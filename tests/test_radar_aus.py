"""Offline tests for the NCI/AURA (Australia) radar provider."""

import concurrent.futures as cf
import datetime
import http.server
import threading
import zipfile

import numpy
import pytest
import xarray
import xradar

import naaulu.network
from naaulu import radar


# ─── fixtures ───────────────────────────────────────────────────────────────

def make_volume(times, lat=-33.7008, lon=151.2090, dbzh=None, clean=None):
    """Minimal xradar datatree; clean= adds a DBZH_CLEAN moment to test the rename."""
    times = numpy.asarray(times, dtype="datetime64[ns]")
    if dbzh is None:
        dbzh = numpy.full((times.size, 4), 10.0)
    data = {
        "DBZH": xarray.DataArray(
            numpy.asarray(dbzh, dtype=float), dims=("azimuth", "range")
        ),
        "sweep_fixed_angle": 0.5,
        "sweep_mode": "azimuth_surveillance",
        "sweep_number": 0,
    }
    if clean is not None:
        data["DBZH_CLEAN"] = xarray.DataArray(
            numpy.asarray(clean, dtype=float),
            dims=("azimuth", "range"),
            attrs={"_Undetect": 1.0, "units": "dBZ"},
        )
    sweep = xarray.Dataset(
        data,
        coords={
            "azimuth": numpy.arange(times.size, dtype=float),
            "range": numpy.arange(4, dtype=float) * 1000.0,
            "elevation": numpy.full(times.size, 0.5),
            "time": ("azimuth", times),
        },
    )
    root = xarray.Dataset(
        coords={
            "latitude": lat,
            "longitude": lon,
            "altitude": 195.0,
            "sweep_group_name": ["sweep_0"],
            "sweep_fixed_angle": [0.5],
        }
    )
    return xarray.DataTree.from_dict(
        {"": xarray.DataTree(dataset=root), "sweep_0": xarray.DataTree(dataset=sweep)}
    )


def write_member(target):
    """A tiny but valid HDF5 file to stand in for a .pvol.h5 member."""
    import h5py

    with h5py.File(target, "w") as handle:
        handle.create_dataset("what", data=b"PVOL")
    return target.read_bytes()


@pytest.fixture
def volume_zip(tmp_path):
    """Daily volume zip: two valid members plus a file to be ignored."""
    member_a = "71_20261005_080500.pvol.h5"
    member_b = "71_20261005_081000.pvol.h5"
    archive_path = tmp_path / "71_20261005.pvol.zip"
    payloads = {
        member_a: write_member(tmp_path / "a.h5"),
        member_b: write_member(tmp_path / "b.h5"),
        "README.txt": b"not a volume",
    }
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in payloads.items():
            archive.writestr(name, data)
    return archive_path


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    """HTTP server exposing fixed bytes, with or without Range support."""

    payload = b""
    fail_gets = 0        # GETs to answer with 503 first
    stats = None         # concurrency counters

    def log_message(self, *args):
        pass

    @classmethod
    def track(cls, delta):
        with cls.stats["lock"]:
            cls.stats["in_flight"] += delta
            if delta > 0:
                cls.stats["max_in_flight"] = max(
                    cls.stats["max_in_flight"], cls.stats["in_flight"]
                )

    def do_HEAD(self):
        self.track(1)
        try:
            self.send_response(200)
            self.send_header("Content-Length", str(len(self.payload)))
            self.end_headers()
        finally:
            self.track(-1)

    def do_GET(self):
        self.track(1)
        try:
            if type(self).fail_gets > 0:
                type(self).fail_gets -= 1
                self.send_response(503)
                self.send_header("Retry-After", "0")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return

            spec = self.headers.get("Range")
            if spec:
                start_text, _, end_text = spec.partition("=")[2].partition("-")
                start = int(start_text)
                end = min(
                    int(end_text) if end_text else len(self.payload) - 1,
                    len(self.payload) - 1,
                )
                body = self.payload[start:end + 1]
                self.send_response(206)
                self.send_header(
                    "Content-Range", f"bytes {start}-{end}/{len(self.payload)}"
                )
            else:
                body = self.payload
                self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self.wfile.write(body)
        finally:
            self.track(-1)


class _Url(str):
    """The served URL, with handles on the handler behind it."""

    def __new__(cls, text, handler):
        obj = super().__new__(cls, text)
        obj.handler = handler
        return obj

    def set_payload(self, data):
        self.handler.payload = data

    def fail_next_gets(self, count):
        self.handler.fail_gets = count

    @property
    def max_in_flight(self):
        return self.handler.stats["max_in_flight"]


@pytest.fixture
def range_server():
    """serve(payload) -> url; shuts every server down afterwards."""
    servers = []

    def serve(payload, site="71"):
        handler = type(
            "Handler",
            (_RangeHandler,),
            {
                "payload": payload,
                "fail_gets": 0,
                "stats": {
                    "lock": threading.Lock(),
                    "in_flight": 0,
                    "max_in_flight": 0,
                },
            },
        )
        httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        servers.append(httpd)
        host, port = httpd.server_address[:2]
        url = f"http://{host}:{port}/rq0/{site}/2026/vol/{site}_20261005.pvol.zip"
        return _Url(url, handler)

    yield serve

    for httpd in servers:
        httpd.shutdown()
        httpd.server_close()


WINDOW = datetime.datetime(2026, 10, 5, 8, 0)
WINDOW_END = datetime.datetime(2026, 10, 5, 8, 5)


def stamps(*text):
    return [(f"71_{t}.pvol.h5", datetime.datetime.fromisoformat(t)) for t in text]


# ─── ranged access to the daily zip ─────────────────────────────────────────

def test_range_reader_lists_and_reads_members(range_server, volume_zip):
    url = range_server(volume_zip.read_bytes())

    with zipfile.ZipFile(naaulu.network.RangeReader(url)) as archive:
        assert archive.namelist() == sorted(
            ["71_20261005_080500.pvol.h5", "71_20261005_081000.pvol.h5", "README.txt"]
        )
        data = archive.read("71_20261005_080500.pvol.h5")

    with zipfile.ZipFile(volume_zip) as archive:
        assert data == archive.read("71_20261005_080500.pvol.h5")


def test_extract_single_member(volume_zip, tmp_path):
    destination = tmp_path / "cache" / "71_20261005_080500.pvol.h5"

    result = radar._aus_extract(
        str(volume_zip), "71_20261005_080500.pvol.h5", str(destination)
    )

    assert result == str(destination)
    assert destination.exists()
    assert radar._validate_hdf5(str(destination))


# ─── volume selection ───────────────────────────────────────────────────────

def test_select_picks_volume_nominally_inside_grid_window():
    members = stamps("2026-10-05T08:00:00", "2026-10-05T08:05:00", "2026-10-05T08:10:00")

    # the volume starting at time_end falls entirely after the window
    assert radar._aus_select(members, WINDOW, WINDOW_END) == [
        datetime.datetime(2026, 10, 5, 8, 0)
    ]


def test_select_includes_preceding_volume_for_offset_window():
    # offset windows like 07:57:30 .. 08:02:30 come from PT150S sampling
    start = datetime.datetime(2026, 10, 5, 7, 57, 30)
    end = datetime.datetime(2026, 10, 5, 8, 2, 30)
    members = stamps(
        "2026-10-05T07:55:00", "2026-10-05T08:00:00", "2026-10-05T08:05:00"
    )

    assert radar._aus_select(members, start, end) == [
        datetime.datetime(2026, 10, 5, 7, 55),
        datetime.datetime(2026, 10, 5, 8, 0),
    ]


# ─── time clamping ──────────────────────────────────────────────────────────

def test_clamp_keeps_sweeps_that_would_be_filtered_out():
    # every ray of the 08:05 scan lies past the window end
    times = ["2026-10-05T08:05:21", "2026-10-05T08:06:00", "2026-10-05T08:06:40"]
    volume = make_volume(times)

    with pytest.raises(ValueError, match="No sweeps remain"):
        xradar.util.create_volume(
            sweeps=[volume],
            time_coverage_start=WINDOW,
            time_coverage_end=WINDOW_END,
            min_angle=0,
            max_angle=0.5,
        )

    clamped = radar._aus_clamp(make_volume(times), WINDOW, WINDOW_END)
    merged = xradar.util.create_volume(
        sweeps=[clamped],
        time_coverage_start=WINDOW,
        time_coverage_end=WINDOW_END,
        min_angle=0,
        max_angle=0.5,
    )

    assert list(merged.ds.sweep_group_name.values) == ["sweep_0"]
    ray_times = merged["sweep_0"].ds.time.values
    assert ray_times.min() >= numpy.datetime64(WINDOW)
    assert ray_times.max() <= numpy.datetime64(WINDOW_END)


# ─── DBZH_CLEAN exposed as DBZH ─────────────────────────────────────────────

def test_clean_reflectivity_replaces_dbzh():
    times = ["2026-10-05T08:05:21", "2026-10-05T08:06:00"]
    # raw DBZH masks no-echo as NaN, CLEAN as the -31.9 floor
    dbzh = numpy.array(
        [[numpy.nan, 12.0, 8.0, 40.0], [5.0, numpy.nan, 20.0, 33.0]]
    )
    clean = numpy.array(
        [[-31.9, 12.1, numpy.nan, 40.2], [5.1, -31.9, 20.2, numpy.nan]]
    )

    volume = make_volume(times, dbzh=dbzh, clean=clean)
    radar._aus_use_clean_reflectivity(volume)

    dataset = volume["sweep_0"].ds
    assert "DBZH_CLEAN" not in dataset
    assert "DBZH" in dataset
    numpy.testing.assert_allclose(dataset["DBZH"].values, clean, equal_nan=True)
    # metadata travels with the rename
    assert dataset["DBZH"].attrs["_Undetect"] == 1.0


# ─── resilience & politeness ────────────────────────────────────────────────

def test_retry_recovers_from_server_error(range_server, volume_zip):
    url = range_server(volume_zip.read_bytes())
    url.fail_next_gets(2)

    with zipfile.ZipFile(naaulu.network.RangeReader(url)) as archive:
        assert len(archive.namelist()) == 3


def test_semaphore_caps_concurrent_requests(range_server, volume_zip):
    url = range_server(volume_zip.read_bytes())
    urls = [f"{url}?slot={i}" for i in range(16)]

    def fetch(target):
        with zipfile.ZipFile(naaulu.network.RangeReader(target)) as archive:
            return len(archive.namelist())

    with cf.ThreadPoolExecutor(max_workers=16) as pool:
        counts = list(pool.map(fetch, urls))

    assert counts == [3] * 16
    assert url.max_in_flight >= 2          # still parallel
    assert url.max_in_flight <= 4          # but never more than the cap


def test_listing_cache_written_and_reused(range_server, volume_zip, tmp_path, monkeypatch):
    monkeypatch.setattr("naaulu.config.get_cache_dir", lambda subdir=None: str(tmp_path))
    monkeypatch.setattr(radar, "_aus_members_cache", {})
    url = range_server(volume_zip.read_bytes())

    members = radar._aus_list_members(str(url))
    assert len(members) == 2
    assert len(list(tmp_path.glob("aus_*.json"))) == 1

    # a later run starts with an empty in-process cache
    monkeypatch.setattr(radar, "_aus_members_cache", {})
    monkeypatch.setattr(
        radar,
        "_aus_parse_members",
        lambda target: (_ for _ in ()).throw(AssertionError("re-parsed listing")),
    )
    assert radar._aus_list_members(str(url)) == members
