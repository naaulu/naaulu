"""Tests for the NCI/AURA (Australia) radar provider.

Everything runs offline: the daily volume zips are served by a tiny local
HTTP server that implements Range requests, mirroring the NCI archive, which
only exposes multi-GB daily zips over ranged HTTP.
"""

import concurrent.futures as cf
import datetime
import http.server
import io
import threading
import time
import zipfile

import numpy
import pytest
import xarray
import xradar

import naaulu.config
import naaulu.network
from naaulu import radar


# ─── fixtures ───────────────────────────────────────────────────────────────

def make_volume(times, lat=-33.7008, lon=151.2090, dbzh=None, clean=None):
    """Minimal datatree shaped like one xradar opens from an ODIM file.

    clean= adds a DBZH_CLEAN moment with the real files' attributes, so the
    rename to DBZH can be checked.
    """
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
    supports_range = True
    status = 200
    fail_gets = 0        # answer this many GETs with 503 first
    bad_ranges = 0       # answer this many GETs with a wrong Content-Range first
    stats = None         # {"lock", "in_flight", "max_in_flight"}

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
            self.send_response(200 if self.status == 200 else self.status)
            self.send_header("Content-Length", str(len(self.payload)))
            self.end_headers()
        finally:
            self.track(-1)

    def do_GET(self):
        self.track(1)
        try:
            if self.status != 200:
                self.send_response(self.status)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return

            if type(self).fail_gets > 0:
                type(self).fail_gets -= 1
                self.send_response(503)
                self.send_header("Retry-After", "0")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return

            spec = self.headers.get("Range")
            if spec and self.supports_range:
                start_text, _, end_text = spec.partition("=")[2].partition("-")
                start = int(start_text)
                end = min(
                    int(end_text) if end_text else len(self.payload) - 1,
                    len(self.payload) - 1,
                )
                body = self.payload[start:end + 1]
                self.send_response(206)
                if type(self).bad_ranges > 0:
                    type(self).bad_ranges -= 1
                    self.send_header("Content-Range", "bytes 0-0/1")
                else:
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

    def break_next_ranges(self, count):
        self.handler.bad_ranges = count

    @property
    def max_in_flight(self):
        return self.handler.stats["max_in_flight"]


@pytest.fixture
def range_server():
    """serve(payload, suffix='') -> url; shuts every server down afterwards."""
    servers = []

    def serve(payload, supports_range=True, status=200, suffix="", site="71"):
        handler = type(
            "Handler",
            (_RangeHandler,),
            {
                "payload": payload,
                "supports_range": supports_range,
                "status": status,
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
        url = f"http://{host}:{port}/rq0/{site}/2026/vol/{site}_20261005.pvol.zip{suffix}"
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

    with zipfile.ZipFile(radar._RangeReader(url)) as archive:
        assert archive.namelist() == sorted(
            ["71_20261005_080500.pvol.h5", "71_20261005_081000.pvol.h5", "README.txt"]
        )
        data = archive.read("71_20261005_080500.pvol.h5")

    with zipfile.ZipFile(volume_zip) as archive:
        assert data == archive.read("71_20261005_080500.pvol.h5")


def test_range_reader_rejects_server_without_range(range_server, volume_zip):
    url = range_server(volume_zip.read_bytes(), supports_range=False)

    with pytest.raises(RuntimeError, match="range"):
        with zipfile.ZipFile(radar._RangeReader(url, retries=1)) as archive:
            archive.namelist()


def test_range_reader_missing_archive(range_server):
    url = range_server(b"", status=404)

    with pytest.raises(FileNotFoundError):
        radar._RangeReader(url)


def test_parse_members_sorts_and_ignores_other_files(volume_zip):
    members = radar._aus_parse_members(str(volume_zip))

    assert [name for name, _ in members] == [
        "71_20261005_080500.pvol.h5",
        "71_20261005_081000.pvol.h5",
    ]
    assert members[0][1] == datetime.datetime(2026, 10, 5, 8, 5)


def test_member_listing_is_cached(volume_zip, monkeypatch):
    monkeypatch.setattr(radar, "_aus_members_cache", {})
    parsed = []
    original = radar._aus_parse_members

    def counting(url):
        parsed.append(url)
        return original(url)

    monkeypatch.setattr(radar, "_aus_parse_members", counting)

    radar._aus_list_members(str(volume_zip))
    radar._aus_list_members(str(volume_zip))

    assert parsed == [str(volume_zip)]


def test_member_listing_cache_evicts_oldest(monkeypatch):
    monkeypatch.setattr(radar, "_aus_members_cache", {})
    monkeypatch.setattr(radar, "_AUS_MEMBER_CACHE_MAX", 2)
    monkeypatch.setattr(
        radar, "_aus_parse_members", lambda url: [(f"{url}.h5", WINDOW)]
    )

    for i in range(3):
        radar._aus_list_members(f"zip-{i}")

    assert list(radar._aus_members_cache) == ["zip-1", "zip-2"]


def test_extract_single_member(volume_zip, tmp_path):
    destination = tmp_path / "cache" / "71_20261005_080500.pvol.h5"

    result = radar._aus_extract(
        str(volume_zip), "71_20261005_080500.pvol.h5", str(destination)
    )

    assert result == str(destination)
    assert destination.exists()
    assert radar._validate_hdf5(str(destination))
    assert not list(destination.parent.glob("*.part"))
    # cached: a second call must not rewrite the file
    stamp = destination.stat().st_mtime_ns
    radar._aus_extract(str(volume_zip), "71_20261005_080500.pvol.h5", str(destination))
    assert destination.stat().st_mtime_ns == stamp


def test_extract_rejects_corrupt_member(tmp_path):
    archive_path = tmp_path / "corrupt.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("71_20261005_080500.pvol.h5", b"definitely not hdf5")
    destination = tmp_path / "out" / "71_20261005_080500.pvol.h5"

    with pytest.raises(ValueError, match="corrupted"):
        radar._aus_extract(str(archive_path), "71_20261005_080500.pvol.h5", str(destination))

    assert not destination.exists()
    assert destination.parent.exists()
    assert not list(destination.parent.glob("*.part"))


# ─── volume selection ───────────────────────────────────────────────────────

def test_select_picks_volume_nominally_inside_grid_window():
    members = stamps("2026-10-05T08:00:00", "2026-10-05T08:05:00", "2026-10-05T08:10:00")

    # the volume starting at time_end has all of its sweeps after the window
    assert radar._aus_select(members, WINDOW, WINDOW_END) == [
        datetime.datetime(2026, 10, 5, 8, 0)
    ]


def test_select_includes_preceding_volume_for_offset_window():
    # PT150S sampling also produces windows like 07:57:30 .. 08:02:30
    start = datetime.datetime(2026, 10, 5, 7, 57, 30)
    end = datetime.datetime(2026, 10, 5, 8, 2, 30)
    members = stamps(
        "2026-10-05T07:55:00", "2026-10-05T08:00:00", "2026-10-05T08:05:00"
    )

    assert radar._aus_select(members, start, end) == [
        datetime.datetime(2026, 10, 5, 7, 55),
        datetime.datetime(2026, 10, 5, 8, 0),
    ]


def test_select_falls_back_when_no_volume_nominally_inside():
    # 6 minute cadence of the older archive: nothing starts inside the window
    start = datetime.datetime(2026, 10, 5, 7, 57, 30)
    members = stamps("2026-10-05T07:50:00", "2026-10-05T07:56:00")

    assert radar._aus_select(members, start, start + datetime.timedelta(minutes=5)) == [
        datetime.datetime(2026, 10, 5, 7, 56)
    ]


def test_select_ignores_volumes_older_than_max_lag():
    members = stamps("2026-10-05T07:00:00")

    assert radar._aus_select(members, WINDOW, WINDOW_END) == []


def test_select_crosses_midnight():
    start = datetime.datetime(2026, 10, 5, 23, 57, 30)
    end = datetime.datetime(2026, 10, 6, 0, 2, 30)
    members = stamps(
        "2026-10-05T23:55:00", "2026-10-06T00:00:00", "2026-10-06T00:05:00"
    )

    assert radar._aus_select(members, start, end) == [
        datetime.datetime(2026, 10, 5, 23, 55),
        datetime.datetime(2026, 10, 6, 0, 0),
    ]


# ─── time clamping ──────────────────────────────────────────────────────────

def test_clamp_keeps_sweeps_that_would_be_filtered_out():
    # a scan named 08:05 runs to 08:06:40, past the end of an 08:00..08:05 window
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


def test_clamp_leaves_times_inside_the_window_untouched():
    times = ["2026-10-05T08:01:00", "2026-10-05T08:02:00", "2026-10-05T08:03:00"]
    volume = make_volume(times)
    before = volume["sweep_0"].ds.time.values.copy()

    radar._aus_clamp(volume, WINDOW, WINDOW_END)

    numpy.testing.assert_array_equal(volume["sweep_0"].ds.time.values, before)


def test_clamp_clips_both_ends_of_a_straddling_scan():
    times = ["2026-10-05T07:59:00", "2026-10-05T08:02:00", "2026-10-05T08:06:00"]
    clamped = radar._aus_clamp(make_volume(times), WINDOW, WINDOW_END)

    numpy.testing.assert_array_equal(
        clamped["sweep_0"].ds.time.values,
        numpy.array(
            ["2026-10-05T08:00:00", "2026-10-05T08:02:00", "2026-10-05T08:05:00"],
            dtype="datetime64[ns]",
        ),
    )


# ─── DBZH_CLEAN exposed as DBZH ─────────────────────────────────────────────

def test_clean_reflectivity_replaces_dbzh():
    times = ["2026-10-05T08:05:21", "2026-10-05T08:06:00"]
    # raw DBZH masks no-echo to NaN; CLEAN carries it as the -31.9 floor
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
    # the cleaned moment's metadata travels with the renamed variable
    assert dataset["DBZH"].attrs["_Undetect"] == 1.0


def test_clean_reflectivity_falls_back_to_raw_dbzh():
    times = ["2026-10-05T08:05:21"]
    dbzh = numpy.array([[1.0, 2.0, 3.0, 4.0]])

    volume = make_volume(times, dbzh=dbzh)
    radar._aus_use_clean_reflectivity(volume)

    dataset = volume["sweep_0"].ds
    assert "DBZH_CLEAN" not in dataset
    numpy.testing.assert_array_equal(dataset["DBZH"].values, dbzh)


def test_clean_reflectivity_floor_yields_no_rain():
    """The whole point: undetect precipitation must come out as ~0 mm/h."""
    times = ["2026-10-05T08:05:21"]
    volume = make_volume(times, clean=numpy.full((1, 4), -31.9))
    radar._aus_use_clean_reflectivity(volume)

    sweep = volume["sweep_0"].ds
    rainrate = sweep["DBZH"].wrl.trafo.idecibel().wrl.zr.z_to_r()
    assert numpy.nanmax(rainrate.values) < 0.01
    assert not numpy.isnan(rainrate.values).any()


# ─── site mapping ───────────────────────────────────────────────────────────

def test_check_site_accepts_matching_coordinates():
    volume = make_volume(["2026-10-05T08:05:21"])

    assert radar._aus_check_site(volume, "0-21010-0-606")


def test_check_site_rejects_another_radar_from_a_reused_id():
    # archive id 38 holds Charleville until 2006 and Newdegate from 2016
    volume = make_volume(
        ["2026-10-05T08:05:21"], lat=-33.0970, lon=119.0087
    )

    assert not radar._aus_check_site(volume, "0-20010-0-94510")


def test_site_id_uses_bundled_mapping(monkeypatch):
    monkeypatch.setattr(radar, "_get_aus_mapping", lambda: {"0-21010-0-606": "71"})

    assert radar._aus_site_id("0-21010-0-606") == "71"


def test_site_id_warns_and_skips_unmatched_radar(monkeypatch, caplog):
    # Coffs Harbour's nearest NCI site is Grafton, 0.7 degrees away
    monkeypatch.setattr(radar, "_get_aus_mapping", lambda: {})
    monkeypatch.setattr(
        radar,
        "_aus_site_list",
        lambda: [
            {
                "id": "28",
                "lat": -29.6206,
                "lon": 152.9633,
                "wigos": None,
                "start": None,
                "end": None,
                "location": "Grafton",
            }
        ],
    )

    with caplog.at_level("WARNING"):
        assert radar._aus_site_id("0-20010-0-94791") is None
        assert radar._aus_site_id("0-20010-0-94791") is None  # second fetch

    assert "no NCI archive site" in caplog.text
    assert "skipping" in caplog.text
    # the estimate loop fetches twice per event: only the first warns
    assert caplog.text.count("no NCI archive site") == 1


def test_site_id_without_coordinates_is_skipped(monkeypatch, caplog):
    monkeypatch.setattr(radar, "_get_aus_mapping", lambda: {})
    monkeypatch.setattr(
        radar,
        "_aus_site_list",
        lambda: [
            {
                "id": "1",
                "lat": -37.69,
                "lon": 144.95,
                "wigos": None,
                "start": None,
                "end": None,
                "location": "Broadmeadows",
            }
        ],
    )
    monkeypatch.setattr(radar, "get_database", lambda: {"0-00000-0-0": {}})

    with caplog.at_level("WARNING"):
        assert radar._aus_site_id("0-00000-0-0") is None

    assert "no coordinates known" in caplog.text


# ─── resilience & politeness ────────────────────────────────────────────────

def test_retry_recovers_from_server_error(range_server, volume_zip):
    url = range_server(volume_zip.read_bytes())
    url.fail_next_gets(2)

    with zipfile.ZipFile(radar._RangeReader(url)) as archive:
        assert len(archive.namelist()) == 3


def test_retry_honours_retry_after(range_server, volume_zip):
    url = range_server(volume_zip.read_bytes())
    url.fail_next_gets(2)   # 503 with Retry-After: 0

    started = time.monotonic()
    with zipfile.ZipFile(radar._RangeReader(url)) as archive:
        assert archive.namelist()

    # two failures answered with Retry-After: 0 must not cost the 1s+2s backoff
    assert time.monotonic() - started < 1.0


def test_retry_recovers_from_broken_range_header(range_server, volume_zip):
    url = range_server(volume_zip.read_bytes())
    url.break_next_ranges(1)

    with zipfile.ZipFile(radar._RangeReader(url)) as archive:
        assert "71_20261005_080500.pvol.h5" in archive.namelist()


def test_semaphore_caps_concurrent_requests(range_server, volume_zip):
    url = range_server(volume_zip.read_bytes())
    urls = [f"{url}?slot={i}" for i in range(16)]

    def fetch(target):
        with zipfile.ZipFile(radar._RangeReader(target)) as archive:
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

    # a later run starts with an empty in-process cache and must not re-parse
    monkeypatch.setattr(radar, "_aus_members_cache", {})
    monkeypatch.setattr(
        radar,
        "_aus_parse_members",
        lambda target: (_ for _ in ()).throw(AssertionError("re-parsed listing")),
    )
    assert radar._aus_list_members(str(url)) == members


def test_listing_cache_is_keyed_by_site_and_day(range_server, volume_zip, tmp_path, monkeypatch):
    """Regression: the key must include the site, or every site overwrites one file."""
    monkeypatch.setattr("naaulu.config.get_cache_dir", lambda subdir=None: str(tmp_path))
    monkeypatch.setattr(radar, "_aus_members_cache", {})
    payload = volume_zip.read_bytes()
    url_a = range_server(payload, site="71")
    url_b = range_server(payload, site="72")

    members_a = radar._aus_list_members(str(url_a))
    members_b = radar._aus_list_members(str(url_b))

    assert sorted(p.name for p in tmp_path.glob("aus_*.json")) == [
        "aus_71_2026_20261005.json",
        "aus_72_2026_20261005.json",
    ]

    # both entries stay usable after the other site wrote its own file
    monkeypatch.setattr(radar, "_aus_members_cache", {})
    monkeypatch.setattr(
        radar,
        "_aus_parse_members",
        lambda target: (_ for _ in ()).throw(AssertionError("re-parsed listing")),
    )
    assert radar._aus_list_members(str(url_a)) == members_a
    assert radar._aus_list_members(str(url_b)) == members_b


def test_listing_cache_invalidated_when_archive_changes(range_server, volume_zip, tmp_path, monkeypatch):
    monkeypatch.setattr("naaulu.config.get_cache_dir", lambda subdir=None: str(tmp_path))
    monkeypatch.setattr(radar, "_aus_members_cache", {})
    monkeypatch.setattr(radar, "_aus_sizes", {})
    url = range_server(volume_zip.read_bytes())
    assert len(radar._aus_list_members(str(url))) == 2

    # the day zip is republished with one more volume
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in (
            "71_20261005_080500.pvol.h5",
            "71_20261005_081000.pvol.h5",
            "71_20261005_081500.pvol.h5",
        ):
            archive.writestr(name, b"data")
    url.set_payload(buffer.getvalue())

    monkeypatch.setattr(radar, "_aus_members_cache", {})
    monkeypatch.setattr(radar, "_aus_sizes", {})
    parsed = []
    original = radar._aus_parse_members
    monkeypatch.setattr(
        radar,
        "_aus_parse_members",
        lambda target: parsed.append(target) or original(target),
    )

    members = radar._aus_list_members(str(url))
    assert parsed == [str(url)]
    assert len(members) == 3


def test_session_identifies_the_client(monkeypatch):
    monkeypatch.setattr(naaulu.network, "session", None)

    session = naaulu.network.runtime_session()
    try:
        assert session.headers["User-Agent"].startswith("naaulu/")
        adapter = session.get_adapter("https://dapds00.nci.org.au")
        # requests keeps the constructor arguments under private names
        assert adapter._pool_maxsize == naaulu.network.MAX_POOL_SIZE
    finally:
        session.close()


# ─── era of a reused archive id ─────────────────────────────────────────────

def test_era_conflict_skips_before_download(monkeypatch, tmp_path, caplog, volume_zip):
    rows = [
        {
            "id": "38",
            "lat": -26.4139,
            "lon": 146.2558,
            "wigos": "0-0-94510",
            "start": datetime.date(1999, 12, 9),
            "end": datetime.date(2006, 10, 1),
            "location": "Charleville",
        },
        {
            "id": "38",
            "lat": -33.097,
            "lon": 119.0087,
            "wigos": None,
            "start": datetime.date(2016, 10, 1),
            "end": None,
            "location": "Ndegate",
        },
    ]
    monkeypatch.setattr(radar, "_aus_site_list", lambda: rows)
    monkeypatch.setattr(
        radar,
        "_get_aus_mapping",
        lambda: {"0-20010-0-94510": "38", "0-21010-0-658": "38"},
    )
    monkeypatch.setattr(radar, "_aus_zip_url", lambda site, day: str(volume_zip))
    monkeypatch.setattr(radar, "_aus_warned", set())
    monkeypatch.setattr(naaulu.config, "get_download_dir", lambda subdir=None: str(tmp_path))
    extracted = []
    monkeypatch.setattr(
        radar, "_aus_extract", lambda url, name, destination: extracted.append(name)
    )

    # the fixture zip holds volumes at 08:05 and 08:10, so use that window
    when = datetime.datetime(2026, 10, 5, 8, 10)
    window = datetime.timedelta(minutes=5)

    with caplog.at_level("WARNING"):
        for _ in range(2):
            with pytest.raises(RuntimeError, match="archive site 38 holds Ndegate"):
                radar.aus(time=when, duration=window, wsi="0-20010-0-94510")

    assert extracted == []                                    # nothing downloaded
    assert caplog.text.count("archive site 38 holds Ndegate") == 1  # warn once

    # the radar that does own the id in 2026 gets as far as extraction
    with pytest.raises(RuntimeError, match="No usable AURA radar volume"):
        radar.aus(time=when, duration=window, wsi="0-21010-0-658")
    assert extracted


# ─── against the real archive ───────────────────────────────────────────────

@pytest.mark.network
def test_aus_returns_clamped_volumes_from_the_archive():
    time = datetime.datetime(2026, 10, 5, 8, 5)
    duration = datetime.timedelta(minutes=5)

    try:
        volumes = radar.aus(
            time=time, duration=duration, wsi="0-21010-0-606"
        )
    except (RuntimeError, FileNotFoundError) as exc:
        pytest.skip(f"no archive data available: {exc}")

    assert volumes
    for volume in volumes:
        assert float(volume.ds["longitude"].values) == pytest.approx(151.21, abs=0.1)
        for key in xradar.util.get_sweep_keys(volume):
            dataset = volume[key].ds
            times = dataset.time.values
            assert times.min() >= numpy.datetime64(time - duration)
            assert times.max() <= numpy.datetime64(time)
            assert dataset.sizes["azimuth"] == 360
            # DBZH_CLEAN must have been renamed into DBZH
            assert "DBZH_CLEAN" not in dataset
            assert dataset["DBZH"].attrs.get("_Undetect") == 1.0
            # undetect precipitation kept as the floor instead of NaN
            assert numpy.nanmin(dataset["DBZH"].values) >= -31.95
