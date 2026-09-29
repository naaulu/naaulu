"""Offline tests for OPERA PVOL volume selection.

The bucket is stubbed: what matters is which timestamp is picked for a
window, given that nodes are not all on the :00/:05 grid (CZ publishes at
HH:04/HH:09/... and used to be skipped by the former exact match).
"""

import datetime

import pytest

from naaulu import radar

WSI = "0-20000-0-11480"          # Brdy-Praha, node czbrd (offset cadence)
DAY = "2026/10/07"
STAMP_PREFIX = "20261007"


def stamps(minutes):
    """PVOL stamps for one day at the given minutes-of-hour (5 min cadence)."""
    return {f"{STAMP_PREFIX}T{m:02d}{minute:02d}"
            for m in range(24) for minute in minutes}


# ─── candidate ordering ─────────────────────────────────────────────────────

def test_candidates_prefer_volumes_inside_the_window():
    # window [11:00, 11:05] on an on-grid node: the window start wins
    inside = radar._pvol_candidates(
        stamps([0, 5, 10]), "20261007T1100", "20261007T1105"
    )
    assert inside[0] == "20261007T1100"


def test_candidates_handle_an_offset_node():
    # CZ cadence: HH:04, HH:09, ... -> no stamp equals the window start
    cz = {f"{STAMP_PREFIX}T{m:02d}{minute:02d}"
          for m in range(24) for minute in (4, 9, 14, 19, 24, 29)}
    picked = radar._pvol_candidates(cz, "20261007T1100", "20261007T1105")
    assert picked[0] == "20261007T1104"      # first volume inside the window


def test_candidates_fall_back_to_the_latest_before_the_window():
    # the day ends at 11:59, window [12:00, 12:05] has nothing inside
    cz = {f"{STAMP_PREFIX}T{h:02d}59" for h in range(12)}
    picked = radar._pvol_candidates(cz, "20261007T1200", "20261007T1205")
    assert picked[0] == "20261007T1159"

    # a window before the day's first volume falls back to nothing
    assert radar._pvol_candidates(
        {f"{STAMP_PREFIX}T0004"}, "20261006T2300", "20261006T2305"
    ) == []


def test_candidates_are_limited_when_falling_back():
    before = [f"{STAMP_PREFIX}T{m:02d}00" for m in range(11)]   # 00:00 .. 10:00
    picked = radar._pvol_candidates(before, "20261007T1200", "20261007T1205")
    assert len(picked) == 5                      # capped
    assert picked[0] == "20261007T1000"          # closest first
    assert picked[-1] == "20261007T0600"


# ─── _opera end to end with a stubbed bucket ────────────────────────────────

@pytest.fixture
def stub_bucket(monkeypatch):
    """Stub the OPERA bucket for a CZ-like node (PVOL only, no SCAN dir)."""
    downloads, scans = [], []
    config = {"stamps": set(), "skip_dirs": set(), "scan": False}

    def fake_list_pvol_times(datedir, country, node):
        return set(config["stamps"])

    def fake_s3_list(prefix):
        is_pvol = "/PVOL/" in prefix
        if not is_pvol and not config["scan"]:
            return []                       # czbrd has no SCAN directory
        timestr = prefix.split("@")[-1]
        if timestr in config["skip_dirs"]:
            return []
        if not is_pvol:
            scans.append(timestr)
        return [f"{prefix}@DBZH.h5", f"{prefix}@TH.h5"]

    def fake_download(url, filename=None):
        downloads.append(url)
        return "/tmp/fake.h5"

    monkeypatch.setattr(radar, "_list_pvol_times", fake_list_pvol_times)
    monkeypatch.setattr(radar, "_s3_list", fake_s3_list)
    monkeypatch.setattr(radar.naaulu.network, "download", fake_download)
    monkeypatch.setattr(radar, "_validate_hdf5", lambda path: True)
    monkeypatch.setattr(
        radar.xradar.io, "open_odim_datatree", lambda path: f"volume:{path}"
    )
    return downloads, config, scans


def stamps_from(minutes):
    return [f"{STAMP_PREFIX}T{h:02d}{minute:02d}"
            for h in range(24) for minute in minutes]


def test_opera_takes_exactly_one_volume_for_an_offset_node(stub_bucket):
    downloads, config, _ = stub_bucket
    config["stamps"] = stamps_from([4, 9, 14, 19, 24, 29])

    import datetime
    time = datetime.datetime(2026, 10, 7, 11, 5)
    volumes = radar._opera(time, datetime.timedelta(minutes=5), WSI)

    assert len(volumes) == 1
    assert len(downloads) == 1
    assert "T1104" in downloads[0]          # nearest volume inside [11:00, 11:05]


def test_opera_keeps_the_window_start_on_grid_nodes(stub_bucket):
    downloads, config, _ = stub_bucket
    config["stamps"] = stamps_from([0, 5, 10, 15])

    import datetime
    time = datetime.datetime(2026, 10, 7, 11, 5)
    volumes = radar._opera(time, datetime.timedelta(minutes=5), WSI)

    assert len(volumes) == 1
    assert "T1100" in downloads[0]          # unchanged behaviour


def test_opera_skips_candidates_without_a_pvol_directory(stub_bucket):
    """A stamp coming from SCAN only must not end the search."""
    downloads, config, _ = stub_bucket
    config["stamps"] = stamps_from([4, 9])
    config["skip_dirs"] = {"20261007T1104"}  # inside the window, no PVOL dir

    import datetime
    time = datetime.datetime(2026, 10, 7, 11, 5)
    volumes = radar._opera(time, datetime.timedelta(minutes=5), WSI)

    assert len(volumes) == 1
    assert "T1104" not in downloads[0]
    assert "T1009" in downloads[0]          # fell back to the latest before it


def test_opera_still_reads_scan_when_the_node_has_it(stub_bucket):
    """Nodes with a SCAN directory keep the old behaviour (PVOL + SCAN)."""
    downloads, config, scans = stub_bucket
    config["stamps"] = stamps_from([0, 5, 10])
    config["scan"] = True

    import datetime
    time = datetime.datetime(2026, 10, 7, 11, 5)
    volumes = radar._opera(time, datetime.timedelta(minutes=5), WSI)

    assert "/PVOL/" in downloads[0] and "T1100" in downloads[0]
    assert scans == ["20261007T1100"]       # PVOL volume + SCAN elevations
    assert len(volumes) == 1 + 2             # 1 PVOL + 2 SCAN moments


def test_opera_raises_when_no_volume_exists(stub_bucket):
    downloads, config, _ = stub_bucket
    config["stamps"] = set()                 # empty bucket

    import datetime
    with pytest.raises(RuntimeError, match=WSI):
        radar._opera(
            datetime.datetime(2026, 10, 7, 11, 5),
            datetime.timedelta(minutes=5),
            WSI,
        )


# ─── France: the gauge correction is taken back out ─────────────────────────

def fra_volume(dbzh):
    """Minimal volume with one sweep holding the given DBZH values.

    ``dbzh=None`` gives a sweep without a DBZH moment at all.
    """
    import numpy
    import xarray

    if dbzh is None:
        data = {
            "VRADH": xarray.DataArray(
                numpy.zeros((2, 2)), dims=("azimuth", "range")
            )
        }
    else:
        data = {
            "DBZH": xarray.DataArray(
                numpy.asarray(dbzh, dtype=float), dims=("azimuth", "range")
            )
        }
    sweep = xarray.Dataset(
        data,
        coords={"azimuth": [0.0, 1.0], "range": [1000.0, 2000.0]},
    )
    root = xarray.Dataset(
        coords={"sweep_group_name": ["sweep_0"], "sweep_fixed_angle": [0.5]}
    )
    return xarray.DataTree.from_dict(
        {"": xarray.DataTree(dataset=root), "sweep_0": xarray.DataTree(dataset=sweep)}
    )


def test_fra_is_the_reader_for_france():
    assert radar._RADAR["FRA"] is radar.fra


def test_fra_suppresses_the_gauge_correction(monkeypatch):
    """The gauge correction comes out as a flat -1 dB on DBZH."""
    import datetime

    import numpy

    volume = fra_volume([[24.0, numpy.nan], [48.0, 36.0]])
    monkeypatch.setattr(
        radar, "_opera", lambda time, duration, wsi: [volume]
    )

    volumes = radar.fra(
        datetime.datetime(2026, 10, 7, 11, 5),
        datetime.timedelta(minutes=5),
        WSI,
    )

    assert len(volumes) == 1
    got = volumes[0]["sweep_0"].ds["DBZH"].values
    expected = [[23.0, numpy.nan], [47.0, 35.0]]
    numpy.testing.assert_allclose(got, expected)


def test_fra_shifts_every_echo_by_the_same_amount(monkeypatch):
    """A rainrate bias is a constant offset in dBZ.

    Dividing dBZ by a factor instead removes more the stronger the echo
    (5 dB at 30 dBZ, 8 dB at 50 dBZ), which would bend the echo structure
    and drag pixels across the convective threshold in z2r().
    """
    import datetime

    import numpy

    volume = fra_volume([[10.0, 30.0], [44.0, 54.0]])
    monkeypatch.setattr(
        radar, "_opera", lambda time, duration, wsi: [volume]
    )

    volumes = radar.fra(
        datetime.datetime(2026, 10, 7, 11, 5),
        datetime.timedelta(minutes=5),
        WSI,
    )

    deltas = volumes[0]["sweep_0"].ds["DBZH"].values - numpy.asarray(
        [[10.0, 30.0], [44.0, 54.0]]
    )
    numpy.testing.assert_allclose(deltas, -1.0)


def test_fra_leaves_sweeps_without_dbzh_alone(monkeypatch):
    import datetime

    volume = fra_volume(None)
    monkeypatch.setattr(
        radar, "_opera", lambda time, duration, wsi: [volume]
    )

    volumes = radar.fra(
        datetime.datetime(2026, 10, 7, 11, 5),
        datetime.timedelta(minutes=5),
        WSI,
    )

    assert "DBZH" not in volumes[0]["sweep_0"].ds


@pytest.mark.network
def test_opera_listing_real_bucket():
    """The bucket keeps volumes for 24h, so a pinned date ages out forever.

    A day is chosen at run time rather than hard-coded; today is tried
    first, then yesterday, because a run just after midnight can find
    today's directory still empty.
    """
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    stamps = []
    for day_offset in (0, 1):
        day = now - datetime.timedelta(days=day_offset)
        stamps = radar._list_pvol_times(day.strftime("%Y/%m/%d"), "CZ", "czbrd")
        if stamps:
            break

    assert stamps, "no OPERA volumes listed for czbrd in the last two days"
    minutes = {int(s[-2:]) for s in stamps if len(s) == 13}
    assert 4 in minutes or 0 in minutes     # offset or on-grid cadence
