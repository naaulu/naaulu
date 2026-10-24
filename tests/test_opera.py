"""Offline tests for OPERA PVOL volume selection."""

import datetime

import numpy
import pytest
import xarray

from naaulu import radar

WSI = "0-20000-0-11480"          # Brdy-Praha, node czbrd
STAMP_PREFIX = "20261007"


# ─── _opera end to end with a stubbed bucket ────────────────────────────────

@pytest.fixture
def stub_bucket(monkeypatch):
    """Stub the OPERA bucket for a CZ-like node (PVOL only, no SCAN dir)."""
    downloads = []
    config = {"stamps": set(), "skip_dirs": set()}

    def fake_list_pvol_times(datedir, country, node):
        return set(config["stamps"])

    def fake_s3_list(prefix):
        if "/PVOL/" not in prefix:
            return []                       # no SCAN directory here
        timestr = prefix.split("@")[-1]
        if timestr in config["skip_dirs"]:
            return []
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
    return downloads, config


def stamps_from(minutes):
    return [f"{STAMP_PREFIX}T{h:02d}{minute:02d}"
            for h in range(24) for minute in minutes]


def test_opera_takes_exactly_one_volume_for_an_offset_node(stub_bucket):
    downloads, config = stub_bucket
    config["stamps"] = stamps_from([4, 9, 14, 19, 24, 29])

    time = datetime.datetime(2026, 10, 7, 11, 5)
    volumes = radar._opera(time, datetime.timedelta(minutes=5), WSI)

    assert len(volumes) == 1
    assert len(downloads) == 1
    assert "T1104" in downloads[0]          # nearest volume inside [11:00, 11:05]


def test_opera_keeps_the_window_start_on_grid_nodes(stub_bucket):
    downloads, config = stub_bucket
    config["stamps"] = stamps_from([0, 5, 10, 15])

    time = datetime.datetime(2026, 10, 7, 11, 5)
    volumes = radar._opera(time, datetime.timedelta(minutes=5), WSI)

    assert len(volumes) == 1
    assert "T1100" in downloads[0]


def test_opera_raises_when_no_volume_exists(stub_bucket):
    downloads, config = stub_bucket
    config["stamps"] = set()                 # empty bucket

    with pytest.raises(RuntimeError, match=WSI):
        radar._opera(
            datetime.datetime(2026, 10, 7, 11, 5),
            datetime.timedelta(minutes=5),
            WSI,
        )


# ─── France: the gauge correction is taken back out ─────────────────────────

def fra_volume(dbzh):
    """One-sweep volume holding the given DBZH values; None gives no DBZH."""
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


def test_fra_suppresses_the_gauge_correction(monkeypatch):
    """The gauge correction comes out as a flat -1 dB on DBZH."""
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


@pytest.mark.network
def test_opera_listing_real_bucket():
    """Today is tried first, then yesterday: right after midnight today can be empty."""
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
