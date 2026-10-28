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


# ─── per-radar calibration offsets ───────────────────────────────────────────

def stub_volume(dbzh):
    """One-sweep volume holding the given DBZH values."""
    sweep = xarray.Dataset(
        {
            "DBZH": xarray.DataArray(
                numpy.asarray(dbzh, dtype=float), dims=("azimuth", "range")
            )
        },
        coords={"azimuth": [0.0, 1.0], "range": [1000.0, 2000.0]},
    )
    root = xarray.Dataset(
        coords={"sweep_group_name": ["sweep_0"], "sweep_fixed_angle": [0.5]}
    )
    return xarray.DataTree.from_dict(
        {"": xarray.DataTree(dataset=root), "sweep_0": xarray.DataTree(dataset=sweep)}
    )


def test_calibration_offsets_are_applied(monkeypatch):
    """FRA loses the upstream gauge correction (-2 dB), Jabbeke gains its
    +1 dB, radars without an entry pass through unchanged."""
    def fake_opera(time, duration, wsi):
        return [stub_volume([[24.0, numpy.nan], [48.0, 36.0]])]

    monkeypatch.setitem(radar._RADAR, "FRA", fake_opera)
    monkeypatch.setitem(radar._RADAR, "BEL", fake_opera)
    monkeypatch.setattr(radar, "get_database", lambda: {
        "0-20010-0-07083": {"country": "FRA"},   # Avesnes
        "0-20010-0-06410": {"country": "BEL"},   # Jabbeke
        "0-20010-0-06482": {"country": "BEL"},   # Wideumont, no entry
    })

    def dbzh(wsi):
        volumes = radar.get(
            time=datetime.datetime(2026, 10, 7, 11, 5),
            duration=datetime.timedelta(minutes=5),
            wsi=wsi,
        )
        assert len(volumes) == 1
        return volumes[0]["sweep_0"].ds["DBZH"].values

    numpy.testing.assert_allclose(
        dbzh("0-20010-0-07083"),
        [[22.0, numpy.nan], [46.0, 34.0]],
        equal_nan=True,
    )
    numpy.testing.assert_allclose(
        dbzh("0-20010-0-06410"),
        [[25.0, numpy.nan], [49.0, 37.0]],
        equal_nan=True,
    )
    numpy.testing.assert_allclose(
        dbzh("0-20010-0-06482"),
        [[24.0, numpy.nan], [48.0, 36.0]],
        equal_nan=True,
    )
