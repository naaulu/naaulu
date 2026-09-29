"""Offline tests for volume validation and archiving."""

import datetime

import numpy
import pytest
import xarray

from naaulu import radar
from test_radar_aus import make_volume

TIME = datetime.datetime(2026, 10, 5, 8, 6)
DURATION = datetime.timedelta(minutes=10)


def mixed_resolution_volume():
    """A volume whose sweeps disagree: 360 rays and a partial 359-ray scan."""

    def sweep(n, fixed, number):
        times = numpy.datetime64("2026-10-05T08:00:00") + numpy.arange(n) * numpy.timedelta64(1, "s")
        return xarray.DataTree(dataset=xarray.Dataset(
            {
                "DBZH": (("azimuth", "range"), numpy.full((n, 4), 10.0)),
                "sweep_fixed_angle": fixed,
                "sweep_mode": "azimuth_surveillance",
                "sweep_number": number,
            },
            coords={
                "azimuth": numpy.arange(n, dtype=float),
                "range": numpy.arange(4, dtype=float) * 1000.0,
                "elevation": numpy.full(n, fixed),
                "time": ("azimuth", times),
            },
        ))

    root = xarray.DataTree(dataset=xarray.Dataset(
        coords={
            "latitude": -33.7008,
            "longitude": 151.2090,
            "altitude": 195.0,
            # no sweep_group_name/sweep_fixed_angle here: create_volume()
            # rebuilds them, and a length-2 coord would leak into sweep_0
        }
    ))
    return xarray.DataTree.from_dict({
        "": root,
        "sweep_0": sweep(360, 0.5, 0),
        "sweep_1": sweep(359, 1.5, 1),
    })


def test_get_volume_drops_the_partial_sweep_and_keeps_the_volume(monkeypatch):
    volume = mixed_resolution_volume()
    monkeypatch.setattr(radar, "get", lambda **kwargs: [volume])

    merged = radar.get_volume(
        time=TIME, duration=DURATION, wsi="0-21010-0-606", min_angle=0, max_angle=90
    )

    assert list(merged.ds.sweep_group_name.values) == ["sweep_0"]
    numpy.testing.assert_allclose(merged.ds.sweep_fixed_angle.values, [0.5])
    # the partial sweep is gone from the source volume too
    assert "sweep_1" not in list(volume.children)


def test_get_volume_still_fails_when_no_sweep_is_usable(monkeypatch):
    volume = mixed_resolution_volume()
    del volume["sweep_0"]                       # only the 359-ray sweep left
    monkeypatch.setattr(radar, "get", lambda **kwargs: [volume])

    with pytest.raises(RuntimeError, match="No valid sweeps"):
        radar.get_volume(
            time=TIME, duration=DURATION, wsi="0-21010-0-606",
            min_angle=0, max_angle=90,
        )


def test_write_converts_boolean_attributes(tmp_path):
    """NEXRAD VCP flags are Python bools; NetCDF has no boolean attributes."""
    import h5py

    times = ["2026-10-05T08:00:00", "2026-10-05T08:01:00"]
    volume = make_volume(times)
    volume.attrs["mpda_vcp"] = True
    volume.attrs["number_elevation_cuts"] = 14
    volume["sweep_0"].attrs["sails_cut"] = False

    target = tmp_path / "volume.nc"
    radar.write(volume, str(target))            # used to raise CompatibilityError

    assert target.exists()
    with h5py.File(str(target), "r") as handle:
        # h5py hands back size-1 arrays for scalar attributes
        assert numpy.asarray(handle.attrs["mpda_vcp"]).item() == 1
        assert numpy.asarray(handle.attrs["number_elevation_cuts"]).item() == 14
        assert numpy.asarray(handle["sweep_0"].attrs["sails_cut"]).item() == 0
        assert str(handle["sweep_0/DBZH"].dtype).startswith("float")


def test_write_leaves_nothing_behind_when_the_write_fails(tmp_path, monkeypatch):
    """A failed write must not leave a half-written file for later runs to trip on."""
    volume = make_volume(["2026-10-05T08:00:00"])

    def explode(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(xarray.DataTree, "to_netcdf", explode)
    target = tmp_path / "volume.nc"

    with pytest.raises(RuntimeError, match="disk full"):
        radar.write(volume, str(target))

    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_combine_volume_rebuilds_an_unusable_cache(tmp_path, monkeypatch):
    """A corrupt archive file is a cache miss, not a permanent dead end.

    The boolean-attribute failure used to leave 7 kB stubs behind: they opened
    fine but held no sweeps, so every later run died on "No sweeps remain after
    filtering" instead of recomputing the volume.
    """
    target = tmp_path / "volume.nc"
    target.write_bytes(b"not a netcdf file at all")

    volume = make_volume(["2026-10-05T08:00:00", "2026-10-05T08:01:00"])
    monkeypatch.setattr(radar, "path", lambda **kwargs: str(target))
    monkeypatch.setattr(radar, "get_database", lambda: {"0-00000-0-0": {"country": "USA"}})
    monkeypatch.setattr(radar, "get_volume", lambda **kwargs: volume)

    rebuilt = radar.combine_volume(
        time=TIME,
        duration=DURATION,
        wsi="0-00000-0-0",
        azimuth_scale=1,
        range_scale=500,
        max_range=180e3,
        min_angle=0,
        max_angle=90,
        variables=["DBZH"],
        precision=8,
        update=True,
    )

    assert rebuilt is not None
    # the stub was discarded and replaced by a real, readable volume
    assert target.stat().st_size > 20000
    with xarray.open_datatree(str(target), engine="h5netcdf") as cached:
        assert list(cached.children)


def test_missing_cache_still_raises_file_not_found(tmp_path, monkeypatch):
    """The cache-miss path must stay FileNotFoundError: that is what triggers a rebuild."""
    target = tmp_path / "absent.nc"
    monkeypatch.setattr(radar, "path", lambda **kwargs: str(target))
    monkeypatch.setattr(radar, "get_database", lambda: {"0-00000-0-0": {"country": "USA"}})

    with pytest.raises(FileNotFoundError):
        radar.combine_volume(
            time=TIME,
            duration=DURATION,
            wsi="0-00000-0-0",
            azimuth_scale=1,
            range_scale=500,
            max_range=180e3,
            min_angle=0,
            max_angle=90,
            variables=["DBZH"],
            precision=8,
            update=False,
        )
