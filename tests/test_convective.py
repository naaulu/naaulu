"""Offline tests for the Steiner convective classification."""

import numpy
import xarray

from naaulu import radar

NAZ = 360
NRNG = 120
RSCALE = 500.0


def sweep(values):
    """A (azimuth, range) sweep of reflectivity in dBZ."""
    azimuth = numpy.linspace(0, 360, NAZ, endpoint=False)
    range_ = numpy.arange(NRNG, dtype=float) * RSCALE + RSCALE / 2
    data = numpy.broadcast_to(numpy.asarray(values, dtype=float), (NAZ, NRNG)).copy()
    return xarray.DataArray(
        data,
        dims=("azimuth", "range"),
        coords={"azimuth": azimuth, "range": range_},
        name="DBZH",
    )


def field(background=20.0):
    """A sweep filled with a flat background, ready for a core to be planted."""
    return sweep(numpy.full((NAZ, NRNG), background))


def test_flat_field_below_the_intensity_threshold_is_stratiform():
    assert not radar.convective(sweep(30.0)).values.any()


def test_flat_field_above_the_intensity_threshold_is_convective():
    """Echo above 40 dBZ is convective whatever it is embedded in."""
    assert radar.convective(sweep(45.0)).values.all()


def test_isolated_core_is_flagged_and_expanded():
    """A 45 dBZ core in a 20 dBZ field expands by the 1 km convective radius."""
    dbzh = field(20.0)
    dbzh.values[59:62, 59:62] = 45.0
    result = radar.convective(dbzh)
    assert result.values[59:62, 59:62].all()
    assert result.values[60, 63]           # 2 gates beyond the core
    assert not result.values[60, 64]
    assert result.values[57, 60]           # 2 rays beyond the core
    assert not result.values[56, 60]
    assert not result.values[200, 100]     # the background stays stratiform


def test_core_below_the_intensity_threshold_is_found_by_peakedness():
    """A weak but prominent core is caught by the peakedness criterion."""
    dbzh = field(30.0)
    dbzh.values[59:62, 59:62] = 38.0
    result = radar.convective(dbzh)
    assert result.values[59:62, 59:62].all()
    assert result.values[60, 67]           # 3 km expansion: 6 gates
    assert not result.values[60, 68]
    assert not result.values[200, 100]
    # without the peakedness criterion: nothing
    assert not radar.convective(dbzh, peakedness_min=39.0).values.any()


def test_leading_dimensions_are_classified():
    stacked = xarray.concat([sweep(30.0), sweep(45.0)], dim="time")
    result = radar.convective(stacked)
    assert not result.values[0].any()
    assert result.values[1].all()


def test_sweep_convective_adds_the_variable():
    dataset = xarray.Dataset({"DBZH": field(45.0)})
    radar.sweep_convective(dataset)
    assert dataset["convective"].values.all()
