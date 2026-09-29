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
    """A 45 dBZ core in a 20 dBZ field gets the 1 km convective radius.

    At 30 km, 1 degree of azimuth is 0.53 km and a gate is 0.5 km, so the
    expansion reaches 2 rays and 2 gates beyond the core.
    """
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
    """A 38 dBZ core in a 30 dBZ field: too weak to be intense, but prominent.

    38 - (10 - 30**2 / 180) = 33.0 dBZ stands well above the ~30.2 dBZ
    background, so the peakedness criterion flags it, and the 3 km
    convective radius of a 30 dBZ background expands it by 6 gates.
    """
    dbzh = field(30.0)
    dbzh.values[59:62, 59:62] = 38.0
    result = radar.convective(dbzh)
    assert result.values[59:62, 59:62].all()
    assert result.values[60, 67]           # 3 km expansion: 6 gates
    assert not result.values[60, 68]
    assert not result.values[200, 100]
    # the same field without the peakedness criterion finds nothing
    assert not radar.convective(dbzh, peakedness_min=39.0).values.any()


def test_core_straddling_the_azimuth_wrap_is_expanded_both_ways():
    """A core on ray 0 has to be expanded onto rays 358/359 as well."""
    dbzh = field(20.0)
    dbzh.values[0, 60] = 45.0
    result = radar.convective(dbzh)
    assert result.values[[0, 1, 2, 358, 359], 60].all()
    assert not result.values[3, 60]


def test_missing_gates_are_never_convective():
    """NaN and the -32 dBZ recode of sub-noise echo stay out of the class.

    The gates sit inside a 45 dBZ field, so the expansion would drag the
    convective class over them if the nodata mask were not applied last.
    """
    dbzh = sweep(45.0)
    dbzh.values[100, 100] = numpy.nan
    dbzh.values[101, 101] = -32.0
    result = radar.convective(dbzh)
    assert not result.values[100, 100]
    assert not result.values[101, 101]
    assert result.values.sum() == result.size - 2


def test_thin_echo_does_not_get_a_background():
    """A 38 dBZ core in a 30 dBZ disc of 5 km radius.

    The 10 km background circle is three quarters empty, so with the default
    coverage the core is not tested against a background and stays
    stratiform; with a laxer coverage the same core is classified.
    """
    dbzh = sweep(numpy.nan)
    azimuth = numpy.linspace(0, 360, NAZ, endpoint=False)
    range_ = numpy.arange(NRNG) * RSCALE + RSCALE / 2
    x = range_[None, :] * numpy.cos(numpy.deg2rad(azimuth))[:, None]
    y = range_[None, :] * numpy.sin(numpy.deg2rad(azimuth))[:, None]
    disc = numpy.hypot(x, y) < 5000.0
    dbzh.values[disc] = 30.0
    dbzh.values[0, 6] = 38.0               # inside the disc, at 3.25 km

    assert not radar.convective(dbzh).values.any()
    assert radar.convective(dbzh, coverage=0.1).values[0, 6]


def test_leading_dimensions_are_classified():
    stacked = xarray.concat([sweep(30.0), sweep(45.0)], dim="time")
    result = radar.convective(stacked)
    assert not result.values[0].any()
    assert result.values[1].all()


def test_result_keeps_name_dims_and_coords():
    dbzh = sweep(30.0)
    result = radar.convective(dbzh)
    assert result.name == "convective"
    assert result.dtype == bool
    assert result.dims == dbzh.dims
    assert result.coords.identical(dbzh.coords)


def test_sweep_convective_adds_the_variable():
    dataset = xarray.Dataset({"DBZH": field(45.0)})
    radar.sweep_convective(dataset)
    assert dataset["convective"].values.all()
