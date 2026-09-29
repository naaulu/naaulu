"""Offline tests for the z2r reflectivity to rain rate conversion."""

import numpy
import pytest
import xarray

from naaulu import radar

NAZ = 9
NRNG = 17


def sweep(values):
    """A (azimuth, range) sweep of reflectivity in dBZ."""
    azimuth = numpy.linspace(0, 360, NAZ, endpoint=False)
    range_ = numpy.arange(NRNG, dtype=float) * 1000.0
    data = numpy.broadcast_to(numpy.asarray(values, dtype=float), (NAZ, NRNG)).copy()
    return xarray.DataArray(
        data,
        dims=("azimuth", "range"),
        coords={"azimuth": azimuth, "range": range_},
        name="DBZH",
    )


def marshall_palmer(dbzh):
    """Z = 200 R^1.6, solved for R."""
    return (10 ** (numpy.asarray(dbzh, dtype=float) / 10) / 200.0) ** (1 / 1.6)


def convective(dbzh):
    """Z = 300 R^1.4, solved for R."""
    return (10 ** (numpy.asarray(dbzh, dtype=float) / 10) / 300.0) ** (1 / 1.4)


def values(dbzh):
    return radar.z2r(sweep(dbzh)).values


def test_marshall_palmer_on_a_smooth_field():
    """A flat field is not convective: below 40 dBZ it is plain Marshall-Palmer."""
    for dbzh in [0, 10, 20, 30, 36, 39]:
        assert numpy.allclose(values(dbzh), marshall_palmer(dbzh))


@pytest.mark.parametrize("dbzh", [40, 44, 50])
def test_bright_fields_are_convective(dbzh):
    """Reflectivity above the 40 dBZ intensity threshold is deep convection."""
    assert numpy.allclose(values(dbzh), convective(dbzh))


@pytest.mark.parametrize("dbzh", [54, 55, 60, 65, 80])
def test_hail_cap_holds_the_rate_at_54_dbzh(dbzh):
    """Reflectivity is capped first, so the rate never exceeds 54 dBZ's."""
    assert numpy.allclose(values(dbzh), convective(54.0))
    assert convective(54.0) == pytest.approx(122.3969, rel=1e-4)


def test_rate_never_exceeds_the_cap():
    field = numpy.linspace(0, 70, NRNG)
    assert radar.z2r(sweep(field)).max().item() <= convective(54.0)


def test_embedded_core_selects_the_relations():
    """A 45 dBZ core in a 20 dBZ field uses the convective relation.

    On this small grid the 10 km background circle is barely larger than the
    sweep itself, so the core lifts the background at its gate to ~26 dBZ
    and the 2 km convective radius of that background drags the two gates
    behind it into the convective class as well.
    """
    dbzh = sweep(20.0)
    dbzh.values[4, 8] = 45.0
    result = radar.z2r(dbzh)
    assert result.values[4, 8] == pytest.approx(convective(45.0), rel=1e-6)
    assert result.values[4, 10] == pytest.approx(convective(20.0), rel=1e-6)
    assert result.values[4, 11] == pytest.approx(marshall_palmer(20.0), rel=1e-6)
    assert result.values[0, 0] == pytest.approx(marshall_palmer(20.0), rel=1e-6)


def test_nan_stays_nan():
    field = sweep(30.0)
    field.values[4, 8] = numpy.nan
    result = radar.z2r(field)
    assert numpy.isnan(result.values[4, 8])
    assert numpy.isnan(result.values).sum() == 1


def test_keeps_coords_and_dims():
    field = sweep(30.0)
    result = radar.z2r(field)
    assert result.dims == field.dims
    assert result.coords.identical(field.coords)
