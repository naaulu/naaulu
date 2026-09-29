import datetime

import numpy
import pytest
import xarray

from naaulu.precip import accumulate


X = [0.0, 1000.0, 2000.0, 3000.0]
Y = [0.0, 1000.0, 2000.0]


def make_rainrate_dataset(values_2d, x=X, y=Y):
    return xarray.Dataset({
        "rainrate": xarray.DataArray(
            numpy.array(values_2d, dtype=float),
            dims=["y", "x"],
            coords={"x": x, "y": y},
        )
    }, coords={"spatial_ref": 0})


def test_accumulate_constant_rate():
    t0 = datetime.datetime(2025, 7, 11, 18, 50, 0)
    dt = datetime.timedelta(minutes=5)
    timestamps = [t0, t0 + dt, t0 + 2 * dt]
    time_start = t0
    time_end = t0 + 2 * dt
    ds = make_rainrate_dataset(numpy.full((3, 4), 12.0))

    result = accumulate([ds, ds, ds], timestamps, time_start, time_end)

    expected = 12.0 * (10.0 / 60.0)
    numpy.testing.assert_allclose(result["precipitation"].values, expected, atol=1e-6)


def test_accumulate_zero_rate():
    t0 = datetime.datetime(2025, 7, 11, 18, 50, 0)
    dt = datetime.timedelta(minutes=5)
    timestamps = [t0, t0 + dt, t0 + 2 * dt]
    time_start = t0
    time_end = t0 + 2 * dt
    ds = make_rainrate_dataset(numpy.zeros((3, 4)))

    result = accumulate([ds, ds, ds], timestamps, time_start, time_end)

    numpy.testing.assert_allclose(result["precipitation"].values, 0.0, atol=1e-6)


def test_accumulate_nan_interpolation():
    t0 = datetime.datetime(2025, 7, 11, 18, 50, 0)
    dt = datetime.timedelta(minutes=5)
    timestamps = [t0, t0 + dt, t0 + 2 * dt]
    time_start = t0
    time_end = t0 + 2 * dt

    ds1 = make_rainrate_dataset(numpy.full((3, 4), 5.0))
    ds2 = make_rainrate_dataset(numpy.full((3, 4), numpy.nan))
    ds3 = make_rainrate_dataset(numpy.full((3, 4), 10.0))

    result = accumulate([ds1, ds2, ds3], timestamps, time_start, time_end)

    assert not numpy.isnan(result["precipitation"].values).any(), "output contains NaN"

    expected = ((5.0 + 7.5) / 2 + (7.5 + 10.0) / 2) * (5.0 / 60.0)
    numpy.testing.assert_allclose(result["precipitation"].values, expected, atol=1e-6)


def test_accumulate_single_dataset_raises():
    t0 = datetime.datetime(2025, 7, 11, 18, 50, 0)
    ds = make_rainrate_dataset(numpy.full((3, 4), 5.0))

    with pytest.raises(ValueError, match="insufficient data"):
        accumulate([ds], [t0], t0, t0)


def test_accumulate_boundary_padding():
    t0 = datetime.datetime(2025, 7, 11, 18, 50, 0)
    dt = datetime.timedelta(minutes=5)

    timestamps = [t0 + datetime.timedelta(minutes=2, seconds=30),
                  t0 + datetime.timedelta(minutes=7, seconds=30)]
    time_start = t0
    time_end = t0 + 2 * dt

    ds1 = make_rainrate_dataset(numpy.full((3, 4), 6.0))
    ds2 = make_rainrate_dataset(numpy.full((3, 4), 6.0))

    # Caller pads boundaries before calling accumulate (matches compute_rainaccum_tile)
    timestamps.insert(0, time_start)
    timestamps.append(time_end)

    result = accumulate([ds1, ds1, ds2, ds2], timestamps, time_start, time_end)

    expected = 6.0 * (10.0 / 60.0)
    numpy.testing.assert_allclose(result["precipitation"].values, expected, atol=1e-6)


def test_accumulate_two_rates():
    t0 = datetime.datetime(2025, 7, 11, 18, 50, 0)
    dt = datetime.timedelta(minutes=5)
    timestamps = [t0, t0 + dt, t0 + 2 * dt]
    time_start = t0
    time_end = t0 + 2 * dt

    ds1 = make_rainrate_dataset(numpy.full((3, 4), 0.0))
    ds2 = make_rainrate_dataset(numpy.full((3, 4), 6.0))
    ds3 = make_rainrate_dataset(numpy.full((3, 4), 12.0))

    result = accumulate([ds1, ds2, ds3], timestamps, time_start, time_end)

    expected = (0.0 + 6.0) / 2 * (5.0 / 60.0) + (6.0 + 12.0) / 2 * (5.0 / 60.0)
    numpy.testing.assert_allclose(result["precipitation"].values, expected, atol=1e-6)


def test_accumulate_preserves_coords():
    t0 = datetime.datetime(2025, 7, 11, 18, 50, 0)
    dt = datetime.timedelta(minutes=5)
    timestamps = [t0, t0 + dt, t0 + 2 * dt]
    time_start = t0
    time_end = t0 + 2 * dt
    ds = make_rainrate_dataset(numpy.full((3, 4), 1.0))

    result = accumulate([ds, ds, ds], timestamps, time_start, time_end)

    assert "precipitation" in result
    assert "rainrate" not in result
    assert "x" in result.coords
    assert "y" in result.coords
    assert "spatial_ref" in result.coords
    assert result["precipitation"].dims == ("y", "x")
    assert result["precipitation"].shape == (3, 4)


def test_accumulate_2d_grid():
    t0 = datetime.datetime(2025, 7, 11, 18, 50, 0)
    dt = datetime.timedelta(minutes=5)
    timestamps = [t0, t0 + dt, t0 + 2 * dt]
    time_start = t0
    time_end = t0 + 2 * dt

    rates = numpy.array([
        [0.0, 6.0, 12.0, 18.0],
        [2.0, 4.0,  8.0, 16.0],
        [1.0, 3.0,  9.0, 15.0],
    ])

    ds1 = make_rainrate_dataset(rates * 0.0)
    ds2 = make_rainrate_dataset(rates * 0.5)
    ds3 = make_rainrate_dataset(rates * 1.0)

    result = accumulate([ds1, ds2, ds3], timestamps, time_start, time_end)

    expected = rates * (5.0 / 60.0)
    numpy.testing.assert_allclose(result["precipitation"].values, expected, atol=1e-6)
