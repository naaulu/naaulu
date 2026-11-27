import datetime

import numpy
import pytest
import shapely.geometry
import xarray

import naaulu.config
import naaulu.precip
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


# ─── radar_count on combined accumulations ───────────────────────────────────

TILE = shapely.geometry.box(4.0, 50.0, 6.0, 51.0)
BASE_DURATION = datetime.timedelta(minutes=5)
RESOLUTION = 1000
PRODUCT = "dove"
# PT10M ending 18:50 sums the 5min accumulations closing at 18:45 and 18:50
FIRST_BASE = datetime.datetime(2025, 7, 11, 18, 45, 0)


def save_base_array(time, values, radar_count=None, availability=1.0):
    """Write a base accumulation with an explicit 2x3 value grid."""
    dataset = xarray.Dataset(
        {"precipitation": (("y", "x"), numpy.asarray(values, dtype=float))},
        coords={"x": [0.0, 1000.0, 2000.0], "y": [0.0, 1000.0]},
    )
    dataset.attrs = naaulu.precip.set_metadata(
        time=time,
        tile=TILE,
        duration=BASE_DURATION,
        resolution=RESOLUTION,
        product=PRODUCT,
    )
    dataset.attrs["availability"] = availability
    if radar_count is not None:
        dataset.attrs["radar_count"] = radar_count
    naaulu.precip.save(dataset)


def save_base(time, value, radar_count=None, availability=1.0):
    """Write a base accumulation the way `naaulu estimate` saves them."""
    save_base_array(
        time,
        numpy.full((2, 3), value),
        radar_count=radar_count,
        availability=availability,
    )


@pytest.fixture
def archive(monkeypatch, tmp_path):
    """Route the archive to a temporary directory (must be active when saving)."""
    monkeypatch.setattr(naaulu.config, "get_archive_dir", lambda: str(tmp_path))
    return tmp_path


def combine_bases(archive, duration=2 * BASE_DURATION):
    return naaulu.precip.combine(
        time=FIRST_BASE + BASE_DURATION,
        tile=TILE,
        duration=duration,
        resolution=RESOLUTION,
        product=PRODUCT,
    )


def test_combine_averages_radar_count(archive):
    save_base(FIRST_BASE, 1.0, radar_count=4)
    save_base(FIRST_BASE + BASE_DURATION, 2.0, radar_count=6)

    result = combine_bases(archive)

    assert result.attrs["radar_count"] == 5.0
    # a complete combination is fully available
    assert result.attrs["availability"] == pytest.approx(1.0)
    numpy.testing.assert_allclose(result["precipitation"].values, 3.0)


def test_combine_scales_availability_by_missing_slots(archive):
    # one of three slots is missing; the two present are 50% available
    save_base(FIRST_BASE, 1.0, availability=0.5)
    save_base(FIRST_BASE + BASE_DURATION, 1.0, availability=0.5)

    result = combine_bases(archive, duration=3 * BASE_DURATION)

    assert result.attrs["availability"] == pytest.approx(0.5 * 2 / 3)


# ─── coverage guard on combined accumulations ────────────────────────────────


def slot_times(n):
    """The n base slots of the window ending at FIRST_BASE + BASE_DURATION."""
    first = FIRST_BASE + BASE_DURATION * (2 - n)
    return [first + i * BASE_DURATION for i in range(n)]


def test_combine_fills_from_partial_coverage(archive):
    # 11 of 12 slots (0.92 >= 0.8): one file missing, one file NaN at [0, 0]
    for i, t in enumerate(slot_times(12)):
        if i == 3:
            continue
        if i == 5:
            values = numpy.full((2, 3), 1.0)
            values[0, 0] = numpy.nan
            save_base_array(t, values)
        else:
            save_base(t, 1.0)

    result = combine_bases(archive, duration=12 * BASE_DURATION)

    values = result["precipitation"].values
    expected = numpy.full((2, 3), 11.0)
    expected[0, 0] = 10.0    # the pixel that lost both the missing file and the NaN one
    numpy.testing.assert_allclose(values, expected)
    assert result.attrs["availability"] == pytest.approx(11 / 12)


def test_combine_masks_below_coverage(archive):
    # 9 of 12 slots (0.75 < 0.8): every pixel is dropped
    for i, t in enumerate(slot_times(12)):
        if i % 4 == 0:
            continue
        save_base(t, 1.0)

    result = combine_bases(archive, duration=12 * BASE_DURATION)

    assert numpy.isnan(result["precipitation"].values).all()
    assert result.attrs["availability"] == pytest.approx(9 / 12)


def test_combine_missing_file_equals_all_nan_file(archive, monkeypatch):
    def combine_with(skipped):
        directory = archive / skipped
        directory.mkdir()
        monkeypatch.setattr(naaulu.config, "get_archive_dir", lambda: str(directory))
        for i, t in enumerate(slot_times(12)):
            if i == 3 and skipped == "missing":
                continue
            if i == 3:
                save_base_array(t, numpy.full((2, 3), numpy.nan))
            else:
                save_base(t, 1.0)
        return combine_bases(archive, duration=12 * BASE_DURATION)

    missing = combine_with("missing")
    all_nan = combine_with("all_nan")

    # a file that failed to load is a file of NaNs at every pixel
    numpy.testing.assert_allclose(
        missing["precipitation"].values,
        all_nan["precipitation"].values,
        equal_nan=True,
    )
    numpy.testing.assert_allclose(missing["precipitation"].values, 11.0)
