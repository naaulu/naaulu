"""Radar count metadata written on estimated accumulations."""

import datetime

import numpy
import pytest
import shapely.geometry
import xarray

import naaulu.estimation
from naaulu.estimation import Sweep, blend_scans, mean_radar_count


def test_mean_radar_count_averages_the_buckets():
    rainrates_radar = {
        "t1": {"be": None, "nl": None, "de": None},
        "t2": {"be": None, "nl": None},
        "t3": {"be": None},
    }

    count = mean_radar_count(rainrates_radar, ["t1", "t2", "t3"], ["be", "nl", "de"])

    assert count == pytest.approx(2.0)


def rainrate_dataset(value):
    return xarray.Dataset(
        {
            "rainrate": xarray.DataArray(
                numpy.full((2, 3), value),
                dims=["y", "x"],
                coords={"x": [0.0, 1000.0, 2000.0], "y": [0.0, 1000.0]},
            )
        },
        coords={"spatial_ref": 0},
    )


def test_compute_rainaccum_tile_writes_radar_count():
    tile = shapely.geometry.box(4.0, 50.0, 6.0, 51.0)
    time = datetime.datetime(2025, 7, 11, 18, 50, 0)
    duration = datetime.timedelta(minutes=5)
    step = datetime.timedelta(minutes=2)
    buckets = [time - duration + i * step for i in range(3)]

    estimator = object.__new__(naaulu.estimation.Base)
    estimator.duration = duration
    estimator.step = step
    estimator.resolution = 1000
    estimator.method = "dove"
    estimator.mapping = {tile: ["be", "nl", "de"]}
    estimator.rainrates_radar = {
        buckets[0]: {"be": None, "nl": None, "de": None},
        buckets[1]: {"be": None, "nl": None},
        buckets[2]: {"be": None},
    }
    estimator.rainrates = {t: {tile: rainrate_dataset(1.0)} for t in buckets}

    accum = estimator.compute_rainaccum_tile(time, tile)

    # three radars contributed, but two on average per bucket
    assert accum.attrs["radar_count"] == pytest.approx(2.0)
    assert len(accum.attrs["radars"]) == 3
    assert accum.attrs["availability"] == 1.0


UTC = datetime.timezone.utc


def at(hour, minute, second=0):
    return datetime.datetime(2025, 7, 11, hour, minute, second, tzinfo=UTC)


def estimator_with(*radars, step=150, duration=300):
    """Minimal Base carrying only what bucket_sweeps reads."""
    estimator = object.__new__(naaulu.estimation.Base)
    estimator.method = "dove"
    estimator.step = datetime.timedelta(seconds=step)
    estimator.duration = datetime.timedelta(seconds=duration)
    estimator.radar_sweeps = {name: list(entries) for name, entries in radars}
    return estimator


def test_bucket_sweeps_blends_the_bracketing_scans():
    # a radar on a 5 minute cycle against a 150 second sampling grid
    before = Sweep(at(21, 54, 14), 180000.0, object())
    after = Sweep(at(21, 59, 14), 180000.0, object())
    later = Sweep(at(22, 4, 14), 180000.0, object())
    estimator = estimator_with(("fr", [before, after, later]))

    selected = estimator.bucket_sweeps(at(21, 57, 30))

    assert len(selected["fr"]) == 2
    weight_before, sweep_before = selected["fr"][0]
    weight_after, sweep_after = selected["fr"][1]
    assert weight_before == pytest.approx(1 - 196 / 300)
    assert weight_after == pytest.approx(196 / 300)
    assert sweep_before is before.sweep
    assert sweep_after is after.sweep


def test_bucket_sweeps_keeps_slow_and_dense_radars_in_every_bucket():
    """Regression: scans used to be tied to their single nearest bucket.

    A 5 minute scanner then missed every second bucket of a 150 second
    sampling grid, and the dense scanners composed those buckets alone with
    full weight in the accumulation.
    """
    slow = [
        Sweep(at(21, 54, 14), 180000.0, object()),
        Sweep(at(21, 59, 14), 180000.0, object()),
        Sweep(at(22, 4, 14), 180000.0, object()),
    ]
    dense = [
        Sweep(at(21, 54), 179500.0, object()),
        Sweep(at(21, 56, 30), 179500.0, object()),
        Sweep(at(21, 59), 179500.0, object()),
        Sweep(at(22, 1, 30), 179500.0, object()),
    ]
    estimator = estimator_with(("fr", slow), ("nl", dense))

    for bucket in (at(21, 55), at(21, 57, 30), at(22, 0)):
        selected = estimator.bucket_sweeps(bucket)
        assert set(selected) == {"fr", "nl"}


def test_bucket_sweeps_does_not_combine_across_a_missing_scan():
    # one scan of the 5 minute cycle is missing: hold instead of spanning
    before = Sweep(at(21, 54, 14), 180000.0, object())
    scans = [
        Sweep(at(21, 44, 14), 180000.0, object()),
        Sweep(at(21, 49, 14), 180000.0, object()),
        before,
        Sweep(at(22, 4, 14), 180000.0, object()),
    ]
    estimator = estimator_with(("fr", scans))

    selected = estimator.bucket_sweeps(at(21, 57, 30))

    assert selected["fr"] == [(1.0, before.sweep)]


def raster(value, missing=()):
    rainrate = numpy.full((2, 2), value)
    quality = numpy.full((2, 2), 0.5)
    for row, column in missing:
        rainrate[row, column] = numpy.nan
        quality[row, column] = numpy.nan
    return xarray.Dataset(
        {
            "rainrate": xarray.DataArray(rainrate, dims=["y", "x"]),
            "quality": xarray.DataArray(quality, dims=["y", "x"]),
        },
        coords={"x": [0.0, 1000.0], "y": [0.0, 1000.0], "spatial_ref": 0},
    )


def test_blend_scans_combines_and_renormalizes_per_pixel():
    # the head scan has data everywhere except one pixel, the tail scan is
    # missing at another pixel where the head scan does hold data
    head = raster(10.0, missing=[(1, 1)])
    tail = raster(30.0, missing=[(0, 1), (1, 1)])

    combined = blend_scans([(0.25, head), (0.75, tail)])

    values = combined["rainrate"].values
    assert values[0, 0] == pytest.approx(0.25 * 10.0 + 0.75 * 30.0)
    # the tail scan has no data: the head scan keeps the pixel alone
    assert values[0, 1] == pytest.approx(10.0)
    assert values[1, 0] == pytest.approx(0.25 * 10.0 + 0.75 * 30.0)
    # no scan holds data there
    assert numpy.isnan(values[1, 1])
    assert combined["quality"].values[0, 0] == pytest.approx(0.5)


def test_compute_rainrate_retries_radars_missing_from_the_archive():
    """A radar missing at fetch time is retried, never cached as final.

    compute_rainaccum prefetches the next frame (time + duration) while
    computing the current one. The OPERA archive needs ~5 minutes to
    receive the BE/FR/NL volumes, so that speculative fetch fails for
    them; caching the empty result made every real frame of the hour
    find a "cache hit" with those radars absent.
    """
    time = at(22, 0)
    estimator = object.__new__(naaulu.estimation.Base)
    estimator.method = "dove"
    estimator.duration = datetime.timedelta(seconds=300)
    estimator.step = datetime.timedelta(seconds=150)
    estimator.radars = ["be", "de"]
    estimator.tiles = []
    estimator.volumes = {}
    estimator.missing_volumes = {}
    estimator.radar_sweeps = {}
    estimator.rainrates_radar = {}
    estimator.rainrates = {}

    be_available = False
    fetched = []

    def get_radar_volume(time, wsi):
        fetched.append(wsi)
        if wsi == "be" and not be_available:
            return None
        return object()

    estimator.get_radar_volume = get_radar_volume
    estimator.compute_rainrate_radar = lambda time, wsi: None

    # the speculative prefetch runs before the BE volume reached the archive
    estimator.compute_rainrate(time)
    assert sorted(fetched) == ["be", "de"]
    assert estimator.missing_volumes[time] == {"be"}

    # the real frame: only the missing radar is fetched again
    fetched.clear()
    be_available = True
    estimator.compute_rainrate(time)
    assert fetched == ["be"]
    assert estimator.missing_volumes[time] == set()

    # complete: nothing is fetched again
    fetched.clear()
    estimator.compute_rainrate(time)
    assert fetched == []
