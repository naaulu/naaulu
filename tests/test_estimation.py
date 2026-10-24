"""Radar count metadata written on estimated accumulations."""

import datetime

import numpy
import pytest
import shapely.geometry
import xarray

import naaulu.estimation
from naaulu.estimation import mean_radar_count


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
