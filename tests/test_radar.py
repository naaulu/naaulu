import datetime

import shapely.geometry

from naaulu import geography
from naaulu import radar


def test_select_batch_skips_radars_that_do_not_reach_the_tile(monkeypatch):
    """Selection must fit the tile inside the coverage, not its bounding box."""
    tile = shapely.geometry.box(-116, 35, -115, 36)
    far = geography.radar_coverage(location=(-114.0, 37.5), distance=180e3)
    near = geography.radar_coverage(location=(-115.5, 35.5), distance=180e3)
    assert not far.intersects(tile) and near.intersects(tile)

    monkeypatch.setattr(
        radar, "get_database",
        lambda: {
            "far": {"lon": -114.0, "lat": 37.5, "start": None, "country": "USA"},
            "near": {"lon": -115.5, "lat": 35.5, "start": None, "country": "USA"},
        },
    )

    results = radar.select_batch(
        start=datetime.datetime(2026, 10, 5, 7, 55),
        end=datetime.datetime(2026, 10, 5, 8, 5),
        tiles=[tile],
        distance=180e3,
    )

    assert sorted(results[tile][1]) == ["near"]
