"""Offline tests for the VMM (Flanders) hourly gauge feed.

Same KISTERS KiWIS stack as SPW, so the transport, the opening-instant hour
labelling and the metadata caching are shared (_kiwis_*). What is checked
here is what is VMM-specific: the group query that brings coordinates
alongside the series list, the VMM- code prefix, and that the three Belgian
networks merge without collision.
"""

import datetime
import json
import os

import numpy
import pytest
import requests
import shapely.geometry

import naaulu.errors
from naaulu import gauge


TARGET = datetime.datetime(2026, 10, 9, 8, 0)   # naive times are UTC
ONE_HOUR = datetime.timedelta(hours=1)
# covers RMI's station, both of SPW's and VMM's
BELGIUM = shapely.geometry.box(2.0, 49.0, 7.0, 52.0)
STAMP = "2026-10-09T09:00:00.000+02:00"         # == 07:00 UTC: opens the hour
OTHER_STAMP = "2026-10-09T10:00:00.000+02:00"   # == 08:00 UTC: the next hour

SERIES = [
    {"ts_id": "1", "station_no": "01P03_005", "station_id": "1225",
     "station_name": "Vinderhoute_P"},
    {"ts_id": "2", "station_no": "01ALMC_30RT01007", "station_id": "12996",
     "station_name": "Retie_ALMC"},
]

# getTimeseriesValueLayer carries the coordinates with the series list
LAYER = [
    {"ts_id": "1", "station_latitude": 51.0, "station_longitude": 3.65},
    {"ts_id": "2", "station_latitude": 51.28, "station_longitude": 5.10},
]


def block(ts_id, data):
    return {
        "ts_id": ts_id,
        "rows": str(len(data)),
        "columns": "Timestamp,Value",
        "data": data,
    }


@pytest.fixture
def kiwis(monkeypatch, tmp_path):
    """Stub the KiWIS transport, recording every call the feed makes."""
    state = {"calls": [], "values": []}

    def fake(url, params, fmt, label):
        state["calls"].append((params["request"], dict(params), fmt, label, url))
        request = params["request"]
        if request == "getTimeseriesList":
            return SERIES
        if request == "getTimeseriesValueLayer":
            return LAYER
        if request == "getTimeseriesValues":
            return state["values"]
        raise AssertionError(f"unexpected request {request}")

    monkeypatch.setattr(gauge, "_kiwis_get", fake)

    def cache_dir(subdir=None):
        path = tmp_path / subdir if subdir else tmp_path
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    monkeypatch.setattr("naaulu.config.get_cache_dir", cache_dir)
    # an assignment, so monkeypatch can undo it: clearing the dict here
    # would leave the fake series behind for the live test to pick up
    monkeypatch.setattr(gauge, "_series_cache", {})
    return state


def window(state):
    return next(
        (params["from"], params["to"])
        for name, params, *_ in state["calls"]
        if name == "getTimeseriesValues"
    )


def requested(state):
    return [name for name, *_ in state["calls"]]


# ─── the request itself ─────────────────────────────────────────────────────

def test_window_is_the_ending_hour_in_utc(kiwis):
    kiwis["values"] = [block("1", [[STAMP, 2.4]])]

    gauge._bel_vmm(TARGET, datetime.timedelta(hours=1))

    assert window(kiwis) == ("2026-10-09T07:00:00Z", "2026-10-09T08:00:00Z")


def test_values_go_to_the_vmm_endpoint_in_one_call(kiwis):
    kiwis["values"] = [block("1", [[STAMP, 2.4]])]

    gauge._bel_vmm(TARGET, datetime.timedelta(hours=1))

    _, params, fmt, label, url = kiwis["calls"][-1]
    assert fmt == "json"          # objson is rejected by this request
    assert label == "VMM"
    assert url == gauge.VMM_KIWIS_URL
    assert params["ts_id"] == "1,2"
    assert "timespan" not in params   # silently ignored: returns latest only


# ─── how a station ends up in the dataset ───────────────────────────────────

def test_codes_come_from_the_series_and_positions_from_the_layer(kiwis):
    kiwis["values"] = [block("1", [[STAMP, 2.4]]), block("2", [[STAMP, 0.7]])]

    values, codes, coords = gauge._bel_vmm(TARGET, datetime.timedelta(hours=1))

    assert values == [2.4, 0.7]
    assert codes == ["VMM-01P03_005", "VMM-01ALMC_30RT01007"]
    # lon first: create_dataset() reads coords[:,0] as the longitude
    numpy.testing.assert_allclose(coords, [[3.65, 51.0], [5.10, 51.28]])


def test_the_hour_is_the_one_ending_at_time(kiwis):
    """VMM stamps an hourly total with the instant the hour opens.

    naaulu labels an accumulation by the instant it closes, so VMM's 07:00
    row (the 07:00-08:00 hour) belongs under time=08:00, not its 08:00 row.
    """
    kiwis["values"] = [
        block("1", [
            ["2026-10-09T09:00:00.000+02:00", 8.7],   # 07:00Z -> the hour
            ["2026-10-09T10:00:00.000+02:00", 4.5],   # 08:00Z -> next hour
        ])
    ]

    values, _, _ = gauge._bel_vmm(TARGET, datetime.timedelta(hours=1))

    assert values == [8.7]


def test_stations_missing_the_hour_are_dropped(kiwis):
    kiwis["values"] = [
        block("1", [[STAMP, 2.4]]),
        block("2", [[OTHER_STAMP, 9.9]]),   # next hour: no value for our window
    ]

    values, codes, coords = gauge._bel_vmm(TARGET, datetime.timedelta(hours=1))

    assert values == [2.4]
    assert codes == ["VMM-01P03_005"]
    assert coords.shape == (1, 2)


# ─── failure modes ──────────────────────────────────────────────────────────

def test_only_one_hour_is_supported(kiwis):
    with pytest.raises(ValueError, match="duration"):
        gauge._bel_vmm(TARGET, datetime.timedelta(minutes=10))


def test_no_station_matching_the_hour_raises(kiwis):
    kiwis["values"] = [block("1", [[OTHER_STAMP, 9.9]])]

    with pytest.raises(naaulu.errors.NoDataError, match="No VMM data"):
        gauge._bel_vmm(TARGET, datetime.timedelta(hours=1))


# ─── station metadata caching ───────────────────────────────────────────────

def test_metadata_is_read_once(kiwis):
    kiwis["values"] = [block("1", [[STAMP, 2.4]])]

    gauge._bel_vmm(TARGET, datetime.timedelta(hours=1))
    gauge._bel_vmm(TARGET, datetime.timedelta(hours=1))

    fetched = [n for n in requested(kiwis) if n != "getTimeseriesValues"]
    assert fetched == ["getTimeseriesList", "getTimeseriesValueLayer"]


def test_metadata_survives_without_the_in_process_cache(monkeypatch, kiwis, tmp_path):
    kiwis["values"] = [block("1", [[STAMP, 2.4]])]

    gauge._bel_vmm(TARGET, datetime.timedelta(hours=1))
    assert os.path.exists(os.path.join(tmp_path, "vmm", "series.json"))

    # a later run starts with an empty in-process cache
    monkeypatch.setattr(gauge, "_series_cache", {})
    gauge._bel_vmm(TARGET, datetime.timedelta(hours=1))

    fetched = [n for n in requested(kiwis) if n != "getTimeseriesValues"]
    assert fetched == ["getTimeseriesList", "getTimeseriesValueLayer"]


def test_expired_metadata_is_refetched(monkeypatch, kiwis, tmp_path):
    kiwis["values"] = [block("1", [[STAMP, 2.4]])]

    gauge._bel_vmm(TARGET, datetime.timedelta(hours=1))
    filename = tmp_path / "vmm" / "series.json"
    cached = json.loads(filename.read_text())
    cached["generated"] -= gauge.KIWIS_MAX_AGE + 1
    filename.write_text(json.dumps(cached))

    monkeypatch.setattr(gauge, "_series_cache", {})
    gauge._bel_vmm(TARGET, datetime.timedelta(hours=1))

    fetched = [n for n in requested(kiwis) if n != "getTimeseriesValues"]
    assert fetched == ["getTimeseriesList", "getTimeseriesValueLayer"] * 2


# ─── how the three keys sit together ────────────────────────────────────────

def _collect(monkeypatch, tmp_path, countries, hour=TARGET, duration=ONE_HOUR):
    """The public path: dispatch by key, cache write, station concat."""
    def fake_path(time, duration, country):
        return str(tmp_path / f"{country}.nc")

    monkeypatch.setattr(gauge, "path", fake_path)
    return gauge.collect([hour], BELGIUM, duration, countries)


def test_three_keys_merge_without_colliding(monkeypatch, tmp_path):
    """SPW numbers and VMM station codes share no prefix or shape."""
    monkeypatch.setitem(gauge._GAUGE, "bel", lambda t, d: ([1.1], ["6464"],
                        numpy.array([[5.03, 51.22]])))
    monkeypatch.setitem(gauge._GAUGE, "bel_spw", lambda t, d: ([1.5], ["SPW-1043"],
                        numpy.array([[4.64, 50.73]])))
    monkeypatch.setitem(gauge._GAUGE, "bel_vmm", lambda t, d: ([2.0], ["VMM-01P03_005"],
                        numpy.array([[3.65, 51.0]])))

    ds = _collect(monkeypatch, tmp_path, ["bel", "bel_spw", "bel_vmm"])

    codes = list(map(str, ds.station.values))
    assert codes == ["6464", "SPW-1043", "VMM-01P03_005"]
    assert len(set(codes)) == len(codes)


def test_one_key_alone_is_enough(monkeypatch, tmp_path):
    """A single operator can be asked for without dragging in the others."""
    monkeypatch.setitem(gauge._GAUGE, "bel_spw", lambda t, d: ([], [], numpy.empty((0, 2))))
    monkeypatch.setitem(gauge._GAUGE, "bel_vmm", lambda t, d: ([2.0], ["VMM-01P03_005"],
                        numpy.array([[3.65, 51.0]])))

    ds = _collect(monkeypatch, tmp_path, ["bel_vmm"])

    assert list(map(str, ds.station.values)) == ["VMM-01P03_005"]
    assert list(ds.precipitation.values) == [2.0]


# ─── live endpoint ──────────────────────────────────────────────────────────

@pytest.mark.network
def test_vmm_hourly_feed_real_endpoint():
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    hour = now.replace(minute=0, second=0, microsecond=0)
    hour -= datetime.timedelta(hours=3)     # let the hourly series settle

    # the offline stubs must not reach in here: the fixture installs an
    # empty cache via monkeypatch, so anything else means it leaked and the
    # request below would carry the fixture's fake ts_ids
    assert gauge._series_cache.get("vmm") in (None, []), (
        f"offline series fixture leaked into the live test: "
        f"{gauge._series_cache.get('vmm')}"
    )

    try:
        values, codes, coords = gauge._bel_vmm(hour, datetime.timedelta(hours=1))
    except (RuntimeError, naaulu.errors.NoDataError):
        pytest.skip(f"no VMM data available at {hour}")
    except requests.HTTPError as exc:
        # 4xx means we built a bad request, which must not be skipped; 5xx
        # is the provider being down, which is not our failure to report
        if exc.response is not None and exc.response.status_code < 500:
            raise
        pytest.skip(f"VMM KiWIS unavailable: {exc}")

    assert len(values) == len(codes) == len(coords)
    assert len(values) > 20, f"expected most of the 53 VMM gauges, got {len(values)}"
    assert all(code.startswith("VMM-") for code in codes)
    # Flanders: roughly lon 2.6-5.9, lat 50.6-51.5
    assert coords[:, 0].min() > 2.0
    assert coords[:, 0].max() < 6.5
    assert coords[:, 1].min() > 50.0
    assert coords[:, 1].max() < 51.8
    assert min(values) >= 0.0
