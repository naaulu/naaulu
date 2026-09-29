"""Offline tests for the SPW (Wallonia) hourly gauge feed.

The KiWIS endpoint is stubbed: what matters is the window that is asked
for, how its local-offset stamps are matched against that window, which
stations survive into the merged Belgian network, and that the station
metadata is reused between calls instead of being re-fetched.
"""

import datetime
import json
import logging
import os
import time
import zoneinfo

import numpy
import pytest
import requests
import shapely.geometry

import naaulu.config
import naaulu.errors
from naaulu import gauge


TARGET = datetime.datetime(2026, 10, 9, 8, 0)       # naive times are UTC
STAMP = "2026-10-09T09:00:00.000+02:00"             # == 07:00 UTC: opens the hour
OTHER_STAMP = "2026-10-09T10:00:00.000+02:00"       # == 08:00 UTC: the next hour

TIMESERIES = [
    {"ts_id": "1", "station_no": "1043", "station_id": "9520142",
     "station_name": "ANDERLUES"},
    {"ts_id": "2", "station_no": "8063", "station_id": "12556",
     "station_name": "ANSEREMME"},
]

STATIONS = [
    # one record per parameter type: the duplicates must not multiply stations
    {"station_id": "9520142", "station_no": "1043",
     "station_latitude": "50.4239", "station_longitude": "4.2682"},
    {"station_id": "9520142", "station_no": "1043",
     "station_latitude": "50.4239", "station_longitude": "4.2682"},
    {"station_id": "12556", "station_no": "8063",
     "station_latitude": "50.2399", "station_longitude": "4.0943"},
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
            return TIMESERIES
        if request == "getStationList":
            return STATIONS
        if request == "getTimeseriesValues":
            return state["values"]
        raise AssertionError(f"unexpected request {request}")

    monkeypatch.setattr(gauge, "_kiwis_get", fake)

    # same contract as naaulu.config.get_cache_dir: build the directory
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
    """The from/to pair sent with getTimeseriesValues."""
    return next(
        (params["from"], params["to"])
        for name, params, _, _, _ in state["calls"]
        if name == "getTimeseriesValues"
    )


# ─── the request itself ─────────────────────────────────────────────────────

def test_window_is_the_ending_hour_in_utc(kiwis):
    kiwis["values"] = [block("1", [[STAMP, 2.4]])]

    gauge._bel_spw(TARGET, datetime.timedelta(hours=1))

    assert window(kiwis) == ("2026-10-09T07:00:00Z", "2026-10-09T08:00:00Z")


def test_aware_and_naive_time_ask_for_the_same_window(kiwis):
    kiwis["values"] = [block("1", [[STAMP, 2.4]])]

    gauge._bel_spw(TARGET, datetime.timedelta(hours=1))
    first = window(kiwis)

    aware = TARGET.replace(tzinfo=datetime.timezone.utc)
    gauge._bel_spw(aware, datetime.timedelta(hours=1))

    assert window(kiwis) == first


def test_all_stations_go_in_one_call(kiwis):
    kiwis["values"] = [block("1", [[STAMP, 2.4]])]

    gauge._bel_spw(TARGET, datetime.timedelta(hours=1))

    _, params, fmt, label, url = kiwis["calls"][-1]
    assert fmt == "json"              # objson is rejected by this request
    assert label == "SPW"
    assert url == gauge.SPW_KIWIS_URL
    assert params["ts_id"] == "1,2"
    assert "timespan" not in params   # silently ignored: returns latest only


# ─── how a station ends up in the dataset ───────────────────────────────────

def test_station_codes_are_prefixed_and_positions_paired(kiwis):
    kiwis["values"] = [
        block("1", [[STAMP, 2.4]]),
        block("2", [[STAMP, 0.0]]),
    ]

    values, codes, coords = gauge._bel_spw(TARGET, datetime.timedelta(hours=1))

    assert values == [2.4, 0.0]
    assert codes == ["SPW-1043", "SPW-8063"]
    # lon first: create_dataset() reads coords[:,0] as the longitude
    numpy.testing.assert_allclose(
        coords, [[4.2682, 50.4239], [4.0943, 50.2399]]
    )


def test_local_stamp_is_matched_against_utc(kiwis):
    kiwis["values"] = [block("1", [[STAMP, 2.4]])]

    values, _, _ = gauge._bel_spw(TARGET, datetime.timedelta(hours=1))

    assert values == [2.4]


def test_the_hour_is_the_one_ending_at_time(kiwis):
    """SPW stamps an hourly total with the instant the hour opens.

    naaulu labels an accumulation by the instant it closes (precip.combine,
    _bel_rmib), so the 07:00-08:00 hour is the one that belongs under
    time=08:00, not SPW's 08:00 row for 08:00-09:00.
    """
    kiwis["values"] = [
        block("1", [
            ["2026-10-09T09:00:00.000+02:00", 4.5],   # 07:00-08:00 UTC
            ["2026-10-09T10:00:00.000+02:00", 8.7],   # 08:00-09:00 UTC
        ])
    ]

    values, _, _ = gauge._bel_spw(TARGET, datetime.timedelta(hours=1))

    assert values == [4.5]


def test_stations_missing_the_hour_are_dropped(kiwis):
    # station 2 reported, but for the next hour: no value for our window
    kiwis["values"] = [
        block("1", [[STAMP, 2.4]]),
        block("2", [[OTHER_STAMP, 9.9]]),
    ]

    values, codes, coords = gauge._bel_spw(TARGET, datetime.timedelta(hours=1))

    assert values == [2.4]
    assert codes == ["SPW-1043"]
    assert coords.shape == (1, 2)


def test_null_values_are_dropped(kiwis):
    kiwis["values"] = [
        block("1", [[STAMP, None]]),
        block("2", [[STAMP, 0.0]]),
    ]

    values, codes, _ = gauge._bel_spw(TARGET, datetime.timedelta(hours=1))

    assert values == [0.0]
    assert codes == ["SPW-8063"]


# ─── failure modes ──────────────────────────────────────────────────────────

def test_only_one_hour_is_supported(kiwis):
    with pytest.raises(ValueError, match="duration"):
        gauge._bel_spw(TARGET, datetime.timedelta(minutes=10))


def test_no_station_matching_the_hour_raises(kiwis):
    kiwis["values"] = [block("1", [[OTHER_STAMP, 9.9]])]

    with pytest.raises(naaulu.errors.NoDataError, match="No SPW data"):
        gauge._bel_spw(TARGET, datetime.timedelta(hours=1))


def test_empty_payload_raises(kiwis):
    kiwis["values"] = []

    with pytest.raises(naaulu.errors.NoDataError, match="No SPW data"):
        gauge._bel_spw(TARGET, datetime.timedelta(hours=1))


# ─── how a network key reaches a provider ───────────────────────────────────

ONE_HOUR = datetime.timedelta(hours=1)
RMI_OK = ([1.1], ["6464"], numpy.array([[5.03, 51.22]]))
SPW_OK = ([1.5, 0.0], ["SPW-1043", "SPW-8063"],
          numpy.array([[4.4, 50.5], [4.9, 50.2]]))
VMM_OK = ([2.0], ["VMM-01P03_005"], numpy.array([[3.65, 51.0]]))

# covers RMI's station plus both of SPW's and VMM's
BELGIUM = shapely.geometry.box(2.0, 49.0, 7.0, 52.0)


def collect(monkeypatch, tmp_path, countries, hour=TARGET, duration=ONE_HOUR):
    """The public path: dispatch, retry, cache write, station concat."""
    def fake_path(time, duration, country):
        return str(tmp_path / f"{country}.nc")

    monkeypatch.setattr(gauge, "path", fake_path)
    return gauge.collect([hour], BELGIUM, duration, countries)


def test_belgian_networks_are_registered_separately():
    """Each operator is its own key, so the CLI can pick one."""
    assert gauge._GAUGE["bel"] is gauge._bel_rmib
    assert gauge._GAUGE["bel_spw"] is gauge._bel_spw
    assert gauge._GAUGE["bel_vmm"] is gauge._bel_vmm
    assert "bel" not in vars(gauge), "the merged reader must be gone"


def test_missing_data_stays_a_debug_line(monkeypatch, caplog, tmp_path):
    """A gauge that simply has no reading for that hour is normal."""
    def no_data(time, duration):
        raise naaulu.errors.NoDataError("No AWS data from RMI at ...")

    monkeypatch.setitem(gauge._GAUGE, "bel", no_data)
    monkeypatch.setitem(gauge._GAUGE, "bel_spw", lambda t, d: SPW_OK)
    monkeypatch.setitem(gauge._GAUGE, "bel_vmm", lambda t, d: VMM_OK)

    with caplog.at_level(logging.WARNING, logger="naaulu.gauge"):
        ds = collect(monkeypatch, tmp_path, ["bel", "bel_spw", "bel_vmm"])

    # the network that had nothing drops out, the other two survive
    assert sorted(map(str, ds.station.values)) == ["SPW-1043", "SPW-8063", "VMM-01P03_005"]
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_no_archive_still_reads_the_network(monkeypatch, tmp_path):
    """With --no-archive there is nowhere to cache: skip the write.

    Failing on os.makedirs("") here would report the network itself as
    failed, hiding the real error behind an unhelpful one.
    """
    monkeypatch.setitem(gauge._GAUGE, "bel_vmm", lambda t, d: VMM_OK)
    monkeypatch.setattr(naaulu.config, "get_archive_dir", lambda: None)

    ds = gauge.get_network(time=TARGET, duration=ONE_HOUR, country="bel_vmm")

    assert list(map(str, ds.station.values)) == ["VMM-01P03_005"]


def test_a_failed_provider_is_retried_then_warned(monkeypatch, caplog, tmp_path):
    """Throttling must not read as 'no data' and silently drop a network."""
    calls = []

    def broken(time, duration):
        calls.append(time)
        raise requests.HTTPError("429 Client Error: Too Many Requests")

    monkeypatch.setitem(gauge._GAUGE, "bel", broken)
    monkeypatch.setitem(gauge._GAUGE, "bel_spw", lambda t, d: SPW_OK)
    monkeypatch.setitem(gauge._GAUGE, "bel_vmm", lambda t, d: VMM_OK)
    monkeypatch.setattr(gauge.time, "sleep", lambda seconds: None)

    with caplog.at_level(logging.INFO, logger="naaulu.gauge"):
        ds = collect(monkeypatch, tmp_path, ["bel", "bel_spw", "bel_vmm"])

    assert len(calls) == 2, "provider failure should be retried once"
    warning = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warning) == 1
    assert "429" in warning[0].getMessage()
    assert "bel" in warning[0].getMessage()      # names the network asked for
    assert sorted(map(str, ds.station.values)) == ["SPW-1043", "SPW-8063", "VMM-01P03_005"]


def test_unsupported_duration_is_not_a_failure(monkeypatch, caplog, tmp_path):
    """SPW and VMM serve hourly only: pt10min skips them, it does not warn."""
    def unsupported(time, duration):
        raise ValueError("Unsupported duration for SPW AWS: 0:10:00")

    monkeypatch.setitem(gauge._GAUGE, "bel", lambda t, d: RMI_OK)
    monkeypatch.setitem(gauge._GAUGE, "bel_spw", unsupported)

    with caplog.at_level(logging.WARNING, logger="naaulu.gauge"):
        ds = collect(
            monkeypatch, tmp_path, ["bel", "bel_spw"],
            duration=datetime.timedelta(minutes=10),
        )

    assert list(map(str, ds.station.values)) == ["6464"]
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_each_belgian_network_writes_its_own_cache(monkeypatch, tmp_path):
    monkeypatch.setitem(gauge._GAUGE, "bel", lambda t, d: RMI_OK)
    monkeypatch.setitem(gauge._GAUGE, "bel_spw", lambda t, d: SPW_OK)
    monkeypatch.setitem(gauge._GAUGE, "bel_vmm", lambda t, d: VMM_OK)

    collect(monkeypatch, tmp_path, ["bel", "bel_spw", "bel_vmm"])

    assert sorted(p.name for p in tmp_path.glob("*.nc")) == [
        "bel.nc", "bel_spw.nc", "bel_vmm.nc",
    ]


# ─── timezones and DST ─────────────────────────────────────────────────────

@pytest.mark.parametrize("stamp,expected", [
    ("2026-07-15T12:00:00.000+02:00", datetime.datetime(2026, 7, 15, 10)),  # CEST
    ("2026-01-15T11:00:00.000+01:00", datetime.datetime(2026, 1, 15, 10)),  # CET
    ("2026-03-29T03:00:00.000+02:00", datetime.datetime(2026, 3, 29, 1)),   # past the gap
    ("2025-10-26T02:00:00.000+02:00", datetime.datetime(2025, 10, 26, 0)),  # first 02:00
    ("2025-10-26T02:00:00.000+01:00", datetime.datetime(2025, 10, 26, 1)),  # repeated 02:00
    ("2026-07-15T14:00:00", datetime.datetime(2026, 7, 15, 12)),            # no offset
])
def test_stamp_parsing_is_dst_safe(stamp, expected):
    assert gauge._kiwis_parse_stamp(stamp) == expected


def test_naive_stamp_is_brussels_not_the_host_timezone():
    """astimezone() on a naive datetime assumes the host clock.

    SPW times are Brussels times, so a host in Tokyo must not change the
    answer - only an explicit Europe/Brussels reference keeps it fixed.
    """
    if not hasattr(time, "tzset"):
        pytest.skip("time.tzset() is POSIX-only")

    original = os.environ.get("TZ")
    try:
        os.environ["TZ"] = "Asia/Tokyo"
        time.tzset()

        parsed = gauge._kiwis_parse_stamp("2026-07-15T14:00:00")
    finally:
        if original is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = original
        time.tzset()

    assert parsed == datetime.datetime(2026, 7, 15, 12)


def test_the_repeated_autumn_hour_stays_two_hours(kiwis):
    """2025-10-26: 02:00 local happens twice, one hour apart in UTC.

    Matching on wall clock would collapse them and hand the same rainfall
    to both windows.
    """
    kiwis["values"] = [
        block("1", [
            ["2025-10-26T02:00:00.000+02:00", 3.0],   # opens 00:00Z
            ["2025-10-26T02:00:00.000+01:00", 7.0],   # opens 01:00Z
        ])
    ]

    first, _, _ = gauge._bel_spw(
        datetime.datetime(2025, 10, 26, 1, 0), datetime.timedelta(hours=1)
    )
    second, _, _ = gauge._bel_spw(
        datetime.datetime(2025, 10, 26, 2, 0), datetime.timedelta(hours=1)
    )

    assert first == [3.0]
    assert second == [7.0]


def test_rmi_query_is_utc_whatever_timezone_the_caller_holds(monkeypatch):
    """_bel_rmib used to strftime() the caller's own clock and label it Z.

    _bel_spw normalises with naive_utc() first; both networks must ask for
    the same hour when handed the same instant.
    """
    seen = {}

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "features": [
                    {
                        "properties": {"precip_quantity": 1.0, "code": 42},
                        "geometry": {"coordinates": [4.4, 50.5]},
                    }
                ]
            }

    class FakeSession:
        def get(self, url, params=None, timeout=None):
            seen.update(params)
            return Response()

    # _bel_rmib goes through the shared session, not a bare requests.get
    monkeypatch.setattr(
        gauge.naaulu.network, "runtime_session", lambda: FakeSession()
    )

    brussels = datetime.datetime(
        2026, 10, 9, 8, 0, tzinfo=zoneinfo.ZoneInfo("Europe/Brussels")
    )
    values, codes, coords = gauge._bel_rmib(
        brussels, datetime.timedelta(hours=1)
    )

    # 08:00 in Brussels is 06:00 UTC
    assert seen["CQL_FILTER"] == "timestamp='2026-10-09T06:00:00Z'"
    assert seen["typeName"] == "aws:aws_1hour"
    assert values == [1.0] and codes == ["42"]
    numpy.testing.assert_allclose(coords, [[4.4, 50.5]])


# ─── station metadata caching ───────────────────────────────────────────────

def test_station_metadata_is_read_once(kiwis):
    kiwis["values"] = [block("1", [[STAMP, 2.4]])]

    gauge._bel_spw(TARGET, datetime.timedelta(hours=1))
    gauge._bel_spw(TARGET, datetime.timedelta(hours=1))

    fetched = [name for name, *_ in kiwis["calls"] if name != "getTimeseriesValues"]
    assert fetched == ["getTimeseriesList", "getStationList"]


def test_station_metadata_survives_without_the_in_process_cache(monkeypatch, kiwis, tmp_path):
    kiwis["values"] = [block("1", [[STAMP, 2.4]])]

    gauge._bel_spw(TARGET, datetime.timedelta(hours=1))
    assert os.path.exists(os.path.join(tmp_path, "spw", "series.json"))

    # a later run starts with an empty in-process cache
    monkeypatch.setattr(gauge, "_series_cache", {})
    gauge._bel_spw(TARGET, datetime.timedelta(hours=1))

    fetched = [name for name, *_ in kiwis["calls"] if name != "getTimeseriesValues"]
    assert fetched == ["getTimeseriesList", "getStationList"]


def test_expired_station_metadata_is_refetched(monkeypatch, kiwis, tmp_path):
    kiwis["values"] = [block("1", [[STAMP, 2.4]])]

    gauge._bel_spw(TARGET, datetime.timedelta(hours=1))
    filename = tmp_path / "spw" / "series.json"
    cached = json.loads(filename.read_text())
    cached["generated"] -= gauge.KIWIS_MAX_AGE + 1
    filename.write_text(json.dumps(cached))

    monkeypatch.setattr(gauge, "_series_cache", {})
    gauge._bel_spw(TARGET, datetime.timedelta(hours=1))

    fetched = [name for name, *_ in kiwis["calls"] if name != "getTimeseriesValues"]
    assert fetched == ["getTimeseriesList", "getStationList"] * 2


# ─── the KiWIS transport ────────────────────────────────────────────────────

class FakeResponse:
    def __init__(self, payload=None, text="", json_error=False):
        self._payload = payload
        self.text = text
        self._json_error = json_error

    def raise_for_status(self):
        pass

    def json(self):
        if self._json_error:
            raise ValueError("no json")
        return self._payload


@pytest.fixture
def kiwis_session(monkeypatch):
    """Patch the HTTP session behind _kiwis_get."""
    state = {"response": None, "query": None}

    class FakeSession:
        def get(self, url, params=None, timeout=None):
            state["query"] = params
            return state["response"]

    monkeypatch.setattr(
        "naaulu.network.runtime_session", lambda: FakeSession()
    )
    return state


def test_transport_builds_an_anonymous_query(kiwis_session):
    kiwis_session["response"] = FakeResponse(payload=[])

    gauge._kiwis_get(gauge.SPW_KIWIS_URL, {"request": "getStationList"}, "objson", "SPW")

    query = kiwis_session["query"]
    assert query["request"] == "getStationList"
    assert query["format"] == "objson"
    assert query["service"] == "kisters"
    assert query["type"] == "queryServices"
    assert not any(
        key in query for key in ("auth_key", "apikey", "user", "password")
    )


def test_transport_raises_on_a_kiwis_error_object(kiwis_session):
    kiwis_session["response"] = FakeResponse(
        payload={"type": "error", "code": "InvalidParameterValue",
                 "message": "Service parameter is unknown."}
    )

    with pytest.raises(RuntimeError, match="Service parameter"):
        gauge._spw_get({"request": "getTimeseriesValues"}, "json")


def test_transport_raises_on_an_xml_exception_report(kiwis_session):
    kiwis_session["response"] = FakeResponse(
        text="<?xml version='1.0' ?><ExceptionReport/>", json_error=True
    )

    with pytest.raises(RuntimeError, match="no JSON"):
        gauge._spw_get({"request": "getTimeseriesValues"}, "json")


# ─── live endpoint ──────────────────────────────────────────────────────────

@pytest.mark.network
def test_spw_hourly_feed_real_endpoint():
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    hour = now.replace(minute=0, second=0, microsecond=0)
    hour -= datetime.timedelta(hours=3)     # let the hourly series settle

    # the offline stubs must not reach in here: the fixture installs an
    # empty cache via monkeypatch, so anything else means it leaked and the
    # request below would carry the fixture's fake ts_ids
    assert gauge._series_cache.get("spw") in (None, []), (
        f"offline series fixture leaked into the live test: "
        f"{gauge._series_cache.get('spw')}"
    )

    try:
        values, codes, coords = gauge._bel_spw(
            hour, datetime.timedelta(hours=1)
        )
    except (RuntimeError, naaulu.errors.NoDataError):
        pytest.skip(f"no SPW data available at {hour}")
    except requests.HTTPError as exc:
        # 4xx means we built a bad request, which must not be skipped; 5xx
        # is the provider being down, which is not our failure to report
        if exc.response is not None and exc.response.status_code < 500:
            raise
        pytest.skip(f"SPW KiWIS unavailable: {exc}")

    assert len(values) == len(codes) == len(coords)
    assert len(values) > 20, f"expected most of the 97 SPW gauges, got {len(values)}"
    assert all(code.startswith("SPW-") for code in codes)
    # Wallonia: roughly lon 2.8-6.4, lat 49.5-50.9
    assert coords[:, 0].min() > 2.0
    assert coords[:, 0].max() < 7.0
    assert coords[:, 1].min() > 49.0
    assert coords[:, 1].max() < 51.5
    assert min(values) >= 0.0
