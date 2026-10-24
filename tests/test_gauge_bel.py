"""Offline tests for the SPW (Wallonia) hourly gauge feed."""

import datetime
import zoneinfo

import numpy
import pytest
import requests
import shapely.geometry

import naaulu.errors
from naaulu import gauge


TARGET = datetime.datetime(2026, 10, 9, 8, 0)       # naive times are UTC
STAMP = "2026-10-09T09:00:00.000+02:00"             # 07:00 UTC: opens the hour
OTHER_STAMP = "2026-10-09T10:00:00.000+02:00"       # 08:00 UTC: the next hour

TIMESERIES = [
    {"ts_id": "1", "station_no": "1043", "station_id": "9520142",
     "station_name": "ANDERLUES"},
    {"ts_id": "2", "station_no": "8063", "station_id": "12556",
     "station_name": "ANSEREMME"},
]

STATIONS = [
    # duplicate records must not multiply stations
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
    """Stub the KiWIS transport, recording every call."""
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

    # same contract as get_cache_dir: create the directory
    def cache_dir(subdir=None):
        path = tmp_path / subdir if subdir else tmp_path
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    monkeypatch.setattr("naaulu.config.get_cache_dir", cache_dir)
    # an assignment: monkeypatch must be able to undo it
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


# ─── how a station ends up in the dataset ───────────────────────────────────

def test_station_codes_are_prefixed_and_positions_paired(kiwis):
    kiwis["values"] = [
        block("1", [[STAMP, 2.4]]),
        block("2", [[STAMP, 0.0]]),
    ]

    values, codes, coords = gauge._bel_spw(TARGET, datetime.timedelta(hours=1))

    assert values == [2.4, 0.0]
    assert codes == ["SPW-1043", "SPW-8063"]
    # lon first: coords[:,0] is the longitude
    numpy.testing.assert_allclose(
        coords, [[4.2682, 50.4239], [4.0943, 50.2399]]
    )


def test_the_hour_is_the_one_ending_at_time(kiwis):
    """The row asked for is the hour ending at `time`."""
    kiwis["values"] = [
        block("1", [
            ["2026-10-09T09:00:00.000+02:00", 4.5],   # 07:00-08:00 UTC
            ["2026-10-09T10:00:00.000+02:00", 8.7],   # 08:00-09:00 UTC
        ])
    ]

    values, _, _ = gauge._bel_spw(TARGET, datetime.timedelta(hours=1))

    assert values == [4.5]


def test_stations_missing_the_hour_are_dropped(kiwis):
    # station 2 reported for the next hour only
    kiwis["values"] = [
        block("1", [[STAMP, 2.4]]),
        block("2", [[OTHER_STAMP, 9.9]]),
    ]

    values, codes, coords = gauge._bel_spw(TARGET, datetime.timedelta(hours=1))

    assert values == [2.4]
    assert codes == ["SPW-1043"]
    assert coords.shape == (1, 2)


# ─── failure modes ──────────────────────────────────────────────────────────

def test_only_one_hour_is_supported(kiwis):
    with pytest.raises(ValueError, match="duration"):
        gauge._bel_spw(TARGET, datetime.timedelta(minutes=10))


def test_no_station_matching_the_hour_raises(kiwis):
    kiwis["values"] = [block("1", [[OTHER_STAMP, 9.9]])]

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


def test_each_belgian_network_writes_its_own_cache(monkeypatch, tmp_path):
    monkeypatch.setitem(gauge._GAUGE, "bel", lambda t, d: RMI_OK)
    monkeypatch.setitem(gauge._GAUGE, "bel_spw", lambda t, d: SPW_OK)
    monkeypatch.setitem(gauge._GAUGE, "bel_vmm", lambda t, d: VMM_OK)

    collect(monkeypatch, tmp_path, ["bel", "bel_spw", "bel_vmm"])

    assert sorted(p.name for p in tmp_path.glob("*.nc")) == [
        "bel.nc", "bel_spw.nc", "bel_vmm.nc",
    ]


def test_rmi_query_is_utc_whatever_timezone_the_caller_holds(monkeypatch):
    """The RMI query must ask for the same hour whatever the caller's timezone."""
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

    # _bel_rmib goes through the shared session
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


# ─── live endpoint ──────────────────────────────────────────────────────────

@pytest.mark.network
def test_spw_hourly_feed_real_endpoint():
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    hour = now.replace(minute=0, second=0, microsecond=0)
    hour -= datetime.timedelta(hours=3)     # let the hourly series settle

    # the offline fixture must not have leaked its fake series
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
        # 4xx is a bad request of ours and must not be skipped; 5xx is the
        # provider being down
        if exc.response is not None and exc.response.status_code < 500:
            raise
        pytest.skip(f"SPW KiWIS unavailable: {exc}")

    assert len(values) == len(codes) == len(coords)
    assert len(values) > 20, f"expected most of the 97 SPW gauges, got {len(values)}"
    assert all(code.startswith("SPW-") for code in codes)
    # Wallonia
    assert coords[:, 0].min() > 2.0
    assert coords[:, 0].max() < 7.0
    assert coords[:, 1].min() > 49.0
    assert coords[:, 1].max() < 51.5
    assert min(values) >= 0.0
