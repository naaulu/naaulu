"""Offline tests for the Luxembourg (geoportail INSPIRE WFS) gauge feeds."""

import datetime
import zoneinfo

import numpy
import pytest
import shapely.geometry

from naaulu import gauge


TARGET = datetime.datetime(2026, 10, 21, 12, 0)   # naive times are UTC
ONE_HOUR = datetime.timedelta(hours=1)
TWO_HOURS = datetime.timedelta(hours=2)
# the GMT+1 label carrying the hour ending 12:00 UTC
STAMP = "2026-10-21T13:00:00Z"
OTHER_STAMP = "2026-10-21T14:00:00Z"
# Luxembourg, roughly
LUXEMBOURG = shapely.geometry.box(5.7, 49.4, 6.6, 50.2)

ASTA_ARCHIVE = "MF.PointTimeSeriesObservation_Hourly_ASTA_sum_nn050"
ASTA_LIVE = "MF.PointTimeSeriesObservation_Live_ASTA_sum_nn050"
AGE_ARCHIVE = "MF.PointTimeSeriesObservation_Hourly_AGE_sum_nn050"


def feature(station, value, lon=5.9, lat=49.6, stamp=STAMP, href=True):
    """One GeoJSON observation as the WFS returns it."""
    properties = {
        "datetime": stamp,
        "name_descr": station,
        "sum_nn050": value,
    }
    if href:
        properties["featureofinterest_xlink_href"] = (
            "http://catalog.inspire.geoportail.lu/geonetwork/dataset/"
            f"00b35d13-111e-45c7-bc65-556b61d17c7e/{station}"
        )
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "properties": properties,
    }


@pytest.fixture
def wfs(monkeypatch):
    """Stub the WFS transport: canned features per (layer, stamp), calls recorded."""
    state = {"calls": [], "responses": {}}

    def fake(url, typename, stamp):
        state["calls"].append({"url": url, "typename": typename, "stamp": stamp})
        responses = state["responses"]
        return responses.get((typename, stamp), responses.get(typename, []))

    monkeypatch.setattr(gauge, "_lu_features", fake)
    return state


def asked(wfs):
    return [call["typename"] for call in wfs["calls"]]


# ─── the request itself ─────────────────────────────────────────────────────

def test_the_stamp_is_the_gmt_plus_one_label_of_the_ending_hour(wfs):
    wfs["responses"] = {ASTA_ARCHIVE: [feature("AGM_043", 1.2)]}

    gauge.lux_asta(TARGET, ONE_HOUR)

    assert wfs["calls"][0]["stamp"] == STAMP


def test_the_stamp_rolls_over_midnight(wfs):
    wfs["responses"] = {ASTA_ARCHIVE: [feature("AGM_043", 1.2)]}

    gauge.lux_asta(datetime.datetime(2026, 10, 21, 23, 0), ONE_HOUR)

    assert wfs["calls"][0]["stamp"] == "2026-10-22T00:00:00Z"


def test_an_aware_caller_is_converted_to_utc_first(wfs):
    """08:00 in Brussels on 21 October is 06:00 UTC, so labelled 07:00."""
    wfs["responses"] = {ASTA_ARCHIVE: [feature("AGM_043", 1.2)]}
    brussels = datetime.datetime(
        2026, 10, 21, 8, 0, tzinfo=zoneinfo.ZoneInfo("Europe/Brussels")
    )

    gauge.lux_asta(brussels, ONE_HOUR)

    assert wfs["calls"][0]["stamp"] == "2026-10-21T07:00:00Z"


# ─── how a station ends up in the dataset ───────────────────────────────────

def test_values_codes_and_positions_come_from_the_feature(wfs):
    wfs["responses"] = {ASTA_ARCHIVE: [
        feature("AGM_043", 1.2, lon=5.8755, lat=49.5018),
        feature("AGM_103", 0.0, lon=6.0390, lat=49.4786),
    ]}

    values, codes, coords = gauge.lux_asta(TARGET, ONE_HOUR)

    assert values == [1.2, 0.0]
    # zero rain is a reading, the prefix separates ASTA from AGE
    assert codes == ["ASTA-AGM_043", "ASTA-AGM_103"]
    # lon first: coords[:,0] is the longitude
    numpy.testing.assert_allclose(
        coords, [[5.8755, 49.5018], [6.0390, 49.4786]]
    )


def test_a_null_reading_is_dropped_not_zero_filled(wfs):
    wfs["responses"] = {ASTA_ARCHIVE: [
        feature("AGM_043", 1.2),
        feature("AGM_103", None),
    ]}

    values, codes, coords = gauge.lux_asta(TARGET, ONE_HOUR)

    assert values == [1.2]
    assert codes == ["ASTA-AGM_043"]
    assert coords.shape == (1, 2)


def test_the_place_name_backs_up_a_missing_station_href(wfs):
    wfs["responses"] = {ASTA_ARCHIVE: [feature("Arsdorf", 0.4, href=False)]}

    values, codes, _ = gauge.lux_asta(TARGET, ONE_HOUR)

    assert values == [0.4]
    assert codes == ["ASTA-Arsdorf"]


def test_a_feature_without_a_position_is_dropped(wfs):
    no_geometry = feature("AGM_043", 1.2)
    no_geometry["geometry"] = None
    wfs["responses"] = {ASTA_ARCHIVE: [no_geometry, feature("AGM_103", 0.6)]}

    values, codes, coords = gauge.lux_asta(TARGET, ONE_HOUR)

    assert codes == ["ASTA-AGM_103"]
    numpy.testing.assert_allclose(coords, [[5.9, 49.6]])


# ─── which layer answers ────────────────────────────────────────────────────

def test_the_archive_is_asked_first_and_wins(wfs):
    wfs["responses"] = {
        ASTA_ARCHIVE: [feature("AGM_043", 1.2)],
        ASTA_LIVE: [feature("AGM_043", 9.9)],
    }

    values, _, _ = gauge.lux_asta(TARGET, ONE_HOUR)

    assert values == [1.2]
    assert asked(wfs) == [ASTA_ARCHIVE]


def test_the_live_layer_covers_the_freshest_hour(wfs):
    wfs["responses"] = {ASTA_LIVE: [feature("AGM_043", 0.8)]}

    values, codes, _ = gauge.lux_asta(TARGET, ONE_HOUR)

    assert values == [0.8]
    assert codes == ["ASTA-AGM_043"]
    assert asked(wfs) == [ASTA_ARCHIVE, ASTA_LIVE]


def test_age_has_no_live_layer_and_uses_its_own_typename(wfs):
    wfs["responses"] = {AGE_ARCHIVE: [feature("AGE_054", 0.3)]}

    values, codes, coords = gauge.lux_age(TARGET, ONE_HOUR)

    assert values == [0.3]
    assert codes == ["AGE-AGE_054"]
    assert asked(wfs) == [AGE_ARCHIVE]
    assert wfs["calls"][0]["url"] == gauge.LU_WFS_URL


# ─── the WFS transport ──────────────────────────────────────────────────────

class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


@pytest.fixture
def session(monkeypatch):
    """Patch the HTTP session behind _lu_features."""
    state = {"payload": {"features": []}, "url": None, "query": None, "timeout": None}

    class FakeSession:
        def get(self, url, params=None, timeout=None):
            state["url"] = url
            state["query"] = params
            state["timeout"] = timeout
            return FakeResponse(state["payload"])

    monkeypatch.setattr("naaulu.network.runtime_session", lambda: FakeSession())
    return state


def test_the_query_selects_the_single_hour_as_geojson(session):
    session["payload"] = {"features": [feature("AGM_043", 1.2)]}

    features = gauge._lu_features(gauge.LU_WFS_URL, ASTA_ARCHIVE, STAMP)

    assert len(features) == 1
    assert session["url"] == gauge.LU_WFS_URL
    query = session["query"]
    assert query["service"] == "wfs"
    assert query["version"] == "2.0.0"
    assert query["request"] == "GetFeature"
    assert query["typeName"] == ASTA_ARCHIVE
    assert query["outputFormat"] == "application/json"
    assert query["srsName"] == "EPSG:4326"
    assert query["CQL_FILTER"] == f"datetime = {STAMP}"
    assert session["timeout"] == 30


# ─── how a network key reaches a provider ───────────────────────────────────

LUX_ATA_OK = ([1.5], ["ASTA-AGM_043"], numpy.array([[5.9, 49.6]]))
LUX_AGE_OK = ([0.7], ["AGE-AGE_054"], numpy.array([[6.1, 50.1]]))


def collect(monkeypatch, tmp_path, countries, hour=TARGET, duration=ONE_HOUR):
    """The public path: dispatch, retry, cache write, station concat."""
    def fake_path(time, duration, country):
        return str(tmp_path / f"{country}.nc")

    monkeypatch.setattr(gauge, "path", fake_path)
    return gauge.collect([hour], LUXEMBOURG, duration, countries)


def test_the_luxembourg_networks_are_registered_and_lux_stays_free():
    assert gauge._GAUGE["lux_asta"] is gauge.lux_asta
    assert gauge._GAUGE["lux_age"] is gauge.lux_age
    # reserved for MeteoLux, which has no time-series feed
    assert "lux" not in gauge._GAUGE


def test_each_network_writes_its_own_cache(monkeypatch, tmp_path):
    monkeypatch.setitem(gauge._GAUGE, "lux_asta", lambda t, d: LUX_ATA_OK)
    monkeypatch.setitem(gauge._GAUGE, "lux_age", lambda t, d: LUX_AGE_OK)

    collect(monkeypatch, tmp_path, ["lux_asta", "lux_age"])

    assert sorted(p.name for p in tmp_path.glob("*.nc")) == [
        "lux_age.nc", "lux_asta.nc",
    ]


def test_both_networks_reach_their_own_layer(monkeypatch, tmp_path, wfs):
    wfs["responses"] = {
        ASTA_ARCHIVE: [feature("AGM_043", 1.5)],
        AGE_ARCHIVE: [feature("AGE_054", 0.7, lon=6.1, lat=50.1)],
    }

    gauges = collect(monkeypatch, tmp_path, ["lux_asta", "lux_age"])

    assert sorted(gauges.station.values) == ["AGE-AGE_054", "ASTA-AGM_043"]
    assert set(asked(wfs)) == {ASTA_ARCHIVE, AGE_ARCHIVE}


def test_two_hours_sum_the_two_hourly_rows(wfs, monkeypatch, tmp_path):
    """PT2H rides on the existing composition: two stamps, summed."""
    wfs["responses"] = {
        (ASTA_ARCHIVE, STAMP): [feature("AGM_043", 1.5)],
        (ASTA_ARCHIVE, "2026-10-21T12:00:00Z"): [feature("AGM_043", 0.75)],
    }

    def fake_path(time, duration, country):
        return str(tmp_path / f"{time:%Y%m%d%H}.nc")

    monkeypatch.setattr(gauge, "path", fake_path)

    ds = gauge.get_network(time=TARGET, duration=TWO_HOURS, country="lux_asta")

    assert list(ds.precipitation.values) == pytest.approx([2.25])
