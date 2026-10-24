"""Offline tests for the per-country IEM ASOS gauge keys."""

import datetime
import json

import numpy
import pytest
import shapely.geometry

import naaulu.errors
import naaulu.network
from naaulu import gauge


TARGET = datetime.datetime(2026, 10, 22, 13, 0)   # naive times are UTC
ONE_HOUR = datetime.timedelta(hours=1)
TEN_MINUTES = datetime.timedelta(minutes=10)
# Belgium, roughly
BELGIUM = shapely.geometry.box(2.5, 49.5, 6.4, 51.6)

# every country key and the ISO code its ASOS twin must ask for
ASOS_KEYS = {
    "bel_asos": "BE",
    "cze_asos": "CZ",
    "deu_asos": "DE",
    "est_asos": "EE",
    "fin_asos": "FI",
    "fra_asos": "FR",
    "lux_asos": "LU",
    "nld_asos": "NL",
}

GEOJSON = {
    "features": [
        {"properties": {"sid": "EBBR"},
         "geometry": {"type": "Point", "coordinates": [4.53, 50.90]}},
        {"properties": {"sid": "EBLG"},
         "geometry": {"type": "Point", "coordinates": [5.45, 50.65]}},
        {"properties": {"sid": "EBCI"},
         "geometry": {"type": "Point", "coordinates": [4.45, 50.46]}},
    ]
}

CSV = (
    "#DEBUG: Time Period -> 2026-10-22 12:30:00+00:00 2026-10-22 13:30:00+00:00\n"
    "station,valid,lon,lat,p01i\n"
    # two rows for one station: the nearest to the target hour must win
    "EBBR,2026-10-22 12:50,4.5259,50.8962,0.04\n"
    "EBBR,2026-10-22 13:20,4.5259,50.8962,0.12\n"
    # a dry reading is a reading
    "EBLG,2026-10-22 12:50,5.4500,50.6500,0.00\n"
    # no rain gauge reporting that hour
    "EBCI,2026-10-22 12:50,4.4531,50.4583,M\n"
    # a station outside the network has no coordinates
    "ZZZZ,2026-10-22 12:50,1.0000,49.0000,0.99\n"
)


@pytest.fixture
def iem(monkeypatch):
    """Stub IEM: the network geojson and the asos.py CSV, every URL recorded."""
    state = {"urls": [], "geojson": GEOJSON, "csv": CSV}

    def fake_fetch(url, filename=None, stream=False):
        state["urls"].append(url)
        if "geojson/network" in url:
            return json.dumps(state["geojson"]).encode()
        return state["csv"].encode()

    monkeypatch.setattr(naaulu.network, "fetch", fake_fetch)
    # same contract as the module attribute: a fresh cache per test
    monkeypatch.setattr(gauge, "_station_coords_cache", {})
    return state


def geojson_url(iem):
    return next(url for url in iem["urls"] if "geojson/network" in url)


def query_url(iem):
    return next(url for url in iem["urls"] if "asos.py" in url)


# ─── how a key reaches its network ──────────────────────────────────────────

def test_every_country_key_has_an_asos_twin():
    for key, iso in ASOS_KEYS.items():
        assert key in gauge._GAUGE, f"missing {key}"
        assert gauge._GAUGE[key] is getattr(gauge, key)


def test_the_asos_only_country_keys_are_gone():
    """Germany and France publish no open national feed: the twins only."""
    for key in ("deu", "fra"):
        assert key not in gauge._GAUGE
        assert not hasattr(gauge, key)


def test_an_unknown_network_key_names_itself():
    """A dropped key must not surface as a bare xarray concat error."""
    with pytest.raises(ValueError, match="Unknown gauge network.*deu") as excinfo:
        gauge.collect([TARGET], BELGIUM, ONE_HOUR, ["deu"])

    message = str(excinfo.value)
    assert "lux_asta" in message, "the available keys should be listed"
    # a helper, not an addressable key
    assert "ghcn" not in message


def test_a_network_without_any_data_says_so(monkeypatch):
    def empty(time, duration):
        raise naaulu.errors.NoDataError("nothing published")

    monkeypatch.setitem(gauge._GAUGE, "lux_asta", empty)

    with pytest.raises(naaulu.errors.NoDataError, match="No gauge data"):
        gauge.collect([TARGET], BELGIUM, ONE_HOUR, ["lux_asta"])


def test_each_twin_asks_for_its_own_iso_code(monkeypatch):
    asked = []

    def fake_asos(time, duration, country):
        asked.append(country)
        return [], [], numpy.empty((0, 2))

    monkeypatch.setattr(gauge, "asos", fake_asos)

    for key in ASOS_KEYS:
        gauge._GAUGE[key](TARGET, ONE_HOUR)

    assert asked == list(ASOS_KEYS.values())


@pytest.mark.parametrize("key,iso", sorted(ASOS_KEYS.items()))
def test_the_network_name_keeps_the_double_underscore(iem, key, iso):
    """IEM spells international networks XX__ASOS, US states XX_ASOS."""
    gauge._GAUGE[key](TARGET, ONE_HOUR)

    assert geojson_url(iem).endswith(f"/geojson/network/{iso}__ASOS.geojson")
    assert f"network={iso}__ASOS" in query_url(iem)


# ─── the request itself ─────────────────────────────────────────────────────

def test_the_window_brackets_the_target_by_thirty_minutes(iem):
    gauge._GAUGE["bel_asos"](TARGET, ONE_HOUR)

    query = query_url(iem)
    assert "year1=2026" in query and "month1=10" in query and "day1=22" in query
    assert "hour1=12" in query and "minute1=30" in query
    assert "hour2=13" in query and "minute2=30" in query
    assert "data=p01i" in query
    assert "tz=UTC" in query


def test_stations_come_from_the_network_geojson(iem):
    values, codes, coords = gauge._GAUGE["bel_asos"](TARGET, ONE_HOUR)

    # inches are converted, the dry EBLG reading stays, the station missing
    # a reading (EBCI) and the unknown one (ZZZZ) are dropped
    assert values == pytest.approx([0.04 * 25.4, 0.0])
    assert codes == ["EBBR", "EBLG"]
    # lon first: coords[:,0] is the longitude
    numpy.testing.assert_allclose(
        coords, [[4.53, 50.90], [5.45, 50.65]]
    )


def test_the_row_nearest_the_target_hour_wins(iem):
    """ASOS reports every 30 minutes: keep the closest one to the hour."""
    values, codes, _ = gauge._GAUGE["bel_asos"](TARGET, ONE_HOUR)

    ebbbr = values[codes.index("EBBR")]
    # 12:50 (0.04 in) rather than 13:20 (0.12 in)
    assert ebbbr == pytest.approx(0.04 * 25.4)


def test_a_network_without_any_positioned_station_returns_empty(iem):
    iem["geojson"] = {"features": []}
    iem["csv"] = "station,valid,lon,lat,p01i\nEBBR,2026-10-22 12:50,4.5,50.9,0.04\n"

    values, codes, coords = gauge._GAUGE["bel_asos"](TARGET, ONE_HOUR)

    assert values == [] and codes == []
    assert coords.shape == (0, 2)


# ─── failure modes ──────────────────────────────────────────────────────────

def test_only_one_hour_is_supported(iem):
    with pytest.raises(ValueError, match="1-hour"):
        gauge._GAUGE["bel_asos"](TARGET, TEN_MINUTES)


# ─── live endpoint ──────────────────────────────────────────────────────────

@pytest.mark.network
def test_bel_asos_feed_real_endpoint():
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    hour = now.replace(minute=0, second=0, microsecond=0)
    hour -= datetime.timedelta(hours=1)
    hour = hour.replace(tzinfo=None)      # naive is UTC here

    try:
        values, codes, coords = gauge._GAUGE["bel_asos"](hour, ONE_HOUR)
    except naaulu.errors.NoDataError:
        pytest.skip(f"no Belgian ASOS data available at {hour}")
    except requests.exceptions.ConnectionError as exc:
        pytest.skip(f"IEM unreachable: {exc}")
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        if status is not None and status >= 500:
            pytest.skip(f"IEM error {status}: {exc}")
        raise

    assert len(values) == len(codes) == len(coords)
    assert len(values) > 1, f"expected the Belgian ASOS stations, got {len(values)}"
    # ICAO identifiers, unlike the national feeds
    assert all(len(code) == 4 for code in codes)
    # Belgium (the network reaches into the neighbouring countries)
    assert coords[:, 1].min() > 49.0
    assert coords[:, 1].max() < 52.5
    assert min(values) >= 0.0
