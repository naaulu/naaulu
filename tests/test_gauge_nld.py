"""Offline tests for the NLD (KNMI) near-real-time gauge feed."""

import datetime
import os

import numpy
import pytest
import requests
import shapely.geometry
import xarray

import naaulu.config
import naaulu.errors
import naaulu.network
from naaulu import gauge


TARGET = datetime.datetime(2026, 10, 17, 15, 0)   # naive times are UTC
ONE_HOUR = datetime.timedelta(hours=1)
TWO_HOURS = datetime.timedelta(hours=2)
# the Netherlands, roughly mainland
NETHERLANDS = shapely.geometry.box(3.3, 50.7, 7.3, 53.6)


def write_knmi_file(directory, stations, r1h):
    """A minimal 10-minute KNMI file: one instant, one R1H column."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "KMDS__OPER_P___10M_OBS_L2_202610171500.nc")
    dataset = xarray.Dataset(
        {
            "R1H": (("station", "time"), numpy.array([r1h], dtype=float).T),
        },
        coords={
            "station": ("station", [code for code, _, _ in stations]),
            "time": [numpy.datetime64("2026-10-17T15:00:00")],
            "lon": ("station", [lon for _, lon, _ in stations]),
            "lat": ("station", [lat for _, _, lat in stations]),
        },
    )
    dataset.to_netcdf(path, engine="h5netcdf")
    return path


@pytest.fixture
def knmi(monkeypatch, tmp_path):
    """Stub the KNMI transport: the /url handshake and the file download."""
    state = {"requests": [], "filepath": None, "status": 200}
    # the real lookup, for the test that goes through the environment
    state["real_get_secret"] = naaulu.config.get_secret

    def fake_get_temp(url, key, filename, field):
        state["requests"].append({"url": url, "key": key, "filename": filename,
                                  "field": field})
        if state["status"] == 404:
            # what naaulu.network.get_temp raises for an unpublished file
            raise FileNotFoundError(url)
        return state["filepath"]

    monkeypatch.setattr(naaulu.network, "get_temp", fake_get_temp)
    # empty: the shared anonymous key is the fallback
    monkeypatch.setattr(naaulu.config, "get_secret", lambda name: None)
    return state


# ─── the request itself ─────────────────────────────────────────────────────

def test_file_is_named_after_the_interval_end(knmi, tmp_path):
    """No listing call: the file name carries the instant."""
    knmi["filepath"] = write_knmi_file(
        str(tmp_path), [("06209", 5.2, 52.1)], [1.5]
    )

    gauge.nld(TARGET, ONE_HOUR)

    request = knmi["requests"][-1]
    assert request["filename"] == "KMDS__OPER_P___10M_OBS_L2_202610171500.nc"
    assert request["filename"] in request["url"]
    assert request["url"].startswith(gauge.KNMI_OPEN_DATA)
    assert "10-minute-in-situ-meteorological-observations" in request["url"]
    assert request["field"] == "temporaryDownloadUrl"


def test_anonymous_key_is_the_fallback(knmi, tmp_path):
    knmi["filepath"] = write_knmi_file(
        str(tmp_path), [("06209", 5.2, 52.1)], [1.5]
    )

    gauge.nld(TARGET, ONE_HOUR)
    assert knmi["requests"][-1]["key"] == gauge.KNMI_ANONYMOUS_KEY


def test_personal_key_overrides_the_anonymous_one(knmi, tmp_path, monkeypatch):
    knmi["filepath"] = write_knmi_file(
        str(tmp_path), [("06209", 5.2, 52.1)], [1.5]
    )
    monkeypatch.setattr(naaulu.config, "get_secret", lambda name: "personal-key")

    gauge.nld(TARGET, ONE_HOUR)

    assert knmi["requests"][-1]["key"] == "personal-key"


def test_the_key_can_come_from_the_environment(knmi, tmp_path, monkeypatch):
    """No keyring needed: the container exports KNMI_API_KEY and that is used."""
    knmi["filepath"] = write_knmi_file(
        str(tmp_path), [("06209", 5.2, 52.1)], [1.5]
    )
    monkeypatch.setattr(naaulu.config, "get_secret", knmi["real_get_secret"])
    monkeypatch.setenv("KNMI_API_KEY", "from-the-environment")

    gauge.nld(TARGET, ONE_HOUR)

    assert knmi["requests"][-1]["key"] == "from-the-environment"


# ─── how a station ends up in the dataset ───────────────────────────────────

def test_values_come_from_r1h_and_positions_from_the_file(knmi, tmp_path):
    knmi["filepath"] = write_knmi_file(
        str(tmp_path),
        [("06209", 5.20, 52.10), ("06225", 4.47, 52.47)],
        [1.5, 0.0],
    )

    values, codes, coords = gauge.nld(TARGET, ONE_HOUR)

    assert values == [1.5, 0.0]
    assert codes == ["06209", "06225"]
    # lon first: coords[:,0] is the longitude
    numpy.testing.assert_allclose(coords, [[5.20, 52.10], [4.47, 52.47]])


def test_stations_without_a_gauge_are_dropped(knmi, tmp_path):
    """Wind poles report no R1H; they must not appear as zeros."""
    knmi["filepath"] = write_knmi_file(
        str(tmp_path),
        [("06201", 4.0, 53.0), ("06209", 5.20, 52.10)],
        [numpy.nan, 2.25],
    )

    values, codes, coords = gauge.nld(TARGET, ONE_HOUR)

    assert values == [2.25]
    assert codes == ["06209"]
    assert coords.shape == (1, 2)


# ─── failure modes ──────────────────────────────────────────────────────────

def test_only_one_hour_is_supported(knmi, tmp_path):
    knmi["filepath"] = write_knmi_file(
        str(tmp_path), [("06209", 5.2, 52.1)], [1.5]
    )

    with pytest.raises(ValueError, match="duration"):
        gauge.nld(TARGET, datetime.timedelta(minutes=10))


def test_an_unpublished_file_is_no_data(knmi, tmp_path):
    """Too fresh to exist, or too old to be archived: both are NoDataError."""
    knmi["status"] = 404

    with pytest.raises(naaulu.errors.NoDataError, match="no KMDS"):
        gauge.nld(TARGET, ONE_HOUR)


# ─── the public path ────────────────────────────────────────────────────────

def test_two_hours_sum_the_two_hourly_files(knmi, tmp_path, monkeypatch):
    """PT2H rides on the existing composition: two R1H files, summed."""
    knmi["files"] = {
        TARGET: write_knmi_file(
            str(tmp_path / "a"), [("06209", 5.2, 52.1)], [1.5]
        ),
        TARGET - ONE_HOUR: write_knmi_file(
            str(tmp_path / "b"), [("06209", 5.2, 52.1)], [0.75]
        ),
    }

    def fake_get_temp(url, key, filename, field):
        time = _target_of(filename)
        return knmi["files"][time]

    monkeypatch.setattr(naaulu.network, "get_temp", fake_get_temp)

    def fake_path(time, duration, country):
        return str(tmp_path / f"{time:%Y%m%d%H%M}.nc")

    monkeypatch.setattr(gauge, "path", fake_path)

    ds = gauge.get_network(time=TARGET, duration=TWO_HOURS, country="nld")

    assert list(ds.precipitation.values) == pytest.approx([2.25])


def _target_of(filename):
    stamp = filename.removesuffix(".nc").rsplit("_", 1)[-1]
    return datetime.datetime.strptime(stamp, "%Y%m%d%H%M")


# ─── live endpoint ──────────────────────────────────────────────────────────

@pytest.mark.network
def test_knmi_hourly_feed_real_endpoint():
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    hour = now.replace(minute=0, second=0, microsecond=0)
    hour -= datetime.timedelta(hours=1)   # the file lands ~3m40s after the hour
    hour = hour.replace(tzinfo=None)      # naive is UTC here

    try:
        values, codes, coords = gauge.nld(hour, ONE_HOUR)
    except naaulu.errors.NoDataError:
        pytest.skip(f"no KNMI data available at {hour}")
    except requests.exceptions.ConnectionError as exc:
        pytest.skip(f"KNMI Data Platform unreachable: {exc}")
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        if status == 429:
            pytest.skip(f"KNMI anonymous key rate limited: {exc}")
        if status is not None and status >= 500:
            pytest.skip(f"KNMI Data Platform error {status}: {exc}")
        raise

    assert len(values) == len(codes) == len(coords)
    assert len(values) > 10, f"expected the KNMI rain gauges, got {len(values)}"
    # the Netherlands (incl. the Caribbean stations far outside)
    assert coords[:, 1].max() < 54.0
    assert min(values) >= 0.0
