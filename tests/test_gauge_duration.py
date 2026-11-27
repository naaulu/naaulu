"""Offline tests for gauge durations the providers do not serve natively."""

import datetime

import numpy
import pytest

import naaulu.errors
import naaulu.util
from naaulu import gauge


TARGET = datetime.datetime(2026, 10, 17, 15, 0)   # naive times are UTC
ONE_HOUR = datetime.timedelta(hours=1)
TWO_HOURS = datetime.timedelta(hours=2)
TEN_MINUTES = datetime.timedelta(minutes=10)


def reading(**stations):
    """The provider-shaped (values, codes, coords) for code=value pairs."""
    codes = list(stations)
    values = [stations[code] for code in codes]
    coords = numpy.array([[3.0 + i, 50.0 + i] for i in range(len(codes))])
    return values, codes, coords


@pytest.fixture
def network(monkeypatch, tmp_path):
    """A stub provider serving only what the test staged."""
    state = {"calls": [], "values": {}}

    def provider(time, duration):
        state["calls"].append((time, duration))
        if (time, duration) not in state["values"]:
            raise naaulu.errors.NoDataError(f"no stub data at {time} for {duration}")
        return state["values"][(time, duration)]

    monkeypatch.setitem(gauge._GAUGE, "stub", provider)

    def path(time, duration, country):
        filename = (
            f"{naaulu.util.format_time(time)}."
            f"{naaulu.util.format_duration(duration)}.{country}.nc"
        )
        return str(tmp_path / filename)

    monkeypatch.setattr(gauge, "path", path)
    return state


def test_two_hours_sum_the_two_ending_hours(network):
    """2h ending 15:00 = hour ending 15:00 + hour ending 14:00."""
    network["values"] = {
        (TARGET, ONE_HOUR): reading(A=3.0),
        (TARGET - ONE_HOUR, ONE_HOUR): reading(A=1.5),
    }

    ds = gauge.get_network(time=TARGET, duration=TWO_HOURS, country="stub")

    assert network["calls"] == [
        (TARGET, ONE_HOUR),
        (TARGET - ONE_HOUR, ONE_HOUR),
    ]
    assert list(map(str, ds.station.values)) == ["A"]
    assert ds.precipitation.values == [4.5]
    # the units must survive the sum
    assert ds.precipitation.attrs["units"] == "mm"


def test_a_station_missing_one_hour_drops_out(network):
    """Only stations present in every part remain."""
    network["values"] = {
        (TARGET, ONE_HOUR): reading(A=3.0, B=9.9),
        (TARGET - ONE_HOUR, ONE_HOUR): reading(A=1.5),
    }

    ds = gauge.get_network(time=TARGET, duration=TWO_HOURS, country="stub")

    assert list(map(str, ds.station.values)) == ["A"]
    assert ds.precipitation.values == [4.5]


def test_twenty_minutes_sum_ten_minute_parts(network):
    """10 minutes is the largest native step that divides 20 minutes."""
    network["values"] = {
        (TARGET, TEN_MINUTES): reading(A=0.4),
        (TARGET - TEN_MINUTES, TEN_MINUTES): reading(A=0.2),
    }

    ds = gauge.get_network(
        time=TARGET, duration=datetime.timedelta(minutes=20), country="stub"
    )

    assert network["calls"] == [
        (TARGET, TEN_MINUTES),
        (TARGET - TEN_MINUTES, TEN_MINUTES),
    ]
    assert ds.precipitation.values == pytest.approx([0.6])


def test_the_sum_and_its_parts_are_cached(network, tmp_path):
    """A second request is served from disk: no provider is called again."""
    network["values"] = {
        (TARGET, ONE_HOUR): reading(A=3.0),
        (TARGET - ONE_HOUR, ONE_HOUR): reading(A=1.5),
    }

    first = gauge.get_network(time=TARGET, duration=TWO_HOURS, country="stub")
    network["calls"].clear()
    second = gauge.get_network(time=TARGET, duration=TWO_HOURS, country="stub")

    assert network["calls"] == []
    assert second.precipitation.values == first.precipitation.values == [4.5]
    # each part is cached under its own closing instant
    assert (tmp_path / "20261017150000.pt1h.stub.nc").exists()
    assert (tmp_path / "20261017140000.pt1h.stub.nc").exists()
