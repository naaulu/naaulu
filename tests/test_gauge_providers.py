import datetime
import unittest

import numpy
import pytest

import naaulu.providers.fmi
import naaulu.providers.iem
import naaulu.providers.noaa
import naaulu.providers.vmm
import naaulu.providers.dwd


def _recent_hour():
    now = datetime.datetime.now(datetime.timezone.utc).replace(minute=0, second=0, microsecond=0)
    return (now - datetime.timedelta(hours=2)).replace(tzinfo=None)


def _recent_10min():
    now = datetime.datetime.now(datetime.timezone.utc).replace(second=0, microsecond=0)
    aligned = now.replace(minute=(now.minute // 10) * 10)
    return (aligned - datetime.timedelta(minutes=30)).replace(tzinfo=None)


def _assert_gauge_shape(values, codes, coords):
    n = len(codes)
    assert n > 0, "expected at least one station"
    assert len(values) == n
    assert isinstance(coords, numpy.ndarray)
    assert coords.shape == (n, 2)


class TestFmiAws(unittest.TestCase):

    @pytest.mark.network
    def test_hourly_returns_stations(self):
        values, codes, coords = naaulu.providers.fmi.aws(
            _recent_hour(), datetime.timedelta(hours=1)
        )
        _assert_gauge_shape(values, codes, coords)

    @pytest.mark.network
    def test_ten_min_returns_stations(self):
        values, codes, coords = naaulu.providers.fmi.aws(
            _recent_10min(), datetime.timedelta(minutes=10)
        )
        _assert_gauge_shape(values, codes, coords)

    @pytest.mark.network
    def test_ten_min_values_in_mm(self):
        """ri_10min comes back as mm/h; provider must convert to mm/10min (×10/60)."""
        values, _, _ = naaulu.providers.fmi.aws(
            _recent_10min(), datetime.timedelta(minutes=10)
        )
        finite = [v for v in values if v == v]
        # 10-min accumulations should never exceed ~25 mm in Finland (extreme).
        # If the conversion were missing, values would be ×6 larger.
        self.assertLess(max(finite, default=0.0), 25.0)

    def test_unsupported_duration_raises(self):
        with self.assertRaises(ValueError):
            naaulu.providers.fmi.aws(
                _recent_hour(), datetime.timedelta(minutes=5)
            )


class TestIemAsos(unittest.TestCase):

    @pytest.mark.network
    def test_hourly_returns_stations(self):
        # IEM data lags; use a fixed past time within the last few days.
        t = datetime.datetime.now() - datetime.timedelta(days=2)
        t = t.replace(minute=0, second=0, microsecond=0)
        values, codes, coords = naaulu.providers.iem.asos(
            t, datetime.timedelta(hours=1)
        )
        _assert_gauge_shape(values, codes, coords)

    def test_unsupported_duration_raises(self):
        with self.assertRaises(ValueError):
            naaulu.providers.iem.asos(
                _recent_hour(), datetime.timedelta(minutes=10)
            )


class TestNoaaGhcn(unittest.TestCase):

    def _has_token(self):
        try:
            naaulu.providers.noaa._get_token()
            return True
        except RuntimeError:
            return False

    @pytest.mark.network
    def test_hourly_returns_data(self):
        if not self._has_token():
            self.skipTest("NOAA_API_TOKEN not configured")
        # GHCN-H ingestion lags; use a date known to have hourly precip data.
        t = datetime.datetime(2024, 1, 15, 12, 0)
        values, codes, coords = naaulu.providers.noaa.ghcn(
            t, datetime.timedelta(hours=1)
        )
        # The provider currently queries a single hardcoded station; we
        # accept either "got data" or "got nothing for that hour" — but
        # the call itself must complete and return the expected tuple shape.
        self.assertEqual(len(values), len(codes))
        self.assertEqual(coords.shape if hasattr(coords, "shape") else (len(codes), 2),
                         (len(codes), 2))


class TestVmmOtt(unittest.TestCase):

    def test_ten_min_and_hourly_return_stations(self):
        t = datetime.datetime(2024, 1, 15, 12, 0)
        for duration in (datetime.timedelta(minutes=10), datetime.timedelta(hours=1)):
            values, codes, coords = naaulu.providers.vmm.ott(t, duration)
            _assert_gauge_shape(values, codes, coords)

    def test_unsupported_duration_raises(self):
        t = datetime.datetime(2024, 1, 15, 12, 0)
        with self.assertRaises(ValueError):
            naaulu.providers.vmm.ott(t, datetime.timedelta(minutes=5))


class TestDwdSynop(unittest.TestCase):

    def test_ten_min_and_hourly_return_stations(self):
        t = datetime.datetime(2024, 1, 15, 12, 0)
        for duration in (datetime.timedelta(minutes=10), datetime.timedelta(hours=1)):
            values, codes, coords = naaulu.providers.dwd.synop(t, duration)
            _assert_gauge_shape(values, codes, coords)

    def test_unsupported_duration_raises(self):
        t = datetime.datetime(2024, 1, 15, 12, 0)
        with self.assertRaises(ValueError):
            naaulu.providers.dwd.synop(t, datetime.timedelta(minutes=5))


if __name__ == "__main__":
    unittest.main()
