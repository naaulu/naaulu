import datetime
import unittest

import pytest

import naaulu.providers.chmi
import naaulu.providers.estea
import naaulu.providers.fmi
import naaulu.providers.noaa
import xradar
from naaulu.radar import get_radar

from tests.conftest import (
    COVERED_DURATION,
    COVERED_TIME,
    RADARS,
    RADARS_DE,
    SHORT_DURATION,
    TOO_OLD_DURATION,
    TOO_OLD_TIME,
    UNKNOWN_WSI,
)

LONG_DURATION = datetime.timedelta(minutes=30)

EST_WSI = "0-21010-0-42"
CHMI_WSI = "0-20000-0-11718"

COVERED_TIME_UTC = COVERED_TIME
TOO_OLD_TIME_UTC = TOO_OLD_TIME


class TestEstea(unittest.TestCase):

    @pytest.mark.network
    @pytest.mark.slow
    def test_short_window(self):
        vols = naaulu.providers.estea.radar(
            time=COVERED_TIME_UTC, duration=SHORT_DURATION, wsi=EST_WSI
        )
        self.assertGreater(len(vols), 0)  # period is covered (any number of elevations)

    @pytest.mark.network
    @pytest.mark.slow
    def test_long_window(self):
        vols = naaulu.providers.estea.radar(
            time=COVERED_TIME_UTC, duration=LONG_DURATION, wsi=EST_WSI
        )
        self.assertGreater(len(vols), 1)

    @pytest.mark.network
    @pytest.mark.slow
    def test_old_period_raises(self):
        with self.assertRaises(RuntimeError):
            naaulu.providers.estea.radar(
                time=TOO_OLD_TIME_UTC, duration=SHORT_DURATION, wsi=EST_WSI
            )

    def test_unknown_wsi_raises(self):
        with self.assertRaises(ValueError):
            naaulu.providers.estea.radar(
                time=COVERED_TIME_UTC, duration=SHORT_DURATION, wsi=UNKNOWN_WSI
            )

    @pytest.mark.network
    @pytest.mark.slow
    def test_one_hour_ago(self):
        t = datetime.datetime.now(tz=datetime.timezone.utc) - datetime.timedelta(hours=1)
        vols = naaulu.providers.estea.radar(
            time=t, duration=SHORT_DURATION, wsi=EST_WSI
        )
        self.assertGreater(len(vols), 0)

    @pytest.mark.network
    @pytest.mark.slow
    def test_one_month_ago(self):
        t = datetime.datetime.now(tz=datetime.timezone.utc) - datetime.timedelta(days=30)
        with self.assertRaises(RuntimeError):
            naaulu.providers.estea.radar(
                time=t, duration=SHORT_DURATION, wsi=EST_WSI
            )


class TestTimingChmi(unittest.TestCase):

    @pytest.mark.network
    @pytest.mark.slow
    def test_short_window(self):
        vols = naaulu.providers.chmi.get_radar(
            time=COVERED_TIME_UTC, duration=SHORT_DURATION, wsi=CHMI_WSI
        )
        self.assertEqual(len(vols), 1)

    @pytest.mark.network
    @pytest.mark.slow
    def test_long_window(self):
        vols = naaulu.providers.chmi.get_radar(
            time=COVERED_TIME_UTC, duration=LONG_DURATION, wsi=CHMI_WSI
        )
        self.assertGreater(len(vols), 1)

    @pytest.mark.network
    @pytest.mark.slow
    def test_old_period_raises(self):
        with self.assertRaises(RuntimeError):
            naaulu.providers.chmi.get_radar(
                time=TOO_OLD_TIME_UTC, duration=SHORT_DURATION, wsi=CHMI_WSI
            )

    def test_unknown_wsi_raises(self):
        with self.assertRaises(ValueError):
            naaulu.providers.chmi.get_radar(
                time=COVERED_TIME_UTC, duration=SHORT_DURATION, wsi=UNKNOWN_WSI
            )


class TestTimingOpera(unittest.TestCase):

    RADAR_WSI = "0-20000-0-06482"

    @pytest.mark.network
    @pytest.mark.slow
    def test_short_window(self):
        vols = naaulu.providers.opera.get_radar(
            time=COVERED_TIME_UTC, duration=SHORT_DURATION, wsi=self.RADAR_WSI
        )
        self.assertGreater(len(vols), 0)

    @pytest.mark.network
    @pytest.mark.slow
    def test_long_window(self):
        vols = naaulu.providers.opera.get_radar(
            time=COVERED_TIME_UTC, duration=LONG_DURATION, wsi=self.RADAR_WSI
        )
        self.assertGreater(len(vols), 1)

    @pytest.mark.network
    @pytest.mark.slow
    def test_old_period_raises(self):
        with self.assertRaises(RuntimeError):
            naaulu.providers.opera.get_radar(
                time=TOO_OLD_TIME_UTC, duration=SHORT_DURATION, wsi=self.RADAR_WSI
            )

    @pytest.mark.network
    def test_unknown_wsi_raises(self):
        with self.assertRaises(ValueError):
            naaulu.providers.opera.get_radar(
                time=COVERED_TIME_UTC, duration=SHORT_DURATION, wsi=UNKNOWN_WSI
            )


@pytest.mark.network
class TestGetRadarCoverage(unittest.TestCase):

    def _radars(self):
        return {**RADARS, **RADARS_DE}

    def test_recent_period(self):
        for country, (wsi, node) in self._radars().items():
            with self.subTest(country=country, node=node):
                volumes = opera_get_radar(
                    time=COVERED_TIME,
                    duration=COVERED_DURATION,
                    wsi=wsi,
                )
                self.assertGreater(
                    len(volumes), 0,
                    f"{country}/{node} ({wsi}) should return volumes at {COVERED_TIME}",
                )

    def test_short_window(self):
        """Verify [time – 5 min, time] is covered for every OPERA country."""
        for country, (wsi, node) in self._radars().items():
            with self.subTest(country=country, node=node):
                volumes = opera_get_radar(
                    time=COVERED_TIME,
                    duration=SHORT_DURATION,
                    wsi=wsi,
                )
                self.assertGreater(
                    len(volumes), 0,
                    f"{country}/{node} ({wsi}) — no data in PT5M window at {COVERED_TIME}",
                )

    def test_old_period_raises(self):
        for country, (wsi, node) in self._radars().items():
            with self.subTest(country=country, node=node):
                with self.assertRaises(RuntimeError):
                    opera_get_radar(
                        time=TOO_OLD_TIME,
                        duration=TOO_OLD_DURATION,
                        wsi=wsi,
                    )

    def test_unknown_wsi_raises(self):
        with self.assertRaises(ValueError):
            opera_get_radar(
                time=COVERED_TIME,
                duration=COVERED_DURATION,
                wsi=UNKNOWN_WSI,
            )


class TestTimingNoaa(unittest.TestCase):

    @pytest.mark.network
    @pytest.mark.slow
    def test_short_window(self):
        vols = naaulu.providers.noaa.get_radar(
            time=COVERED_TIME_UTC, duration=SHORT_DURATION, wsi="0-21010-0-190"
        )
        self.assertEqual(len(vols), 1)

    @pytest.mark.network
    @pytest.mark.slow
    def test_long_window(self):
        vols = naaulu.providers.noaa.get_radar(
            time=COVERED_TIME_UTC, duration=LONG_DURATION, wsi="0-21010-0-190"
        )
        self.assertGreater(len(vols), 1)

    @pytest.mark.network
    @pytest.mark.slow
    def test_old_period_raises(self):
        with self.assertRaises(FileNotFoundError):
            naaulu.providers.noaa.get_radar(
                time=TOO_OLD_TIME_UTC, duration=SHORT_DURATION, wsi="0-21010-0-190"
            )

    def test_unknown_wsi_raises(self):
        with self.assertRaises(ValueError):
            naaulu.providers.noaa.get_radar(
                time=COVERED_TIME_UTC, duration=SHORT_DURATION, wsi=UNKNOWN_WSI
            )


class TestFmi(unittest.TestCase):

    FMI_WSI = "0-20010-0-02954"

    @pytest.mark.network
    @pytest.mark.slow
    def test_short_window(self):
        volume = naaulu.providers.fmi.radar(
            time=COVERED_TIME_UTC, duration=SHORT_DURATION, wsi=self.FMI_WSI
        )
        self.assertIsNotNone(volume)

    @pytest.mark.network
    @pytest.mark.slow
    def test_long_window(self):
        volume = naaulu.providers.fmi.radar(
            time=COVERED_TIME_UTC, duration=LONG_DURATION, wsi=self.FMI_WSI
        )
        self.assertIsNotNone(volume)

    @pytest.mark.network
    @pytest.mark.slow
    def test_old_period_raises(self):
        with self.assertRaises(RuntimeError):
            naaulu.providers.fmi.radar(
                time=TOO_OLD_TIME_UTC, duration=SHORT_DURATION, wsi=self.FMI_WSI
            )

    def test_unknown_wsi_raises(self):
        with self.assertRaises(ValueError):
            naaulu.providers.fmi.radar(
                time=COVERED_TIME_UTC, duration=SHORT_DURATION, wsi=UNKNOWN_WSI
            )

    @pytest.mark.network
    @pytest.mark.slow
    def test_one_hour_ago(self):
        t = datetime.datetime.now(tz=datetime.timezone.utc) - datetime.timedelta(hours=1)
        volume = naaulu.providers.fmi.radar(
            time=t, duration=SHORT_DURATION, wsi=self.FMI_WSI
        )
        self.assertIsNotNone(volume)

    @pytest.mark.network
    @pytest.mark.slow
    def test_one_month_ago(self):
        t = datetime.datetime.now(tz=datetime.timezone.utc) - datetime.timedelta(days=30)
        with self.assertRaises(RuntimeError):
            naaulu.providers.fmi.radar(
                time=t, duration=SHORT_DURATION, wsi=self.FMI_WSI
            )


if __name__ == "__main__":
    unittest.main()
