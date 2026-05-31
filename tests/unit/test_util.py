import datetime
import unittest

from naaulu.util import (
    format_duration,
    format_lat,
    format_lon,
    format_time,
    format_distance,
    get_bracketing_times,
    parse_distance,
    parse_duration,
    time_arange,
)


class TestFormatLon(unittest.TestCase):
    def test_east(self):
        self.assertEqual(format_lon(5.7), "E005")

    def test_west(self):
        self.assertEqual(format_lon(-10.2), "W010")


class TestFormatLat(unittest.TestCase):
    def test_north(self):
        self.assertEqual(format_lat(50.1), "N050")

    def test_south(self):
        self.assertEqual(format_lat(-33.8), "S033")


class TestFormatTime(unittest.TestCase):
    def test_default(self):
        t = datetime.datetime(2024, 6, 5, 14, 30, 45)
        self.assertEqual(format_time(t), "20240605143045")

    def test_show(self):
        t = datetime.datetime(2024, 6, 5, 14, 30, 45)
        self.assertEqual(format_time(t, show=True), "2024-06-05 14:30")


class TestParseDuration(unittest.TestCase):
    def test_hours(self):
        self.assertEqual(parse_duration("pt1h"), datetime.timedelta(hours=1))

    def test_minutes(self):
        self.assertEqual(parse_duration("pt30m"), datetime.timedelta(minutes=30))


class TestFormatDuration(unittest.TestCase):
    def test_default(self):
        d = datetime.timedelta(hours=1, minutes=30)
        self.assertEqual(format_duration(d), "pt1h30m")

    def test_show(self):
        d = datetime.timedelta(hours=1)
        self.assertEqual(format_duration(d, show=True), "1 hour")


class TestParseDistance(unittest.TestCase):
    def test_km(self):
        self.assertEqual(parse_distance("5km"), 5000)

    def test_meters(self):
        self.assertEqual(parse_distance("500m"), 500)


class TestFormatDistance(unittest.TestCase):
    def test_km(self):
        self.assertEqual(format_distance(5000), "5km")

    def test_meters(self):
        self.assertEqual(format_distance(500), "500m")


class TestTimeArange(unittest.TestCase):
    def test_basic(self):
        start = datetime.datetime(2024, 1, 1, 0, 0)
        end = datetime.datetime(2024, 1, 1, 2, 0)
        step = datetime.timedelta(hours=1)
        result = time_arange(start, end, step)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], start)
        self.assertEqual(result[-1], end)

    def test_empty(self):
        start = datetime.datetime(2024, 1, 1, 2, 0)
        end = datetime.datetime(2024, 1, 1, 1, 0)
        step = datetime.timedelta(hours=1)
        self.assertEqual(time_arange(start, end, step), [])


class TestGetBracketingTimes(unittest.TestCase):
    def test_exact_match(self):
        timestamps = [
            datetime.datetime(2024, 1, 1, 0, 0),
            datetime.datetime(2024, 1, 1, 1, 0),
        ]
        target = datetime.datetime(2024, 1, 1, 1, 0)
        before, after = get_bracketing_times(timestamps, target)
        self.assertEqual(before, target)
        self.assertEqual(after, target)

    def test_between(self):
        timestamps = [
            datetime.datetime(2024, 1, 1, 0, 0),
            datetime.datetime(2024, 1, 1, 2, 0),
        ]
        target = datetime.datetime(2024, 1, 1, 1, 0)
        before, after = get_bracketing_times(timestamps, target)
        self.assertEqual(before, timestamps[0])
        self.assertEqual(after, timestamps[1])

    def test_outside_range(self):
        timestamps = [
            datetime.datetime(2024, 1, 1, 2, 0),
        ]
        target = datetime.datetime(2024, 1, 1, 1, 0)
        self.assertEqual(get_bracketing_times(timestamps, target), (None, None))


if __name__ == "__main__":
    unittest.main()
