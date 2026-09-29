import datetime
import warnings

import numpy

from naaulu.util import naive_utc, to_datetime64


def test_naive_utc_preserves_naive_values():
    naive = datetime.datetime(2025, 7, 11, 18, 50, 0)
    assert naive_utc(naive) is naive


def test_naive_utc_converts_tz_aware_to_naive_utc():
    tz = datetime.timezone(datetime.timedelta(hours=2))
    aware = datetime.datetime(2025, 7, 11, 20, 50, 0, tzinfo=tz)
    assert naive_utc(aware) == datetime.datetime(2025, 7, 11, 18, 50, 0)


def test_to_datetime64_converts_tz_aware_without_warning():
    tz = datetime.timezone(datetime.timedelta(hours=2))
    aware = datetime.datetime(2025, 7, 11, 20, 50, 0, tzinfo=tz)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        value = to_datetime64(aware)

    assert value == numpy.datetime64("2025-07-11T18:50:00")


def test_to_datetime64_passes_through_naive_values():
    naive = datetime.datetime(2025, 7, 11, 18, 50, 0)
    assert to_datetime64(naive) == numpy.datetime64("2025-07-11T18:50:00")
