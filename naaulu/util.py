import datetime
import isodate
import os

import numpy


def format_lon(value):
    """Longitude as E003/W003."""
    if value >= 0:
        return f"E{int(value):03}"
    else:
        return f"W{abs(int(value)):03}"


def format_lat(value):
    """Latitude as N003/S003."""
    if value >= 0:
        return f"N{int(value):03}"
    else:
        return f"S{abs(int(value)):03}"


def format_tile(tile):
    """Tile name from its lon/lat bounds."""
    lonmin, latmin, lonmax, latmax = tile.bounds

    name = f"{format_lon(lonmin)}_{format_lon(lonmax)}.{format_lat(latmin)}_{format_lat(latmax)}"
    return name


def format_time(time, *, show=False):
    """Time as YYYYmmddHHMMSS, or a readable UTC string when show."""
    if show:
        time_str = time.strftime("%Y-%m-%d %H:%M UTC")
    else:
        time_str = time.strftime("%Y%m%d%H%M%S")
    return time_str


def format_duration(duration, *, show=False):
    """Duration as a lowercase ISO 8601 string, or a readable one when show."""
    duration_str = isodate.isoduration.duration_isoformat(duration)
    if show:
        duration_str = duration_str.replace("P1D", "1 day")
        duration_str = duration_str.replace("PT1H", "1 hour")
        duration_str = duration_str.replace("PT", "")
        duration_str = duration_str.replace("H", " hours")
        duration_str = duration_str.replace("M", " minutes")
        duration_str = duration_str.replace("S", " seconds")
    else:
        duration_str = duration_str.lower()
    return duration_str


def parse_duration(duration_str):
    """ISO 8601 duration string as a timedelta."""
    d = isodate.parse_duration(duration_str.upper())
    if isinstance(d, datetime.timedelta):
        return d
    return datetime.timedelta(days=d.days, seconds=d.seconds)


def parse_time(time_str):
    """Time as tz-aware UTC datetime from an ISO or basic-format string."""
    # basic formats without the ISO 'T': 8 digits as a day, 14 as a datetime
    if len(time_str) == 8 and time_str.isdigit():
        time_str = time_str + "T000000"
    elif len(time_str) == 14 and time_str.isdigit():
        time_str = time_str[:8] + "T" + time_str[8:]
    time = isodate.parse_datetime(time_str)
    if time.tzinfo is None:
        time = time.replace(tzinfo=datetime.timezone.utc)
    return time


def parse_distance(distance):
    """Distance in metres from a number or a string with an m/km/E suffix."""
    if isinstance(distance, str):
        s = distance.strip().upper()
        if "E" in s:
            return int(float(s))
        if "KM" in s:
            s = s.replace("KM", "")
            return int(float(s) * 1000)
        if "M" in s:
            s = s.replace("M", "")
            return int(float(s))
        distance = s
    return int(distance)


def format_distance(distance):
    """Distance in km when a whole number of them, else in metres."""
    if (distance % 1000) == 0:
        distance = distance // 1000
        return f"{distance}km"
    else:
        return f"{distance}m"


def get_path(root, filename):
    """Path under root derived from the filename's own parts."""
    parts = filename.split(".")
    time = parts[0]
    dirname = os.path.join(root, time[0:4], time[4:6], time[6:8], *parts[1:-1])

    path = os.path.join(dirname, filename)

    return path


def naive_utc(value):
    """Normalize to tz-naive UTC; naive input is assumed UTC."""
    if getattr(value, "tzinfo", None) is None:
        return value
    return value.astimezone(datetime.timezone.utc).replace(tzinfo=None)


def to_datetime64(value):
    """Convert a datetime to numpy.datetime64 without timezone warnings."""
    return numpy.datetime64(naive_utc(value))


def time_mean(datetimes):
    """Mean timestamp of a datetime64 array, as tz-aware UTC datetime."""
    timestamps = numpy.asarray(datetimes).astype('datetime64[s]').astype('int64')
    time_mean = numpy.mean(timestamps)
    time_mean = datetime.datetime.fromtimestamp(time_mean, tz=datetime.timezone.utc)

    return time_mean


def time_arange(start, end, step):
    """Timestamps from start to end inclusive, spaced by step."""
    t = start
    times = []
    while t <= end:
        times.append(t)
        t = t + step

    return times


def malloc_trim():
    """Release free memory back to the OS (no-op on non-Linux)."""
    try:
        import sys
        import ctypes
        if sys.platform == "linux":
            ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass
