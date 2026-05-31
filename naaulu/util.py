import bisect
import datetime
import isodate
import os

import numpy


def format_lon(value):
    if value >= 0:
        return f"E{int(value):03}"
    else:
        return f"W{abs(int(value)):03}"


def format_lat(value):
    if value >= 0:
        return f"N{int(value):03}"
    else:
        return f"S{abs(int(value)):03}"


def format_tile(tile):
    lonmin, latmin, lonmax, latmax = tile.bounds

    name = f"{format_lon(lonmin)}_{format_lon(lonmax)}.{format_lat(latmin)}_{format_lat(latmax)}"
    return name


def format_time(time, show=False):
    if show:
        time_str = time.strftime("%Y-%m-%d %H:%M")
    else:
        time_str = time.strftime("%Y%m%d%H%M%S")
    return time_str


def format_duration(duration, show=False):
    duration_str = isodate.isoduration.duration_isoformat(duration)
    if show:
        duration_str = duration_str.replace("PT1H", "1 hour")
        duration_str = duration_str.replace("PT", "")
        duration_str = duration_str.replace("H", " hours")
        duration_str = duration_str.replace("M", " minutes")
        duration_str = duration_str.replace("S", " seconds")
    else:
        duration_str = duration_str.lower()
    return duration_str


def parse_duration(duration_str):
    d = isodate.parse_duration(duration_str.upper())
    if isinstance(d, datetime.timedelta):
        return d
    return datetime.timedelta(days=d.days, seconds=d.seconds)


def parse_time(time_str):
    time = isodate.parse_datetime(time_str)
    return time


def parse_distance(distance):

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
    if (distance % 1000) == 0:
        distance = distance // 1000
        return f"{distance}km"
    else:
        return f"{distance}m"


def get_path(root, filename):
    parts = filename.split(".")
    time = parts[0]
    dirname = os.path.join(root, time[0:4], time[4:6], time[6:8], *parts[1:-1])

    path = os.path.join(dirname, filename)

    return path


def get_bracketing_times(timestamps, target):
    """
    timestamps: sorted list of datetime objects
    target: datetime to bracket
    Returns: (time_before, time_after) or (None, None)
             If target matches an element, returns (target, target)
    """    

    idx = bisect.bisect_left(timestamps, target)
    if idx < len(timestamps) and timestamps[idx] == target:
        return target, target
    if 0 < idx < len(timestamps):
        return timestamps[idx - 1], timestamps[idx]
    return None, None


def round_time_to_interval(dt, interval):
    interval_ns = interval.astype('timedelta64[ns]').astype(numpy.int64)
    dt_ns = dt.astype('datetime64[ns]').astype(numpy.int64)
    rounded_ns = ((dt_ns + interval_ns // 2) // interval_ns) * interval_ns
    return numpy.datetime64(int(rounded_ns), 'ns')


def time_mean(datetimes):
    """
    Computes the average (mean) timestamp from a datetime64 array.

    Parameters
    ----------
    datetimes : array-like
        Array of datetime64 values (e.g., numpy.datetime64 or compatible).

    Returns
    -------
    numpy.datetime64
        The mean timestamp.
    """
    timestamps = numpy.asarray(datetimes).astype('datetime64[s]').astype('int64')
    time_mean = numpy.mean(timestamps)
    time_mean = datetime.datetime.utcfromtimestamp(time_mean)

    return time_mean

def time_arange(start, end, step):
    t = start
    times = []
    while t <= end:
        times.append(t)
        t = t + step

    return times
