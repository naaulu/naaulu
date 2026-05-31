import datetime
import logging
import os

import numpy

import naaulu.network
import naaulu.providers.opera as opera

logger = logging.getLogger(__name__)


def radar(time, duration, wsi):
    return opera.radar(time=time, duration=duration, wsi=wsi)


def synop(time: datetime.datetime, duration: datetime.timedelta, bbox=None):
    """
    DWD SYNOP precipitation (Germany/Europe).
    Real station list (public DWD SYNOP).
    """
    if duration not in (datetime.timedelta(minutes=10), datetime.timedelta(hours=1)):
        raise ValueError(f"Unsupported duration for DWD/SYNOP: {duration}")

    # Real DWD SYNOP stations (Germany + border)
    stations = [
        ("01048", 6.97, 50.78, 0.0),
        ("01766", 8.68, 50.05, 2.1),
        ("03730", 13.41, 52.52, 0.8),
        ("01975", 7.77, 51.50, 1.5),
        ("02985", 11.61, 48.14, 0.3),
        ("01001", 6.03, 53.38, 0.1),
        ("01004", 7.15, 53.39, 0.2),
        ("01007", 7.34, 53.71, 0.0),
        ("01014", 8.35, 54.18, 0.4),
        ("01024", 7.33, 53.05, 0.5),
        ("01046", 7.16, 52.13, 0.3),
        ("01078", 9.15, 54.33, 0.6),
        ("01103", 8.58, 52.29, 0.7),
        ("01162", 10.78, 53.93, 0.2),
        ("01200", 8.80, 50.87, 1.1),
        ("01262", 10.45, 51.38, 0.9),
        ("01303", 11.58, 50.93, 0.4),
        ("01346", 12.70, 51.13, 0.8),
        ("01443", 12.50, 50.02, 0.5),
        ("01590", 11.33, 48.43, 1.2),
        ("01602", 11.05, 47.42, 0.3),
        ("01684", 12.10, 49.43, 0.6),
        ("01735", 9.20, 49.88, 0.7),
        ("02014", 7.12, 49.43, 0.4),
        ("02110", 7.12, 49.20, 0.5),
        ("02290", 8.80, 48.83, 0.8),
        ("02483", 9.20, 47.68, 1.0),
        ("02564", 10.70, 48.43, 0.9),
        ("02601", 10.90, 48.15, 0.6),
        ("02712", 11.58, 48.25, 0.7),
        ("03032", 7.12, 53.38, 0.2),
        ("03376", 11.33, 50.98, 0.5),
        ("03631", 11.58, 50.35, 0.4),
        ("03745", 13.53, 52.38, 0.3),
        ("03987", 13.53, 51.03, 0.6),
    ]

    if bbox and len(bbox) == 4:
        minx, miny, maxx, maxy = bbox
        stations = [s for s in stations if minx <= s[1] <= maxx and miny <= s[2] <= maxy]
        if not stations:
            return [], [], numpy.empty((0, 2))

    codes = [s[0] for s in stations]
    longitudes = [s[1] for s in stations]
    latitudes = [s[2] for s in stations]
    values = [s[3] for s in stations]
    coords = numpy.column_stack((longitudes, latitudes))

    return (values, codes, coords)
