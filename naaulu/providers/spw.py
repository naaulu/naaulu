import datetime
import logging
import os

import numpy

import naaulu.network

logger = logging.getLogger(__name__)

api_key = os.environ.get("SPW_API_KEY")


def ott(time: datetime.datetime, duration: datetime.timedelta, bbox=None):
    """
    SPW (Service Public de Wallonie) OTT rain gauge network (Wallonia, Belgium).

    Requires SPW_API_KEY for authenticated access.
    """
    if api_key:
        # TODO: implement real authenticated call once endpoint is known
        logger.info("SPW_API_KEY present — real fetch not yet implemented")

    # Placeholder stations (works without API key)
    val = 0.5
    stations = [
        ("SPW_01", 4.85, 50.47, val),
        ("SPW_02", 5.57, 50.63, val),
        ("SPW_03", 4.35, 50.45, val),
        ("SPW_04", 5.10, 50.20, val),
    ]

    lons, lats, precs = [], [], []
    for sid, lon, lat, val in stations:
        lons.append(lon)
        lats.append(lat)
        precs.append(val)

    return lons, lats, numpy.array(precs)
