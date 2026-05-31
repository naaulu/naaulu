import datetime
import logging

import numpy
import requests

import naaulu.providers.opera as opera

logger = logging.getLogger(__name__)


def radar(time, duration, wsi):
    return opera.radar(time=time, duration=duration, wsi=wsi)


def synop(time: datetime.datetime, duration: datetime.timedelta, bbox=None):
    """
    RMIB SYNOP / automatic weather stations (Belgium).
    Calls RMIB open data API.
    """
    if duration not in (datetime.timedelta(minutes=10), datetime.timedelta(hours=1)):
        raise ValueError(f"Unsupported duration for RMIB/SYNOP: {duration}")

    url = "https://opendata.meteo.be/api/observations"
    params = {
        "dataset": "synop",
        "start": (time - duration).isoformat(),
        "end": time.isoformat(),
    }

    try:
        r = requests.get(url, params=params, timeout=30)
        if r.status_code != 200:
            logger.warning(f"rmib/synop HTTP {r.status_code}")
            return [], [], numpy.empty((0, 2))
        data = r.json()
    except Exception as e:
        logger.warning(f"rmib/synop API error: {e}")
        return [], [], numpy.empty((0, 2))

    stations = []
    for item in data.get("results", []):
        code = item.get("station_code") or item.get("code")
        lon = item.get("longitude") or item.get("lon")
        lat = item.get("latitude") or item.get("lat")
        val = item.get("precipitation") or item.get("precip") or 0.0
        if lon is None or lat is None:
            continue
        if bbox and len(bbox) == 4:
            minx, miny, maxx, maxy = bbox
            if not (minx <= lon <= maxx and miny <= lat <= maxy):
                continue
        stations.append((code, lon, lat, val))

    if not stations:
        logger.info("rmib/synop: no stations returned")
        return [], [], numpy.empty((0, 2))

    codes = [s[0] for s in stations]
    longitudes = [s[1] for s in stations]
    latitudes = [s[2] for s in stations]
    values = [s[3] for s in stations]
    coords = numpy.column_stack((longitudes, latitudes))

    logger.debug(f"rmib/synop: returned {len(values)} stations at {time}")
    return (values, codes, coords)
