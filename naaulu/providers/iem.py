import datetime
import json
import logging

import numpy as np
import naaulu.network

logger = logging.getLogger(__name__)

gauges = ["asos"]

# Default IEM ASOS network. Callers can pass any per-state code (e.g. "WI_ASOS",
# "AZ_ASOS"); see https://mesonet.agron.iastate.edu/sites/networks.php for the list.
DEFAULT_NETWORK = "IA_ASOS"

# Inches -> millimetres (ASOS p01i is reported in inches per METAR convention)
INCH_TO_MM = 25.4

_station_coords_cache = {}


def _station_coords(network):
    """Fetch and cache station metadata (sid, lon, lat) for a network."""
    if network in _station_coords_cache:
        return _station_coords_cache[network]

    url = f"https://mesonet.agron.iastate.edu/geojson/network/{network}.geojson"
    raw = naaulu.network.fetch(url)
    geo = json.loads(raw.decode("utf-8"))
    coords = {}
    for feat in geo.get("features", []):
        sid = feat["properties"].get("sid")
        lon, lat = feat["geometry"]["coordinates"]
        if sid:
            coords[sid] = (lon, lat)
    _station_coords_cache[network] = coords
    return coords


def asos(time: datetime.datetime, duration: datetime.timedelta, network: str = DEFAULT_NETWORK, bbox=None):
    """
    Fetch ASOS hourly precipitation via IEM.

    ASOS METARs report `p01i` (1-hour precipitation, inches) at non-regular times
    (usually :54). For a requested `time`, we query a ±30-min window and take the
    observation closest to `time` per station that has a non-missing p01i.
    """
    if duration != datetime.timedelta(hours=1):
        raise ValueError("Only 1-hour duration supported")

    station_meta = _station_coords(network)
    if not station_meta:
        raise RuntimeError(f"No stations discovered for IEM network {network}")

    if bbox and len(bbox) == 4:
        minx, miny, maxx, maxy = bbox
        station_meta = {
            sid: (lon, lat)
            for sid, (lon, lat) in station_meta.items()
            if minx <= lon <= maxx and miny <= lat <= maxy
        }
        if not station_meta:
            logger.info(f"iem/asos: no stations in bbox {bbox}")
            return [], [], np.empty((0, 2))

    window_start = time - datetime.timedelta(minutes=30)
    window_end = time + datetime.timedelta(minutes=30)

    url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
    params = {
        "data": "p01i",
        "tz": "UTC",
        "format": "comma",
        "network": network,
        "station": ",".join(sorted(station_meta.keys())),
        "latlon": "yes",
        "year1": window_start.year, "month1": window_start.month,
        "day1": window_start.day, "hour1": window_start.hour, "minute1": window_start.minute,
        "year2": window_end.year, "month2": window_end.month,
        "day2": window_end.day, "hour2": window_end.hour, "minute2": window_end.minute,
    }

    response = naaulu.network.fetch(url + "?" + "&".join(f"{k}={v}" for k, v in params.items()))
    raw = response.decode("utf-8")

    header = None
    best = {}  # station -> (abs_dt_seconds, value_mm)
    for line in raw.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        fields = line.split(",")
        if header is None:
            header = fields
            continue
        if len(fields) != len(header):
            continue
        row = dict(zip(header, fields))
        try:
            value_in = float(row["p01i"])
        except (ValueError, KeyError):
            continue  # 'M' or other missing markers
        try:
            obs_time = datetime.datetime.strptime(row["valid"], "%Y-%m-%d %H:%M")
        except ValueError:
            continue
        dt = abs((obs_time - time).total_seconds())
        station = row["station"]
        if station not in best or dt < best[station][0]:
            best[station] = (dt, value_in * INCH_TO_MM)

    codes, values, longitudes, latitudes = [], [], [], []
    for station, (_, value_mm) in best.items():
        lon, lat = station_meta.get(station, (None, None))
        if lon is None:
            continue
        codes.append(station)
        values.append(value_mm)
        longitudes.append(lon)
        latitudes.append(lat)

    coords = np.column_stack((longitudes, latitudes)) if codes else np.empty((0, 2))
    return (values, codes, coords)
