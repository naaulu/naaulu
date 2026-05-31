import datetime
import json
import logging
import os
import signal
import traceback

import fsspec
import numpy
import xradar.io
import xradar.util
import naaulu.config
import naaulu.network
import naaulu.radar

logger = logging.getLogger(__name__)

_NOAA_API_TOKEN = None


class TimeoutException(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutException("Operation timed out")


def _call_with_timeout(func, args, timeout_sec=10):
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_sec)
    try:
        result = func(*args)
        signal.alarm(0)
        return result
    except TimeoutException:
        logger.warning(f"Operation timed out after {timeout_sec} seconds")
        raise
    finally:
        signal.alarm(0)


def _get_token():
    global _NOAA_API_TOKEN
    if _NOAA_API_TOKEN is None:
        _NOAA_API_TOKEN = naaulu.config.get_secret("NOAA_API_TOKEN")
    if not _NOAA_API_TOKEN:
        raise RuntimeError("NOAA_API_TOKEN not found in .naaulu_secrets")
    return _NOAA_API_TOKEN


# Global cache for station coordinates
_station_coords = {}


def ghcn(time: datetime.datetime, duration, bbox=None):
    """GHCN hourly precipitation. Requires NOAA_API_TOKEN in ~/.naaulu_secrets."""
    startdate = time.isoformat()
    enddate = (time + datetime.timedelta(hours=1)).isoformat()

    try:
        headers = {"token": _get_token()}
    except RuntimeError:
        logger.warning("NOAA_API_TOKEN not configured — skipping ghcn gauges")
        return [], [], numpy.array([]).reshape((0, 2))

    url = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"
    params = {
        "datasetid": "PRECIP_HLY",
        "datatypeid": "HPCP",
        "startdate": startdate,
        "enddate": enddate,
        "limit": 1000,
        "units": "metric"
    }
    if bbox and len(bbox) == 4:
        params["extent"] = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"

    try:
        response = naaulu.network.api_get(url, headers=headers, params=params, timeout=20)
        if response.status_code != 200:
            logger.warning(f"NOAA API error {response.status_code}")
            return [], [], numpy.array([]).reshape((0, 2))
        data = response.json()
    except Exception:
        logger.debug("failed to fetch GHCN data", exc_info=True)
        return [], [], numpy.array([]).reshape((0, 2))

    vals, codes, coords_list = [], [], []

    for item in data.get("results", []):
        value = item.get("value")
        if value is None or value <= 0:
            continue
        lat = item.get("latitude")
        lon = item.get("longitude")
        if lat is None or lon is None:
            continue

        vals.append(value)
        codes.append(item.get("station"))
        coords_list.append([float(lon), float(lat)])

    coords = numpy.array(coords_list) if coords_list else numpy.array([]).reshape((0, 2))

    logger.info(f"ghcn: found {len(vals)} stations with precip")
    return vals, codes, coords


# ─── Radar (moved from usa.py) ─────────────────────────────────────────────────

radars = None  # global cache of {wigos_id: [network, code]}


def add_nexrad(wigos_db: dict, tolerance: float = 0.01) -> dict:
    url = "https://www.ncei.noaa.gov/access/homr/file/nexrad-stations.txt"
    txt_path = naaulu.network.download(url, "nexrad-stations.txt")

    stations = {}
    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            try:
                icao = line[9:13].strip()
                lat = float(line[106:115].strip())
                lon = float(line[116:126].strip())
                stations[icao] = {"lat": lat, "lon": lon}
            except Exception:
                continue

    cache = naaulu.config.get_cache_dir()
    codes_path = os.path.join(cache, "radars_usa.json")

    if os.path.exists(codes_path):
        with open(codes_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
    else:
        mapping = {}

    updated = False
    for wigos_id, meta in wigos_db.items():
        if wigos_id in mapping:
            continue
        try:
            lat_wigos = float(meta["latitude"])
            lon_wigos = float(meta["longitude"])
        except (KeyError, ValueError):
            continue

        for icao, site in stations.items():
            if abs(lat_wigos - site["lat"]) < tolerance and abs(lon_wigos - site["lon"]) < tolerance:
                mapping[wigos_id] = ["nexrad", icao]
                updated = True
                break

    if updated:
        with open(codes_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2)

    return mapping


def clean_sweep(sweep):
    nrays = len(sweep["azimuth"].values)
    if nrays not in [360, 720]:
        return None
    angle_res = 360.0 / nrays
    return xradar.util.reindex_angle(
        ds=sweep,
        start_angle=0.0,
        stop_angle=360.0,
        angle_res=angle_res,
        direction=1,
    )


def nexrad(time: datetime.datetime, duration: datetime.timedelta, code: str):
    time_start = time - duration
    time_end = time

    icao = code.upper()
    files = []
    for shift in [-1, 0, 1]:
        day = time_start + datetime.timedelta(days=shift)
        date = day.strftime("%Y/%m/%d")
        prefix = f"{date}/{icao}/"
        pattern = f"{prefix}{icao}*_V0?"
        fs = fsspec.filesystem("s3", anon=True)
        files.extend(fs.glob(f"s3://unidata-nexrad-level2/{pattern}"))

    if not files:
        raise FileNotFoundError(f"No radar volumes found for {icao} near {date}")

    def extract_timestamp(key: str) -> datetime.datetime:
        fname = key.split("/")[-1]
        ts = fname[len(icao):len(icao)+15]
        return datetime.datetime.strptime(ts, "%Y%m%d_%H%M%S")

    keys = []
    for f in files:
        ts = extract_timestamp(f)
        if ts < time_start:
            keys = [f]
        if time_start <= ts <= time_end:
            keys.append(f)
        if ts > time_end:
            break

    dtrees = []
    for key in keys:
        local_path = naaulu.network.download_s3_file(s3_url=key)
        try:
            dtree = xradar.io.open_nexradlevel2_datatree(local_path)
        except KeyError as e:
            logger.debug(traceback.print_exc())
            logger.info(f"Skipping {key} due to sweep error: {e}")
            continue
        for s in xradar.util.get_sweep_keys(dtree):
            sweep = clean_sweep(dtree[s].ds)
            if sweep is None:
                del dtree[s]
                continue
            dtree[s].ds = sweep
        dtrees.append(dtree)
        print(dtree)

    return dtrees


def radar(time, duration, wsi):
    # wsi here is expected to be the NEXRAD code
    return nexrad(time=time, duration=duration, code=wsi)
