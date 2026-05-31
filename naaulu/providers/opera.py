import datetime
import json
import logging
import os
import threading
import xml.etree.ElementTree as ET

import pycountry
import requests
import xradar

import naaulu.config
import naaulu.network

logger = logging.getLogger(__name__)

S3_ENDPOINT = "https://s3.waw3-1.cloudferro.com"
S3_BUCKET = "openradar-24h"

database = None
_database_lock = threading.Lock()


def _s3_list(prefix):
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    keys = []
    continuation_token = None
    while True:
        url = f"{S3_ENDPOINT}/{S3_BUCKET}/?prefix={prefix}&list-type=2&max-keys=1000"
        if continuation_token:
            url += f"&continuation-token={continuation_token}"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        root = ET.fromstring(response.text)
        for content in root.findall("s3:Contents", ns):
            key = content.find("s3:Key", ns).text
            keys.append(key)
        is_truncated = root.find("s3:IsTruncated", ns)
        if is_truncated is None or is_truncated.text != "true":
            break
        token_elem = root.find("s3:NextContinuationToken", ns)
        if token_elem is None:
            break
        continuation_token = token_elem.text
    return keys


def _country_alpha2(country_name):
    country = pycountry.countries.lookup(country_name)
    return country.alpha_2


def get_database():

    global database
    if database is not None:
        return database

    with _database_lock:
        if database is not None:
            return database

        filename = os.path.join(naaulu.config.get_data_dir(), "radars_opera.json")
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                database = json.load(f)
        else:
            raise ValueError(f"file {os.path.basename(filename)} missing, check installation")

    return database


def _list_pvol_times(datedir, country, node):
    times = set()
    for subdir in ("PVOL", "SCAN"):
        prefix = f"{datedir}/{country}/{node}/{subdir}/"
        try:
            keys = _s3_list(prefix)
        except Exception:
            continue
        for k in keys:
            fname = k.split("/")[-1]
            try:
                timestr = fname.split("@")[1]
                times.add(timestr)
            except IndexError:
                continue
    return sorted(times)


def radar(time, duration, wsi):
    db = get_database()
    if wsi not in db:
        # tolerate different organisation identifiers (20000/20010/21010)
        parts = wsi.split("-")
        if len(parts) >= 4:
            local = parts[3]
            for org in ("20000", "20010", "21010"):
                alt = f"0-{org}-0-{local}"
                if alt in db:
                    wsi = alt
                    break
        if wsi not in db:
            raise ValueError(f"WIGOS ID {wsi} not found in OPERA database")
    entry = db[wsi]
    node = entry.get("odimcode")
    country_name = entry["country"]
    country = _country_alpha2(country_name)

    time_start = time - duration
    start_str = time_start.strftime("%Y%m%dT%H%M")
    end_str = time.strftime("%Y%m%dT%H%M")

    volumes = []
    day = time
    for _ in range(3):
        datedir = day.strftime("%Y/%m/%d")
        timestrs = _list_pvol_times(datedir, country, node)

        for subdir in ("PVOL", "SCAN"):
            if subdir == "PVOL":
                # PVOL: exact match on volume start time
                matching = [t for t in timestrs if t == start_str]
            else:
                # SCAN: any file whose timestamp falls inside [time-duration, time-1min]
                end_minus_1 = (time - datetime.timedelta(minutes=1)).strftime("%Y%m%dT%H%M")
                matching = [t for t in timestrs if start_str <= t <= end_minus_1]
            for timestr in matching:
                prefix = f"{datedir}/{country}/{node}/{subdir}/{node}@{timestr}"
                try:
                    keys = _s3_list(prefix)
                except Exception:
                    continue
                if not keys:
                    continue

                dbzh_key = None
                for k in keys:
                    if k.endswith("@DBZH.h5") or "@DBZH." in k:
                        dbzh_key = k
                        break
                if dbzh_key is None:
                    dbzh_key = keys[0]

                url = f"{S3_ENDPOINT}/{S3_BUCKET}/{dbzh_key}"
                try:
                    filepath = naaulu.network.download(url)
                    volume = xradar.io.open_odim_datatree(filepath)
                    volumes.append(volume)
                except Exception:
                    logger.debug(f"failed to download/open {dbzh_key}", exc_info=True)

        if volumes:
            break

        day -= datetime.timedelta(days=1)

    if not volumes:
        raise RuntimeError(
            f"No OPERA radar data for WIGOS {wsi} between "
            f"{start_str} and {end_str}"
        )

    return volumes


def extract_odim_georef(filehandle) -> dict:
    """
    Extract geospatial referencing info from an ODIM HDF5 file.

    Parameters
    ----------
    filehandle : h5py
        File opened by h5py

    Returns
    -------
    dict
        Dictionary containing 'crs', 'bounds', and 'resolution'.
    """
    
    where = f["dataset1/data1/where"]
    what = f["dataset1/data1/what"]

    # Grid geometry
    xsize = int(where.attrs["xsize"])
    ysize = int(where.attrs["ysize"])
    xscale = float(where.attrs["xscale"])
    yscale = float(where.attrs["yscale"])
    xstart = float(where.attrs["xstart"])
    ystart = float(where.attrs["ystart"])

    # Bounding box (upper-left origin)
    minx = int(xstart)
    maxx = int(xstart + xsize * xscale)
    maxy = int(ystart)
    miny = int(ystart - ysize * yscale)

    bounds = (minx, miny, maxx, maxy)
    resolution = (int(xscale), int(yscale))

    # CRS from projdef string
    projdef = what.attrs.get("projdef", None)
    if projdef is not None:
        crs = pyproj.CRS.from_proj4(projdef)
    else:
        raise ValueError("Missing 'projdef' attribute in ODIM file.")

    return crs, bounds, resolution
