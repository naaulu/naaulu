import datetime
import json
import logging
import os
import math
import re
import threading
import warnings
import xml.etree.ElementTree

import numpy
import requests
import wradlib
import xradar
import xarray

import naaulu.config
import naaulu.geography
import naaulu.network
import naaulu.util

logger = logging.getLogger(__name__)


# ─── OPERA constants ─────────────────────────────────────────────────────────

S3_ENDPOINT = "https://s3.waw3-1.cloudferro.com"
S3_BUCKET = "openradar-24h"

ODIM_FALLBACK = {
    "0-20010-0-06356": ("nlhrw", "Netherlands"),
}

_opera_database = None
_opera_database_lock = threading.Lock()


# ─── OPERA helpers ───────────────────────────────────────────────────────────

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
        root = xml.etree.ElementTree.fromstring(response.text)
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


def get_opera_database():
    global _opera_database
    if _opera_database is not None:
        return _opera_database

    with _opera_database_lock:
        if _opera_database is not None:
            return _opera_database

        filename = naaulu.config.get_bundled_data_path("radars_opera.json")
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                _opera_database = json.load(f)
        else:
            raise ValueError(f"file {os.path.basename(filename)} missing, check installation")

    return _opera_database


def extract_odim_georef(filehandle) -> dict:
    where = filehandle["dataset1/data1/where"]
    what = filehandle["dataset1/data1/what"]

    xsize = int(where.attrs["xsize"])
    ysize = int(where.attrs["ysize"])
    xscale = float(where.attrs["xscale"])
    yscale = float(where.attrs["yscale"])
    xstart = float(where.attrs["xstart"])
    ystart = float(where.attrs["ystart"])

    minx = int(xstart)
    maxx = int(xstart + xsize * xscale)
    maxy = int(ystart)
    miny = int(ystart - ysize * yscale)

    bounds = (minx, miny, maxx, maxy)
    resolution = (int(xscale), int(yscale))

    projdef = what.attrs.get("projdef", None)
    if projdef is not None:
        import pyproj
        crs = pyproj.CRS.from_proj4(projdef)
    else:
        raise ValueError("Missing 'projdef' attribute in ODIM file.")

    return crs, bounds, resolution


def _validate_hdf5(filepath):
    try:
        import h5py
        with h5py.File(filepath, "r") as f:
            _ = f.keys()
        return True
    except Exception:
        return False


def _opera(time, duration, wsi):
    db = get_opera_database()
    if wsi not in db:
        parts = wsi.split("-")
        if len(parts) >= 4:
            local = parts[3]
            for org in ("20000", "20010", "21010"):
                alt = f"0-{org}-0-{local}"
                if alt in db:
                    wsi = alt
                    break
    entry = db.get(wsi)
    if entry is not None:
        node = entry.get("odimcode")
        country_name = entry["country"]
        country = naaulu.geography.country_code(country_name, alpha=2)
    elif wsi in ODIM_FALLBACK:
        node, country_name = ODIM_FALLBACK[wsi]
        country = naaulu.geography.country_code(country_name, alpha=2)
    else:
        raise ValueError(f"WIGOS ID {wsi} not found in OPERA database")

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
                matching = [t for t in timestrs if t == start_str]
            else:
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
                    if not _validate_hdf5(filepath):
                        logger.warning(f"corrupted radar file, removing cache: {filepath}")
                        os.remove(filepath)
                        continue
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


# ─── Finland (opera for recent, FMI S3 for historical) ──────────────────────

def fin(time, duration, wsi):
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    if time.tzinfo is None:
        time = time.replace(tzinfo=datetime.timezone.utc)
    if (now - time).total_seconds() < 86400:
        return _opera(time, duration, wsi)

    db = get_opera_database()
    if wsi not in db:
        raise ValueError(f"WIGOS ID {wsi} not found")
    node = db[wsi].get("odimcode")
    t = time - duration
    t = t.replace(minute=(t.minute // 5) * 5, second=0, microsecond=0)
    timestamp = t.strftime("%Y%m%d%H%M")
    datedir = t.strftime("%Y/%m/%d")
    filename = f"{timestamp}_{node}_PVOL.h5"
    link = f"http://s3-eu-west-1.amazonaws.com/fmi-opendata-radar-volume-hdf5/{datedir}/{node}/{filename}"
    try:
        local_path = naaulu.network.download(link)
        volume = xradar.io.open_odim_datatree(local_path)
    except Exception:
        raise RuntimeError(f"No radar file available for {node} at {timestamp}")
    return volume


# ─── Estonia (opera for recent, portal API for historical) ───────────────────

_EST_RADAR_NAME = {
    "0-21010-0-42": "SUR",
    "0-21010-0-41": "HAR",
}

_EST_BASE_URL = "https://avaandmed.keskkonnaportaal.ee/api/lists/active"


def est(time, duration, wsi):
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    if time.tzinfo is None:
        time = time.replace(tzinfo=datetime.timezone.utc)
    if (now - time).total_seconds() < 86400:
        return _opera(time, duration, wsi)

    name = _EST_RADAR_NAME[wsi]
    t = time - duration
    timestamp = t.strftime("%Y%m%d%H%M")
    query = {
        "filter": {
            "and": {
                "children": [
                    {"isEqual": {"field": "RadarStation", "value": name}},
                    {"isEqual": {"field": "RadarDataType", "value": "VOL"}},
                    {"contains": {"field": "RMTitle", "value": timestamp}},
                ]
            }
        },
        "limit": 1,
    }

    response = requests.post(
        f"{_EST_BASE_URL}/items/query",
        json=query,
        headers={"Accept": "application/json"},
    )
    response.raise_for_status()
    data = response.json()

    if not data.get("documents"):
        raise RuntimeError(f"No radar file available for {name} at {timestamp}")

    doc = data["documents"][0]
    doc_id = doc["id"]
    filename = f"{name}.{timestamp}.VOL.h5"
    download_url = f"{_EST_BASE_URL}/items/{doc_id}/files/0"

    max_retries = 3
    for attempt in range(max_retries):
        filepath = naaulu.network.download(download_url, filename)
        if _validate_hdf5(filepath):
            volume = xradar.io.open_odim_datatree(filepath)
            return volume
        logger.warning(f"EST radar file corrupted (attempt {attempt + 1}/{max_retries}): {filepath}")
        os.remove(filepath)

    raise RuntimeError(f"EST radar file {filename} is corrupted after {max_retries} attempts")


# ─── USA (NEXRAD S3) ────────────────────────────────────────────────────────


def _usa_clean_sweep(sweep):
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


def _nexrad(time, duration, code):
    time_start = time - duration
    time_end = time

    icao = code.upper()
    files = []
    for shift in [-1, 0, 1]:
        day = time_start + datetime.timedelta(days=shift)
        date = day.strftime("%Y/%m/%d")
        prefix = f"{date}/{icao}/"
        pattern = f"{prefix}{icao}*_V0?"
        import fsspec
        fs = fsspec.filesystem("s3", anon=True)
        files.extend(fs.glob(f"s3://unidata-nexrad-level2/{pattern}"))

    if not files:
        raise FileNotFoundError(f"No radar volumes found for {icao} near {date}")

    def extract_timestamp(key):
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
            logger.debug(f"Skipping {key} due to sweep error: {e}")
            continue
        for s in xradar.util.get_sweep_keys(dtree):
            sweep = _usa_clean_sweep(dtree[s].ds)
            if sweep is None:
                del dtree[s]
                continue
            dtree[s].ds = sweep
        dtrees.append(dtree)

    return dtrees


_usa_mapping = None


def _get_usa_mapping():
    global _usa_mapping
    if _usa_mapping is not None:
        return _usa_mapping

    filename = naaulu.config.get_bundled_data_path("radars_usa.json")
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            _usa_mapping = json.load(f)
    else:
        _usa_mapping = {}
    return _usa_mapping


def usa(time, duration, wsi):
    mapping = _get_usa_mapping()
    if wsi in mapping:
        icao = mapping[wsi][0]
    elif isinstance(wsi, str) and len(wsi) == 4:
        icao = wsi
    else:
        raise ValueError(f"No ICAO code found for WSI {wsi!r}")
    return _nexrad(time=time, duration=duration, code=icao)


# ─── Australia (THREDDS/NCI) ─────────────────────────────────────────────────

def _aus_build_radar_mapping(wigos_db, tolerance=0.01):
    url = "https://dapds00.nci.org.au/thredds/catalog/rrqpe/level_1/catalog.xml"
    xml_path = naaulu.network.download(url, "aura_catalog.xml")

    tree = xml.etree.ElementTree.parse(xml_path)
    root = tree.getroot()

    stations = {}
    for dataset in root.iter("{http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0}dataset"):
        name = dataset.attrib.get("name", "")
        if "_" in name and name.endswith(".vol"):
            icao = name.split("_")[0]
            if icao not in stations:
                stations[icao] = {"lat": None, "lon": None}

    mapping = {}
    for wigos_id, meta in wigos_db.items():
        try:
            lat_wigos = float(meta["latitude"])
            lon_wigos = float(meta["longitude"])
        except (KeyError, ValueError):
            continue

        for icao, site in stations.items():
            if site["lat"] is None or site["lon"] is None:
                continue
            if abs(lat_wigos - site["lat"]) < tolerance and abs(lon_wigos - site["lon"]) < tolerance:
                mapping[wigos_id] = icao
                break

    return mapping


def aus(time, duration, wsi):
    icao = wsi
    date = time.strftime("%Y/%m/%d")
    base_url = "https://dapds00.nci.org.au/thredds/fileServer/rrqpe/level_1"
    path = f"{base_url}/{date}/{icao}/{icao}_{time.strftime('%Y%m%d_%H%M%S')}.vol"
    return path


# ─── WMO database ────────────────────────────────────────────────────────────

_database = None


def get_database():
    global _database
    if _database is not None:
        return _database

    filename = naaulu.config.get_bundled_data_path("radars.json")
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            _database = json.load(f)
    else:
        raise ValueError(f"file {os.path.basename(filename)} missing, check installation")

    return _database


# ─── Build functions ──────────────────────────────────────────────────────────

def build_radar_database(data_dir=None):
    """Build radars.json from WMO WRD."""
    if data_dir is None:
        data_dir = naaulu.config.get_data_dir()
    os.makedirs(data_dir, exist_ok=True)

    logger.info("fetching WMO radar database from WRD")
    search_url = "https://wrd.mgm.gov.tr/Radar/Search"
    payload = {
        "draw": 1,
        "start": 0,
        "length": 10000,
        "INSTALL_YEAR_MIN": 1900,
        "INSTALL_YEAR_MAX": 2026,
    }
    resp = requests.post(search_url, data=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    radars = {}
    for item in data.get("data", []):
        name = item.get("RADAR_NAME", "")
        wsi = item.get("WSI") or name.replace(" ", "_")
        if not wsi:
            continue
        start = item.get("INSTALL_DATE")
        if start:
            try:
                start = start[:10]
            except Exception:
                start = None
        try:
            lon = float(item.get("RADAR_LON")) if item.get("RADAR_LON") else None
            lat = float(item.get("RADAR_LAT")) if item.get("RADAR_LAT") else None
        except (TypeError, ValueError):
            lon = lat = None
        country = item.get("COUNTRY_NAME") or ""
        if country:
            try:
                country = naaulu.geography.country_code(country)
            except ValueError:
                stripped = re.sub(r"\s*\(.*\)\s*$", "", country)
                try:
                    country = naaulu.geography.country_code(stripped)
                except ValueError:
                    country = ""
        radars[wsi] = {"lon": lon, "lat": lat, "start": start, "country": country, "name": name}

    RADAR_COUNTRIES = {"EST", "BEL", "FIN", "DEU", "CZE", "FRA", "NLD", "AUT", "USA", "AUS"}
    filtered = {wsi: r for wsi, r in radars.items() if r["country"] in RADAR_COUNTRIES}

    raw_out = os.path.join(data_dir, "radars_wmo.json")
    with open(raw_out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info(f"wrote {raw_out}")

    out = os.path.join(data_dir, "radars.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2)
    logger.info(f"wrote {out} ({len(filtered)} entries)")
    return filtered


def build_opera_database(data_dir=None):
    """Fetch and build the OPERA radar database."""
    if data_dir is None:
        data_dir = naaulu.config.get_data_dir()
    os.makedirs(data_dir, exist_ok=True)

    logger.info("fetching OPERA radar database")
    url = (
        "https://www.eumetnet.eu/wp-content/themes/aeron-child/"
        "observations-programme/current-activities/opera/database/"
        "OPERA_Database/OPERA_RADARS_DB.json"
    )
    downloaded_path = naaulu.network.download(url)
    with open(downloaded_path, "r", encoding="utf-8") as f:
        radars = json.load(f)
    db = {}
    for radar in radars:
        if radar.get("wigosid", "") != "":
            db[radar["wigosid"]] = radar

    out = os.path.join(data_dir, "radars_opera.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)
    logger.info(f"wrote {out} ({len(db)} entries)")
    return db


def build_usa_database(data_dir=None):
    """Build radars_usa.json: WIGOS ID to NEXRAD ICAO code mapping."""
    if data_dir is None:
        data_dir = naaulu.config.get_data_dir()
    os.makedirs(data_dir, exist_ok=True)

    logger.info("fetching NEXRAD station list")
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

    logger.info(f"parsed {len(stations)} NEXRAD stations")

    radars_path = os.path.join(data_dir, "radars.json")
    with open(radars_path) as f:
        wmo = json.load(f)
    usa = {k: v for k, v in wmo.items() if v.get("country") == "USA"}

    mapping = {}
    tolerance = 0.01
    for wigos_id, meta in usa.items():
        try:
            lat_w = float(meta["lat"])
            lon_w = float(meta["lon"])
        except (KeyError, ValueError):
            continue
        for icao, site in stations.items():
            if abs(lat_w - site["lat"]) < tolerance and abs(lon_w - site["lon"]) < tolerance:
                mapping[wigos_id] = [icao, meta.get("name", "")]
                break

    out = os.path.join(data_dir, "radars_usa.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    logger.info(f"wrote {out} ({len(mapping)}/{len(usa)} mapped)")
    return mapping

    filename = os.path.join(naaulu.config.get_data_dir(), "radars.json")
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            _database = json.load(f)
    else:
        raise ValueError(f"file {os.path.basename(filename)} missing, check installation")

    return _database


def select(*, start, end, geom, distance, key=None):

    radars = get_database()

    selection = None
    if key is not None:
        cache = naaulu.config.get_cache_dir()
        filename = f"radar_select_{distance}.json"
        filename = os.path.join(cache, filename)
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                selections = json.load(f)
                if key in selections:
                    selection = selections[key]
        else:
            selections = {}

    if selection is None:
        selection = []
        for wsi, radar in radars.items():
            location = (radar["lon"], radar["lat"])
            coverage = naaulu.geography.radar_coverage(
                location=location,
                distance=distance,
                )
            if not coverage.intersects(geom):
                continue
            selection.append(wsi)
        if key is not None:
            selections[key] = selection
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(selections, f)

    radars = {wsi: radars[wsi] for wsi in selection}
    radars = {
        k: v
        for k, v in radars.items()
        if v["start"] is None or datetime.date.fromisoformat(v["start"]) <= end.date()
    }
    radars = {
        k: v
        for k, v in radars.items()
        if v.get("end") is None or datetime.date.fromisoformat(v["end"]) >= start.date()
    }

    logger.info(f"{key} => {len(radars)} radars")
    return radars.keys()


# ─── Dispatch ────────────────────────────────────────────────────────────────

_RADAR = {
    "EST": est,
    "BEL": _opera,
    "FIN": fin,
    "DEU": _opera,
    "CZE": _opera,
    "FRA": _opera,
    "NLD": _opera,
    "USA": usa,
    "AUS": aus,
}


def get(time: datetime.datetime, duration: datetime.timedelta, wsi: str):
    radars = get_database()
    if wsi not in radars:
        raise ValueError(f"No radar provider for WSI {wsi}")

    entry = radars[wsi]
    country = entry.get("country")
    if not country or country not in _RADAR:
        raise ValueError(f"No radar provider for country {country}")

    func = _RADAR[country]
    return func(time=time, duration=duration, wsi=wsi)


# ─── Path / I/O ──────────────────────────────────────────────────────────────

def path(*,
    time,
    duration,
    country,
    wsi,
    min_angle=0,
    max_angle=90,
    ):
    time_str = naaulu.util.format_time(time)
    wsi_str = wsi.replace("-","_")
    duration_str = naaulu.util.format_duration(duration)
    angle_str = f"{min_angle:.1f}_{max_angle:.1f}".replace(".", "")
    filename = ".".join(
        [
            time_str,
            duration_str,
            country.lower(),
            wsi_str,
            angle_str,
            "nc"
        ]
    )
    archive = naaulu.config.get_archive_dir()
    if archive is not None:
        root = os.path.join(archive, "radar")
        filename = naaulu.util.get_path(root, filename)

    return filename


def write(volume, filename):

    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(os.path.dirname(filename), exist_ok=True)

    for sweep_name in volume.children:
        sweep = volume[sweep_name]
        if sweep.ds is not None:
            for var in xradar.util.get_sweep_dataset_vars(sweep.ds):
                sweep.ds[var].encoding["zlib"] = True
                sweep.ds[var].encoding["complevel"] = 4

    volume.to_netcdf(filename, engine="h5netcdf")


def get_volume(*, time, duration, wsi, min_angle, max_angle):
    sweeps = get(time=time, duration=duration, wsi=wsi)
    if not isinstance(sweeps, list):
        sweeps = [sweeps]
    return xradar.util.create_volume(
        sweeps=sweeps,
        time_coverage_start=time - duration,
        time_coverage_end=time,
        min_angle=min_angle,
        max_angle=max_angle,
    )


def combine_volume(
    time,
    duration,
    wsi,
    azimuth_scale,
    range_scale,
    max_range,
    min_angle,
    max_angle,
    variables,
    precision,
    update=True,
    ):

    if update == True:
        try:
            volume = combine_volume(
                time,
                duration,
                wsi,
                azimuth_scale,
                range_scale,
                max_range,
                min_angle,
                max_angle,
                variables,
                precision,
                update=False,
            )
            return volume
        except FileNotFoundError:
            pass

    radars = get_database()
    radar = radars[wsi]
    country = radar.get("country")

    filename = path(
        time=time,
        country=country,
        wsi=wsi,
        duration=duration,
        min_angle=min_angle,
        max_angle=max_angle,
        )

    if update == False:
        volume = xarray.open_datatree(filename, engine="h5netcdf")
        volume = volume.load()
        volume.close()
        volume = xradar.util.create_volume(
            sweeps=[volume],
            min_angle=min_angle,
            max_angle=max_angle,
        )
    else:
        volume = get_volume(
            time=time,
            duration=duration,
            wsi=wsi,
            min_angle=min_angle,
            max_angle=max_angle,
        )

    volume = xradar.util.apply_to_volume(
        volume,
        cut,
        max_range
        )
    sweeps = volume.ds.sweep_group_name.values

    volume = select_sweeps(volume, sweeps)
    volume = xradar.util.create_volume([volume])

    volume = xradar.util.apply_to_volume(
        volume,
        rescale,
        azimuth_scale,
        range_scale
        )

    volume = xradar.util.apply_to_volume(
        volume,
        recode_dbzh,
        precision)


    volume = xradar.util.apply_to_volume(
        volume, xradar.util.select_sweep_dataset_vars, variables
    )

    if update:
        logger.info(f"saving radar volume: {time} {country} {wsi}")
        write(volume, filename)

    volume = xradar.util.apply_to_volume(
        volume,
        retype,
        numpy.float32
        )

    sweep_keys = xradar.util.get_sweep_keys(volume)
    for key in sweep_keys:
        dbzh = volume[key].ds["DBZH"].values
        total = dbzh.size
        nan_count = numpy.isnan(dbzh).sum()
        below_noise = (dbzh < 5).sum()
        logger.debug(f"volume {time} {country} {wsi} {key}: DBZH NaN={nan_count}/{total} ({100*nan_count/total:.1f}%), below5dBZ={below_noise}/{total} ({100*below_noise/total:.1f}%)")

    return volume


# ─── Sweep utilities ─────────────────────────────────────────────────────────

def retype(sweep, type):
    for var in xradar.util.get_sweep_dataset_vars(sweep):
        sweep[var] = sweep[var].astype(type)

    return sweep


def cut(sweep, max_range):

    if sweep.range.values[-1] >= max_range:
        sweep = sweep.sel(range=slice(0, max_range))
    else:
        dr = float(sweep.range.values[1] - sweep.range.values[0])
        n = int(max_range / dr)
        m = sweep.range.size
        if n > m:
            new_range = numpy.arange(0, n * dr, dr)
            vars = {}
            for var in xradar.util.get_sweep_dataset_vars(sweep):
                old = sweep[var].values
                new = numpy.full((sweep.azimuth.size, n), numpy.nan, dtype=old.dtype)
                new[:, :m] = old
                vars[var] = xarray.DataArray(
                    new, dims=["azimuth", "range"],
                    coords={"azimuth": sweep.azimuth, "range": new_range},
                )
            ds = xarray.Dataset(vars, coords={"azimuth": sweep.azimuth, "range": new_range})
            for coord in sweep.coords:
                if coord not in ("azimuth", "range"):
                    ds[coord] = sweep[coord]
            for var in xradar.util.get_sweep_metadata_vars(sweep):
                if var not in ds:
                    ds[var] = sweep[var]
            sweep = ds

    return sweep


def rescale(sweep, ascale, rscale):
    range_res = sweep.range.values[1] - sweep.range.values[0]
    nrange = max(round(rscale / range_res), 1)

    azimuth_res = sweep.azimuth.values[1] - sweep.azimuth.values[0]
    nazimuth = max(round(ascale / azimuth_res), 1)

    n = int(sweep.range.values[-1] / rscale) + 1
    target_range = numpy.arange(0, n * rscale, rscale)

    sweep_new = []

    for var in xradar.util.get_sweep_dataset_vars(sweep):
        sweep_var_ds = sweep[[var]]
        if var == "DBZH":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                sweep_var_ds[var].values = 10 ** (sweep_var_ds[var].values / 10)

        if nrange > 1 or nazimuth > 1:
            sweep_var_ds = sweep_var_ds.coarsen(
                azimuth=nazimuth, range=nrange, boundary="trim"
            ).mean()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sweep_var_ds = sweep_var_ds.interp(range=target_range)

        if var == "DBZH":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                sweep_var_ds[var].values = 10 * numpy.log10(sweep_var_ds[var].values)
        sweep_new.append(sweep_var_ds)

    sweep_new = xarray.merge(sweep_new, compat='override')

    for var in xradar.util.get_sweep_metadata_vars(sweep):
        sweep_new[var] = sweep[var]

    return sweep_new


def recode_dbzh(sweep, precision):

    sweep["DBZH"] = sweep["DBZH"].clip(min=-32, max=95.5)
    sweep["DBZH"] = xarray.where(sweep["DBZH"] < 5, -32, sweep["DBZH"])

    enc = {}
    if precision == 8:
        enc["scale_factor"] = 0.5
        enc["add_offset"] = -32
        enc["_FillValue"] = 255
        enc["dtype"] = numpy.uint8

    if precision == 16:
        enc["scale_factor"] = 0.01
        enc["add_offset"] = -32
        enc["_FillValue"] = 65535
        enc["dtype"] = numpy.uint16

    sweep["DBZH"].encoding.update(enc)

    return sweep


def spatial_reference(sweep):

    elangle = float(sweep["sweep_fixed_angle"].values)
    nrays = sweep.azimuth.size
    nbins = sweep.range.size
    rscale = sweep.range.values[1] - sweep.range.values[0]
    reference = (elangle, nrays, nbins, rscale)

    return reference


def timestamp(sweep, mode="min"):
    time = sweep.time.values
    if mode == "min":
        time = numpy.min(time)
    if mode == "max":
        time = numpy.max(time)

    return time


def select_sweeps(dt: xarray.DataTree, sweep_names: list[str]) -> xarray.DataTree:
    selected = {}
    for name in sweep_names:
        if name in dt:
            selected[name] = dt[name]
            selected[name].ds = selected[name].ds.assign_coords(
                sweep_fixed_angle =     selected[name].ds.sweep_fixed_angle
                )
    dt = xarray.DataTree(name=dt.name, dataset=dt.ds, children=selected)
    return dt


def add_quality(volume, *, name, fun, long_name, units="1", fun_kwargs=None):
    def _apply(sweep):
        sweep[name] = fun(sweep, **(fun_kwargs or {}))
        sweep[name].attrs = {"long_name": long_name, "units": units}
        return sweep
    return xradar.util.apply_to_sweeps(volume, _apply)


def quality_filter_window_distance(sweep, fsize=1500, tr1=6.0):
    return sweep.wrl.classify.filter_window_distance(fsize=fsize, tr1=tr1).DBZH


def quality_distance(sweep):
    return numpy.clip(1.0 - sweep.range / 300_000, 0, 1)


def quality_height(sweep, site_alt, max_alt=3000.0):
    h = (sweep.z - site_alt) / max_alt
    return numpy.clip((1.0 - h) / 0.3, 0, 1)


def echotop(volume, level=7):
    sweep_keys = list(volume.ds.sweep_group_name.values)
    fixed_angles = volume.ds.sweep_fixed_angle.values
    sorted_indices = numpy.argsort(fixed_angles)
    sorted_keys = [sweep_keys[i] for i in sorted_indices]

    if not sorted_keys:
        return None

    ref = volume[sorted_keys[0]].ds.copy()
    et = numpy.full(ref.DBZH.shape, numpy.nan)

    if logging.getLogger().isEnabledFor(logging.DEBUG):
        logging.debug(f"echotop: {len(sorted_keys)} sweeps, DBZH shape {ref.DBZH.shape}")

    for key in sorted_keys:
        sweep = volume[key].ds
        dbzh = sweep.DBZH.values
        z = sweep.z.values
        nrng = min(ref.sizes["range"], sweep.sizes["range"])
        mask = (dbzh[:, :nrng] >= level) & ~numpy.isnan(dbzh[:, :nrng])
        et[:, :nrng] = numpy.where(
            mask, numpy.fmax(et[:, :nrng], z[:, :nrng]), et[:, :nrng],
        )
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            above = mask.sum()
            logging.debug(
                f"  sweep {key}: DBZH {dbzh.shape}, range {nrng}, "
                f"dbzh [{numpy.nanmin(dbzh):.1f}, {numpy.nanmax(dbzh):.1f}], "
                f"gates >= {level}dBZ: {above}/{mask.size}"
            )

    ref["echotop"] = (("azimuth", "range"), et)
    ref["echotop"].attrs = {
        "long_name": f"echo top height ({level} dBZ)",
        "units": "meters",
    }
    return ref


def _ensure_georeferenced(ds):
    if "x" in ds.coords and "y" in ds.coords:
        return ds
    return ds.xradar.georeference()


def fill_gap_nearest_elevation(volume, *, base_key):
    if base_key not in volume:
        return None

    sweep_keys = list(volume.ds.sweep_group_name.values)
    fixed_angles = volume.ds.sweep_fixed_angle.values
    base_idx = sweep_keys.index(base_key)
    base_angle = float(fixed_angles[base_idx])

    higher = [
        (float(fixed_angles[i]), sweep_keys[i])
        for i in range(len(sweep_keys))
        if float(fixed_angles[i]) >= base_angle and sweep_keys[i] != base_key
    ]
    higher.sort(key=lambda p: p[0])
    higher_keys = [k for _, k in higher]

    base = _ensure_georeferenced(volume[base_key].ds.copy(deep=True))
    base_ground = numpy.sqrt(
        numpy.asarray(base["x"].values) ** 2
        + numpy.asarray(base["y"].values) ** 2
    )
    base_gr = base_ground[0]
    base_az = numpy.asarray(base["azimuth"].values)
    n_az_base = base.sizes["azimuth"]

    dbzh = numpy.asarray(base["DBZH"].values, dtype=numpy.float64).copy()

    for higher_key in higher_keys:
        nan_mask = numpy.isnan(dbzh)
        if not nan_mask.any():
            break

        src = _ensure_georeferenced(volume[higher_key].ds)
        src_ground = numpy.sqrt(
            numpy.asarray(src["x"].values) ** 2
            + numpy.asarray(src["y"].values) ** 2
        )
        src_gr = src_ground[0]
        src_vals = numpy.asarray(src["DBZH"].values, dtype=numpy.float64)

        if src_gr.size < 2 or not numpy.all(numpy.diff(src_gr) > 0):
            order = numpy.argsort(src_gr)
            src_gr = src_gr[order]
            src_vals = src_vals[:, order]

        idx = numpy.searchsorted(src_gr, base_gr)
        valid_range = (idx > 0) & (idx < src_gr.size)
        idx_safe = numpy.clip(idx, 1, src_gr.size - 1)
        left = idx_safe - 1
        right = idx_safe
        gl = src_gr[left]
        gr = src_gr[right]
        denom = gr - gl
        w = numpy.where(denom > 0, (base_gr - gl) / denom, 0.0)

        n_az_src = src.sizes["azimuth"]
        if n_az_src == n_az_base:
            az_map = numpy.arange(n_az_base)
        else:
            src_az = numpy.asarray(src["azimuth"].values)
            diff = numpy.abs(((base_az[:, None] - src_az[None, :] + 180) % 360) - 180)
            az_map = numpy.argmin(diff, axis=1)

        vals_left = src_vals[az_map][:, left]
        vals_right = src_vals[az_map][:, right]
        interp = vals_left * (1.0 - w[None, :]) + vals_right * w[None, :]
        interp = numpy.where(valid_range[None, :], interp, numpy.nan)

        fill = nan_mask & ~numpy.isnan(interp)
        dbzh = numpy.where(fill, interp, dbzh)

    base["DBZH"].values[:] = dbzh
    return base


def plot(sweep, variable="DBZH"):
    sweep = sweep.xradar.georeference()
    sweep[variable].plot.pcolormesh(shading="auto")
