import datetime
import json
import logging
import os
import sqlite3
import subprocess
import tempfile
import threading
import time
import xml.etree.ElementTree
import zoneinfo
from concurrent.futures import ThreadPoolExecutor, as_completed


import numpy
import xarray

import naaulu.config
import naaulu.errors
import naaulu.geography
import naaulu.network
import naaulu.util

logger = logging.getLogger(__name__)


# ─── Core gauge infrastructure ───────────────────────────────────────────────

# Durations served natively; longer ones are summed from these by get_network.
SUPPORTED_DURATIONS = {
    datetime.timedelta(hours=1),
    datetime.timedelta(minutes=10),
}


def _composing_step(duration):
    """The largest native step that evenly divides `duration`, or None if it cannot be composed."""
    steps = [
        step for step in SUPPORTED_DURATIONS
        if duration > step and duration % step == datetime.timedelta(0)
    ]
    return max(steps, default=None)


def can_serve(duration):
    """True when get_network can produce `duration` (native or composed)."""
    return duration in SUPPORTED_DURATIONS or _composing_step(duration) is not None


def path(time, duration, country):

    time_str = naaulu.util.format_time(time)
    duration_str = naaulu.util.format_duration(duration)
    filename = f"{time_str}.{duration_str}.{country.lower()}.nc"
    archive = naaulu.config.get_archive_dir()
    if archive is not None:
        root = os.path.join(archive, "gauge")
        filename = naaulu.util.get_path(root, filename)

    return filename


def create_dataset(*, codes, coords, values):
    longitudes = coords[:,0]
    latitudes = coords[:,1]

    dataset = xarray.Dataset(
        {
            "precipitation": xarray.DataArray(
                values,
                dims=["station"],
                coords={"station": codes},
                attrs={"units": "mm", "description": "Accumulated precipitation"}
            )
        },
        coords={
            "longitude": ("station", longitudes),
            "latitude": ("station", latitudes)
        }
    )

    return dataset


def get_dataset_coordinates(ds):
    longitudes = ds.longitude.values
    latitudes = ds.latitude.values
    coords = numpy.column_stack((longitudes, latitudes))

    return coords


def _get_network_single(func, time, duration):
    """One provider call at a native duration, as a station dataset."""
    values, codes, coords = _fetch_network(func, time, duration)

    return create_dataset(
        coords=coords,
        codes=codes,
        values=values,
        )


def _compose_network(time, duration, step, country):
    """`duration` as the sum of the consecutive `step` parts ending `time`.

    A station missing from any part drops out of the sum.
    """
    parts = duration // step
    logger.info(
        f"composing gauge duration {duration} for {country} from "
        f"{parts} {step} accumulations ending {time}"
    )
    total = None
    attrs = {}
    for i in range(parts):
        part = get_network(time=time - i * step, duration=step, country=country)
        attrs = part["precipitation"].attrs
        total = part if total is None else total + part

    # arithmetic strips the attributes
    total["precipitation"].attrs = attrs
    return total


def get_network(*, time, duration, country):
    """The `duration` accumulation ending at `time` for one network.

    Non-native durations are summed from the consecutive native parts;
    a station missing from a part drops out of the sum.
    """
    filename = path(time, duration, country)
    if os.path.exists(filename):
        logger.info(f"Using cached gauge data at {time} for duration {duration} from {country}")
        ds = xarray.open_dataset(filename, engine="h5netcdf")
        return ds

    func = _GAUGE.get(country.lower())
    if func is None:
        raise ValueError(f"Gauge network {country} not available")

    # native durations are never composed
    step = None if duration in SUPPORTED_DURATIONS else _composing_step(duration)
    if step is None:
        ds = _get_network_single(func, time, duration)
    else:
        ds = _compose_network(time, duration, step, country)

    # with --no-archive there is nowhere to cache: skip the write rather
    # than failing on os.makedirs("")
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
        ds.to_netcdf(filename, engine="h5netcdf")

    return ds


def concat_datasets(*, datasets, times):
    dataset = xarray.concat(
        datasets,
        xarray.DataArray(times, dims=["time"]),
        coords="minimal",
        compat="no_conflicts",
    )
    return dataset


def _fetch_network(func, when, duration, attempts=2):
    """One provider call, retried once on a transport failure."""
    for attempt in range(attempts):
        try:
            return func(when, duration)
        except (naaulu.errors.NoDataError, ValueError):
            raise
        except Exception as exc:
            if attempt == attempts - 1:
                raise
            logger.info(f"{when} {duration}: {exc}, retrying once")
            time.sleep(1.0)


def collect(times, geometry, duration, countries):

    # ghcn and asos are helpers needing a country argument, not keys of their own
    helpers = ("ghcn", "asos")
    unknown = [name for name in countries if name.lower() not in _GAUGE]
    if unknown:
        raise ValueError(
            f"Unknown gauge network(s): {', '.join(unknown)}; available: "
            + ", ".join(sorted(key for key in _GAUGE if key not in helpers))
        )

    gauges = []
    for time in times:
        datasets = []
        for country in countries:
            try:
                dataset = get_network(
                    time=time,
                    duration=duration,
                    country=country,
                    )
                datasets.append(dataset)
            except naaulu.errors.NoDataError:
                logger.debug(f"no gauge data available: {time} {duration} {country}")
            except ValueError:
                logger.debug(f"{country}: duration {duration} not served at {time}")
            except Exception as exc:
                logger.warning(f"gauge network {country} failed at {time}: {exc}")

        if not datasets:
            raise naaulu.errors.NoDataError(
                f"No gauge data at {naaulu.util.format_time(time, show=True)} "
                f"for {', '.join(countries)}"
            )

        ds = xarray.concat(datasets, dim="station")
        ds = naaulu.geography.cut(ds, geometry)
        gauges.append(ds)

    gauges = concat_datasets(
        datasets=gauges,
        times=times
    )

    return gauges


# ─── GHCN (hourly, global) ──────────────────────────────────────────────────

_ghcnh_monthly_cache = {}
_ghcnh_station_cache = None


def _ghcn_get_inventory_path():
    return os.path.join(naaulu.config.get_data_dir("ghcnh"), "ghcnh-inventory.txt")


def _ghcn_get_station_list_path():
    return os.path.join(naaulu.config.get_data_dir("ghcnh"), "ghcnh-station-list.txt")


def _ghcn_load_station_list():
    global _ghcnh_station_cache
    if _ghcnh_station_cache is not None:
        return _ghcnh_station_cache

    stn_path = _ghcn_get_station_list_path()
    if not os.path.exists(stn_path):
        url = "https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/doc/ghcnh-station-list.txt"
        os.makedirs(os.path.dirname(stn_path), exist_ok=True)
        subprocess.run(["curl", "-sf", "--max-time", "60", "-o", stn_path, url], check=True, timeout=65)

    stations = []
    with open(stn_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            stations.append({
                "id": line[0:11].strip(),
                "lat": float(line[12:20].strip()),
                "lon": float(line[21:30].strip()),
                "elevation": float(line[31:37].strip() or "0"),
                "name": line[38:71].strip(),
                "country": line[86:88].strip(),
            })
    _ghcnh_station_cache = stations
    return stations


def _ghcn_get_active_station_ids(year, month):
    for y, m in _ghcn_iter_months_back(year, month, max_months=6):
        key = (y, m)
        if key in _ghcnh_monthly_cache:
            active = _ghcnh_monthly_cache[key]
            if active:
                if (y, m) != (year, month):
                    logger.info(f"ghcn: using {y}-{m:02d} inventory (no data for {year}-{month:02d})")
                return active
            continue

        inv_path = _ghcn_get_inventory_path()
        if not os.path.exists(inv_path):
            url = "https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/doc/ghcnh-inventory.txt"
            os.makedirs(os.path.dirname(inv_path), exist_ok=True)
            subprocess.run(["curl", "-sf", "--max-time", "120", "-o", inv_path, url], check=True, timeout=125)

        col = m

        active = set()
        with open(inv_path, encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                try:
                    yr = int(parts[1])
                except ValueError:
                    continue
                if yr == y:
                    count = int(parts[col + 1])
                    if count > 0:
                        active.add(parts[0])

        _ghcnh_monthly_cache[key] = active
        if active:
            if (y, m) != (year, month):
                logger.info(f"ghcn: using {y}-{m:02d} inventory ({len(active)} stations, no data for {year}-{month:02d})")
            else:
                logger.info(f"ghcn: {len(active)} stations with data for {year}-{month:02d}")
            return active

    logger.warning(f"ghcn: no stations found in inventory for {year}-{month:02d} or prior months")
    return set()


def _ghcn_iter_months_back(year, month, max_months=6):
    y, m = year, month
    for _ in range(max_months):
        yield y, m
        m -= 1
        if m < 1:
            m = 12
            y -= 1


def _ghcn_fetch_s3_station(args):
    import pyarrow.parquet as pq

    sid, year, target_month, target_day, target_hour = args
    s3_key = f"s3://noaa-ghcnh-pds/hourly/access/by-year/{year}/parquet/GHCNh_{sid}_{year}.parquet"

    data_dir = naaulu.config.get_data_dir("ghcnh")
    filename = f"GHCNh_{sid}_{year}.parquet"
    local_path = os.path.join(data_dir, filename)

    try:
        if not os.path.exists(local_path):
            import boto3
            from botocore import UNSIGNED
            from botocore.config import Config
            bucket, key = s3_key.replace("s3://", "").split("/", 1)
            s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3_client.download_file(bucket, key, local_path)

        columns = ["STATION", "precipitation", "LATITUDE", "LONGITUDE", "Month", "Day", "Hour"]
        target_m = f"{target_month:02d}"
        target_d = f"{target_day:02d}"
        target_h = f"{target_hour:02d}"

        table = pq.read_table(
            local_path,
            columns=columns,
            filters=[
                ("Month", "==", target_m),
                ("Day", "==", target_d),
                ("Hour", "==", target_h),
            ],
        )
        if table.num_rows == 0:
            return None

        precip = table.column("precipitation")[0].as_py()
        return {
            "val": float(precip) if precip is not None else None,
            "code": sid,
            "coords": [float(table.column("LONGITUDE")[0].as_py()),
                        float(table.column("LATITUDE")[0].as_py())],
        }
    except Exception:
        return None


def _ghcn_s3(time, duration, station_ids):
    year = time.year
    m, d, h = time.month, time.day, time.hour

    args_list = [(sid, year, m, d, h) for sid in station_ids]

    vals, codes, coords_list = [], [], []

    max_workers = 20
    logger.info(f"ghcn_s3: querying {len(station_ids)} stations on S3 for {time.isoformat()}")

    from tqdm import tqdm
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_ghcn_fetch_s3_station, a): a[0] for a in args_list}
        disable = not logging.getLogger().isEnabledFor(logging.INFO)
        with tqdm(total=len(futures), desc="ghcn_s3", unit="stn", disable=disable) as pbar:
            for future in as_completed(futures):
                res = future.result()
                if res and res["val"] is not None and res["val"] >= 0:
                    vals.append(res["val"])
                    codes.append(res["code"])
                    coords_list.append(res["coords"])
                pbar.update(1)

    return vals, codes, coords_list


def ghcn(time: datetime.datetime, duration, country: str):
    """Global GHCN hourly gauges; country is a mandatory ISO 2-letter code."""
    country = country.upper()
    station_list = _ghcn_load_station_list()
    station_list = [s for s in station_list if s["country"] == country]

    active_ids = _ghcn_get_active_station_ids(time.year, time.month)
    if active_ids is not None:
        station_list = [s for s in station_list if s["id"] in active_ids]

    if not station_list:
        logger.info("ghcn: no stations found")
        return [], [], numpy.array([]).reshape((0, 2))

    station_ids = [s["id"] for s in station_list]

    res = _ghcn_s3(time, duration, station_ids)
    if res:
        vals, codes, coords_list = res
        if vals:
            coords = numpy.array(coords_list)
            logger.info(f"ghcn: returning {len(vals)} stations via S3 Parquet")
            return vals, codes, coords

    return [], [], numpy.array([]).reshape((0, 2))


# ─── ASOS (hourly, global via IEM) ──────────────────────────────────────────

INCH_TO_MM = 25.4
_station_coords_cache = {}


def _asos_station_coords(network):
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


def asos(time: datetime.datetime, duration: datetime.timedelta, country: str):
    """Global ASOS hourly gauges via IEM; country is a mandatory ISO 2-letter code."""
    if duration != datetime.timedelta(hours=1):
        raise ValueError("Only 1-hour duration supported")

    country = country.upper()
    # IEM spells the international networks XX__ASOS and the US state ones
    # XX_ASOS; naaulu keys are ISO country codes, so always the double form.
    network = f"{country}__ASOS"

    station_meta = _asos_station_coords(network)
    if not station_meta:
        logger.info(f"asos: no stations found for network {network}")
        return [], [], numpy.empty((0, 2))

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
    best = {}
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
            continue
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

    coords = numpy.column_stack((longitudes, latitudes)) if codes else numpy.empty((0, 2))
    return (values, codes, coords)


# One key per country covered by the IEM ASOS feed, so a network can be
# picked either through its national feed or through this uniform one.
# The national feeds reach further back and denser, ASOS is the common
# fallback; Germany and France publish no open national feed, so their
# country keys are deu_asos and fra_asos.

def bel_asos(time, duration):
    """Belgian ASOS hourly gauges via IEM."""
    return asos(time, duration, "BE")


def cze_asos(time, duration):
    """Czech ASOS hourly gauges via IEM."""
    return asos(time, duration, "CZ")


def deu_asos(time, duration):
    """German ASOS hourly gauges via IEM."""
    return asos(time, duration, "DE")


def est_asos(time, duration):
    """Estonian ASOS hourly gauges via IEM."""
    return asos(time, duration, "EE")


def fin_asos(time, duration):
    """Finnish ASOS hourly gauges via IEM."""
    return asos(time, duration, "FI")


def fra_asos(time, duration):
    """French ASOS hourly gauges via IEM."""
    return asos(time, duration, "FR")


def lux_asos(time, duration):
    """Luxembourgish ASOS hourly gauges via IEM (Findel only)."""
    return asos(time, duration, "LU")


def nld_asos(time, duration):
    """Dutch ASOS hourly gauges via IEM."""
    return asos(time, duration, "NL")


# ─── Belgium ─────────────────────────────────────────────────────────────────

BEL_WFS_URL = "https://opendata.meteo.be/geoserver/ows"


def _bel_rmib(time, duration):
    """RMI AWS observations from the opendata.meteo.be WFS.

    Stamps mark the end of the period; timestamps are sent as UTC.
    """
    # normalise to UTC: strftime() would print the caller's clock
    timestamp = naaulu.util.naive_utc(time).strftime("%Y-%m-%dT%H:%M:%SZ")

    if duration == datetime.timedelta(minutes=10):
        layer = "aws:aws_10min"
    elif duration == datetime.timedelta(hours=1):
        layer = "aws:aws_1hour"
    else:
        raise ValueError(f"Unsupported duration for RMI AWS: {duration}")

    params = {
        "service": "wfs",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeName": layer,
        "CQL_FILTER": f"timestamp='{timestamp}'",
        "outputFormat": "application/json",
    }

    response = naaulu.network.runtime_session().get(BEL_WFS_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    features = data.get("features", [])

    values = []
    codes = []
    longitudes = []
    latitudes = []

    for f in features:
        props = f["properties"]
        precip = props.get("precip_quantity")
        if precip is None:
            continue

        code = str(props["code"])
        lon, lat = f["geometry"]["coordinates"]

        values.append(float(precip))
        codes.append(code)
        longitudes.append(lon)
        latitudes.append(lat)

    if not values:
        raise naaulu.errors.NoDataError(f"No AWS data from RMI at {timestamp}")

    coords = numpy.column_stack((longitudes, latitudes))
    return values, codes, coords


# ─── Belgium: the two KiWIS deployments (SPW Wallonia, VMM Flanders) ────────

KIWIS_BASE = {"service": "kisters", "type": "queryServices", "datasource": 0}
KIWIS_TZ = zoneinfo.ZoneInfo("Europe/Brussels")
KIWIS_MAX_AGE = 7 * 24 * 3600.0    # station metadata, re-read weekly

_series_cache = {}
_series_lock = threading.Lock()


def _kiwis_get(url, params, fmt, label):
    """One anonymous KiWIS call; provider errors are raised here as exceptions."""
    query = dict(KIWIS_BASE)
    query.update(params)
    query["format"] = fmt

    response = naaulu.network.runtime_session().get(url, params=query, timeout=30)
    response.raise_for_status()

    try:
        payload = response.json()
    except ValueError:
        raise RuntimeError(f"{label} KiWIS returned no JSON: {response.text[:200]}")

    if isinstance(payload, dict) and payload.get("type") == "error":
        raise RuntimeError(f"{label} KiWIS error: {payload.get('message')}")

    return payload


def _load_series(name, fetch):
    """Station metadata for one deployment, cached in-process and on disk."""
    entries = _series_cache.get(name)
    if entries is not None:
        return entries

    with _series_lock:
        entries = _series_cache.get(name)
        if entries is not None:
            return entries

        filename = os.path.join(naaulu.config.get_cache_dir(name), "series.json")
        entries = None
        if os.path.exists(filename):
            try:
                with open(filename, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                if time.time() - cached["generated"] < KIWIS_MAX_AGE:
                    entries = cached["series"]
            except (OSError, ValueError, KeyError, TypeError):
                logger.debug(f"ignoring unreadable {name} series cache {filename}", exc_info=True)

        if entries is None:
            entries = fetch()
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump({"generated": time.time(), "series": entries}, f)
            except OSError:
                logger.debug(f"cannot write {name} series cache {filename}", exc_info=True)

        _series_cache[name] = entries

    return entries


def _kiwis_parse_stamp(value):
    """Turn a KiWIS timestamp into naive UTC.

    The offset is honoured so DST-repeated hours stay distinct; offset-less
    stamps are read as Brussels time.
    """
    stamp = datetime.datetime.fromisoformat(value)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=KIWIS_TZ)
    return stamp.astimezone(datetime.timezone.utc).replace(tzinfo=None)


def _kiwis_hourly(url, label, series, time, duration):
    """The hour ending at `time` for every series, as (values, codes, coords).

    These networks stamp the hour's opening instant, so the row asked for
    is the one stamped `time - duration`. Stations missing that hour are
    dropped, not zero-filled.
    """
    end = naaulu.util.naive_utc(time)
    opening = end - duration

    payload = _kiwis_get(url, {
        "request": "getTimeseriesValues",
        "ts_id": ",".join(entry["ts_id"] for entry in series),
        "from": opening.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "to": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "returnfields": "Timestamp,Value",
    }, "json", label)

    by_ts_id = {entry["ts_id"]: entry for entry in series}

    values = []
    codes = []
    longitudes = []
    latitudes = []

    for series_block in payload:
        entry = by_ts_id.get(series_block.get("ts_id"))
        if entry is None:
            continue
        for stamp, value in series_block.get("data") or []:
            if value is None:
                continue
            if _kiwis_parse_stamp(stamp) != opening:
                continue
            values.append(float(value))
            codes.append(entry["code"])
            longitudes.append(entry["lon"])
            latitudes.append(entry["lat"])
            break

    if not values:
        raise naaulu.errors.NoDataError(
            f"No {label} data at {naaulu.util.format_time(time, show=True)}"
        )

    coords = numpy.column_stack((longitudes, latitudes))
    return values, codes, coords


SPW_KIWIS_URL = "https://hydrometrie.wallonie.be/services/KiWIS/KiWIS"
_SPW_STATION_PREFIX = "SPW-"


def _spw_get(params, fmt):
    """SPW-specific anonymous KiWIS call."""
    return _kiwis_get(SPW_KIWIS_URL, params, fmt, "SPW")


def _spw_fetch_series():
    """Metadata for every SPW rain gauge with an hourly series."""
    series = _spw_get(
        {
            "request": "getTimeseriesList",
            "stationparameter_no": "Precip",
            "ts_shortname": "h.Total",
        },
        "objson",
    )

    stations = _spw_get(
        {"request": "getStationList", "stationparameter_no": "Precip"},
        "objson",
    )

    # one station appears once per parameter type: keep the first
    coords = {}
    for station in stations:
        lon = station.get("station_longitude")
        lat = station.get("station_latitude")
        if lon and lat and station["station_id"] not in coords:
            coords[station["station_id"]] = (float(lon), float(lat))

    entries = []
    for ts in series:
        position = coords.get(ts["station_id"])
        if position is None:
            continue
        entries.append(
            {
                "ts_id": ts["ts_id"],
                # prefix: plain numbers would collide with the RMI codes
                "code": _SPW_STATION_PREFIX + ts["station_no"],
                "lon": position[0],
                "lat": position[1],
            }
        )

    if not entries:
        raise RuntimeError("no SPW hourly precipitation series found")

    return entries


def _spw_load_series():
    """SPW station metadata, cached in-process and on disk."""
    return _load_series("spw", _spw_fetch_series)


def _bel_spw(time, duration):
    """SPW rain gauges (Wallonia); hourly series only."""
    if duration != datetime.timedelta(hours=1):
        raise ValueError(f"Unsupported duration for SPW AWS: {duration}")

    return _kiwis_hourly(SPW_KIWIS_URL, "SPW", _spw_load_series(), time, duration)


# ─── Belgium, Flanders (VMM via waterinfo.be) ───────────────────────────────

VMM_KIWIS_URL = "https://download.waterinfo.be/tsmdownload/KiWIS/KiWIS"
VMM_HOURLY_GROUP = "01192897"    # WEBLayer_Download_Neerslag_uur, ts P.60
_VMM_STATION_PREFIX = "VMM-"


def _vmm_get(params, fmt):
    """VMM-specific anonymous KiWIS call."""
    return _kiwis_get(VMM_KIWIS_URL, params, fmt, "VMM")


def _vmm_fetch_series():
    """Metadata for every Flemish rain gauge with an hourly series."""
    series = _vmm_get(
        {"request": "getTimeseriesList", "timeseriesgroup_id": VMM_HOURLY_GROUP},
        "objson",
    )

    layer = _vmm_get(
        {"request": "getTimeseriesValueLayer", "timeseriesgroup_id": VMM_HOURLY_GROUP},
        "objson",
    )

    coords = {
        row["ts_id"]: (float(row["station_longitude"]), float(row["station_latitude"]))
        for row in layer
        if row.get("station_longitude") and row.get("station_latitude")
    }

    entries = []
    for ts in series:
        position = coords.get(ts["ts_id"])
        if position is None:
            continue
        entries.append(
            {
                "ts_id": ts["ts_id"],
                "code": _VMM_STATION_PREFIX + ts["station_no"],
                "lon": position[0],
                "lat": position[1],
            }
        )

    if not entries:
        raise RuntimeError("no VMM hourly precipitation series found")

    return entries


def _vmm_load_series():
    """VMM station metadata, cached in-process and on disk."""
    return _load_series("vmm", _vmm_fetch_series)


def _bel_vmm(time, duration):
    """VMM rain gauges (Flanders); hourly series only."""
    if duration != datetime.timedelta(hours=1):
        raise ValueError(f"Unsupported duration for VMM AWS: {duration}")

    return _kiwis_hourly(VMM_KIWIS_URL, "VMM", _vmm_load_series(), time, duration)


# ─── Finland ─────────────────────────────────────────────────────────────────

def fin(time, duration):
    """Finnish AWS gauge observations."""
    if duration == datetime.timedelta(hours=1):
        param = "r_1h"
        result_key = "Precipitation amount"
        to_mm = 1.0
    elif duration == datetime.timedelta(minutes=10):
        param = "ri_10min"
        result_key = "Precipitation intensity"
        to_mm = 10.0 / 60.0
    else:
        raise ValueError(f"Unsupported duration for FMI AWS: {duration}")

    import fmiopendata.wfs

    start_time = time.isoformat(timespec="seconds") + "Z"
    end_time = time.isoformat(timespec="seconds") + "Z"
    args = [
        "bbox=18,55,35,75",
        "starttime=" + start_time,
        "endtime=" + end_time,
        f"parameters={param}",
    ]

    obs = fmiopendata.wfs.download_stored_query(
        "fmi::observations::weather::multipointcoverage",
        args=args,
    )

    meta = obs.location_metadata

    values = []
    codes = []
    longitudes = []
    latitudes = []

    for station, data in obs.data[time].items():
        codes.append(meta[station]["fmisid"])
        lon = meta[station]["longitude"]
        lat = meta[station]["latitude"]
        longitudes.append(lon)
        latitudes.append(lat)
        values.append(data[result_key]["value"] * to_mm)

    coords = numpy.column_stack((longitudes, latitudes))

    return (values, codes, coords)


# ─── Czech Republic ─────────────────────────────────────────────────────────

CZE_GEOPACKAGE_URL = "https://geoportal.gov.cz/atom/CHMU/stanice_CHMU_2024_epsg4258.gpkg"
CZE_PRECIP_ELEMENT = "SRA10M"


def cze_stations():
    """Download CHMI GeoPackage and extract AWS station coordinates."""
    logger.info("fetching CHMI station GeoPackage")
    resp = naaulu.network.runtime_session().get(CZE_GEOPACKAGE_URL, timeout=120)
    resp.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False) as f:
        f.write(resp.content)
        gpkg_path = f.name

    try:
        conn = sqlite3.connect(gpkg_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT Wsi, Geogr1, Geogr2 "
            "FROM stanice_CHMU_2024_EPSG_4258 "
            "WHERE Wsi LIKE '0-20000-0-%'"
        )
        stations = {row[0]: (row[1], row[2]) for row in cursor.fetchall()}
        conn.close()
    finally:
        os.unlink(gpkg_path)

    logger.info(f"found {len(stations)} CHMI AWS stations")
    return stations


def build_chmi_aws(data_dir=None):
    """Build chmi_aws_stations.json from CHMI GeoPackage."""
    if data_dir is None:
        data_dir = naaulu.config.get_data_dir()
    os.makedirs(data_dir, exist_ok=True)

    stations = cze_stations()
    out = os.path.join(data_dir, "chmi_aws_stations.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(stations, f, indent=2)
    logger.info(f"wrote {out} ({len(stations)} entries)")
    return stations


def _cze_load_stations():
    filename = naaulu.config.get_bundled_data_path("chmi_aws_stations.json")
    with open(filename, "r", encoding="utf-8") as f:
        return {k: tuple(v) for k, v in json.load(f).items()}


def cze(time, duration):
    """Czech CHMI AWS gauge observations."""
    stations = _cze_load_stations()

    values = []
    codes = []
    longitudes = []
    latitudes = []

    for wsi, (lon, lat) in stations.items():
        date_str = time.strftime("%Y%m%d")
        url = f"https://opendata.chmi.cz/meteorology/climate/now/data/10m-{wsi}-{date_str}.json"
        try:
            resp = naaulu.network.runtime_session().get(url, timeout=30)
            if resp.status_code != 200:
                continue
            data = resp.json()
        except Exception:
            logger.debug(f"failed to fetch CHMI data for {wsi}", exc_info=True)
            continue

        time_start = time - duration
        for row in data.get("data", {}).get("data", {}).get("values", []):
            if len(row) < 4:
                continue
            if row[1] != CZE_PRECIP_ELEMENT:
                continue

            try:
                dt_str = row[2].rstrip("Z")
                dt = datetime.datetime.fromisoformat(dt_str)
                if dt < time_start or dt >= time:
                    continue
            except (ValueError, IndexError):
                continue

            value = row[3]
            if value is None or value == "null":
                continue

            try:
                precip = float(value)
            except (ValueError, TypeError):
                continue

            values.append(precip)
            codes.append(wsi)
            longitudes.append(lon)
            latitudes.append(lat)
            break

    if not values:
        raise naaulu.errors.NoDataError(f"No CHMI AWS data at {time}")

    coords = numpy.column_stack((longitudes, latitudes))
    return values, codes, coords


# ─── Estonia ─────────────────────────────────────────────────────────────────

_EST_BASE_URL = "https://avaandmed.keskkonnaportaal.ee/api/lists/active"
_EST_RECENT_THRESHOLD = datetime.timedelta(days=4)


def _est_fetch_xml():
    url = "https://www.ilmateenistus.ee/ilma_andmed/xml/observations.php"
    response = naaulu.network.runtime_session().get(url, timeout=30)
    response.raise_for_status()

    root = xml.etree.ElementTree.fromstring(response.text)

    values = []
    codes = []
    longitudes = []
    latitudes = []

    for station in root.findall("station"):
        wmo = station.findtext("wmocode", "").strip()
        precip = station.findtext("precipitations", "").strip()
        lon = station.findtext("longitude", "").strip()
        lat = station.findtext("latitude", "").strip()

        if not wmo or not precip or not lon or not lat:
            continue

        try:
            precip_val = float(precip)
        except ValueError:
            continue

        try:
            lon_val = float(lon)
            lat_val = float(lat)
        except ValueError:
            continue

        codes.append(wmo)
        longitudes.append(lon_val)
        latitudes.append(lat_val)
        values.append(precip_val)

    if not codes:
        raise naaulu.errors.NoDataError(
            "No valid precipitation data from Estonian AWS stations"
        )

    coords = numpy.column_stack((longitudes, latitudes))
    return (values, codes, coords)


def _est_gauge(time, duration):
    if duration != datetime.timedelta(hours=1):
        raise ValueError("Only 1-hour duration supported for Estonian AWS")
    return _est_fetch_xml()


def est(time, duration):
    """Estonian gauges: national feed for recent data (<4 days), GHCN for older."""
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    if time.tzinfo is None:
        time_utc = time.replace(tzinfo=datetime.timezone.utc)
    else:
        time_utc = time

    if (now - time_utc) < _EST_RECENT_THRESHOLD:
        return _est_gauge(time, duration)
    else:
        return ghcn(time, duration, "EE")


# ─── Netherlands (KNMI Data Platform) ───────────────────────────────────────

KNMI_OPEN_DATA = "https://api.dataplatform.knmi.nl/open-data/v1"
KNMI_10MIN_DATASET = "10-minute-in-situ-meteorological-observations"
KNMI_10MIN_VERSION = "1.0"

# Shared anonymous key published on the KNMI Developer Portal
# (developer.dataplatform.knmi.nl/open-data-api); it rotates roughly yearly.
# A personal key via `naaulu set KNMI_API_KEY` gets dedicated quota instead.
KNMI_ANONYMOUS_KEY = (
    "eyJvcmciOiI1ZTU1NGUxOTI3NGE5NjAwMDEyYTNlYjEiLCJpZCI6IjUzYTg1ZDBhMmQ5"
    "YzRkYzJiYWNlNzQ4NTQ2Zjk4ODExIiwiaCI6Im11cm11cjEyOCJ9"
)


def _knmi_file(time):
    """The 10-minute file whose interval ends at `time`."""
    stamp = naaulu.util.naive_utc(time).strftime("%Y%m%d%H%M")
    return f"KMDS__OPER_P___10M_OBS_L2_{stamp}.nc"


def _knmi_read(filepath):
    """(values, codes, coords) of the hourly rainfall sum in one KNMI file."""
    with xarray.open_dataset(filepath, engine="h5netcdf") as dataset:
        rainfall = dataset["R1H"].values[:, 0]
        stations = dataset["station"].values
        longitudes = dataset["lon"].values
        latitudes = dataset["lat"].values

    # only the precipitation sites report R1H; the wind poles stay NaN
    valid = ~numpy.isnan(rainfall)

    values = [float(v) for v in rainfall[valid]]
    codes = [str(code) for code in stations[valid]]
    coords = numpy.column_stack((longitudes[valid], latitudes[valid]))

    return values, codes, coords


def nld(time, duration):
    """KNMI gauges from the near real-time 10-minute observations.

    Serves the 1-hour accumulation natively through R1H, the rolling hourly
    rainfall sum stamped at the interval end; longer durations are summed from
    consecutive hours by get_network. The file is named after its interval end,
    so it is fetched directly: about 3m40s after the hour KNMI publishes it,
    and it is rewritten some 10 minutes later with the stations that report
    late. Stations without a rain gauge are dropped.
    """
    if duration != datetime.timedelta(hours=1):
        raise ValueError(f"Only 1-hour duration supported for KNMI, got {duration}")

    api_key = naaulu.config.get_secret("KNMI_API_KEY") or KNMI_ANONYMOUS_KEY
    filename = _knmi_file(time)
    url = f"{KNMI_OPEN_DATA}/datasets/{KNMI_10MIN_DATASET}/versions/{KNMI_10MIN_VERSION}/files/{filename}/url"

    try:
        filepath = naaulu.network.get_temp(url, api_key, filename, "temporaryDownloadUrl")
    except FileNotFoundError:
        # the file for that instant was never published
        raise naaulu.errors.NoDataError(
            f"KNMI has no {filename}: not published or too old"
        )

    return _knmi_read(filepath)


# ─── Luxembourg (geoportail INSPIRE WFS) ────────────────────────────────────

# `lux` is deliberately left free for MeteoLux, the national weather service:
# it runs a single station (Findel, WMO 06590) and only publishes last-minute
# snapshots through metapi.ana.lu, with no time-series API to ask for a past
# hour. The networks below are ASTA (agriculture) and AGE (water agency).

LU_WFS_URL = "https://wms.inspire.geoportail.lu/geoserver/mf/wfs"
LU_WFS_LIVE_URL = "https://wms.inspire.geoportail.lu/geoserver/ml/wfs"
LU_VALUE_FIELD = "sum_nn050"    # precipitation incl. snow and hail, mm

# Hour-ending stamps are labelled in fixed GMT+1 (no DST) and printed as if
# they were UTC: `datetime 13:00:00Z` carries `time 13:00:00+01:00`. Confirmed
# against the published totals, which are the sum of the hourly stamps from
# 01:00 to the next day 00:00, i.e. the stamps end the hours they belong to.
# So the hour ending at UTC `time` is stamped `time + 1h`.
LU_STAMP_OFFSET = datetime.timedelta(hours=1)

# archive first (measured about half a day to a day behind), then the
# near-real-time layer, which holds only the freshest hour and covers ASTA
# alone: AGE has no live layer
LU_ASTA_LAYERS = [
    (LU_WFS_URL, "MF.PointTimeSeriesObservation_Hourly_ASTA_sum_nn050"),
    (LU_WFS_LIVE_URL, "MF.PointTimeSeriesObservation_Live_ASTA_sum_nn050"),
]
LU_AGE_LAYERS = [
    (LU_WFS_URL, "MF.PointTimeSeriesObservation_Hourly_AGE_sum_nn050"),
]


def _lu_stamp(time):
    """The provider label of the hour ending at naive-UTC `time`."""
    stamp = naaulu.util.naive_utc(time) + LU_STAMP_OFFSET
    return stamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def _lu_features(url, typename, stamp):
    """One WFS GetFeature call for a single hour-ending stamp."""
    params = {
        "service": "wfs",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeName": typename,
        "srsName": "EPSG:4326",
        "outputFormat": "application/json",
        "CQL_FILTER": f"datetime = {stamp}",
    }

    response = naaulu.network.runtime_session().get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    return data.get("features") or []


def _lux_gauge(layers, prefix, time, duration):
    """The hour ending at `time` for one Luxembourg network, as station data.

    Stations missing the hour are dropped, not zero-filled, and neither are
    null readings. The first layer holding the stamp wins: the archive is
    asked first, the live layer only covers the most recent hour.
    """
    if duration != datetime.timedelta(hours=1):
        raise ValueError(
            f"Only 1-hour duration supported for Luxembourg gauges, got {duration}"
        )

    stamp = _lu_stamp(time)

    features = []
    for url, typename in layers:
        features = _lu_features(url, typename, stamp)
        if features:
            break

    values = []
    codes = []
    longitudes = []
    latitudes = []

    for feature in features:
        properties = feature.get("properties") or {}
        value = properties.get(LU_VALUE_FIELD)
        if value is None:
            continue

        # AGM_043 / AGE_054: unique per station, unlike the place name
        href = properties.get("featureofinterest_xlink_href") or ""
        station = href.rsplit("/", 1)[-1] or properties.get("name_descr")
        if not station:
            continue

        geometry = feature.get("geometry") or {}
        coordinates = geometry.get("coordinates")
        if not coordinates or len(coordinates) < 2:
            continue

        values.append(float(value))
        # prefix: plain numbers would collide with the other networks
        codes.append(f"{prefix}-{station}")
        longitudes.append(float(coordinates[0]))
        latitudes.append(float(coordinates[1]))

    if not values:
        raise naaulu.errors.NoDataError(
            "No Luxembourg "
            f"{prefix} data at {naaulu.util.format_time(naaulu.util.naive_utc(time), show=True)}"
        )

    coords = numpy.column_stack((longitudes, latitudes))
    return values, codes, coords


def lux_asta(time, duration):
    """ASTA rain gauges (agriculture) from the geoportail INSPIRE WFS.

    Serves the 1-hour accumulation natively; longer durations are summed from
    consecutive hours by get_network. The archive layer is about half a day
    to a day behind, so recent hours come from the near-real-time layer,
    which carries the freshest hour only and is published just after the hour.
    """
    return _lux_gauge(LU_ASTA_LAYERS, "ASTA", time, duration)


def lux_age(time, duration):
    """AGE rain gauges (water agency) from the geoportail INSPIRE WFS.

    Hourly series only, about half a day to a day behind: AGE has no
    near-real-time layer, so recent hours raise NoDataError.
    """
    return _lux_gauge(LU_AGE_LAYERS, "AGE", time, duration)


# ─── Dispatch ────────────────────────────────────────────────────────────────

# Belgium publishes through three disjoint networks: RMI nationwide,
# SPW Wallonia, VMM Flanders.
# Luxembourg likewise publishes through two: ASTA and AGE; `lux` stays
# reserved for MeteoLux, the national weather service.
# Each country key also has an IEM ASOS twin under `<iso>_asos`
# (bel_asos, deu_asos, ...); the national sub-keys do not need one.
_GAUGE = {
    "bel": _bel_rmib,
    "bel_spw": _bel_spw,
    "bel_vmm": _bel_vmm,
    "bel_asos": bel_asos,
    "deu_asos": deu_asos,
    "fin": fin,
    "fin_asos": fin_asos,
    "cze": cze,
    "cze_asos": cze_asos,
    "est": est,
    "est_asos": est_asos,
    "ghcn": ghcn,
    "asos": asos,
    "fra_asos": fra_asos,
    "nld": nld,
    "nld_asos": nld_asos,
    "lux_asta": lux_asta,
    "lux_age": lux_age,
    "lux_asos": lux_asos,
}


def get_gauge(time, duration, country):
    country = country.upper()
    func = _GAUGE.get(country.lower())
    if func is None:
        raise ValueError(f"No gauge provider for country: {country}")
    return func(time, duration)
