import csv
import datetime
import hashlib
import json
import logging
import os
import math
import re
import tempfile
import threading
import time
import warnings
import xml.etree.ElementTree
import zipfile

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


def _pvol_candidates(timestrs, start_str, end_str, max_before=5):
    """PVOL volumes for a window, nearest to its start first.

    Nodes are not all on the :00/:05 grid - CZ publishes at HH:04/HH:09/... -
    so an exact match on the window start would skip them entirely. Volumes
    inside the window come first (earliest, i.e. closest to the start), then
    the most recent ones before it.
    """
    inside = sorted(t for t in timestrs if start_str <= t <= end_str)
    before = sorted((t for t in timestrs if t < start_str), reverse=True)
    return inside + before[:max_before]


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
                matching = _pvol_candidates(timestrs, start_str, end_str)
            else:
                matching = [t for t in timestrs if start_str <= t < end_str]
            for timestr in matching:
                prefix = f"{datedir}/{country}/{node}/{subdir}/{node}@{timestr}"
                try:
                    keys = _s3_list(prefix)
                except Exception:
                    continue
                if not keys:
                    continue

                if subdir == "PVOL":
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
                        # one PVOL volume per window is enough: try the next
                        # candidate only when this one failed above
                        break
                    except Exception:
                        logger.debug(f"failed to download/open {dbzh_key}", exc_info=True)
                else:
                    elev_files = {}
                    for k in keys:
                        fname = k.split("/")[-1]
                        parts = fname.split("@")
                        if len(parts) >= 3:
                            elev = parts[2]
                            if elev not in elev_files or "@DBZH" in k:
                                elev_files[elev] = k

                    for elev, key in sorted(elev_files.items()):
                        url = f"{S3_ENDPOINT}/{S3_BUCKET}/{key}"
                        try:
                            filepath = naaulu.network.download(url)
                            if not _validate_hdf5(filepath):
                                logger.warning(f"corrupted radar file, removing cache: {filepath}")
                                os.remove(filepath)
                                continue
                            volume = xradar.io.open_odim_datatree(filepath)
                            volumes.append(volume)
                        except Exception:
                            logger.debug(f"failed to download/open {key}", exc_info=True)

        if volumes:
            break

        day -= datetime.timedelta(days=1)

    if not volumes:
        raise RuntimeError(
            f"No OPERA radar data for WIGOS {wsi} between "
            f"{start_str} and {end_str}"
        )

    return volumes


# ─── France (opera, minus the gauge correction baked into the data) ──────────

def fra(time, duration, wsi):
    """French volumes, with the upstream gauge correction suppressed.

    Météo-France delivers reflectivity that has already been corrected
    against its gauge network, unlike the other OPERA nodes which ship raw
    reflectivity. The correction is baked into the pixels and carries no
    flag in the file's metadata (see the cached frave files: nothing about
    a bias there), so it cannot be undone from the attributes.

    A rainrate bias is a constant offset in dBZ, so the correction is a
    flat offset rather than a division: dbzh / 1.2 would remove more the
    stronger the echo (5 dB at 30 dBZ, 8 dB at 50 dBZ) and warp the echo
    structure. For scale, the ~1.4 rainrate bias seen against gauges would
    be 10 * b * log10(1.4) = 2.3 dB of reflectivity with Marshall-Palmer
    (b = 1.6); the correction applied here is -1 dB.
    """
    volumes = _opera(time, duration, wsi)
    if not isinstance(volumes, list):
        volumes = [volumes]

    for volume in volumes:
        for key in xradar.util.get_sweep_keys(volume):
            sweep = volume[key].ds
            if sweep is None or "DBZH" not in sweep:
                continue
            # .ds is a read-only view; a shallow copy shares the arrays
            sweep = sweep.copy(deep=False)
            sweep["DBZH"] = sweep["DBZH"] - 1.0    # suppress gauge correction
            volume[key].ds = sweep

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
    # filenames carry naive UTC timestamps; normalize so tz-aware callers
    # (parse_time returns UTC-aware datetimes) don't break the comparison
    time = naaulu.util.naive_utc(time)
    time_start = time - duration
    time_end = time

    icao = code.upper()
    bucket = "unidata-nexrad-level2"
    files = []
    for shift in [-1, 0, 1]:
        day = time_start + datetime.timedelta(days=shift)
        date = day.strftime("%Y/%m/%d")
        prefix = f"{date}/{icao}/"
        try:
            keys = naaulu.network.list_s3_objects(bucket, prefix=prefix)
        except Exception as exc:
            logger.debug(f"cannot list {bucket}/{prefix}: {exc}")
            continue
        # same selection as the former glob f"{prefix}{icao}*_V0?"
        files.extend(
            key for key in keys
            if re.fullmatch(rf"{icao}.*_V0.", os.path.basename(key))
        )

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
        # boto3 keys are bucket-less, download_s3_file wants bucket/key
        local_path = naaulu.network.download_s3_file(s3_url=f"{bucket}/{key}")
        try:
            dtree = xradar.io.open_nexradlevel2_datatree(local_path)
        except Exception as e:
            # one unparsable volume must not cost the whole radar (mixed
            # 360/720 ray counts make some volumes fail to parse)
            logger.warning(f"skipping unreadable radar volume {key}: {type(e).__name__}: {e}")
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


# ─── Australia (AURA / NCI THREDDS) ─────────────────────────────────────────

_AUS_BASE = "https://dapds00.nci.org.au/thredds/fileServer/rq0"
_AUS_SITE_LIST = f"{_AUS_BASE}/radar_site_list.csv"
# Only used when matching by coordinates alone, so that nearby radars are not
# confused with each other. Sites whose WMO and NCI coordinates differ by more
# than this (up to 0.4 degrees, e.g. Mildura) are matched by WIGOS id instead.
_AUS_TOLERANCE = 0.1
# How far before the window the fallback volume may still be useful. Volumes
# come every 5 minutes nowadays (6 minutes in the older archive), so this
# covers a couple of missed cycles without dragging in stale data: a volume
# entirely before the window gets clamped onto its start.
_AUS_MAX_LAG = datetime.timedelta(minutes=15)

_aus_sites = None
_aus_mapping = None
_aus_members_cache = {}
_aus_sizes = {}
_aus_warned = set()
_aus_lock = threading.Lock()
# One entry holds ~289 members, so a year-long backfill across several radars
# would otherwise pile up megabytes of listings.
_AUS_MEMBER_CACHE_MAX = 256
# NCI resets connections when a handful of clients open too many at once
# (observed: 16 of 36 requests reset at 24 threads), so ranged access runs
# behind a small semaphore no matter how many estimator workers there are.
_AUS_SEMAPHORE = threading.Semaphore(4)
_AUS_RETRIES = 3
_AUS_BACKOFF = (1.0, 2.0)


def _aus_warn_once(key, message):
    """Log message at WARNING on first sight, at DEBUG afterwards.

    The estimate loop fetches each radar twice per event, so without this
    every provider warning shows up doubled.
    """
    with _aus_lock:
        first = key not in _aus_warned
        _aus_warned.add(key)
    logger.log(logging.WARNING if first else logging.DEBUG, message)


class _RangeReader:
    """Minimal seekable file-like backed by HTTP Range requests.

    Lets the stdlib zipfile module pull a single volume out of the multi-GB
    daily zip archives in the NCI AURA collection without downloading them.
    """

    def __init__(self, url, chunk=1 << 20, session=None, retries=_AUS_RETRIES):
        self.url = url
        self.session = session or naaulu.network.runtime_session()
        self.chunk = chunk
        self.retries = max(1, retries)
        self.size = self._content_length()
        self.pos = 0
        self._buffer = b""
        self._buffer_start = 0

    @staticmethod
    def _delay(response, attempt):
        """Backoff before a retry, honouring Retry-After when the server sent one."""
        if response is not None:
            value = response.headers.get("retry-after")
            if value:
                try:
                    return min(float(value), 60.0)
                except ValueError:
                    pass
        return _AUS_BACKOFF[min(attempt - 1, len(_AUS_BACKOFF) - 1)]

    def _request(self, method, headers=None, expect=None):
        """Perform one request, retrying transient failures.

        expect: (start, end) of a requested byte range. Such a response is
        only accepted when status, Content-Range and body length all match, so
        a truncated or misrouted body can never be mistaken for archive data.
        """
        last = None
        response = None
        for attempt in range(self.retries):
            if attempt:
                delay = self._delay(response, attempt)
                logger.debug(
                    f"retry {attempt}/{self.retries - 1} {method} {self.url} "
                    f"in {delay:.1f}s after {last}"
                )
                time.sleep(delay)
            response = None
            try:
                with _AUS_SEMAPHORE:
                    if method == "HEAD":
                        response = self.session.head(
                            self.url, headers=headers, timeout=60
                        )
                    else:
                        response = self.session.get(
                            self.url, headers=headers, timeout=300
                        )
            except (
                requests.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
            ) as exc:
                last = exc
                continue

            if response.status_code == 404:
                raise FileNotFoundError(f"radar archive not found: {self.url}")
            if response.status_code in (429, 500, 502, 503, 504):
                last = requests.HTTPError(
                    f"HTTP {response.status_code} for {self.url}"
                )
                continue
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                last = exc
                continue

            if expect is not None:
                start, end = expect
                wanted = f"bytes {start}-{end}/{self.size}"
                content_range = response.headers.get("content-range", "")
                body = response.content
                if (
                    response.status_code != 206
                    or content_range != wanted
                    or len(body) != end - start + 1
                ):
                    last = RuntimeError(
                        f"bad range response from {self.url}: "
                        f"{response.status_code} {content_range!r} "
                        f"{len(body)} bytes, wanted {wanted!r}"
                    )
                    continue
            return response

        raise last

    def _content_length(self):
        with _aus_lock:
            cached = _aus_sizes.get(self.url)
        if cached is not None:
            return cached

        response = self._request("HEAD")
        size = int(response.headers.get("content-length") or 0)
        if size <= 0:
            # no usable Content-Length: ask the server where the body ends
            response = self._request("GET", headers={"Range": "bytes=0-0"})
            content_range = response.headers.get("content-range", "")
            if "/" not in content_range:
                raise RuntimeError(
                    f"server does not support range requests: {self.url}"
                )
            size = int(content_range.rsplit("/", 1)[1])
        if size <= 0:
            raise RuntimeError(f"empty archive at {self.url}")

        with _aus_lock:
            _aus_sizes[self.url] = size
        return size

    def _fill(self):
        start = self.pos
        end = min(start + self.chunk, self.size) - 1
        response = self._request(
            "GET", headers={"Range": f"bytes={start}-{end}"}, expect=(start, end)
        )
        self._buffer = response.content
        self._buffer_start = start

    def read(self, size=-1):
        if size is None or size < 0:
            size = self.size - self.pos
        size = min(size, self.size - self.pos)
        if size <= 0:
            return b""
        parts = []
        remaining = size
        while remaining > 0:
            if not self._buffer_start <= self.pos < self._buffer_start + len(self._buffer):
                self._fill()
            offset = self.pos - self._buffer_start
            take = min(remaining, len(self._buffer) - offset)
            if take <= 0:
                raise RuntimeError(f"empty range response for {self.url}")
            parts.append(self._buffer[offset:offset + take])
            self.pos += take
            remaining -= take
        return b"".join(parts)

    def seek(self, offset, whence=0):
        if whence == 0:
            position = offset
        elif whence == 1:
            position = self.pos + offset
        elif whence == 2:
            position = self.size + offset
        else:
            raise ValueError(f"invalid whence ({whence})")
        self.pos = max(position, 0)
        return self.pos

    def tell(self):
        return self.pos

    def readable(self):
        return True

    def seekable(self):
        return True

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _aus_open(url):
    """File-like for a daily volume zip: local file if it exists, else ranged HTTP."""
    if os.path.exists(url):
        return open(url, "rb")
    return _RangeReader(url)


_MEMBER_RE = re.compile(r"_(\d{8})_(\d{6})\.pvol\.h5$")


def _aus_parse_members(url):
    """Sorted [(name, naive datetime)] for the daily volume zip at url.

    All timestamps in the AURA archive are UTC (see rq0_level1_readme.pdf).
    """
    with _aus_open(url) as source:
        with zipfile.ZipFile(source) as archive:
            names = archive.namelist()

    members = []
    for name in names:
        match = _MEMBER_RE.search(name)
        if match is None:
            continue
        stamp = datetime.datetime.strptime(
            match.group(1) + match.group(2), "%Y%m%d%H%M%S"
        )
        members.append((name, stamp))

    members.sort(key=lambda member: member[1])
    return members


_AUS_ARCHIVE_RE = re.compile(r"/rq0/(\d+)/(\d{4})/vol/\1_(\d{8})\.pvol\.zip$")


def _aus_listing_path(url):
    """On-disk listing cache for a daily zip, or None when not cacheable.

    Keyed by site and day (an archive id can host different radars in
    different eras, and each day has its own zip). Only real archive URLs
    qualify; local paths (tests, mirrors) go straight to the parser.
    """
    match = _AUS_ARCHIVE_RE.search(url)
    if match is None:
        return None
    site, year, day = match.group(1), match.group(2), match.group(3)
    return os.path.join(
        naaulu.config.get_cache_dir("aus"), f"aus_{site}_{year}_{day}.json"
    )


def _aus_content_length(url):
    """Archive size in bytes (HEAD, cached per url for this run)."""
    return _RangeReader(url).size


def _aus_read_listing(url):
    """Members from the on-disk listing cache, if it still matches the archive."""
    path = _aus_listing_path(url)
    if path is None or not os.path.exists(path):
        return None

    try:
        with open(path, encoding="utf-8") as handle:
            stored = json.load(handle)
        # the archive can still be rewritten (a day zip is published the next
        # morning), so the size is checked against the server before reuse
        if stored.get("url") != url:
            logger.debug(f"listing cache {path} is for another archive, ignoring")
            return None
        if stored.get("size") != _aus_content_length(url):
            logger.debug(f"stale listing cache {path}, relisting {url}")
            return None
        members = [
            (name, datetime.datetime.fromisoformat(stamp))
            for name, stamp in stored["members"]
        ]
    except FileNotFoundError:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.debug(f"ignoring listing cache {path}: {exc}")
        return None

    return members


def _aus_write_listing(url, members):
    path = _aus_listing_path(url)
    if path is None:
        return
    try:
        size = _aus_content_length(url)
    except (FileNotFoundError, RuntimeError, requests.RequestException):
        return

    dirname = os.path.dirname(path)
    os.makedirs(dirname, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=dirname, suffix=".part")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(
                {
                    "url": url,
                    "size": size,
                    "members": [
                        [name, stamp.isoformat()] for name, stamp in members
                    ],
                },
                stream,
            )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _aus_list_members(url):
    """Members of the daily zip at url, listed once and then cached.

    Cached in-process for the run and on disk between runs, so repeated
    estimates do not re-read ~1 MB of central directory per site and day.
    """
    with _aus_lock:
        cached = _aus_members_cache.get(url)
    if cached is not None:
        return cached

    members = _aus_read_listing(url)
    if members is None:
        members = _aus_parse_members(url)
        _aus_write_listing(url, members)

    with _aus_lock:
        while len(_aus_members_cache) >= _AUS_MEMBER_CACHE_MAX:
            _aus_members_cache.pop(next(iter(_aus_members_cache)))
        _aus_members_cache[url] = members
    return members


def _aus_extract(url, name, destination):
    """Extract a single volume from the daily zip into destination (atomic)."""
    if os.path.exists(destination):
        return destination

    with _aus_open(url) as source:
        with zipfile.ZipFile(source) as archive:
            data = archive.read(name)

    dirname = os.path.dirname(destination)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=dirname, suffix=".part")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)

        if not _validate_hdf5(temporary):
            raise ValueError(f"corrupted radar volume {name} in {url}")

        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)

    logger.debug(f"extracted {name} from {url} ({len(data)} bytes)")
    return destination


def _aus_zip_url(site, day):
    return f"{_AUS_BASE}/{site}/{day.year}/vol/{site}_{day:%Y%m%d}.pvol.zip"


def _aus_select(members, time_start, time_end, max_lag=_AUS_MAX_LAG):
    """Volume stamps to use for [time_start, time_end).

    Those nominally inside the window (end exclusive: a volume starting
    exactly at time_end has all of its sweeps after the window), plus at most
    the closest volume before it. Scans are only ~4.5 minutes long and cannot
    always be placed inside an offset window, so that fallback keeps the
    lowest elevation available.
    """
    inside = [stamp for _, stamp in members if time_start <= stamp < time_end]
    before = [
        stamp for _, stamp in members
        if time_start - max_lag <= stamp < time_start
    ]
    selected = set(inside)
    if before:
        selected.add(max(before))
    return sorted(selected)


def _aus_clamp(volume, time_start, time_end):
    """Clamp sweep ray times to the accumulation window.

    A BOM volume scan starts ~20 s after its nominal filename time and ends
    within seconds of the next 5 minute boundary, with the lowest elevation
    ending last. create_volume prunes any sweep that reaches past the window,
    which would drop exactly the sweep Dove depends on, so out-of-window rays
    are clipped onto the window bounds instead.
    """
    start = numpy.datetime64(naaulu.util.naive_utc(time_start), "ns")
    end = numpy.datetime64(naaulu.util.naive_utc(time_end), "ns")

    for key in xradar.util.get_sweep_keys(volume):
        dataset = volume[key].ds
        if "time" not in dataset.coords:
            continue
        times = dataset["time"].values.astype("datetime64[ns]")
        clamped = numpy.clip(times, start, end)
        if numpy.array_equal(times, clamped):
            continue
        dataset = dataset.assign_coords(
            time=xarray.Variable(dataset["time"].dims, clamped)
        )
        volume[key].ds = dataset
        logger.debug(
            f"{key}: clamped {int((times != clamped).sum())} ray times into the window"
        )

    return volume


def _aus_wigos_key(wigos):
    """WIGOS id without its organisation segment.

    NCI and WMO disagree on that segment for the same radar, e.g.
    Charleville is 0-20000-0-94510 in the NCI list and 0-20010-0-94510 in
    the WMO database.
    """
    parts = str(wigos).split("-")
    if len(parts) != 4:
        return None
    return "-".join((parts[0], parts[2], parts[3]))


def _aus_parse_date(value):
    """Parse the site list's d/m/yyyy dates; '-' or empty means unbounded."""
    text = (value or "").strip()
    if not text or text == "-":
        return None
    for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _aus_site_list():
    """Rows of the NCI site list: id, coordinates, WIGOS, validity, location.

    One row per radar generation, and an archive id can be reused by a
    different radar in another era (id 38 is Charleville until 2006 and
    Newdegate from 2016), so all rows are kept.
    """
    global _aus_sites
    if _aus_sites is not None:
        return _aus_sites

    path = naaulu.network.download(_AUS_SITE_LIST, "radar_site_list.csv")
    sites = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                lat = float(row["site_lat"])
                lon = float(row["site_lon"])
            except (KeyError, TypeError, ValueError):
                continue
            sites.append(
                {
                    "id": row["id"],
                    "lat": lat,
                    "lon": lon,
                    "wigos": _aus_wigos_key(row.get("WIGOS")),
                    "start": _aus_parse_date(row.get("postchange_start")),
                    "end": _aus_parse_date(row.get("prechange_end")),
                    "location": (
                        (row.get("location") or row.get("short_name") or "").strip()
                    ),
                }
            )

    if not sites:
        raise ValueError(f"no sites parsed from {path}")

    _aus_sites = sites
    return sites


def _aus_distance(lat, lon, row):
    """Approximate distance in degrees between a point and a site row."""
    return math.hypot(
        row["lat"] - lat, (row["lon"] - lon) * math.cos(math.radians(lat))
    )


def _aus_nearest_site(lat, lon, sites=None):
    """(site_id, distance in degrees) of the closest NCI site, or None."""
    if sites is None:
        sites = _aus_site_list()

    best = None
    for row in sites:
        distance = _aus_distance(lat, lon, row)
        if best is None or distance < best[1]:
            best = (row["id"], distance)

    return best


def _get_aus_mapping():
    global _aus_mapping
    if _aus_mapping is None:
        filename = naaulu.config.get_bundled_data_path("radars_aus.json")
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                _aus_mapping = json.load(f)
        else:
            _aus_mapping = {}
    return _aus_mapping


def _aus_site_id(wsi):
    """NCI site id for a WIGOS id, or None when there is no plausible match."""
    mapping = _get_aus_mapping()
    if wsi in mapping:
        return mapping[wsi]

    radar = get_database().get(wsi) or {}
    try:
        lat, lon = float(radar["lat"]), float(radar["lon"])
    except (KeyError, TypeError, ValueError):
        _aus_warn_once(
            ("unmapped", wsi), f"cannot map {wsi} to an NCI site: no coordinates known"
        )
        return None

    match = _aus_nearest_site(lat, lon)
    if match is None:
        _aus_warn_once(
            ("unmapped", wsi),
            f"cannot map {wsi} to an NCI site: no site list available",
        )
        return None

    site_id, distance = match
    if distance > _AUS_TOLERANCE:
        _aus_warn_once(
            ("unmapped", wsi),
            f"no NCI archive site for {wsi} ({radar.get('name', '?')} at "
            f"{lat:.4f}, {lon:.4f}): nearest is {site_id} "
            f"{distance:.3f} degrees away, skipping",
        )
        return None

    logger.info(f"mapped {wsi} to NCI site {site_id} by coordinates ({distance:.4f} degrees)")
    return site_id


def _aus_use_clean_reflectivity(volume):
    """Expose DBZH_CLEAN as DBZH in every sweep of an opened volume.

    BOM encodes "no echo" in DBZH with the same raw code as nodata (0), which
    xradar masks to NaN, so undetect precipitation reaches the estimator as
    missing instead of 0 mm. DBZH_CLEAN is the reflectivity after the
    radar's clutter and non-meteorological echo filtering, and it uses ODIM's
    codes properly: rejected gates are nodata, clear air is undetect
    (-31.9 dBZ, i.e. about 0 rain). Volumes without the moment keep their raw
    DBZH.
    """
    renamed = 0
    for key in xradar.util.get_sweep_keys(volume):
        # .ds is a read-only view; a shallow copy shares the arrays
        dataset = volume[key].ds.copy(deep=False)
        if "DBZH_CLEAN" not in dataset:
            continue
        dataset["DBZH"] = dataset["DBZH_CLEAN"].rename("DBZH")
        volume[key].ds = dataset.drop_vars("DBZH_CLEAN")
        renamed += 1

    if renamed:
        logger.debug(f"using DBZH_CLEAN as DBZH in {renamed} sweeps")
    else:
        logger.debug("volume has no DBZH_CLEAN, keeping raw DBZH")

    return volume


def _aus_check_site(volume, wsi, max_distance=0.5):
    """True when the opened volume really is for wsi's radar.

    Archive ids are reused across eras (rq0/38 holds Charleville until 2006
    and Newdegate from 2016), so a mapping can point at another radar. The
    threshold stays loose because WMO and NCI coordinates for the same radar
    differ by up to 0.4 degrees.
    """
    radar = get_database().get(wsi) or {}
    try:
        lat, lon = float(radar["lat"]), float(radar["lon"])
    except (KeyError, TypeError, ValueError):
        return True

    try:
        site_lat = float(volume.ds["latitude"].values)
        site_lon = float(volume.ds["longitude"].values)
    except (KeyError, IndexError, TypeError, ValueError):
        return True

    distance = math.hypot(
        site_lat - lat, (site_lon - lon) * math.cos(math.radians(lat))
    )
    if distance > max_distance:
        _aus_warn_once(
            ("site", wsi),
            f"{wsi}: volume was recorded at {site_lat:.4f}, {site_lon:.4f}, "
            f"{distance:.3f} degrees from the expected radar, skipping",
        )
        return False

    return True


def _aus_valid_on(row, day):
    """Whether a site row covers the given date."""
    if row["start"] is not None and day < row["start"]:
        return False
    if row["end"] is not None and day > row["end"]:
        return False
    return True


def _aus_era_conflict(site, wsi, day):
    """Reason not to fetch this archive site for this date, or None.

    An archive id can be handed to a different radar in another era, so the
    rows occupying the id on that date are compared against the radar we
    asked for before any bytes are downloaded.
    """
    rows = [row for row in _aus_site_list() if row["id"] == site]
    active = [row for row in rows if _aus_valid_on(row, day)]
    if not active:
        # nobody recorded there on that date: let the archive answer (404)
        return None

    radar = get_database().get(wsi) or {}
    try:
        lat, lon = float(radar["lat"]), float(radar["lon"])
    except (KeyError, TypeError, ValueError):
        return None

    if any(_aus_distance(lat, lon, row) <= _AUS_TOLERANCE for row in active):
        return None

    occupants = ", ".join(
        f"{row['location'] or row['id']} (since {row['start'] or 'unknown'})"
        for row in active
    )
    ended = [row["end"] for row in rows if row["end"] is not None and row not in active]
    detail = f"; {radar.get('name', wsi)} data there ended {max(ended)}" if ended else ""
    return (
        f"archive site {site} holds {occupants} on {day.isoformat()} "
        f"instead of {radar.get('name', wsi)}{detail}, skipping"
    )


def aus(time, duration, wsi):
    """Volumes from the NCI AURA archive covering [time - duration, time].

    Returns a list of ODIM datatrees (read with xradar's ODIM reader), with
    the clutter-filtered reflectivity DBZH_CLEAN exposed as DBZH and ray
    times clamped to the requested window.
    """
    time = naaulu.util.naive_utc(time)
    time_start = time - duration
    time_end = time

    site = _aus_site_id(wsi)
    if site is None:
        raise RuntimeError(f"no NCI archive site for WIGOS {wsi}")

    conflict = _aus_era_conflict(site, wsi, time_start.date())
    if conflict is not None:
        _aus_warn_once(("era", wsi), conflict)
        raise RuntimeError(conflict)

    days = {time_start.date(), time_end.date()}
    days.add((time_start - _AUS_MAX_LAG).date())

    members = []
    for day in sorted(days):
        url = _aus_zip_url(site, day)
        try:
            members.extend(_aus_list_members(url))
        except FileNotFoundError:
            logger.debug(f"no AURA volume archive for site {site} on {day}")
        except (
            zipfile.BadZipFile,
            RuntimeError,
            ValueError,
            requests.RequestException,
        ) as exc:
            logger.warning(f"cannot list {url}: {exc}")

    if not members:
        raise RuntimeError(
            f"No AURA radar data for {wsi} around {time:%Y%m%d%H%M%S}"
        )
    members.sort(key=lambda member: member[1])

    stamps = _aus_select(members, time_start, time_end)
    if not stamps:
        raise RuntimeError(
            f"No AURA radar volume for {wsi} between "
            f"{time_start:%Y%m%d%H%M%S} and {time_end:%Y%m%d%H%M%S}"
        )

    by_stamp = {stamp: name for name, stamp in members}
    volumes = []
    for stamp in stamps:
        name = by_stamp[stamp]
        url = _aus_zip_url(site, stamp.date())
        destination = os.path.join(naaulu.config.get_download_dir(), name)
        try:
            _aus_extract(url, name, destination)
        except (
            FileNotFoundError,
            zipfile.BadZipFile,
            RuntimeError,
            ValueError,
            requests.RequestException,
        ) as exc:
            logger.warning(f"cannot extract {name}: {exc}")
            continue
        try:
            volume = xradar.io.open_odim_datatree(destination)
        except Exception as exc:
            logger.warning(f"cannot open {destination}: {exc}")
            continue
        _aus_use_clean_reflectivity(volume)
        if not _aus_check_site(volume, wsi):
            continue
        volumes.append(_aus_clamp(volume, time_start, time_end))

    if not volumes:
        raise RuntimeError(
            f"No usable AURA radar volume for {wsi} around {time:%Y%m%d%H%M%S}"
        )

    return volumes


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
    # GET: the endpoint is CodeIgniter CSRF-protected, a tokenless POST gets 403
    resp = requests.get(search_url, params=payload, timeout=120)
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


def build_aus_database(data_dir=None):
    """Build radars_aus.json: WIGOS ID to NCI/AURA site id mapping."""
    if data_dir is None:
        data_dir = naaulu.config.get_data_dir()
    os.makedirs(data_dir, exist_ok=True)

    sites = _aus_site_list()
    logger.info(f"parsed {len(sites)} NCI/AURA radar site rows")

    by_wigos = {}
    for row in sites:
        if row["wigos"]:
            by_wigos.setdefault(row["wigos"], []).append(row)

    radars_path = os.path.join(data_dir, "radars.json")
    with open(radars_path) as f:
        wmo = json.load(f)
    australian = {k: v for k, v in wmo.items() if v.get("country") == "AUS"}

    mapping = {}
    for wigos_id, meta in australian.items():
        try:
            lat, lon = float(meta["lat"]), float(meta["lon"])
        except (KeyError, TypeError, ValueError):
            continue

        site_id = None
        candidates = by_wigos.get(_aus_wigos_key(wigos_id), [])
        if len(candidates) == 1:
            site_id = candidates[0]["id"]
        elif len(candidates) > 1:
            # same WIGOS id for several archive ids: let coordinates decide
            best = min(candidates, key=lambda row: _aus_distance(lat, lon, row))
            if _aus_distance(lat, lon, best) <= _AUS_TOLERANCE:
                site_id = best["id"]

        if site_id is None:
            match = _aus_nearest_site(lat, lon, sites)
            if match is not None and match[1] <= _AUS_TOLERANCE:
                site_id = match[0]

        if site_id is None:
            nearest = _aus_nearest_site(lat, lon, sites)
            logger.info(
                f"{wigos_id} ({meta.get('name')}): no NCI site within "
                f"{_AUS_TOLERANCE} degrees (nearest {nearest}), leaving unmapped"
            )
            continue

        mapping[wigos_id] = site_id

    out = os.path.join(data_dir, "radars_aus.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, sort_keys=True)
    logger.info(f"wrote {out} ({len(mapping)}/{len(australian)} mapped)")
    return mapping


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

    return selection, radars.keys()


def select_batch(*, start, end, tiles, distance):

    radars = get_database()

    coverages = {}
    for wsi, info in radars.items():
        location = (info["lon"], info["lat"])
        coverages[wsi] = naaulu.geography.radar_coverage(
            location=location,
            distance=distance,
        )

    from shapely.strtree import STRtree
    coverage_list = list(coverages.values())
    wsi_list = list(coverages.keys())
    tree = STRtree(coverage_list)

    results = {}
    for tile in tiles:
        # query() filters on bounding boxes only, which lets through radars
        # whose coverage circle never reaches the tile (a circle's box covers
        # its inscribed square): without the predicate, a tile picks up radars
        # hundreds of km away, wasting a download and erroring on WSIs that
        # have no ICAO mapping.
        indices = tree.query(tile, predicate="intersects")
        selection = [wsi_list[i] for i in indices]

        filtered = {
            wsi: radars[wsi] for wsi in selection
            if radars[wsi]["start"] is None or datetime.date.fromisoformat(radars[wsi]["start"]) <= end.date()
        }
        filtered = {
            wsi: v for wsi, v in filtered.items()
            if v.get("end") is None or datetime.date.fromisoformat(v["end"]) >= start.date()
        }

        results[tile] = (selection, filtered.keys())

    return results


def write_select_cache(*, distance, key, selection):
    cache_dir = naaulu.config.get_cache_dir()
    filename = f"radar_select_{distance}.json"
    filename = os.path.join(cache_dir, filename)
    selections = {}
    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                selections = json.load(f)
        except (json.JSONDecodeError, ValueError):
            os.remove(filename)
    selections[key] = selection
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(selections, f)


# ─── Dispatch ────────────────────────────────────────────────────────────────

_RADAR = {
    "EST": est,
    "BEL": _opera,
    "FIN": fin,
    "DEU": _opera,
    "CZE": _opera,
    "FRA": fra,
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


def _sanitize_attrs(node):
    """Store boolean attributes as 0/1 before writing.

    NetCDF has no boolean attribute type, and NEXRAD volumes carry VCP flags
    (mpda_vcp, sails_cut, ...) as Python bools, which h5netcdf refuses to
    write unless invalid_netcdf=True.
    """
    for key, value in list(node.attrs.items()):
        if isinstance(value, (bool, numpy.bool_)):
            node.attrs[key] = int(value)
    for name in node.children:
        _sanitize_attrs(node[name])


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

    _sanitize_attrs(volume)
    # Write beside the target and rename: a failure mid-write (bad attrs, a
    # full disk, a crash) must not leave a half-written .nc behind, since the
    # archive reader treats an existing file as a valid cache and would then
    # fail on it for every later run instead of rebuilding it.
    handle, tmp = tempfile.mkstemp(
        dir=dirname or ".", prefix=os.path.basename(filename), suffix=".tmp"
    )
    os.close(handle)
    try:
        volume.to_netcdf(tmp, engine="h5netcdf")
        os.replace(tmp, filename)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def get_volume(*, time, duration, wsi, min_angle, max_angle):
    sweeps = get(time=time, duration=duration, wsi=wsi)
    if not isinstance(sweeps, list):
        sweeps = [sweeps]

    valid = []
    for vol in sweeps:
        # drop only the unusable sweep, not the whole volume: partial scans
        # (357/359/719 rays) do happen and losing the radar for one of them
        # is much worse than losing that sweep
        for name in list(xradar.util.get_sweep_keys(vol)):
            ds = vol[name].ds
            if ds is None or "azimuth" not in ds.sizes:
                continue
            az_count = ds.sizes["azimuth"]
            if az_count not in (360, 720):
                logger.warning(f"{wsi} {name}: azimuth count is {az_count}, expected 360 or 720, discarding sweep")
                del vol[name]
        if xradar.util.get_sweep_keys(vol):
            valid.append(vol)
        else:
            logger.warning(f"{wsi}: no usable sweep left in volume")

    if not valid:
        raise RuntimeError(f"No valid sweeps for {wsi}")

    return xradar.util.create_volume(
        sweeps=valid,
        time_coverage_start=naaulu.util.naive_utc(time - duration),
        time_coverage_end=naaulu.util.naive_utc(time),
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
        try:
            volume = xarray.open_datatree(filename, engine="h5netcdf")
            volume = volume.load()
            volume.close()
            volume = xradar.util.create_volume(
                sweeps=[volume],
                min_angle=min_angle,
                max_angle=max_angle,
            )
        except FileNotFoundError:
            raise
        except Exception as e:
            # A leftover half-written file (or one that predates the atomic
            # write above) opens but holds no usable sweeps. Treat it as a
            # cache miss so this run rebuilds it, rather than failing every
            # run until someone deletes the file by hand.
            logger.warning(f"discarding unusable radar cache {filename}: {e}")
            os.remove(filename)
            raise FileNotFoundError(filename) from e
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


def quality_filter_window_distance(sweep, fsize=2000, tr1=6.0):
    return sweep.wrl.classify.filter_window_distance(fsize=fsize, tr1=tr1).DBZH


def quality_distance(sweep):
    return numpy.clip(1.0 - sweep.range / 300_000, 0, 1)


def quality_height(sweep, site_alt, max_alt=3000.0):
    h = (sweep.z - site_alt) / max_alt
    return numpy.clip((1.0 - h) / 0.3, 0, 1)


def _polar_window(dbzh, wsize, fun):
    """wradlib's metric window filter on the (azimuth, range) dims.

    wradlib's own xarray wrapper feeds the whole array to its numpy
    implementation, which reads the azimuth count off the first axis: it is
    only correct for a plain 2-D sweep. The slices are therefore filtered
    one by one here, over any leading dimensions.
    """
    rscale = float(dbzh["range"].values[1] - dbzh["range"].values[0])
    shape = dbzh.shape
    img = numpy.asarray(dbzh.values, dtype=float).reshape(-1, shape[-2], shape[-1])
    out = numpy.stack(
        [wradlib.util.filter_window_polar(s, wsize, fun, rscale) for s in img]
    ).reshape(shape)
    return xarray.DataArray(out, coords=dbzh.coords, dims=dbzh.dims)


def _steiner_background(dbzh, bkg_radius):
    """Background reflectivity of a polar sweep and its data coverage.

    The background is the mean of the valid gates within a circle of
    ``bkg_radius`` around every gate, evaluated on the native polar grid via
    wradlib's metric window filter. ``dbzh`` holds a sweep in dBZ laid out as
    (azimuth, range) with nodata already turned into NaN.

    Returns the background in dBZ and the fraction of the circle holding
    valid data.
    """
    valid = dbzh.notnull().astype(float)
    linear = dbzh.wrl.trafo.idecibel().fillna(0.0)
    num = _polar_window(linear, bkg_radius, "uniform")
    den = _polar_window(valid, bkg_radius, "uniform")
    mean = num / den.where(den > 0)
    background = 10 * numpy.log10(mean.where(mean > 0))
    return background, den


def convective(dbzh, *, intense=40.0, bkg_radius=10_000.0,
               min_reflectivity=10.0, peakedness_min=35.0,
               expand=True, coverage=0.75):
    """Steiner et al. (1995) convective/stratiform classification on a polar sweep.

    ``dbzh`` holds a sweep in dBZ laid out as (..., azimuth, range); a plain
    2-D sweep is the normal case. The grid must be polar and regular in range
    (as produced by :func:`combine_volume`), which is what
    ``wradlib.util.filter_window_polar`` assumes.

    Echo is convective when it exceeds ``intense``, or when it stands out of
    its 10 km background by more than Steiner's peakedness relation
    (difference = 10 - background**2 / 180, only above ``peakedness_min``).
    The cores found this way are then expanded by a radius growing with the
    background reflectivity (1 to 5 km), which drags the echo around a core
    into the convective class as well.

    The background is averaged over the valid gates only; gates whose circle
    holds less than ``coverage`` valid data are not tested against it.
    Gates without data (NaN, or -32 dBZ where recode_dbzh stored sub-noise
    echo) are never convective.
    """
    nodata = dbzh.isnull() | (dbzh <= -32.0)
    data = dbzh.where(~nodata)

    background, valid_fraction = _steiner_background(data, bkg_radius)
    covered = valid_fraction >= coverage

    conv = data >= intense

    # Steiner's peakedness: how far above its background an echo must sit
    difference = 10.0 - background**2 / 180.0
    conv = conv | (
        (data > peakedness_min)
        & (background > 0)
        & (background < 42)
        & (data - difference > background)
        & covered
    )

    if expand:
        # convective radius as a function of the background reflectivity
        steps = (
            (None, 25.0, 1000.0),
            (25.0, 30.0, 2000.0),
            (30.0, 35.0, 3000.0),
            (35.0, 40.0, 4000.0),
            (40.0, None, 5000.0),
        )
        cores = conv
        for lower, upper, radius in steps:
            in_step = background.notnull()
            if lower is not None:
                in_step = in_step & (background >= lower)
            if upper is not None:
                in_step = in_step & (background < upper)
            temp = (cores & in_step).astype(float)
            dilated = _polar_window(temp, radius, "maximum") > 0
            conv = conv | dilated

    result = (conv & ~nodata & (dbzh >= min_reflectivity)).fillna(False)
    result.name = "convective"
    result.attrs = {
        "long_name": "convective echo classification (Steiner et al. 1995)",
    }
    return result


def sweep_convective(sweep, **kwargs):
    """Add the Steiner convective classification of sweep["DBZH"] to the sweep."""
    sweep["convective"] = convective(sweep["DBZH"], **kwargs)
    return sweep


def z2r(dbzh):
    """Reflectivity to rain rate.

    ``dbzh`` holds a sweep in dBZ laid out as (..., azimuth, range); a plain
    2-D sweep is the normal case.

    Marshall-Palmer, or the convective relation where the Steiner et al.
    (1995) classification (see :func:`convective`) says the rain is
    convective. Reflectivity is capped at 54 dBZ first, so hail cores cannot
    run the rate away (max 122.4 mm/h).
    """
    dbzh = dbzh.clip(max=54.0)
    z = dbzh.wrl.trafo.idecibel()
    strat = z.wrl.zr.z_to_r(a=200.0, b=1.6)     # Marshall-Palmer
    conv = z.wrl.zr.z_to_r(a=300.0, b=1.4)      # WSR-88D convective
    return xarray.where(convective(dbzh), conv, strat)


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
