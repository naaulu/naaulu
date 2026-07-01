import appdirs
import keyring
import logging
import os
import pathlib
import shutil
from importlib.resources import files

import numpy

import naaulu.geography
import naaulu.util

logger = logging.getLogger(__name__)

ARCH_DIR = os.path.join(appdirs.user_cache_dir("naaulu"), "archive")


def get_bundled_data_path(filename):
    """Get path to a bundled data file in naaulu/data/."""
    return files("naaulu").joinpath("data", filename)

SECRETS_FILE = pathlib.Path.home() / ".naaulu_secrets"
_secret_cache = {}

def _load_secrets():
    """Internal: Load secrets from file into the cache."""
    if not _secret_cache and SECRETS_FILE.exists():
        with open(SECRETS_FILE) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    _secret_cache[key.strip()] = value.strip()

def get_secret(key):
    """Get a secret: try keyring first, fall back to .naaulu_secrets file."""
    value = keyring.get_password("naaulu", key)
    if value is not None:
        return value
    _load_secrets()
    return _secret_cache.get(key)


def set_secret(key, value):
    """Store a secret in the system keyring."""
    keyring.set_password("naaulu", key, value)


def set_archive_dir(clean=False, disabled=False):

    global ARCH_DIR

    if disabled:
        ARCH_DIR = None
        return

    if clean and os.path.exists(ARCH_DIR):
        shutil.rmtree(ARCH_DIR)

    os.makedirs(ARCH_DIR, exist_ok=True)


def get_archive_dir():

    global ARCH_DIR

    return ARCH_DIR


def get_cache_dir(subdir=None):
    """Main cache directory for derived artifacts (e.g. ~/.cache/naaulu/main)."""
    dir = os.path.join(appdirs.user_cache_dir("naaulu"), "main")
    if subdir:
        dir = os.path.join(dir, subdir)
    os.makedirs(dir, exist_ok=True)
    return dir


def get_download_dir(subdir=None):
    """Download cache for files fetched from HTTP/S3 (e.g. ~/.cache/naaulu/download)."""
    dir = os.path.join(appdirs.user_cache_dir("naaulu"), "download")
    if subdir:
        dir = os.path.join(dir, subdir)
    os.makedirs(dir, exist_ok=True)
    return dir


def get_data_dir(subdir=None):
    """Persistent reference data dir (XDG data, e.g. ~/.local/share/naaulu).

    Installed by install.sh, copied from repo data. Contains stable inputs
    that are always needed (country boundaries, radar databases). Use
    get_download_dir() for files fetched at runtime.
    """
    dir = appdirs.user_data_dir("naaulu")
    if subdir:
        dir = os.path.join(dir, subdir)
    os.makedirs(dir, exist_ok=True)
    return dir


def add_time_args(parser):
    parser.add_argument(
        "--first",
        help="first rainfall accumulation event in ISO 8601 format",
        type=str,
    )
    parser.add_argument(
        "--last",
        help="last rainfall accumulation event in ISO 8601 format",
        type=str,
    )
    parser.add_argument(
        "--step",
        help="time step between rainfall accumulation events in ISO 8601 format",
        type=str,
        default="PT5M"
    )


def add_spatial_args(parser):
    parser.add_argument(
        "--window",
        help="geographic window in WGS84",
        type=float,
        nargs=4,
        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
    )
    parser.add_argument(
        "--country",
        help="country name",
        type=str,
    )
    parser.add_argument(
        "--no-mainland",
        help="include all islands (default: mainland only)",
        action="store_true",
    )
    parser.add_argument(
        "--geofile",
        help="geojson file containing one feature)",
        type=str,
    )
    parser.add_argument(
        "--name",
        help="name of the area of interest",
        type=str,
    )
    parser.add_argument(
        "--chunk",
        help="size of geographic chunk in degrees",
        default=2,
        type=int,
    )
    parser.add_argument(
        "--buffer",
        help="buffer around the area of interest",
        type=int,
        default=40E3,
    )
 

def add_product_args(parser):
    parser.add_argument(
        "--duration",
        help="rainfall accumulation duration [P10D, PT3H, PT5M, ...]",
        type=str,
        default="PT5M",
    )
    parser.add_argument(
        "--resolution",
        help="raster pixel resolution [1km, 500m, ...]",
        default="1km"
    )
    parser.add_argument(
        "--product",
        help="rainfall estimation product",
        default="dove"
    )

def add_system_args(parser):
    parser.add_argument(
        "--log",
        help="log level [critical, error, warning, info, debug]",
        type=str,
        default="warning",
    )    
    parser.add_argument(
        "--no-archive",
        help="disable archive output",
        action="store_true",
    )
    parser.add_argument(
        "--clean",
        help="switch to clean cache",
        action="store_true",
    )
    

def setup_logging(log):

    level = getattr(logging, log.upper(), logging.WARNING)
    logging.basicConfig(level=level)


def parse_times(args):
    
    if args.first is None:
        raise ValueError("provide --first and (optionally) --last event")
    first = naaulu.util.parse_time(args.first)

    if args.last is None:
        last = first
    else:
        last = naaulu.util.parse_time(args.last)

    step = naaulu.util.parse_duration(args.step)  # returns datetime.timedelta
    if args.step is None:
        raise ValueError("provide --step between event")
    
    times = naaulu.util.time_arange(
        start=first,
        end=last,
        step=step,
        )

    return times, step


def parse_area(args):

    area = args.name
    geometry = None

    if args.window:
        window = [numpy.round(coord, 1) for coord in args.window]
        geometry = naaulu.geography.import_window(window)

    if args.country:
        area = args.country
        iso_code = naaulu.geography.country_code(area)
        geometry = naaulu.geography.import_country(iso_code, mainland_only=not args.no_mainland)

    if args.geofile:
        geometry = naaulu.geography.import_geometry(args.geofile)

    if args.buffer is not None:
        geometry = naaulu.geography.buffer_geometry(geometry, distance=args.buffer)

    if geometry is None:
        raise ValueError("provide --country, --geofile or --window to define area")

    if area is None:
        raise ValueError("provide --name to reference your area")
    
    chunk = args.chunk
    
    return area, chunk, geometry


def parse_product(args):

    product = args.product
    product = product.lower()

    duration = args.duration
    duration = naaulu.util.parse_duration(duration)

    resolution = args.resolution
    resolution = naaulu.util.parse_distance(resolution)

    return duration, resolution, product


