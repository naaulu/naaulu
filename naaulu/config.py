import appdirs
import logging
import os
import pathlib
import shutil
import tempfile
import atexit

import numpy

import naaulu.geography
import naaulu.util

logger = logging.getLogger(__name__)

TEMP_DIR = None
ARCH_DIR = os.path.join(appdirs.user_cache_dir("naaulu"), "archive")

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
    """Get a secret from the session cache or file."""
    _load_secrets()
    return _secret_cache.get(key)


def get_temp_dir():

    global TEMP_DIR

    if TEMP_DIR is None:
        if logger.isEnabledFor(logging.DEBUG):
            TEMP_DIR = os.path.join(tempfile.gettempdir(), "naaulu")
            logger.debug(f"using debug temp dir: {TEMP_DIR}")
        else:
            TEMP_DIR = tempfile.mkdtemp(prefix="naaulu_")
            logger.info(f"using temp dir: {TEMP_DIR}")

        os.makedirs(TEMP_DIR, exist_ok=True)

    return TEMP_DIR


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
    dir = appdirs.user_cache_dir("naaulu")
    if subdir:
        dir = os.path.join(dir, subdir)
    os.makedirs(dir, exist_ok=True)
    return dir


def get_data_dir(subdir=None):
    """Persistent reference data dir (XDG data, e.g. ~/.local/share/naaulu).

    Can be cleaned with `naaulu clean --data`. Use for stable inputs that take
    real time to fetch (country boundaries, radar databases). Derived artifacts
    belong in get_cache_dir instead.
    """
    dir = appdirs.user_data_dir("naaulu")
    if subdir:
        dir = os.path.join(dir, subdir)
    os.makedirs(dir, exist_ok=True)
    return dir


def cleanup_temp_dir():
    if TEMP_DIR and not logger.isEnabledFor(logging.DEBUG):
        shutil.rmtree(TEMP_DIR, ignore_errors=True)


atexit.register(cleanup_temp_dir)


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
        "--organisation",
        help="organisation providing rainfall estimation product",
        default="naaulu",
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

    if geometry is None:
        raise ValueError("provide --country, --geofile or --window to define area")

    if area is None:
        raise ValueError("provide --name to reference your area")
    
    chunk = args.chunk
    
    return area, chunk, geometry


def parse_product(args):

    organisation = args.organisation
    organisation = organisation.lower()

    product = args.product
    product = product.lower()

    duration = args.duration
    duration = naaulu.util.parse_duration(duration)

    resolution = args.resolution
    resolution = naaulu.util.parse_distance(resolution)

    return duration, resolution, organisation, product


