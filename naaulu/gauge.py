import datetime
import importlib
import json
import logging
import os
import traceback
from typing import Callable, Dict, List, Tuple

import numpy
import xarray

import naaulu.config
import naaulu.geography
import naaulu.util
from naaulu.providers import noaa

NETWORKS_REGISTRY = os.path.join(os.path.dirname(__file__), "data", "networks.json")

logger = logging.getLogger(__name__)

GaugeProvider = Callable[
    [datetime.datetime, datetime.timedelta], Tuple[List[float], List[str], numpy.ndarray]
]

PROVIDERS: Dict[str, GaugeProvider] = {}


def register_provider(name: str, provider_func: GaugeProvider):
    PROVIDERS[name] = provider_func


def get_gauge(time: datetime.datetime, duration: datetime.timedelta, provider: str = "default"):
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown gauge provider: {provider}")
    return PROVIDERS[provider](time, duration)


register_provider("noaa", noaa.ghcn)
register_provider("fmi", lambda t, d: __import__("naaulu.providers.fin").fmi(t, d))
register_provider("asos", lambda t, d: __import__("naaulu.providers.iem").asos(t, d))
register_provider("rmib", lambda t, d: __import__("naaulu.providers.rmib").synop(t, d))

def _provider_args_suffix(provider_args):
    if not provider_args:
        return ""
    parts = []
    for k in sorted(provider_args):
        v = str(provider_args[k]).replace(os.sep, "_").replace(" ", "_")
        parts.append(f"{k}={v}")
    return "." + ",".join(parts)


def path(time, duration, organisation, network, provider_args=None):

    time_str = naaulu.util.format_time(time)
    duration_str = naaulu.util.format_duration(duration)
    suffix = _provider_args_suffix(provider_args)
    filename = f"{time_str}.{duration_str}.{organisation.lower()}.{network.lower()}{suffix}.nc"
    archive = naaulu.config.get_archive_dir()
    if archive is not None:
        root = os.path.join(archive, "gauge")
        filename = naaulu.util.get_path(root, filename)

    return filename


def create_dataset(*, codes, coords, values):
    """
    Create a gauge dataset from longitude/latitude coordinates and precip values.

    Parameters:
    - codes: list of codes identifying stations
    - coords: 2D array of geographic coordinates
    - values: 1D array of precip values, one per station

    Returns:
    - xarray.Dataset with 'station' dimension and 'precipitation' variable
    """
    
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


def get(*, time, duration, organisation, network, provider_args=None):

    filename = path(time, duration, organisation, network, provider_args)
    if os.path.exists(filename):
        logger.info(f"Using cached gauge data at {time} for duration {duration} from {organisation}/{network}")
        ds = xarray.open_dataset(filename, engine="h5netcdf")
        return ds
    module = f"naaulu.providers.{organisation}"
    try:
        module = importlib.import_module(module)
        fun = getattr(module, network)
    except (ModuleNotFoundError, AttributeError):
        raise ValueError(f"Gauge network {organisation}/{network} not available")

    values, codes, coords = fun(time, duration, **(provider_args or {}))

    ds = create_dataset(
        coords=coords,
        codes=codes,        
        values=values,
        )

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    ds.to_netcdf(filename, engine="h5netcdf")

    return ds

def list_networks(*, country=None, region=None, organisation=None, duration=None, weather=None, climat=None):
    with open(NETWORKS_REGISTRY) as f:
        registry = json.load(f)

    networks = registry["networks"]
    if country is not None:
        try:
            country = naaulu.geography.country_code(country)
        except Exception:
            country = str(country).upper()
        networks = [n for n in networks if n.get("country") is None or n.get("country") == country]
    if region is not None:
        networks = [n for n in networks if n.get("region") == region]
    if organisation is not None:
        networks = [n for n in networks if n.get("organisation") == organisation]
    if duration is not None:
        import isodate
        networks = [
            n for n in networks
            if isodate.parse_duration(n["temporal_resolution"]) == duration
        ]
    if weather is not None:
        networks = [n for n in networks if n.get("weather") == weather]
    if climat is not None:
        networks = [n for n in networks if n.get("climat") == climat]

    return networks


def concat_datasets(*, datasets, times):
    dataset = xarray.concat(
        datasets,
        xarray.DataArray(times, dims=["time"]),
        coords="minimal",
        compat="no_conflicts",
    )
    return dataset

def collect(times, geometry, duration, networks):

    gauges = []
    for time in times:
        datasets = []
        for organisation, network in networks:            
            try:
                dataset = naaulu.gauge.get(
                    time = time,
                    duration = duration,
                    organisation = organisation,
                    network = network,
                    )
                datasets.append(dataset)
            except Exception:
                logger.debug(traceback.format_exc())
                logger.warning(f"no gauge data available: {time} {duration} {organisation}/{network} (duration not supported)")
        
        ds = xarray.concat(datasets, dim="station")
        ds = naaulu.geography.cut(ds, geometry)
        gauges.append(ds)

    gauges = concat_datasets(
        datasets = gauges,
        times = times
    )    

    return gauges
