import datetime
import importlib
import logging
import os
import pkgutil
import warnings
from typing import Any, Callable, Dict

import numpy
import xarray

import wradlib.ipol
import wradlib.georef

import naaulu.util
import naaulu.config
import naaulu.errors
import naaulu.geography
import naaulu.providers

logger = logging.getLogger(__name__)

database = None

PrecipProvider = Callable[[datetime.datetime, Any, Any], Any]
PROVIDERS: Dict[str, PrecipProvider] = {}


def register_provider(name: str, provider_func: PrecipProvider):
    PROVIDERS[name] = provider_func


def get_precip(time: datetime.datetime, duration: Any = None, resolution: Any = None, provider: str = "default"):
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown precipitation provider: {provider}")
    return PROVIDERS[provider](time, duration, resolution)


register_provider("radklim", lambda t, d, r: __import__("naaulu.providers.deu").radklim(t, d, r))
register_provider("euradclim", lambda t, d, r: __import__("naaulu.providers.ndl").euradclim(t, d, r))


def write_netcdf(dataset, filename):

    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    # Remove existing file to avoid OSError when h5py tries to truncate
    # a file that is still open from a previous xarray.open_dataset() call.
    if os.path.exists(filename):
        os.remove(filename)

    encoding = {
        "precipitation": {
            "dtype": "uint16",
            "scale_factor": 1 / 100,
            "_FillValue": 65535,
            "zlib": True,
            "complevel": 4,
        }
    }

    dataset.to_netcdf(filename, engine="h5netcdf", encoding=encoding)


def read(filename):

    dataset = xarray.open_dataset(filename, engine="h5netcdf")

    return dataset


def path(metadata, format="nc"):

    filename = ".".join(
        [
            metadata["time"],
            metadata["tile"],
            metadata["duration"],
            metadata["resolution"],
            metadata["organisation"],
            metadata["product"],
            format,
        ]
    )

    archive = naaulu.config.get_archive_dir()
    if archive is not None:        
        root = os.path.join(archive, "precip")
        filename = naaulu.util.get_path(root, filename)

    return filename


def save(dataset, format="nc"):

    filename = path(dataset.attrs, format=format)
    basename = os.path.basename(filename)

    logger.info(f"saving to {basename}")

    if format == "nc":
        write_netcdf(dataset, filename)


def load(time, tile, duration, resolution, organisation, product):

    logger.info(
        f"load dataset: {time} {tile} {duration} {resolution} {organisation} {product}"
    )

    attributes = what(time, tile, duration, resolution, organisation, product)

    filename = path(attributes)

    dataset = read(filename)

    return dataset


def set_metadata(*, time, tile, duration, resolution, organisation, product):

    metadata = {}
    metadata["time"] = naaulu.util.format_time(time)
    metadata["tile"] = naaulu.util.format_tile(tile)
    metadata["duration"] = naaulu.util.format_duration(duration)
    metadata["resolution"] = naaulu.util.format_distance(resolution)
    metadata["organisation"] = organisation
    metadata["product"] = product

    return metadata


def get_base(duration, resolution, organisation, product):

    module = importlib.import_module(f"naaulu.providers.{organisation}")
    products = getattr(module, "products")
    bases = products[product]

    base = None
    for base in reversed(bases):
        base_duration = naaulu.util.parse_duration(base[0])
        base_resolution = naaulu.util.parse_distance(base[1])
        if duration.total_seconds() % base_duration.total_seconds() == 0:
            if resolution % base_resolution == 0:
                return (base_duration, base_resolution)

    raise ValueError(f"duration and resolution should be a multiple of: {bases}  ")


def get(time, tile, duration, resolution, organisation, product):

    logger.info(
        f"get dataset: {time} {tile.bounds} {duration} {resolution} {organisation} {product}"
    )
    metadata = set_metadata(
        time=time,
        tile=tile,
        duration=duration,
        resolution=resolution,
        organisation=organisation,
        product=product,
        )
    filename = path(metadata)

    if os.path.exists(filename):
        dataset = read(filename)
        return dataset

    module = importlib.import_module(f"naaulu.providers.{organisation}")
    fun = getattr(module, product)
    dataset_source = fun(time, duration, resolution)
    dataset = naaulu.geography.tile_dataset(tile, resolution)
    dataset = warp(dataset_source, dataset)

    try:
        availability = dataset.attrs["availability"]
    except KeyError:
        availability = 1.0

    dataset.attrs = metadata
    dataset.attrs["availability"] = availability

    save(dataset)

    return dataset


def combine(*, time, tile, duration, resolution, organisation, product, base=None):

    logger.info(
        f"combine dataset: {time} {tile.bounds} {duration} {resolution} {organisation} {product}"
    )

    if base is None:
        try:
            base = get_base(duration, resolution, organisation, product)
        except (ModuleNotFoundError, AttributeError):
            base = (duration, resolution)

    base_duration, base_resolution = base

    ts = time - duration + base_duration
    rainaccums = []

    nc = int(resolution / base_resolution)
    while ts <= time:
        try:
            rainaccum = get(
                ts,
                tile,
                base_duration,
                base_resolution,
                organisation,
                product,
            )

            if nc > 1:
                rainaccum = rainaccum.coarsen(x=nc, y=nc, boundary="trim").mean()
        except Exception:
            rainaccum = None
        rainaccums.append(rainaccum)
        ts = ts + base_duration

    avail = rainaccums.count(None) / len(rainaccums)

    rainaccums = [x for x in rainaccums if x is not None]
    if len(rainaccums) == 0:
        raise naaulu.errors.NoDataError(f"No {organisation}/{product} data at time {time} on tile {tile.bounds} for duration {duration}")

    rainsum = sum(rainaccums)
    metadata = set_metadata(
        time=time,
        tile=tile,
        duration=duration,
        resolution=resolution,
        organisation=organisation,
        product=product,
        )
    rainsum.attrs.update(metadata)

    avails = [x.attrs["availability"] for x in rainaccums]
    availability = numpy.mean(avails) * avail
    if availability < 1.0:
        missing = 100 * (1 - availability)
        logger.info(
            f"{missing}% of missing data at time {time} on tile {tile.bounds} for duration {duration}"
        )
    rainsum.attrs["availability"] = availability

    counts = [x.attrs["radar_count"] for x in rainaccums if "radar_count" in x.attrs]
    if counts:
        rainsum.attrs["radar_count"] = numpy.mean(counts)

    return rainsum


def extract(*, times, coords, codes, chunk, duration, resolution, organisation, product):

    rasters = {}
    datasets = []

    tiles = naaulu.geography.coords_to_tiles(
        coords=coords,
        chunk_height=chunk,
        chunk_width=chunk,
        )
    
    coords_crs = None
    for time in times:        
        # Build datasets for each tile at this time
        for tile in tiles:
            if tile not in rasters:
                rasters[tile] = combine(
                    time=time,
                    tile=tile,
                    duration=duration,
                    resolution=resolution,
                    organisation=organisation,
                    product=product
                )
        if coords_crs is None:
            coords_crs = {}
            for tile in rasters.keys():
                if rasters[tile] is None:
                    continue
                coords_crs[tile] = wradlib.georef.transform_coords(
                    coords = coords,
                    source_crs = 4326,
                    target_crs = rasters[tile].spatial_ref.attrs["crs_wkt"],
                    )
        values = []
        # Select values for each gauge location
        for pt, tile in zip(coords, tiles):
            if rasters[tile] is None:
                val = numpy.nan
            else:
                val = rasters[tile]["precipitation"].sel(x=pt[0], y=pt[1], method="nearest")
                val = val.item()
            values.append(val)

        dataset = naaulu.gauge.create_dataset(
            codes = codes,
            coords = coords,
            values = values,
            )
        datasets.append(dataset)

    dataset = naaulu.gauge.concat_datasets(
        datasets=datasets,
        times=times,
        )
    
    return dataset


def get_list():
    global database

    if database is not None:
        return database

    database = {}

    organisations = [
        name for _, name, _ in pkgutil.iter_modules(naaulu.providers.__path__)
    ]
    for organisation in organisations:
        module = importlib.import_module(f"naaulu.providers.{organisation}")
        try:
            database[organisation] = getattr(module, "products")
        except AttributeError:
            pass

    return database


def get_transform(
    source: xarray.Dataset, target: xarray.Dataset
) -> wradlib.ipol.RectBin:
    """
    Create a binned transformation from source raster to target raster grid.
    Only target pixels whose centers fall within the source extent are used.
    """
    crs_source = source["spatial_ref"].attrs["crs_wkt"]
    crs_target = target["spatial_ref"].attrs["crs_wkt"]

    coords_source = wradlib.georef.get_raster_coordinates(source, mode="center")
    coords_source = wradlib.georef.transform_coords(
        coords_source, crs_source, crs_target
    )

    coords_target = wradlib.georef.get_raster_coordinates(target, mode="center")

    xmin, ymin = coords_source[:, 0].min(), coords_source[:, 1].min()
    xmax, ymax = coords_source[:, 0].max(), coords_source[:, 1].max()

    x, y = coords_target[:, 0], coords_target[:, 1]
    fill = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)

    return wradlib.ipol.RectBin(coords_source, coords_target, fill=fill)


def warp(
    source: xarray.Dataset,
    target: xarray.Dataset,
    statistic=numpy.nanmean,
    transform=None,
) -> xarray.Dataset:
    """
    Warp source dataset to target dataset using binned transformation.
    If a precomputed transform is provided, it will be reused.
    """
    if transform is None:
        transform = get_transform(source, target)

    data_vars = {}

    for var in source.data_vars:
        values = source[var].values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            binned = transform(values, statistic=statistic)
        data_vars[var] = (("y", "x"), binned)

    dataset = xarray.Dataset(data_vars, coords={"x": target.x, "y": target.y})

    return dataset


def accumulate(datasets, timestamps, time_start, time_end, var="rainrate"):
    """
    Accumulate a rate variable over a time interval using trapezoidal integration.

    Parameters
    ----------
    datasets : list of xarray.Dataset
        Each must contain the variable `var`.
    timestamps : list of numpy.datetime64
        Matching timestamps for each dataset.
    time_start : datetime.datetime
        Start of integration window.
    time_end : datetime.datetime
        End of integration window.
    var : str
        Name of the variable to integrate (default "rainrate").

    Returns
    -------
    xarray.Dataset
        Dataset containing 'precipitation' with all static coords preserved.
    """
    
    # Convert time_start and time_end to numpy.datetime64
    time_start = numpy.datetime64(time_start)
    time_end = numpy.datetime64(time_end)

    # Validate coverage
    if not datasets or min(timestamps) > time_start or max(timestamps) < time_end:
        raise ValueError("datasets coverage does not span the full accumulation period")

    # Convert timestamps to hours since time_start
    timestamps = [numpy.datetime64(t) for t in timestamps]
    time_hr = [
        (t - time_start).astype("timedelta64[s]").astype(float) / 3600.0
        for t in timestamps
    ]
    t_start_hr = 0.0
    t_end_hr = (time_end - time_start).astype("timedelta64[s]").astype(float) / 3600.0

    # Stack only the variable of interest
    time_hr, arrays = zip(*sorted([
        (t_hr, ds[[var]].drop_vars("time", errors="ignore").expand_dims(synthetic_time=[t_hr]))
        for t_hr, ds in zip(time_hr, datasets)
    ]))
    stacked = xarray.concat(arrays, dim="synthetic_time", coords="minimal")

    # Interpolate boundaries
    start = stacked.interp(synthetic_time=t_start_hr).expand_dims(synthetic_time=[t_start_hr])
    end   = stacked.interp(synthetic_time=t_end_hr).expand_dims(synthetic_time=[t_end_hr])

    # Select interior points
    mask = (stacked.synthetic_time > t_start_hr) & (stacked.synthetic_time < t_end_hr)
    middle = stacked.sel(synthetic_time=mask)

    # Final stack
    final = xarray.concat([start, middle, end], dim="synthetic_time")
    time_final = numpy.concatenate([[t_start_hr], stacked.synthetic_time.sel(synthetic_time=mask).values, [t_end_hr]])
    final = final.assign_coords(synthetic_time=time_final)

    # Integrate and rename to 'precipitation'
    accumulated = final[[var]].integrate("synthetic_time").rename({var: "precipitation"})

    # Re‑attach only static coords (no synthetic_time dependence)
    for name, coord in datasets[0].coords.items():
        if "synthetic_time" not in coord.dims:
            accumulated = accumulated.assign_coords({name: coord})

    return accumulated