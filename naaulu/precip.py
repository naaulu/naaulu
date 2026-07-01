import datetime
import logging
import os
import warnings

import numpy
import xarray

import wradlib.ipol
import wradlib.georef

import naaulu.util
import naaulu.config
import naaulu.errors
import naaulu.geography

logger = logging.getLogger(__name__)


PRODUCTS = {
    "dove":     [("PT5M", "1km"), ("PT1H", "2km"), ("PT24H", "2km")],
    "eider":    [("PT5M", "1km"), ("PT1H", "2km"), ("PT24H", "2km")],
    "fulmar":   [("PT5M", "1km"), ("PT1H", "2km"), ("PT24H", "2km")],
    "gadwall":  [("PT5M", "1km"), ("PT1H", "2km"), ("PT24H", "2km")],
    "euradclim": [("PT1H", "2.5km")],
    "radklim":  [("PT5M", "1km"), ("PT1H", "2km"), ("PT24H", "2km")],
}


def get_dove(time, duration=None, resolution=None):
    raise NotImplementedError("dove product is estimated, not fetched")


def get_eider(time, duration=None, resolution=None):
    raise NotImplementedError("eider product is estimated, not fetched")


def get_fulmar(time, duration=None, resolution=None):
    raise NotImplementedError("fulmar product not yet implemented")


def get_gadwall(time, duration=None, resolution=None):
    raise NotImplementedError("gadwall product not yet implemented")


def get_euradclim(time, duration=None, resolution=None):
    import zipfile
    import h5py
    import naaulu.network
    from naaulu.radar import extract_odim_georef

    dataset_name = "RAD_OPERA_HOURLY_RAINFALL_ACCUMULATION_EURADCLIM"
    version = "2.0"
    timestamp = time.strftime("%Y%m%d%H%M")
    yearmonth = time.strftime("%Y%m")
    temp_dir = naaulu.config.get_download_dir()
    file_name = f"RAD_OPERA_HOURLY_RAINFALL_ACCUMULATION_{timestamp}.h5"
    target_path = os.path.join(temp_dir, file_name)

    if not os.path.exists(target_path):
        base_url = "https://api.dataplatform.knmi.nl/open-data/v1"
        archive_name = f"{dataset_name}_{yearmonth}_0002.zip"
        url = f"{base_url}/datasets/{dataset_name}/versions/{version}/files/{archive_name}/url"
        archive_path = naaulu.network.get_temp(
            url, api_key, archive_name, "temporaryDownloadUrl"
        )
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            for entry in zip_ref.infolist():
                if os.path.basename(entry.filename) == file_name:
                    with zip_ref.open(entry) as source, open(
                        target_path, "wb"
                    ) as target:
                        target.write(source.read())
                    break
            else:
                raise FileNotFoundError(f"{file_name} not found in archive")

    with h5py.File(target_path, "r") as f:
        crs, bounds, resolution = extract_odim_georef(f)
        precip = f["dataset1/data1/data"][:]

    dataset["precipitation"] = dataset["precipitation"].where(
        dataset["precipitation"] != -9999000, numpy.nan
    )
    dataset["precipitation"].attrs = {}

    return dataset


def get_radklim(time, duration=None, resolution=None):
    raise NotImplementedError("RadKlim dataset provider not yet implemented")


_PRODUCTS = {
    "dove":     get_dove,
    "eider":    get_eider,
    "fulmar":   get_fulmar,
    "gadwall":  get_gadwall,
    "euradclim": get_euradclim,
    "radklim":  get_radklim,
}


def get_product(time, duration, resolution, product):
    if product not in _PRODUCTS:
        raise ValueError(f"Unknown product: {product}")
    return _PRODUCTS[product](time, duration, resolution)


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


def set_metadata(*, time, tile, duration, resolution, product):

    metadata = {}
    metadata["time"] = naaulu.util.format_time(time)
    metadata["tile"] = naaulu.util.format_tile(tile)
    metadata["duration"] = naaulu.util.format_duration(duration)
    metadata["resolution"] = naaulu.util.format_distance(resolution)
    metadata["product"] = product

    return metadata


def get_base(duration, resolution, product):

    bases = PRODUCTS[product]

    base = None
    for base in reversed(bases):
        base_duration = naaulu.util.parse_duration(base[0])
        base_resolution = naaulu.util.parse_distance(base[1])
        if duration > base_duration and duration.total_seconds() % base_duration.total_seconds() == 0:
            if resolution % base_resolution == 0:
                return (base_duration, base_resolution)

    raise ValueError(f"duration and resolution should be a multiple of: {bases}  ")


def get(time, tile, duration, resolution, product):

    logger.info(
        f"get dataset: {time} {tile.bounds} {duration} {resolution} {product}"
    )
    metadata = set_metadata(
        time=time,
        tile=tile,
        duration=duration,
        resolution=resolution,
        product=product,
        )
    filename = path(metadata)

    if os.path.exists(filename):
        dataset = read(filename)
        return dataset

    dataset_source = get_product(time, duration, resolution, product)
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


def combine(*, time, tile, duration, resolution, product, base=None):

    logger.info(
        f"combine dataset: {time} {tile.bounds} {duration} {resolution} {product}"
    )

    if base is None:
        try:
            base = get_base(duration, resolution, product)
        except (ValueError, KeyError):
            base = (duration, resolution)

    base_duration, base_resolution = base

    ts = time - duration + base_duration
    rainaccums = []

    if resolution != base_resolution:
        target = naaulu.geography.tile_dataset(tile, resolution)
        source = naaulu.geography.tile_dataset(tile, base_resolution)
        transform = get_transform(source, target)

    while ts <= time:
        try:
            rainaccum = get(
                ts,
                tile,
                base_duration,
                base_resolution,
                product,
            )

            if resolution != base_resolution:
                rainaccum = warp(rainaccum, target, transform=transform)
        except Exception:
            rainaccum = None
        rainaccums.append(rainaccum)
        ts = ts + base_duration

    avail = rainaccums.count(None) / len(rainaccums)

    rainaccums = [x for x in rainaccums if x is not None]
    if len(rainaccums) == 0:
        raise naaulu.errors.NoDataError(f"No {product} data at time {time} on tile {tile.bounds} for duration {duration}")

    rainsum = sum(rainaccums)
    metadata = set_metadata(
        time=time,
        tile=tile,
        duration=duration,
        resolution=resolution,
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


def extract(*, times, coords, codes, chunk, duration, resolution, product):

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
                try:
                    rasters[tile] = get(
                        time=time,
                        tile=tile,
                        duration=duration,
                        resolution=resolution,
                        product=product
                    )
                except naaulu.errors.NoDataError:
                    logger.info(f"no data at {time} for tile {tile.bounds}")
                    rasters[tile] = None
        if coords_crs is None:
            coords_crs = {}
            for tile in rasters.keys():
                if rasters[tile] is None:
                    continue
                coords_crs[tile] = wradlib.georef.reproject(
                    coords,
                    src_crs=4326,
                    trg_crs=rasters[tile].spatial_ref.attrs["crs_wkt"],
                )
        values = []
        for i, (pt, tile) in enumerate(zip(coords, tiles)):
            if rasters[tile] is None:
                val = numpy.nan
            elif tile in coords_crs:
                val = rasters[tile]["precipitation"].sel(
                    x=coords_crs[tile][i, 0],
                    y=coords_crs[tile][i, 1],
                    method="nearest",
                ).item()
            else:
                val = numpy.nan
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


def get_transform(
    source: xarray.Dataset, target: xarray.Dataset
) -> wradlib.ipol.RectBin:
    """
    Create a binned transformation from source raster to target raster grid.
    Only target pixels whose centers fall within the source extent are used.
    """
    coords_source = wradlib.georef.get_raster_coordinates(source, mode="center").values.reshape(-1, 2)

    coords_target = wradlib.georef.get_raster_coordinates(target, mode="center").values

    xmin, ymin = coords_source[:, 0].min(), coords_source[:, 1].min()
    xmax, ymax = coords_source[:, 0].max(), coords_source[:, 1].max()

    x, y = coords_target[..., 0], coords_target[..., 1]
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
    dataset.attrs = source.attrs

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

    # Validate coverage — require at least 2 data points within the window
    if len(datasets) < 2:
        raise ValueError("insufficient data for accumulation")

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
        (t_hr, ds[[var]].drop_vars("time", errors="ignore").drop_vars("sweep_fixed_angle", errors="ignore").expand_dims(synthetic_time=[t_hr]))
        for t_hr, ds in zip(time_hr, datasets)
    ]))
    stacked = xarray.concat(arrays, dim="synthetic_time")
    for i, t in enumerate(stacked.synthetic_time.values):
        vals = stacked[var].isel(synthetic_time=i).values
        n = vals.size
        n_nan = numpy.isnan(vals).sum()
        n_pos = (vals > 0).sum()
        logger.debug(f"accumulate stacked t={t:.4f}h: NaN {n_nan}/{n} ({100*n_nan/n:.1f}%), positive {n_pos}/{n} ({100*n_pos/n:.1f}%)")

    # Interpolate boundaries (fill NaN along time axis first)
    stacked = stacked.interpolate_na(dim="synthetic_time", method="linear")
    # Fill remaining boundary NaN: propagate nearest valid value to edges
    arr = stacked[var].values  # shape: (time, y, x)
    ntime = arr.shape[0]
    for t in range(ntime):
        nan_mask = numpy.isnan(arr[t])
        if not nan_mask.any():
            continue
        # find first and last valid for each pixel
        valid_any = ~nan_mask  # (y, x)
        # forward fill: if this pixel is NaN but previous had a value, use it
        if t > 0:
            prev_valid = ~numpy.isnan(arr[t-1])
            fill = prev_valid & nan_mask
            arr[t][fill] = arr[t-1][fill]
        # backward fill: if this pixel is NaN but next has a value, use it
        if t < ntime - 1:
            next_valid = ~numpy.isnan(arr[t+1])
            fill = next_valid & numpy.isnan(arr[t])
            arr[t][fill] = arr[t+1][fill]
    for i, t in enumerate(stacked.synthetic_time.values):
        vals = stacked[var].isel(synthetic_time=i).values
        n = vals.size
        n_nan = numpy.isnan(vals).sum()
        n_pos = (vals > 0).sum()
        logger.debug(f"accumulate interpolated t={t:.4f}h: NaN {n_nan}/{n} ({100*n_nan/n:.1f}%), positive {n_pos}/{n} ({100*n_pos/n:.1f}%)")
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
