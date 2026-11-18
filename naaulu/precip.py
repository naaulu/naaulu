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
    "nimbus":   [("PT1H", "2km")],
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


def get_nimbus(time, duration=None, resolution=None):
    """OPERA NIMBUS 1-hour rainfall accumulation composite (mm, 2 km).

    Published every 15 minutes on the :00/:15/:30/:45 grid; the file stamp
    is the END of the accumulation window, so the file at `time` holds
    (time - 1h, time] — the same convention naaulu uses for its own hourly
    product. The newest file lags its stamp by roughly 8-13 minutes (the
    rain-rate frames it sums are issued HH+5..HH+7).

    Composite files carry no xstart/ystart: the projection is
    axis-aligned, so the LL/UL lon/lat corners pin the grid origin.
    """
    import h5py
    import pyproj
    from naaulu.radar import _s3_download

    if duration is not None and duration != datetime.timedelta(hours=1):
        raise ValueError(
            "nimbus is a 1-hour product; use combine() for longer durations"
        )

    # snap to the 15-minute publication grid (down)
    stamp = time.replace(minute=(time.minute // 15) * 15, second=0, microsecond=0)
    if stamp != time:
        logger.debug(f"nimbus: requested {time}, publication stamp {stamp}")
    key = f"{stamp:%Y/%m/%d}/OPERA/COMP/OPERA@{stamp:%Y%m%dT%H%M}@0@ACRR.h5"

    try:
        filepath = _s3_download(key)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"NIMBUS ACRR not available for {stamp:%Y-%m-%d %H:%M} "
            f"(published ~10 min after the stamp)"
        ) from e

    with h5py.File(filepath, "r") as f:
        # materialize before the file closes: AttributeManager dies with it
        where = dict(f["where"].attrs)
        what = dict(f["dataset1/data1/what"].attrs)
        raw = f["dataset1/data1/data"][:]

    gain = float(what.get("gain", 1.0))
    offset = float(what.get("offset", 0.0))
    nodata = float(what.get("nodata", -9999000.0))
    undetect = float(what.get("undetect", -8888000.0))

    values = raw * gain + offset
    values[raw == nodata] = numpy.nan    # no radar coverage: missing
    values[raw == undetect] = 0.0        # no echo: dry, a real zero

    projdef = where["projdef"]
    crs = pyproj.CRS.from_proj4(
        projdef.decode() if isinstance(projdef, bytes) else projdef
    )
    xsize, ysize = int(where["xsize"]), int(where["ysize"])
    xscale, yscale = float(where["xscale"]), float(where["yscale"])

    # grid origin from the LL/UL corners (lon/lat rounded: snap to the grid)
    to_laea = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    ll_x, ll_y = to_laea.transform(
        float(where["LL_lon"]), float(where["LL_lat"])
    )
    ul_x, ul_y = to_laea.transform(
        float(where["UL_lon"]), float(where["UL_lat"])
    )
    xmin = round(ll_x / xscale) * xscale
    ymax = round(ul_y / yscale) * yscale

    # WGS84 source grid over the domain: the LAEA rectangle's image in
    # lon/lat is curved, so densify the edges rather than trusting corners.
    width, height = xsize * xscale, ysize * yscale
    t = numpy.linspace(0, 1, 100)
    edge_x = numpy.concatenate(
        [t * width, numpy.full(100, width), (1 - t) * width, numpy.zeros(100)]
    )
    edge_y = numpy.concatenate(
        [numpy.zeros(100), t * height, numpy.full(100, height), (1 - t) * height]
    )
    to_wgs = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lons, lats = to_wgs.transform(xmin + edge_x, ymax - edge_y)
    bounds = (
        float(lons.min()), float(lats.min()),
        float(lons.max()), float(lats.max()),
    )

    # Integer-arcsecond grid at native resolution, built the way
    # create_raster_geographic builds it internally: that wrapper computes
    # bounds*3600 in floats and a projected bbox carries sub-arcsecond dust
    # into snap_resolution's integer check, which then rejects it.
    # Bounds are grown (by <1 px) to divide the native resolution exactly:
    # snap_resolution hunts for *any* divisor and can collapse to a
    # needlessly fine grid when the extent's divisor structure is poor
    # (it returned 9 arcsec where 166 was asked for).
    bounds_arc = tuple(int(round(b * 3600)) for b in bounds)
    res_deg = wradlib.georef.meters_to_degrees(
        meters=int(xscale),
        longitude=(bounds[0] + bounds[2]) / 2,
        latitude=(bounds[1] + bounds[3]) / 2,
    )
    xres, yres = (max(1, int(round(r * 3600))) for r in res_deg)
    minx, miny, maxx, maxy = bounds_arc
    maxx = minx + ((maxx - minx + xres - 1) // xres) * xres
    maxy = miny + ((maxy - miny + yres - 1) // yres) * yres
    source = wradlib.georef.create_raster_xarray(
        wradlib.georef.get_earth_projection(arcsecond=True),
        (minx, miny, maxx, maxy),
        (xres, yres),
    )

    # nearest source pixel for every source-grid centre (coords are
    # arcseconds, matching create_raster_geographic's CF convention)
    lon = source.x.values / 3600.0
    lat = source.y.values / 3600.0
    grid_lon, grid_lat = numpy.meshgrid(lon, lat)
    gx, gy = to_laea.transform(grid_lon, grid_lat)
    ix = numpy.floor((gx - xmin) / xscale).astype(numpy.int32)
    iy = numpy.floor((ymax - gy) / yscale).astype(numpy.int32)
    inside = (ix >= 0) & (ix < xsize) & (iy >= 0) & (iy < ysize)
    sampled = numpy.full(ix.shape, numpy.nan)
    sampled[inside] = values[iy[inside], ix[inside]]

    dataset = xarray.Dataset(
        {"precipitation": (("y", "x"), sampled)},
        coords={"x": source.x, "y": source.y, "spatial_ref": source.spatial_ref},
    )
    dataset.attrs = dict(source.attrs)

    return dataset


_PRODUCTS = {
    "dove":     get_dove,
    "eider":    get_eider,
    "fulmar":   get_fulmar,
    "gadwall":  get_gadwall,
    "euradclim": get_euradclim,
    "radklim":  get_radklim,
    "nimbus":   get_nimbus,
}


def get_product(time, duration, resolution, product):
    if product not in _PRODUCTS:
        raise ValueError(f"Unknown product: {product}")
    return _PRODUCTS[product](time, duration, resolution)


def write_netcdf(dataset, filename):

    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    # remove existing file: h5py cannot truncate an open file
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

    # count by identity: xarray == None is elementwise and truthy
    missing = sum(1 for x in rainaccums if x is None)
    avail = 1.0 - missing / len(rainaccums)

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

    # mean radar_count over base files; files without it are skipped
    counts = [x.attrs["radar_count"] for x in rainaccums if "radar_count" in x.attrs]
    if counts:
        rainsum.attrs["radar_count"] = float(numpy.mean(counts))

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
    """Binned transform from source to target grid.
    Only target pixels inside the source extent are filled.
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
    """Warp source dataset onto the target grid; a precomputed transform may be reused."""
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
    """Accumulate a rate variable over a time interval by trapezoidal integration.
    Returns a dataset with 'precipitation'.
    """

    # normalize to naive UTC (numpy has no tz)
    time_start = naaulu.util.to_datetime64(time_start)
    time_end = naaulu.util.to_datetime64(time_end)

    # require at least 2 points
    if len(datasets) < 2:
        raise ValueError("insufficient data for accumulation")

    # hours since time_start
    timestamps = [naaulu.util.to_datetime64(t) for t in timestamps]
    time_hr = [
        (t - time_start).astype("timedelta64[s]").astype(float) / 3600.0
        for t in timestamps
    ]
    t_start_hr = 0.0
    t_end_hr = (time_end - time_start).astype("timedelta64[s]").astype(float) / 3600.0

    # keep only the target variable
    time_hr, arrays = zip(*sorted([
        (t_hr, ds[[var]].drop_vars("time", errors="ignore").drop_vars("sweep_fixed_angle", errors="ignore").expand_dims(synthetic_time=[t_hr]))
        for t_hr, ds in zip(time_hr, datasets)
    ]))
    stacked = xarray.concat(arrays, dim="synthetic_time",coords='different',compat="equals")
    for i, t in enumerate(stacked.synthetic_time.values):
        vals = stacked[var].isel(synthetic_time=i).values
        n = vals.size
        n_nan = numpy.isnan(vals).sum()
        n_pos = (vals > 0).sum()
        logger.debug(f"accumulate stacked t={t:.4f}h: NaN {n_nan}/{n} ({100*n_nan/n:.1f}%), positive {n_pos}/{n} ({100*n_pos/n:.1f}%)")

    # fill NaN along time, then edges from neighbors
    stacked = stacked.interpolate_na(dim="synthetic_time", method="linear")
    arr = stacked[var].values
    ntime = arr.shape[0]
    for t in range(ntime):
        nan_mask = numpy.isnan(arr[t])
        if not nan_mask.any():
            continue
        valid_any = ~nan_mask
        # forward fill from previous time
        if t > 0:
            prev_valid = ~numpy.isnan(arr[t-1])
            fill = prev_valid & nan_mask
            arr[t][fill] = arr[t-1][fill]
        # backward fill from next time
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

    # interior points
    mask = (stacked.synthetic_time > t_start_hr) & (stacked.synthetic_time < t_end_hr)
    middle = stacked.sel(synthetic_time=mask)

    final = xarray.concat([start, middle, end], dim="synthetic_time")
    time_final = numpy.concatenate([[t_start_hr], stacked.synthetic_time.sel(synthetic_time=mask).values, [t_end_hr]])
    final = final.assign_coords(synthetic_time=time_final)

    accumulated = final[[var]].integrate("synthetic_time").rename({var: "precipitation"})

    # re-attach static coords
    for name, coord in datasets[0].coords.items():
        if "synthetic_time" not in coord.dims:
            accumulated = accumulated.assign_coords({name: coord})

    return accumulated
