import importlib
import json
import logging
import os
import math

import datetime
import warnings

import numpy
import wradlib
import xradar
import xarray

import naaulu.config
import naaulu.geography
import naaulu.util
import naaulu.network
import pkgutil
import requests
from bs4 import BeautifulSoup
import naaulu.providers


database = None

logger = logging.getLogger(__name__)


def get_database():
    global database
    if database is not None:
        return database

    filename = os.path.join(naaulu.config.get_data_dir(), "radars.json")
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            database = json.load(f)
    else:
        raise ValueError(f"file {os.path.basename(filename)} missing, check installation")

    return database


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
            location = (radar["longitude"], radar["latitude"])
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
        if v["end"] is None or datetime.date.fromisoformat(v["end"]) >= start.date()
    }

    logger.info(f"{key} => {len(radars)} radars")
    return radars.keys()


def get_radar(time: datetime.datetime, duration: datetime.timedelta, wsi: str):
    radars = get_database()
    if wsi not in radars:
        raise ValueError(f"No providers found for WSI {wsi}")

    entry = radars[wsi]
    prov = entry.get("provider")
    candidates = [(0, prov, None)] if prov else []

    if not candidates:
        raise ValueError(f"No providers found for WSI {wsi}")

    candidates.sort(reverse=True)  # highest quality first

    last_error = None
    for _, provider, _ in candidates:
        try:
            module = importlib.import_module(f"naaulu.providers.{provider}")
            return module.radar(time=time, duration=duration, wsi=wsi)
        except Exception as e:
            last_error = e
            logger.debug(f"provider {provider} failed for {wsi}: {e}")

    raise RuntimeError(f"Could not fetch radar data for {wsi} from provider {provider}") from last_error


def path(*,
    time,
    duration,
    provider,
    wsi,
    min_angle=0,
    max_angle=90,
    ):
    time_str = naaulu.util.format_time(time)
    wsi_str = wsi.replace("-","_")
    duration_str = naaulu.util.format_duration(duration)
    angle_str = f"{int(min_angle)}_{int(max_angle)}"
    filename = ".".join(
        [
            time_str,
            duration_str,
            provider,
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
    sweeps = get_radar(time=time, duration=duration, wsi=wsi)
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
    provider = radar.get("provider")
    
    filename = path(
        time=time,
        provider=provider,
        wsi=wsi,
        duration=duration,
        min_angle=min_angle,
        max_angle=max_angle,
        )
    
    if update == False:
        volume = xarray.open_datatree(filename, engine="h5netcdf")
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
        logger.info(f"saving radar volume: {time} {provider} {wsi}")
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
        logger.debug(f"volume {time} {provider} {wsi} {key}: DBZH NaN={nan_count}/{total} ({100*nan_count/total:.1f}%), below5dBZ={below_noise}/{total} ({100*below_noise/total:.1f}%)")

    return volume


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
    """Fill DBZH NaNs in the base sweep from higher elevations.

    For each NaN gate in the base sweep, look up the same ground range
    (sqrt(x**2 + y**2)) on the next-higher elevation at the same azimuth
    and linearly interpolate between the two bracketing range bins. If the
    higher sweep is also NaN at that ground range, the process recurses to
    the next-higher elevation until all higher sweeps are exhausted.

    Other sweeps at the same fixed angle as the base are also used as
    fill sources (ordered ahead of strictly higher elevations).
    """
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
