import bisect
import collections
import datetime
import logging
import traceback
import warnings

import numpy
import wradlib
import xarray
import xradar.util

import naaulu.precip
import naaulu.radar
import naaulu.util
import naaulu.multi

logger = logging.getLogger(__name__)

# one radar scan: mean timestamp, outermost gate still holding data, sweep
Sweep = collections.namedtuple("Sweep", ["time", "rmax", "sweep"])

# a missing radar volume may still arrive while its window is this recent
RETRY_GRACE = datetime.timedelta(minutes=20)


def blend_scans(grids):
    """Weighted combination of two time-adjacent rasterized scans.

    ``grids`` holds (weight, dataset) pairs; weights are renormalized per
    pixel over the scans that hold data there.
    """
    (head_weight, head), (tail_weight, tail) = grids
    combined = head.copy(deep=True)
    for var in head.data_vars:
        head_values = head[var].values
        tail_values = tail[var].values if var in tail else numpy.full(head_values.shape, numpy.nan)
        head_valid = numpy.isfinite(head_values)
        tail_valid = numpy.isfinite(tail_values)
        numerator = (
            head_weight * numpy.where(head_valid, head_values, 0.0)
            + tail_weight * numpy.where(tail_valid, tail_values, 0.0)
        )
        denominator = numpy.where(head_valid, head_weight, 0.0) + numpy.where(
            tail_valid, tail_weight, 0.0
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            combined[var].values = numpy.where(denominator > 0, numerator / denominator, numpy.nan)

    return combined


def mean_radar_count(rainrates_radar, timestamps, tile_radars):
    """Average number of radars contributing to each bucket.

    Returns None when no bucket is available.
    """
    counts = [
        sum(1 for wsi in rainrates_radar.get(t, {}) if wsi in tile_radars)
        for t in timestamps
    ]
    if not counts:
        return None

    return float(numpy.mean(counts))


class Base:
    """Estimator over a time range: radar volumes to rainrates to accumulations.

    Subclasses override the method settings below and implement
    ``compute_rainrate_radar``.
    """

    method = None
    variables = ["DBZH"]
    min_angle = 0
    max_angle = 90

    def __init__(
        self,
        *,
        start,
        end,
        duration,
        step,
        geometry,
        chunk,
        resolution,
        radars,
        azimuth_scale=1,
        range_scale=500,
        max_range=180e3,
        precision=8,
    ):
        """Configure the run and select the radars per tile."""
        self.start = start
        self.end = end
        self.duration = duration
        self.step = step
        self.geometry = geometry
        self.chunk = chunk
        self.resolution = resolution
        self.azimuth_scale = azimuth_scale
        self.range_scale = range_scale
        self.max_range = max_range
        self.precision = precision

        self.set_tiles()
        self.set_radars(radars=radars)

        self.transform = {}

        logger.info(f"selected radars: {self.radars}")

        self.volumes = {}
        self.missing_volumes = {}
        self.radar_sweeps = {}
        self.rainrates_radar = {}
        self.rainrates = {}
        self.rainaccums = {}

    def clean(self, time):
        """Drop stored data older than the retention window."""

        prev = time - self.duration
        prev2 = prev - self.duration
        self.rainrates = {
            k: v for k, v in self.rainrates.items() if k >= prev2
        }
        self.rainrates_radar = {
            k: v for k, v in self.rainrates_radar.items() if k >= prev2
        }
        self.radar_sweeps = {
            wsi: [entry for entry in sweeps if entry.time >= prev2]
            for wsi, sweeps in self.radar_sweeps.items()
        }
        self.missing_volumes = {
            k: v for k, v in self.missing_volumes.items() if k >= prev2
        }
        self.rainaccums = {
            k: v for k, v in self.rainaccums.items() if k >= prev
        }
        naaulu.util.malloc_trim()

    def set_tiles(self):
        """Split the geometry into tiles of chunk width and height."""
        self.tiles = naaulu.geography.chunk_geometry(
            geometry=self.geometry,
            chunk_width=self.chunk,
            chunk_height=self.chunk,
            )

    def set_radars(self, radars):
        """Assign radars to tiles within max_range, or force a given list."""
        results = naaulu.radar.select_batch(
            start=self.start,
            end=self.end,
            tiles=self.tiles,
            distance=self.max_range,
        )
        self.mapping = {}
        self.radars = []
        for tile, (selection, radars_keys) in results.items():
            tile_name = naaulu.util.format_tile(tile)
            logger.info(f"{tile_name} => {len(radars_keys)} radars")
            if radars is not None:
                radars_keys = radars
            self.mapping[tile] = radars_keys
            self.radars.extend(radars_keys)
            naaulu.radar.write_select_cache(
                distance=self.max_range,
                key=tile_name,
                selection=selection,
            )
        self.radars = sorted(set(self.radars))


    def get_radar_volume(self, time, wsi):
        """Radar volume for wsi at time, georeferenced in float32; None on failure."""

        try:
            volume = naaulu.radar.combine_volume(
                time=time,
                duration=self.duration,
                wsi=wsi,
                variables=self.variables,
                azimuth_scale=self.azimuth_scale,
                range_scale=self.range_scale,
                max_range=self.max_range,
                min_angle=self.min_angle,
                max_angle=self.max_angle,
                precision=self.precision,
                )
            volume = xradar.georeference.transforms.get_x_y_z_tree(volume)
            # Downcast georeference coordinates to float32
            for key in list(volume.children):
                if "sweep" in key:
                    ds = volume[key].ds
                    az_count = ds.sizes.get("azimuth", 0)
                    if az_count not in (360, 720):
                        logger.warning(f"{wsi} {key}: azimuth count is {az_count}, expected 360 or 720")
                    for coord in ["x", "y", "z"]:
                        if coord in ds.coords:
                            ds.coords[coord] = ds.coords[coord].astype(numpy.float32)
        except Exception as e:
            logger.debug(traceback.format_exc())
            logger.info(f"cannot get volume {wsi}: {e}")
            volume = None

        return volume


    def compute_sweep_raster(self, pair, raster, tile_bounds):
        """Rasterize one radar's sweeps for a time bucket, blending them if several."""
        t, wsi = pair
        if wsi not in self.rainrates_radar[t]:
            return None
        grids = []
        for weight, sweep in self.rainrates_radar[t][wsi]:
            sweep_ref = naaulu.radar.spatial_reference(sweep)
            ref = (tile_bounds, wsi, sweep_ref)
            if ref not in self.transform:
                sweep_f32 = sweep.copy()
                for coord in ["x", "y", "z"]:
                    if coord in sweep_f32.coords:
                        sweep_f32.coords[coord] = sweep_f32.coords[coord].astype(numpy.float32)
                self.transform[ref] = wradlib.comp.transform_binned(
                    sweep=sweep_f32,
                    raster=raster
                    )
                del sweep_f32
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                grid = wradlib.comp.sweep_to_raster(
                    sweep=sweep,
                    raster=raster,
                    transform=self.transform[ref],
                )
            grids.append((weight, grid))
        if len(grids) == 1:
            return (t, wsi, grids[0][1])
        return (t, wsi, blend_scans(grids))

    def compute_rainrate_tile(self, time, tile):
        """Composite rainrate per time bucket for one tile; None where no radar holds data."""

        times = naaulu.util.time_arange(
            start=time - self.duration,
            end=time,
            step=self.step,
            )
        raster = wradlib.georef.create_raster_geographic(
            bounds=tile.bounds,
            resolution=self.resolution,
            resolution_in_meters=True
            )
        # Downcast raster coordinates to float32
        for coord in ["x", "y"]:
            if coord in raster.coords:
                raster.coords[coord] = raster.coords[coord].astype(numpy.float32)
        radars = self.mapping[tile]
        tile_name = naaulu.util.format_tile(tile)

        items = [(t, wsi) for t in times for wsi in radars]
        results = naaulu.multi.run_parallel(
            func=self.compute_sweep_raster,
            key="pair",
            items=items,
            raster=raster,
            tile_bounds=tile.bounds,
        )

        rainrates_radar = {t: [] for t in times}
        for result in results.values():
            if result is not None:
                t, wsi, rainrate = result
                rainrates_radar[t].append(rainrate)

        for t in times:
            logger.debug(f"tile {tile_name} bucket {t}: {len(rainrates_radar[t])} radars")

        rainrates_tile = {}

        for ts, rainrates in rainrates_radar.items():
            if len(rainrates) == 0:
                rainrates_tile[ts] = None
                continue
            radargrids = xarray.concat([ds["rainrate"] for ds in rainrates], dim="radar").astype("float32")
            qualitygrids = xarray.concat([ds["quality"] for ds in rainrates], dim="radar").astype("float32")
            composite = wradlib.comp.compose_weighted(radargrids, qualitygrids)
            tile_rainrate = composite.values
            if numpy.isnan(tile_rainrate).all():
                rainrates_tile[ts] = None
            else:
                ds = composite.to_dataset()
                for coord in ("latitude", "longitude", "altitude"):
                    if coord not in ds.coords:
                        ds = ds.assign_coords({coord: rainrates[0][coord]})
                rainrates_tile[ts] = ds

            if logger.isEnabledFor(logging.DEBUG):
                tile_rainrate = rainrates_tile[ts]["rainrate"].values if rainrates_tile[ts] is not None else None
                if tile_rainrate is not None:
                    tr_nan = numpy.isnan(tile_rainrate).sum()
                    tr_total = tile_rainrate.size
                    tr_positive = (tile_rainrate > 0).sum()
                    logger.debug(f"tile rainrate {ts} {naaulu.util.format_tile(tile)}: NaN {tr_nan}/{tr_total} ({100*tr_nan/tr_total:.1f}%), positive {tr_positive}/{tr_total} ({100*tr_positive/tr_total:.1f}%)")
                else:
                    logger.debug(f"tile rainrate {ts} {naaulu.util.format_tile(tile)}: all NaN -> None")

        return rainrates_tile


    def may_arrive(self, time):
        """Whether data for this window may still arrive."""
        if time.tzinfo is None:
            time = time.replace(tzinfo=datetime.timezone.utc)
        now = datetime.datetime.now(datetime.timezone.utc)
        return time + self.duration > now - RETRY_GRACE

    def compute_rainrate_radar(self, time, wsi):
        """Rainrate sweeps for one radar; only the concrete methods provide it."""
        raise NotImplementedError(f"{type(self).__name__} has no radar method")

    def compute_rainrate(self, time):
        """Compute radar sweeps and tile rainrates for one timestamp, retrying missing radars."""

        missing = self.missing_volumes.get(time, set())
        cached = time in self.volumes

        # reuse unless the window may still gain data
        if cached and (not missing or not self.may_arrive(time)):
            logger.info(f"cache hit: rainrates for {time} already computed")
            return

        radars = self.radars
        if missing:
            radars = sorted(missing)
            logger.info(f"retrying {len(radars)} radars missing at {time}: {radars}")

        times = naaulu.util.time_arange(
            start=time - self.duration,
            end=time,
            step=self.step,
            )
        for t in times:
            if t not in self.rainrates_radar:
                self.rainrates_radar[t] = {}

        logger.info(f"start getting radar volumes for {time}")
        self.volumes[time] = {}
        results = naaulu.multi.run_parallel(
            func=self.get_radar_volume,
            key="wsi",
            items=radars,
            time=time,
            )

        for wsi, volume in results.items():
            self.volumes[time][wsi] = volume

        # radars that delivered nothing are retried on the next call
        self.missing_volumes[time] = {
            wsi for wsi, volume in self.volumes[time].items() if volume is None
        }

        logger.info(f"start computing radar rainrates for {time}")
        results = naaulu.multi.run_parallel(
            func=self.compute_rainrate_radar,
            key="wsi",
            items=radars,
            time=time,
            )

        # store the sweeps
        for wsi, rainrates in results.items():
            if rainrates is None:
                logger.debug(f"{self.method} {time} {wsi}: no rainrate (volume=None)")
                continue
            if len(rainrates) == 0:
                logger.debug(f"{self.method} {time} {wsi}: no rainrate (empty list)")
                continue
            for rainrate in rainrates:
                valid = ~numpy.isnan(rainrate["rainrate"].values)
                has_valid = valid.any(axis=0)
                rmax = float(rainrate["range"].values[has_valid][-1]) if has_valid.any() else 0.0
                if logger.isEnabledFor(logging.DEBUG):
                    el = float(rainrate["sweep_fixed_angle"].values)
                    if has_valid.any():
                        rr_vals = rainrate["rainrate"].values
                        non_nan = rr_vals[~numpy.isnan(rr_vals)]
                        rr_mean = non_nan.mean()
                        rr_max = non_nan.max()
                        logger.debug(f"{self.method} {time} {wsi} el={el} valid_range={rmax:.0f}m rainrate_mean={rr_mean:.4f} rainrate_max={rr_max:.4f}")
                    else:
                        logger.debug(f"{self.method} {time} {wsi} el={el} valid_range=0 ALL_NAN rainrate")
                time_sweep = naaulu.util.time_mean(rainrate.time.values)
                logger.debug(f"{self.method} {time} {wsi} sweep_time={time_sweep} rmax={rmax:.0f}m")
                self.store_sweep(wsi, time_sweep, rmax, rainrate)

        # free volumes
        self.volumes[time] = None
        naaulu.util.malloc_trim()

        # a retry with no new data keeps the cached tile rainrates
        if cached and not any(results.values()):
            logger.info(f"retry for {time} brought no new radar data; keeping the cached tile rainrates")
            return

        # fill every bucket of this run from the sweep store
        logger.info(f"start assigning radar sweeps to time buckets for {time}")
        for t in times:
            self.rainrates_radar[t] = self.bucket_sweeps(t)
            logger.debug(f"bucket {t}: {len(self.rainrates_radar[t])} radars")

        logger.info(f"start computing tile rainrates for {time}")
        for t in times:
            if t not in self.rainrates:
                self.rainrates[t] = {}

        results = naaulu.multi.run_parallel(
            func=self.compute_rainrate_tile,
            key="tile",
            items=self.tiles,
            time=time,
        )

        for tile, rainrates in results.items():
            for ts, rainrate in rainrates.items():
                self.rainrates[ts][tile] = rainrate


    def store_sweep(self, wsi, time_sweep, rmax, sweep):
        """Register one radar scan, dropping a duplicate with less coverage."""
        sweeps = self.radar_sweeps.setdefault(wsi, [])
        for i, entry in enumerate(sweeps):
            if entry.time == time_sweep:
                if rmax > entry.rmax:
                    sweeps[i] = Sweep(time_sweep, rmax, sweep)
                return
        sweeps.append(Sweep(time_sweep, rmax, sweep))
        sweeps.sort(key=lambda entry: entry.time)


    def bucket_sweeps(self, bucket):
        """Scans standing in for one time bucket, as (weight, sweep) per radar.

        The two scans bracketing the bucket are combined linearly. A scan
        only stands in for a bucket within one revisit of its own radar, and
        scans are never combined across a missing scan: both bracketing scans
        must be that close. Without a bracketing pair the nearest scan is
        held, as long as it is within one revisit.
        """
        selected = {}
        for wsi, sweeps in self.radar_sweeps.items():
            if not sweeps:
                continue
            revisits = [
                (b.time - a.time).total_seconds()
                for a, b in zip(sweeps, sweeps[1:])
            ]
            if revisits:
                revisit = float(numpy.median(revisits))
                max_offset = max(revisit, self.step.total_seconds())
                max_interval = 2 * revisit
            else:
                # a lone scan reaches at most one volume
                max_offset = self.duration.total_seconds()
                max_interval = max_offset

            times = [entry.time for entry in sweeps]
            index = bisect.bisect_right(times, bucket)
            before = sweeps[index - 1] if index > 0 else None
            after = sweeps[index] if index < len(sweeps) else None

            pair = None
            if before is not None and after is not None:
                head = (bucket - before.time).total_seconds()
                tail = (after.time - bucket).total_seconds()
                gap = head + tail
                if head <= max_offset and tail <= max_offset and gap <= max_interval:
                    if head > 0:
                        alpha = head / gap
                        pair = [(1 - alpha, before), (alpha, after)]
                    else:
                        pair = [(1.0, before)]
            if pair is None:
                candidates = [
                    entry
                    for entry in (before, after)
                    if entry is not None
                    and abs((bucket - entry.time).total_seconds()) <= max_offset
                ]
                if not candidates:
                    logger.debug(
                        f"{self.method} {wsi} has no scan within {max_offset:.0f}s of bucket {bucket}"
                    )
                    continue
                nearest = min(
                    candidates,
                    key=lambda entry: (abs((bucket - entry.time).total_seconds()), -entry.rmax),
                )
                pair = [(1.0, nearest)]

            selected[wsi] = [(weight, entry.sweep) for weight, entry in pair]

        return selected


    def compute_rainaccum_tile(self, time, tile):
        """Accumulate one tile's rainrates over the window; None when no data."""

        tile_name = naaulu.util.format_tile(tile)
        time_start = time - self.duration
        time_end = time

        rainrates = []
        timestamps = []
        ts = time_start
        while ts <= time_end:
            if ts in self.rainrates and tile in self.rainrates[ts] and self.rainrates[ts][tile] is not None:
                rainrates.append(self.rainrates[ts][tile])
                timestamps.append(ts)
            ts = ts + self.step

        if len(rainrates) == 0:
            logger.info(f"no rainrate data for accumulation on tile {tile_name}")
            return None

        tile_radars = self.mapping.get(tile, [])

        # counted before the boundary padding below
        radar_count = mean_radar_count(self.rainrates_radar, timestamps, tile_radars)

        # pad boundaries to the exact window bounds
        if timestamps[0] != time_start:
            timestamps.insert(0, time_start)
            rainrates.insert(0, rainrates[0])
        if timestamps[-1] != time_end:
            timestamps.append(time_end)
            rainrates.append(rainrates[-1])

        for rr, ts in zip(rainrates, timestamps):
            if logger.isEnabledFor(logging.DEBUG):
                rr_vals = rr["precipitation"].values if "precipitation" in rr else rr["rainrate"].values
                rr_nan = numpy.isnan(rr_vals).sum()
                rr_total = rr_vals.size
                rr_pos = (rr_vals > 0).sum()
                logger.debug(f"accum {time} {tile_name} input ts={ts}: NaN {rr_nan}/{rr_total} ({100*rr_nan/rr_total:.1f}%), positive {rr_pos}/{rr_total} ({100*rr_pos/rr_total:.1f}%)")

        try:
            accum = naaulu.precip.accumulate(
                datasets=rainrates,
                timestamps=timestamps,
                time_start=time_start,
                time_end=time_end)
        except ValueError:
            logger.debug(traceback.format_exc())
            logger.info(f"cannot compute radar accumulation on tile {tile_name}")
            return None

        precip_vals = accum["precipitation"].values
        p_nan = numpy.isnan(precip_vals).sum()
        p_total = precip_vals.size
        p_positive = (precip_vals > 0).sum()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"precip accumulation {time} {tile_name}: NaN {p_nan}/{p_total} ({100*p_nan/p_total:.1f}%), positive {p_positive}/{p_total} ({100*p_positive/p_total:.1f}%)")
        if p_nan == p_total:
            logger.warning(f"precip accumulation ALL NaN at {time} on tile {naaulu.util.format_tile(tile)}")
        elif p_positive == 0:
            logger.info(f"precip accumulation ALL ZERO at {time} on tile {naaulu.util.format_tile(tile)}")

        metadata = naaulu.precip.set_metadata(
            time=time,
            tile=tile,
            duration=self.duration,
            resolution=self.resolution,
            product=self.method,
        )
        accum.attrs.update(metadata)
        accum.attrs["availability"] = 1.0

        actual = set()
        for t in timestamps:
            actual.update(k for k in self.rainrates_radar.get(t, {}) if k in tile_radars)
        db = naaulu.radar.get_database()
        accum.attrs["radars"] = sorted(db.get(wsi, {}).get("name", wsi) for wsi in actual)
        if radar_count is not None:
            accum.attrs["radar_count"] = radar_count

        return accum


    def compute_rainaccum(self, time):
        """Accumulate every tile for one timestamp, computing rainrates first."""

        self.rainaccums[time] = {}

        for t in [time - self.duration, time, time + self.duration]:
            self.compute_rainrate(t)

        logger.info(f"start computing rainaccums for {time}")

        results = naaulu.multi.run_parallel(
            func=self.compute_rainaccum_tile,
            key="tile",
            items=self.tiles,
            time=time,
            )

        for tile, rainaccum in results.items():
            if rainaccum is None:
                tile_str = naaulu.util.format_tile(tile)
                logger.info(f"cannot compute rainaccum for tile {tile_str}")
            self.rainaccums[time][tile] = rainaccum

        # clear only for a single timestamp
        if self.start == self.end:
            self.transform.clear()


class Dove(Base):
    """Reflectivity-only estimator (DBZH, elevations up to 0.5 degrees)."""

    method = "dove"
    min_angle = 0
    max_angle = 0.5

    def compute_rainrate_radar(self, time, wsi):
        """Rainrate per sweep from DBZH, with a range-decaying quality."""

        if self.volumes[time][wsi] is None:
            return None

        volume = self.volumes[time][wsi]
        keys = xradar.util.get_sweep_keys(volume)
        if not keys:
            logger.debug(f"{self.method} {time} {wsi}: no sweep keys in volume")
            return None

        rainrates = []
        for key in keys:
            sweep = volume[key].ds.copy()
            if logger.isEnabledFor(logging.DEBUG):
                dbzh = sweep["DBZH"].values
                dbzh_nan = numpy.isnan(dbzh).sum()
                dbzh_total = dbzh.size
                logger.debug(f"{self.method} {time} {wsi} {key}: DBZH NaN {dbzh_nan}/{dbzh_total} ({100*dbzh_nan/dbzh_total:.1f}%)")

            z = sweep["DBZH"].wrl.trafo.idecibel()
            sweep["rainrate"] = z.wrl.zr.z_to_r()
            sweep["rainrate"].attrs = {
                "standard_name": "rainfall_rate",
                "long_name": "rainfall_rate",
                "units": "mm h-1",
                "ancillary_variables": "quality",
            }
            sweep = sweep.get(["rainrate"])

            if logger.isEnabledFor(logging.DEBUG):
                rainrate_vals = sweep["rainrate"].values
                rr_nan = numpy.isnan(rainrate_vals).sum()
                rr_total = rainrate_vals.size
                rr_positive = (rainrate_vals > 0).sum()
                logger.debug(f"{self.method} {time} {wsi} {key}: rainrate NaN {rr_nan}/{rr_total} ({100*rr_nan/rr_total:.1f}%), positive {rr_positive}/{rr_total} ({100*rr_positive/rr_total:.1f}%)")

            n_azimuth = sweep.azimuth.size
            range_values = sweep["range"]
            normalized_quality = 1.0 - range_values / range_values.max()
            quality = numpy.broadcast_to(normalized_quality.values, (n_azimuth, len(range_values)))
            sweep["quality"] = xarray.DataArray(
                quality,
                dims=["azimuth", "range"],
                coords={"azimuth": sweep["azimuth"], "range": range_values}
                )
            rainrates.append(sweep)

        if not rainrates:
            logger.debug(f"{self.method} {time} {wsi}: no valid sweeps produced rainrate")
            return None

        return rainrates


class Eider(Base):
    """Reflectivity estimator (DBZH, elevations up to 8 degrees) with
    clutter, range and height quality."""

    method = "eider"
    min_angle = 0
    max_angle = 8

    # minimum finite echo-top gates before the height quality is trusted
    ECHO_TOP_MIN_GATES = 500

    def compute_rainrate_radar(self, time, wsi):
        """Rainrate per lowest-elevation sweep, with quality from clutter
        filtering, range and echo-top height."""

        volume = self.volumes[time][wsi]
        if volume is None:
            return None

        # echo top of the precipitation layer (18 dBZ), heights in ASL;
        # below the gate threshold the height term is off, floored at 2000 m
        et_sweep = naaulu.radar.echotop(volume, level=18)
        max_alt = numpy.inf
        if et_sweep is not None:
            tops = et_sweep.echotop.values
            if int(numpy.isfinite(tops).sum()) >= self.ECHO_TOP_MIN_GATES:
                max_alt = max(float(numpy.nanmedian(tops)), 2000.0)

        volume = naaulu.radar.add_quality(
            volume, name="quality_filter_window_distance",
            func=naaulu.radar.quality_filter_window_distance,
            long_name="filter_window_distance clutter quality index",
        )
        volume = naaulu.radar.add_quality(
            volume, name="quality_distance",
            func=naaulu.radar.quality_distance,
            long_name="range quality indicator",
        )
        height_label = (
            f"{max_alt:.0f}m ASL" if numpy.isfinite(max_alt)
            else "no echo-top sample"
        )
        volume = naaulu.radar.add_quality(
            volume, name="quality_height",
            func=naaulu.radar.quality_height,
            func_kwargs={"max_alt": max_alt},
            long_name=f"height quality index (1 at 0m, 0 at {height_label})",
        )
        volume = naaulu.radar.add_quality(
            volume, name="quality",
            func=lambda s: s.quality_filter_window_distance * s.quality_distance * s.quality_height,
            long_name="aggregated quality (product of filter_window_distance, distance, height)",
        )

        def _zero_nan_quality(sweep):
            """Missing quality gates hold 0 rather than NaN."""
            attrs = sweep["quality"].attrs
            sweep["quality"] = sweep["quality"].fillna(0.0)
            sweep["quality"].attrs = attrs
            return sweep

        volume = xradar.util.apply_to_sweeps(volume, _zero_nan_quality)

        sweep_keys = list(volume.ds.sweep_group_name.values)
        fixed_angles = volume.ds.sweep_fixed_angle.values
        min_angle = min(fixed_angles)
        base_keys = [k for i, k in enumerate(sweep_keys) if fixed_angles[i] == min_angle]

        log_fill = logger.isEnabledFor(logging.DEBUG)
        rainrates = []
        for base_key in base_keys:
            if log_fill:
                ds_before = volume[base_key].ds
                nan0 = int(numpy.isnan(ds_before["DBZH"].values).sum())
                q0 = int(numpy.isnan(ds_before["quality"].values).sum())
                z0 = int((ds_before["quality"].values == 0).sum())

            sweep = naaulu.radar.fill_gap_nearest_elevation(
                volume, base_key=base_key, quality="quality", threshold=0.2,
            )
            if sweep is None:
                continue

            if log_fill:
                nan1 = int(numpy.isnan(sweep["DBZH"].values).sum())
                q1 = int(numpy.isnan(sweep["quality"].values).sum())
                z1 = int((sweep["quality"].values == 0).sum())
                logger.debug(
                    f"{self.method} {time} {wsi} {base_key}: fill DBZH NaN {nan0} -> {nan1}, "
                    f"quality NaN {q0} -> {q1}, quality==0 {z0} -> {z1}"
                )

            sweep["rainrate"] = naaulu.radar.z2r(sweep["DBZH"])
            sweep["rainrate"].attrs = {
                "standard_name": "rainfall_rate",
                "long_name": "rainfall_rate",
                "units": "mm h-1",
                "ancillary_variables": "quality",
            }
            rainrates.append(sweep.get(["rainrate", "quality"]))

        return rainrates


class Fulmar(Base):
    """Dual-polarization estimator (not yet implemented)."""

    method = "fulmar"
    variables = ["DBZH", "VRADH", "PHIDP", "RHOHV", "ZDR"]


class Gadwall(Base):
    """Dual-polarization estimator (not yet implemented)."""

    method = "gadwall"
    variables = ["DBZH", "VRADH", "PHIDP", "RHOHV", "ZDR"]
