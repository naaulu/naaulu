import concurrent
import ctypes
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

class Base:

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
        min_angle=0,
        max_angle=90,
        azimuth_scale=1,
        range_scale=500,
        max_range=150e3,
        variables=["DBZH"],
        precision=8,
    ):

        self.start = start
        self.end = end
        self.duration = duration        
        self.step = step
        self.geometry = geometry
        self.chunk = chunk        
        self.resolution = resolution
        self.min_angle = min_angle
        self.max_angle = max_angle        
        self.azimuth_scale = azimuth_scale
        self.range_scale = range_scale
        self.max_range = max_range
        self.variables = variables
        self.precision = precision        
        
        self.set_tiles()
        self.set_radars()

        self.transform = {}        
        
        logger.info(f"selected radars: {self.radars}")

        self.volumes = {}
        self.rainrates_radar = {}
        self.rainrates = {}
        self.rainaccums = {}

    def clean(self, time):

        prev = time - self.duration
        prev2 = prev - self.duration
        self.rainrates = {
            k: v for k, v in self.rainrates.items() if k >= prev2
        }
        self.rainrates_radar = {
            k: v for k, v in self.rainrates_radar.items() if k >= prev2
        }
        self.rainaccums = {
            k: v for k, v in self.rainaccums.items() if k >= prev
        }
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    
    def set_tiles(self):
        self.tiles = naaulu.geography.chunk_geometry(
            geometry=self.geometry,
            chunk_width = self.chunk,
            chunk_height = self.chunk,
            )    
    
    def set_radars(self):
        self.mapping = {}
        self.radars = []
        for tile in self.tiles:
            radars = naaulu.radar.select(
                start=self.start,
                end=self.end,
                geom=tile,
                distance=self.max_range,
                key=naaulu.util.format_tile(tile),
                )
            self.mapping[tile] = radars
            self.radars.extend(radars)
        self.radars = sorted(set(self.radars))


    def get_radar_volume(self, time, wsi):    
    
        try:
            volume = naaulu.radar.combine_volume(
                time = time,
                duration = self.duration,
                wsi = wsi,
                variables = self.variables,
                azimuth_scale = self.azimuth_scale,
                range_scale = self.range_scale,
                max_range = self.max_range,
                min_angle = self.min_angle,
                max_angle = self.max_angle,
                precision= self.precision,                        
                )
            volume = xradar.georeference.transforms.get_x_y_z_tree(volume)
            # Downcast georeference coordinates to float32
            for key in list(volume.children):
                if "sweep" in key:
                    ds = volume[key].ds
                    for coord in ["x", "y", "z"]:
                        if coord in ds.coords:
                            ds.coords[coord] = ds.coords[coord].astype(numpy.float32)
        except Exception as e:                    
            logger.debug(traceback.format_exc())
            logger.info(f"cannot get volume {wsi}: {e}")
            volume = None

        return volume
               

    def compute_sweep_raster(self, pair, raster, tile_bounds):
        t, wsi = pair
        if wsi not in self.rainrates_radar[t]:
            return None
        sweep = self.rainrates_radar[t][wsi]
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
            rainrate = wradlib.comp.sweep_to_raster(
                sweep=sweep,
                raster=raster,
                transform=self.transform[ref],
            )
        return (t, wsi, rainrate)

    def compute_rainrate_tile(self, time, tile):

        times = naaulu.util.time_arange(
            start = time - self.duration,
            end = time,
            step = self.step,
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
            composite = wradlib.comp._compose_weighted_xarray(
                radargrids = radargrids,
                qualitygrids = qualitygrids,
            )            
            rainrates_tile[ts] = composite.to_dataset()

            tile_rainrate = rainrates_tile[ts]["rainrate"].values
            tr_nan = numpy.isnan(tile_rainrate).sum()
            tr_total = tile_rainrate.size
            tr_positive = (tile_rainrate > 0).sum()
            logger.debug(f"tile rainrate {ts} {naaulu.util.format_tile(tile)}: NaN {tr_nan}/{tr_total} ({100*tr_nan/tr_total:.1f}%), positive {tr_positive}/{tr_total} ({100*tr_positive/tr_total:.1f}%)")
    
        return rainrates_tile


    def compute_rainrate(self, time):
        
        logger.info(f"start getting radar volumes for {time}")
        self.volumes[time] = {}
        results = naaulu.multi.run_parallel(
            func=self.get_radar_volume,
            key="wsi",
            items=self.radars,           
            time=time,
            )
        
        for wsi, volume in results.items():
            self.volumes[time][wsi] = volume       

        logger.info(f"start computing radar rainrates for {time}")
        times = naaulu.util.time_arange(
            start = time - self.duration,
            end = time,
            step = self.step,
            )       
        for t in times:
            if t not in self.rainrates_radar:
                self.rainrates_radar[t] = {}
        
        results = naaulu.multi.run_parallel(
            func=self.compute_rainrate_radar,
            key = "wsi",
            items = self.radars,
            time = time,
            )
        
        # Free volumes immediately — no longer needed
        self.volumes.pop(time, None)
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        
        for wsi, rainrates in results.items():
            if rainrates is None:
                logger.debug(f"{self.method} {time} {wsi}: no rainrate (volume=None)")
                continue
            if len(rainrates) == 0:
                logger.debug(f"{self.method} {time} {wsi}: no rainrate (empty list)")
                continue
            for rainrate in rainrates:
                el = float(rainrate["sweep_fixed_angle"].values)
                valid = ~numpy.isnan(rainrate["rainrate"].values)
                has_valid = valid.any(axis=0)
                rmax = float(rainrate["range"].values[has_valid][-1]) if has_valid.any() else 0.0
                rr_vals = rainrate["rainrate"].values
                if has_valid.any():
                    non_nan = rr_vals[~numpy.isnan(rr_vals)]
                    rr_mean = non_nan.mean()
                    rr_max = non_nan.max()
                    logger.debug(f"{self.method} {time} {wsi} el={el} valid_range={rmax:.0f}m rainrate_mean={rr_mean:.4f} rainrate_max={rr_max:.4f}")
                else:
                    logger.debug(f"{self.method} {time} {wsi} el={el} valid_range=0 ALL_NAN rainrate")
                time_sweep = naaulu.util.time_mean(rainrate.time.values)
                diff = [abs((t - time_sweep).total_seconds()) for t in times]
                idx = numpy.argmin(diff)
                bucket = times[idx]
                skip = False
                if wsi in self.rainrates_radar[bucket]:
                    existing = self.rainrates_radar[bucket][wsi]
                    evalid = ~numpy.isnan(existing["rainrate"].values)
                    ehas = evalid.any(axis=0)
                    ermax = float(existing["range"].values[ehas][-1]) if ehas.any() else 0.0
                    if rmax <= ermax:
                        skip = True
                logger.debug(f"{self.method} {time} {wsi} sweep_time={time_sweep} bucket={bucket} rmax={rmax:.0f}m skip={skip}")
                if not skip:
                    self.rainrates_radar[bucket][wsi] = rainrate
        
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


    def compute_rainaccum_tile(self, time, tile):
        
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
            return

        # Pad boundaries so accumulate can interpolate to exact time_start/time_end
        if timestamps[0] != time_start:
            timestamps.insert(0, time_start)
            rainrates.insert(0, rainrates[0])
        if timestamps[-1] != time_end:
            timestamps.append(time_end)
            rainrates.append(rainrates[-1])

        for rr, ts in zip(rainrates, timestamps):
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
        except ValueError as e:
            import traceback
            logger.debug(traceback.format_exc())
            logger.info(f"cannot compute radar accumulation on tile {tile_name}")
            return
        
        precip_vals = accum["precipitation"].values
        p_nan = numpy.isnan(precip_vals).sum()
        p_total = precip_vals.size
        p_positive = (precip_vals > 0).sum()
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
        tile_radars = self.mapping.get(tile, [])
        for t in timestamps:
            actual.update(k for k in self.rainrates_radar.get(t, {}) if k in tile_radars)
        db = naaulu.radar.get_database()
        accum.attrs["radars"] = sorted(db.get(wsi, {}).get("name", wsi) for wsi in actual)

        return accum
    

    def compute_rainaccum(self, time):
        
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

        # Clear transform cache only for single timestamp (batch mode reuses it)
        if self.start == self.end:
            self.transform.clear()

class Dove(Base):

    def __init__(self, *args, **kwargs):
        Base.__init__(self, *args, **kwargs)
        self.method = "dove"
        self.variables = ["DBZH"]
        self.min_angle = 0
        self.max_angle = 0.5

    def compute_rainrate_radar(self, time, wsi):

        if self.volumes[time][wsi] is None:
            return

        volume = self.volumes[time][wsi]
        keys = xradar.util.get_sweep_keys(volume)
        if not keys:
            logger.debug(f"{self.method} {time} {wsi}: no sweep keys in volume")
            return

        rainrates = []
        for key in keys:
            sweep = volume[key].ds.copy()
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
            return
        
        return rainrates       

class Eider(Base):

    def __init__(self, *args, **kwargs):
        Base.__init__(self, *args, **kwargs)
        self.method = "eider"
        self.variables = ["DBZH"]
        self.min_angle = 0
        self.max_angle = 8

    def compute_rainrate_radar(self, time, wsi):

        volume = self.volumes[time][wsi]
        if volume is None:
            return

        site_alt = float(volume.ds.altitude.values)

        et_sweep = naaulu.radar.echotop(volume)
        if et_sweep is None:
            max_alt = 3000.0
        else:
            max_alt = float(numpy.nanmedian(et_sweep.echotop.values))
            if not numpy.isfinite(max_alt):
                max_alt = 3000.0
        max_alt = max(max_alt, 2000.0)

        volume = naaulu.radar.add_quality(
            volume, name="quality_filter_window_distance",
            fun=naaulu.radar.quality_filter_window_distance,
            long_name="filter_window_distance clutter quality index",
        )
        volume = naaulu.radar.add_quality(
            volume, name="quality_distance",
            fun=naaulu.radar.quality_distance,
            long_name="range quality indicator",
        )
        volume = naaulu.radar.add_quality(
            volume, name="quality_height",
            fun=naaulu.radar.quality_height,
            fun_kwargs={"site_alt": site_alt, "max_alt": max_alt},
            long_name=f"height quality index (1 at 0m, 0 at {max_alt:.0f}m)",
        )
        volume = naaulu.radar.add_quality(
            volume, name="quality",
            fun=lambda s: s.quality_filter_window_distance * s.quality_distance * s.quality_height,
            long_name="aggregated quality (product of filter_window_distance, distance, height)",
        )

        sweep_keys = list(volume.ds.sweep_group_name.values)
        fixed_angles = volume.ds.sweep_fixed_angle.values
        min_angle = min(fixed_angles)
        base_keys = [k for i, k in enumerate(sweep_keys) if fixed_angles[i] == min_angle]

        rainrates = []
        for base_key in base_keys:
            sweep = naaulu.radar.fill_gap_nearest_elevation(
                volume, base_key=base_key,
            )
            if sweep is None:
                continue

            z = sweep["DBZH"].wrl.trafo.idecibel()
            sweep["rainrate"] = z.wrl.zr.z_to_r_enhanced()[0]
            sweep["rainrate"].attrs = {
                "standard_name": "rainfall_rate",
                "long_name": "rainfall_rate",
                "units": "mm h-1",
                "ancillary_variables": "quality",
            }
            rainrates.append(sweep.get(["rainrate", "quality"]))

        return rainrates
    
class Fulmar(Base):

    def __init__(self, *args, **kwargs):
        Base.__init__(self, *args, **kwargs)
        self.method = "fulmar"
        self.variables = ["DBZH", "VRADH", "PHIDP", "RHOHV", "ZDR"]

class Gadwall(Base):

    def __init__(self, *args, **kwargs):
        Base.__init__(self, *args, **kwargs)
        self.method = "gadwall"
        self.variables = ["DBZH", "VRADH", "PHIDP", "RHOHV", "ZDR"]
