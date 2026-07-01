#!/usr/bin/env python

"""Generate plots, animations and 3D renders of rainfall data"""

import argparse
import sys
import logging
import traceback

import naaulu.config
import naaulu.errors
import naaulu.gauge
import naaulu.geography
import naaulu.precip
import naaulu.visual


def main():
    parser = argparse.ArgumentParser()

    naaulu.config.add_system_args(parser)
    naaulu.config.add_time_args(parser)
    naaulu.config.add_spatial_args(parser)
    naaulu.config.add_product_args(parser)

    parser.add_argument(
        "--clim", help="colorbar limits", type=float, nargs=2, default=[1, 100]
    )    
    parser.add_argument("--movie", help="create an animation", action="store_true")
    parser.add_argument("--show", help="show 2d figure", action="store_true")
    parser.add_argument("--render", help="render in 3d", action="store_true")
    parser.add_argument(
        "--network", help="overlay gauge values for a country code; repeatable",
        action="append", default=[],
    )
    parser.add_argument(
        "--provinces", help="show admin-1 and admin-2 boundaries",
        action="store_true",
    )

    args = parser.parse_args()

    naaulu.config.setup_logging(args.log)
    naaulu.config.set_archive_dir(disabled=args.no_archive)

    logger = logging.getLogger(__name__)
    times, step = naaulu.config.parse_times(args)
    area, chunk, geometry = naaulu.config.parse_area(args)
    duration, resolution, product = naaulu.config.parse_product(args)

    plot_crs = naaulu.visual.get_plot_crs(geometry.bounds)
    tiles = naaulu.geography.chunk_geometry(
        geometry=geometry,
        chunk_width=chunk,
        chunk_height=chunk,
    )

    if args.movie:
        filename = naaulu.visual.path(
            time = times,
            area = area,
            duration = duration,
            resolution = resolution,
            product = product,
            format = "mp4"
        )
        
        container, stream = naaulu.visual.create_movie(filename)        
    
    for time in times:

        datasets = []
        for tile in tiles:
            try:
                dataset = naaulu.precip.get(
                    time=time,
                    tile=tile,
                    duration=duration,
                    resolution=resolution,
                    product=product,
                )
                datasets.append(dataset)
            except naaulu.errors.NoDataError:
                logger.info(f"no data at time {time} for tile {tile}")
            except Exception:
                logger.debug(traceback.format_exc())
                logger.warning(f"cannot get product at time {time} for tile {tile}")

        if args.render:
            naaulu.visual.render(datasets)
            continue

        fig, ax = naaulu.visual.figure()
        cmap, norm = naaulu.visual.get_cmap(*args.clim)

        if not datasets:
            logger.warning("no precipitation data available — showing empty map with gauges only")
        else:
            for dataset in datasets:
                naaulu.visual.plot(ax, plot_crs, dataset, cmap, norm)

        naaulu.visual.add_borders(ax=ax, crs=plot_crs, geom=geometry, chunk_width=chunk, provinces=args.provinces)
        naaulu.visual.add_axis(ax, geometry.bounds, plot_crs)

        radar_count = None
        counts = [ds.attrs.get("radar_count") for ds in datasets if "radar_count" in ds.attrs]
        if counts:
            import numpy
            radar_count = numpy.mean(counts)

        naaulu.visual.add_title(fig, time, area, duration, resolution, product, radar_count=radar_count)
        naaulu.visual.add_colorbar(fig, ax, cmap, norm)

        if args.network:
            if duration not in naaulu.gauge.SUPPORTED_DURATIONS:
                logger.warning(f"gauge overlay not available for duration {duration}")
            else:
                gauge_datasets = {}
                for country in args.network:
                    try:
                        ds = naaulu.gauge.get_network(
                            time=time,
                            duration=duration,
                            country=country,
                        )
                        ds = naaulu.geography.cut(ds, geometry)
                    except Exception:
                        logger.debug(traceback.format_exc())
                        logger.warning(f"no gauge data for {country} at {time} (duration {duration} not available)")
                        continue
                    if ds.sizes.get("station", 0) == 0:
                        logger.info(f"{country} returned no stations inside the plot area at {time}")
                        continue
                    gauge_datasets[country] = ds

                if gauge_datasets:
                    naaulu.visual.plot_gauges(ax, plot_crs, gauge_datasets, cmap=cmap, norm=norm)
                else:
                    logger.warning("no gauge stations found in area (no data for requested duration)")

        if args.movie:
            naaulu.visual.add_frame(container, fig)
        else:
            if args.show:
                naaulu.visual.show()
            else:
                filename = naaulu.visual.path(
                    time,
                    area,
                    duration,
                    resolution,
                    product,
                    "png"
                )                
                fig.savefig(filename)
                logger.info(f"saved {filename}")

    if args.movie:
        naaulu.visual.close_movie(container)
        logger.info(f"saved {filename}")
    

if __name__ == "__main__":
    main()
