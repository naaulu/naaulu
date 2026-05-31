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
        "--gauge", help="overlay gauge values for a network (org/network); repeatable",
        action="append", default=[],
    )

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    naaulu.config.setup_logging(args.log)
    naaulu.config.set_archive_dir(disabled=args.no_archive)

    logger = logging.getLogger(__name__)
    times, step = naaulu.config.parse_times(args)
    area, chunk, geometry = naaulu.config.parse_area(args)
    duration, resolution, organisation, product = naaulu.config.parse_product(args)

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
            organisation = organisation,
            product = product,
            format = "mp4"
        )
        print(filename)
        
        container, stream = naaulu.visual.create_movie(filename)        
    
    for time in times:

        datasets = []
        for tile in tiles:
            try:
                dataset = naaulu.precip.combine(
                    time=time,
                    tile=tile,
                    duration=duration,
                    resolution=resolution,
                    organisation=organisation,
                    product=product,
                )
                datasets.append(dataset)
            except naaulu.errors.NoDataError:
                logger.info(f"no data at time {time} for tile {tile}")
            except Exception:
                logger.debug(traceback.format_exc())
                logger.warning(f"cannot combine product at time {time} for tile {tile}")

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

        naaulu.visual.add_borders(ax=ax, crs=plot_crs, geom=geometry, chunk_width=chunk)
        naaulu.visual.add_axis(ax, geometry.bounds, plot_crs)

        radar_count = None
        counts = [ds.attrs.get("radar_count") for ds in datasets if "radar_count" in ds.attrs]
        if counts:
            import numpy
            radar_count = numpy.mean(counts)

        naaulu.visual.add_title(fig, time, area, duration, resolution, organisation, product, radar_count=radar_count)
        naaulu.visual.add_colorbar(fig, ax, cmap, norm)

        if args.gauge:
            gauge_datasets = {}
            for spec in args.gauge:
                try:
                    org, net = spec.split("/", 1)
                except ValueError:
                    logger.warning(f"ignoring --gauge {spec!r}: expected 'org/network'")
                    continue

                provider_args = {"bbox": geometry.bounds}

                # Choose sensible network for non-US areas
                if org == "iem" and net == "asos" and area.lower() in ("fra", "bel", "france", "belgium"):
                    provider_args["network"] = "FR_ASOS" if area.lower().startswith("fr") else "BE_ASOS"

                try:
                    ds = naaulu.gauge.get(
                        time=time,
                        duration=duration,
                        organisation=org,
                        network=net,
                        provider_args=provider_args,
                    )
                    ds = naaulu.geography.cut(ds, geometry)
                except Exception:
                    logger.debug(traceback.format_exc())
                    logger.warning(f"no gauge data for {spec} at {time} (duration {duration} not available)")
                    continue
                if ds.sizes.get("station", 0) == 0:
                    logger.info(f"{spec} returned no stations inside the plot area at {time}")
                    continue
                gauge_datasets[spec] = ds

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
                    organisation,
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
