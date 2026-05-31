#!/usr/bin/env python

"""Merge partial rainfall estimates into final products"""

import argparse
import sys
import logging
import traceback

import naaulu.config
import naaulu.precip

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Rainfall estimation")

    naaulu.config.add_time_args(parser)
    naaulu.config.add_spatial_args(parser)
    naaulu.config.add_system_args(parser)
    naaulu.config.add_product_args(parser)

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    naaulu.config.setup_logging(args.log)
    naaulu.config.set_archive_dir(disabled=args.no_archive)

    times, step = naaulu.config.parse_times(args)
    area, chunk, geometry  = naaulu.config.parse_area(args)
    duration, resolution, organisation, product = naaulu.config.parse_product(args)

    tiles = naaulu.geography.chunk_geometry(
        geometry=geometry,
        chunk_width = chunk,
        chunk_height = chunk,
        )

    for time in times:
        for tile in tiles:
            try:
                dataset = naaulu.precip.combine(
                    time=time,
                    tile=tile,
                    duration=duration,
                    resolution = resolution,
                    organisation= organisation,
                    product= product
                )
                naaulu.precip.save(dataset)
            except Exception as e:
                logger.debug(traceback.format_exc())
                logger.warning(
                    f"failed to combine dataset at {time} on tile {tile.bounds}: {e}"
                )
                continue


if __name__ == "__main__":
    main()
