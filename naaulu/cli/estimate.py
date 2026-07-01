#!/usr/bin/env python

"""Rainfall estimation from radar data to accumulation products"""

import os
import sys
if sys.platform == "win32":
    os.add_dll_directory(r"C:\Users\egoud\vcpkg\installed\x64-windows\bin")

import argparse
import datetime
import logging

import naaulu.config
import naaulu.estimation
import naaulu.precip
import naaulu.util

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Rainfall estimation",
    )

    naaulu.config.add_system_args(parser)
    naaulu.config.add_time_args(parser)
    naaulu.config.add_spatial_args(parser)
    naaulu.config.add_product_args(parser)
    parser.add_argument(
        "--sampling",
        help="time sampling for estimation [P10D, PT3H, PT5M, ...]",
        type=str,
        default="PT150S"
    )
    args = parser.parse_args()

    naaulu.config.setup_logging(args.log)
    naaulu.config.set_archive_dir(disabled=args.no_archive)

    times, step = naaulu.config.parse_times(args)
    area, chunk, geometry = naaulu.config.parse_area(args)
    duration, resolution, product =  naaulu.config.parse_product(args)
    if step != duration:
        raise ValueError("step and duration must be the same for estimation")
    sampling = naaulu.util.parse_duration(args.sampling)

    estimator = getattr(naaulu.estimation, product.capitalize())
    estimator = estimator(
        start=times[0],
        end=times[-1],        
        duration=duration,
        step=sampling,
        geometry=geometry,
        chunk=chunk,
        resolution=resolution,
    )

    tic = datetime.datetime.now()
    for time in times:
        logger.info(f"building rainfall dataset at {time} for {area} area")
        base_start = time - duration + step
        base_t = base_start
        while base_t <= time:
            estimator.clean(base_t)
            estimator.compute_rainaccum(base_t)
            for tile, rainaccum in estimator.rainaccums[base_t].items():
                if rainaccum is not None:
                    naaulu.precip.save(rainaccum)
            base_t += step
        delay = (datetime.datetime.now() - tic).total_seconds()
        logger.info(f"estimation at {time} completed with delay of {delay} seconds")
        

if __name__ == "__main__":
    main()
