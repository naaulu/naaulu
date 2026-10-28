#!/usr/bin/env python

"""Rainfall estimation from radar data to accumulation products"""

import argparse
import datetime
import logging
from time import sleep

import naaulu.config
import naaulu.estimation
import naaulu.multi
import naaulu.network
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
    parser.add_argument(
        "--delay",
        help="seconds to wait after each timestamp for the radar volume to arrive [100]",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--radar",
        help="force using specific radars [WSI number]",
        type=str,
        nargs="+",
        default=None
    )
    args = parser.parse_args()

    naaulu.config.setup_logging(args.log)
    naaulu.config.set_archive_dir(disabled=args.no_archive)
    naaulu.multi.set_workers(args.jobs)

    times, step = naaulu.config.parse_times(args)
    area, chunk, geometry = naaulu.config.parse_area(args)
    duration, resolution, product =  naaulu.config.parse_product(args)
    if step != duration:
        raise ValueError("step and duration must be the same for estimation")
    sampling = naaulu.util.parse_duration(args.sampling)
    radars = args.radar

    estimator = getattr(naaulu.estimation, product.capitalize())
    estimator = estimator(
        start=times[0],
        end=times[-1],        
        duration=duration,
        step=sampling,
        geometry=geometry,
        chunk=chunk,
        resolution=resolution,
        radars=radars
    )

    # Radar volume for a nominal timestamp lands late: every frame waits
    # for it (--delay, a plain sleep, no polling). Timestamps already in
    # the past never wait, so backfill runs stay instant.
    wait_for = datetime.timedelta(seconds=args.delay)
    tic = datetime.datetime.now()
    for time in times:
        due = time + wait_for
        now = datetime.datetime.now(datetime.timezone.utc)
        if due > now:
            logger.info(f"waiting {(due - now).total_seconds():.0f}s for radar volume at {time}")
            sleep((due - now).total_seconds())
        try:
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
        except Exception:
            # One bad frame must not kill a long background run.
            logger.exception(f"estimation at {time} failed, continuing with the next timestamp")

        # Kill every pooled connection between timestamps: nothing (sockets,
        # keep-alives, half-open peers) may survive the long wait below —
        # each frame starts with a completely fresh connection state.
        naaulu.network.close_session()
        

if __name__ == "__main__":
    main()
