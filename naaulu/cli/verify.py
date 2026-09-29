#!/usr/bin/env python

"""Verify rainfall estimates against gauge measurements"""

import argparse
import logging
import os
import sys

logger = logging.getLogger(__name__)

import naaulu.config
import naaulu.gauge
import naaulu.geography
import naaulu.precip
import naaulu.util
import naaulu.verification

import matplotlib.pyplot


def add_network_args(parser):
    """--network is both repeatable and space separated.

    repeatable is what the user types: --network bel --network bel_spw
    space separated is what this repo has always allowed: --network bel deu
    With action="append" and nargs="+" both forms arrive as a list of
    lists, which flatten_networks() then unwraps.
    """
    parser.add_argument(
        "--network",
        help=(
            "gauge network key, repeatable and space separated "
            "[BEL, BEL_SPW, BEL_VMM, DEU, FIN, CZE, EST, FRA]; the three "
            "Belgian keys are disjoint and can be combined"
        ),
        type=str,
        action="append",
        nargs="+",
        required=True,
    )


def flatten_networks(groups):
    """Lowercased network keys from --network, however they were grouped.

    nargs="+" on its own would let a later --network replace an earlier
    one, silently dropping networks from the verification.
    """
    return [name.lower() for group in groups for name in group]


def main():

    parser = argparse.ArgumentParser()
    naaulu.config.add_time_args(parser)
    naaulu.config.add_spatial_args(parser)
    naaulu.config.add_product_args(parser)
    naaulu.config.add_system_args(parser)

    parser.add_argument(
        "--verification",
        help="name of the verification method",
        type=str,
        default="beaver"
    )
    add_network_args(parser)
    parser.add_argument(
        "--availability",
        help="minimal sampling availability (0.0 - 1.0)",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--show", help="show figure interactively",
        action="store_true",
    )


    args = parser.parse_args()

    naaulu.config.setup_logging(args.log)
    naaulu.config.set_archive_dir(disabled=args.no_archive)

    times, step = naaulu.config.parse_times(args)
    area, chunk, geometry = naaulu.config.parse_area(args)
    duration, resolution, product = naaulu.config.parse_product(args)

    countries = flatten_networks(args.network)

    gauges = naaulu.gauge.collect(
        times,
        geometry,
        duration,
        countries,
    )
    
    coords = naaulu.gauge.get_dataset_coordinates(gauges)
    estimates = naaulu.precip.extract(
        times=times,
        codes=gauges.station.values,
        coords=coords,
        chunk=chunk,
        duration= duration,
        resolution= resolution,
        product = product,
    )
    
    verification = args.verification

    fun = getattr(naaulu.verification, verification)
    results, fig = fun(gauges, estimates)

    time_str = "_".join(naaulu.util.format_time(t) for t in times)
    try:
        area_part = naaulu.geography.country_code(area).lower()
    except Exception:
        area_part = area.lower()

    filename = ".".join([
        time_str,
        area_part,
        verification,
        naaulu.util.format_duration(duration),
        naaulu.util.format_distance(resolution),
        product,
        "png",
    ])

    archive = naaulu.config.get_archive_dir()
    if archive is not None:
        root = os.path.join(archive, "verification")
        filepath = naaulu.util.get_path(root, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fig.savefig(filepath)
        logger.info(f"saved {filepath}")

    if args.show:
        matplotlib.pyplot.show()
    else:
        matplotlib.pyplot.close(fig)

if __name__ == "__main__":
    main()
