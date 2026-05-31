#!/usr/bin/env python

"""Verify rainfall estimates against gauge measurements"""

import argparse
import sys

import naaulu.gauge
import naaulu.precip
import naaulu.verification


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
        default="flea"
    )
    parser.add_argument(
        "--gauge",
        help="list of gauge network [ORGANISATION/NETWORK]",
        type=str,
        nargs='+',
        default=None
    )
    parser.add_argument(
        "--availability",
        help="minimal sampling availability (0.0 - 1.0)",
        type=float,
        default=1.0,
    )


    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    naaulu.config.setup_logging(args.log)
    naaulu.config.set_archive_dir(disabled=args.no_archive)

    times, step = naaulu.config.parse_time(args)
    area, chunk, geometry = naaulu.config.parse_area(args)
    duration, resolution, organisation, product = naaulu.config.parse_product(args)

    networks = []
    if args.gauge is None:
        countries = naaulu.geography.get_countries(
            geom=geometry,
            iso_only=True,
            min_ratio=0.05,
            )
        for country in countries:
            networks.append((country.lower(), "aws"))

    else:        
        for item in args.gauge:
            organisation, network = item.split("/")
            networks.append(organisation.lower(), network.lower())
    
    gauges = naaulu.gauge.collect(
        times,
        geometry,
        duration,
        networks,
    )
    
    coords = naaulu.gauge.get_dataset_coordinates(gauges)
    estimates = naaulu.precip.extract(
        times=times,
        codes=gauges.station.values,
        coords=coords,
        chunk=chunk,
        duration= duration,
        resolution= resolution,
        organisation= organisation,
        product = product,
    )
    
    verification = args.verification

    fun = getattr(naaulu.verification, verification)
    fun(gauges, estimates)

if __name__ == "__main__":
    main()
