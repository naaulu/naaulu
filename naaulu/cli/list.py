#!/usr/bin/env python

"""List available gauge network(s) and radar(s)"""

import argparse
import json

import naaulu.config
import naaulu.gauge
import naaulu.radar
import naaulu.util


def main():
    parser = argparse.ArgumentParser(description="List available networks and radars")
    naaulu.config.add_system_args(parser)

    subparsers = parser.add_subparsers(dest="subcommand", required=True, help="subcommand to run")

    # network subcommand
    net_parser = subparsers.add_parser("network", help="list gauge networks (filtered by area)")
    naaulu.config.add_spatial_args(net_parser)
    net_parser.add_argument("--organisation", "--org", help="filter by organisation (e.g. vmm, fmi)")
    net_parser.add_argument("--duration", help="filter by ISO duration (e.g. PT1H, PT10M)")
    net_parser.add_argument("--weather", action="store_const", const=True, default=None, help="weather/realtime only")
    net_parser.add_argument("--climat", action="store_const", const=True, default=None, help="climat/historical only")

    # radar subcommand
    rad_parser = subparsers.add_parser("radar", help="list radars with actual providers (overlapping area)")
    naaulu.config.add_spatial_args(rad_parser)
    rad_parser.add_argument("--provider", help="filter by provider (e.g. opera, dwd)")
    rad_parser.add_argument(
        "--distance",
        help="radar coverage radius (e.g. 150km, 150E3)",
        default="150km",
    )

    args = parser.parse_args()
    naaulu.config.setup_logging(args.log or "info")
    naaulu.config.set_archive_dir(disabled=True)

    # parse optional spatial (same logic as plot/verify)
    geometry = None
    country_filter = getattr(args, "country", None)
    try:
        area_name, _, geometry = naaulu.config.parse_area(args)
        if not country_filter and len(area_name) == 2:
            country_filter = area_name
    except ValueError:
        pass  # no spatial filter provided

    if args.subcommand == "network":
        duration = None
        if args.duration:
            try:
                duration = naaulu.util.parse_duration(args.duration)
            except Exception:
                print(f"Invalid duration: {args.duration}")
                return

        networks = naaulu.gauge.list_networks(
            country=country_filter,
            organisation=args.organisation,
            duration=duration,
            weather=args.weather,
            climat=args.climat,
        )
        # For geometry, further dynamic filter would require per-network station check (omitted for speed)

        print(f"Available gauge networks ({len(networks)}):")
        print(f"{'ORG/NET':<15} {'DUR':<6} {'COUNTRY':<8} {'REGION':<12}")
        print("-" * 44)
        for n in networks:
            orgnet = f"{n['organisation']}/{n['network']}"
            print(f"{orgnet:<15} {n['temporal_resolution']:<6} {(n.get('country') or '---'):<8} {(n.get('region') or ''):<12}")

    elif args.subcommand == "radar":
        db = naaulu.radar.get_database()
        radars = db

        distance = None
        if geometry is not None:
            import datetime
            distance = naaulu.util.parse_distance(args.distance)
            now = datetime.datetime.now()
            try:
                overlapping = list(naaulu.radar.select(
                    start=now - datetime.timedelta(hours=6),
                    end=now,
                    geom=geometry,
                    distance=distance,
                    key=None,
                ))
                radars = {k: v for k, v in db.items() if k in overlapping}
            except Exception as e:
                print(f"Note: spatial filter failed ({e}), showing all")

        total = len(radars)
        print(f"Available radar providers ({total}):")
        print(f"{'WSI':<16} {'NAME':<12} {'PROVIDER':<8} {'LON':>8} {'LAT':>6} {'DIST':>4} {'START':<10} {'END':<10}")
        print("-" * 81)
        for wsi, r in sorted(radars.items()):
            prov = r.get("provider")
            if prov:
                name = (r.get("name") or "")[:12]
                lon = r.get("longitude")
                lat = r.get("latitude")
                start = r.get("start") or ""
                end = r.get("end") or ""
                if geometry is not None:
                    dist_m = naaulu.geography.radar_distance_to_geometry(
                        location=(lon, lat), geometry=geometry
                    )
                    dist = dist_m / 1000  # km
                    print(f"{wsi:<16} {name:<12} {prov:<8} {lon:>8.2f} {lat:>6.2f} {dist:>4.0f} {start:<10} {end:<10}")
                else:
                    print(f"{wsi:<16} {name:<12} {prov:<8} {lon:>8.2f} {lat:>6.2f} {'-':>4} {start:<10} {end:<10}")


if __name__ == "__main__":
    main()
