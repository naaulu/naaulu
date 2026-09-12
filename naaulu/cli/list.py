#!/usr/bin/env python

"""List available radar(s)"""

import argparse

import naaulu.config
import naaulu.geography
import naaulu.radar
import naaulu.util


def main():
    parser = argparse.ArgumentParser(description="List available radars")
    naaulu.config.add_system_args(parser)

    naaulu.config.add_spatial_args(parser)
    parser.add_argument(
        "--distance",
        help="radar coverage radius (e.g. 150km, 150E3)",
        default="150km",
    )

    args = parser.parse_args()
    naaulu.config.setup_logging(args.log or "info")
    naaulu.config.set_archive_dir(disabled=True)

    geometry = None
    country_filter = getattr(args, "country", None)
    try:
        area_name, _, geometry = naaulu.config.parse_area(args)
        if not country_filter and len(area_name) == 2:
            country_filter = area_name
    except ValueError:
        pass

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
    print(f"{'WSI':<16} {'NAME':<12} {'COUNTRY':<8} {'LON':>8} {'LAT':>6} {'DIST':>4} {'START':<10} {'END':<10}")
    print("-" * 81)
    for wsi, r in sorted(radars.items()):
        country = r.get("country")
        if country:
            name = (r.get("name") or "")[:12]
            lon = r.get("lon")
            lat = r.get("lat")
            start = r.get("start") or ""
            end = r.get("end") or ""
            lon_str = f"{lon:>8.2f}" if lon is not None else f"{'---':>8}"
            lat_str = f"{lat:>6.2f}" if lat is not None else f"{'--':>6}"
            if geometry is not None:
                dist_m = naaulu.geography.radar_distance_to_geometry(
                    location=(lon, lat), geometry=geometry
                )
                dist = dist_m / 1000
                print(f"{wsi:<16} {name:<12} {country:<8} {lon_str} {lat_str} {dist:>4.0f} {start:<10} {end:<10}")
            else:
                print(f"{wsi:<16} {name:<12} {country:<8} {lon_str} {lat_str} {'-':>4} {start:<10} {end:<10}")


if __name__ == "__main__":
    main()
