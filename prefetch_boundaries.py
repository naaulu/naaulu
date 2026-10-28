#!/usr/bin/env python
"""
Fill the OSM boundary + city caches (~/.local/share/naaulu/boundaries) for a
set of countries from Geofabrik Postpass (SQL over OSM's planet database),
so that later renders start from the baked cache (a missing tier still fills
itself on demand).

Intended for image builds and realtime containers:

    python prefetch_boundaries.py FRA,BEL,DEU
    python prefetch_boundaries.py FRA --levels ADM0,ADM2,CITIES
    python prefetch_boundaries.py FRA --refresh   # re-query cached tiers

A prefetch is a handful of small queries per country (megabytes, never a
planet download); each tier is cached in the data dir and reused. Pair the
result with NAAULU_BOUNDARIES_OFFLINE=1 at runtime so plots only ever read
the baked cache. Repo reference data (naaulu/data/) is a different thing —
that is rebuilt by build_data.py.

Exits non-zero if any requested tier could not be cached (service
unreachable, bad ISO3), so a container build fails loudly instead of
shipping a half-filled cache.

Tiers: ADM0 (countries), ADM2 (provinces), CITIES (place nodes — scoped by
each country's own outline, Natural Earth standing in where OSM has none).
"""

import logging
import os
import sys

# Ensure we can import naaulu even if not installed
sys.path.insert(0, os.path.dirname(__file__))

import naaulu.config
from naaulu import geography


logger = logging.getLogger("prefetch")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Prefetch OSM boundary/city caches from Geofabrik "
                    "Postpass for given countries")
    parser.add_argument(
        "countries",
        help="comma-separated ISO3 codes, e.g. FRA,BEL,DEU",
    )
    parser.add_argument(
        "--levels",
        default="ADM0,ADM2",
        help="tiers to fetch, comma-separated [ADM0,ADM2,CITIES]",
        type=str,
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="re-query Postpass for tiers that are already cached",
    )
    parser.add_argument(
        "--log",
        help="log level [critical, error, warning, info, debug]",
        type=str,
        default="info",
    )
    args = parser.parse_args()

    naaulu.config.setup_logging(args.log)

    iso_codes = [c.strip().upper() for c in args.countries.split(",") if c.strip()]
    levels = tuple(l.strip().upper() for l in args.levels.split(",") if l.strip())

    logger.info(f"prefetching boundaries: {','.join(iso_codes)} ({','.join(levels)})")
    try:
        filled = geography.prefetch_boundaries(
            iso_codes, levels=levels, refresh=args.refresh)
    except RuntimeError as e:
        sys.exit(f"error: {e}")

    for level in levels:
        cached = filled[level]
        print(f"{level}: {len(cached)}/{len(iso_codes)} countries with units "
              f"({', '.join(cached) if cached else 'none'})")
    print(f"boundary cache filled in {naaulu.config.get_data_dir('boundaries')}")


if __name__ == "__main__":
    main()
