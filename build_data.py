#!/usr/bin/env python
"""
Maintainer build script.
Rebuilds reference data files in naaulu/data/
Run with: uv run python build_data.py
"""

import logging
import os
import sys

# Ensure we can import naaulu even if not installed
sys.path.insert(0, os.path.dirname(__file__))

import naaulu
import naaulu.config
import naaulu.geography
import naaulu.gauge
import naaulu.radar


logger = logging.getLogger("build")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Rebuild reference data in naaulu/data/")
    parser.add_argument(
        "--log",
        help="log level [critical, error, warning, info, debug]",
        type=str,
        default="info",
    )
    args = parser.parse_args()

    naaulu.config.setup_logging(args.log)

    # Use naaulu/data/ directory for bundled data
    data_dir = os.path.join(os.path.dirname(__file__), "naaulu", "data")
    os.makedirs(data_dir, exist_ok=True)

    naaulu.radar.build_radar_database(data_dir)
    naaulu.radar.build_opera_database(data_dir)
    naaulu.radar.build_usa_database(data_dir)
    naaulu.gauge.build_chmi_aws(data_dir)

    print("loading country boundaries...")
    countries = naaulu.geography._load_countries()
    print(f"  {len(countries)} countries")

    print("reference data rebuilt in naaulu/data/")


if __name__ == "__main__":
    main()
