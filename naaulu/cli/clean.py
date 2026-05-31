#!/usr/bin/env python

"""Clean cache (incl. archive), temp, and reference data directories"""

import argparse
import logging
import os
import shutil
import sys
import tempfile

import naaulu.config

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Clean naaulu cache (incl. archive), temp, and data directories")

    parser.add_argument(
        "--log",
        help="log level [critical, error, warning, info, debug]",
        type=str,
        default="warning",
    )
    parser.add_argument(
        "--cache",
        help="clean the cache directory",
        action="store_true",
    )
    parser.add_argument(
        "--temp",
        help="clean the temp directory",
        action="store_true",
    )
    parser.add_argument(
        "--all",
        help="clean cache (incl. archive), temp, and data directories",
        action="store_true",
    )
    parser.add_argument(
        "--data",
        help="clean the reference data directory",
        action="store_true",
    )

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    naaulu.config.setup_logging(args.log)

    if args.all:
        args.cache = True
        args.temp = True
        args.data = True

    if args.cache:
        cache_dir = naaulu.config.get_cache_dir()
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            logger.info(f"cleaned cache directory: {cache_dir}")
        else:
            logger.info(f"cache directory does not exist: {cache_dir}")

    if args.temp:
        if naaulu.config.TEMP_DIR and os.path.exists(naaulu.config.TEMP_DIR):
            shutil.rmtree(naaulu.config.TEMP_DIR)
            logger.info(f"cleaned temp directory: {naaulu.config.TEMP_DIR}")

        system_temp = tempfile.gettempdir()
        debug_temp = os.path.join(system_temp, "naaulu")
        if os.path.exists(debug_temp):
            shutil.rmtree(debug_temp, ignore_errors=True)
            logger.info(f"cleaned debug temp directory: {debug_temp}")

        for entry in os.listdir(system_temp):
            if entry.startswith("naaulu_"):
                path = os.path.join(system_temp, entry)
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                    logger.info(f"cleaned temp directory: {path}")

    if args.data:
        data_dir = naaulu.config.get_data_dir()
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)
            logger.info(f"cleaned data directory: {data_dir}")
        else:
            logger.info(f"data directory does not exist: {data_dir}")


if __name__ == "__main__":
    main()
