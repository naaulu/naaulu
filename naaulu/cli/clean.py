#!/usr/bin/env python

"""Clean cache directories (main, download, archive)"""

import argparse
import logging
import os
import shutil
import sys

import naaulu.config

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Clean naaulu cache directories")

    parser.add_argument(
        "--log",
        help="log level [critical, error, warning, info, debug]",
        type=str,
        default="warning",
    )
    parser.add_argument(
        "--cache",
        help="clean the main cache directory",
        action="store_true",
    )
    parser.add_argument(
        "--download",
        help="clean the download cache directory",
        action="store_true",
    )
    parser.add_argument(
        "--archive",
        help="clean the archive directory",
        action="store_true",
    )

    args = parser.parse_args()

    naaulu.config.setup_logging(args.log)

    if not args.cache and not args.download and not args.archive:
        args.cache = True
        args.download = True
        args.archive = True

    if args.cache:
        cache_dir = naaulu.config.get_cache_dir()
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            logger.info(f"cleaned main cache directory: {cache_dir}")
        else:
            logger.info(f"main cache directory does not exist: {cache_dir}")

    if args.download:
        download_dir = naaulu.config.get_download_dir()
        if os.path.exists(download_dir):
            shutil.rmtree(download_dir)
            logger.info(f"cleaned download cache directory: {download_dir}")
        else:
            logger.info(f"download cache directory does not exist: {download_dir}")

    if args.archive:
        archive_dir = naaulu.config.get_archive_dir()
        if archive_dir and os.path.exists(archive_dir):
            shutil.rmtree(archive_dir)
            logger.info(f"cleaned archive directory: {archive_dir}")
        else:
            logger.info(f"archive directory does not exist: {archive_dir}")


if __name__ == "__main__":
    main()
