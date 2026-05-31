#!/usr/bin/env python

"""Export archived data to a custom directory"""

import argparse
import logging
import os
import shutil
import sys

import naaulu.config
import naaulu.util

logger = logging.getLogger(__name__)

TYPE_SUBDIRS = {
    "precip": "precip",
    "radar": "radar",
    "gauge": "gauge",
    "visual": "visual",
}


def main():
    parser = argparse.ArgumentParser(description="Export archived data to a custom directory")

    naaulu.config.add_time_args(parser)
    naaulu.config.add_system_args(parser)

    parser.add_argument(
        "--to",
        required=True,
        help="destination directory",
    )
    parser.add_argument(
        "--precip",
        action="store_true",
        help="export precipitation products",
    )
    parser.add_argument(
        "--radar",
        action="store_true",
        help="export radar products",
    )
    parser.add_argument(
        "--gauge",
        action="store_true",
        help="export gauge products",
    )
    parser.add_argument(
        "--visual",
        action="store_true",
        help="export visual products",
    )

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    naaulu.config.setup_logging(args.log)

    archive_dir = naaulu.config.get_archive_dir()
    if archive_dir is None:
        logger.error("archive is disabled")
        sys.exit(1)

    if not os.path.exists(archive_dir):
        logger.error(f"archive directory does not exist: {archive_dir}")
        sys.exit(1)

    times, step = naaulu.config.parse_times(args)

    selected = []
    for key in TYPE_SUBDIRS:
        if getattr(args, key):
            selected.append(key)

    if not selected:
        logger.error("select at least one data type: --precip --radar --gauge --visual")
        sys.exit(1)

    dest = os.path.abspath(args.to)
    os.makedirs(dest, exist_ok=True)

    for data_type in selected:
        subdir = TYPE_SUBDIRS[data_type]
        src_root = os.path.join(archive_dir, subdir)
        if not os.path.exists(src_root):
            logger.info(f"no {subdir} data in archive, skipping")
            continue

        extensions = (".png", ".mp4") if data_type == "visual" else (".nc",)

        for time in times:
            time_str = naaulu.util.format_time(time)
            year = time_str[0:4]
            month = time_str[4:6]
            day = time_str[6:8]
            src_dir = os.path.join(src_root, year, month, day)
            dest_dir = os.path.join(dest, subdir, year, month, day)

            if not os.path.exists(src_dir):
                continue

            for root, dirs, files in os.walk(src_dir):
                for fname in files:
                    if any(fname.endswith(ext) for ext in extensions):
                        src_file = os.path.join(root, fname)
                        rel = os.path.relpath(root, src_root)
                        dst_path = os.path.join(dest, subdir, rel, fname)
                        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                        shutil.copy2(src_file, dst_path)
                        logger.info(f"copied {src_file} -> {dst_path}")

    logger.info(f"export completed to {dest}")


if __name__ == "__main__":
    main()
