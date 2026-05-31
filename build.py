#!/usr/bin/env python
"""
Maintainer build script.
Generates radars.json and radars_opera.json in naaulu/data/
Run with: python build.py
"""

import importlib
import json
import logging
import os
import pkgutil
import sys

import requests

# Ensure we can import naaulu even if not installed
sys.path.insert(0, os.path.dirname(__file__))

import naaulu
import naaulu.providers
import naaulu.geography
import naaulu.network


logger = logging.getLogger("build")


def build_radar_database():
    """Build radars.json from WMO WRD and save to package data dir."""
    logger.info("fetching WMO radar database from WRD")

    search_url = "https://wrd.mgm.gov.tr/Radar/Search"
    payload = {
        "draw": 1,
        "start": 0,
        "length": 10000,
        "INSTALL_YEAR_MIN": 1900,
        "INSTALL_YEAR_MAX": 2026,
    }
    resp = requests.post(search_url, data=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    radars = {}
    for item in data.get("data", []):
        name = item.get("RADAR_NAME", "")
        wsi = item.get("WSI") or name.replace(" ", "_")
        if not wsi:
            continue
        start = item.get("INSTALL_DATE")
        if start:
            try:
                start = start[:10]
            except Exception:
                start = None
        try:
            lon = float(item.get("RADAR_LON")) if item.get("RADAR_LON") else None
            lat = float(item.get("RADAR_LAT")) if item.get("RADAR_LAT") else None
        except (TypeError, ValueError):
            lon = lat = None
        raw = item.get("OWNER_SHORT_NAME") or item.get("OWNER_NAME") or ""
        if raw and not item.get("OWNER_SHORT_NAME"):
            parts = [p for p in raw.replace("-", " ").split() if p]
            if len(parts) >= 2:
                caps = "".join(w[0] for w in parts if w and w[0].isupper())
                if caps:
                    raw = caps
        owner = raw.lower().strip().replace(" ", "").replace("-", "").replace("/", "")
        owner = "".join(c for c in __import__("unicodedata").normalize("NFKD", owner) if not __import__("unicodedata").combining(c))
        radars[wsi] = {"lon": lon, "lat": lat, "start": start, "owner": owner, "name": name}

    # providers with radar
    radar_providers = []
    for _, name, _ in pkgutil.iter_modules(naaulu.providers.__path__):
        try:
            mod = importlib.import_module(f"naaulu.providers.{name}")
            if hasattr(mod, "radar"):
                radar_providers.append(name)
        except Exception:
            pass

    filtered = {}
    for wsi, tmp in radars.items():
        prov = tmp["owner"] if tmp["owner"] in radar_providers else None
        if prov:
            filtered[wsi] = {
                "name": tmp["name"],
                "longitude": tmp["lon"],
                "latitude": tmp["lat"],
                "start": tmp["start"],
                "end": None,
                "provider": prov,
            }

    data_dir = os.path.join(os.path.dirname(__file__), "naaulu", "data")
    os.makedirs(data_dir, exist_ok=True)
    out = os.path.join(data_dir, "radars.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2)
    logger.info(f"wrote {out} ({len(filtered)} entries)")
    return filtered


def build_opera_database():
    """Fetch and build the OPERA radar database, save to data/radars_opera.json."""
    logger.info("fetching OPERA radar database")
    filename = "OPERA_RADARS_DB.json"
    url = "https://www.eumetnet.eu/wp-content/themes/aeron-child/observations-programme/current-activities/opera/database/OPERA_Database"
    url = f"{url}/{filename}"
    downloaded_path = naaulu.network.download(url)
    with open(downloaded_path, "r", encoding="utf-8") as f:
        radars = json.load(f)
    db = {}
    for radar in radars:
        if radar.get("wigosid", "") != "":
            db[radar["wigosid"]] = radar

    data_dir = os.path.join(os.path.dirname(__file__), "naaulu", "data")
    os.makedirs(data_dir, exist_ok=True)
    out = os.path.join(data_dir, "radars_opera.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)
    logger.info(f"wrote {out} ({len(db)} entries)")
    return db


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build reference data into repo")
    parser.add_argument(
        "--log",
        help="log level [critical, error, warning, info, debug]",
        type=str,
        default="info",
    )
    args = parser.parse_args()

    naaulu.config.setup_logging(args.log)

    build_radar_database()
    build_opera_database()

    print("loading country boundaries...")
    countries = naaulu.geography._load_countries()
    print(f"  {len(countries)} countries")

    print("reference data built and installed")


if __name__ == "__main__":
    main()
