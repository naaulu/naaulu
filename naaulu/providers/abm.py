import datetime
import os
import json
import naaulu.config
import naaulu.radar

radar_mapping = None  # global cache


def build_radar_mapping(wigos_db: dict, tolerance: float = 0.01) -> dict:
    url = "https://dapds00.nci.org.au/thredds/catalog/rrqpe/level_1/catalog.xml"
    xml_path = naaulu.network.download(url, "aura_catalog.xml")

    # Parse radar site codes from THREDDS catalog
    from xml.etree import ElementTree as ET
    tree = ET.parse(xml_path)
    root = tree.getroot()

    stations = {}
    for dataset in root.iter("{http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0}dataset"):
        name = dataset.attrib.get("name", "")
        if "_" in name and name.endswith(".vol"):
            icao = name.split("_")[0]
            # You may need to hardcode or fetch lat/lon separately
            # For now, assume you have a lookup table or external source
            if icao not in stations:
                stations[icao] = {"lat": None, "lon": None}  # Fill later

    # TODO: Replace with actual lat/lon for each ICAO
    # You can scrape BoM radar site info or use a CSV if available

    mapping = {}
    for wigos_id, meta in wigos_db.items():
        try:
            lat_wigos = float(meta["latitude"])
            lon_wigos = float(meta["longitude"])
        except (KeyError, ValueError):
            continue

        for icao, site in stations.items():
            if site["lat"] is None or site["lon"] is None:
                continue
            if abs(lat_wigos - site["lat"]) < tolerance and abs(lon_wigos - site["lon"]) < tolerance:
                mapping[wigos_id] = icao
                break

    codes_path = os.path.join(os.path.dirname(__file__), "radar_aus_codes.json")
    with open(codes_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)

    return mapping


def radar(time, duration, wsi):
    icao = wsi
    date = time.strftime("%Y/%m/%d")
    base_url = "https://dapds00.nci.org.au/thredds/fileServer/rrqpe/level_1"
    path = f"{base_url}/{date}/{icao}/{icao}_{time.strftime('%Y%m%d_%H%M%S')}.vol"
    return path