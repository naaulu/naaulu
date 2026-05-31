import datetime
import requests
import xradar
import naaulu.network
import naaulu.providers.opera

BASE_URL = "https://avaandmed.keskkonnaportaal.ee/api/lists/active"

radar_name = {
    "0-21010-0-42": "SUR",
    "0-21010-0-41": "HAR",
}

def radar(time, duration, wsi):
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    if time.tzinfo is None:
        time = time.replace(tzinfo=datetime.timezone.utc)
    if (now - time).total_seconds() < 86400:
        return naaulu.providers.opera.radar(time, duration, wsi)

    name = radar_name[wsi]
    t = time - duration
    timestamp = t.strftime("%Y%m%d%H%M")
    query = {
        "filter": {
            "and": {
                "children": [
                    {"isEqual": {"field": "RadarStation", "value": name}},
                    {"isEqual": {"field": "RadarDataType", "value": "VOL"}},
                    {"contains": {"field": "RMTitle", "value": timestamp}},
                ]
            }
        },
        "limit": 1,
    }

    response = requests.post(
        f"{BASE_URL}/items/query",
        json=query,
        headers={"Accept": "application/json"},
    )
    response.raise_for_status()
    data = response.json()

    if not data.get("documents"):
        raise RuntimeError(f"No radar file available for {name} at {timestamp}")

    doc = data["documents"][0]
    doc_id = doc["id"]
    filename = f"{name}.{timestamp}.VOL.h5"
    download_url = f"{BASE_URL}/items/{doc_id}/files/0"
    filepath = naaulu.network.download(download_url, filename)
    volume = xradar.io.open_odim_datatree(filepath)
    return volume


def aws(time, duration, **kwargs):
    if duration not in (datetime.timedelta(minutes=10), datetime.timedelta(hours=1)):
        raise ValueError(f"Unsupported duration for ESTEA AWS: {duration}")

    import numpy

    # Estonian AWS stations (approx values for demo; replace with real KAIA API query)
    stations = [
        ("26038", 24.4, 59.4, 1.2),  # Harku/Tallinn
        ("26242", 26.5, 58.3, 0.8),  # Tartu
        ("26247", 24.5, 58.4, 2.1),  # Pärnu
        ("26346", 27.0, 57.8, 0.5),  # Võru
    ]

    codes = [s[0] for s in stations]
    lons = [s[1] for s in stations]
    lats = [s[2] for s in stations]
    values = [s[3] for s in stations]  # mm accumulation

    coords = numpy.column_stack((lons, lats))
    return (values, codes, coords)
