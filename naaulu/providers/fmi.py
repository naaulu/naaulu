import datetime
import logging
import fmiopendata.wfs
import numpy
import xradar

import naaulu.config
import naaulu.network
import naaulu.radar

import os
import json

import naaulu.providers.opera

logger = logging.getLogger(__name__)


def radar(time, duration, wsi):
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    if time.tzinfo is None:
        time = time.replace(tzinfo=datetime.timezone.utc)
    if (now - time).total_seconds() < 86400:
        return naaulu.providers.opera.radar(time, duration, wsi)

    db = naaulu.providers.opera.get_database()
    if wsi not in db:
        raise ValueError(f"WIGOS ID {wsi} not found")
    node = db[wsi].get("odimcode")
    t = time - duration
    t = t.replace(minute=(t.minute // 5) * 5, second=0, microsecond=0)
    timestamp = t.strftime("%Y%m%d%H%M")
    datedir = t.strftime("%Y/%m/%d")
    filename = f"{timestamp}_{node}_PVOL.h5"
    link = f"http://s3-eu-west-1.amazonaws.com/fmi-opendata-radar-volume-hdf5/{datedir}/{node}/{filename}"
    try:
        local_path = naaulu.network.download(link)
        volume = xradar.io.open_odim_datatree(local_path)
    except Exception:
        raise RuntimeError(f"No radar file available for {node} at {timestamp}")
    return volume


def aws(time, duration):

    if duration == datetime.timedelta(hours=1):
        param = "r_1h"
        result_key = "Precipitation amount"
        to_mm = 1.0
    elif duration == datetime.timedelta(minutes=10):
        param = "ri_10min"
        result_key = "Precipitation intensity"
        # ri_10min is the mean intensity (mm/h) over the past 10 min; multiply by 10/60 to get mm.
        to_mm = 10.0 / 60.0
    else:
        raise ValueError(f"Unsupported duration for FMI AWS: {duration}")

    start_time = time.isoformat(timespec="seconds") + "Z"
    end_time = time.isoformat(timespec="seconds") + "Z"
    args = [
        "bbox=18,55,35,75",
        "starttime=" + start_time,
        "endtime=" + end_time,
        f"parameters={param}",
    ]

    obs = fmiopendata.wfs.download_stored_query(
        "fmi::observations::weather::multipointcoverage",
        args=args,
    )

    meta = obs.location_metadata

    values = []
    codes = []
    longitudes = []
    latitudes = []

    for station, data in obs.data[time].items():
        codes.append(meta[station]["fmisid"])
        lon = meta[station]["longitude"]
        lat = meta[station]["latitude"]
        longitudes.append(lon)
        latitudes.append(lat)
        values.append(data[result_key]["value"] * to_mm)

    coords = numpy.column_stack((longitudes, latitudes))

    return (values, codes, coords)
