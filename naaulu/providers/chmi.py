import datetime
import logging

import naaulu
import xradar

logger = logging.getLogger(__name__)


_radar_map = {
    "0-20000-0-11718": ("ska", 50),
    "0-20000-0-11480": ("brd", 60),
}


def radar(time, duration, wsi):
    short, code = _radar_map[wsi]
    t = time - duration
    timestamp = t.strftime("%Y%m%d%H%M%S")
    filename = f"T_PAGZ{code}_C_OKPR_{timestamp}.hdf"
    url = f"http://opendata.chmi.cz/meteorology/weather/radar/sites/{short}/vol_z/hdf5/{filename}"
    try:
        filepath = naaulu.network.download(url)
        volume = xradar.io.open_odim_datatree(filepath)
    except Exception as e:
        raise RuntimeError(f"No radar file available for {wsi} at {timestamp}") from e
    return volume
