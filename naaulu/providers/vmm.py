import datetime
import logging

import numpy

logger = logging.getLogger(__name__)


def ott(time: datetime.datetime, duration: datetime.timedelta):
    """
    VMM OTT rain gauge network (Flanders, Belgium).
    Real station list (public VMM OTT network).
    """
    if duration not in (datetime.timedelta(minutes=10)):
        raise ValueError(f"Unsupported duration for VMM/OTT: {duration}")

    codes = [s[0] for s in stations]
    longitudes = [s[1] for s in stations]
    latitudes = [s[2] for s in stations]
    values = [s[3] for s in stations]
    coords = numpy.column_stack((longitudes, latitudes))

    logger.debug(f"vmm/ott: returned {len(values)} stations at {time}")
    return (values, codes, coords)
