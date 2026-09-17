import datetime

import numpy as np
import pytest

from naaulu import radar


@pytest.mark.parametrize("wsi", [
    "0-20010-0-06482",  # Wideumont, BEL
    "0-21010-0-42",     # Sürgavere, EST
    "0-20010-0-07005",  # Abbeville, FRA
    "0-20010-0-10132",  # BOO, DEU
    "0-20010-0-02954",  # Anjalankoski, FIN
    "0-20010-0-06356",  # Herwijnen, NLD
    "0-20000-0-11480",  # Brdy-Praha, CZE
    "0-21010-0-190",    # Aberdeen, USA
])
@pytest.mark.network
def test_get_volume_realtime(wsi):
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    time = (now - datetime.timedelta(hours=2)).replace(minute=0, second=0, microsecond=0)
    duration = datetime.timedelta(hours=1)

    try:
        volume = radar.get_volume(
            time=time,
            duration=duration,
            wsi=wsi,
            min_angle=0,
            max_angle=90,
        )
    except (RuntimeError, ImportError):
        pytest.skip("no data available for this radar at this time")

    sweep_keys = list(volume.ds.sweep_group_name.values)
    assert len(sweep_keys) > 1, f"expected > 1 sweep, got {len(sweep_keys)}"

    all_times = []
    for key in sweep_keys:
        t = volume[key].ds.time.values
        all_times.extend([t.min(), t.max()])

    all_times = np.array(all_times)
    min_time = all_times.min()
    max_time = all_times.max()

    naive_time = time.replace(tzinfo=None)
    naive_start = (time - duration).replace(tzinfo=None)

    assert min_time >= np.datetime64(naive_start), \
        f"earliest time {min_time} is before {naive_start}"
    assert max_time <= np.datetime64(naive_time), \
        f"latest time {max_time} is after {naive_time}"
