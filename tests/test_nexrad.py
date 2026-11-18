"""Offline tests for the NEXRAD (USA) provider's volume selection."""

import datetime

import pytest

from naaulu import radar
from test_radar_aus import make_volume

BUCKET = "unidata-nexrad-level2"
ICAO = "KESX"

# time window used by the tests: [07:55, 08:00]
TIME = datetime.datetime(2026, 10, 5, 8, 0)
DURATION = datetime.timedelta(minutes=5)

LISTING = {
    "2026/10/04/KESX/": [
        "2026/10/04/KESX/KESX20261004_235500_V06",        # too early, only a fallback
    ],
    "2026/10/05/KESX/": [
        "2026/10/05/KESX/KESX20261005_075000_V06",        # before the window
        "2026/10/05/KESX/KESX20261005_075500_V06",        # window start
        "2026/10/05/KESX/KESX20261005_080000_V06",        # window end
        "2026/10/05/KESX/KESX20261005_080500_V06",        # after the window
        "2026/10/05/KESX/KESX20261005_081000_V06.gz",     # not a volume
        "2026/10/05/KESX/KESX20261005_081500_V06.txt",    # not a volume
        "2026/10/05/KESX/OTHER20261005_080000_V06",       # another radar
        "2026/10/05/KESX/KESX_INDEX.txt",                 # not a volume
    ],
    "2026/10/06/KESX/": [],
}

# the last volume before the window is a fallback, then everything inside it
EXPECTED = [
    "2026/10/05/KESX/KESX20261005_075000_V06",
    "2026/10/05/KESX/KESX20261005_075500_V06",
    "2026/10/05/KESX/KESX20261005_080000_V06",
]


@pytest.fixture
def stub_s3(monkeypatch):
    listed, downloaded = [], []

    def fake_list(bucket, prefix, suffix=None, max_keys=1000):
        assert bucket == BUCKET
        listed.append(prefix)
        return LISTING.get(prefix, [])

    def fake_download(s3_url, dest_path=None):
        downloaded.append(s3_url)
        return f"/tmp/{s3_url.rsplit('/', 1)[-1]}"

    monkeypatch.setattr(radar.naaulu.network, "list_s3_objects", fake_list)
    monkeypatch.setattr(radar.naaulu.network, "download_s3_file", fake_download)
    # keep the stub simple
    monkeypatch.setattr(radar, "_usa_clean_sweep", lambda ds: ds.copy(deep=False))
    monkeypatch.setattr(
        radar.xradar.io,
        "open_nexradlevel2_datatree",
        lambda path: make_volume(["2026-10-05T08:00:00"]),
    )
    return listed, downloaded


def test_nexrad_selects_volumes_and_downloads_bucket_keys(stub_s3):
    listed, downloaded = stub_s3

    volumes = radar._nexrad(TIME, DURATION, ICAO)

    # day-1, day and day+1 prefixes are probed
    assert listed == ["2026/10/04/KESX/", "2026/10/05/KESX/", "2026/10/06/KESX/"]
    # bucket/key pairs, volumes only
    assert downloaded == [f"{BUCKET}/{key}" for key in EXPECTED]
    assert len(volumes) == len(EXPECTED)
