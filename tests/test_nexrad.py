"""Offline tests for the NEXRAD (USA) provider's volume selection.

The S3 listing and download are stubbed: what matters here is which keys are
picked out of the bucket, that only real volume files are considered, and that
the download gets the bucket-prefixed key that network.download_s3_file needs.
"""

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
        "2026/10/05/KESX/KESX20261005_082000_V06.md5",    # not a volume
        "2026/10/05/KESX/KESX20261005_082500_V06.bak",    # not a volume
        "2026/10/05/KESX/OTHER20261005_080000_V06",       # another radar
        "2026/10/05/KESX/KESX_INDEX.txt",                 # not a volume
    ],
    "2026/10/06/KESX/": [],
}

# the last volume before the window is kept as a fallback, then everything
# inside it; the first volume after the window stops the scan
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
    # a real NEXRAD sweep would have 360/720 rays; keep the stub simple
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
    # only genuine volumes, and download_s3_file gets bucket/key
    assert downloaded == [f"{BUCKET}/{key}" for key in EXPECTED]
    assert len(volumes) == len(EXPECTED)


def test_nexrad_ignores_non_volume_keys(stub_s3):
    _, downloaded = stub_s3

    radar._nexrad(TIME, DURATION, ICAO)

    for key in downloaded:
        assert key.endswith("_V06")
        assert "/KESX" in key
    assert not any(key.endswith((".gz", ".txt", ".md5", ".bak")) for key in downloaded)
    assert not any("OTHER" in key for key in downloaded)


def test_nexrad_raises_without_volumes(monkeypatch):
    monkeypatch.setattr(
        radar.naaulu.network, "list_s3_objects",
        lambda bucket, prefix, suffix=None, max_keys=1000: [],
    )

    with pytest.raises(FileNotFoundError, match=ICAO):
        radar._nexrad(TIME, DURATION, ICAO)


def test_nexrad_skips_an_unreadable_volume(monkeypatch):
    """One unparsable file must not cost the whole radar (mixed ray counts)."""
    monkeypatch.setattr(
        radar.naaulu.network, "list_s3_objects",
        lambda bucket, prefix, suffix=None, max_keys=1000: LISTING.get(prefix, []),
    )
    monkeypatch.setattr(
        radar.naaulu.network, "download_s3_file",
        lambda s3_url, dest_path=None: s3_url,
    )
    monkeypatch.setattr(radar, "_usa_clean_sweep", lambda ds: ds.copy(deep=False))

    opened = []

    def open_volume(path):
        opened.append(path)
        if len(opened) == 1:
            raise ValueError(
                "conflicting sizes for dimension 'azimuth': length 360 on "
                "'azimuth' and length 720 on DBZH"
            )
        return make_volume(["2026-10-05T08:00:00"])

    monkeypatch.setattr(radar.xradar.io, "open_nexradlevel2_datatree", open_volume)

    volumes = radar._nexrad(TIME, DURATION, ICAO)

    assert len(opened) == len(EXPECTED)          # every candidate was attempted
    assert len(volumes) == len(EXPECTED) - 1     # only the broken one was lost


def test_nexrad_survives_a_failing_day_listing(monkeypatch):
    """One unreachable day must not sink the whole fetch."""
    calls = []

    def flaky(bucket, prefix, suffix=None, max_keys=1000):
        calls.append(prefix)
        if prefix.startswith("2026/10/04"):
            raise ConnectionError("day before unreachable")
        return LISTING.get(prefix, [])

    downloaded = []
    monkeypatch.setattr(radar.naaulu.network, "list_s3_objects", flaky)
    monkeypatch.setattr(
        radar.naaulu.network, "download_s3_file",
        lambda s3_url, dest_path=None: downloaded.append(s3_url) or f"/tmp/{s3_url}",
    )
    monkeypatch.setattr(radar, "_usa_clean_sweep", lambda ds: ds.copy(deep=False))
    monkeypatch.setattr(
        radar.xradar.io, "open_nexradlevel2_datatree",
        lambda path: make_volume(["2026-10-05T08:00:00"]),
    )

    volumes = radar._nexrad(TIME, DURATION, ICAO)

    assert len(calls) == 3
    assert len(volumes) == len(EXPECTED)


@pytest.mark.network
def test_nexrad_listing_real_bucket():
    """The Unidata bucket is near real time, so a pinned date rots.

    Today is tried first, then the two days before it, because a run just
    after midnight can find today's directory still empty.
    """
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    volumes = []
    for day_offset in (0, 1, 2):
        day = now - datetime.timedelta(days=day_offset)
        keys = radar.naaulu.network.list_s3_objects(
            BUCKET, prefix=f"{day.strftime('%Y/%m/%d')}/{ICAO}/"
        )
        volumes = [k for k in keys if k.rsplit("/", 1)[-1].endswith("_V06")]
        if volumes:
            break

    assert volumes, "no NEXRAD volumes listed for KESX in the last three days"
