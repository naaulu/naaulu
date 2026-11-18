"""Offline tests for the NIMBUS ACRR (1-hour accumulation) product."""

import datetime

import h5py
import numpy
import pytest

import naaulu.precip
from naaulu import radar


PROJDEF = (
    b"+proj=laea +lat_0=55.0 +lon_0=10.0 +x_0=1950000.0 "
    b"+y_0=-2100000.0 +units=m +ellps=WGS84"
)
NODATA = -9999000.0
UNDETECT = -8888000.0

# constant light rain, one dry cell, one uncovered cell, one downpour
DATA = numpy.full((30, 40), 1.5)
DATA[1, 0] = UNDETECT
DATA[0, 0] = NODATA
DATA[15, 20] = 42.5


def write_acrr(path):
    """Minimal ODIM composite: axis-aligned LAEA grid at origin (0, 0)."""
    import pyproj

    xscale = yscale = 2000.0
    xsize, ysize = DATA.shape[1], DATA.shape[0]
    width, height = xsize * xscale, ysize * yscale

    crs = pyproj.CRS.from_proj4(PROJDEF.decode())
    fwd = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    corners = {
        "LL": (0.0, -height),
        "LR": (width, -height),
        "UL": (0.0, 0.0),
        "UR": (width, 0.0),
    }

    with h5py.File(path, "w") as f:
        where = f.create_group("where")
        where.attrs["xsize"] = xsize
        where.attrs["ysize"] = ysize
        where.attrs["xscale"] = xscale
        where.attrs["yscale"] = yscale
        where.attrs["projdef"] = PROJDEF
        for name, (x, y) in corners.items():
            lon, lat = fwd.transform(x, y)
            where.attrs[f"{name}_lon"] = float(lon)
            where.attrs[f"{name}_lat"] = float(lat)

        data1 = f.create_group("dataset1").create_group("data1")
        data1.create_dataset("data", data=DATA)
        what = data1.create_group("what")
        what.attrs["quantity"] = b"ACRR"
        what.attrs["gain"] = 1.0
        what.attrs["offset"] = 0.0
        what.attrs["nodata"] = NODATA
        what.attrs["undetect"] = UNDETECT


@pytest.fixture
def stub_bucket(monkeypatch, tmp_path):
    """Serve a synthetic ACRR file for every requested key."""
    filepath = tmp_path / "OPERA@fake@0@ACRR.h5"
    write_acrr(str(filepath))
    calls = []

    def fake_download(key):
        calls.append(key)
        return str(filepath)

    monkeypatch.setattr(radar, "_s3_download", fake_download)
    return calls


def test_stamp_snapped_to_publication_grid(stub_bucket):
    naaulu.precip.get_nimbus(datetime.datetime(2026, 11, 16, 16, 7))
    assert stub_bucket == [
        "2026/11/16/OPERA/COMP/OPERA@20261116T1600@0@ACRR.h5"
    ]


def test_stamp_on_grid_passthrough(stub_bucket):
    naaulu.precip.get_nimbus(datetime.datetime(2026, 11, 16, 16, 45))
    assert stub_bucket[0].endswith("OPERA@20261116T1645@0@ACRR.h5")


def test_values_scaled_and_masked(stub_bucket):
    ds = naaulu.precip.get_nimbus(datetime.datetime(2026, 11, 16, 16, 0))
    values = ds["precipitation"].values

    # bbox corners fall outside the rotated source rectangle: NaN there is
    # correct, so compare on the valid subset
    assert numpy.nanmax(values) == 42.5                    # downpour survived
    assert numpy.isnan(values).any()                       # outside + nodata
    assert (values == 0.0).any()                           # undetect -> zero
    assert numpy.nanmin(values) >= 0.0                     # no sentinels leaked
    valid = values[~numpy.isnan(values)]
    assert (valid > 1.0).sum() > 0                         # the constant rain


def test_grid_is_wgs84_arcsecond(stub_bucket):
    ds = naaulu.precip.get_nimbus(datetime.datetime(2026, 11, 16, 16, 0))
    assert ds.sizes["x"] > 10 and ds.sizes["y"] > 10
    # arcsecond-valued coords (degrees would be single digits)
    assert abs(ds.x.values).min() > 360
    assert "spatial_ref" in ds.coords
    assert "precipitation" in ds.data_vars


def test_hourly_product_only(stub_bucket):
    with pytest.raises(ValueError, match="1-hour product"):
        naaulu.precip.get_nimbus(
            datetime.datetime(2026, 11, 16, 16, 0),
            duration=datetime.timedelta(days=1),
        )


def test_missing_file_raises_clear_error(monkeypatch):
    def fake_download(key):
        raise FileNotFoundError(key)

    monkeypatch.setattr(radar, "_s3_download", fake_download)
    with pytest.raises(FileNotFoundError, match="NIMBUS ACRR not available"):
        naaulu.precip.get_nimbus(datetime.datetime(2026, 11, 16, 16, 0))


def test_registered_in_products():
    assert "nimbus" in naaulu.precip.PRODUCTS
    assert naaulu.precip.PRODUCTS["nimbus"] == [("PT1H", "2km")]
    assert naaulu.precip._PRODUCTS["nimbus"] is naaulu.precip.get_nimbus
