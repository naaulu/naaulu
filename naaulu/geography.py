import logging
import os
import shutil
import textwrap

import numpy
import pycountry
import shapely.geometry
import xarray
import wradlib.georef
import wradlib.ipol

import json
import math

import naaulu.network
import naaulu.config
import naaulu.util

logger = logging.getLogger(__name__)

_country_cache: dict[str, shapely.geometry.MultiPolygon] = {}
_country_tiles_cache: dict[str, dict] = {}
_countries_geojson_cache: dict[str, shapely.geometry.MultiPolygon] | None = None


def _load_countries():
    """Load Natural Earth 1:50m admin-0 countries.

    Returns a dict mapping ISO3 code -> MultiPolygon. Falls back to ADM0_A3
    when ISO_A3 is missing or '-99' (Natural Earth uses this for disputed
    territories like Norway, France-with-overseas, Kosovo, etc.).
    """
    global _countries_geojson_cache
    if _countries_geojson_cache is not None:
        return _countries_geojson_cache

    data_dir = naaulu.config.get_data_dir()
    filename = "ne_50m_admin_0_countries.geojson"
    url = f"https://github.com/nvkelso/natural-earth-vector/raw/master/geojson/{filename}"
    path = os.path.join(data_dir, filename)

    if not os.path.exists(path):
        downloaded_path = naaulu.network.download(url, filename=filename)
        if downloaded_path != path:
            shutil.move(downloaded_path, path)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    by_iso: dict[str, list] = {}
    for feature in data["features"]:
        props = feature["properties"]
        iso = props.get("ISO_A3") or ""
        if not iso or iso == "-99":
            iso = props.get("ADM0_A3") or ""
        if not iso or iso == "-99":
            continue
        g = shapely.geometry.shape(feature["geometry"])
        geoms = list(g.geoms) if isinstance(g, shapely.geometry.MultiPolygon) else [g]
        by_iso.setdefault(iso, []).extend(geoms)

    _countries_geojson_cache = {
        iso: shapely.geometry.MultiPolygon(geoms) for iso, geoms in by_iso.items()
    }
    return _countries_geojson_cache


def get_countries(*, geom=None, iso_only=False, year="2023", adm="0", min_ratio=0, iso_codes=None):
    """
    Get country boundaries from geometry.

    Parameters:    
    - geom: Shapely geometry to filter countries by intersection.
    - iso_only: return only ISO codes
    - year: Dataset year (default "2023").
    - adm: Administrative level (default "0").
    - min_ratio: min overlapping ratio with geom
    - iso_codes: optional set of ISO3 codes to restrict iteration

    Returns:
    - shapely.geometry.MultiPolygon of selected country or countries.
    """
    selected = []
    for country in pycountry.countries:
        if iso_codes is not None and country.alpha_3 not in iso_codes:
            continue
        try:
            country_geom = import_country(iso_code=country.alpha_3, year=year, adm=adm)
            if geom is not None:
                if min_ratio > 0:
                    intersection = geom.intersection(country_geom)
                    overlap_ratio = intersection.area / country_geom.area
                    if overlap_ratio < min_ratio:
                        continue
                else:
                    if not country_geom.intersects(geom):
                        continue

            if iso_only:
                selected.append(country.alpha_3)
            else:
                selected.extend(country_geom.geoms)
        except Exception as e:
            logger.debug(f"Skipping {country.alpha_3}: {e}")

    if iso_only:
        return selected
    
    return shapely.geometry.MultiPolygon(selected)

def import_country(iso_code=None, year="2023", adm="0", mainland_only=True):
    """
    Load country boundary from Natural Earth 1:50m admin-0.

    Parameters:
    - iso_code: ISO3 country code (e.g. "FRA").
    - year: kept for signature compatibility (single global snapshot).
    - adm: Administrative level. Only "0" is supported.
    - mainland_only: If True, keep only the largest polygon (mainland) and discard small islands.

    Returns:
    - shapely.geometry.MultiPolygon of selected country.
    """
    if adm != "0":
        raise NotImplementedError(f"only ADM0 is supported, got ADM{adm}")

    cache_key = f"{iso_code}_{year}_ADM{adm}{'_mainland' if mainland_only else ''}"
    if cache_key in _country_cache:
        return _country_cache[cache_key]

    countries = _load_countries()
    if iso_code not in countries:
        raise KeyError(f"ISO3 {iso_code} not found in Natural Earth admin-0")

    geoms = list(countries[iso_code].geoms)

    if mainland_only and geoms:
        geoms.sort(key=lambda g: g.area, reverse=True)
        geoms = [geoms[0]]

    geom = shapely.geometry.MultiPolygon(geoms)
    _country_cache[cache_key] = geom
    return geom


def build_country_tiles(chunk_width=2, chunk_height=None):
    """Precompute tile→countries mapping and save to cache."""
    if chunk_height is None:
        chunk_height = chunk_width

    cache_dir = naaulu.config.get_cache_dir()
    cache_key = f"country_tiles_{chunk_width}x{chunk_height}"
    path = os.path.join(cache_dir, f"{cache_key}.json")

    tile_countries = {}
    countries = _load_countries()

    for iso_code, country_geom in countries.items():
        minx, miny, maxx, maxy = country_geom.bounds
        x_start = math.floor(minx / chunk_width) * chunk_width
        x_end = math.ceil(maxx / chunk_width) * chunk_width
        y_start = math.floor(miny / chunk_height) * chunk_height
        y_end = math.ceil(maxy / chunk_height) * chunk_height

        x = x_start
        while x < x_end:
            y = y_start
            while y < y_end:
                tile = shapely.geometry.box(x, y, x + chunk_width, y + chunk_height)
                if tile.intersects(country_geom):
                    key = f"{naaulu.util.format_lon(x)}_{naaulu.util.format_lon(x + chunk_width)}.{naaulu.util.format_lat(y)}_{naaulu.util.format_lat(y + chunk_height)}"
                    if key not in tile_countries:
                        tile_countries[key] = []
                    tile_countries[key].append(iso_code)
                y += chunk_height
            x += chunk_width

    logger.info(f"building country tile cache: {path}")
    with open(path, "w") as f:
        json.dump(tile_countries, f)

    return tile_countries


def get_country_tiles(geom, chunk_width=2, chunk_height=None):
    """Get set of ISO3 codes intersecting tiles covering geom."""
    if chunk_height is None:
        chunk_height = chunk_width

    cache_key = f"country_tiles_{chunk_width}x{chunk_height}"

    if cache_key not in _country_tiles_cache:
        cache_dir = naaulu.config.get_cache_dir()
        path = os.path.join(cache_dir, f"{cache_key}.json")

        if not os.path.exists(path):
            tile_countries = build_country_tiles(chunk_width, chunk_height)
        else:
            with open(path) as f:
                tile_countries = json.load(f)

        _country_tiles_cache[cache_key] = tile_countries

    tile_countries = _country_tiles_cache[cache_key]

    minx, miny, maxx, maxy = geom.bounds
    x_start = math.floor(minx / chunk_width) * chunk_width
    x_end = math.ceil(maxx / chunk_width) * chunk_width
    y_start = math.floor(miny / chunk_height) * chunk_height
    y_end = math.ceil(maxy / chunk_height) * chunk_height

    iso_codes = set()
    x = x_start
    while x < x_end:
        y = y_start
        while y < y_end:
            key = f"{naaulu.util.format_lon(x)}_{naaulu.util.format_lon(x + chunk_width)}.{naaulu.util.format_lat(y)}_{naaulu.util.format_lat(y + chunk_height)}"
            iso_codes.update(tile_countries.get(key, []))
            y += chunk_height
        x += chunk_width

    return iso_codes


def country_code(name):
    """Returns the ISO 3166-1 alpha-3 country code for a given country name."""
    try:
        country = pycountry.countries.lookup(name)
    except LookupError:
        try:
            country = pycountry.countries.search_fuzzy(name)[0]
        except LookupError:            
            valid_names = sorted([c.name for c in pycountry.countries])
            formatted_list = textwrap.fill("\n".join(valid_names), width=40)
            raise ValueError(
                f"Country '{name}' not found.\n\nValid country names:\n\n{formatted_list}"
            )
    return country.alpha_3


def import_window(window):

    lon_min, lat_min, lon_max, lat_max = window
    geometry = shapely.geometry.box(lon_min, lat_min, lon_max, lat_max)

    return geometry


def chunk_geometry(*, geometry, chunk_width, chunk_height):
    """Returns clipped tiles aligned to a global grid, covering the input geometry."""
    minx, miny, maxx, maxy = geometry.bounds

    # Snap bounds to global grid
    x_start = math.floor(minx / chunk_width) * chunk_width
    x_end = math.ceil(maxx / chunk_width) * chunk_width
    y_start = math.floor(miny / chunk_height) * chunk_height
    y_end = math.ceil(maxy / chunk_height) * chunk_height

    tiles = []
    x = x_start
    while x < x_end:
        y = y_start
        while y < y_end:
            tile = shapely.geometry.box(x, y, x + chunk_width, y + chunk_height)
            if tile.intersects(geometry):
                tiles.append(tile)
            y += chunk_height
        x += chunk_width

    return tiles


def radar_coverage(*, location, distance):

    coverage = shapely.geometry.Point(0, 0).buffer(distance)
    radar_crs = wradlib.georef.get_radar_projection(site=location)
    wgs = wradlib.georef.get_earth_projection()
    coords = list(coverage.exterior.coords)
    transformed_coords = wradlib.georef.reproject(
        coords, 
        src_crs=radar_crs,
        trg_crs=wgs,
        )
    coverage = shapely.geometry.Polygon(transformed_coords)

    return coverage


def radar_distance_to_geometry(*, location, geometry):
    """Return geodesic distance (meters) from radar location to geometry.
    0 if the point is inside the geometry.
    """
    from pyproj import Geod
    from shapely.geometry import Point
    pt = Point(location)
    if geometry.contains(pt):
        return 0.0
    geod = Geod(ellps="WGS84")
    # nearest point on boundary
    nearest = geometry.boundary.interpolate(geometry.boundary.project(pt))
    lon1, lat1 = location
    lon2, lat2 = nearest.x, nearest.y
    _, _, dist = geod.inv(lon1, lat1, lon2, lat2)
    return dist


def tile_dataset(tile, resolution):
    """Create a raster dataset for a given tile and resolution."""
    dataset = wradlib.georef.create_raster_geographic(
        tile.bounds, resolution, resolution_in_meters=True
    )

    return dataset


def points(coords):

    geometry = shapely.geometry.MultiPoint(coords)

    return geometry


def coords_to_tiles(*, coords, chunk_width, chunk_height):

    tiles = []
    for xy in coords:
        out = naaulu.geography.chunk_geometry(
            geometry=shapely.geometry.Point(xy),
            chunk_height=chunk_height,
            chunk_width=chunk_width,
            )
        tiles.append(out[0])

    return tiles


    



def cut(ds, geometry):
    selected = []
    for i, (lon, lat) in enumerate(zip(ds.longitude.values, ds.latitude.values)):
        pt = shapely.geometry.Point(lon, lat)
        if pt.intersects(geometry):
            selected.append(i)
    dim = list(ds.dims)[0]
    ds = ds.isel({dim: selected})

    return ds
