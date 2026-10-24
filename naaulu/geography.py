
import datetime
import json
import math
import logging
import os
import shutil
import textwrap
import time

import numpy
import pycountry
import pyproj.crs
import pyproj.transformer
import requests
import shapely
import shapely.geometry
import shapely.ops
import xarray
import wradlib.georef
import wradlib.ipol

import naaulu.network
import naaulu.config
import naaulu.util

logger = logging.getLogger(__name__)

_country_cache: dict[str, shapely.geometry.MultiPolygon] = {}
_country_tiles_cache: dict[str, dict] = {}
_countries_geojson_cache: dict[str, shapely.geometry.MultiPolygon] | None = None
_countries_10m_cache: dict[str, shapely.geometry.MultiPolygon] | None = None
# (ISO3, level) -> assembled OSM boundary set (ADM0/ADM1/ADM2)
_osm_cache: dict[tuple[str, str], shapely.geometry.MultiPolygon] = {}


# ─── Boundaries ────────────────────────────────────────────────────────────
# All drawn boundaries (country outlines + the one province tier) come from
# OpenStreetMap, queried through Geofabrik Postpass (postpass.geofabrik.de —
# SQL over OSM's full planet database, served as GeoJSON: no planet download
# to store, and unlike Overpass no rate-limit bans whenever the whole cache
# had to fill). Fills happen at prefetch or on a plot's first use, never
# under NAAULU_BOUNDARIES_OFFLINE=1: adjacent units are served from
# the *same* boundary ways in one database, so neighbours share exact
# linework and can never disagree — the problem with geoBoundaries, which
# standardises each unit from its own national source (FRA/BEL ADM0
# overlapped by ~650 km², the région tier by ~236 km²).
# Geometry is stored exactly as served and is never simplified: simplifying
# two neighbours independently is what opens hairline seams between them —
# only untouched shared linework stays bit-identical on both sides of a
# border, so the seam cannot open.
# OSM relations close their ring with the territorial sea (maritime=yes ways)
# instead of the physical coastline, and are drawn as stored: the country
# outline follows the ~12 nm territorial line, and no coastline is fetched.
# OSM is ODbL: figures based on it must credit "© OpenStreetMap contributors";
# the per-file licence is recorded in a .meta.json beside each cached file.

# OSM has no ADM numbering — admin_level conventions are per country. naaulu
# draws exactly one sub-national tier: the *province* tier (GADM's ADM2 —
# départements, provincien, cantons, Länder, counties …); the
# state/region tier (ADM1: régions, Länder, gewesten) is not drawn. Map the
# province tier to the admin_level that actually holds it in each country,
# verified against OSM relation counts:
#   default 6: FR départements · BE provincien · IT/ES provincias ·
#              EE maakonnad · LUX cantons · CZ obvody · US counties ·
#              AU LGAs · GB counties-ish …
#   NLD 4: the provincie IS the province (nothing below it worth drawing)
#   CHE 4: cantons · FIN 4: maakunnat (2020 reform: no middle tier)
#   DEU 4: the 16 Länder — Regierungsbezirke at 5 cover only
#          BW/Bayern/Hessen/NRW and leave the rest of the country without
#          any sub-tier lines; kreise at 6 are the too-fine tier
#   HRV 4: županije (21, Grad Zagreb included) · ROU 4: județe (42) ·
#          MDA 4: raions + municipalities (37) · POL 4: województwa (16 —
#          powiaty at 6 are the too-fine tier for a drawn map) ·
#          CZE 4: kraje (14 — SO ORP at 6 is the too-fine tier)
#   AND 7: the parroquies (ISO3166-2 AD-02..AD-08)
#   DNK 7: kommuner (regioner at 4 = the state tier) · NOR 7: kommuner
#          (fylker at 4) · SWE 7: kommuner (lv6 has only 2 samebyar) ·
#          ALA 8: kommuner · JEY 8: the 12 parishes · GGY 8: its parishes
#          (lv4 is the bailiwick itself, an ADM0 duplicate)
#   LVA 5: novadi (42, the 2021 tier — lv6 is no Latvian tier at all) ·
#          MKD 7: opštini (80) · SVN 8: občine · MLT 8: local councils (68)
#          · SMR 8: castelli (9) · LIE 8: Gemeinden (11)
#   MCO 10: the quartiers (9) — Monaco's only subdivision, so it draws its
#          actual tiers instead of nothing at the default
# A wrong or missing level degrades to no lines for that country, never to
# an error — and the prefetch warns on an empty tier, so the mapping can
# be fixed instead of silently shipping no lines.
OSM_PROVINCE_LEVEL: dict[str, int] = {
    "NLD": 4,
    "CHE": 4,
    "FIN": 4,
    "DEU": 4,
    "HRV": 4,
    "ROU": 4,
    "MDA": 4,
    "POL": 4,
    "CZE": 4,
    "LVA": 5,
    "AND": 7,
    "MKD": 7,
    "SWE": 7,
    "DNK": 7,
    "NOR": 7,
    "GGY": 8,
    "ALA": 8,
    "JEY": 8,
    "SVN": 8,
    "MLT": 8,
    "SMR": 8,
    "LIE": 8,
    "MCO": 10,
}
OSM_PROVINCE_LEVEL_DEFAULT = 6

# Top-N largest ADM0 parts whose bounding boxes scope a spatial query: the
# mainland comes first, but the overseas parts hold that country's other
# units and cities (Natural Earth does not link overseas territories to
# their parent, so only the fetched ADM0 knows about them).
QUERY_BOUNDS_LIMIT = 64

# Padding around the Natural Earth fallback envelope (degrees, ~20 km): NE's
# outline can be tighter than OSM's own units — its Åland polygon stops north
# of the southern kommuner — and the pad keeps them inside the query. Spill
# is dropped at attribution anyway.
NE_BOUNDS_PAD = 0.2

# How far (degrees, ~65 km) a unit's representative point may sit from the
# requested country's own land before attribution gives up on it. The outer
# Åland kommuner reach ~0.50° outside NE's outline, the Finnish municipality
# the query drags in sits at 0.89°, open ocean tens of degrees — so the cap
# keeps the first and drops the rest.
EXTENT_MAX_DEG = 0.6


def _load_ne_countries(filename):
    """Natural Earth admin-0 countries from a bundled file: ISO3 -> MultiPolygon."""
    bundled = naaulu.config.get_bundled_data_path(filename)
    with open(bundled, "r", encoding="utf-8") as f:
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

    return {
        iso: shapely.geometry.MultiPolygon(geoms) for iso, geoms in by_iso.items()
    }


def _load_countries():
    """Natural Earth 50m countries: ISO3 -> MultiPolygon.

    The plumbing file: window→country tile lookup and the attribution and
    scoping fallbacks. Never drawn.
    """
    global _countries_geojson_cache
    if _countries_geojson_cache is None:
        _countries_geojson_cache = _load_ne_countries(
            "ne_50m_admin_0_countries.geojson")
    return _countries_geojson_cache


def _load_countries_10m():
    """Natural Earth 10m countries: ISO3 -> MultiPolygon.

    The bundled *drawn* source behind `--adm 1` — Belgium: 648 vertices
    vs 164 at the 50m plumbing scale, still far coarser than OSM.
    """
    global _countries_10m_cache
    if _countries_10m_cache is None:
        _countries_10m_cache = _load_ne_countries(
            "ne_10m_admin_0_countries.geojson")
    return _countries_10m_cache


def _osm_paths(iso3, level):
    """(geometry, metadata) file paths of one cached OSM boundary set."""
    directory = naaulu.config.get_data_dir("boundaries")
    stem = os.path.join(directory, f"openstreetmap-{iso3}-{level}")
    return stem + ".geojson", stem + ".meta.json"


# Geofabrik Postpass serves OSM's full planet database over SQL, as GeoJSON.
# A prefetch is therefore a handful of small queries per country (megabytes,
# mostly that country's own detailed outline) instead of a planet extract on
# disk — and unlike the old Overpass days, bulk queries are not rate-limited.
POSTPASS_URL = "https://postpass.geofabrik.de/api/interpreter"


def _postpass(sql, attempts=3):
    """Run one SQL statement against Postpass; return its GeoJSON features.

    Transient failures are retried with backoff (Postpass answers 504 under
    load, and neither a 15-minute image bake nor an on-demand fill may die
    on the first one); a persistent failure still raises, so prefetches
    fail loud instead of shipping a half-filled cache.
    """
    delay, payload = 2.0, None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.post(
                POSTPASS_URL, data={"data": sql}, timeout=(10, 600),
                headers={"User-Agent": "naaulu (rainfall tools; git.naaulu.org)"})
            response.raise_for_status()
            payload = response.json()
            break
        except (requests.RequestException, ValueError) as e:
            if attempt == attempts:
                raise RuntimeError(f"Postpass query failed: {e}") from e
            logger.warning(
                f"Postpass query failed ({attempt}/{attempts}): {e} — "
                f"retrying in {delay:.0f}s")
            time.sleep(delay)
            delay *= 3
    features = payload.get("features") if isinstance(payload, dict) else None
    if features is None:
        raise RuntimeError(
            f"Postpass query returned no FeatureCollection: "
            f"{str(payload)[:200]}")
    return features


def _envelope_sql(bounds):
    """`geom && envelope` clauses scoping a query to bounding boxes."""
    return " OR ".join(
        f"geom && ST_MakeEnvelope({minx}, {miny}, {maxx}, {maxy}, 4326)"
        for minx, miny, maxx, maxy in bounds)


def _fetch_units(iso3, admin_level):
    """Admin boundary polygons near one country, as GeoJSON features.

    ADM0 (the country outline) is selected by its ISO3166-1 tag: country
    relations carry it, and the tag filter returns that one relation instead
    of every neighbour a spatial filter would drag in. The province tier is
    selected spatially — units rarely carry usable ISO tags — against the
    country's own extent, so foreign units arrive as spill and are dropped
    at attribution below.
    """
    admin_level = int(admin_level)
    if admin_level == 2:
        iso2 = _iso2_of(iso3)
        if not iso2:
            raise RuntimeError(
                f"No ISO3166-1 code for {iso3}; cannot query its outline")
        where = (f"(tags->>'ISO3166-1' = '{iso2}' "
                 f"OR tags->>'ISO3166-1:alpha2' = '{iso2}')")
    else:
        where = f"({_envelope_sql(_query_bounds(iso3))})"
    return _postpass(
        "SELECT * FROM postpass_polygon "
        "WHERE tags->>'boundary' = 'administrative' "
        f"AND tags->>'admin_level' = '{admin_level}' AND {where}")


def _query_bounds(iso3):
    """Bounding boxes scoping a spatial query for one country.

    The cached ADM0 geometry bounds it exactly — its overseas parts
    included, which Natural Earth does not link to their parent country.
    Before ADM0 exists (the first tier of a run) the bundled Natural Earth
    country stands in; its bounds cover the mainland, which is why ADM0 is
    fetched first.
    """
    cached = _read_osm_cache(iso3, "ADM0")
    if cached is not None and not cached.is_empty:
        parts = sorted(_parts(cached), key=lambda g: g.area, reverse=True)
        return [p.bounds for p in parts[:QUERY_BOUNDS_LIMIT]]
    country = _load_countries().get(iso3)
    if country is None or country.is_empty:
        raise RuntimeError(
            f"No Natural Earth coverage for {iso3}; wrong ISO3 code?")
    minx, miny, maxx, maxy = country.bounds
    return [(minx - NE_BOUNDS_PAD, miny - NE_BOUNDS_PAD,
             maxx + NE_BOUNDS_PAD, maxy + NE_BOUNDS_PAD)]


def _own_extent(iso3):
    """One country's own geometry, for scoping and placing its city nodes.

    The fetched ADM0 is exact (OSM stores the territorial sea and every
    overseas part). Where OSM has no country-level relation at all — Åland
    is modelled only as a level-4 region plus its municipalities — the
    bundled Natural Earth country stands in: labels are decoration, and NE
    is already the attribution fallback everywhere else. Drawn *borders*
    never fall back to NE.
    """
    cached = _read_osm_cache(iso3, "ADM0")
    if cached is not None and not cached.is_empty:
        return cached
    fallback = _load_countries().get(iso3)
    if fallback is not None and not fallback.is_empty:
        logger.debug(
            f"Places for {iso3} scoped by Natural Earth: OSM has no "
            f"country-level outline for it")
        return fallback
    return None


def _fetch_places(iso3):
    """place=city|town nodes inside one country's own extent.

    Returns None when neither the ADM0 cache nor Natural Earth knows the
    country: there is then no scope for the query and no country to place
    the nodes in.
    """
    extent = _own_extent(iso3)
    if extent is None:
        logger.debug(f"Places for {iso3} have no extent to scope by; "
                     f"skipping")
        return None
    parts = sorted(_parts(extent), key=lambda g: g.area, reverse=True)
    bounds = [p.bounds for p in parts[:QUERY_BOUNDS_LIMIT]]
    return _postpass(
        "SELECT * FROM postpass_point "
        "WHERE tags->>'place' IN ('city', 'town') AND tags ? 'name' "
        f"AND ({_envelope_sql(bounds)})")


def _store_cities(iso3, places):
    """Attribute place nodes to one country (own-extent test) and cache them."""
    geom_obj = _own_extent(iso3)
    if geom_obj is None:
        logger.debug(f"Cities for {iso3} have no extent to fall in; skipping")
        return
    shapely.prepare(geom_obj)
    features = []
    for place in places:
        tags = place.get("properties", {}).get("tags") or {}
        lon, lat = place["geometry"]["coordinates"][:2]
        if not geom_obj.covers(shapely.geometry.Point(lon, lat)):
            continue  # query spill across the border
        features.append({
            "type": "Feature",
            "properties": {
                "ISO_A3": iso3,
                "name": tags.get("name"),
                "place": tags.get("place"),
                "population": tags.get("population"),
                "capital": tags.get("capital"),
            },
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
        })
    timestamp = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
    _write_cities_cache(iso3, features, timestamp)
    _cities_cache[iso3] = features


def _ingest_units(units, iso3_codes, level):
    """Attribute and cache one level of fetched unit polygons.

    Units arrive as polygon features (each boundary relation Postpass
    serves as one feature) and are stored as served — never simplified, so
    the linework two neighbours share stays bit-identical and their seam
    cannot open. A level with no matching units is negative-cached as an
    empty file.
    """
    requested = set(iso3_codes)

    # Assign each unit to a country — ISO tags first, bundled Natural Earth
    # as fallback (a spatial query also returns units whose members merely
    # touch the area, e.g. a cross-border district of a neighbour, and those
    # must be attributed or dropped, not merged into the wrong country).
    by_country = {}  # iso3 -> parts, kept separate per unit
    for unit in units:
        tags = unit.get("properties", {}).get("tags") or {}
        name = tags.get("name", "?")
        try:
            polygon = shapely.geometry.shape(unit["geometry"])
            if polygon.is_empty:
                continue
            iso3 = (_iso3_from_tags(tags, level)
                    or _iso3_by_ne_point(polygon)
                    or _iso3_by_padded_ne(polygon, requested))
            if iso3 is None or iso3 not in requested:
                logger.debug(
                    f"Skipping {level} unit outside the batch: {name}")
                continue
            by_country.setdefault(iso3, []).extend(_parts(polygon))
        except Exception as e:
            logger.debug(f"Skipping OSM unit for {name}: {e}")

    # Cache every unit's polygons *separately* — a unary_union would merge
    # adjacent units and erase the interior province lines that --adm 2
    # exists to draw (the old geoBoundaries loader also returned unmerged unit
    # lists).
    timestamp = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
    for iso3, parts in by_country.items():
        if parts:
            geometry = (parts[0] if len(parts) == 1
                        else shapely.geometry.MultiPolygon(parts))
            _write_osm_cache(iso3, level, geometry, timestamp,
                             admin_level=_province_level(iso3)
                             if level == "ADM2" else None)
            _osm_cache[(iso3, level)] = geometry

    # A fetch that matched no units for a requested country (an empty admin
    # level, an unmapped microstate) is remembered as an empty result —
    # otherwise every draw would look for it forever.
    for iso3 in requested:
        if iso3 not in by_country:
            empty = shapely.geometry.MultiPolygon()
            _write_osm_cache(iso3, level, empty, timestamp,
                             admin_level=_province_level(iso3)
                             if level == "ADM2" else None)
            _osm_cache[(iso3, level)] = empty
            if level == "ADM2":
                logger.warning(
                    f"No {iso3} units at admin_level {_province_level(iso3)} "
                    f"— OSM_PROVINCE_LEVEL in naaulu/geography.py likely "
                    f"needs the right tier for this country")


def _prefetch_country(iso3, levels):
    """Fill the requested caches for one country from Postpass.

    Tiers run in a fixed order: ADM0 first, because the province tier and
    the place nodes are scoped by the country's own outline.
    """
    if "ADM0" in levels:
        _ingest_units(_fetch_units(iso3, 2), [iso3], "ADM0")
    if "ADM2" in levels:
        _ingest_units(
            _fetch_units(iso3, _province_level(iso3)), [iso3], "ADM2")
    if "CITIES" in levels:
        places = _fetch_places(iso3)
        if places is not None:
            _store_cities(iso3, places)


def _province_level(iso3):
    """OSM admin_level of one country's province tier."""
    return OSM_PROVINCE_LEVEL.get(iso3, OSM_PROVINCE_LEVEL_DEFAULT)


def _iso3_from_tags(tags, level="ADM0"):
    """ISO3 from a relation's ISO3166 tags, or None.

    For units ISO3166-2 must be read first: overseas French regions carry
    their own ISO3166-1 country code (GF, RE, MQ …) *alongside* FR-GF/FR-RE,
    and only the -2 tag names the parent country whose batch requested them.
    ADM0 relations are countries themselves, so -1 leads there.
    """
    keys = (("ISO3166-1", "ISO3166-1:alpha2", "ISO3166-2") if level == "ADM0"
            else ("ISO3166-2", "ISO3166-1", "ISO3166-1:alpha2"))
    for key in keys:
        value = tags.get(key) or ""
        if not value:
            continue
        iso2 = value.split("-")[0] if key == "ISO3166-2" else value
        iso2 = iso2.upper()
        for country in pycountry.countries:
            if country.alpha_2 == iso2:
                return country.alpha_3
    return None


def _iso3_by_ne_point(polygon):
    """ISO3 of the bundled Natural Earth country covering a representative
    point — the assignment fallback for relations without ISO tags."""
    point = polygon.representative_point()
    for iso3, country in _load_countries().items():
        minx, miny, maxx, maxy = country.bounds
        if minx <= point.x <= maxx and miny <= point.y <= maxy \
                and country.covers(point):
            return iso3
    return None


def _iso3_by_padded_ne(polygon, iso3_codes):
    """Attribution last resort: the requested country whose own land lies
    within EXTENT_MAX_DEG of the unit.

    ISO tags and exact Natural Earth placement come first; this catches what
    NE's too-tight outline cannot reach — Åland's tagless kommuner sit
    outside its polygon, out to ~0.50°. The cap keeps the ocean dropped
    (tens of degrees from any French land) and foreign units the query
    drags in (the Finnish Kustavi at 0.89°).
    """
    point = polygon.representative_point()
    for iso3 in sorted(iso3_codes):
        country = _load_countries().get(iso3)
        if country is not None and country.distance(point) <= EXTENT_MAX_DEG:
            return iso3
    return None


def _read_osm_cache(iso3, level):
    """Cached OSM polygon for one ISO3 and level, or None if never fetched.

    An empty MultiPolygon means a successful fetch found no units at this
    level (negative cache) — present-but-empty, so draws do not re-query it.
    """
    key = (iso3, level)
    json_path, _ = _osm_paths(iso3, level)
    if not os.path.exists(json_path):
        return None
    if level == "ADM2":
        # a file fetched at another admin_level is the wrong tier entirely
        _, meta_path = _osm_paths(iso3, level)
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                cached_level = json.load(f).get("adminLevel")
        except Exception:
            cached_level = None  # legacy sidecar without the key: trust it
        if cached_level is not None and cached_level != _province_level(iso3):
            logger.debug(
                f"Cached {iso3} ADM2 holds admin_level {cached_level} but the "
                f"mapping now wants {_province_level(iso3)}; refetching")
            return None
    if key in _osm_cache:
        return _osm_cache[key]
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            features = json.load(f)["features"]
        geometry = (shapely.geometry.MultiPolygon() if not features
                    else shapely.geometry.shape(features[0]["geometry"]))
        _osm_cache[key] = geometry
        return geometry
    except Exception as e:
        logger.debug(f"Skipping cached OSM {iso3} {level}: {e}")
        return None


def _rounded(mapping, digits=7):
    """Round GeoJSON coordinates to OSM's native 1e-7 grid (~1 cm).

    Deterministic and applied identically to neighbours, so shared linework
    stays byte-identical across cached countries.
    """
    if isinstance(mapping, float):
        return round(mapping, digits)
    if isinstance(mapping, (list, tuple)):
        return [_rounded(item, digits) for item in mapping]
    return mapping


def _write_osm_cache(iso3, level, geometry, timestamp, admin_level=None):
    """Persist one assembled boundary set with its ODbL sidecar.

    ADM2 files record the admin_level they were fetched at, so a later
    mapping change (OSM_PROVINCE_LEVEL) refetches instead of silently
    serving the old tier.
    """
    json_path, meta_path = _osm_paths(iso3, level)
    feature = {
        "type": "Feature",
        "properties": {"ISO_A3": iso3, "level": level},
        "geometry": _rounded(shapely.geometry.mapping(geometry)),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": [feature]}, f)
    sidecar = {
        "source": "OpenStreetMap (via Geofabrik Postpass)",
        "url": "https://postpass.geofabrik.de/",
        "license": "Open Data Commons Open Database License (ODbL) v1.0",
        "licenseSource": "www.openstreetmap.org/copyright",
        "attribution": "© OpenStreetMap contributors",
        "level": level,
        "osmTimestamp": timestamp,
        "downloaded": datetime.datetime.now(
            tz=datetime.timezone.utc
        ).isoformat(),
    }
    if admin_level is not None:
        sidecar["adminLevel"] = admin_level
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)
    logger.info(f"cached OSM boundaries {iso3} (ODbL, © OpenStreetMap contributors)")


def _iso2_of(iso3):
    try:
        return pycountry.countries.get(alpha_3=iso3).alpha_2
    except Exception:
        return None


def _boundaries_offline():
    """Realtime deployments set NAAULU_BOUNDARIES_OFFLINE=1: renders then
    only ever read the (image-baked) cache and never touch the network — no
    latency, no rate limits, no transient gaps mid-serve."""
    return os.environ.get("NAAULU_BOUNDARIES_OFFLINE", "").lower() not in (
        "", "0", "false")


def _tier_missing(iso3, level):
    """True when one tier must be (re)fetched: never cached, unreadable, or
    cached at another admin_level than the mapping wants now — the rule
    `_read_osm_cache` applies to readers, so a changed OSM_PROVINCE_LEVEL
    refetches the tier instead of serving the voided file forever."""
    if level == "CITIES":
        return _read_cities_cache(iso3) is None
    return _read_osm_cache(iso3, level) is None


def _ensure_fetched(iso_codes, level):
    """Fill the tiers a render is about to read (fetch on demand).

    A cold plot fills exactly the window's countries — once, seconds and a
    few megabytes — and every later render reads the cache. ADM0 is fetched
    first where it is not the requested tier, because the province tier and
    the place nodes scope by the country's own outline. NAAULU_BOUNDARIES_
    OFFLINE=1 keeps serves strictly local (a gap stays a gap), and a failed
    fetch degrades to omitted lines instead of a failed plot.
    """
    if _boundaries_offline():
        return
    tiers = ("ADM0", level) if level != "ADM0" else ("ADM0",)
    for iso3 in sorted({c for c in iso_codes if c}):
        for tier in tiers:
            if not _tier_missing(iso3, tier):
                continue
            logger.info(f"Cache miss for {iso3} {tier} — fetching on demand")
            try:
                _prefetch_country(iso3, (tier,))
            except Exception as e:
                logger.warning(
                    f"On-demand fetch failed for {iso3} {tier}: {e}")


def prefetch_boundaries(iso_codes, levels=("ADM0", "ADM2"), refresh=False):
    """Fill the boundary and city caches from Geofabrik Postpass.

    Intended for image builds and machine provisioning: one set of small
    queries per country feeds every requested level (megabytes, never a
    planet download), so a later render starts warm — though it still fills
    a missing tier itself on demand. `refresh` re-queries levels that are
    already cached, and a tier voided by a changed province mapping
    refetches on its own. Raises if any
    requested tier ends up with no cache file (unreachable service, bad
    ISO3) — a build step must fail loudly instead of shipping a half-filled
    cache.
    """
    if _boundaries_offline():
        raise RuntimeError(
            "NAAULU_BOUNDARIES_OFFLINE is set; unset it to prefetch")
    iso_codes = sorted(set(iso_codes))
    for iso3 in iso_codes:
        missing = [level for level in levels
                   if refresh or _tier_missing(iso3, level)]
        if not missing:
            continue
        _prefetch_country(iso3, missing)

    filled, failed = {}, []
    for level in levels:
        data = (_load_cities(iso_codes) if level == "CITIES"
                else _load_osm(list(iso_codes), level))
        filled[level] = sorted(iso for iso, items in data.items() if items)
        for iso3 in iso_codes:
            if not os.path.exists(_osm_paths(iso3, level)[0]):
                failed.append(f"{iso3} {level}")
    if failed:
        raise RuntimeError(
            "boundary prefetch failed for: " + ", ".join(failed))
    return filled


def _load_osm(iso_codes, level):
    """Load boundary sets of one level: ISO3 -> MultiPolygon (OpenStreetMap).

    Read-only: a missing set is a prefetch gap — those lines are simply not
    drawn (fills run through `prefetch_boundaries` or the on-demand step of
    the public draw helpers, never inside a bare load), and
    a cached-but-empty set (no units exist at this level) counts as fetched —
    it is absent from the result.
    """
    result, missing = {}, []
    for iso3 in iso_codes:
        cached = _read_osm_cache(iso3, level)
        if cached is None:
            missing.append(iso3)
        elif not cached.is_empty:
            result[iso3] = cached

    if missing:
        logger.debug(
            f"No cached {level} for {', '.join(sorted(missing))} — lines "
            f"omitted; fill with prefetch_boundaries.py")
    return result


def _load_admin0(iso_codes):
    """Admin-0 (country) boundaries: ISO3 -> MultiPolygon (OpenStreetMap)."""
    return _load_osm(iso_codes, "ADM0")


def _load_admin2(iso_codes):
    """Admin-2 (province) boundaries: ISO3 -> MultiPolygon (OpenStreetMap)."""
    return _load_osm(iso_codes, "ADM2")


# ─── populated places (cities) ──────────────────────────────────────────────

# Rank stand-ins when OSM carries no population tag on a place node.
CITIES_DEFAULT_POPULATION = {"city": 50_000, "town": 10_000}

_cities_cache = {}  # ISO3 -> list of GeoJSON point features


def _parse_population(value):
    """OSM population tags are free-form ('12 345', '1,234', '~5000')."""
    if not value:
        return None
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return int(digits) if digits else None


def _write_cities_cache(iso3, features, timestamp):
    """Persist one country's place nodes with their ODbL sidecar."""
    json_path, meta_path = _osm_paths(iso3, "CITIES")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)
    sidecar = {
        "source": "OpenStreetMap (via Geofabrik Postpass)",
        "url": "https://postpass.geofabrik.de/",
        "license": "Open Data Commons Open Database License (ODbL) v1.0",
        "licenseSource": "www.openstreetmap.org/copyright",
        "attribution": "© OpenStreetMap contributors",
        "level": "CITIES",
        "osmTimestamp": timestamp,
        "downloaded": datetime.datetime.now(
            tz=datetime.timezone.utc
        ).isoformat(),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)
    logger.info(f"cached OSM places {iso3} (ODbL, © OpenStreetMap contributors)")


def _read_cities_cache(iso3):
    """Cached place nodes for one country, or None if never fetched.

    An empty feature list means the fetch found no places (negative cache) —
    present-but-empty, so draws do not re-query it.
    """
    if iso3 in _cities_cache:
        return _cities_cache[iso3]
    json_path, _ = _osm_paths(iso3, "CITIES")
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            features = json.load(f)["features"]
        _cities_cache[iso3] = features
        return features
    except Exception as e:
        logger.debug(f"Skipping cached places {iso3}: {e}")
        return None


def _load_cities(iso_codes):
    """Place-node features per ISO3, from the cache (read-only).

    Read-only: fills run through `prefetch_boundaries` or `get_cities`'s
    on-demand step (never under NAAULU_BOUNDARIES_OFFLINE=1). A
    cached-but-empty country (no places) counts as
    fetched and is absent from the result.
    """
    result, missing = {}, []
    for iso3 in iso_codes:
        cached = _read_cities_cache(iso3)
        if cached is None:
            missing.append(iso3)
        elif cached:
            result[iso3] = cached
    if missing:
        logger.debug(
            f"No cached places for {', '.join(sorted(missing))} — labels "
            f"omitted; fill with prefetch_boundaries.py --levels CITIES")
    return result


def get_cities(geom, iso_codes, limit=25):
    """Major cities inside the geometry, ranked for drawing.

    Score = OSM `population` (a place-class default when the tag is missing);
    capitals are boosted so they survive the cut. `limit` bounds label
    clutter, shared-border places deduplicate, and the result is deterministic.
    Missing caches fill on demand (never under NAAULU_BOUNDARIES_OFFLINE=1).
    Returns dicts with name/lon/lat/place/population.
    """
    _ensure_fetched(iso_codes, "CITIES")
    candidates, seen = [], set()
    for iso3, features in sorted(_load_cities(sorted(iso_codes)).items()):
        for feature in features:
            props = feature.get("properties") or {}
            name = props.get("name")
            if not name:
                continue
            lon, lat = feature["geometry"]["coordinates"][:2]
            if geom is not None and not geom.intersects(
                    shapely.geometry.Point(lon, lat)):
                continue
            key = (name, round(lon, 4), round(lat, 4))
            if key in seen:
                continue
            seen.add(key)
            place = props.get("place") or "town"
            population = _parse_population(props.get("population"))
            score = population if population is not None \
                else CITIES_DEFAULT_POPULATION.get(place, 5_000)
            if props.get("capital"):
                score = max(score, 100_000)
            candidates.append({"name": name, "lon": lon, "lat": lat,
                               "place": place, "population": population,
                               "score": score})
    candidates.sort(key=lambda c: (-c["score"], c["name"]))
    return [{k: c[k] for k in
             ("name", "lon", "lat", "place", "population")}
            for c in (candidates if not limit else candidates[:limit])]


def get_countries(*, geom=None, iso_only=False, year="2023", adm="0", min_ratio=0, iso_codes=None):
    """Country boundaries (or ISO codes) intersecting a geometry."""
    selected = []
    for country in pycountry.countries:
        if iso_codes is not None and country.alpha_3 not in iso_codes:
            continue
        try:
            # bulk enumeration: read-only, a missing country is skipped
            # instead of triggering ~250 on-demand fetches
            country_geom = import_country(
                iso_code=country.alpha_3, year=year, adm=adm, fetch=False)
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


def get_admin0_ne10m(geom=None):
    """Country outlines from the bundled Natural Earth 10m file (--adm 1).

    A pure local read: no boundary cache, no fetch — the outlines are one
    file, so neighbouring countries match each other exactly; what it costs
    is detail (Belgium: 648 vertices — finer than the 50m plumbing file,
    still far coarser than OSM, whose linework --adm 2 and up use).
    """
    selected = []
    for country in _load_countries_10m().values():
        for poly in _parts(country):
            if geom is not None and not poly.intersects(geom):
                continue
            selected.append(poly)
    return shapely.geometry.MultiPolygon(selected)


def get_admin0(geom, iso_codes):
    """Admin-0 boundaries intersecting the geometry (filled on demand)."""
    _ensure_fetched(iso_codes, "ADM0")
    admin0_data = _load_admin0(iso_codes)
    selected = []
    for iso, geom_obj in admin0_data.items():
        if isinstance(geom_obj, shapely.geometry.MultiPolygon):
            for poly in geom_obj.geoms:
                if geom is not None and not poly.intersects(geom):
                    continue
                selected.append(poly)
        else:
            if geom is not None and not geom_obj.intersects(geom):
                continue
            selected.append(geom_obj)
    return shapely.geometry.MultiPolygon(selected)


def _parts(geometry):
    """Component polygons of a (Multi)Polygon."""
    return list(geometry.geoms) if hasattr(geometry, "geoms") else [geometry]


def get_admin2(geom, iso_codes):
    """Admin-2 boundaries intersecting the geometry (filled on demand)."""
    _ensure_fetched(iso_codes, "ADM2")
    admin2_data = _load_admin2(iso_codes)
    selected = []
    for iso, units in admin2_data.items():
        for region_geom in _parts(units):
            if geom is not None and not region_geom.intersects(geom):
                continue
            selected.append(region_geom)
    return shapely.geometry.MultiPolygon(selected)


def import_country_ne10m(iso_code, mainland_only=True):
    """Country window from the bundled Natural Earth 10m file (--adm 1).

    Same contract as import_country, different source: at --adm 1 the
    geometry that clips the data and the outlines that get drawn are this
    one NE geometry — what you see is the mask edge. Pure local read (no
    cache, no fetch, offline-safe); mainland_only keeps the largest part,
    mirroring import_country's OSM behaviour. Raises for a country the
    bundled file does not cover.
    """
    country = _load_countries_10m().get(iso_code)
    if country is None or country.is_empty:
        raise KeyError(
            f"No Natural Earth 10m boundary for {iso_code}: the bundled "
            f"file covers {len(_load_countries_10m())} countries — check "
            f"the ISO3 code")
    geoms = _parts(country)
    if mainland_only and len(geoms) > 1:
        geoms = sorted(geoms, key=lambda g: g.area, reverse=True)[:1]
    return shapely.geometry.MultiPolygon(geoms)


def import_country(iso_code=None, year="2023", adm="0", mainland_only=True,
                   fetch=True):
    """Load a country boundary from the OSM boundary cache (ADM0).

    mainland_only keeps just the largest polygon (the mainland; OSM country
    relations also carry islands and overseas territories). `year` is ignored:
    OSM geometry is versioned by its osmTimestamp sidecar, not release years.
    `fetch` fills a missing outline on demand (one Postpass query, then it
    is cache); bulk enumerations pass fetch=False to stay read-only.
    """
    if adm != "0":
        raise NotImplementedError(f"only ADM0 is supported, got ADM{adm}")

    cache_key = f"{iso_code}_{year}_ADM{adm}{'_mainland' if mainland_only else ''}"
    if cache_key in _country_cache:
        return _country_cache[cache_key]

    if fetch and iso_code:
        _ensure_fetched([iso_code], "ADM0")
    boundaries = _load_admin0([iso_code]) if iso_code else {}
    geom_obj = boundaries.get(iso_code)
    if geom_obj is None or geom_obj.is_empty:
        hint = ("NAAULU_BOUNDARIES_OFFLINE is set — prefetch this country's "
                "boundaries at build time" if _boundaries_offline()
                else "its on-demand Postpass fetch failed or OSM has no "
                     "country-level relation for it — run "
                     "prefetch_boundaries.py to see the same error fixed-"
                     "quietly at build time")
        raise KeyError(f"No OSM boundary for {iso_code}: {hint}")

    geoms = _parts(geom_obj)
    if mainland_only and len(geoms) > 1:
        geoms = sorted(geoms, key=lambda g: g.area, reverse=True)[:1]

    geometry = shapely.geometry.MultiPolygon(geoms)
    _country_cache[cache_key] = geometry
    return geometry


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


def country_code(name, alpha=3):
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

    if alpha == 3:
        return country.alpha_3

    if alpha == 2:
        return country.alpha_2


def import_window(window):

    lon_min, lat_min, lon_max, lat_max = window
    geometry = shapely.geometry.box(lon_min, lat_min, lon_max, lat_max)

    return geometry


def buffer_geometry(geometry, distance):
    """Buffer a geometry by a distance in meters."""
    if distance == 0:
        return(geometry)
    centroid = geometry.centroid
    aeqd_crs = pyproj.crs.CRS(
        f"+proj=aeqd +lat_0={centroid.y} +lon_0={centroid.x} +datum=WGS84"
    )
    wgs84_crs = pyproj.crs.CRS("EPSG:4326")
    to_aeqd = pyproj.transformer.Transformer.from_crs(wgs84_crs, aeqd_crs, always_xy=True)
    to_wgs = pyproj.transformer.Transformer.from_crs(aeqd_crs, wgs84_crs, always_xy=True)

    geom_aeqd = shapely.ops.transform(to_aeqd.transform, geometry)
    buffered_aeqd = geom_aeqd.buffer(distance)
    return shapely.ops.transform(to_wgs.transform, buffered_aeqd)


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

    if distance == 0:
        return shapely.geometry.Point(location)

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
    """Geodesic distance in meters from location to geometry (0 if inside)."""
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
