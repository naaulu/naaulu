"""Offline tests for the OSM boundary/city caches (Geofabrik Postpass)."""

import json
import os

import pytest
import shapely.geometry

import naaulu.config
from naaulu import geography


# ─── synthetic OSM world ────────────────────────────────────────────────────
# BEL is a plain unit square; FRA shares BEL's east edge *verbatim* (as
# adjacent OSM relations share boundary ways in one database) and extends a
# territorial sea bulge eastwards, closed by maritime=yes ways — stored as
# OSM serves it, then cut back at the synthetic land mask (x <= 2.2) at
# ingest. Everything below the fetch layer consumes polygon features: the
# shape Postpass serves a boundary relation in.

def _feature(tags, *coords):
    """One polygon feature: tags plus a closed ring of (lon, lat) points."""
    return {"type": "Feature",
            "properties": {"tags": tags},
            "geometry": {"type": "Polygon",
                         "coordinates": [[(x, y) for x, y in coords]]}}


def _point(lon, lat, tags):
    return {"type": "Feature",
            "properties": {"tags": tags},
            "geometry": {"type": "Point", "coordinates": [lon, lat]}}


BE_RING = ((0, 0), (1, 0), (1, 1), (0, 1), (0, 0))
FR_RING = ((1, 0), (1, 1), (2, 1), (2.5, 1), (2.5, 0), (2, 0), (1, 0))

BE_ADM0_F = _feature({"boundary": "administrative", "ISO3166-1": "BE",
                      "admin_level": "2", "name": "Belgium"}, *BE_RING)
FR_ADM0_F = _feature({"boundary": "administrative", "ISO3166-1": "FR",
                      "admin_level": "2", "name": "France"}, *FR_RING)
BE_UNIT_F = _feature({"boundary": "administrative", "ISO3166-2": "BE-XX",
                      "admin_level": "6", "name": "Belgique"}, *BE_RING)
FR_UNIT_F = _feature({"boundary": "administrative", "ISO3166-2": "FR-XX",
                      "admin_level": "6", "name": "Région"}, *FR_RING)

PLACES = [
    # inside FRA (1..2.5 x 0..1) unless noted
    _point(1.5, 0.5, {"place": "city", "name": "Metropolis",
                      "population": "120000"}),
    _point(2.2, 0.5, {"place": "town", "name": "Bourg", "population": "3 000"}),
    _point(1.8, 0.8, {"place": "town", "name": "Capitale",
                      "population": "500", "capital": "yes"}),
    # outside every fixture country: must never be attributed to FRA
    _point(10.0, 0.5, {"place": "city", "name": "Elsewhere",
                       "population": "900000"}),
]


@pytest.fixture
def cache(monkeypatch, tmp_path):
    """Isolated data dir; tests never touch the network."""
    monkeypatch.setattr(naaulu.config, "get_data_dir",
                        lambda subdir=None: str(tmp_path))
    monkeypatch.setattr(naaulu.config, "get_cache_dir",
                        lambda subdir=None: str(tmp_path / "cache"))
    monkeypatch.setattr(geography, "_osm_cache", {})
    monkeypatch.setattr(geography, "_cities_cache", {})
    monkeypatch.setattr(geography, "_country_cache", {})
    # Synthetic coast cut for the fixture world (0..2.5 E): land up to 2.2,
    # so the FR maritime bulge is clipped exactly like real data is.
    monkeypatch.setattr(geography, "_load_ne10m_land",
                        lambda: shapely.geometry.box(-10.0, -10.0, 2.2, 10.0))
    return tmp_path


def fake_fetch(monkeypatch, units=(), places=(), calls=None):
    """Fake the two Postpass fetchers with a fixed feature world.

    The fake honours the admin_level argument the way the real SQL does, so
    a test can prove which tier a prefetch asked for.
    """
    units = list(units)

    def fetch_units(iso3, admin_level):
        if calls is not None:
            calls.append((iso3, admin_level))
        return [u for u in units
                if u["properties"]["tags"].get("admin_level")
                == str(admin_level)]

    def fetch_places(iso3):
        if calls is not None:
            calls.append((iso3, "CITIES"))
        return list(places)

    monkeypatch.setattr(geography, "_fetch_units", fetch_units)
    monkeypatch.setattr(geography, "_fetch_places", fetch_places)


# ─── ingestion (the polygons below the fetch layer) ─────────────────────────

def test_borders_come_from_one_shared_linework(cache):
    """Neighbours built from the *same* linework yield an exact seam."""
    geography._ingest_units([BE_ADM0_F, FR_ADM0_F], ["BEL", "FRA"], "ADM0")

    result = geography._load_admin0(["BEL", "FRA"])
    assert sorted(result) == ["BEL", "FRA"]
    assert result["BEL"].intersection(result["FRA"]).area == 0
    shared = result["BEL"].boundary.intersection(result["FRA"].boundary)
    assert shared.length == 1.0, "the whole facing edge must be one line"

    _, meta_path = geography._osm_paths("BEL", "ADM0")
    with open(meta_path, encoding="utf-8") as handle:
        sidecar = json.load(handle)
    assert "Open Database License" in sidecar["license"]
    assert "OpenStreetMap contributors" in sidecar["attribution"]


def test_territorial_sea_is_cut_at_the_land_mask(cache):
    """The relation stores the maritime bulge; the prefetch cuts it back at
    the shared land line — the drawn outline stops where land does."""
    geography._ingest_units([FR_ADM0_F], ["FRA"], "ADM0")

    fra = geography._load_admin0(["FRA"])["FRA"]
    assert abs(fra.area - 1.2) < 1e-9, \
        "sea beyond the mask (x>2.2) is cut, land+inner water kept"
    assert fra.bounds == (1.0, 0.0, 2.2, 1.0), \
        "the outline ends at the land line, not at x=2.5"


def test_units_without_iso_tags_fall_back_to_natural_earth(cache):
    """Spatial queries return neighbours too; tagless units get NE."""
    in_france = _feature(
        {"boundary": "administrative", "admin_level": "6", "name": "Untagged"},
        (2.0, 47.0), (2.5, 47.0), (2.5, 47.5), (2.0, 47.5), (2.0, 47.0))

    geography._ingest_units([in_france], ["FRA"], "ADM2")

    assert "FRA" in geography._load_admin2(["FRA"]), \
        "square inside France must be attributed to FRA"


def test_a_mapping_change_invalidates_the_cached_level(cache, monkeypatch):
    """ADM2 files record the admin_level they were fetched at: a mapping
    change must void the file instead of silently serving the old tier."""
    geography._ingest_units([FR_UNIT_F], ["FRA"], "ADM2")
    assert "FRA" in geography._load_admin2(["FRA"])

    monkeypatch.setitem(geography.OSM_PROVINCE_LEVEL, "FRA", 4)

    assert geography._read_osm_cache("FRA", "ADM2") is None, \
        "mapping change must void the cached tier"


# ─── province tier mapping (ADM2, the one drawn sub-national level) ─────────

def test_province_levels_follow_the_per_country_table():
    """The province tier sits at a different admin_level per country."""
    assert geography._province_level("FRA") == 6, "FR département"
    assert geography._province_level("NLD") == 4, "NL provincie"
    assert geography._province_level("CHE") == 4, "CH canton"
    assert geography._province_level("DEU") == 4, "DE Land"
    assert geography._province_level("LVA") == 5, "LV novads"
    assert geography._province_level("HRV") == 4, "HR županija"
    assert geography._province_level("ROU") == 4, "RO județ"
    assert geography._province_level("MDA") == 4, "MD raion"
    assert geography._province_level("POL") == 4, "PL województwo"
    assert geography._province_level("CZE") == 4, "CZ kraj"
    assert geography._province_level("AND") == 7, "AND parroquia"
    assert geography._province_level("DNK") == 7, "DK kommune"
    assert geography._province_level("NOR") == 7, "NO kommune"
    assert geography._province_level("SWE") == 7, "SE kommun"
    assert geography._province_level("MKD") == 7, "MK opština"
    assert geography._province_level("ALA") == 8, "Åland kommune"
    assert geography._province_level("JEY") == 8, "JE parish"
    assert geography._province_level("GGY") == 8, "GG parish"
    assert geography._province_level("LIE") == 8, "FL Gemeinde"
    assert geography._province_level("SVN") == 8, "SI občina"
    assert geography._province_level("MLT") == 8, "MT local council"
    assert geography._province_level("SMR") == 8, "SMR castello"
    assert geography._province_level("MCO") == 10, "MC quartier"
    assert geography._province_level("XXX") == geography.OSM_PROVINCE_LEVEL_DEFAULT


# ─── prefetch from Postpass (queries, never a planet download) ──────────────

def test_prefetch_fills_every_requested_tier(cache, monkeypatch):
    """One country's queries fill ADM0, ADM2 and CITIES caches, in the same
    shapes the rest of the pipeline has always consumed."""
    fake_fetch(monkeypatch, units=[FR_ADM0_F, FR_UNIT_F], places=PLACES[:3])

    geography._prefetch_country("FRA", ["ADM0", "ADM2", "CITIES"])

    adm0 = geography._load_admin0(["FRA"])["FRA"]
    assert adm0.bounds == (1.0, 0.0, 2.2, 1.0)
    adm2 = geography._load_admin2(["FRA"])["FRA"]
    assert adm2.bounds == (1.0, 0.0, 2.2, 1.0)
    cities = geography.get_cities(
        shapely.geometry.box(-20, -10, 20, 10), ["FRA"])
    assert [c["name"] for c in cities] == ["Metropolis", "Capitale", "Bourg"], \
        "population-ranked, capital boosted above raw size"


def test_prefetch_keeps_only_the_province_level(cache, monkeypatch):
    """The country's mapping decides which units become the ADM2 tier: the
    fetch asks for that level and never for any other."""
    lv9 = _feature({"boundary": "administrative", "ISO3166-2": "FR-99",
                    "admin_level": "9", "name": "too fine"},
                   (5, 5), (6, 5), (6, 6), (5, 6), (5, 5))
    calls = []
    fake_fetch(monkeypatch, units=[FR_ADM0_F, FR_UNIT_F, lv9], calls=calls)

    geography._prefetch_country("FRA", ["ADM0", "ADM2"])

    adm2 = geography._load_admin2(["FRA"])["FRA"]
    assert adm2.bounds == (1.0, 0.0, 2.2, 1.0), "only the lv6 unit is kept"
    assert ("FRA", 9) not in calls, "the province tier must not be over-fetched"


def test_prefetch_drops_foreign_spill(cache, monkeypatch):
    """A spatial query returns neighbouring units too; only this country's
    land ends up in its cache."""
    fake_fetch(monkeypatch,
               units=[BE_ADM0_F, FR_ADM0_F, BE_UNIT_F, FR_UNIT_F])

    geography._prefetch_country("BEL", ["ADM0", "ADM2"])

    assert geography._load_admin2(["BEL"])["BEL"].bounds == (0.0, 0.0, 1.0, 1.0)
    assert "FRA" not in geography._load_admin2(["BEL"])


def test_cities_fall_back_to_natural_earth_without_adm0(cache, monkeypatch):
    """OSM can lack a country-level relation entirely (Åland: only a
    level-4 region and its municipalities) — labels then scope and
    attribute against bundled Natural Earth instead."""
    monkeypatch.setattr(
        geography, "_load_countries",
        lambda: {"FRA": shapely.geometry.box(0, 0, 3, 2)})
    fake_fetch(monkeypatch, places=PLACES[:3])

    geography._prefetch_country("FRA", ["CITIES"])

    cities = geography.get_cities(
        shapely.geometry.box(-20, -10, 20, 10), ["FRA"])
    assert [c["name"] for c in cities] == ["Metropolis", "Capitale", "Bourg"]


def test_prefetch_queries_once_and_fills(cache, monkeypatch):
    """Cached tiers are not re-queried; only missing ones are."""
    calls = []
    fake_fetch(monkeypatch, units=[FR_ADM0_F, FR_UNIT_F], calls=calls)

    filled = geography.prefetch_boundaries(["FRA"], levels=("ADM0", "ADM2"))
    assert filled["ADM0"] == ["FRA"]
    assert calls == [("FRA", 2), ("FRA", 6)]

    assert geography.prefetch_boundaries(["FRA"], levels=("ADM0", "ADM2"))
    assert calls == [("FRA", 2), ("FRA", 6)], "cached tiers must not re-query"


def test_prefetch_refills_after_a_mapping_change(cache, monkeypatch, caplog):
    """A tier voided by a changed OSM_PROVINCE_LEVEL is re-queried at the
    next prefetch — never served from the stale file, never left empty
    without a word."""
    fake_fetch(monkeypatch, units=[FR_UNIT_F])
    geography.prefetch_boundaries(["FRA"], levels=("ADM2",))
    assert "FRA" in geography._load_admin2(["FRA"])

    monkeypatch.setitem(geography.OSM_PROVINCE_LEVEL, "FRA", 4)
    calls = []
    fake_fetch(monkeypatch, units=[], calls=calls)  # lv4 finds nothing

    with caplog.at_level("WARNING"):
        geography.prefetch_boundaries(["FRA"], levels=("ADM2",))

    assert calls == [("FRA", 4)], "the voided tier must be re-queried at lv4"
    assert any("OSM_PROVINCE_LEVEL" in record.message
               for record in caplog.records), \
        "an empty province tier must warn about the mapping"


def test_prefetch_refresh_re_queries(cache, monkeypatch):
    calls = []
    fake_fetch(monkeypatch, units=[FR_ADM0_F], calls=calls)
    geography.prefetch_boundaries(["FRA"], levels=("ADM0",))
    geography.prefetch_boundaries(["FRA"], levels=("ADM0",), refresh=True)

    assert calls == [("FRA", 2), ("FRA", 2)]


# ─── offline runtime (realtime containers) ──────────────────────────────────

def test_internal_loaders_stay_read_only(cache):
    """Raw reads never fetch or write: fills happen in prefetch or in the
    on-demand step of the public draw helpers, never inside a bare load."""
    assert geography._load_admin0(["FRA"]) == {}

    geojson, _ = geography._osm_paths("FRA", "ADM0")
    assert not os.path.exists(geojson), "a read must not create files"


def test_draws_fetch_missing_boundaries_on_demand(cache, monkeypatch):
    """A cold plot fills exactly what the window needs — once — and every
    later render reads the cache."""
    calls = []
    fake_fetch(monkeypatch, units=[FR_ADM0_F, FR_UNIT_F], calls=calls)
    window = shapely.geometry.box(0, 0, 3, 2)

    first = geography.get_admin0(window, ["FRA"])
    assert len(first.geoms) == 1, "the window got its country outline"
    assert calls == [("FRA", 2)], "one fetch for the missing tier"

    geography.get_admin0(window, ["FRA"])
    assert calls == [("FRA", 2)], "the warm cache must not re-fetch"


def test_draws_fetch_adm2_with_its_adm0_scope_first(cache, monkeypatch):
    """The province tier scopes by the country outline, so an on-demand
    ADM2 fill fetches ADM0 first (when it is missing too)."""
    calls = []
    fake_fetch(monkeypatch, units=[FR_ADM0_F, FR_UNIT_F], calls=calls)

    selected = geography.get_admin2(
        shapely.geometry.box(0, 0, 3, 2), ["FRA"])

    assert len(selected.geoms) == 1, "the FR unit is drawn"
    assert calls == [("FRA", 2), ("FRA", 6)], "ADM0 first, then the tier"


# ─── country windows (--country geometry) ───────────────────────────────────

def test_import_country_window_comes_from_the_osm_cache(cache, monkeypatch):
    """--country windows are OSM ADM0 geometry (largest part = mainland);
    Natural Earth must not be consulted anywhere in that path."""
    def no_ne():
        raise AssertionError("import_country must not use Natural Earth")
    monkeypatch.setattr(geography, "_load_countries", no_ne)

    island = _feature({"ISO3166-1": "FR"},
                      (5, 0), (5.5, 0), (5.5, 0.5), (5, 0.5), (5, 0))
    geography._ingest_units([FR_ADM0_F, island], ["FRA"], "ADM0")

    full = geography.import_country("FRA", mainland_only=False)
    mainland = geography.import_country("FRA", mainland_only=True)

    assert len(full.geoms) == 2, \
        "every OSM part survives (the islet outside the mask is kept)"
    assert len(mainland.geoms) == 1, "mainland_only keeps the largest part"
    assert mainland.bounds == (1.0, 0.0, 2.2, 1.0), \
        "the mainland outline stops at the land mask"


def test_import_country_fetches_on_demand(cache, monkeypatch):
    """A cold --country window fills itself instead of raising."""
    fake_fetch(monkeypatch, units=[FR_ADM0_F])

    geometry = geography.import_country("FRA", mainland_only=False)

    assert geometry.bounds == (1.0, 0.0, 2.2, 1.0)
    geojson, _ = geography._osm_paths("FRA", "ADM0")
    assert os.path.exists(geojson), "the on-demand fill must persist"


def test_coast_clip_rules(cache):
    """The three rules of the land clip: no-op on land, keeps islets the
    mask does not know, and repairs invalid OSM polygons without losing area."""
    # covered by the fixture mask -> byte-identical result
    square = shapely.geometry.box(0, 0, 1, 1)
    assert geography._clip_to_land(square).equals(square)

    # a piece outside the mask entirely = land below NE's mapping threshold
    # (Sark-class islet): kept as OSM drew it, so no unit vanishes
    islet = shapely.geometry.box(5.0, 5.0, 5.5, 5.5)
    assert geography._clip_to_land(islet).equals(islet)

    # invalid self-touching ring (two lobes pinched at one vertex) is
    # repaired with buffer(0): valid output, area exactly preserved
    bowtie = shapely.geometry.Polygon(
        [(0.2, 0.2), (1.4, 0.2), (0.8, 0.5), (1.4, 0.8),
         (0.2, 0.8), (0.8, 0.5), (0.2, 0.2)])
    assert not bowtie.is_valid
    clipped = geography._clip_to_land(bowtie)
    assert clipped.is_valid, "the stored geometry must come out valid"
    assert abs(clipped.area - 0.36) < 1e-9, "repair keeps area exactly"


def test_pre_clip_caches_are_voided_and_refetch(cache):
    """Every current ADM0/ADM2 sidecar records its coast line; files without
    it predate the clip (12 nm sea still stored) and must be voided instead
    of served."""
    # every current ADM0/ADM2 sidecar records the clip recipe it was made
    # with; files with the wrong recipe (pre-clip, or ADM2 from the broken
    # aggregate repair) must not be served
    geography._ingest_units([FR_ADM0_F], ["FRA"], "ADM0")
    geojson, meta_path = geography._osm_paths("FRA", "ADM0")
    meta = json.load(open(meta_path))
    assert meta["coastline"] == "ne10m", "ADM0 caches record their coast"
    assert geography._read_osm_cache("FRA", "ADM0") is not None

    meta.pop("coastline")
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle)
    geography._osm_cache.pop(("FRA", "ADM0"), None)

    assert geography._read_osm_cache("FRA", "ADM0") is None, \
        "a pre-clip cache must not be served"
    assert geography._tier_missing("FRA", "ADM0"), \
        "prefetch/on-demand must refill it"

    # the same rule guards ADM2 (its adminLevel check still applies too)
    geography._ingest_units([FR_UNIT_F], ["FRA"], "ADM2")
    _, meta_path = geography._osm_paths("FRA", "ADM2")
    meta = json.load(open(meta_path))
    assert meta["coastline"] == "ne10m:pieces", \
        "ADM2 records the per-piece recipe"
    meta.pop("coastline")
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle)
    geography._osm_cache.pop(("FRA", "ADM2"), None)
    assert geography._read_osm_cache("FRA", "ADM2") is None


def test_coast_clip_keeps_touching_units_separate(cache):
    """Provinces share borders — the clip must never dissolve them into one
    country silhouette. (buffer(0) on the *aggregate* does exactly that:
    area identical, interior lines gone. The repair runs per piece.)"""
    paired = shapely.geometry.MultiPolygon(
        [shapely.geometry.box(0, 0, 1, 1), shapely.geometry.box(1, 0, 2, 1)])
    assert not paired.is_valid, "edge-touching members are OGC-invalid"

    out = geography._clip_to_land(paired)

    assert out.geom_type == "MultiPolygon" and len(out.geoms) == 2, \
        "adjacent units must stay separate after the clip"
    assert all(abs(p.area - 1.0) < 1e-9 for p in out.geoms), \
        "both units keep their full shape inside the mask"


def test_ingest_keeps_interior_borders_between_touching_units(cache):
    """End to end: two adjacent units of one country stay two polygons with
    a visible shared border in the cache — the province lines survive."""
    neighbour = _feature({"boundary": "administrative", "admin_level": "6",
                          "ISO3166-2": "BE-YY", "name": "Second unit"},
                         (1, 0), (2, 0), (2, 1), (1, 1), (1, 0))

    geography._ingest_units([BE_UNIT_F, neighbour], ["BEL"], "ADM2")

    units = geography._load_admin2(["BEL"])["BEL"]
    parts = list(units.geoms) if hasattr(units, "geoms") else [units]
    assert len(parts) == 2, \
        f"two adjacent units must be stored separately, got {len(parts)}"
    assert abs(units.area - 2.0) < 1e-9
    shared = parts[0].boundary.intersection(parts[1].boundary)
    assert abs(shared.length - 1.0) < 1e-9, \
        "the shared border must remain drawn (1 deg of common edge)"


# ─── bundled Natural Earth 10m outlines (--adm 1) ──────────────────────────

def _vertices(g):
    parts = g.geoms if g.geom_type == "MultiPolygon" else [g]
    return sum(len(p.exterior.coords)
               + sum(len(r.coords) for r in p.interiors) for p in parts)


def test_ne10m_is_the_finer_bundled_scale(cache):
    """Both NE scales load keyed by ISO3, and the drawn 10m file is
    genuinely finer — Belgium is the documented case (648 vs 164)."""
    coarse = geography._load_countries()
    fine = geography._load_countries_10m()

    assert "BEL" in coarse and "BEL" in fine
    assert _vertices(fine["BEL"]) > 3 * _vertices(coarse["BEL"]), \
        "the drawn 10m file must be the finer one"


def test_admin0_ne10m_is_a_pure_local_read(cache, monkeypatch):
    """--adm 1's country outlines come from the bundled 10m file: no
    fetch, no boundary cache — and only window countries come back."""
    def boom(*args, **kwargs):
        raise AssertionError("the NE path must not fetch")
    monkeypatch.setattr(geography, "_prefetch_country", boom)

    window = shapely.geometry.box(2.0, 49.0, 6.5, 52.0)
    outlines = geography.get_admin0_ne10m(window)

    assert not outlines.is_empty
    assert any(p.covers(shapely.geometry.Point(4.35, 50.85))
               for p in outlines.geoms), "Belgium is in the window"
    assert not any(p.covers(shapely.geometry.Point(-3.7, 40.4))
                   for p in outlines.geoms), "Spain is not"
    assert geography.get_admin0_ne10m(
        shapely.geometry.box(-40, 0, -39, 1)).is_empty, \
        "a mid-ocean window draws nothing"


def test_import_country_ne10m_is_the_drawn_geometry(cache, monkeypatch):
    """The --adm 1 window IS the NE10m geometry the same detail draws —
    one source for clip and border alike, read locally without any fetch."""
    def boom(*args, **kwargs):
        raise AssertionError("the NE window must not fetch")
    monkeypatch.setattr(geography, "_prefetch_country", boom)

    full = geography._load_countries_10m()["BEL"]
    default = geography.import_country_ne10m("BEL")
    everything = geography.import_country_ne10m("BEL", mainland_only=False)

    assert everything.equals(full), "the window is the bundled geometry"
    largest = max(full.geoms, key=lambda g: g.area)
    assert default.equals(shapely.geometry.MultiPolygon([largest])), \
        "mainland_only mirrors import_country: largest part only"

    with pytest.raises(KeyError, match="Natural Earth 10m"):
        geography.import_country_ne10m("XXX")


# ─── cities (OSM populated places) ──────────────────────────────────────────

def test_draws_fetch_missing_cities_on_demand(cache, monkeypatch):
    """A cold plot fetches its labels once (ADM0 first — it attributes the
    places), then the cache serves them like every other tier."""
    calls = []
    fake_fetch(monkeypatch, units=[FR_ADM0_F], places=PLACES[:3], calls=calls)
    window = shapely.geometry.box(-20, -10, 20, 10)

    cities = geography.get_cities(window, ["FRA"])
    assert [c["name"] for c in cities] == ["Metropolis", "Capitale", "Bourg"]
    assert ("FRA", 2) in calls and ("FRA", "CITIES") in calls, \
        "ADM0 first, then the place nodes"

    before = len(calls)
    geography.get_cities(window, ["FRA"])
    assert len(calls) == before, "the warm cache must not re-fetch"


def test_cities_carry_the_odbl_sidecar(cache, monkeypatch):
    fake_fetch(monkeypatch, units=[FR_ADM0_F], places=PLACES[:3])
    geography._prefetch_country("FRA", ["ADM0", "CITIES"])

    _, meta_path = geography._osm_paths("FRA", "CITIES")
    with open(meta_path, encoding="utf-8") as handle:
        sidecar = json.load(handle)
    assert "OpenStreetMap" in sidecar["source"]
    assert "Open Database License" in sidecar["license"]
    assert sidecar["level"] == "CITIES"


def test_cities_rank_capitals_and_honour_the_limit(cache, monkeypatch):
    """A capital's boost outranks much bigger plain towns; the limit bounds
    label clutter deterministically."""
    fake_fetch(monkeypatch, units=[FR_ADM0_F], places=PLACES[:3])
    geography._prefetch_country("FRA", ["ADM0", "CITIES"])
    window = shapely.geometry.box(-20, -10, 20, 10)

    top = geography.get_cities(window, ["FRA"], limit=1)
    two = geography.get_cities(window, ["FRA"], limit=2)

    assert [c["name"] for c in top] == ["Metropolis"]
    assert top[0]["population"] == 120000
    assert two[1]["name"] == "Capitale", "capital (pop 500) beats Bourg (3 000)"


def test_cities_filter_to_the_country(cache, monkeypatch):
    """Nodes outside every requested country drop at attribution."""
    fake_fetch(monkeypatch, units=[FR_ADM0_F], places=PLACES)
    geography._prefetch_country("FRA", ["ADM0", "CITIES"])

    cities = geography.get_cities(
        shapely.geometry.box(-20, -10, 20, 10), ["FRA"], limit=10)

    assert "Elsewhere" not in [c["name"] for c in cities]


# ─── Postpass query layer (SQL over OSM's planet database) ──────────────────

class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _capture_post(monkeypatch, sent):
    """Fake requests.post, recording every SQL statement sent."""
    def post(url, data=None, timeout=None, headers=None):
        sent.append((url, data["data"]))
        return _FakeResponse({"type": "FeatureCollection", "features": []})
    monkeypatch.setattr(geography.requests, "post", post)


def test_adm0_is_selected_by_its_iso_tag(cache, monkeypatch):
    """Country outlines query by ISO3166-1 tag: no neighbour spill."""
    sent = []
    _capture_post(monkeypatch, sent)

    geography._fetch_units("FRA", 2)

    url, sql = sent[0]
    assert url == geography.POSTPASS_URL
    assert "postpass_polygon" in sql
    assert "tags->>'admin_level' = '2'" in sql
    assert "tags->>'ISO3166-1' = 'FR'" in sql
    assert "ST_MakeEnvelope" not in sql, "the outline is tag-selected"


def test_adm2_is_scoped_spatially(cache, monkeypatch):
    """Province units carry no usable tags; they are bbox-scoped instead."""
    sent = []
    _capture_post(monkeypatch, sent)

    geography._fetch_units("FRA", 6)

    _, sql = sent[0]
    assert "tags->>'admin_level' = '6'" in sql
    assert "ST_MakeEnvelope" in sql
    assert "ISO3166-1" not in sql


def test_query_bounds_prefer_the_fetched_adm0(cache):
    """The cached ADM0 bounds a spatial query — overseas parts included,
    which Natural Earth does not link to their parent country."""
    geography._write_osm_cache(
        "FRA", "ADM0",
        shapely.geometry.MultiPolygon(
            [shapely.geometry.box(1.0, 0.0, 2.5, 1.0),    # mainland
             shapely.geometry.box(55.2, -21.0, 55.7, -20.9)]),  # overseas
        "test")

    bounds = geography._query_bounds("FRA")

    assert (55.2, -21.0, 55.7, -20.9) in bounds, \
        "the overseas part must scope its own queries"


def test_query_bounds_fall_back_to_natural_earth(cache):
    """Before ADM0 is fetched, the bundled Natural Earth country stands in;
    an unknown ISO3 fails loud instead of querying the whole planet."""
    bounds = geography._query_bounds("FRA")
    assert len(bounds) == 1 and bounds[0][0] < bounds[0][2]

    with pytest.raises(RuntimeError, match="Natural Earth"):
        geography._query_bounds("XXX")


def test_places_query_scopes_to_the_adm0_extent(cache, monkeypatch):
    """City nodes are fetched per country, bounded by its own outline."""
    geography._write_osm_cache(
        "FRA", "ADM0", shapely.geometry.box(1.0, 0.0, 2.5, 1.0), "test")
    sent = []
    _capture_post(monkeypatch, sent)

    geography._fetch_places("FRA")

    _, sql = sent[0]
    assert "postpass_point" in sql
    assert "tags->>'place' IN ('city', 'town')" in sql
    assert "ST_MakeEnvelope(1.0, 0.0, 2.5, 1.0, 4326)" in sql


def test_postpass_retries_transient_failures(cache, monkeypatch):
    """A 504 under load must not kill an image bake or an on-demand fill:
    the query retries with backoff and succeeds on a later attempt."""
    answers = [504, 504, None]  # None = success
    sent = []

    def post(url, data=None, timeout=None, headers=None):
        status = answers[len(sent)]
        sent.append(status)
        if status is not None:
            raise geography.requests.HTTPError(f"{status} Server Error")
        return _FakeResponse({"type": "FeatureCollection", "features": []})

    sleeps = []
    monkeypatch.setattr(geography.requests, "post", post)
    monkeypatch.setattr(geography.time, "sleep", sleeps.append)

    assert geography._postpass("SELECT 1") == []
    assert len(sent) == 3, "two failures, then success"
    assert sleeps == [2.0, 6.0], "exponential backoff between attempts"
