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
# territorial sea bulge eastwards, closed by maritime=yes ways — drawn as
# stored, so FRA's outline includes the bulge. Everything below the fetch
# layer consumes polygon features: the shape Postpass serves a boundary
# relation in.

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


def test_territorial_sea_is_drawn_as_stored(cache):
    """The maritime=yes ring closes the relation; the bulge is the outline."""
    geography._ingest_units([FR_ADM0_F], ["FRA"], "ADM0")

    fra = geography._load_admin0(["FRA"])["FRA"]
    assert abs(fra.area - 1.5) < 1e-9, "the sea bulge beyond x=2 is the shape"
    assert fra.bounds == (1.0, 0.0, 2.5, 1.0)


def test_empty_levels_are_negative_cached(cache):
    """A level with no units writes an empty file, so it is not re-ingested."""
    geography._ingest_units([], ["FRA"], "ADM0")

    assert geography._load_admin0(["FRA"]) == {}
    geojson, _ = geography._osm_paths("FRA", "ADM0")
    assert os.path.exists(geojson), "absent level must leave a marker"
    assert geography._load_admin0(["FRA"]) == {}


def test_units_without_iso_tags_fall_back_to_natural_earth(cache):
    """Spatial queries return neighbours too; tagless units get NE."""
    in_france = _feature(
        {"boundary": "administrative", "admin_level": "6", "name": "Untagged"},
        (2.0, 47.0), (2.5, 47.0), (2.5, 47.5), (2.0, 47.5), (2.0, 47.0))

    geography._ingest_units([in_france], ["FRA"], "ADM2")

    assert "FRA" in geography._load_admin2(["FRA"]), \
        "square inside France must be attributed to FRA"


def test_units_far_from_every_country_are_dropped(cache):
    """Geometry in the ocean (no NE country) is skipped, not guessed."""
    ocean = _feature({}, (0, 0), (1, 0), (1, 1), (0, 1), (0, 0))

    geography._ingest_units([ocean], ["FRA"], "ADM2")

    assert geography._load_admin2(["FRA"]) == {}


def test_units_outside_natural_earth_fall_back_to_the_query_extent(
        cache, monkeypatch):
    """Units NE cannot place attribute to the requested country holding them
    in its padded outline — Åland's tagless kommuner sit outside NE's
    too-tight archipelago polygon."""
    _, miny, _, _ = geography._load_countries()["ALA"].bounds
    outside_ne = _feature({"boundary": "administrative", "admin_level": "8",
                           "name": "Södra skären"},
                          (20.0, miny - 0.10), (20.1, miny - 0.10),
                          (20.1, miny - 0.05), (20.0, miny - 0.05),
                          (20.0, miny - 0.10))
    assert geography._iso3_by_ne_point(
        shapely.geometry.shape(outside_ne["geometry"])) is None, \
        "the fixture must sit outside exact Natural Earth"

    geography._ingest_units([outside_ne], ["ALA"], "ADM2")

    assert "ALA" in geography._load_admin2(["ALA"]), \
        "the padded outline must catch what NE cannot place"


def test_a_mapping_change_invalidates_the_cached_level(cache, monkeypatch):
    """ADM2 files record the admin_level they were fetched at: a mapping
    change must void the file instead of silently serving the old tier."""
    geography._ingest_units([FR_UNIT_F], ["FRA"], "ADM2")
    assert "FRA" in geography._load_admin2(["FRA"])

    monkeypatch.setitem(geography.OSM_PROVINCE_LEVEL, "FRA", 4)

    assert geography._read_osm_cache("FRA", "ADM2") is None, \
        "mapping change must void the cached tier"


def test_legacy_level_files_without_the_key_still_load(cache):
    """Sidecars written before adminLevel existed keep working."""
    baked = shapely.geometry.box(1.0, 0.0, 2.0, 1.0)
    geography._write_osm_cache("FRA", "ADM2", baked, "test")  # no admin_level

    result = geography._load_admin2(["FRA"])

    assert "FRA" in result
    assert geography._load_admin2(["FRA"])["FRA"].equals(baked)


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
    assert adm0.bounds == (1.0, 0.0, 2.5, 1.0)
    adm2 = geography._load_admin2(["FRA"])["FRA"]
    assert adm2.bounds == (1.0, 0.0, 2.5, 1.0)
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
    assert adm2.bounds == (1.0, 0.0, 2.5, 1.0), "only the lv6 unit is kept"
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


def test_cities_need_some_geometry_to_scope_by(cache, monkeypatch):
    """No ADM0 and no Natural Earth coverage: no query and no file — the
    prefetch fails loud instead of guessing where the country is."""
    monkeypatch.setattr(geography, "_load_countries", lambda: {})
    fake_fetch(monkeypatch, places=PLACES[:1])

    geography._prefetch_country("FRA", ["CITIES"])

    geojson, _ = geography._osm_paths("FRA", "CITIES")
    assert not os.path.exists(geojson), "no extent → no city file"


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


def test_prefetch_fails_loudly_instead_of_shipping_gaps(cache, monkeypatch):
    """A build step must exit non-zero when the service is unreachable."""
    def broken_fetch(iso3, admin_level):
        raise RuntimeError(f"Postpass query failed for {iso3}: 503")

    monkeypatch.setattr(geography, "_fetch_units", broken_fetch)

    with pytest.raises(RuntimeError, match="Postpass query failed"):
        geography.prefetch_boundaries(["FRA"], levels=("ADM0",))


def test_unknown_countries_fail_with_guidance(cache):
    """An ISO3 code without an ISO3166-1 mapping fails loud before any
    query is sent, never with a silent empty cache."""
    with pytest.raises(RuntimeError, match="ISO3166-1"):
        geography.prefetch_boundaries(["XXX"], levels=("ADM0",))


def test_prefetch_refuses_to_run_in_offline_mode(cache, monkeypatch):
    monkeypatch.setenv("NAAULU_BOUNDARIES_OFFLINE", "1")

    with pytest.raises(RuntimeError, match="OFFLINE"):
        geography.prefetch_boundaries(["FRA"])


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


def test_offline_draws_never_fetch_on_demand(cache, monkeypatch):
    """NAAULU_BOUNDARIES_OFFLINE=1 keeps serves strictly local: a gap stays
    a gap instead of a network call mid-serve."""
    monkeypatch.setenv("NAAULU_BOUNDARIES_OFFLINE", "1")

    def boom(*args, **kwargs):
        raise AssertionError("offline mode must never fetch")
    monkeypatch.setattr(geography, "_prefetch_country", boom)

    assert geography.get_admin0(
        shapely.geometry.box(0, 0, 3, 2), ["FRA"]).is_empty
    assert geography.get_admin2(
        shapely.geometry.box(0, 0, 3, 2), ["FRA"]).is_empty


def test_offline_mode_still_reads_the_baked_cache(cache, monkeypatch):
    monkeypatch.setenv("NAAULU_BOUNDARIES_OFFLINE", "1")
    baked = shapely.geometry.box(1.0, 0.0, 2.0, 1.0)
    geography._write_osm_cache("FRA", "ADM0", baked, "test")

    result = geography._load_admin0(["FRA"])

    assert "FRA" in result


def test_offline_mode_changes_nothing_for_reads(cache, monkeypatch):
    """Draws are read-only either way; the flag only guards prefetches."""
    geography._ingest_units([FR_ADM0_F], ["FRA"], "ADM0")
    monkeypatch.setenv("NAAULU_BOUNDARIES_OFFLINE", "1")

    assert "FRA" in geography._load_admin0(["FRA"])


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

    assert len(full.geoms) == 2, "OSM carries every part (islands included)"
    assert len(mainland.geoms) == 1, "mainland_only keeps the largest part"
    assert mainland.bounds == (1.0, 0.0, 2.5, 1.0)


def test_import_country_fails_loud_without_a_cache_entry(cache, monkeypatch):
    """A missing ADM0 cache entry must explain the fix, not draw nothing."""
    monkeypatch.setenv("NAAULU_BOUNDARIES_OFFLINE", "1")

    with pytest.raises(KeyError, match="OFFLINE"):
        geography.import_country("FRA")


def test_import_country_fetches_on_demand(cache, monkeypatch):
    """A cold --country window fills itself instead of raising."""
    fake_fetch(monkeypatch, units=[FR_ADM0_F])

    geometry = geography.import_country("FRA", mainland_only=False)

    assert geometry.bounds == (1.0, 0.0, 2.5, 1.0)
    geojson, _ = geography._osm_paths("FRA", "ADM0")
    assert os.path.exists(geojson), "the on-demand fill must persist"


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


def test_cities_respect_offline_mode(cache, monkeypatch):
    """Offline renders label only baked places — never the network."""
    monkeypatch.setenv("NAAULU_BOUNDARIES_OFFLINE", "1")

    result = geography.get_cities(shapely.geometry.box(0, 0, 3, 2), ["FRA"])

    assert result == []


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


def test_postpass_gives_up_when_the_service_is_down(cache, monkeypatch):
    """A persistent failure still raises, so prefetches (and image bakes)
    fail loud instead of shipping a half-filled cache."""
    sent = []

    def post(url, data=None, timeout=None, headers=None):
        sent.append(1)
        raise geography.requests.HTTPError("503 Service Unavailable")

    monkeypatch.setattr(geography.requests, "post", post)
    monkeypatch.setattr(geography.time, "sleep", lambda s: None)

    with pytest.raises(RuntimeError, match="Postpass query failed"):
        geography._postpass("SELECT 1")
    assert len(sent) == 3, "three attempts before giving up"


# ─── live service endpoint ──────────────────────────────────────────────────

@pytest.mark.network
def test_postpass_real_endpoint():
    """The real Postpass service serves Andorra into real caches."""
    geography._osm_cache.clear()
    for level in ("ADM0", "ADM2"):
        geojson, _ = geography._osm_paths("AND", level)
        if os.path.exists(geojson):
            os.remove(geojson)
    try:
        geography._prefetch_country("AND", ["ADM0", "ADM2"])
    except RuntimeError as exc:
        pytest.skip(f"Postpass unreachable: {exc}")

    adm0 = geography._load_admin0(["AND"])
    assert "AND" in adm0, "expected Andorra from the service"
    assert 0.03 < adm0["AND"].area < 0.06  # ~468 km²

    adm2 = geography._load_admin2(["AND"])
    assert "AND" in adm2, "expected Andorra's parroquies"
    assert 0.03 < adm2["AND"].area < 0.06, "units cover the same territory"


@pytest.mark.network
def test_postpass_countries_without_a_country_relation():
    """Åland has no admin_level=2 relation in OSM (only a level-4 region
    and its municipalities): its outline stays empty, but its city labels
    still fill through the Natural Earth fallback."""
    geography._osm_cache.clear()
    geojson, _ = geography._osm_paths("ALA", "CITIES")
    if os.path.exists(geojson):
        os.remove(geojson)
    try:
        geography._prefetch_country("ALA", ["ADM0", "CITIES"])
    except RuntimeError as exc:
        pytest.skip(f"Postpass unreachable: {exc}")

    assert geography._load_admin0(["ALA"]) == {}, \
        "OSM carries no country-level relation for Åland"
    cities = geography.get_cities(
        shapely.geometry.box(19.0, 59.5, 21.5, 61.0), ["ALA"])
    assert "Mariehamn" in [c["name"] for c in cities], \
        "labels must come from the NE-scoped query"
