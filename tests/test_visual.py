"""Offline tests for the plotting helpers (visual)."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot
import numpy
import pyproj

import naaulu.visual


# NE France mainland + the default 10 km buffer — what --country france plots
FRANCE_BOUNDS = (-4.9, 42.25, 8.28, 51.19)


def test_add_axis_frames_the_whole_window():
    """Axes limits must contain the full projected window.

    The image of a lon/lat box under LAEA is not a rectangle spanned by two
    opposite corners: the east edge bows toward the equator, so its SE corner
    sits ~80 km east of the NE corner for the France window. Framing by
    corners alone cut the eastern third (Lorraine/Alsace) off the plot.
    """
    crs = naaulu.visual.get_plot_crs(FRANCE_BOUNDS)
    fig, ax = matplotlib.pyplot.subplots()
    naaulu.visual.add_axis(ax, FRANCE_BOUNDS, crs)

    # dense sampling of the window outline, projected like add_axis does
    lon_min, lat_min, lon_max, lat_max = FRANCE_BOUNDS
    tf = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    n = 200
    lons = numpy.linspace(lon_min, lon_max, n)
    lats = numpy.linspace(lat_min, lat_max, n)
    edge_lons = numpy.concatenate(
        [lons, lons, numpy.full(n, lon_min), numpy.full(n, lon_max)])
    edge_lats = numpy.concatenate(
        [numpy.full(n, lat_min), numpy.full(n, lat_max), lats, lats])
    xs, ys = tf.transform(edge_lons, edge_lats)

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    tol = 1.0  # metres
    assert x0 <= xs.min() + tol and x1 >= xs.max() - tol, \
        "window must fit inside the frame on the x axis"
    assert y0 <= ys.min() + tol and y1 >= ys.max() - tol, \
        "window must fit inside the frame on the y axis"
    matplotlib.pyplot.close(fig)


def test_add_cities_draws_markers_and_labels(monkeypatch):
    """add_cities plots one marker per city and labels it with its name."""
    import shapely.geometry
    monkeypatch.setattr(naaulu.geography, "get_country_tiles",
                        lambda geom, cw, ch=None: {"BEL"})
    monkeypatch.setattr(naaulu.geography, "get_cities",
                        lambda geom, iso, limit=25: [
                            {"name": "Bruxelles", "lon": 4.35, "lat": 50.85,
                             "place": "city", "population": 1000000}])

    crs = naaulu.visual.get_plot_crs((2.0, 49.0, 6.0, 52.0))
    fig, ax = matplotlib.pyplot.subplots()
    naaulu.visual.add_cities(ax, crs, shapely.geometry.box(2.0, 49.0, 6.0, 52.0))

    assert len(ax.lines) == 1, "one marker line"
    assert ax.lines[0].get_marker() == "s", "cities are drawn as squares"
    assert [t.get_text() for t in ax.texts] == ["Bruxelles"]
    matplotlib.pyplot.close(fig)


def test_add_borders_adm1_draws_the_bundled_ne10m(monkeypatch):
    """--adm 1 draws country outlines from the bundled NE10m file: the
    OSM layer (and any fetch) must not be involved at all."""
    import shapely.geometry

    def boom(*args, **kwargs):
        raise AssertionError("adm 1 must read the bundle, not OSM")
    monkeypatch.setattr(naaulu.geography, "get_admin0", boom)
    monkeypatch.setattr(naaulu.geography, "get_country_tiles", boom)
    monkeypatch.setattr(naaulu.geography, "_prefetch_country", boom)

    crs = naaulu.visual.get_plot_crs((2.0, 49.0, 6.0, 52.0))
    fig, ax = matplotlib.pyplot.subplots()
    naaulu.visual.add_borders(
        ax, crs, shapely.geometry.box(2.0, 49.0, 6.0, 52.0), adm=1)

    assert len(ax.lines) >= 1, "country outlines drawn from the bundle"
    matplotlib.pyplot.close(fig)


def test_add_borders_adm2_draws_osm_end_to_end(monkeypatch):
    """--adm 2 (the default) keeps OSM fully: country outline and province
    tier both come from the boundary cache, drawn as one linework."""
    import shapely.geometry

    calls = {"adm0": 0, "adm2": 0}
    monkeypatch.setattr(naaulu.geography, "get_country_tiles",
                        lambda geom, cw, ch=None: {"BEL"})

    def fake_admin0(geom, iso_codes):
        calls["adm0"] += 1
        return shapely.geometry.MultiPolygon(
            [shapely.geometry.box(2, 49, 6, 52)])

    def fake_admin2(geom, iso_codes):
        calls["adm2"] += 1
        return shapely.geometry.MultiPolygon(
            [shapely.geometry.box(3, 50, 4, 51)])
    monkeypatch.setattr(naaulu.geography, "get_admin0", fake_admin0)
    monkeypatch.setattr(naaulu.geography, "get_admin2", fake_admin2)

    crs = naaulu.visual.get_plot_crs((2.0, 49.0, 6.0, 52.0))
    fig, ax = matplotlib.pyplot.subplots()
    naaulu.visual.add_borders(
        ax, crs, shapely.geometry.box(2.0, 49.0, 6.0, 52.0), adm=2)

    assert calls == {"adm0": 1, "adm2": 1}, "both tiers from OSM"
    assert len(ax.lines) == 2, "country outline + one province unit"
    matplotlib.pyplot.close(fig)
