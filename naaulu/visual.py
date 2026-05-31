import io
import logging
import os

import av
import geovista
import geovista.bridge
import geovista.pantry
import matplotlib.cm
import matplotlib.pyplot
import numpy
import pyproj
import shapely.ops
import wradlib

import naaulu.geography

logger = logging.getLogger(__name__)


def show():
    matplotlib.pyplot.show()

def figure():

    fig, ax = matplotlib.pyplot.subplots(figsize=(16, 9))

    return fig, ax

def path(time, area, duration, resolution, organisation, product, format):

    if format == "mp4":
        times = time
        time_str = f"{naaulu.util.format_time(times[0])}"
        time_str = f"{time_str}_{naaulu.util.format_time(times[-1])}"
    else:
        time_str = f"{naaulu.util.format_time(time)}"
        
    try:
        area_part = naaulu.geography.country_code(area).lower()
    except Exception:
        area_part = area.lower()

    filename = ".".join(
        [
            time_str,
            area_part,
            naaulu.util.format_duration(duration),
            naaulu.util.format_distance(resolution),
            organisation,
            product,
            format,
        ]
    )

    archive = naaulu.config.get_archive_dir()
    if archive is not None:    
        root = os.path.join(archive, "figure")
        filename = naaulu.util.get_path(root, filename)
        os.makedirs(os.path.dirname(filename), exist_ok=True)

    return filename


def get_plot_crs(bounds):
    lon_min, lat_min, lon_max, lat_max = bounds
    central_lon = (lon_min + lon_max) / 2
    central_lat = (lat_min + lat_max) / 2
    projstr = f"+proj=laea +lon_0={central_lon} +lat_0={central_lat}"
    plot_crs = pyproj.CRS.from_proj4(projstr)

    return plot_crs


def plot(ax, crs, dataset, cmap, norm):
    precip = dataset["precipitation"]
    xy = wradlib.georef.get_raster_coordinates(precip).values
    xy = wradlib.georef.reproject(xy, src_crs = dataset.spatial_ref.attrs["crs_wkt"], trg_crs= crs)
    ax.pcolormesh(xy[...,0], xy[...,1], precip.values, cmap=cmap, norm=norm, shading="auto")


def add_borders(ax, crs, geom, chunk_width=2, chunk_height=None):
    if chunk_height is None:
        chunk_height = chunk_width
    transformer = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    iso_codes = naaulu.geography.get_country_tiles(geom, chunk_width, chunk_height)
    countries = naaulu.geography.get_countries(geom=geom, iso_codes=iso_codes)
    countries = shapely.ops.transform(transformer.transform, countries)
    for poly in countries.geoms:
        x, y = poly.exterior.xy
        ax.plot(x, y, color="black", linewidth=1)

def add_axis(ax, bounds, crs):
    lon_min, lat_min, lon_max, lat_max = bounds
    transformer = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)        
    x_min, y_min = transformer.transform(lon_min, lat_min)
    x_max, y_max = transformer.transform(lon_max, lat_max)               
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{x/1000:.1f}" for x in xticks])
    ax.set_xlabel("km")
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y/1000:.1f}" for y in yticks])
    ax.set_ylabel("km")
    ax.set_facecolor("lightgrey")
    ax.grid(True, linestyle="--", linewidth=0.5, color="gray", alpha=0.5)
    ax.set_aspect('equal')

def add_title(fig, time, area, duration, resolution, organisation, product, radar_count=None):

    where = f"{area.capitalize()} / {naaulu.util.format_time(time, show=True)}"
    what = f"{naaulu.util.format_duration(duration, show=True)} / {naaulu.util.format_distance(resolution)}"
    how = f"{organisation.upper()} / {product.upper()}"
    if radar_count is not None:
        how += f" / radars:{int(round(radar_count))}"
    title_str = " || ".join(
        [where, what, how]
    )
    fig.suptitle(title_str, fontsize=14)


def get_cmap_levels(vmin, vmax):

    levels = numpy.array(
        [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 1000]
    )
    levels = [level for level in levels if level >= vmin]
    levels = [level for level in levels if level <= vmax]
    levels = numpy.concatenate(([1e-6], levels))

    return levels

def get_cmap(cmin, cmax):

    levels = get_cmap_levels(cmin, cmax)
    cmap = matplotlib.cm.get_cmap("YlGnBu")
    cmap.set_bad("lightgrey")
    cmap.set_under("white")
    norm = matplotlib.cm.colors.BoundaryNorm(levels, ncolors=cmap.N, extend="max")

    return cmap, norm

def plot_gauges(ax, crs, datasets_by_network, cmap=None, norm=None):
    """
    Overlay gauge stations on a map, colored by precipitation value.
    """
    if not datasets_by_network:
        return

    transformer = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    for label, ds in datasets_by_network.items():
        lon = numpy.asarray(ds.longitude.values, dtype=float)
        lat = numpy.asarray(ds.latitude.values, dtype=float)
        x, y = transformer.transform(lon, lat)
        prec = ds["precipitation"].values if "precipitation" in ds else numpy.full(len(x), numpy.nan)
        logger.info(f"gauge {label}: {list(zip(ds.station.values if 'station' in ds else range(len(x)), prec))}")

        if cmap is not None and norm is not None:
            colors = cmap(norm(prec))
        else:
            colors = "black"

        ax.scatter(
            x, y,
            c=colors,
            edgecolors="black",
            linewidths=0.5,
            s=50,
            label=label,
            zorder=5,
        )

    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)


def add_colorbar(fig, ax, cmap, norm):
    mappable = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(mappable, ax=ax, ticks=norm.boundaries, extend="max")
    cbar.ax.set_yticklabels(["0" if lvl == 1e-6 else str(lvl) for lvl in norm.boundaries])
    cbar.set_label("Rainfall Accumulation (mm)", fontsize=12)

def create_movie(filename):

    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    container = av.open(filename, mode="w", format="mp4")
    stream = container.add_stream("libx265", rate=2)
    stream.width = 1920
    stream.height = 1080
    stream.pix_fmt = "yuv420p"

    return container, stream

def add_frame(container, fig):

    stream = container.streams.video[0]
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.05)
    buf.seek(0)
    image = matplotlib.pyplot.imread(buf)
    image = (image * 255).astype("uint8")[:, :, :3]    
    frame = av.VideoFrame.from_ndarray(image, format="rgb24")
    packet = stream.encode(frame)
    if packet:
        container.mux(packet)

def close_movie(container):    
    stream = container.streams.video[0]
    packet = stream.encode(None)
    if packet:
        container.mux(packet)
    container.close()

def render(datasets):
    
    arrays = [ds["precipitation"].squeeze() for ds in datasets]
    merged = xarray.concat(arrays, dim="x")
    lats = merged.y.values
    lons = merged.x.values
    values = merged.values

    mesh = geovista.bridge.Transform.from_1d(lons, lats, data=values)
    plotter = geovista.GeoPlotter()
    plotter.add_mesh(mesh, cmap="YlGnBu", clim=args.clim, show_scalar_bar=True)
    coastlines = geovista.pantry.fetch_coastlines()
    plotter.add_mesh(coastlines, color="black", line_width=1)
    plotter.add_text(
        f"{naaulu.util.format_time(time, show=True)} | {area['name']}",
        position="upper_left",
        font_size=10,
    )
    plotter.show()