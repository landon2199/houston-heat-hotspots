"""Static figures (matplotlib) and an optional interactive folium map."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from .stats import GI_CLASSES  # noqa: E402

# Esri Hot Spot Analysis renderer colours (cold 99 -> hot 99)
GI_COLORS = ["#4575b5", "#849eba", "#c0ccbe", "#f0f0e6", "#fab984", "#ed7551", "#d62f27"]
LISA_COLORS = {
    "High-High": "#d7191c",
    "High-Low": "#fdae61",
    "Low-High": "#abd9e9",
    "Low-Low": "#2c7bb6",
    "Not significant": "#eeeeee",
}
INK = "#222222"
MUTED = "#666666"
EDGE = "#ffffff"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.edgecolor": "#bbbbbb",
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "figure.facecolor": "white",
    }
)


def _basemap(ax, crs):
    """Light basemap if tiles are reachable; silently skipped otherwise."""
    try:
        import contextily as cx

        cx.add_basemap(ax, crs=crs, source=cx.providers.Esri.WorldGrayCanvas, attribution_size=6, zorder=0,
                       attribution="Basemap: Powered by Esri · Esri, HERE, Garmin, © OpenStreetMap contributors")
        return True
    except Exception as exc:  # offline, tile server error, etc.
        print(f"  basemap skipped ({type(exc).__name__})")
        return False


def _frame(ax, title=None):
    ax.set_axis_off()
    if title:
        ax.set_title(title, loc="left", fontsize=13, color=INK, fontweight="bold")


def _to_web(gdf):
    return gdf.to_crs(3857)


def map_lst(gdf, dropped, path, basemap=True):
    g = _to_web(gdf)
    fig, ax = plt.subplots(figsize=(11, 8))
    if len(dropped):
        _to_web(dropped).plot(ax=ax, color="#d9d9d9", edgecolor=EDGE, linewidth=0.2, zorder=1)
    g.plot(
        column="lst_c", ax=ax, cmap="OrRd", scheme="quantiles", k=7, edgecolor=EDGE, linewidth=0.15,
        legend=True, zorder=2,
        legend_kwds={"title": "Mean LST (°C), quantiles", "loc": "upper left", "bbox_to_anchor": (1.0, 1.0), "fontsize": 8,
                     "title_fontsize": 9, "frameon": False, "fmt": "{:.1f}"},
    )
    if basemap:
        _basemap(ax, g.crs)
    _frame(ax, "Summer land surface temperature by census tract")
    ax.text(0, -0.02, "Harris County, TX · Landsat 8/9 C2 L2, Jun–Aug median · grey = excluded tracts",
            transform=ax.transAxes, fontsize=8, color=MUTED)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _gi_plot(ax, g, legend=True, counts=None, short=False, below=False):
    colors = dict(zip(GI_CLASSES, GI_COLORS))
    g = g.assign(_c=g["gi_class"].map(colors))
    g.plot(color=g["_c"], ax=ax, edgecolor=EDGE, linewidth=0.15, zorder=2)
    if legend:
        handles = []
        for lab in reversed(GI_CLASSES):
            n = f" ({counts.get(lab, 0)})" if counts is not None else ""
            text = lab.replace(" Spot - ", " ").replace(" Confidence", "") if short else lab
            handles.append(Patch(facecolor=colors[lab], edgecolor="#999999", linewidth=0.3, label=text + n))
        where = dict(loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=4) if below else             dict(loc="upper left", bbox_to_anchor=(1.0, 1.0))
        ax.legend(handles=handles, fontsize=10 if below else 9, frameon=False, title="Gi* (FDR-corrected)",
                  title_fontsize=11 if below else 10, alignment="left", **where)


def map_gi(gdf, dropped, path, basemap=True):
    g = _to_web(gdf)
    fig, ax = plt.subplots(figsize=(11, 8))
    if len(dropped):
        _to_web(dropped).plot(ax=ax, color="#d9d9d9", edgecolor=EDGE, linewidth=0.2, zorder=1)
    _gi_plot(ax, g, counts=g["gi_class"].value_counts().to_dict())
    if basemap:
        _basemap(ax, g.crs)
    _frame(ax, "Getis-Ord Gi* hot and cold spots of surface temperature")
    ax.text(0, -0.02, "Queen contiguity · Benjamini–Hochberg FDR · counts = tracts", transform=ax.transAxes,
            fontsize=8, color=MUTED)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def map_lisa(gdf, dropped, path, basemap=True):
    g = _to_web(gdf)
    fig, ax = plt.subplots(figsize=(11, 8))
    if len(dropped):
        _to_web(dropped).plot(ax=ax, color="#d9d9d9", edgecolor=EDGE, linewidth=0.2, zorder=1)
    g.assign(_c=g["lisa"].map(LISA_COLORS)).plot(color=g["lisa"].map(LISA_COLORS), ax=ax, edgecolor=EDGE,
                                                linewidth=0.15, zorder=2)
    counts = g["lisa"].value_counts().to_dict()
    handles = [Patch(facecolor=c, edgecolor="#999999", linewidth=0.3, label=f"{k} ({counts.get(k, 0)})")
               for k, c in LISA_COLORS.items()]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=9, frameon=False,
              title="Local Moran's I (p < 0.05)", title_fontsize=10, alignment="left")
    if basemap:
        _basemap(ax, g.crs)
    _frame(ax, "LISA clusters of surface temperature")
    ax.text(0, -0.02, "999 conditional permutations · pseudo p < 0.05, not FDR-corrected", transform=ax.transAxes,
            fontsize=8, color=MUTED)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def scatter_lst_ndvi(gdf, slope, intercept, r2, path):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(gdf["ndvi"], gdf["lst_c"], s=12, color="#c4532d", alpha=0.55, edgecolor="white", linewidth=0.3)
    xs = np.linspace(gdf["ndvi"].min(), gdf["ndvi"].max(), 50)
    ax.plot(xs, intercept + slope * xs, color=INK, linewidth=2)
    ax.set_xlabel("Mean NDVI (tract)")
    ax.set_ylabel("Mean land surface temperature (°C)")
    ax.grid(True, color="#eeeeee", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("Greener tracts are cooler", loc="left", fontsize=13, color=INK, fontweight="bold")
    ax.text(0.98, 0.96, f"OLS: {slope * 0.1:+.2f} °C per +0.1 NDVI\nR² = {r2:.2f}, n = {len(gdf)}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color=INK)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def hero(gdf, dropped, metrics, path, basemap=True):
    """16:9 portfolio card: LST map + Gi* map with headline numbers."""
    g = _to_web(gdf)
    d = _to_web(dropped) if len(dropped) else None
    fig = plt.figure(figsize=(16, 9), dpi=100)
    fig.patch.set_facecolor("white")
    fig.text(0.02, 0.93, "Where is Houston hottest?", fontsize=28, fontweight="bold", color=INK)
    m = metrics
    sub = (f"Summer land surface temperature, {m['n_tracts_analyzed']} Harris County census tracts · "
           f"Moran's I = {m['global_moran']['I']:.2f} · "
           f"{m['gi_star']['hot_spots']} hot-spot and {m['gi_star']['cold_spots']} cold-spot tracts (FDR) · "
           f"{m['regression']['headline_ndvi_c_per_0.1']:+.1f} °C per +0.1 NDVI")
    fig.text(0.02, 0.885, sub, fontsize=13, color=MUTED)
    ax1 = fig.add_axes([0.01, 0.12, 0.475, 0.70])
    ax2 = fig.add_axes([0.515, 0.12, 0.475, 0.70])
    for ax in (ax1, ax2):
        if d is not None:
            d.plot(ax=ax, color="#d9d9d9", edgecolor=EDGE, linewidth=0.2, zorder=1)
    g.plot(column="lst_c", ax=ax1, cmap="OrRd", scheme="quantiles", k=7, edgecolor=EDGE, linewidth=0.1, zorder=2,
           legend=True, legend_kwds={"title": "Mean LST (°C)", "loc": "upper center", "bbox_to_anchor": (0.5, -0.01),
                                     "ncol": 4, "fontsize": 10, "title_fontsize": 11, "frameon": False,
                                     "fmt": "{:.1f}", "alignment": "left"})
    _gi_plot(ax2, g, counts=g["gi_class"].value_counts().to_dict(), short=True, below=True)
    for ax, t in ((ax1, "Land surface temperature"), (ax2, "Gi* hot & cold spots")):
        if basemap:
            _basemap(ax, g.crs)
        _frame(ax, t)
        ax.set_anchor("N")
    fig.text(0.97, 0.012, "Data: USGS Landsat 8/9 C2 L2 via Microsoft Planetary Computer · US Census Bureau TIGER/Line",
             ha="right", fontsize=9, color=MUTED)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def folium_map(gdf, path):
    """Small interactive choropleth (simplified geometry, rounded attributes)."""
    import branca
    import folium

    g = gdf[["GEOID", "lst_c", "ndvi", "gi_class", "lisa", "geometry"]].copy()
    g = g.to_crs(3857)
    g["geometry"] = g.geometry.simplify(60)
    g = g.to_crs(4326)
    g["geometry"] = g.geometry.set_precision(1e-5)
    g["lst_c"] = g["lst_c"].round(1)
    g["ndvi"] = g["ndvi"].round(2)
    c = g.geometry.union_all().centroid
    m = folium.Map(location=[c.y, c.x], zoom_start=9, tiles=None)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Powered by <a href='https://www.esri.com'>Esri</a> | Esri, HERE, Garmin, &copy; OpenStreetMap contributors, "
             "and the GIS user community | Landsat: USGS | Tracts: U.S. Census Bureau",
        name="Esri World Light Gray Canvas", max_zoom=16,
    ).add_to(m)
    bins = list(np.quantile(g["lst_c"], np.linspace(0, 1, 8)))
    cmap = branca.colormap.StepColormap(
        [matplotlib.colors.to_hex(plt.get_cmap("OrRd")(v)) for v in np.linspace(0.08, 1, 7)],
        index=bins, vmin=bins[0], vmax=bins[-1], caption="Mean summer LST (°C), quantiles")
    folium.GeoJson(
        g.to_json(drop_id=True, to_wgs84=False),
        style_function=lambda f: {"fillColor": cmap(f["properties"]["lst_c"]), "fillOpacity": 0.75,
                                  "color": "#ffffff", "weight": 0.3},
        tooltip=folium.GeoJsonTooltip(fields=["GEOID", "lst_c", "ndvi", "gi_class", "lisa"],
                                      aliases=["Tract", "LST (°C)", "NDVI", "Gi*", "LISA"]),
        name="Tracts",
    ).add_to(m)
    cmap.add_to(m)
    m.get_root().header.add_child(folium.Element("<title>Houston Heat Hot Spots</title>"))
    m.get_root().html.add_child(folium.Element(
        '<div style="position:fixed;top:12px;left:56px;z-index:1000;background:rgba(255,255,255,0.92);'
        'padding:8px 12px;border-radius:4px;box-shadow:0 1px 4px rgba(0,0,0,0.25);'
        'font:13px/1.35 sans-serif;color:#222;max-width:340px">'
        '<b style="font-size:15px">Houston Heat Hot Spots</b><br>'
        "Mean summer land surface temperature (°C) by census tract, Harris County, TX. "
        "Landsat 8/9, Jun–Aug 2024–2025. Hover a tract for NDVI, Gi* and LISA class.</div>"))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    m.save(str(path))
