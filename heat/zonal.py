"""Zonal statistics: tract-level mean LST and NDVI from the composite grid."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from rasterio.transform import from_origin


def grid_transform(x, y, res):
    """Affine transform from pixel-centre coordinates (y descending)."""
    x = np.asarray(x)
    y = np.asarray(y)
    return from_origin(x.min() - res / 2, y.max() + res / 2, res, res)


def zone_ids(tracts: gpd.GeoDataFrame, shape, transform) -> np.ndarray:
    """Raster of tract row index (0..n-1) at pixel centres, -1 outside all tracts."""
    shapes = ((geom, i) for i, geom in enumerate(tracts.geometry))
    return rasterize(shapes, out_shape=shape, transform=transform, fill=-1, dtype="int32")


def zonal_means(zones: np.ndarray, values: dict, valid: np.ndarray, n_zones: int) -> dict:
    """Mean of each value array per zone over ``valid`` pixels, plus pixel counts.

    ``zones``: int raster (-1 = no zone); ``values``: name -> float raster;
    ``valid``: boolean raster of usable pixels.
    """
    inside = zones >= 0
    z_all = zones[inside]
    out = {"n_pixels": np.bincount(z_all, minlength=n_zones)}
    ok = inside & valid
    z_ok = zones[ok]
    n_ok = np.bincount(z_ok, minlength=n_zones)
    out["n_valid"] = n_ok
    for name, arr in values.items():
        s = np.bincount(z_ok, weights=arr[ok].astype("float64"), minlength=n_zones)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[name] = np.where(n_ok > 0, s / n_ok, np.nan)
    return out


def tract_stats(tracts: gpd.GeoDataFrame, comp) -> gpd.GeoDataFrame:
    """Attach mean LST/NDVI, pixel counts and water share to each tract."""
    res = float(comp.attrs["resolution"])
    tracts = tracts.to_crs(comp.attrs["crs"])
    transform = grid_transform(comp["x"].values, comp["y"].values, res)
    zones = zone_ids(tracts, comp["lst"].shape, transform)

    lst = comp["lst"].values
    ndvi = comp["ndvi"].values
    water = comp["water"].values
    valid = np.isfinite(lst) & np.isfinite(ndvi) & ~water
    stats = zonal_means(zones, {"lst_c": lst, "ndvi": ndvi}, valid, len(tracts))
    water_stats = zonal_means(zones, {}, water, len(tracts))

    out = tracts.copy()
    out["n_pixels"] = stats["n_pixels"]
    out["n_valid"] = stats["n_valid"]
    out["water_pixels"] = water_stats["n_valid"]
    with np.errstate(invalid="ignore", divide="ignore"):
        out["water_share_px"] = np.where(out["n_pixels"] > 0, out["water_pixels"] / out["n_pixels"], np.nan)
    out["lst_c"] = stats["lst_c"]
    out["ndvi"] = stats["ndvi"]
    return out
