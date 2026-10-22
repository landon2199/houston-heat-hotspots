"""Data access: Census tracts, ACS income, and Landsat Collection 2 Level-2 composites.

All sources are public. Raw downloads are cached under ``data/raw/`` (git-ignored).
"""
from __future__ import annotations

import calendar
import os
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import xarray as xr

TIGER_URL = "https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/tl_{year}_{state}_tract.zip"
ACS_URL = "https://api.census.gov/data/{year}/acs/acs5"
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Landsat Collection 2 Level-2 scale factors (USGS LSDS-1619 product guide)
ST_SCALE, ST_OFFSET = 0.00341802, 149.0  # surface temperature, Kelvin
SR_SCALE, SR_OFFSET = 0.0000275, -0.2  # surface reflectance

# QA_PIXEL bit positions (Landsat 8/9 Collection 2)
QA_FILL, QA_DILATED, QA_CIRRUS, QA_CLOUD, QA_SHADOW, QA_SNOW, QA_CLEAR, QA_WATER = range(8)
CLOUD_BITS = (QA_FILL, QA_DILATED, QA_CIRRUS, QA_CLOUD, QA_SHADOW, QA_SNOW)


# --------------------------------------------------------------------------- scaling / masks
def st_to_celsius(dn):
    """Landsat C2 L2 ST_B10 digital number -> land surface temperature in deg C (0 = nodata)."""
    dn = np.asarray(dn, dtype="float64")
    out = dn * ST_SCALE + ST_OFFSET - 273.15
    return np.where(dn == 0, np.nan, out)


def sr_to_reflectance(dn):
    """Landsat C2 L2 surface-reflectance DN -> reflectance (0 = nodata)."""
    dn = np.asarray(dn, dtype="float64")
    out = dn * SR_SCALE + SR_OFFSET
    return np.where(dn == 0, np.nan, out)


def qa_bits_set(qa, bits):
    """Boolean array: True where ANY of the given QA_PIXEL bits is set."""
    qa = np.asarray(qa).astype("uint16")
    mask = np.uint16(sum(1 << b for b in bits))
    return (qa & mask) != 0


def cloud_mask(qa):
    """True where a pixel is fill, dilated cloud, cirrus, cloud, cloud shadow or snow."""
    return qa_bits_set(qa, CLOUD_BITS)


def water_flag(qa):
    return qa_bits_set(qa, (QA_WATER,))


# --------------------------------------------------------------------------- vector data
def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


def load_tracts(raw_dir: Path, state: str = "48", county: str = "201", year: int = 2020) -> gpd.GeoDataFrame:
    """2020 TIGER/Line census tracts for one county."""
    url = TIGER_URL.format(year=year, state=state)
    zpath = _download(url, Path(raw_dir) / Path(url).name)
    gdf = gpd.read_file(f"zip://{zpath.as_posix()}")
    gdf = gdf[gdf["COUNTYFP"] == county].copy()
    gdf = gdf[["GEOID", "NAMELSAD", "ALAND", "AWATER", "geometry"]].reset_index(drop=True)
    gdf["water_share_tiger"] = gdf["AWATER"] / (gdf["ALAND"] + gdf["AWATER"])
    return gdf


def load_acs_income(state: str = "48", county: str = "201", year: int = 2023) -> pd.DataFrame | None:
    """ACS 5-year median household income (B19013_001E) per tract, no API key.

    Uses the ``CENSUS_API_KEY`` environment variable if set. Returns None if the
    request fails, so the pipeline continues without it.
    """
    params = {"get": "NAME,B19013_001E", "for": "tract:*", "in": f"state:{state} county:{county}"}
    if os.environ.get("CENSUS_API_KEY"):
        params["key"] = os.environ["CENSUS_API_KEY"]
    try:
        r = requests.get(ACS_URL.format(year=year), params=params, timeout=60, allow_redirects=False)
        if r.status_code in (301, 302) or "KeyError" in str(r.headers):
            print("  Census API asked for a key (set CENSUS_API_KEY); continuing without income")
            return None
        r.raise_for_status()
        rows = r.json()
    except Exception as exc:  # network, JSON or key-required response
        print(f"  ACS request failed ({exc}); continuing without income")
        return None
    df = pd.DataFrame(rows[1:], columns=rows[0])
    df["GEOID"] = df["state"] + df["county"] + df["tract"]
    inc = pd.to_numeric(df["B19013_001E"], errors="coerce")
    df["income"] = inc.where(inc > 0)  # negative values are Census sentinel codes
    return df[["GEOID", "income"]]


# --------------------------------------------------------------------------- Landsat
def search_landsat(bbox, years, months=(6, 8), max_cloud=40):
    """Landsat 8/9 C2 L2 items over bbox (lon/lat) for the given summer months."""
    import planetary_computer
    import pystac_client

    catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    items = []
    for y in years:
        last = calendar.monthrange(y, months[1])[1]
        dt = f"{y}-{months[0]:02d}-01/{y}-{months[1]:02d}-{last}"
        search = catalog.search(
            collections=["landsat-c2-l2"],
            bbox=bbox,
            datetime=dt,
            query={"eo:cloud_cover": {"lt": max_cloud}, "platform": {"in": ["landsat-8", "landsat-9"]}},
        )
        items.extend(search.item_collection())
    return items


def composite_lst_ndvi(items, bbox, crs="EPSG:32615", resolution=90, min_obs=3, water_frac=0.5):
    """Per-pixel median summer LST (deg C) and NDVI from cloud-masked scenes.

    Returns an xarray Dataset with ``lst``, ``ndvi``, ``n_obs`` (clear observations)
    and ``water`` (True where the pixel is flagged water in > ``water_frac`` of clear looks).
    """
    import odc.stac

    ds = odc.stac.load(
        items,
        bands=["lwir11", "red", "nir08", "qa_pixel"],
        crs=crs,
        resolution=resolution,
        bbox=bbox,
        groupby="solar_day",
        resampling="nearest",
        chunks={},
    ).compute()

    qa = ds["qa_pixel"].values
    bad = cloud_mask(qa) | (ds["lwir11"].values == 0)
    lst = st_to_celsius(ds["lwir11"].values)
    red = sr_to_reflectance(ds["red"].values)
    nir = sr_to_reflectance(ds["nir08"].values)
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = (nir - red) / (nir + red)
    lst[bad | (lst < -10) | (lst > 80)] = np.nan
    ndvi[bad | ~np.isfinite(ndvi)] = np.nan

    clear = np.isfinite(lst)
    n_obs = clear.sum(axis=0)
    wet = (water_flag(qa) & clear).sum(axis=0) / np.maximum(n_obs, 1)
    with warnings.catch_warnings():  # all-NaN pixels are expected
        warnings.simplefilter("ignore", RuntimeWarning)
        lst_med = np.nanmedian(lst, axis=0)
        ndvi_med = np.nanmedian(ndvi, axis=0)
    few = n_obs < min_obs
    lst_med[few] = np.nan
    ndvi_med[few] = np.nan

    coords = {"y": ds["y"], "x": ds["x"]}
    out = xr.Dataset(
        {
            "lst": (("y", "x"), lst_med.astype("float32")),
            "ndvi": (("y", "x"), ndvi_med.astype("float32")),
            "n_obs": (("y", "x"), n_obs.astype("int16")),
            "water": (("y", "x"), wet > water_frac),
        },
        coords=coords,
    )
    out.attrs.update(
        crs=crs,
        resolution=resolution,
        n_dates=int(ds.sizes["time"]),
        dates=[str(t)[:10] for t in ds["time"].values],
    )
    return out
