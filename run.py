"""Urban heat islands by census tract: Landsat LST/NDVI -> hot spots -> spatial regression.

Example:
    python run.py --county 201 --years 2024 2025 --resolution 90
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import xarray as xr

from heat import data, maps, stats, zonal

ROOT = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--state", default="48", help="state FIPS (default 48 = Texas)")
    p.add_argument("--county", default="201", help="county FIPS (default 201 = Harris)")
    p.add_argument("--years", nargs="+", type=int, default=[2024, 2025], help="summer(s) to composite")
    p.add_argument("--resolution", type=int, default=90, help="composite grid size in metres")
    p.add_argument("--crs", default="EPSG:32615", help="projected CRS for the composite (UTM 15N)")
    p.add_argument("--max-cloud", type=float, default=40, help="scene cloud-cover filter (%%)")
    p.add_argument("--min-pixels", type=int, default=20, help="min valid land pixels per tract")
    p.add_argument("--max-water", type=float, default=0.5, help="max share of water pixels per tract")
    p.add_argument("--acs-year", type=int, default=2023, help="ACS 5-year vintage for income")
    p.add_argument("--no-income", action="store_true", help="skip the ACS income covariate")
    p.add_argument("--no-basemap", action="store_true", help="skip contextily basemap tiles")
    return p.parse_args()


def cached_composite(args, bbox):
    cache = ROOT / "data" / "cache" / (
        f"composite_{args.state}{args.county}_{'-'.join(map(str, args.years))}_{args.resolution}m.npz")
    if cache.exists():
        print(f"  using cached composite {cache.name}")
        z = np.load(cache, allow_pickle=False)
        attrs = json.loads(str(z["attrs"]))
        ds = xr.Dataset(
            {k: (("y", "x"), z[k]) for k in ("lst", "ndvi", "n_obs", "water")},
            coords={"y": z["y"], "x": z["x"]},
        )
        ds.attrs.update(attrs)
        return ds
    items = data.search_landsat(bbox, args.years, max_cloud=args.max_cloud)
    print(f"  {len(items)} Landsat 8/9 C2 L2 scenes (cloud < {args.max_cloud}%)")
    ds = data.composite_lst_ndvi(items, bbox, crs=args.crs, resolution=args.resolution)
    ds.attrs["n_scenes"] = len(items)
    ds.attrs["scene_ids"] = sorted(i.id for i in items)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, lst=ds["lst"].values, ndvi=ds["ndvi"].values, n_obs=ds["n_obs"].values,
                        water=ds["water"].values, x=ds["x"].values, y=ds["y"].values,
                        attrs=json.dumps(ds.attrs))
    return ds


def main():
    args = parse_args()
    t0 = time.time()
    raw = ROOT / "data" / "raw"
    figs, res, docs = ROOT / "figures", ROOT / "results", ROOT / "docs"
    for d in (figs, res, docs):
        d.mkdir(exist_ok=True)

    print("1/6 tracts")
    tracts = data.load_tracts(raw, args.state, args.county)
    bbox = list(tracts.to_crs(4326).total_bounds + np.array([-0.01, -0.01, 0.01, 0.01]))

    print("2/6 Landsat composite")
    comp = cached_composite(args, bbox)
    print(f"  {comp.attrs['n_dates']} acquisition dates, grid {comp['lst'].shape}")

    print("3/6 zonal statistics")
    t = zonal.tract_stats(tracts, comp)
    too_wet = (t["water_share_px"] > args.max_water) | (t["ALAND"] == 0)
    too_few = (~too_wet) & (t["n_valid"] < args.min_pixels)
    keep = ~(too_wet | too_few) & t["lst_c"].notna() & t["ndvi"].notna()
    dropped = t[~keep].copy()
    g = t[keep].reset_index(drop=True)
    print(f"  kept {len(g)} of {len(t)} tracts (water: {int(too_wet.sum())}, too few pixels: {int(too_few.sum())})")

    print("4/6 spatial autocorrelation")
    w, winfo = stats.queen_weights(g)
    y = g["lst_c"].values
    gm = stats.global_moran(y, w)
    g["gi_z"], g["gi_p_fdr"], g["gi_class"] = stats.gi_star(y, w, fdr=True)
    _, gi_p_raw, gi_raw = stats.gi_star(y, w, fdr=False)
    g["lisa_I"], g["lisa_p"], g["lisa"] = stats.lisa(y, w)
    gi_counts = g["gi_class"].value_counts().to_dict()
    hot = int(g["gi_class"].str.startswith("Hot").sum())
    cold = int(g["gi_class"].str.startswith("Cold").sum())
    print(f"  Moran's I = {gm['I']:.3f} (p_sim = {gm['p_sim_one_sided']:.3f}); Gi* hot {hot}, cold {cold}")

    print("5/6 regression")
    income_ok = False
    if not args.no_income:
        inc = data.load_acs_income(args.state, args.county, args.acs_year)
        if inc is not None:
            g = g.merge(inc, on="GEOID", how="left")
            income_ok = g["income"].notna().sum() > 0.8 * len(g)
    if income_ok:
        r = g[g["income"].notna()].reset_index(drop=True)
        wr, _ = stats.queen_weights(r)
        r["income_10k"] = r["income"] / 1e4
        names = ["ndvi", "income_10k"]
    else:
        r, wr, names = g, w, ["ndvi"]
    reg = stats.regressions(r["lst_c"].values, r[names].values, names, wr)
    best = min(("ols", "spatial_lag", "spatial_error"), key=lambda k: reg[k]["aic"])
    if best == "spatial_lag":
        ndvi_eff = reg["spatial_lag"]["total_effect"]["ndvi"]
    else:
        ndvi_eff = reg[best]["coef"]["ndvi"]
    # bivariate fit for the scatter plot
    slope, intercept = np.polyfit(g["ndvi"], g["lst_c"], 1)
    r2_bi = float(np.corrcoef(g["ndvi"], g["lst_c"])[0, 1] ** 2)

    metrics = {
        "study_area": {"state_fips": args.state, "county_fips": args.county, "name": "Harris County, TX"},
        "imagery": {
            "collection": "landsat-c2-l2 (Microsoft Planetary Computer)",
            "platforms": ["landsat-8", "landsat-9"],
            "years": args.years, "months": "Jun-Aug", "max_scene_cloud_pct": args.max_cloud,
            "n_scenes": comp.attrs.get("n_scenes"), "n_dates": comp.attrs["n_dates"],
            "resolution_m": args.resolution, "crs": args.crs,
            "pixel_clear_obs_median": float(np.median(comp["n_obs"].values[comp["n_obs"].values > 0])),
        },
        "tracts": {"total": int(len(t)), "kept": int(len(g)), "dropped_water": int(too_wet.sum()),
                   "dropped_too_few_pixels": int(too_few.sum()),
                   "rule": f"drop if water pixels > {args.max_water:.0%} or ALAND = 0, or < {args.min_pixels} valid pixels"},
        "n_tracts_analyzed": int(len(g)),
        "lst_c": {"mean": float(y.mean()), "min": float(y.min()), "max": float(y.max()),
                  "p10": float(np.percentile(y, 10)), "p90": float(np.percentile(y, 90))},
        "ndvi": {"mean": float(g["ndvi"].mean()), "min": float(g["ndvi"].min()), "max": float(g["ndvi"].max())},
        "weights": winfo,
        "global_moran": gm,
        "gi_star": {"fdr": "Benjamini-Hochberg", "hot_spots": hot, "cold_spots": cold, "counts": gi_counts,
                    "without_fdr": {"hot_spots": int(sum(str(c).startswith("Hot") for c in gi_raw)),
                                    "cold_spots": int(sum(str(c).startswith("Cold") for c in gi_raw))}},
        "lisa": {"alpha": 0.05, "permutations": 999, "counts": g["lisa"].value_counts().to_dict()},
        "regression": {
            "covariates": names, "income_source": f"ACS {args.acs_year} 5-year B19013_001E" if income_ok else None,
            **reg, "best_by_aic": best,
            "headline_ndvi_c_per_0.1": float(ndvi_eff * 0.1),
            "bivariate_ols": {"slope_c_per_0.1_ndvi": float(slope * 0.1), "r2": r2_bi},
        },
    }

    print("6/6 outputs")
    out = g.drop(columns=[c for c in ("water_share_tiger",) if c in g]).copy()
    out["gi_class"] = out["gi_class"].astype(str)
    out["lisa"] = out["lisa"].astype(str)
    gpkg = res / "tracts.gpkg"
    if gpkg.exists():
        gpkg.unlink()
    out.to_crs(4326).to_file(gpkg, layer="tracts", driver="GPKG")
    bm = not args.no_basemap
    maps.map_lst(g, dropped, figs / "lst_map.png", basemap=bm)
    maps.map_gi(g, dropped, figs / "gi_star_map.png", basemap=bm)
    maps.map_lisa(g, dropped, figs / "lisa_map.png", basemap=bm)
    maps.scatter_lst_ndvi(g, slope, intercept, r2_bi, figs / "lst_vs_ndvi.png")
    maps.hero(g, dropped, metrics, figs / "hero.png", basemap=bm)
    try:
        maps.folium_map(g, docs / "map.html")
    except Exception as exc:
        print(f"  folium map skipped ({exc})")
    metrics["runtime_s"] = round(time.time() - t0, 1)
    (res / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(f"done in {metrics['runtime_s']} s")


if __name__ == "__main__":
    main()
