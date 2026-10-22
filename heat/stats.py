"""Spatial statistics: weights, global/local autocorrelation, and regression."""
from __future__ import annotations

import numpy as np
from scipy import stats as sps

# Esri "Hot Spot Analysis" class labels, ordered cold -> hot
GI_CLASSES = [
    "Cold Spot - 99% Confidence",
    "Cold Spot - 95% Confidence",
    "Cold Spot - 90% Confidence",
    "Not Significant",
    "Hot Spot - 90% Confidence",
    "Hot Spot - 95% Confidence",
    "Hot Spot - 99% Confidence",
]
LISA_LABELS = {0: "Not significant", 1: "High-High", 2: "Low-High", 3: "Low-Low", 4: "High-Low"}


# --------------------------------------------------------------------------- weights
def queen_weights(gdf, attach_islands=True):
    """Queen contiguity weights (row-standardised).

    Tracts with no queen neighbour (islands) are joined to their single nearest
    neighbour (centroid k=1) so every observation enters the statistics.
    Returns (w, info dict).
    """
    from libpysal.weights import KNN, Queen
    from libpysal.weights.util import attach_islands as _attach

    w = Queen.from_dataframe(gdf, use_index=False, silence_warnings=True)
    islands = list(w.islands)
    if islands and attach_islands:
        cents = gdf.geometry.centroid
        knn = KNN.from_array(np.column_stack([cents.x, cents.y]), k=1)
        w = _attach(w, knn)
    w.transform = "R"
    info = {
        "type": "Queen contiguity, row-standardised",
        "n": int(w.n),
        "mean_neighbors": round(float(w.mean_neighbors), 2),
        "islands_attached_knn1": len(islands),
        "components": int(w.n_components),
    }
    return w, info


# --------------------------------------------------------------------------- multiple testing
def bh_adjust(p):
    """Benjamini-Hochberg (1995) adjusted p-values (q-values), same order as input."""
    p = np.asarray(p, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


# --------------------------------------------------------------------------- global / local
def global_moran(y, w, permutations=999, seed=42):
    from esda.moran import Moran

    np.random.seed(seed)
    m = Moran(y, w, permutations=permutations)
    return {
        "I": float(m.I),
        "expected_I": float(m.EI),
        "z_norm": float(m.z_norm),
        "p_norm_two_sided": float(m.p_norm),
        "p_sim_one_sided": float(m.p_sim),
        "permutations": permutations,
    }


def gi_star(y, w, fdr=True, seed=42):
    """Getis-Ord Gi* z-scores with Esri-style confidence classes.

    Uses the analytical z-score (as in ArcGIS Hot Spot Analysis), two-sided
    normal p-values, and optionally Benjamini-Hochberg FDR correction.
    Returns (z, p_used, class_label array).
    """
    from esda.getisord import G_Local

    np.random.seed(seed)
    g = G_Local(y, w, transform="B", star=True, permutations=0)
    z = np.asarray(g.Zs, dtype=float)
    p = 2 * sps.norm.sf(np.abs(z))
    p_used = bh_adjust(p) if fdr else p
    return z, p_used, classify_gi(z, p_used)


def classify_gi(z, p):
    z = np.asarray(z)
    p = np.asarray(p)
    conf = np.select([p < 0.01, p < 0.05, p < 0.10], [99, 95, 90], default=0)
    labels = np.full(z.shape, "Not Significant", dtype=object)
    for c in (90, 95, 99):
        labels[(conf == c) & (z > 0)] = f"Hot Spot - {c}% Confidence"
        labels[(conf == c) & (z < 0)] = f"Cold Spot - {c}% Confidence"
    return labels


def lisa(y, w, permutations=999, seed=42, alpha=0.05):
    """Local Moran's I (Anselin 1995). Returns (Ii, p_sim, cluster label array)."""
    from esda.moran import Moran_Local

    lm = Moran_Local(y, w, permutations=permutations, seed=seed)
    q = np.where(lm.p_sim < alpha, lm.q, 0)
    labels = np.array([LISA_LABELS[int(v)] for v in q], dtype=object)
    return lm.Is, lm.p_sim, labels


# --------------------------------------------------------------------------- regression
def _residual_moran(e, w, seed=42):
    from esda.moran import Moran

    np.random.seed(seed)
    m = Moran(np.asarray(e).ravel(), w, permutations=999)
    return {"I": float(m.I), "p_sim": float(m.p_sim), "z_norm": float(m.z_norm)}


def regressions(y, X, names, w):
    """OLS with spatial diagnostics, then ML spatial lag and spatial error models."""
    from spreg import ML_Error, ML_Lag, OLS

    y = np.asarray(y, dtype=float).reshape(-1, 1)
    X = np.asarray(X, dtype=float)
    ols = OLS(y, X, w=w, spat_diag=True, moran=True, name_y="lst_c", name_x=names)
    out = {
        "n": int(ols.n),
        "ols": {
            "coef": dict(zip(["const"] + names, map(float, ols.betas.ravel()))),
            "se": dict(zip(["const"] + names, map(float, np.sqrt(np.diag(ols.vm))))),
            "r2": float(ols.r2),
            "adj_r2": float(ols.ar2),
            "aic": float(ols.aic),
            "log_lik": float(ols.logll),
            "residual_moran": _residual_moran(ols.u, w),
            "lm_lag": [float(v) for v in ols.lm_lag],
            "lm_error": [float(v) for v in ols.lm_error],
            "robust_lm_lag": [float(v) for v in ols.rlm_lag],
            "robust_lm_error": [float(v) for v in ols.rlm_error],
        },
    }
    lag = ML_Lag(y, X, w=w, method="full", name_y="lst_c", name_x=names)
    b = lag.betas.ravel()
    rho = float(b[-1])
    out["spatial_lag"] = {
        "coef": dict(zip(["const"] + names, map(float, b[:-1]))),
        "se": dict(zip(["const"] + names + ["rho"], map(float, np.sqrt(np.diag(lag.vm))))),
        "rho": rho,
        "total_effect": {n: float(v / (1 - rho)) for n, v in zip(names, b[1:-1])},
        "pseudo_r2": float(lag.pr2),
        "aic": float(lag.aic),
        "log_lik": float(lag.logll),
        "residual_moran": _residual_moran(lag.u, w),
    }
    err = ML_Error(y, X, w=w, method="full", name_y="lst_c", name_x=names)
    b = err.betas.ravel()
    out["spatial_error"] = {
        "coef": dict(zip(["const"] + names, map(float, b[:-1]))),
        "se": dict(zip(["const"] + names + ["lambda"], map(float, np.sqrt(np.diag(err.vm))))),
        "lambda": float(b[-1]),
        "pseudo_r2": float(err.pr2),
        # spreg's ML_Error.aic counts only the betas; add 1 for lambda so the AIC is
        # comparable with ML_Lag (whose k includes rho) and OLS.
        "aic": float(-2 * err.logll + 2 * (err.k + 1)),
        "log_lik": float(err.logll),
        "residual_moran": _residual_moran(err.e_filtered, w),
    }
    return out
