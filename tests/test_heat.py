import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heat import data, stats, zonal  # noqa: E402


def test_lst_scaling():
    # DN 44177 -> 44177 * 0.00341802 + 149 = 299.997 K -> ~26.85 C
    assert data.st_to_celsius(44177) == pytest.approx(26.85, abs=0.01)
    assert np.isnan(data.st_to_celsius(0))


def test_sr_scaling():
    assert data.sr_to_reflectance(7273) == pytest.approx(0.0, abs=1e-3)
    assert data.sr_to_reflectance(43636) == pytest.approx(1.0, abs=1e-3)


def test_qa_cloud_mask_bits():
    clear_land = 1 << 6  # 64
    cloud = (1 << 3) | (1 << 1)  # cloud + dilated cloud
    shadow = 1 << 4
    water = (1 << 7) | (1 << 6)  # clear water is not a cloud
    qa = np.array([clear_land, cloud, shadow, water, 1], dtype="uint16")  # 1 = fill
    assert data.cloud_mask(qa).tolist() == [False, True, True, False, True]
    assert data.water_flag(qa).tolist() == [False, False, False, True, False]


def test_zonal_means():
    zones = np.array([[0, 0, 1], [1, -1, 1]])
    vals = np.array([[1.0, 3.0, 10.0], [20.0, 99.0, np.nan]])
    valid = np.isfinite(vals)
    out = zonal.zonal_means(zones, {"v": vals}, valid, 2)
    assert out["n_pixels"].tolist() == [2, 3]
    assert out["n_valid"].tolist() == [2, 2]
    assert out["v"].tolist() == [2.0, 15.0]


def test_bh_adjust_matches_definition():
    p = np.array([0.01, 0.04, 0.03, 0.20])
    q = stats.bh_adjust(p)
    # explicit: sorted p = .01,.03,.04,.20 -> p*n/rank = .04,.06,.0533,.20 -> cummin from top
    assert q.tolist() == pytest.approx([0.04, 0.05333333, 0.05333333, 0.20])


def test_morans_i_on_synthetic_lattice():
    from libpysal.weights import lat2W

    w = lat2W(10, 10, rook=False)  # queen lattice
    w.transform = "R"
    rows = np.repeat(np.arange(10), 10).astype(float)  # smooth north-south gradient
    trend = stats.global_moran(rows, w, permutations=99)
    assert trend["I"] > 0.7 and trend["p_sim_one_sided"] <= 0.05
    checker = ((np.arange(100) // 10 + np.arange(100) % 10) % 2).astype(float)  # rook checkerboard
    rook = lat2W(10, 10, rook=True)
    rook.transform = "R"
    assert stats.global_moran(checker, rook, permutations=99)["I"] < -0.9


def test_gi_classes():
    z = np.array([3.0, -2.2, 1.7, 0.1])
    p = np.array([0.003, 0.03, 0.09, 0.9])
    assert stats.classify_gi(z, p).tolist() == [
        "Hot Spot - 99% Confidence", "Cold Spot - 95% Confidence", "Hot Spot - 90% Confidence", "Not Significant"]
