"""Empirical test of the Compression-by-Contradiction bound on real WILDTRACK views.

For a zone with redundancy r (cameras whose FOV contains it) and k colluding
Byzantine witnesses inside that coverage set, the adversary injects a phantom
that is *exactly* consistent across all colluding views (it knows the
calibration). Honest witnesses report their real annotated detections. We then
ask whether (a) some honest witness contradicts the phantom (detection) and
(b) contradictions outnumber support (masking).

Ground mode: phantom foot point on the ground plane (homography test, 1 m gate).
Off-plane mode: worst-case epipolar placement -- the phantom 3-D point lies on
the viewing ray of a real head seen by an honest camera h0, so h0's report is
consistent with it (p^T E p' = 0 holds); other witnesses test the head-point
reprojection (25 px gate at 1920x1080).
"""
from __future__ import annotations

import json

import numpy as np

from data_pipeline.provenance import ARTIFACTS
from experiments.scene import load_scene
from theory.quorum_bound import m_detect, m_mask

GATE_CM, GATE_PX = 100.0, 25.0
IMG_W, IMG_H = 1920, 1080


def _visible(sc, g, X, z=0.0):
    uv, d = sc.project(g, X[None], z)
    return bool(d[0] > 0 and 20 <= uv[0, 0] < IMG_W - 20 and 20 <= uv[0, 1] < IMG_H - 20), uv[0]


def ground_trial(sc, rng, f, k):
    gw = sc.gt_world[f]
    for _ in range(50):
        z = int(rng.integers(len(sc.cells)))
        p = sc.cells[z] + rng.uniform(-40, 40, 2)
        if len(gw) and np.min(np.linalg.norm(gw - p, axis=1)) < GATE_CM:
            continue
        cov = [g for g in range(sc.n_cams) if _visible(sc, g, p)[0]]
        if len(cov) >= 1:
            break
    else:
        return None
    r = len(cov)
    B = set(rng.choice(cov, size=min(k, r), replace=False).tolist())
    contra = support = 0
    for g in cov:
        if g in B:
            support += 1
            continue
        d = sc.det[f][g]
        near = len(d) and np.min(np.linalg.norm(d[:, :2] - p, axis=1)) < GATE_CM
        if near:
            support += 1
        else:
            contra += 1
    return r, contra > 0, contra > support, contra, r - len(B)


def offplane_trial(sc, rng, f, k):
    gw, gid = sc.gt_world[f], sc.gt_ids[f]
    if len(gw) == 0:
        return None
    i = int(rng.integers(len(gw)))
    pid = gid[i]
    seen = [g for g in range(sc.n_cams) if pid in sc.det_ids[f][g]]
    if not seen:
        return None
    h0 = int(rng.choice(seen))
    X = np.array([gw[i, 0], gw[i, 1], 170.0])
    Cw = sc.centers[h0]
    lam = rng.uniform(0.45, 0.8)                 # phantom on h0's ray, closer to the camera
    Xp = Cw + lam * (X - Cw)
    cov = []
    for g in range(sc.n_cams):
        ok, _ = _visible(sc, g, Xp[:2], Xp[2])
        if ok:
            cov.append(g)
    if h0 not in cov:
        return None
    r = len(cov)
    others = [g for g in cov if g != h0]
    B = set(rng.choice(others, size=min(k, len(others)), replace=False).tolist()) if others else set()
    contra = support = 0
    for g in cov:
        if g in B or g == h0:
            support += 1
            continue
        uv, _ = sc.project(g, Xp[None, :2], Xp[2])
        d = sc.det[f][g]
        near = len(d) and np.min(np.linalg.norm(d[:, 4:6] - uv[0], axis=1)) < GATE_PX
        if near:
            support += 1
        else:
            contra += 1
    # h0 sees the real head: its report is consistent with BOTH hypotheses
    truth_support = contra + 1
    return r, contra > 0, truth_support > support, contra, r - len(B) - 1


def run(n_trials: int = 4000, seed: int = 7) -> dict:
    sc = load_scene()
    rng = np.random.default_rng(seed)
    out = {"ground": {}, "offplane": {}, "witness": {}}
    for mode, fn in (("ground", ground_trial), ("offplane", offplane_trial)):
        tab = {}
        wit_hits = wit_n = 0
        for k in (1, 2, 3):
            for _ in range(n_trials):
                f = int(rng.integers(200, 400))
                res = fn(sc, rng, f, k)
                if res is None:
                    continue
                r, det, mask, contra, honest = res
                cell = tab.setdefault(f"{r},{k}", [0, 0, 0])
                cell[0] += 1
                cell[1] += int(det)
                cell[2] += int(mask)
                wit_hits += contra
                wit_n += max(honest, 0)
        out[mode] = {key: {"n": v[0], "detect_rate": v[1] / v[0], "mask_rate": v[2] / v[0],
                           "r": int(key.split(",")[0]), "k": int(key.split(",")[1]),
                           "detect_theory": int(key.split(",")[0]) >= m_detect(int(key.split(",")[1]), mode == "offplane"),
                           "mask_theory": int(key.split(",")[0]) >= m_mask(int(key.split(",")[1]), mode == "offplane")}
                     for key, v in sorted(tab.items())}
        out["witness"][mode] = wit_hits / max(wit_n, 1)
    # agreement with theory where enough samples
    for mode in ("ground", "offplane"):
        cells = [c for c in out[mode].values() if c["n"] >= 30]
        agree_det = np.mean([(c["detect_rate"] >= 0.9) == c["detect_theory"] for c in cells])
        agree_mask = np.mean([(c["mask_rate"] >= 0.9) == c["mask_theory"] for c in cells])
        out[f"{mode}_agreement"] = {"detect": float(agree_det), "mask": float(agree_mask), "cells": len(cells)}
    (ARTIFACTS / "phase_boundary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    o = run()
    print(json.dumps({k: v for k, v in o.items() if k.endswith("agreement") or k == "witness"}, indent=1))
