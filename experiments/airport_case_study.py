"""Airport-scale hand-off case study on real ATL Concourse F OSM geometry.

Real inputs: OSM terminal / apron / taxiway / stand geometry, OpenSky on-ground
and low-altitude aircraft positions (weak prior). Design input (stated as such):
a hypothetical camera deployment along the terminal facade -- public camera
locations do not exist. Detection delays of compromised cameras are resampled
from the WILDTRACK benchmark (our attestor, test episodes).
"""
from __future__ import annotations

import json

import numpy as np

from data_pipeline.provenance import ARTIFACTS
from experiments.adsb_prior import aircraft_xy, point_in_poly, zone_polygons
from theory.handoff_milp import greedy_handoff, solve_handoff
from theory.quorum_bound import max_tolerable_k

CLASS_W = {"terminal": 1.5, "apron": 1.0, "parking_position": 1.0, "taxiway": 0.6}
CELL = 40.0
RANGE, FOV_DEG, N_CAM = 320.0, 110.0, 16
BETA = 0.5


def _seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-9), 0, 1)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


def build(code: str = "ATL", terminal: str = "Concourse F") -> dict:
    zs = zone_polygons(code)
    term = next(z for z in zs if z["kind"] == "terminal" and z["name"] == terminal)
    T = term["xy"]
    lo, hi = T.min(0) - 380, T.max(0) + 380
    xs = np.arange(lo[0] + CELL / 2, hi[0], CELL)
    ys = np.arange(lo[1] + CELL / 2, hi[1], CELL)
    cells = np.array([(x, y) for y in ys for x in xs])
    cls = np.full(len(cells), "", dtype=object)
    for z in zs:
        if z["kind"] in ("apron", "terminal") and z["closed"]:
            inside = point_in_poly(cells, z["xy"])
            if z["kind"] == "terminal":
                # terminal interior is covered by indoor cameras; outdoor cams see a facade band
                continue
            cls[inside & (cls == "")] = z["kind"]
    for z in zs:
        if z["kind"] in ("taxiway", "parking_position"):
            d = np.min([_seg_dist(cells, z["xy"][i], z["xy"][i + 1]) for i in range(len(z["xy"]) - 1)], axis=0)
            cls[(d < 25) & (cls == "")] = z["kind"]
    inside_t = point_in_poly(cells, T)
    d_t = np.min([_seg_dist(cells, T[i], T[i + 1]) for i in range(len(T) - 1)], axis=0)
    cls[(~inside_t) & (d_t < 45)] = "terminal"   # facade / gate frontage band
    keep = cls != ""
    cells, cls = cells[keep], cls[keep]
    # camera deployment: evenly spaced along the facade, facing outward
    per = np.r_[0, np.cumsum(np.linalg.norm(np.diff(T, axis=0), axis=1))]
    pos, heading = [], []
    ctr = T[:-1].mean(0)
    for s in np.linspace(0, per[-1], N_CAM, endpoint=False):
        i = int(np.searchsorted(per, s, side="right") - 1)
        a, b = T[i], T[i + 1]
        p = a + (b - a) * (s - per[i]) / max(per[i + 1] - per[i], 1e-9)
        n = np.array([-(b - a)[1], (b - a)[0]])
        n /= np.linalg.norm(n) + 1e-9
        if (p + n * 5 - ctr) @ n < 0:
            n = -n
        pos.append(p + n * 8)
        heading.append(np.arctan2(n[1], n[0]))
    pos = np.array(pos)
    cover = np.zeros((N_CAM, len(cells)), bool)
    for c in range(N_CAM):
        v = cells - pos[c]
        dist = np.linalg.norm(v, axis=1)
        ang = np.abs((np.arctan2(v[:, 1], v[:, 0]) - heading[c] + np.pi) % (2 * np.pi) - np.pi)
        cover[c] = (dist < RANGE) & (ang < np.radians(FOV_DEG / 2))
    base_w = np.array([CLASS_W[k] for k in cls]) * cover.any(0)
    return {"cells": cells, "cls": cls, "terminal": T, "cams": pos, "heading": np.array(heading),
            "cover": cover, "base_w": base_w, "zones": zs, "bbox": (lo, hi)}


def adsb_activity(cells: np.ndarray, code: str = "ATL", states_code: str | None = None) -> np.ndarray:
    """Fraction of snapshots with a real aircraft within 60 m of each cell."""
    hits = np.zeros(len(cells))
    snaps = aircraft_xy(code, states_code)
    for pts in snaps:
        if len(pts):
            d = np.linalg.norm(cells[:, None] - pts[None], axis=2).min(1)
            hits += d < 60
    return hits / max(len(snaps), 1)


def weights(cs: dict, act: np.ndarray) -> np.ndarray:
    a = act / max(act.max(), 1e-9)
    w = cs["base_w"] * (1 + BETA * np.clip(a, 0, 1))      # bounded: x in [1, 1+beta]
    return w / w.sum()


def simulate(cs: dict, w: np.ndarray, byz: list[int], delays_s: np.ndarray, rng, policy: str,
             budget: int, horizon_s: int = 600, onset_s: int = 60, dt: int = 2) -> dict:
    C = cs["cover"].shape[0]
    blind, series, sels = 0.0, [], []
    q = np.zeros(C, bool)
    cur = solve_handoff(cs["cover"], w, np.zeros(C), q, budget)
    det_t = {b: onset_s + float(d) for b, d in zip(byz, delays_s)}
    for t in range(0, horizon_s, dt):
        attacked = np.zeros(C, bool)
        if t >= onset_s:
            attacked[byz] = True
        q = np.array([c in det_t and t >= det_t[c] for c in range(C)])
        if policy == "milp":
            cur = solve_handoff(cs["cover"], w, q.astype(float), q, budget)
        elif policy == "greedy":
            cur = greedy_handoff(cs["cover"], w, q, budget, current=cur)
        trusted = cur & ~q & ~attacked
        cov = cs["cover"][trusted].any(0)
        frac = float(w[~cov].sum())
        blind += frac * dt / 60
        series.append(frac)
        sels.append(cur.copy())
    return {"blind_min": blind, "series": series, "selected": np.array(sels)}


def run(seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    cs = build()
    act = adsb_activity(cs["cells"])
    act_spoof = adsb_activity(cs["cells"], states_code="JFK")
    w = weights(cs, act)
    w_spoof = weights(cs, act_spoof)
    C = cs["cover"].shape[0]
    budget = int(np.ceil(0.4 * C))
    A = np.array([[(cs["cover"][i] & cs["cover"][j]).sum() if i != j else 0 for j in range(C)] for i in range(C)])
    delta = int((A > 0).sum(1).max())
    sel0 = solve_handoff(cs["cover"], w, np.zeros(C), np.zeros(C, bool), budget)
    # adversary compromises the k selected cameras carrying the most weight (airside protocol attack)
    k = 4
    gain = np.array([(w * cs["cover"][c]).sum() if sel0[c] else -1 for c in range(C)])
    byz = sorted(np.argsort(-gain)[:k].tolist())
    lat_file = ARTIFACTS / "results_summary.json"
    pool = None
    if lat_file.exists():
        pool = json.loads(lat_file.read_text(encoding="utf-8")).get("detection_latency_pool_s")
    pool = np.array(pool if pool else [5.0])
    res = {}
    for pol in ("none", "greedy", "milp"):
        vals = []
        for r in range(50):
            d = rng.choice(pool, size=k)
            sim = simulate(cs, w, byz, d, rng, pol if pol != "none" else "static", budget)
            vals.append(sim["blind_min"])
            if r == 0:
                res[f"{pol}_example"] = {"series": sim["series"], "selected": sim["selected"].astype(int).tolist()}
        res[pol] = {"blind_min_mean": float(np.mean(vals)), "blind_min_std": float(np.std(vals))}
    out = {"n_cameras": C, "budget": budget, "n_cells": int(len(cs["cells"])), "max_degree": delta,
           "max_tolerable_k_mask_ground": max_tolerable_k(delta), "byzantine": byz, "k": k,
           "initial_selection": sel0.astype(int).tolist(),
           "weight_shift_spoof_l1": float(np.abs(w - w_spoof).sum()),
           "adsb_snapshots": len(aircraft_xy("ATL")),
           "adsb_cells_active": int((act > 0).sum()),
           "class_counts": {k2: int((cs["cls"] == k2).sum()) for k2 in CLASS_W},
           "policies": {p: res[p] for p in ("none", "greedy", "milp")},
           "examples": {p: res[f"{p}_example"] for p in ("none", "greedy", "milp")}}
    (ARTIFACTS / "airport_case_study.json").write_text(json.dumps(out), encoding="utf-8")
    np.savez_compressed(ARTIFACTS / "airport_case_study_geometry.npz", cells=cs["cells"], cls=cs["cls"].astype(str),
                        terminal=cs["terminal"], cams=cs["cams"], heading=cs["heading"], cover=cs["cover"],
                        w=w, act=act)
    return out


if __name__ == "__main__":
    o = run()
    print({k: v for k, v in o.items() if k not in ("examples",)})
