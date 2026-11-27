"""Defence strategies, quarantine logic and availability / integrity metrics."""
from __future__ import annotations

import numpy as np

from experiments.scene import FPS
from experiments.streams import NODE_FEATS
from theory.handoff_milp import greedy_handoff, solve_handoff, static_redundant_set

FI = {n: i for i, n in enumerate(NODE_FEATS)}
ON_FRAMES, OFF_FRAMES = 2, 6


def quarantine(score: np.ndarray, theta: float) -> np.ndarray:
    """Hysteresis: >= theta for ON_FRAMES consecutive frames -> quarantined; release after OFF_FRAMES."""
    C, T = score.shape
    Q = np.zeros((C, T), bool)
    for c in range(C):
        on, run_hi, run_lo = False, 0, 0
        for t in range(T):
            if score[c, t] >= theta:
                run_hi, run_lo = run_hi + 1, 0
            else:
                run_lo, run_hi = run_lo + 1, 0
            if not on and run_hi >= ON_FRAMES:
                on = True
            elif on and run_lo >= OFF_FRAMES:
                on = False
            Q[c, t] = on
    return Q


# ------------------------------------------------------------------------ scores
def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def score_single_camera(Xn: np.ndarray) -> np.ndarray:
    """Per-camera analytics using only the camera's own stream (no cross-view data)."""
    lat = Xn[..., FI["lat_z"]] * 5
    jit = Xn[..., FI["jitter"]] * 50
    gap = Xn[..., FI["gap"]]
    frz = Xn[..., FI["freeze"]]
    res = np.abs(Xn[..., FI["flow_resid_z"]] * 5)
    dif = np.abs(Xn[..., FI["diff_z"]] * 5)
    s = np.maximum.reduce([_sig(lat - 4), _sig((jit - 60) / 15), np.minimum(gap, 1), frz,
                           _sig(res - 5), _sig(dif - 5)])
    return s.T  # (C, T)


def score_majority(Xn: np.ndarray) -> np.ndarray:
    """Naive majority vote: fraction of overlapping neighbours that disagree."""
    s = Xn[..., FI["incons_frac"]].copy()
    s[Xn[..., FI["delivered"]] == 0] = 1.0  # a silent camera loses the vote
    return s.T


def score_static_group(Xe: np.ndarray, Xn: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Majority inside a fixed pre-provisioned redundancy group (no other neighbours)."""
    T, C = Xn.shape[:2]
    inc = np.maximum(Xe[..., 0], Xe[..., 1]) > 0.5
    val = Xe[..., 4] > 0
    g = groups[None]
    nv = (val & g).sum(2)
    s = np.where(nv > 0, (inc & val & g).sum(2) / np.maximum(nv, 1), 0.0)
    s[Xn[..., FI["delivered"]] == 0] = 1.0
    return s.T


# ---------------------------------------------------------------------- selection
def run_selection(policy: str, cover: np.ndarray, w: np.ndarray, Q: np.ndarray, risk: np.ndarray,
                  budget: int, static_set: np.ndarray | None = None, cache: dict | None = None) -> np.ndarray:
    C, T = Q.shape
    sel = np.zeros((C, T), bool)
    cur = None
    cache = {} if cache is None else cache
    for t in range(T):
        q = Q[:, t]
        if policy == "static":
            cur = static_set.copy()
        elif policy == "greedy":
            cur = greedy_handoff(cover, w, q, budget, current=cur)
        elif policy == "milp":
            key = (q.tobytes(), np.round(risk[:, t], 1).tobytes())
            if key not in cache:
                cache[key] = solve_handoff(cover, w, risk[:, t], q, budget)
            cur = cache[key]
        sel[:, t] = cur
    return sel


# ------------------------------------------------------------------------ metrics
def metrics(ep, cover: np.ndarray, w: np.ndarray, sel: np.ndarray, Q: np.ndarray, score: np.ndarray) -> dict:
    trusted = sel & ~Q
    good = ep.good
    covered_good = ((trusted & good).astype(float).T @ cover.astype(float)) > 0          # (T, Z)
    bad_trusted = trusted & ~good & ep.label
    deceived = (((bad_trusted).astype(float).T @ cover.astype(float)) > 0) & ~covered_good
    dt_min = 1.0 / FPS / 60.0
    ep_hours = ep.T / FPS / 3600.0
    blind = float(((~covered_good) * w[None]).sum() * dt_min)
    dec = float((deceived * w[None]).sum() * dt_min)
    # false alarms: quarantine onsets on cameras not under active attack
    onset = Q & ~np.roll(Q, 1, axis=1)
    onset[:, 0] = Q[:, 0]
    fa_events = int((onset & ~ep.label).sum())
    fa_cam_minutes = float((Q & ~ep.label).sum() * dt_min)
    out = {"blind_min": blind, "blind_min_per_h": blind / ep_hours, "deceived_min": dec,
           "deceived_min_per_h": dec / ep_hours, "fa_per_h": fa_events / ep_hours,
           "fa_cam_min_per_h": fa_cam_minutes / ep_hours,
           "disruption_cost_per_h": (fa_events + fa_cam_minutes) / ep_hours}
    # detection latency on compromised cameras (seconds); censored at window end
    lats = []
    for b in ep.byz:
        hit = np.where(Q[b, ep.onset:ep.onset + ep.dur])[0]
        lats.append(hit[0] / FPS if len(hit) else ep.dur / FPS)
    out["det_latency_s"] = float(np.mean(lats)) if lats else 0.0
    out["det_rate"] = float(np.mean([Q[b, ep.onset:ep.onset + ep.dur].any() for b in ep.byz])) if ep.byz else 1.0
    y, s = ep.label.ravel(), score.ravel()
    out["frame_tpr"] = float(Q.ravel()[y].mean()) if y.any() else np.nan
    out["frame_fpr"] = float(Q.ravel()[~y].mean())
    out["_y"], out["_s"] = y, s
    return out
