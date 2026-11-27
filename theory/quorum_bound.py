r"""Bounded-degree visibility-graph redundancy bounds (Compression-by-Contradiction).

Model
-----
A target at world point X is observed by its coverage set C(X), r = |C(X)|.
At most k cameras (set B) are Byzantine and may report anything; honest cameras
report the projection of the true target. A camera's report *supports* a
hypothesis X' when it is geometrically consistent with X' (ground plane:
homography transfer; off-plane: the epipolar constraint p^T E p' = 0).

* Ground-plane targets: an honest view determines the ground point uniquely
  (invertible homography), so it supports exactly one hypothesis.
* Off-plane targets: an honest view only fixes a ray; a fabricated X' placed on
  that ray is supported by the honest view (epipolar degeneracy).

Results (proved in the paper, verified exhaustively below):
  detection  (some honest witness contradicts any fabrication):
      ground  m_det  = k + 1,   off-plane  m_det  = k + 2
  masking    (truth is recovered by maximum support despite B):
      ground  m_mask = 2k + 1,  off-plane  m_mask = 2k + 2
Bounded degree: all cameras covering X are pairwise adjacent in the visibility
graph G, so r(X) <= omega(G) <= Delta(G) + 1. Hence attestation at level m is
only possible if Delta + 1 >= m, i.e. k <= floor(Delta / 2) for ground masking.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np


# ----------------------------------------------------------------------- closed forms
def m_detect(k: int, off_plane: bool = False) -> int:
    return k + 1 + int(off_plane)


def m_mask(k: int, off_plane: bool = False) -> int:
    return 2 * k + 1 + int(off_plane)


def m_required(n: int, k: int, mode: str = "mask", off_plane: bool = False,
               delta: int | None = None) -> float:
    """Minimum per-zone redundancy m(n,k); ``inf`` when the network cannot supply it."""
    m = m_mask(k, off_plane) if mode == "mask" else m_detect(k, off_plane)
    cap = n if delta is None else min(n, delta + 1)
    return float(m) if m <= cap else math.inf


def m_probabilistic(k: int, p_witness: float, delta_fail: float, off_plane: bool = False) -> int:
    """Detection redundancy when each honest witness flags a fabrication w.p. p.

    Miss probability (1-p)^(m-k-[off_plane]) <= delta_fail.
    """
    if p_witness >= 1.0:
        return m_detect(k, off_plane)
    if p_witness <= 0.0:
        return int(1e9)
    need = math.ceil(math.log(delta_fail) / math.log(1.0 - p_witness))
    return k + int(off_plane) + max(1, need)


def max_tolerable_k(delta: int, mode: str = "mask", off_plane: bool = False) -> int:
    """Largest k attestable on a visibility graph of maximum degree ``delta``."""
    k = 0
    while (m_mask(k + 1, off_plane) if mode == "mask" else m_detect(k + 1, off_plane)) <= delta + 1:
        k += 1
    return k


# ------------------------------------------------------------- exhaustive verification
@dataclass
class Outcome:
    r: int
    k: int
    off_plane: bool
    detect_always: bool
    mask_always: bool
    witness_world_pair: tuple | None


def _support(reports: list[set], hyp: int) -> int:
    return sum(hyp in s for s in reports)


def verify(r: int, k: int, off_plane: bool) -> Outcome:
    """Exhaustive check over Byzantine sets and report strategies.

    Hypothesis space {0 = truth, 1 = fabrication}; each report is the set of
    hypotheses it is consistent with. Honest camera 0 is the *degenerate* view in
    the off-plane case (fabrication placed on its ray), so it supports both.
    Byzantine reports range over {{0}, {1}, {0,1}, {}} (anything).
    """
    choices = [frozenset({0}), frozenset({1}), frozenset({0, 1}), frozenset()]
    detect_ok, mask_ok, pair = True, True, None
    for B in itertools.combinations(range(r), min(k, r)):
        honest = [i for i in range(r) if i not in B]
        for strat in itertools.product(choices, repeat=len(B)):
            reports = [None] * r
            for i, s in zip(B, strat):
                reports[i] = set(s)
            for i in honest:
                reports[i] = {0, 1} if (off_plane and i == 0) else {0}
            fabricated = any(1 in reports[i] for i in B) or not B
            contradicted = any(reports[i] == {0} for i in honest)
            if fabricated and B and not contradicted:
                detect_ok = False
            s0, s1 = _support(reports, 0), _support(reports, 1)
            if not s0 > s1:
                mask_ok = False
    if not mask_ok:
        pair = indistinguishable_pair(r, k, off_plane)
    return Outcome(r, k, off_plane, detect_ok, mask_ok, pair)


def indistinguishable_pair(r: int, k: int, off_plane: bool):
    """Construct two worlds (truth 0 vs truth 1) with identical report vectors.

    Exists iff masking is impossible: this is the contradiction at the heart of
    the lower bound (the adversary 'compresses' the fabricated trajectory into
    its k views plus any degenerate honest view and matches the truth's support).
    """
    for B0 in itertools.combinations(range(r), min(k, r)):
        for B1 in itertools.combinations(range(r), min(k, r)):
            rep0, rep1 = [], []
            for i in range(r):
                degenerate = off_plane and i == 0
                h0 = {0, 1} if degenerate else {0}
                h1 = {0, 1} if degenerate else {1}
                # world 0 (truth 0): honest outside B0; Byzantine in B0 mimic world 1 honest
                rep0.append(h1 if i in B0 else h0)
                rep1.append(h0 if i in B1 else h1)
            if rep0 == rep1:
                return {"B_world_truth0": B0, "B_world_truth1": B1,
                        "reports": [sorted(s) for s in rep0]}
    return None


def verification_table(rmax: int = 9, kmax: int = 4) -> list[dict]:
    rows = []
    for off in (False, True):
        for k in range(1, kmax + 1):
            for r in range(1, rmax + 1):
                o = verify(r, k, off)
                rows.append({"r": r, "k": k, "off_plane": off, "detect": o.detect_always,
                             "mask": o.mask_always,
                             "detect_theory": r >= m_detect(k, off), "mask_theory": r >= m_mask(k, off)})
    return rows


def worst_case_zone_survival(cover: np.ndarray, weights: np.ndarray, k: int, m: int) -> tuple[float, tuple]:
    """Adversary picks k cameras to minimise weight of zones keeping >= m honest witnesses."""
    C = cover.shape[0]
    best, argbest = math.inf, ()
    red = cover.sum(0)
    for B in itertools.combinations(range(C), k):
        honest_red = red - cover[list(B)].sum(0)
        val = float(weights[honest_red >= m].sum())
        if val < best:
            best, argbest = val, B
    return best, argbest


if __name__ == "__main__":
    tab = verification_table()
    bad = [t for t in tab if t["detect"] != t["detect_theory"] or t["mask"] != t["mask_theory"]]
    print(f"{len(tab)} (r,k,geometry) cases checked; mismatches with closed form: {len(bad)}")
