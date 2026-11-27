r"""Availability-weighted MILP optical hand-off (scipy.optimize.milp).

Decision epoch: choose which camera streams receive the B analytics/decoder
slots, given attestation risk r_c in [0,1] and quarantine flags q_c.

    max   sum_z w_z y_z + gamma * sum_z w_z u_z - lambda * sum_c r_c x_c
    s.t.  y_z        <= sum_c a_cz (1-q_c) x_c          (one trusted witness)
          y_z + u_z  <= sum_c a_cz (1-q_c) x_c          (second witness bonus)
          sum_c x_c  <= B
          x in {0,1}^C, y in {0,1}^Z, u in [0,1]^Z

w_z are SCSPV-scaled coverage weights (availability-first: passenger density x
zone class, optionally modulated by a *bounded* ADS-B/Remote-ID prior). The
second-witness term rewards keeping cross-view attestation alive (m >= 2) without
letting it dominate first-witness availability (gamma < 1).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, hstack, identity, vstack


def aggregate_zones(cover: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Merge zones with identical coverage signatures (exact: their constraints coincide)."""
    keep = w > 0
    cols = cover[:, keep]
    sig, inv = np.unique(cols.T, axis=0, return_inverse=True)
    wz = np.zeros(len(sig))
    np.add.at(wz, inv.ravel(), w[keep])
    return sig.T.astype(float), wz


def solve_handoff(cover: np.ndarray, w: np.ndarray, risk: np.ndarray, quarantined: np.ndarray,
                  budget: int, gamma: float = 0.25, lam: float = 0.02,
                  time_limit: float = 2.0) -> np.ndarray:
    """Return boolean selection x (C,)."""
    C, Z = cover.shape
    A, wz = aggregate_zones(cover, w)
    Zk = A.shape[1]
    trust = (~quarantined).astype(float)
    At = (A * trust[:, None]).T                        # (Zk, C)
    # variable order: x (C), y (Zk), u (Zk)
    c = np.concatenate([lam * risk, -wz, -gamma * wz])
    I = identity(Zk, format="csr")
    Z0 = csr_matrix((Zk, Zk))
    At_s = csr_matrix(At)
    cons1 = hstack([-At_s, I, Z0])
    cons2 = hstack([-At_s, I, I])
    cons3 = hstack([csr_matrix(np.ones((1, C))), csr_matrix((1, 2 * Zk))])
    Acon = vstack([cons1, cons2, cons3]).tocsr()
    ub = np.concatenate([np.zeros(2 * Zk), [budget]])
    lb = np.full(len(ub), -np.inf)
    integrality = np.concatenate([np.ones(C), np.ones(Zk), np.zeros(Zk)])
    bounds = Bounds(np.zeros(C + 2 * Zk), np.ones(C + 2 * Zk))
    res = milp(c, constraints=LinearConstraint(Acon, lb, ub), integrality=integrality,
               bounds=bounds, options={"time_limit": time_limit, "disp": False})
    if res.x is None:
        return greedy_handoff(cover, w, quarantined, budget)
    return res.x[:C] > 0.5


def greedy_handoff(cover: np.ndarray, w: np.ndarray, quarantined: np.ndarray, budget: int,
                   current: np.ndarray | None = None) -> np.ndarray:
    """Greedy max-coverage selection (baseline hand-off). Keeps current members if allowed."""
    C = cover.shape[0]
    sel = np.zeros(C, bool)
    if current is not None:
        sel = current & ~quarantined
    covered = cover[sel].any(0) if sel.any() else np.zeros(cover.shape[1], bool)
    while sel.sum() < budget:
        gains = np.array([(w * (cover[c] & ~covered)).sum() if (not sel[c] and not quarantined[c]) else -1
                          for c in range(C)])
        best = int(np.argmax(gains))
        if gains[best] < 0:
            break
        sel[best] = True
        covered |= cover[best]
    return sel


def static_redundant_set(cover: np.ndarray, w: np.ndarray, budget: int) -> np.ndarray:
    """Pre-provisioned fixed set maximising two-witness coverage (no runtime hand-off)."""
    return solve_handoff(cover, w, np.zeros(cover.shape[0]), np.zeros(cover.shape[0], bool),
                         budget, gamma=1.0, lam=0.0)
