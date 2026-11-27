import itertools

import numpy as np
import pytest

from theory.handoff_milp import aggregate_zones, greedy_handoff, solve_handoff
from theory.quorum_bound import (indistinguishable_pair, m_detect, m_mask, m_probabilistic, m_required,
                                 max_tolerable_k, verification_table, verify)


def test_closed_forms():
    assert [m_detect(k) for k in (1, 2, 3)] == [2, 3, 4]
    assert [m_mask(k) for k in (1, 2, 3)] == [3, 5, 7]
    assert m_detect(2, off_plane=True) == 4
    assert m_mask(2, off_plane=True) == 6


def test_exhaustive_verification_matches_closed_form():
    tab = verification_table(rmax=8, kmax=3)
    assert all(t["detect"] == t["detect_theory"] and t["mask"] == t["mask_theory"] for t in tab)


@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("off", [False, True])
def test_lower_bound_witness_worlds(k, off):
    """At r = m_mask - 1 two worlds with different truth produce identical reports."""
    r = m_mask(k, off) - 1
    pair = indistinguishable_pair(r, k, off)
    assert pair is not None
    assert verify(m_mask(k, off), k, off).mask_always


def test_bounded_degree_cap():
    assert max_tolerable_k(6) == 3            # WILDTRACK: Delta = 6 -> 2k+1 <= 7
    assert max_tolerable_k(6, off_plane=True) == 2
    assert np.isinf(m_required(5, 3))
    assert m_required(7, 3) == 7


def test_probabilistic_redundancy_monotone():
    a = m_probabilistic(2, 0.99, 0.05)
    b = m_probabilistic(2, 0.80, 0.05)
    assert a <= b
    assert m_probabilistic(1, 1.0, 0.05) == m_detect(1)


def test_zone_aggregation_is_exact():
    rng = np.random.default_rng(0)
    cover = rng.random((5, 60)) < 0.4
    w = rng.random(60)
    A, wz = aggregate_zones(cover, w)
    assert np.isclose(wz.sum(), w.sum())
    assert A.shape[0] == 5


def test_milp_matches_bruteforce():
    rng = np.random.default_rng(1)
    cover = rng.random((6, 40)) < 0.35
    w = rng.random(40)
    w /= w.sum()
    q = np.zeros(6, bool)
    x = solve_handoff(cover, w, np.zeros(6), q, 2, gamma=0.0, lam=0.0)
    best = max(w[cover[list(s)].any(0)].sum() for s in itertools.combinations(range(6), 2))
    assert np.isclose(w[cover[x].any(0)].sum(), best)
    assert x.sum() <= 2


def test_milp_respects_quarantine_and_greedy_budget():
    cover = np.eye(4, dtype=bool).repeat(3, axis=1)
    w = np.ones(12) / 12
    q = np.array([True, False, False, False])
    x = solve_handoff(cover, w, np.zeros(4), q, 2)
    assert not x[0] or (cover[0] & ~q[0]).sum() == 0
    g = greedy_handoff(cover, w, q, 2)
    assert g.sum() == 2 and not g[0]
