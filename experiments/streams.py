"""Attack emulation on real camera streams + per-frame attestation features.

The content of every stream is real WILDTRACK data. Attacks only *re-route* real
content (replay of the camera's own past frames, freezes, drops) or tamper with
the analytics metadata a compromised edge camera reports (phantom / erased
detections). No pixels are synthesised. Transport telemetry (arrival latency,
jitter, drops) is produced by standard channel models (Gaussian jitter,
Gilbert-Elliott burst loss, saturation ramp) because WILDTRACK carries none;
this is stated explicitly in the paper.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from data_pipeline.wildtrack import IMG_H, IMG_W
from experiments import flow as flowmod
from experiments.scene import Scene

ATTACKS = ["replay", "deauth", "saturation", "fabrication"]
ATTACK_SET = ATTACKS + ["mixed"]
GATE_CM = 100.0
NODE_FEATS = ["delivered", "lat_z", "jitter", "gap", "flow_mag_z", "flow_resid_z", "diff_z", "freeze",
              "ndet", "incons_mean", "incons_max", "incons_frac", "resid_mean", "epi_mean", "adsb_prior"]
EDGE_FEATS = ["incons_ij", "incons_ji", "resid", "epi", "valid", "overlap"]


@dataclass
class Episode:
    cams: list[int]                    # global camera indices in this episode
    byz: list[int]                     # local indices of compromised cameras
    attack: dict[int, str]             # local idx -> attack type
    base: int                          # first global frame index
    T: int
    onset: int
    dur: int
    src: np.ndarray                    # (C, T) source frame index, -1 when dropped
    latency: np.ndarray                # (C, T) ms
    label: np.ndarray                  # (C, T) bool compromised & active
    good: np.ndarray                   # (C, T) bool: delivers live, untampered content
    benign: np.ndarray                 # (C, T) bool benign congestion
    fab_add: dict = field(default_factory=dict)      # (c, t) -> (m, 6) phantom rows
    fab_remove: dict = field(default_factory=dict)   # (c, t) -> set(person ids)
    adsb: np.ndarray | None = None     # (T,) weak prior (untrusted)

    @property
    def C(self) -> int:
        return len(self.cams)


# ----------------------------------------------------------------- episode synthesis
def make_episode(sc: Scene, rng: np.random.Generator, cams: list[int], k: int, attack: str,
                 base: int, T: int, adsb: np.ndarray | None = None,
                 benign_prob: float = 0.35) -> Episode:
    C = len(cams)
    byz = sorted(rng.choice(C, size=k, replace=False).tolist()) if k else []
    onset = int(rng.integers(int(0.2 * T), int(0.45 * T)))
    dur = int(rng.integers(int(0.3 * T), int(0.5 * T)))
    types = {b: (rng.choice(ATTACKS) if attack == "mixed" else attack) for b in byz}
    src = np.tile(np.arange(base, base + T), (C, 1))
    mu = rng.uniform(40, 120, C)
    sig = rng.uniform(2, 8, C)
    lat = mu[:, None] + rng.normal(0, 1, (C, T)) * sig[:, None]
    drop = rng.random((C, T)) < 0.005
    benign = np.zeros((C, T), bool)
    label = np.zeros((C, T), bool)
    fab = np.zeros((C, T), bool)
    ep = Episode(cams, byz, types, base, T, onset, dur, src, lat, label, None, benign, adsb=adsb)

    # benign congestion on an honest camera (false-alarm pressure)
    honest = [c for c in range(C) if c not in byz]
    if honest and rng.random() < benign_prob:
        c = int(rng.choice(honest))
        s = int(rng.integers(25, T - 35))
        d = int(rng.integers(10, 30))
        ramp = np.linspace(0, 1, d)
        lat[c, s:s + d] += rng.uniform(150, 600) * ramp + rng.normal(0, sig[c] * 6, d)
        drop[c, s:s + d] |= rng.random(d) < 0.15
        frz = rng.random(d) < 0.2
        for i in np.where(frz)[0]:
            src[c, s + i] = src[c, s + i - 1]
        benign[c, s:s + d] = True

    w = slice(onset, onset + dur)
    for b in byz:
        a = types[b]
        label[b, w] = True
        if a == "replay":
            delta = int(rng.integers(20, 60))
            src[b, w] = np.maximum(src[b, w] - delta, 0)
            # a careful relay buffers to mask its added path delay (telemetry-stealthy)
            lat[b, w] += rng.uniform(0, 8) + rng.normal(0, sig[b] * 0.3, dur)
        elif a == "deauth":
            state, p_gb, p_bg = 0, 0.15, 0.30
            for t in range(onset, onset + dur):
                state = (1 if rng.random() < p_gb else 0) if state == 0 else (0 if rng.random() < p_bg else 1)
                if state == 1 and rng.random() < 0.9:
                    drop[b, t] = True
                elif t > 0 and drop[b, t - 1]:
                    lat[b, t] += rng.uniform(200, 800)   # re-association spike
        elif a == "saturation":
            L = rng.uniform(400, 1500)
            lat[b, w] += L * np.minimum(1, np.linspace(0, 2, dur)) + rng.normal(0, sig[b] * rng.uniform(5, 12), dur)
            drop[b, w] |= rng.random(dur) < 0.2
            for t in range(onset, onset + dur):
                if rng.random() < 0.4:
                    src[b, t] = src[b, t - 1]
        elif a == "fabrication":
            fab[b, w] = True
    if any(types[b] == "fabrication" for b in byz):
        _fabricate(sc, rng, ep, [b for b in byz if types[b] == "fabrication"])
    src[drop] = -1
    ep.src, ep.latency = src, np.maximum(lat, 1.0)
    ep.good = (src == np.arange(base, base + T)[None, :]) & (ep.latency < 1000) & ~fab
    return ep


def _fabricate(sc: Scene, rng: np.random.Generator, ep: Episode, fab_cams: list[int]) -> None:
    """Colluding metadata attack: a geometrically consistent phantom + suppression.

    The adversary knows the calibration, so its phantom is consistent across all
    colluding views (ground homography and epipolar geometry). It places the
    phantom where the fewest honest cameras can contradict it.
    """
    cams = ep.cams
    gcov = sc.cover[cams]                               # (C, Z)
    byz_cov = gcov[fab_cams].any(0)
    honest = [c for c in range(ep.C) if c not in ep.byz]
    honest_red = gcov[honest].sum(0) if honest else np.zeros(gcov.shape[1])
    cand = np.where(byz_cov)[0]
    if len(cand) == 0:
        return
    hr = honest_red[cand]
    pool = cand[hr == hr.min()]
    z0 = int(rng.choice(pool))
    p = sc.cells[z0].astype(float) + rng.uniform(-40, 40, 2)
    speed = rng.uniform(30, 70)
    ang = rng.uniform(0, 2 * np.pi)
    v = speed * np.array([np.cos(ang), np.sin(ang)])
    # suppression target: real person nearest the phantom start at onset
    gidx = ep.base + ep.onset
    gw, gid = sc.gt_world[gidx], sc.gt_ids[gidx]
    victim = int(gid[np.argmin(np.linalg.norm(gw - p, axis=1))]) if len(gw) else None
    for t in range(ep.onset, ep.onset + ep.dur):
        cand_p = p + v
        if not (-300 <= cand_p[0] <= 900 and -900 <= cand_p[1] <= 2700):
            v = -v
            cand_p = p + v
        p = cand_p
        for b in fab_cams:
            g = sc.cams.index(sc.cams[cams[b]])
            foot, d1 = sc.project(g, p[None])
            head, d2 = sc.project(g, p[None], z=170.0)
            if d1[0] > 0 and 0 <= foot[0, 0] < IMG_W and 0 <= foot[0, 1] < IMG_H:
                gp = sc.to_ground(g, foot)[0]
                ep.fab_add[(b, t)] = np.array([[gp[0], gp[1], foot[0, 0], foot[0, 1], head[0, 0], head[0, 1]]])
            if victim is not None:
                ep.fab_remove[(b, t)] = {victim}


# ----------------------------------------------------------------------- observations
def detections(sc: Scene, ep: Episode, c: int, t: int):
    s = ep.src[c, t]
    if s < 0:
        return None
    g = ep.cams[c]
    d, ids = sc.det[s][g], sc.det_ids[s][g]
    rm = ep.fab_remove.get((c, t))
    if rm:
        keep = ~np.isin(ids, list(rm))
        d = d[keep]
    add = ep.fab_add.get((c, t))
    if add is not None:
        d = np.vstack([d, add])
    return d


def _in_view(sc: Scene, g: int, ground: np.ndarray) -> np.ndarray:
    if len(ground) == 0:
        return np.zeros(0, bool)
    uv, depth = sc.project(g, ground)
    return (depth > 0) & (uv[:, 0] >= 20) & (uv[:, 0] < IMG_W - 20) & (uv[:, 1] >= 20) & (uv[:, 1] < IMG_H - 20)


_PAIR_CACHE: dict = {}
_F_CACHE: dict = {}


def pair_consistency(sc: Scene, gi: int, gj: int, di: np.ndarray, dj: np.ndarray, key=None):
    """Cross-view consistency of two cameras' reports.

    Returns (incons_ij, incons_ji, mean matched ground residual [cm],
    mean symmetric epipolar distance of head points [px], valid).
    """
    if key is not None and key in _PAIR_CACHE:
        return _PAIR_CACHE[key]
    a = di[_in_view(sc, gj, di[:, :2])] if len(di) else di
    b = dj[_in_view(sc, gi, dj[:, :2])] if len(dj) else dj
    if len(a) == 0 and len(b) == 0:
        out = (0.0, 0.0, 0.0, 0.0, 0.0)
    elif len(a) == 0 or len(b) == 0:
        out = (1.0 if len(a) else 0.0, 1.0 if len(b) else 0.0, 0.0, 0.0, 1.0)
    else:
        D = np.linalg.norm(a[:, None, :2] - b[None, :, :2], axis=2)
        r, c = linear_sum_assignment(D)
        ok = D[r, c] < GATE_CM
        r, c = r[ok], c[ok]
        inc_ij = 1.0 - len(r) / len(a)
        inc_ji = 1.0 - len(c) / len(b)
        resid = float(D[r, c].mean()) if len(r) else GATE_CM
        epi = 0.0
        if len(r):
            if (gi, gj) not in _F_CACHE:
                _F_CACHE[(gi, gj)] = sc.fundamental(gi, gj)
            Fm = _F_CACHE[(gi, gj)]
            xi = np.c_[a[r, 4:6], np.ones(len(r))]
            xj = np.c_[b[c, 4:6], np.ones(len(r))]
            l2 = xi @ Fm.T
            l1 = xj @ Fm
            num = np.abs(np.sum(xj * l2, 1))
            epi = float(np.mean(0.5 * (num / np.hypot(l2[:, 0], l2[:, 1]) + num / np.hypot(l1[:, 0], l1[:, 1]))))
        out = (inc_ij, inc_ji, resid, epi, 1.0)
    if key is not None:
        _PAIR_CACHE[key] = out
    return out


def _robust(x: np.ndarray):
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826
    return med, max(mad, 1e-3)


def episode_features(sc: Scene, ep: Episode, overlap: np.ndarray, warm: int = 20,
                     use_flow: bool = True):
    """Node features (T, C, Fn), edge features (T, C, C, Fe)."""
    C, T = ep.C, ep.T
    Xn = np.zeros((T, C, len(NODE_FEATS)), np.float32)
    Xe = np.zeros((T, C, C, len(EDGE_FEATS)), np.float32)
    ov = overlap[np.ix_(ep.cams, ep.cams)]
    ovn = ov / max(ov.max(), 1)
    dets = [[detections(sc, ep, c, t) for t in range(T)] for c in range(C)]
    fl = np.zeros((C, T, 3))
    last = [None] * C
    gap = np.zeros((C, T))
    for c in range(C):
        g = ep.cams[c]
        for t in range(T):
            s = ep.src[c, t]
            if s < 0:
                gap[c, t] = (gap[c, t - 1] + 1) if t else 1
                continue
            if last[c] is not None and use_flow:
                fl[c, t] = flowmod.flow_stats(g, int(last[c]), int(s))
            last[c] = s
    lat = ep.latency
    for c in range(C):
        dl = ep.src[c] >= 0
        lw = lat[c, :warm][dl[:warm]] if dl[:warm].any() else lat[c, :warm]
        lm, ls = _robust(lw)
        jit = np.abs(np.diff(lat[c], prepend=lat[c, 0]))
        stats = [_robust(fl[c, 1:warm, i][dl[1:warm]]) if dl[1:warm].sum() > 3 else (0.0, 1.0) for i in range(3)]
        for t in range(T):
            d = dets[c][t]
            if d is None:
                Xn[t, c, 3] = min(gap[c, t], 8) / 4
                continue
            Xn[t, c, 0] = 1
            Xn[t, c, 1] = np.clip((lat[c, t] - lm) / ls, -5, 50) / 5
            Xn[t, c, 2] = min(jit[t] / 50.0, 20)
            Xn[t, c, 3] = 0
            for i in range(3):
                Xn[t, c, 4 + i] = np.clip((fl[c, t, i] - stats[i][0]) / stats[i][1], -10, 10) / 5
            Xn[t, c, 7] = float(fl[c, t, 2] == 0.0 and t > 0 and use_flow)
            Xn[t, c, 8] = len(d) / 20.0
    for t in range(T):
        for i in range(C):
            di = dets[i][t]
            for j in range(i + 1, C):
                dj = dets[j][t]
                if ov[i, j] <= 0 or di is None or dj is None:
                    continue
                gi, gj = ep.cams[i], ep.cams[j]
                key = None
                if (i, t) not in ep.fab_add and (i, t) not in ep.fab_remove and \
                        (j, t) not in ep.fab_add and (j, t) not in ep.fab_remove:
                    key = (gi, gj, int(ep.src[i, t]), int(ep.src[j, t]))
                inc_ij, inc_ji, res, epi, val = pair_consistency(sc, gi, gj, di, dj, key)
                Xe[t, i, j] = (inc_ij, inc_ji, res / 50, epi / 20, val, ovn[i, j])
                Xe[t, j, i] = (inc_ji, inc_ij, res / 50, epi / 20, val, ovn[i, j])
        val = Xe[t, :, :, 4]
        inc = np.maximum(Xe[t, :, :, 0], Xe[t, :, :, 1])
        nv = val.sum(1)
        wsum = (val * ovn).sum(1)
        Xn[t, :, 9] = np.where(wsum > 0, (inc * val * ovn).sum(1) / np.maximum(wsum, 1e-9), 0)
        Xn[t, :, 10] = (inc * val).max(1)
        Xn[t, :, 11] = np.where(nv > 0, ((inc > 0.5) * val).sum(1) / np.maximum(nv, 1), 0)
        Xn[t, :, 12] = np.where(nv > 0, (Xe[t, :, :, 2] * val).sum(1) / np.maximum(nv, 1), 0)
        Xn[t, :, 13] = np.where(nv > 0, (Xe[t, :, :, 3] * val).sum(1) / np.maximum(nv, 1), 0)
    if ep.adsb is not None:
        Xn[:, :, 14] = ep.adsb[:, None]
    return Xn, Xe, ov
