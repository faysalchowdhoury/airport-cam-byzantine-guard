"""Per-frame latency of the online attestation pipeline (CPU, single process).

Stages measured on real frames for the full 7-camera network:
  flow      Horn-Schunck optical flow, one camera (runs at each camera/edge node)
  geometry  all pairwise epipolar/homography consistency checks for the frame
  gnn       one causal ST-GNN step for all cameras
  milp      one uncached MILP hand-off solve
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from data_pipeline.provenance import ARTIFACTS
from experiments import flow as flowmod
from experiments.scene import load_scene, visibility_graph
from experiments.streams import EDGE_FEATS, NODE_FEATS, pair_consistency
from models.st_gnn_attestor import STGNNAttestor
from theory.handoff_milp import solve_handoff


def run(n_frames: int = 150, seed: int = 42) -> dict:
    torch.set_num_threads(1)
    sc = load_scene()
    ov = visibility_graph(sc)
    C = sc.n_cams
    m = STGNNAttestor(len(NODE_FEATS), len(EDGE_FEATS), node_names=NODE_FEATS)
    p = ARTIFACTS / f"st_gnn_seed{seed}.pt"
    if p.exists():
        m.load_state_dict(torch.load(p))
    m.eval()
    rng = np.random.default_rng(seed)
    adj = torch.from_numpy((ov > 0).astype(np.float32))[None]
    state = m.init_state(1, C)
    rec = {"flow_ms_per_camera": [], "geometry_ms": [], "gnn_ms": [], "milp_ms": [], "attestor_ms": []}
    flowmod.flow_stats.cache_clear()
    for t in range(201, 201 + n_frames):
        t0 = time.perf_counter()
        flowmod.flow_stats(int(rng.integers(C)), t - 1, t)
        rec["flow_ms_per_camera"].append((time.perf_counter() - t0) * 1e3)
        t1 = time.perf_counter()
        for i in range(C):
            for j in range(i + 1, C):
                if ov[i, j] > 0:
                    pair_consistency(sc, i, j, sc.det[t][i], sc.det[t][j], key=None)
        t2 = time.perf_counter()
        xn = torch.randn(1, C, len(NODE_FEATS)) * 0.1
        xe = torch.rand(1, C, C, len(EDGE_FEATS))
        with torch.no_grad():
            _, state = m.step(xn, xe, adj, state)
        t3 = time.perf_counter()
        rec["geometry_ms"].append((t2 - t1) * 1e3)
        rec["gnn_ms"].append((t3 - t2) * 1e3)
        rec["attestor_ms"].append((t3 - t1) * 1e3)
        if t % 5 == 0:
            t4 = time.perf_counter()
            solve_handoff(sc.cover, sc.weights, rng.random(C), rng.random(C) < 0.2, 3)
            rec["milp_ms"].append((time.perf_counter() - t4) * 1e3)
    out = {k: {"mean": float(np.mean(v)), "p50": float(np.median(v)), "p95": float(np.percentile(v, 95)),
               "p99": float(np.percentile(v, 99)), "samples": [float(x) for x in v]} for k, v in rec.items()}
    out["n_params"] = int(sum(p.numel() for p in m.parameters()))
    out["threads"] = 1
    (ARTIFACTS / "latency.json").write_text(json.dumps(out), encoding="utf-8")
    return out


if __name__ == "__main__":
    o = run()
    print({k: (round(v["p50"], 2), round(v["p95"], 2)) for k, v in o.items() if isinstance(v, dict)})
