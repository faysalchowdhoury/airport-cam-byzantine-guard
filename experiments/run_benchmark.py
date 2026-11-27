"""Multi-seed benchmark: baselines vs. the availability-first SCSPV attestor.

Protocol
  * real WILDTRACK cameras; camera subsets n' in {5,6,7}, k in {1,2,3}
    -> k/n' in {0.14, 0.17, 0.20, 0.29, 0.33, 0.40, 0.43};
  * train / validation episodes use frames 0-199, test episodes frames 200-399
    (temporally disjoint);
  * attacks: replay (drone-relayed re-injection of the camera's own past frames),
    Wi-Fi deauth burst loss, RTSP saturation, colluding metadata fabrication,
    and a mixed campaign;
  * decision thresholds chosen on validation episodes by minimising the airport
    SCSPV loss; Pareto curves sweep the threshold on test episodes.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch

from data_pipeline.gate0 import require_pass
from data_pipeline.provenance import ARTIFACTS
from experiments import controller as ctl
from experiments.adsb_prior import prior_for_episode
from experiments.scene import load_scene, visibility_graph
from experiments.streams import ATTACK_SET, EDGE_FEATS, NODE_FEATS, episode_features, make_episode
from models.st_gnn_attestor import STGNNAttestor
from theory.handoff_milp import static_redundant_set

SEEDS = [42, 101, 202, 303, 404]
CONFIGS = [(7, 1), (6, 1), (5, 1), (7, 2), (6, 2), (5, 2), (7, 3)]
# Site-class-aware security priority vector (availability, integrity, disruption)
SCSPV = {"airport": {"availability": 0.55, "integrity": 0.30, "disruption": 0.15},
         "nuclear": {"availability": 0.15, "integrity": 0.70, "disruption": 0.15}}
THETAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
METHODS = ["single_camera", "majority_vote", "static_redundancy", "scspv_attestor"]
ABLATIONS = {"full": {}, "no_epipolar": {"use_epipolar": False},
             "no_telemetry": {"use_telemetry": False}, "no_adsb": {"use_adsb": False}}


def budget(n: int) -> int:
    """Analytics/decoder slots: airports run far more cameras than live-analytics capacity."""
    return math.ceil(0.4 * n)


def zone_weights(sc, cams):
    cov = sc.cover[cams]
    w = sc.weights * cov.any(0)
    return cov, w / w.sum()


def build(sc, overlap, rng, n_eps, base, T=200, spoof_p=0.3, configs=None, attacks=None):
    eps = []
    plan = []
    if configs is None:
        for _ in range(n_eps):
            plan.append((CONFIGS[rng.integers(len(CONFIGS))], ATTACK_SET[rng.integers(len(ATTACK_SET))]))
    else:
        for cfg in configs:
            for a in attacks:
                for _ in range(n_eps):
                    plan.append((cfg, a))
    for (n, k), a in plan:
        cams = sorted(rng.choice(7, size=n, replace=False).tolist())
        spoof = rng.random() < spoof_p
        prior = prior_for_episode(rng, T, spoof_from="ATL" if spoof else None)
        ep = make_episode(sc, rng, cams, k, a, base, T, adsb=prior)
        ep.spoofed = spoof
        ep.cfg = (n, k)
        Xn, Xe, ov = episode_features(sc, ep, overlap)
        eps.append((ep, Xn, Xe, (ov > 0).astype(np.float32)))
    return eps


def train_gnn(train, seed, epochs, **kw):
    torch.manual_seed(seed)
    m = STGNNAttestor(len(NODE_FEATS), len(EDGE_FEATS), node_names=NODE_FEATS, **kw)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3, weight_decay=1e-4)
    pos = sum(e[0].label.sum() for e in train)
    neg = sum((~e[0].label).sum() for e in train)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / max(pos, 1)))
    rng = np.random.default_rng(seed)
    by_c: dict[int, list] = {}
    for e in train:
        by_c.setdefault(e[0].C, []).append(e)
    bs = 8
    for ep_i in range(epochs):
        batches = []
        for C, lst in by_c.items():
            idx = rng.permutation(len(lst))
            batches += [[lst[j] for j in idx[s:s + bs]] for s in range(0, len(idx), bs)]
        rng.shuffle(batches)
        for bt in batches:
            Xn = torch.from_numpy(np.stack([b[1] for b in bt]))
            Xe = torch.from_numpy(np.stack([b[2] for b in bt]))
            adj = torch.from_numpy(np.stack([b[3] for b in bt]))
            y = torch.from_numpy(np.stack([b[0].label.T for b in bt]).astype(np.float32))
            lg = m(Xn, Xe, adj)
            loss = lossf(lg, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
    m.eval()
    return m


def gnn_scores(m, data):
    out = []
    with torch.no_grad():
        for ep, Xn, Xe, adj in data:
            lg = m(torch.from_numpy(Xn)[None], torch.from_numpy(Xe)[None], torch.from_numpy(adj)[None])
            out.append(torch.sigmoid(lg[0]).numpy().T)
    return out


def method_scores(method, data, gnn_s=None, static_sets=None):
    out = []
    for i, (ep, Xn, Xe, adj) in enumerate(data):
        if method == "single_camera":
            out.append(ctl.score_single_camera(Xn))
        elif method == "majority_vote":
            out.append(ctl.score_majority(Xn))
        elif method == "static_redundancy":
            S = static_sets[i]
            groups = (adj > 0) & S[None, :] & S[:, None]
            out.append(ctl.score_static_group(Xe, Xn, groups))
        else:
            out.append(gnn_s[i])
    return out


def evaluate(method, data, scores, theta, sc, static_sets, milp_cache, policy=None):
    rows = []
    for i, (ep, Xn, Xe, adj) in enumerate(data):
        cov, w = zone_weights(sc, ep.cams)
        Q = ctl.quarantine(scores[i], theta)
        B = budget(ep.C)
        pol = policy or {"single_camera": "greedy", "majority_vote": "greedy",
                         "static_redundancy": "static", "scspv_attestor": "milp"}[method]
        risk = np.round(np.clip(scores[i], 0, 1) * 4) / 4
        cache = milp_cache.setdefault(tuple(ep.cams), {})
        sel = ctl.run_selection(pol, cov, w, Q, risk, B, static_set=static_sets[i], cache=cache)
        r = ctl.metrics(ep, cov, w, sel, Q, scores[i])
        r.update(attack=("mixed" if len(set(ep.attack.values())) > 1 else next(iter(ep.attack.values()), "none")),
                 attack_label=getattr(ep, "attack_label", None), n=ep.C, k=len(ep.byz),
                 spoofed=bool(ep.spoofed))
        rows.append(r)
    return rows


def scspv_loss(rows, site="airport"):
    v = SCSPV[site]
    b = np.mean([r["blind_min_per_h"] for r in rows])
    d = np.mean([r["deceived_min_per_h"] for r in rows])
    f = np.mean([r["disruption_cost_per_h"] for r in rows])
    return v["availability"] * b + v["integrity"] * d + v["disruption"] * f


def auroc(y, s):
    y = np.asarray(y, bool)
    if y.all() or (~y).all():
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    npos = y.sum()
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * (len(y) - npos)))


def strip(rows):
    return [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]


def run_seed(seed, args, sc, overlap):
    rng = np.random.default_rng(seed)
    t0 = time.time()
    train = build(sc, overlap, rng, args.train_eps, base=0)
    val = build(sc, overlap, rng, args.val_eps, base=0)
    test = build(sc, overlap, rng, args.test_eps, base=200, configs=CONFIGS, attacks=ATTACK_SET)
    for ep in test:
        ep[0].attack_label = None
    # remember the scheduled attack label per test episode
    idx = 0
    for cfg in CONFIGS:
        for a in ATTACK_SET:
            for _ in range(args.test_eps):
                test[idx][0].attack_label = a
                idx += 1
    print(f"[seed {seed}] features built in {time.time() - t0:.0f}s", flush=True)
    models = {}
    for name, kw in ABLATIONS.items():
        t1 = time.time()
        models[name] = train_gnn(train, seed, args.epochs, **kw)
        print(f"[seed {seed}] trained {name} in {time.time() - t1:.0f}s", flush=True)
    milp_cache: dict = {}
    static_val = [static_redundant_set(*zone_weights(sc, e[0].cams), budget(e[0].C)) for e in val]
    static_test = [static_redundant_set(*zone_weights(sc, e[0].cams), budget(e[0].C)) for e in test]
    out = {"seed": seed, "methods": {}, "ablations": {}, "pareto": {}, "spoof": {}}
    gv = {n: gnn_scores(m, val) for n, m in models.items()}
    gt = {n: gnn_scores(m, test) for n, m in models.items()}
    for method in METHODS:
        sv = method_scores(method, val, gv["full"], static_val)
        st = method_scores(method, test, gt["full"], static_test)
        losses = {th: scspv_loss(evaluate(method, val, sv, th, sc, static_val, milp_cache)) for th in THETAS}
        th_star = min(losses, key=losses.get)
        pareto = []
        rows_star = None
        for th in THETAS:
            rows = evaluate(method, test, st, th, sc, static_test, milp_cache)
            pareto.append({"theta": th, "blind_min_per_h": float(np.mean([r["blind_min_per_h"] for r in rows])),
                           "disruption_cost_per_h": float(np.mean([r["disruption_cost_per_h"] for r in rows])),
                           "deceived_min_per_h": float(np.mean([r["deceived_min_per_h"] for r in rows]))})
            if th == th_star:
                rows_star = rows
        y = np.concatenate([r["_y"] for r in rows_star])
        s = np.concatenate([r["_s"] for r in rows_star])
        out["methods"][method] = {"theta": th_star, "val_loss": losses[th_star], "frame_auroc": auroc(y, s),
                                  "rows": strip(rows_star)}
        out["pareto"][method] = pareto
        print(f"[seed {seed}] {method}: theta={th_star} blind/h={np.mean([r['blind_min_per_h'] for r in rows_star]):.2f} "
              f"FA/h={np.mean([r['fa_per_h'] for r in rows_star]):.1f} auroc={out['methods'][method]['frame_auroc']:.3f}",
              flush=True)
    # ablations (ST-GNN variants, MILP vs greedy hand-off)
    th = out["methods"]["scspv_attestor"]["theta"]
    for name in ABLATIONS:
        rows = evaluate("scspv_attestor", test, gt[name], th, sc, static_test, milp_cache)
        y = np.concatenate([r["_y"] for r in rows])
        s = np.concatenate([r["_s"] for r in rows])
        out["ablations"][name] = {"frame_auroc": auroc(y, s), "rows": strip(rows)}
    rows = evaluate("scspv_attestor", test, gt["full"], th, sc, static_test, milp_cache, policy="greedy")
    out["ablations"]["greedy_handoff"] = {"frame_auroc": out["methods"]["scspv_attestor"]["frame_auroc"],
                                          "rows": strip(rows)}
    rows = evaluate("scspv_attestor", test, gt["full"], th, sc, static_test, milp_cache, policy="static")
    out["ablations"]["no_handoff"] = {"frame_auroc": out["methods"]["scspv_attestor"]["frame_auroc"],
                                      "rows": strip(rows)}
    # untrusted-prior robustness: score shift when the real JFK prior is swapped for real ATL traffic
    shifts = []
    with torch.no_grad():
        for ep, Xn, Xe, adj in test[:20]:
            Xs = Xn.copy()
            Xs[..., NODE_FEATS.index("adsb_prior")] = prior_for_episode(np.random.default_rng(seed), ep.T,
                                                                       spoof_from="ATL")[:, None]
            a = torch.sigmoid(models["full"](torch.from_numpy(Xn)[None], torch.from_numpy(Xe)[None],
                                             torch.from_numpy(adj)[None]))
            b = torch.sigmoid(models["full"](torch.from_numpy(Xs)[None], torch.from_numpy(Xe)[None],
                                             torch.from_numpy(adj)[None]))
            shifts.append(float((a - b).abs().max()))
    out["spoof"] = {"max_score_shift": float(np.max(shifts)), "mean_max_score_shift": float(np.mean(shifts)),
                    "logit_bound_beta": models["full"].beta}
    torch.save(models["full"].state_dict(), ARTIFACTS / f"st_gnn_seed{seed}.pt")
    print(f"[seed {seed}] done in {time.time() - t0:.0f}s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--train-eps", type=int, default=60)
    ap.add_argument("--val-eps", type=int, default=20)
    ap.add_argument("--test-eps", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()
    require_pass()
    sc = load_scene()
    overlap = visibility_graph(sc)
    for seed in args.seeds:
        res = run_seed(seed, args, sc, overlap)
        (ARTIFACTS / "runs").mkdir(parents=True, exist_ok=True)
        (ARTIFACTS / "runs" / f"seed_{seed}.json").write_text(json.dumps(res), encoding="utf-8")
    json.dump(vars(args), open(ARTIFACTS / "runs" / "config.json", "w"), indent=2)


if __name__ == "__main__":
    main()
