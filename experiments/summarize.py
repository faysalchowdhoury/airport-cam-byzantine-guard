"""Aggregate all artifacts into artifacts/results_summary.json (single source for paper tables)."""
from __future__ import annotations

import json

import numpy as np

from data_pipeline.provenance import ARTIFACTS, now_utc
from theory.quorum_bound import m_probabilistic, verification_table

METRICS = ["blind_min_per_h", "deceived_min_per_h", "fa_per_h", "disruption_cost_per_h",
           "det_rate", "det_latency_s", "frame_tpr", "frame_fpr"]


def _ms(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return {"mean": float(np.mean(vals)) if vals else None, "std": float(np.std(vals)) if vals else None}


def _agg(seed_rows: dict[int, list[dict]], filt=lambda r: True):
    out = {}
    for m in METRICS:
        per_seed = []
        for rows in seed_rows.values():
            rr = [r[m] for r in rows if filt(r) and r[m] is not None and not np.isnan(r[m])]
            if rr:
                per_seed.append(float(np.mean(rr)))
        out[m] = _ms(per_seed)
    return out


def main():
    runs = sorted((ARTIFACTS / "runs").glob("seed_*.json"))
    seeds = {int(p.stem.split("_")[1]): json.loads(p.read_text(encoding="utf-8")) for p in runs}
    methods = list(next(iter(seeds.values()))["methods"].keys())
    bench = {"seeds": sorted(seeds), "methods": {}, "ablations": {}, "pareto": {}, "spoof": {}}
    attacks = sorted({r["attack_label"] for s in seeds.values() for r in s["methods"][methods[0]]["rows"]})
    kns = sorted({(r["n"], r["k"]) for s in seeds.values() for r in s["methods"][methods[0]]["rows"]},
                 key=lambda x: x[1] / x[0])
    for m in methods:
        rows = {sd: s["methods"][m]["rows"] for sd, s in seeds.items()}
        bench["methods"][m] = {
            "overall": _agg(rows),
            "frame_auroc": _ms([s["methods"][m]["frame_auroc"] for s in seeds.values()]),
            "theta": [s["methods"][m]["theta"] for s in seeds.values()],
            "by_attack": {a: _agg(rows, lambda r, a=a: r["attack_label"] == a) for a in attacks},
            "by_kn": {f"{k}/{n}": {"kn": k / n, **_agg(rows, lambda r, n=n, k=k: r["n"] == n and r["k"] == k)}
                      for n, k in kns},
            "episode_blind_min_per_h": {str(sd): [r["blind_min_per_h"] for r in rr] for sd, rr in rows.items()},
        }
        th = sorted({p["theta"] for s in seeds.values() for p in s["pareto"][m]})
        bench["pareto"][m] = [{"theta": t,
                               **{k: _ms([p[k] for s in seeds.values() for p in s["pareto"][m] if p["theta"] == t])
                                  for k in ("blind_min_per_h", "disruption_cost_per_h", "deceived_min_per_h")}}
                              for t in th]
        bench["pareto_per_seed"] = bench.get("pareto_per_seed", {})
        bench["pareto_per_seed"][m] = {str(sd): s["pareto"][m] for sd, s in seeds.items()}
    for a in next(iter(seeds.values()))["ablations"]:
        rows = {sd: s["ablations"][a]["rows"] for sd, s in seeds.items()}
        bench["ablations"][a] = {"overall": _agg(rows),
                                 "frame_auroc": _ms([s["ablations"][a]["frame_auroc"] for s in seeds.values()])}
    bench["spoof"] = {k: _ms([s["spoof"][k] for s in seeds.values()]) for k in ("max_score_shift", "mean_max_score_shift")}
    bench["spoof"]["beta"] = next(iter(seeds.values()))["spoof"]["logit_bound_beta"]
    bench["attacks"] = attacks
    bench["kn_configs"] = [f"{k}/{n}" for n, k in kns]
    pool = [r["det_latency_s"] for s in seeds.values() for r in s["methods"]["scspv_attestor"]["rows"]
            if r["det_rate"] > 0]
    summary = {"generated_utc": now_utc(), "benchmark": bench, "detection_latency_pool_s": pool}
    for name in ("gate0", "phase_boundary", "latency", "airport_case_study"):
        p = ARTIFACTS / f"{name}.json"
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            if name == "latency":
                d = {k: ({kk: vv for kk, vv in v.items() if kk != "samples"} if isinstance(v, dict) else v)
                     for k, v in d.items()}
            if name == "airport_case_study":
                d = {k: v for k, v in d.items() if k != "examples"}
            summary[name] = d
    tab = verification_table()
    summary["theory"] = {
        "cases_checked": len(tab),
        "mismatches": sum(t["detect"] != t["detect_theory"] or t["mask"] != t["mask_theory"] for t in tab),
    }
    if "phase_boundary" in summary:
        pw = summary["phase_boundary"]["witness"]
        summary["theory"]["m_probabilistic_delta0.05"] = {
            mode: {str(k): m_probabilistic(k, pw[mode], 0.05, off_plane=(mode == "offplane")) for k in (1, 2, 3)}
            for mode in ("ground", "offplane")}
    prov = json.loads((ARTIFACTS / "data_provenance.json").read_text(encoding="utf-8"))
    summary["provenance_sources"] = sorted(prov["sources"])
    (ARTIFACTS / "results_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("wrote results_summary.json;", {m: round(bench["methods"][m]["overall"]["blind_min_per_h"]["mean"], 3)
                                          for m in methods})


if __name__ == "__main__":
    main()
