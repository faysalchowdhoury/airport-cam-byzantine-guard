"""Generate LaTeX tables and numeric macros from artifacts/results_summary.json.

The manuscript never hard-codes a result: every number comes through
generated/macros.tex or generated/*.tex produced here.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ART = HERE.parent / "artifacts"
GEN = HERE / "generated"
GEN.mkdir(exist_ok=True)
S = json.loads((ART / "results_summary.json").read_text(encoding="utf-8"))
B = S["benchmark"]
NAMES = {"single_camera": "Single-camera analytics", "majority_vote": "Naive majority vote",
         "static_redundancy": "Static redundancy", "scspv_attestor": "\\textbf{SCSPV attestor + MILP (ours)}"}
ATT = {"replay": "Replay", "deauth": "Deauth", "saturation": "RTSP sat.", "fabrication": "Fabrication",
       "mixed": "Mixed"}
ABL = {"full": "Full model", "no_epipolar": "$-$ epipolar residuals", "no_telemetry": "$-$ frame-jitter telemetry",
       "no_adsb": "$-$ ADS-B/Remote-ID prior", "greedy_handoff": "greedy hand-off (no MILP)",
       "no_handoff": "no hand-off (static set)"}


def pm(d, f="{:.2f}"):
    if d is None or d.get("mean") is None:
        return "--"
    return f"{f.format(d['mean'])}\\,$\\pm$\\,{f.format(d['std'])}"


def macro(name, val):
    return f"\\newcommand{{\\{name}}}{{{val}}}\n"


def best(method_vals, lower=True):
    vals = {m: v for m, v in method_vals.items() if v is not None}
    return (min if lower else max)(vals, key=vals.get)


def main():
    M = B["methods"]
    # -------- main table
    cols = [("blind_min_per_h", True, "{:.2f}"), ("deceived_min_per_h", True, "{:.2f}"),
            ("fa_per_h", True, "{:.1f}"), ("det_rate", False, "{:.2f}"), ("det_latency_s", True, "{:.1f}")]
    bests = {c: best({m: M[m]["overall"][c]["mean"] for m in M}, low) for c, low, _ in cols}
    bauc = best({m: M[m]["frame_auroc"]["mean"] for m in M}, False)
    lines = []
    for m in M:
        cells = []
        for c, low, f in cols:
            s = pm(M[m]["overall"][c], f)
            cells.append(f"\\textbf{{{s}}}" if bests[c] == m else s)
        a = pm(M[m]["frame_auroc"], "{:.3f}")
        cells.append(f"\\textbf{{{a}}}" if bauc == m else a)
        lines.append(NAMES[m] + " & " + " & ".join(cells) + " \\\\")
    (GEN / "table_main.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # -------- by attack (blind min/h and detection rate)
    lines = []
    for a in B["attacks"]:
        row = [ATT.get(a, a)]
        bb = best({m: M[m]["by_attack"][a]["blind_min_per_h"]["mean"] for m in M})
        for m in M:
            s = pm(M[m]["by_attack"][a]["blind_min_per_h"])
            row.append(f"\\textbf{{{s}}}" if bb == m else s)
        for m in M:
            row.append(f"{M[m]['by_attack'][a]['det_rate']['mean']:.2f}")
        lines.append(" & ".join(row) + " \\\\")
    (GEN / "table_attack.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # -------- by k/n
    lines = []
    for key in B["kn_configs"]:
        kn = M["scspv_attestor"]["by_kn"][key]["kn"]
        row = [f"{key} ({kn:.2f})"]
        bb = best({m: M[m]["by_kn"][key]["blind_min_per_h"]["mean"] for m in M})
        for m in M:
            s = pm(M[m]["by_kn"][key]["blind_min_per_h"])
            row.append(f"\\textbf{{{s}}}" if bb == m else s)
        lines.append(" & ".join(row) + " \\\\")
    (GEN / "table_kn.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # -------- ablation
    lines = []
    for a, lab in ABL.items():
        v = B["ablations"][a]
        lines.append(f"{lab} & {pm(v['frame_auroc'], '{:.3f}')} & {pm(v['overall']['blind_min_per_h'])} & "
                     f"{pm(v['overall']['deceived_min_per_h'])} & {pm(v['overall']['fa_per_h'], '{:.1f}')} \\\\")
    (GEN / "table_ablation.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # -------- phase boundary agreement
    pb = S["phase_boundary"]
    lines = []
    for mode, lab in (("ground", "Ground plane"), ("offplane", "Off-plane (epipolar)")):
        ag = pb[f"{mode}_agreement"]
        mp = S["theory"]["m_probabilistic_delta0.05"][mode]
        lines.append(f"{lab} & {ag['cells']} & {ag['detect']:.2f} & {ag['mask']:.2f} & {pb['witness'][mode]:.3f} & "
                     f"{mp['1']}, {mp['2']}, {mp['3']} \\\\")
    (GEN / "table_phase.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # -------- macros
    g0 = S["gate0"]["checks"]
    L = S["latency"]
    cs = S["airport_case_study"]
    ours, maj = M["scspv_attestor"]["overall"], M["majority_vote"]["overall"]
    sc, st = M["single_camera"]["overall"], M["static_redundancy"]["overall"]
    base_best_blind = min(sc["blind_min_per_h"]["mean"], maj["blind_min_per_h"]["mean"], st["blind_min_per_h"]["mean"])
    mac = ""
    mac += macro("GzeroReprojMax", f"{max(g0['G0.2_label_reprojection_px']['median_px'].values()):.1f}")
    mac += macro("GzeroReprojMin", f"{min(g0['G0.2_label_reprojection_px']['median_px'].values()):.1f}")
    mac += macro("GzeroTransferMed", f"{g0['G0.3_cross_view_transfer_cm']['median_cm']:.1f}")
    mac += macro("GzeroTransferPninety", f"{g0['G0.3_cross_view_transfer_cm']['p90_cm']:.1f}")
    mac += macro("GzeroPairs", f"{g0['G0.3_cross_view_transfer_cm']['n_pairs']:,}".replace(",", "{,}"))
    mac += macro("GzeroBoxes", f"{sum(g0['G0.2_label_reprojection_px']['n_boxes'].values()):,}".replace(",", "{,}"))
    mac += macro("GzeroMaxDegree", str(g0["G0.4_fov_overlap_graph"]["max_degree"]))
    mac += macro("TheoryCases", str(S["theory"]["cases_checked"]))
    mac += macro("TheoryMismatch", str(S["theory"]["mismatches"]))
    mac += macro("WitnessGround", f"{pb['witness']['ground']:.3f}")
    mac += macro("WitnessOff", f"{pb['witness']['offplane']:.3f}")
    mac += macro("AgreeGroundDet", f"{pb['ground_agreement']['detect']:.2f}")
    mac += macro("AgreeOffDet", f"{pb['offplane_agreement']['detect']:.2f}")
    mac += macro("OursBlind", f"{ours['blind_min_per_h']['mean']:.2f}")
    mac += macro("OursBlindStd", f"{ours['blind_min_per_h']['std']:.2f}")
    mac += macro("BestBaseBlind", f"{base_best_blind:.2f}")
    mac += macro("BlindReductionPct", f"{100 * (1 - ours['blind_min_per_h']['mean'] / base_best_blind):.0f}")
    mac += macro("OursFA", f"{ours['fa_per_h']['mean']:.1f}")
    mac += macro("MajFA", f"{maj['fa_per_h']['mean']:.1f}")
    mac += macro("SingleFA", f"{sc['fa_per_h']['mean']:.1f}")
    mac += macro("OursFAStd", f"{ours['fa_per_h']['std']:.1f}")
    mac += macro("StaticFA", f"{st['fa_per_h']['mean']:.1f}")
    mac += macro("StaticBlind", f"{st['blind_min_per_h']['mean']:.2f}")
    mac += macro("SingleBlind", f"{sc['blind_min_per_h']['mean']:.2f}")
    mac += macro("MajBlind", f"{maj['blind_min_per_h']['mean']:.2f}")
    mac += macro("StaticAUROC", f"{M['static_redundancy']['frame_auroc']['mean']:.3f}")
    ths = M["scspv_attestor"]["theta"]
    mac += macro("OursThetaRange", f"{min(ths):.1f}" if min(ths) == max(ths) else f"{min(ths):.1f}--{max(ths):.1f}")
    mac += macro("OursDeceived", f"{ours['deceived_min_per_h']['mean']:.2f}")
    mac += macro("MajDeceived", f"{maj['deceived_min_per_h']['mean']:.2f}")
    mac += macro("OursDetRateFab", f"{M['scspv_attestor']['by_attack'].get('fabrication', {}).get('det_rate', {}).get('mean', float('nan')):.2f}")
    mac += macro("OursDetRateReplay", f"{M['scspv_attestor']['by_attack'].get('replay', {}).get('det_rate', {}).get('mean', float('nan')):.2f}")
    mac += macro("SpoofBound", f"{2 * B['spoof']['beta']:.1f}")
    mac += macro("OursAUROC", f"{M['scspv_attestor']['frame_auroc']['mean']:.3f}")
    mac += macro("SingleAUROC", f"{M['single_camera']['frame_auroc']['mean']:.3f}")
    mac += macro("MajAUROC", f"{M['majority_vote']['frame_auroc']['mean']:.3f}")
    mac += macro("OursDetRate", f"{ours['det_rate']['mean']:.2f}")
    mac += macro("OursDetLat", f"{ours['det_latency_s']['mean']:.1f}")
    mac += macro("SpoofShift", f"{B['spoof']['max_score_shift']['mean']:.3f}")
    mac += macro("SpoofBeta", f"{B['spoof']['beta']:.1f}")
    for k in ("gnn_ms", "geometry_ms", "attestor_ms", "milp_ms", "flow_ms_per_camera"):
        nm = {"gnn_ms": "Gnn", "geometry_ms": "Geom", "attestor_ms": "Attest", "milp_ms": "Milp",
              "flow_ms_per_camera": "Flow"}[k]
        mac += macro(f"Lat{nm}Pfifty", f"{L[k]['p50']:.2f}")
        mac += macro(f"Lat{nm}Pninetyfive", f"{L[k]['p95']:.2f}")
    mac += macro("GnnParams", f"{L['n_params']:,}".replace(",", "{,}"))
    mac += macro("CaseCams", str(cs["n_cameras"]))
    mac += macro("CaseBudget", str(cs["budget"]))
    mac += macro("CaseCells", str(cs["n_cells"]))
    mac += macro("CaseK", str(cs["k"]))
    mac += macro("CaseDelta", str(cs["max_degree"]))
    mac += macro("CaseKmax", str(cs["max_tolerable_k_mask_ground"]))
    for p in ("none", "greedy", "milp"):
        mac += macro(f"Case{p.capitalize()}Blind", f"{cs['policies'][p]['blind_min_mean']:.2f}")
    mac += macro("CaseSpoofLone", f"{cs['weight_shift_spoof_l1']:.3f}")
    mac += macro("CaseAdsbCells", str(cs["adsb_cells_active"]))
    abl = B["ablations"]
    for a, nm in (("full", "Full"), ("no_epipolar", "NoEpi"), ("no_telemetry", "NoTel"), ("no_adsb", "NoAdsb"),
                  ("greedy_handoff", "Greedy"), ("no_handoff", "NoHand")):
        mac += macro(f"Abl{nm}AUROC", f"{abl[a]['frame_auroc']['mean']:.3f}")
        mac += macro(f"Abl{nm}Blind", f"{abl[a]['overall']['blind_min_per_h']['mean']:.2f}")
    (GEN / "macros.tex").write_text(mac, encoding="utf-8")
    print("tables + macros written to", GEN)


if __name__ == "__main__":
    main()
