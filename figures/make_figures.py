"""Publication figures (400 DPI, white background, Times New Roman, Okabe-Ito / Viridis).

Every number plotted is read from artifacts/*.json produced by the pipeline.
Identity is never colour-only: methods also differ by marker / hatch and are
direct-labelled.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Polygon  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
OUT = ROOT / "figures"
DPI = 400

OKABE = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "blue": "#0072B2",
         "vermilion": "#D55E00", "purple": "#CC79A7", "yellow": "#F0E442", "black": "#000000"}
METHOD_STYLE = {
    "single_camera": ("Single-camera analytics", OKABE["orange"], "s", "//"),
    "majority_vote": ("Naive majority vote", OKABE["sky"], "^", "\\\\"),
    "static_redundancy": ("Static redundancy", OKABE["green"], "D", ".."),
    "scspv_attestor": ("SCSPV attestor + MILP (ours)", OKABE["vermilion"], "o", ""),
}
INK, MUTED, GRID = "#1a1a1a", "#555555", "#e6e6e6"


def setup():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "figure.facecolor": "#FFFFFF", "axes.facecolor": "#FFFFFF", "savefig.facecolor": "#FFFFFF",
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0,
        "savefig.dpi": DPI, "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
    })


def load(name):
    return json.loads((ART / name).read_text(encoding="utf-8"))


def save(fig, name):
    fig.savefig(OUT / name, dpi=DPI, facecolor="#FFFFFF")
    plt.close(fig)


# ---------------------------------------------------------------------------- Fig 1
def fig1():
    from experiments.scene import load_scene, visibility_graph
    sc = load_scene()
    A = visibility_graph(sc)
    S = load("results_summary.json")
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 54)
    ax.axis("off")
    lanes = [("Airside /\nterminal camera\nvisibility graph", 37, "#f4f8fb"),
             ("Untrusted\nADS-B / Remote-ID\nweak-prior layer", 19, "#fbf7ef"),
             ("ST-GNN epipolar\nattestation &\nMILP hand-off", 1, "#f3faf6")]
    for title, y0, col in lanes:
        ax.add_patch(FancyBboxPatch((0.5, y0), 99, 16, boxstyle="round,pad=0,rounding_size=1.2",
                                    fc=col, ec="#c9c9c9", lw=0.8))
        ax.text(2, y0 + 8, title, va="center", ha="left", fontsize=7.6, color=INK, fontweight="bold",
                linespacing=1.15)

    def box(x, y, w, h, text, ec):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8",
                                    fc="#FFFFFF", ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.2, color=INK)

    def arrow(p, q, col=MUTED, style="-|>", ls="-"):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=9, color=col, lw=1.1, ls=ls))

    # lane 1: real camera graph (top-down WILDTRACK camera centres)
    cen = np.array([c[:2] for c in sc.centers])
    cen = (cen - cen.min(0)) / np.ptp(cen, 0)
    gx, gy = 24 + cen[:, 0] * 14, 41.5 + cen[:, 1] * 9
    for i in range(7):
        for j in range(i + 1, 7):
            if A[i, j] > 0:
                ax.plot([gx[i], gx[j]], [gy[i], gy[j]], color="#9ab", lw=0.25 + 1.6 * A[i, j] / A.max(), zorder=2)
    ax.scatter(gx, gy, s=34, c=OKABE["blue"], edgecolor="white", lw=0.8, zorder=3)
    g0 = S["gate0"]["checks"]
    ax.text(31, 38.0, f"WILDTRACK: n = 7, $\\Delta$ = {g0['G0.4_fov_overlap_graph']['max_degree']}",
            ha="center", fontsize=6.8, color=MUTED)
    box(43, 40, 17, 10, "RTSP streams\nframe-rate / jitter\ntelemetry", OKABE["blue"])
    box(63, 40, 16, 10, "Detections\n(foot & head\nkeypoints)", OKABE["blue"])
    box(82, 40, 16, 10, "Gate 0: homography\n" + f"{g0['G0.3_cross_view_transfer_cm']['median_cm']:.1f} cm median\ntransfer error", OKABE["blue"])
    arrow((38.5, 45), (43, 45))
    arrow((60, 45), (63, 45))
    arrow((79, 45), (82, 45))
    # lane 2
    box(24, 21.5, 20, 11, "OpenSky ADS-B states\n(real, JFK / ATL / DCA)", OKABE["orange"])
    box(48, 21.5, 20, 11, "OSM aeroway zones\n(runway / apron /\nterminal polygons)", OKABE["orange"])
    box(72, 21.5, 26, 11, "Bounded prior gate\n$|\\Delta\\mathrm{logit}|\\leq\\beta$,  $w_z\\in[1,1+\\beta]$\nnever ground truth", OKABE["orange"])
    arrow((44, 27), (48, 27))
    arrow((68, 27), (72, 27))
    # lane 3
    box(24, 3.5, 17, 11, "Epipolar residuals\n$p^{\\top}E\\,p'=0$\n+ homography transfer", OKABE["green"])
    box(44, 3.5, 16, 11, "ST-GNN\ncausal GRU +\nedge-conditioned MP", OKABE["green"])
    box(63, 3.5, 16, 11, "Quorum check\n$m(n,k)$ bound\n(compression\ncontradiction)", OKABE["green"])
    box(82, 3.5, 16, 11, "MILP hand-off\n(SCSPV availability\nweights, budget $B$)", OKABE["vermilion"])
    arrow((41, 9), (44, 9))
    arrow((60, 9), (63, 9))
    arrow((79, 9), (82, 9))
    # detections -> epipolar: down through the gap between lane-2 boxes, then along the lane boundary
    ax.plot([70, 70, 32.5], [40, 18, 18], color=OKABE["blue"], lw=1.1, zorder=4)
    arrow((32.5, 18.05), (32.5, 14.6), col=OKABE["blue"])
    arrow((46, 40), (46, 14.6), col=OKABE["blue"])          # telemetry -> GNN (through lane-2 gap)
    arrow((85, 21.5), (56, 14.6), col=OKABE["orange"], ls="--")   # prior -> GNN (dashed: untrusted)
    arrow((90, 21.5), (90, 14.6), col=OKABE["orange"], ls="--")   # prior -> MILP weights
    ax.annotate("", xy=(98.8, 45), xytext=(98.8, 9),
                arrowprops=dict(arrowstyle="-|>", color=OKABE["vermilion"], lw=1.1, ls=":"))
    ax.text(99.6, 27, "re-allocate\nstreams", rotation=90, va="center", ha="left", fontsize=6.5,
            color=OKABE["vermilion"])
    save(fig, "fig1_airport_architecture.png")


# ---------------------------------------------------------------------------- Fig 2
def fig2():
    from theory.quorum_bound import m_mask
    pb = load("phase_boundary.json")
    fig = plt.figure(figsize=(7.2, 2.9))
    ax3 = fig.add_subplot(1, 3, 1, projection="3d")
    n = np.arange(5, 41)
    kn = np.linspace(0.05, 0.48, 44)
    Nn, KK = np.meshgrid(n, kn)
    k = np.maximum(1, np.floor(KK * Nn)).astype(int)
    M = np.vectorize(lambda kk: m_mask(kk))(k).astype(float)
    M[M > Nn] = np.nan
    surf = ax3.plot_surface(KK, Nn, M, cmap="viridis", linewidth=0, antialiased=True, alpha=0.95)
    ax3.set_xlabel("$k/n$", labelpad=-2)
    ax3.set_ylabel("$n$ cameras", labelpad=-2)
    ax3.set_zlabel("$m_{\\mathrm{mask}}(n,k)$", labelpad=-4)
    ax3.tick_params(pad=-2, labelsize=6.5)
    ax3.view_init(elev=24, azim=-128)
    ax3.set_title("(a) Analytical bound $2k{+}1$", pad=0)
    ax3.xaxis.pane.set_facecolor("#FFFFFF")
    ax3.yaxis.pane.set_facecolor("#FFFFFF")
    ax3.zaxis.pane.set_facecolor("#FFFFFF")

    for idx, mode, title, off in ((2, "ground", "(b) Ground plane (empirical)", 0),
                                  (3, "offplane", "(c) Off-plane (epipolar)", 1)):
        ax = fig.add_subplot(1, 3, idx)
        cells = pb[mode]
        rs = sorted({v["r"] for v in cells.values()})
        grid = np.full((3, max(rs)), np.nan)
        for v in cells.values():
            if v["n"] >= 10:
                grid[v["k"] - 1, v["r"] - 1] = v["detect_rate"]
        im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1,
                       extent=(0.5, max(rs) + 0.5, 0.5, 3.5))
        kk = np.array([0.5, 1.5, 1.5, 2.5, 2.5, 3.5])
        kv = np.array([1, 1, 2, 2, 3, 3])
        ax.plot(kv + 0.5 + off, kk, color="white", lw=2.0, ls="-")
        ax.plot(kv + 0.5 + off, kk, color=OKABE["vermilion"], lw=1.2, ls="-", label=f"detect: $r=k+{1 + off}$")
        ax.plot(2 * kv + 0.5 + off, kk, color="white", lw=2.0)
        ax.plot(2 * kv + 0.5 + off, kk, color=OKABE["yellow"], lw=1.2, ls="--", label=f"mask: $r=2k+{1 + off}$")
        for v in cells.values():
            if v["n"] >= 10:
                ax.text(v["r"], v["k"], f"{v['detect_rate']:.2f}", ha="center", va="center", fontsize=5.6,
                        color="white" if v["detect_rate"] < 0.6 else INK)
        ax.set_xticks(range(1, max(rs) + 1))
        ax.set_yticks([1, 2, 3])
        ax.set_xlabel("zone redundancy $r$")
        ax.set_ylabel("compromised witnesses $k$")
        ax.set_title(title)
        ax.grid(False)
        ax.legend(loc="upper left", fontsize=6, framealpha=0.9, handlelength=1.6, borderpad=0.3)
    cax = fig.add_axes([0.93, 0.2, 0.012, 0.6])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("detection rate", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    fig.subplots_adjust(left=0.0, right=0.91, wspace=0.42, bottom=0.17, top=0.88)
    save(fig, "fig2_byzantine_quorum_bound.png")


# ---------------------------------------------------------------------------- Fig 3
def fig3():
    S = load("results_summary.json")["benchmark"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.15, 1.0, 1.0]})
    ax = axes[0]
    for m, (lab, col, mk, _) in METHOD_STYLE.items():
        for sd, pts in S["pareto_per_seed"][m].items():
            x = [p["disruption_cost_per_h"] for p in pts]
            y = [p["blind_min_per_h"] for p in pts]
            o = np.argsort(x)
            ax.plot(np.array(x)[o], np.array(y)[o], color=col, lw=0.6, alpha=0.35)
        P = S["pareto"][m]
        x = np.array([p["disruption_cost_per_h"]["mean"] for p in P])
        y = np.array([p["blind_min_per_h"]["mean"] for p in P])
        o = np.argsort(x)
        ax.plot(x[o], y[o], color=col, lw=2, marker=mk, ms=4.5, mec="white", mew=0.6, label=lab)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("false-alarm disruption cost (per h)")
    ax.set_ylabel("coverage-weighted blind min / h")
    ax.set_title("(a) Pareto frontier (5 seeds)")
    ax.legend(loc="upper right", fontsize=6.2, framealpha=0.92)

    ax = axes[1]
    data, cols, labs = [], [], []
    for m, (lab, col, mk, hatch) in METHOD_STYLE.items():
        vals = np.concatenate([np.array(v) for v in S["methods"][m]["episode_blind_min_per_h"].values()])
        data.append(vals)
        cols.append(col)
        labs.append(lab.split(" (")[0].replace("SCSPV attestor + MILP", "Ours"))
    parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False, widths=0.8)
    for pc, col, (_, (_, _, _, hatch)) in zip(parts["bodies"], cols, METHOD_STYLE.items()):
        pc.set_facecolor(col)
        pc.set_edgecolor(INK)
        pc.set_linewidth(0.5)
        pc.set_alpha(0.75)
        pc.set_hatch(hatch)
    for i, d in enumerate(data, 1):
        q1, med, q3 = np.percentile(d, [25, 50, 75])
        ax.vlines(i, q1, q3, color=INK, lw=2.2)
        ax.scatter([i], [np.mean(d)], s=16, color="white", edgecolor=INK, zorder=3, lw=0.8)
    ax.set_xticks(range(1, 5))
    ax.set_xticklabels(["Single\ncamera", "Majority\nvote", "Static\nredund.", "Ours"], fontsize=7)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("blind min / h (per episode)")
    ax.set_title("(b) Episode distribution")

    ax = axes[2]
    for m, (lab, col, mk, _) in METHOD_STYLE.items():
        bk = S["methods"][m]["by_kn"]
        items = sorted(bk.values(), key=lambda v: v["kn"])
        x = [v["kn"] for v in items]
        y = np.array([v["blind_min_per_h"]["mean"] for v in items])
        e = np.array([v["blind_min_per_h"]["std"] for v in items])
        ax.plot(x, y, color=col, marker=mk, ms=4, mec="white", mew=0.5, lw=1.4)
        ax.fill_between(x, y - e, y + e, color=col, alpha=0.12, lw=0)
    ax.set_xlabel("compromised fraction $k/n$")
    ax.set_ylabel("blind min / h")
    ax.set_title("(c) Blind time vs. $k/n$")
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig3_blind_time_pareto.png")


# ---------------------------------------------------------------------------- Fig 4
def fig4():
    from experiments.adsb_prior import zone_polygons
    cs = load("airport_case_study.json")
    g = np.load(ART / "airport_case_study_geometry.npz", allow_pickle=False)
    cells, w, cams, cover, T = g["cells"], g["w"], g["cams"], g["cover"], g["terminal"]
    byz = cs["byzantine"]
    sel_none = np.array(cs["examples"]["none"]["selected"])
    sel_milp = np.array(cs["examples"]["milp"]["selected"])
    lo, hi = cells.min(0) - 60, cells.max(0) + 60
    zs = zone_polygons("ATL")
    fig = plt.figure(figsize=(7.2, 3.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.42], width_ratios=[1, 1, 0.05], hspace=0.42, wspace=0.12)
    t_after = -1
    for col, (sel, title) in enumerate([(sel_none[0], "(a) Pre-attack MILP allocation"),
                                        (sel_milp[t_after], "(b) After detection: MILP hand-off")]):
        ax = fig.add_subplot(gs[0, col])
        for z in zs:
            xy = z["xy"]
            if not ((xy[:, 0] > lo[0] - 300).any() and (xy[:, 0] < hi[0] + 300).any()
                    and (xy[:, 1] > lo[1] - 300).any() and (xy[:, 1] < hi[1] + 300).any()):
                continue
            if z["kind"] in ("apron", "terminal", "hangar") and z["closed"]:
                ax.add_patch(Polygon(xy, closed=True, fc="#eeeeee" if z["kind"] != "terminal" else "#d9d9d9",
                                     ec="#bdbdbd", lw=0.4, zorder=1))
            elif z["kind"] in ("taxiway", "runway"):
                ax.plot(xy[:, 0], xy[:, 1], color="#c8c8c8", lw=0.5 if z["kind"] == "taxiway" else 2.0, zorder=1)
        trusted = sel.astype(bool).copy()
        if col == 1:
            trusted[byz] = False
        covered = cover[trusted].any(0)
        risk = w * (~covered)
        sc_ = ax.scatter(cells[:, 0], cells[:, 1], c=np.where(covered, w, np.nan), cmap="viridis", s=7,
                         marker="s", lw=0, zorder=2, vmin=0, vmax=w.max())
        ax.scatter(cells[~covered, 0], cells[~covered, 1], s=7, marker="s", facecolor="none",
                   edgecolor=OKABE["vermilion"], lw=0.5, zorder=2)
        C = len(cams)
        for i in range(C):
            for j in range(i + 1, C):
                if (cover[i] & cover[j]).any():
                    ax.plot(cams[[i, j], 0], cams[[i, j], 1], color="#7a8a99", lw=0.3, alpha=0.5, zorder=3)
        for c in range(C):
            is_b = c in byz and col == 1
            mk = "X" if is_b else ("o" if sel[c] else "o")
            fc = OKABE["vermilion"] if is_b else (OKABE["blue"] if sel[c] else "white")
            ax.scatter(cams[c, 0], cams[c, 1], s=30 if sel[c] or is_b else 16, marker=mk, c=fc,
                       edgecolor=INK, lw=0.6, zorder=5)
        if col == 1:
            new = np.where(sel.astype(bool) & ~sel_none[0].astype(bool))[0]
            for c in new:
                ax.scatter(cams[c, 0], cams[c, 1], s=80, marker="o", facecolor="none", edgecolor=OKABE["green"],
                           lw=1.4, zorder=6)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color("#c9c9c9")
        ax.set_title(title, fontsize=8.5)
        ax.plot([hi[0] - 230, hi[0] - 30], [lo[1] + 40] * 2, color=INK, lw=1.4)
        ax.text(hi[0] - 130, lo[1] + 55, "200 m", ha="center", fontsize=6.5)
    cax = fig.add_subplot(gs[0, 2])
    cb = fig.colorbar(sc_, cax=cax)
    cb.set_label("zone criticality $w_z$", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    leg = [Line2D([], [], marker="o", ls="", mfc=OKABE["blue"], mec=INK, label="streamed camera"),
           Line2D([], [], marker="o", ls="", mfc="white", mec=INK, label="idle camera"),
           Line2D([], [], marker="X", ls="", mfc=OKABE["vermilion"], mec=INK, label="compromised (quarantined)"),
           Line2D([], [], marker="o", ls="", mfc="none", mec=OKABE["green"], mew=1.4, label="hand-off target"),
           Patch(fc="none", ec=OKABE["vermilion"], label="blind zone")]
    fig.legend(handles=leg, loc="upper center", ncol=5, fontsize=6.4, frameon=False, bbox_to_anchor=(0.47, 1.0))
    ax = fig.add_subplot(gs[1, :2])
    styles = {"none": ("no hand-off", OKABE["orange"], "--"), "greedy": ("greedy hand-off", OKABE["sky"], "-."),
              "milp": ("MILP hand-off (ours)", OKABE["vermilion"], "-")}
    for p, (lab, col, ls) in styles.items():
        s = np.array(cs["examples"][p]["series"])
        t = np.arange(len(s)) * 2
        ax.plot(t, s, color=col, ls=ls, lw=1.6,
                label=f"{lab}: {cs['policies'][p]['blind_min_mean']:.2f} blind min")
    ax.axvline(60, color=MUTED, lw=0.8, ls=":")
    ax.text(63, 0.15, "attack onset", fontsize=6.5, color=MUTED)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("blind weight")
    ax.set_title(f"(c) ATL Concourse F, k = {cs['k']} of {cs['n_cameras']} cameras, budget B = {cs['budget']}",
                 fontsize=8.5)
    ax.legend(fontsize=6.4, loc="upper right", framealpha=0.9)
    save(fig, "fig4_zone_criticality_heatmap.png")


# ---------------------------------------------------------------------------- Fig 5
def fig5():
    S = load("results_summary.json")
    L = load("latency.json")
    ab = S["benchmark"]["ablations"]
    names = [("full", "Full model"), ("no_epipolar", "$-$ epipolar"), ("no_telemetry", "$-$ jitter/telemetry"),
             ("no_adsb", "$-$ ADS-B prior"), ("greedy_handoff", "greedy hand-off"), ("no_handoff", "no hand-off")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    cols = [OKABE["vermilion"], OKABE["blue"], OKABE["sky"], OKABE["orange"], OKABE["green"], OKABE["purple"]]
    hatches = ["", "//", "\\\\", "..", "xx", "--"]
    for ai, (metric, ylab, title) in enumerate([("frame_auroc", "frame AUROC", "(a) Detection ablation"),
                                                ("blind_min_per_h", "blind min / h", "(b) Availability ablation")]):
        ax = axes[ai]
        sub = names[:4] if metric == "frame_auroc" else names
        for i, (k, lab) in enumerate(sub):
            v = ab[k]["frame_auroc"] if metric == "frame_auroc" else ab[k]["overall"][metric]
            ax.bar(i, v["mean"], yerr=v["std"], color=cols[i], hatch=hatches[i], edgecolor="white", lw=0.6,
                   width=0.72, capsize=2.5, error_kw={"lw": 0.9, "ecolor": INK})
            ax.text(i, v["mean"] + (v["std"] or 0) + (0.01 if metric == "frame_auroc" else 0.03),
                    f"{v['mean']:.3f}" if metric == "frame_auroc" else f"{v['mean']:.2f}",
                    ha="center", va="bottom", fontsize=6)
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels([lab for _, lab in sub], rotation=35, ha="right", fontsize=6.8)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        if metric == "frame_auroc":
            lo = min(ab[k]["frame_auroc"]["mean"] - ab[k]["frame_auroc"]["std"] for k, _ in sub)
            ax.set_ylim(max(0.0, lo - 0.08), 1.0)
    ax = axes[2]
    comps = [("gnn_ms", "ST-GNN step (all cams)", OKABE["vermilion"], "-"),
             ("geometry_ms", "epipolar/homography checks", OKABE["blue"], "--"),
             ("attestor_ms", "attestor total (geom.+GNN)", OKABE["black"], "-"),
             ("milp_ms", "MILP solve (uncached)", OKABE["green"], "-."),
             ("flow_ms_per_camera", "optical flow (per camera)", OKABE["orange"], ":")]
    for k, lab, col, ls in comps:
        s = np.sort(np.array(L[k]["samples"]))
        ax.plot(s, np.arange(1, len(s) + 1) / len(s), color=col, ls=ls, lw=1.6,
                label=f"{lab} (p95 {L[k]['p95']:.1f} ms)")
    ax.axvline(15, color=MUTED, lw=0.9, ls=":")
    ax.text(16.5, 0.62, "15 ms", fontsize=6.5, color=MUTED)
    ax.set_xscale("log")
    ax.set_xlabel("latency per frame (ms, CPU, 1 thread)")
    ax.set_ylabel("CDF")
    ax.set_title("(c) Latency CDF")
    ax.legend(fontsize=5.6, loc="lower right", framealpha=0.92)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig5_ablation_latency.png")


def main():
    setup()
    for f in (fig1, fig2, fig3, fig4, fig5):
        f()
        print("ok", f.__name__)


if __name__ == "__main__":
    main()
