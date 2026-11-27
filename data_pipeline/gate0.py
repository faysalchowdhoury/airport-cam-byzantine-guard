"""Gate 0: fail-closed feasibility check before any experiment runs.

Checks (all on real WILDTRACK data):
  G0.1  every calibration file parses and yields an invertible ground homography;
  G0.2  calibration <-> label alignment: annotated ground positions re-projected
        into each view land on the annotated box foot point (median error);
  G0.3  cross-camera homography transfer: the same person's foot points from two
        overlapping views map to the same ground location (median error);
  G0.4  overlapping FOV: the visibility graph is connected and every camera has
        at least one overlapping neighbour;
  G0.5  frame payload present for all cameras (when frames were downloaded).
Status ``PASS`` or ``NOT_EVALUABLE``; downstream code refuses to run otherwise.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from .provenance import ARTIFACTS, DATA, now_utc

GATE_FILE = ARTIFACTS / "gate0.json"
REPROJ_MAX_PX = 30.0      # at 1920x1080
TRANSFER_MAX_CM = 60.0


def run(require_frames: bool = True) -> dict:
    from experiments.scene import load_scene, visibility_graph

    report: dict = {"checked_utc": now_utc(), "checks": {}}
    ok = True
    try:
        sc = load_scene()
    except Exception as exc:  # noqa: BLE001
        report.update(status="NOT_EVALUABLE", reason=f"scene load failed: {exc}")
        GATE_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    conds = [float(np.linalg.cond(h)) for h in sc.H]
    g1 = all(np.isfinite(c) and c < 1e12 for c in conds)
    report["checks"]["G0.1_homography_invertible"] = {"pass": g1, "condition_numbers": conds}
    ok &= g1

    from data_pipeline.wildtrack import load_annotations, position_id_to_world
    ann = load_annotations()
    errs = {c: [] for c in range(sc.n_cams)}
    for frame in ann:
        for p in frame:
            X = np.array([position_id_to_world(p["positionID"])])
            for v in p["views"]:
                if v["xmin"] < 0:
                    continue
                uv, _ = sc.project(v["viewNum"], X)
                foot = np.array([0.5 * (v["xmin"] + v["xmax"]), v["ymax"]])
                errs[v["viewNum"]].append(float(np.linalg.norm(uv[0] - foot)))
    med = {sc.cams[c]: float(np.median(e)) for c, e in errs.items()}
    g2 = all(m < REPROJ_MAX_PX for m in med.values())
    report["checks"]["G0.2_label_reprojection_px"] = {"pass": g2, "median_px": med,
                                                     "threshold_px": REPROJ_MAX_PX,
                                                     "n_boxes": {sc.cams[c]: len(e) for c, e in errs.items()}}
    ok &= g2

    tr = []
    for f in range(sc.n_frames):
        for i in range(sc.n_cams):
            for j in range(i + 1, sc.n_cams):
                ids_i, ids_j = sc.det_ids[f][i], sc.det_ids[f][j]
                common, ia, ja = np.intersect1d(ids_i, ids_j, return_indices=True)
                if len(common):
                    d = np.linalg.norm(sc.det[f][i][ia, :2] - sc.det[f][j][ja, :2], axis=1)
                    tr.extend(d.tolist())
    tr = np.array(tr)
    g3 = len(tr) > 1000 and float(np.median(tr)) < TRANSFER_MAX_CM
    report["checks"]["G0.3_cross_view_transfer_cm"] = {
        "pass": bool(g3), "median_cm": float(np.median(tr)), "p90_cm": float(np.percentile(tr, 90)),
        "n_pairs": int(len(tr)), "threshold_cm": TRANSFER_MAX_CM}
    ok &= bool(g3)

    A = visibility_graph(sc)
    reach = np.eye(sc.n_cams, dtype=bool)
    for _ in range(sc.n_cams):
        reach = reach | ((reach.astype(int) @ (A > 0).astype(int)) > 0)
    g4 = bool(reach.all() and ((A > 0).sum(1) > 0).all())
    report["checks"]["G0.4_fov_overlap_graph"] = {
        "pass": g4, "degree": (A > 0).sum(1).tolist(), "max_degree": int((A > 0).sum(1).max()),
        "shared_zones": A.astype(int).tolist(),
        "zone_redundancy_hist": np.bincount(sc.cover.sum(0), minlength=8).tolist()}
    ok &= g4

    frames_dir = DATA / "wildtrack" / "frames"
    present = [(frames_dir / f"C{c + 1}.npy").exists() for c in range(sc.n_cams)]
    g5 = all(present) or not require_frames
    report["checks"]["G0.5_frames_present"] = {"pass": g5, "present": present}
    ok &= g5
    report["status"] = "PASS" if ok else "NOT_EVALUABLE"
    GATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    GATE_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def require_pass() -> None:
    if not GATE_FILE.exists():
        raise SystemExit("NOT_EVALUABLE: Gate 0 has not been run (python -m data_pipeline.gate0)")
    rep = json.loads(GATE_FILE.read_text(encoding="utf-8"))
    if rep.get("status") != "PASS":
        raise SystemExit(f"NOT_EVALUABLE: Gate 0 status={rep.get('status')}")


if __name__ == "__main__":
    rep = run(require_frames="--no-frames" not in sys.argv)
    print(json.dumps({k: (v["pass"] if isinstance(v, dict) else v) for k, v in rep["checks"].items()}, indent=1))
    print("STATUS:", rep["status"])
    sys.exit(0 if rep["status"] == "PASS" else 2)
