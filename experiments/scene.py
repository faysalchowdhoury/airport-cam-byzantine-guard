"""Real multi-camera scene built from WILDTRACK calibration and annotations.

Everything here is derived deterministically from the downloaded dataset:
ground-plane homographies, per-camera detections (annotated boxes), per-cell
camera coverage, the camera visibility graph and passenger-density zone weights.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from data_pipeline.wildtrack import (CAMS, FRAME_IDS, IMG_H, IMG_W, ground_homography,
                                     load_annotations, load_calibration, position_id_to_world)

FPS = 2.0                      # WILDTRACK annotated frame rate
CELL_CM = 100.0                # 1 m x 1 m coverage zones
X_RANGE = (-300.0, 900.0)      # annotated ground area (cm)
Y_RANGE = (-900.0, 2700.0)


@dataclass
class Scene:
    cams: list[str]
    K: list[np.ndarray]
    R: list[np.ndarray]
    t: list[np.ndarray]
    H: list[np.ndarray]                    # ground (cm) -> image px
    Hinv: list[np.ndarray]                 # image px -> ground (cm)
    centers: list[np.ndarray]              # camera centres (world, cm)
    # detections[f][c] -> array (m, 6): ground_x, ground_y, foot_u, foot_v, head_u, head_v; plus ids
    det: list[list[np.ndarray]] = field(default_factory=list)
    det_ids: list[list[np.ndarray]] = field(default_factory=list)
    gt_world: list[np.ndarray] = field(default_factory=list)   # [f] -> (p, 2) cm
    gt_ids: list[np.ndarray] = field(default_factory=list)
    cells: np.ndarray | None = None        # (Z, 2) zone centres (cm)
    cover: np.ndarray | None = None        # (C, Z) bool coverage
    density: np.ndarray | None = None      # (Z,) mean pedestrians per frame
    weights: np.ndarray | None = None      # (Z,) coverage weights, sum 1

    @property
    def n_cams(self) -> int:
        return len(self.cams)

    @property
    def n_frames(self) -> int:
        return len(self.det)

    def project(self, c: int, XY: np.ndarray, z: float = 0.0) -> np.ndarray:
        """World points (cm) at height z -> image px for camera c."""
        XY = np.atleast_2d(XY)
        P = np.c_[XY, np.full(len(XY), z)]
        cam = (self.R[c] @ P.T + self.t[c].reshape(3, 1))
        uv = self.K[c] @ cam
        return (uv[:2] / uv[2]).T, cam[2]

    def to_ground(self, c: int, uv: np.ndarray) -> np.ndarray:
        uv = np.atleast_2d(uv)
        p = self.Hinv[c] @ np.c_[uv, np.ones(len(uv))].T
        return (p[:2] / p[2]).T

    def fundamental(self, i: int, j: int) -> np.ndarray:
        """F_ij with x_j^T F_ij x_i = 0 (built from calibration: E = [t]_x R)."""
        Rij = self.R[j] @ self.R[i].T
        tij = self.t[j] - Rij @ self.t[i]
        tx = np.array([[0, -tij[2], tij[1]], [tij[2], 0, -tij[0]], [-tij[1], tij[0], 0]])
        E = tx @ Rij
        return np.linalg.inv(self.K[j]).T @ E @ np.linalg.inv(self.K[i])


def _in_image(uv: np.ndarray, depth: np.ndarray, margin: float = 0.0) -> np.ndarray:
    return ((depth > 0) & (uv[:, 0] >= margin) & (uv[:, 0] < IMG_W - margin)
            & (uv[:, 1] >= margin) & (uv[:, 1] < IMG_H - margin))


@lru_cache(maxsize=1)
def load_scene() -> Scene:
    cals = [load_calibration(c) for c in CAMS]
    H = [ground_homography(c) for c in cals]
    sc = Scene(cams=list(CAMS), K=[c["K"] for c in cals], R=[c["R"] for c in cals],
               t=[c["t"] for c in cals], H=H, Hinv=[np.linalg.inv(h) for h in H],
               centers=[-c["R"].T @ c["t"] for c in cals])
    ann = load_annotations()
    for frame in ann:
        per_cam = [[] for _ in CAMS]
        per_ids = [[] for _ in CAMS]
        gw, gid = [], []
        for person in frame:
            x, y = position_id_to_world(person["positionID"])
            gw.append((x, y))
            gid.append(person["personID"])
            for v in person["views"]:
                if v["xmin"] < 0 or v["xmax"] < 0:
                    continue
                c = v["viewNum"]
                fu, fv = 0.5 * (v["xmin"] + v["xmax"]), float(v["ymax"])
                hu, hv = fu, float(v["ymin"])
                g = sc.to_ground(c, np.array([fu, fv]))[0]
                per_cam[c].append((g[0], g[1], fu, fv, hu, hv))
                per_ids[c].append(person["personID"])
        sc.det.append([np.array(d, dtype=np.float64).reshape(-1, 6) for d in per_cam])
        sc.det_ids.append([np.array(i, dtype=np.int64) for i in per_ids])
        sc.gt_world.append(np.array(gw, dtype=np.float64).reshape(-1, 2))
        sc.gt_ids.append(np.array(gid, dtype=np.int64))
    # coverage zones
    xs = np.arange(X_RANGE[0] + CELL_CM / 2, X_RANGE[1], CELL_CM)
    ys = np.arange(Y_RANGE[0] + CELL_CM / 2, Y_RANGE[1], CELL_CM)
    cells = np.array([(x, y) for y in ys for x in xs])
    cover = np.zeros((len(CAMS), len(cells)), dtype=bool)
    for c in range(len(CAMS)):
        uv, depth = sc.project(c, cells)
        cover[c] = _in_image(uv, depth, margin=20)
    sc.cells, sc.cover = cells, cover
    dens = np.zeros(len(cells))
    for gw in sc.gt_world:
        if len(gw):
            ix = np.clip(((gw[:, 0] - X_RANGE[0]) // CELL_CM).astype(int), 0, len(xs) - 1)
            iy = np.clip(((gw[:, 1] - Y_RANGE[0]) // CELL_CM).astype(int), 0, len(ys) - 1)
            np.add.at(dens, iy * len(xs) + ix, 1.0)
    sc.density = dens / len(sc.gt_world)
    covered = cover.any(0)
    w = (0.05 + sc.density) * covered          # density-weighted, uncovered cells carry no weight
    sc.weights = w / w.sum()
    return sc


def visibility_graph(sc: Scene, min_shared: int = 1) -> np.ndarray:
    """Weighted camera visibility (FOV-overlap) graph: shared covered zones."""
    C = sc.n_cams
    A = np.zeros((C, C))
    for i in range(C):
        for j in range(C):
            if i != j:
                s = int((sc.cover[i] & sc.cover[j]).sum())
                A[i, j] = s if s >= min_shared else 0
    return A


def frame_time(idx: int) -> float:
    return idx / FPS


__all__ = ["Scene", "load_scene", "visibility_graph", "FPS", "FRAME_IDS"]
