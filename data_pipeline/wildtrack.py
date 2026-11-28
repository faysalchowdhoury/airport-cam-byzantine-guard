"""WILDTRACK downloader (range-request streaming) + calibration / annotation parsing.

Dataset: T. Chavdarova et al., "WILDTRACK: A Multi-camera HD Dataset for Dense
Unscripted Pedestrian Detection", CVPR 2018. Seven synchronized, calibrated,
overlapping 1920x1080 cameras; 400 annotated frames at 2 fps.

Only the members we need are fetched. Full-resolution PNGs are decoded in memory,
SHA-256 hashed (decompressed bytes; the ZIP CRC-32 is verified by ``zipfile``), and
stored as down-sampled grayscale arrays so the pipeline fits on small disks.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .provenance import DATA, record_source, sha256_bytes, sha256_file
from .remote_zip import open_remote_zip

URL = "https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/Wildtrack/Wildtrack_dataset_full.zip"
LANDING = "https://www.epfl.ch/labs/cvlab/data/data-wildtrack/"
PREFIX = "Wildtrack_dataset/"
OUT = DATA / "wildtrack"
CAMS = ["CVLab1", "CVLab2", "CVLab3", "CVLab4", "IDIAP1", "IDIAP2", "IDIAP3"]
IMG_W, IMG_H = 1920, 1080
DOWN = 4  # 1920x1080 -> 480x270
FRAME_IDS = list(range(0, 2000, 5))  # the 400 annotated frames


# --------------------------------------------------------------------------- download
def _small_members(z) -> list[str]:
    keep = []
    for n in z.namelist():
        if n.startswith("__MACOSX") or n.endswith("/"):
            continue
        if n.startswith(PREFIX + "calibrations/") or n.startswith(PREFIX + "annotations_positions/"):
            keep.append(n)
        elif n in (PREFIX + "README.txt",):
            keep.append(n)
    return keep


def download_metadata() -> dict:
    z, f = open_remote_zip(URL)
    hashes = {}
    for n in _small_members(z):
        data = z.read(n)  # zipfile verifies CRC-32 on read
        dst = OUT / "raw" / n[len(PREFIX):]
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        hashes[n] = sha256_bytes(data)
    return {"members": hashes, "bytes_fetched": f.bytes_fetched, "archive_size": f.size}


def _decode_frames(cam_idx: int, frame_ids: list[int]) -> tuple[int, dict]:
    """Decode one camera straight into an on-disk memmap; resumable per frame.

    Progress (per-frame SHA-256) is checkpointed to ``C{k}.partial.json`` so a crash
    or restart only re-fetches frames not yet written. Peak RAM is ~one PNG.
    """
    from PIL import Image

    frames_dir = OUT / "frames"
    final = frames_dir / f"C{cam_idx + 1}.npy"
    tmp = frames_dir / f"C{cam_idx + 1}.partial.npy"
    prog = frames_dir / f"C{cam_idx + 1}.partial.json"
    shape = (len(frame_ids), IMG_H // DOWN, IMG_W // DOWN)
    hashes: dict[str, str] = {}
    if final.exists() and prog.exists():
        hashes = json.loads(prog.read_text(encoding="utf-8"))
        if len(hashes) == len(frame_ids) and np.load(final, mmap_mode="r").shape == shape:
            return cam_idx, hashes
    if tmp.exists() and prog.exists():
        arr = np.lib.format.open_memmap(tmp, mode="r+")
        hashes = json.loads(prog.read_text(encoding="utf-8"))
    else:
        arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=shape)
        hashes = {}
    z, _ = open_remote_zip(URL)
    for i, fid in enumerate(frame_ids):
        name = f"{PREFIX}Image_subsets/C{cam_idx + 1}/{fid:08d}.png"
        if name in hashes:
            continue
        data = z.read(name)
        h = sha256_bytes(data)
        img = Image.open(io.BytesIO(data)).convert("L")
        del data
        if img.size != (IMG_W, IMG_H):
            raise ValueError(f"unexpected frame size {img.size} for {name}")
        arr[i] = np.asarray(img.reduce(DOWN), dtype=np.uint8)
        hashes[name] = h
        if (i + 1) % 10 == 0 or i + 1 == len(frame_ids):
            arr.flush()
            prog.write_text(json.dumps(hashes), encoding="utf-8")
            print(f"C{cam_idx + 1}: {len(hashes)}/{len(frame_ids)}", flush=True)
    arr.flush()
    del arr
    tmp.replace(final)
    prog.write_text(json.dumps(hashes), encoding="utf-8")
    return cam_idx, hashes


def download_frames(n_frames: int = 400, workers: int = 2) -> dict:
    frame_ids = FRAME_IDS[:n_frames]
    (OUT / "frames").mkdir(parents=True, exist_ok=True)
    all_hashes: dict[str, str] = {}
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_decode_frames, c, frame_ids) for c in range(7)]
        for fut in cf.as_completed(futs):
            c, hashes = fut.result()
            all_hashes.update(hashes)
            print(f"camera C{c + 1}: {len(hashes)} frames stored", flush=True)
    manifest = OUT / "frame_sha256_manifest.json"
    manifest.write_text(json.dumps(all_hashes, indent=1, sort_keys=True), encoding="utf-8")
    return {"n_frames_per_camera": len(frame_ids), "manifest": str(manifest.relative_to(DATA.parent)),
            "manifest_sha256": sha256_file(manifest)}


# --------------------------------------------------------------------------- parsing
def _cv_mat(node: ET.Element) -> np.ndarray:
    rows = int(node.find("rows").text)
    cols = int(node.find("cols").text)
    vals = [float(v) for v in node.find("data").text.split()]
    return np.array(vals, dtype=np.float64).reshape(rows, cols)


def load_calibration(cam: str, root: Path | None = None) -> dict:
    """Return K, dist, R, t (t in centimetres, WILDTRACK world units)."""
    root = root or OUT / "raw" / "calibrations"
    intr = ET.parse(root / "intrinsic_zero" / f"intr_{cam}.xml").getroot()
    K = _cv_mat(intr.find("camera_matrix"))
    dist = _cv_mat(intr.find("distortion_coefficients")).ravel()
    extr_txt = (root / "extrinsic" / f"extr_{cam}.xml").read_text(encoding="utf-8")
    ext = ET.fromstring(extr_txt)
    rvec = np.array([float(v) for v in ext.find("rvec").text.split()])
    tvec = np.array([float(v) for v in ext.find("tvec").text.split()])
    return {"K": K, "dist": dist, "R": rodrigues(rvec), "t": tvec}


def rodrigues(r: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(r))
    if th < 1e-12:
        return np.eye(3)
    k = r / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


def ground_homography(cal: dict) -> np.ndarray:
    """3x3 homography mapping ground-plane (X, Y, 1) [cm] -> image pixels (z=0 plane)."""
    P = cal["K"] @ np.hstack([cal["R"], cal["t"].reshape(3, 1)])
    return P[:, [0, 1, 3]]


def position_id_to_world(pos: int) -> tuple[float, float]:
    """WILDTRACK positionID -> ground-plane coordinates in centimetres.

    Grid of 480 x 1440 cells of 2.5 cm, origin at (-300, -900) cm (dataset README).
    """
    x = -300.0 + 2.5 * (pos % 480)
    y = -900.0 + 2.5 * (pos // 480)
    return x, y


def load_annotations(root: Path | None = None) -> list[list[dict]]:
    root = root or OUT / "raw" / "annotations_positions"
    frames = []
    for fid in FRAME_IDS:
        frames.append(json.loads((root / f"{fid:08d}.json").read_text(encoding="utf-8")))
    return frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--skip-frames", action="store_true")
    a = ap.parse_args()
    meta = download_metadata()
    entry = {
        "name": "WILDTRACK multi-camera pedestrian dataset",
        "url": URL,
        "landing_page": LANDING,
        "citation": "Chavdarova et al., CVPR 2018",
        "license": "EPFL CVLab research-use terms (see landing page); not redistributed in this repository",
        "access_method": "HTTP Range requests on the remote ZIP (only required members fetched)",
        "archive_size_bytes": meta["archive_size"],
        "member_sha256": meta["members"],
    }
    record_source("wildtrack_metadata", entry)
    print(f"metadata members: {len(meta['members'])}")
    if not a.skip_frames:
        fr = download_frames(a.frames)
        record_source("wildtrack_frames", {
            "name": "WILDTRACK Image_subsets (C1..C7), decoded and 4x box-downsampled to 480x270 grayscale",
            "url": URL, "license": entry["license"], **fr,
        })


if __name__ == "__main__":
    main()
