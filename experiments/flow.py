"""Dense optical flow (Horn-Schunck, PyTorch) on real WILDTRACK frames.

Implemented without OpenCV so the pipeline has no native-vision dependency.
Frames are the 480x270 grayscale arrays from the downloader, further halved to
240x135 for flow. Results are cached per (camera, previous source frame,
current source frame) so replayed / frozen streams reuse computations.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F

from data_pipeline.provenance import DATA

FRAMES_DIR = DATA / "wildtrack" / "frames"
_KAVG = torch.tensor([[1 / 12, 1 / 6, 1 / 12], [1 / 6, 0, 1 / 6], [1 / 12, 1 / 6, 1 / 12]]).view(1, 1, 3, 3)


@lru_cache(maxsize=8)
def frames(cam: int) -> torch.Tensor:
    arr = np.load(FRAMES_DIR / f"C{cam + 1}.npy", mmap_mode="r")
    t = torch.from_numpy(np.ascontiguousarray(arr)).float().unsqueeze(1) / 255.0
    return F.avg_pool2d(t, 2)  # (N, 1, 135, 240)


def horn_schunck(a: torch.Tensor, b: torch.Tensor, alpha: float = 0.1, iters: int = 60):
    """a, b: (1,1,H,W) in [0,1]. Returns (u, v) flow fields."""
    kx = torch.tensor([[-1, 1], [-1, 1]], dtype=a.dtype).view(1, 1, 2, 2) * 0.25
    ky = torch.tensor([[-1, -1], [1, 1]], dtype=a.dtype).view(1, 1, 2, 2) * 0.25
    kt = torch.ones(1, 1, 2, 2, dtype=a.dtype) * 0.25
    pa, pb = F.pad(a, (0, 1, 0, 1), mode="replicate"), F.pad(b, (0, 1, 0, 1), mode="replicate")
    Ix = F.conv2d(pa, kx) + F.conv2d(pb, kx)
    Iy = F.conv2d(pa, ky) + F.conv2d(pb, ky)
    It = F.conv2d(pb, kt) - F.conv2d(pa, kt)
    u = torch.zeros_like(a)
    v = torch.zeros_like(a)
    den = alpha ** 2 + Ix ** 2 + Iy ** 2
    for _ in range(iters):
        ub = F.conv2d(F.pad(u, (1, 1, 1, 1), mode="replicate"), _KAVG)
        vb = F.conv2d(F.pad(v, (1, 1, 1, 1), mode="replicate"), _KAVG)
        d = (Ix * ub + Iy * vb + It) / den
        u, v = ub - Ix * d, vb - Iy * d
    return u, v


def warp(img: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    _, _, H, W = img.shape
    yy, xx = torch.meshgrid(torch.arange(H, dtype=img.dtype), torch.arange(W, dtype=img.dtype), indexing="ij")
    gx = (xx + u[0, 0]) / (W - 1) * 2 - 1
    gy = (yy + v[0, 0]) / (H - 1) * 2 - 1
    grid = torch.stack([gx, gy], -1).unsqueeze(0)
    return F.grid_sample(img, grid, align_corners=True, padding_mode="border")


@lru_cache(maxsize=200_000)
def flow_stats(cam: int, src_prev: int, src_cur: int) -> tuple[float, float, float]:
    """(mean flow magnitude [px], photometric residual after warp, raw abs frame difference)."""
    if src_prev == src_cur:
        return 0.0, 0.0, 0.0
    fr = frames(cam)
    a, b = fr[src_prev : src_prev + 1], fr[src_cur : src_cur + 1]
    with torch.no_grad():
        u, v = horn_schunck(a, b)
        mag = torch.sqrt(u ** 2 + v ** 2).mean().item()
        resid = (warp(b, u, v) - a).abs().mean().item()
        diff = (b - a).abs().mean().item()
    return mag, resid, diff


def frames_available() -> bool:
    return all((FRAMES_DIR / f"C{c + 1}.npy").exists() for c in range(7))
