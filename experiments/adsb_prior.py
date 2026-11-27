"""Untrusted ADS-B weak prior from real OpenSky state vectors + OSM aerodrome zones.

The prior is *never* ground truth. It only modulates zone criticality (bounded
multiplicatively by 1 +/- beta) and enters the attestor through a bounded gate.
Spoofing is emulated without inventing coordinates: the real JFK stream is
replaced by the real ATL stream ("track transplant"), the attacker's cheapest
way to inject a plausible but false traffic picture.
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np

from data_pipeline.airports import AIRPORTS
from data_pipeline.provenance import DATA


@lru_cache(maxsize=4)
def load_states(code: str) -> list[dict]:
    raw = json.loads((DATA / "opensky" / f"{code}_states.json").read_text(encoding="utf-8"))
    fields = raw["fields"]
    snaps = []
    for s in raw["snapshots"]:
        rows = [dict(zip(fields, r)) for r in (s["body"].get("states") or [])]
        snaps.append({"time": s["body"]["time"], "states": rows})
    return snaps


def activity_series(code: str) -> np.ndarray:
    """Per-snapshot weak-prior activity: aircraft on ground or below 1,000 m in the bbox."""
    out = []
    for s in load_states(code):
        n = 0
        for r in s["states"]:
            if r["latitude"] is None:
                continue
            alt = r["baro_altitude"] if r["baro_altitude"] is not None else r["geo_altitude"]
            if r["on_ground"] or (alt is not None and alt < 1000):
                n += 1
        out.append(n)
    return np.array(out, float)


def prior_for_episode(rng: np.random.Generator, T: int, fps: float = 2.0, code: str = "JFK",
                      spoof_from: str | None = None) -> np.ndarray:
    """Align a random contiguous slice of the real ADS-B series to an episode, scaled to [0,1]."""
    series = activity_series(spoof_from or code)
    ref = activity_series(code)
    snap_dt = 10.0
    need = int(np.ceil(T / fps / snap_dt)) + 1
    start = int(rng.integers(0, max(1, len(series) - need)))
    seg = series[start:start + need]
    tt = np.arange(T) / fps / snap_dt
    val = np.interp(tt, np.arange(len(seg)), seg)
    lo, hi = ref.min(), max(ref.max(), ref.min() + 1)
    return np.clip((val - lo) / (hi - lo), 0, 1.5).astype(np.float32)


# ---------------------------------------------------------------------- OSM zones
def osm_elements(code: str) -> list[dict]:
    return json.loads((DATA / "osm" / f"{code}_aeroway.json").read_text(encoding="utf-8"))["elements"]


def to_local_m(lat: np.ndarray, lon: np.ndarray, code: str) -> np.ndarray:
    lat0, lon0 = AIRPORTS[code]["center"]
    x = (lon - lon0) * 111_320.0 * np.cos(np.radians(lat0))
    y = (lat - lat0) * 110_540.0
    return np.c_[x, y]


def zone_polygons(code: str) -> list[dict]:
    zones = []
    for e in osm_elements(code):
        if e["type"] != "way" or "geometry" not in e:
            continue
        kind = e.get("tags", {}).get("aeroway")
        g = e["geometry"]
        lat = np.array([p["lat"] for p in g])
        lon = np.array([p["lon"] for p in g])
        zones.append({"id": e["id"], "kind": kind, "xy": to_local_m(lat, lon, code),
                      "closed": g[0] == g[-1], "name": e.get("tags", {}).get("name") or e.get("tags", {}).get("ref")})
    return zones


def point_in_poly(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    x, y = pts[:, 0][:, None], pts[:, 1][:, None]
    xi, yi = poly[:-1, 0][None], poly[:-1, 1][None]
    xj, yj = poly[1:, 0][None], poly[1:, 1][None]
    cond = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / np.where(yj - yi == 0, 1e-12, yj - yi) + xi)
    return cond.sum(1) % 2 == 1


def aircraft_xy(code: str, states_code: str | None = None) -> list[np.ndarray]:
    """Per snapshot: (n, 2) local metres of aircraft on ground / below 1,000 m.

    ``states_code`` != ``code`` gives the transplant spoof (another airport's real
    traffic re-referenced onto this airport's frame by relative offset).
    """
    sc = states_code or code
    out = []
    for s in load_states(sc):
        pts = [(r["latitude"], r["longitude"]) for r in s["states"]
               if r["latitude"] is not None and (r["on_ground"] or ((r["baro_altitude"] or 1e9) < 1000))]
        if not pts:
            out.append(np.zeros((0, 2)))
            continue
        a = np.array(pts)
        out.append(to_local_m(a[:, 0], a[:, 1], sc))
    return out
