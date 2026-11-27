"""Real ADS-B state vectors from the OpenSky Network REST API.

Anonymous access serves live state vectors at 10 s resolution (historical
track endpoints require an account). We poll a bounding box around each airport
for a fixed window and keep every raw JSON response byte-for-byte, hashing each
one. Positions are *never* altered; downstream code treats them as an untrusted
weak prior.
"""
from __future__ import annotations

import argparse
import json
import time

import requests

from .airports import AIRPORTS
from .provenance import DATA, record_source, sha256_bytes, sha256_file
from .remote_zip import USER_AGENT

API = "https://opensky-network.org/api/states/all"
OUT = DATA / "opensky"
FIELDS = ["icao24", "callsign", "origin_country", "time_position", "last_contact", "longitude",
          "latitude", "baro_altitude", "on_ground", "velocity", "true_track", "vertical_rate",
          "sensors", "geo_altitude", "squawk", "spi", "position_source"]


def poll(minutes: float, interval: float) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    sess = requests.Session()
    sess.headers["User-Agent"] = USER_AGENT
    n_polls = int(minutes * 60 / interval)
    snaps = {c: [] for c in AIRPORTS}
    for i in range(n_polls):
        t0 = time.time()
        for code, ap in AIRPORTS.items():
            s, w, n, e = ap["bbox"]
            try:
                r = sess.get(API, params={"lamin": s, "lomin": w, "lamax": n, "lomax": e}, timeout=30)
            except requests.RequestException as exc:
                print(code, "error", exc, flush=True)
                continue
            if r.status_code != 200:
                print(code, "HTTP", r.status_code, flush=True)
                continue
            snaps[code].append({"sha256": sha256_bytes(r.content), "body": json.loads(r.content)})
        print(f"poll {i + 1}/{n_polls}", {c: len(v) for c, v in snaps.items()}, flush=True)
        time.sleep(max(0.0, interval - (time.time() - t0)))
    files = {}
    for code, lst in snaps.items():
        path = OUT / f"{code}_states.json"
        path.write_text(json.dumps({"fields": FIELDS, "snapshots": lst}), encoding="utf-8")
        n_states = sum(len(s["body"].get("states") or []) for s in lst)
        files[code] = {"file": f"data/opensky/{path.name}", "sha256": sha256_file(path),
                       "snapshots": len(lst), "state_vectors": n_states,
                       "first_time": lst[0]["body"]["time"] if lst else None,
                       "last_time": lst[-1]["body"]["time"] if lst else None,
                       "per_response_sha256": [s["sha256"] for s in lst]}
    return files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=15)
    ap.add_argument("--interval", type=float, default=10)
    a = ap.parse_args()
    files = poll(a.minutes, a.interval)
    record_source("opensky_adsb_states", {
        "name": "OpenSky Network live ADS-B/Mode-S state vectors (anonymous REST API)",
        "url": API,
        "license": "OpenSky Network terms of use (non-commercial research; cite Schafer et al., IPSN 2014); raw data not redistributed",
        "citation": "M. Schafer et al., 'Bringing up OpenSky', IPSN 2014",
        "poll_interval_s": a.interval, "window_minutes": a.minutes,
        "files": files,
    })


if __name__ == "__main__":
    main()
