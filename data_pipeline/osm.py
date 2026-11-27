"""OpenStreetMap aerodrome geometry (runways, taxiways, aprons, terminals) via Overpass."""
from __future__ import annotations

import json
import time

import requests

from .airports import AIRPORTS
from .provenance import DATA, record_source, sha256_bytes
from .remote_zip import USER_AGENT

ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
OUT = DATA / "osm"
QUERY = """[out:json][timeout:180];
(
  way["aeroway"~"^(runway|taxiway|apron|terminal|hangar|gate|parking_position)$"]({s},{w},{n},{e});
  node["aeroway"~"^(gate|parking_position)$"]({s},{w},{n},{e});
);
out geom;"""


def fetch(code: str) -> tuple[bytes, str]:
    s, w, n, e = AIRPORTS[code]["bbox"]
    q = QUERY.format(s=s, w=w, n=n, e=e)
    last = None
    for attempt in range(3):
        for ep in ENDPOINTS:
            try:
                r = requests.post(ep, data={"data": q}, headers={"User-Agent": USER_AGENT}, timeout=240)
                if r.status_code == 200 and r.content.lstrip().startswith(b"{"):
                    json.loads(r.content)
                    return r.content, ep
                last = f"{ep}: HTTP {r.status_code}"
            except (requests.RequestException, ValueError) as exc:
                last = f"{ep}: {exc}"
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"all Overpass mirrors failed for {code}: {last}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    files = {}
    for code in AIRPORTS:
        raw, ep = fetch(code)
        path = OUT / f"{code}_aeroway.json"
        path.write_bytes(raw)
        n = len(json.loads(raw)["elements"])
        files[code] = {"file": f"data/osm/{path.name}", "sha256": sha256_bytes(raw),
                       "endpoint": ep, "elements": n, "bbox_swne": AIRPORTS[code]["bbox"]}
        print(code, n, "elements from", ep, flush=True)
        time.sleep(5)
    record_source("openstreetmap_aeroways", {
        "name": "OpenStreetMap aeroway features (runway/taxiway/apron/terminal/hangar/gate)",
        "query_template": QUERY,
        "license": "ODbL 1.0, (c) OpenStreetMap contributors",
        "files": files,
    })


if __name__ == "__main__":
    main()
