"""SHA-256 provenance ledger shared by all downloaders."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
LEDGER = ARTIFACTS / "data_provenance.json"
_LOCK = threading.Lock()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_ledger() -> dict:
    if LEDGER.exists():
        return json.loads(LEDGER.read_text(encoding="utf-8"))
    return {"schema": "airport-cam-byzantine-guard/provenance/v1", "sources": {}}


def record_source(key: str, entry: dict) -> None:
    """Insert/replace one source entry in the provenance ledger (thread-safe)."""
    with _LOCK:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        led = load_ledger()
        entry = dict(entry)
        entry.setdefault("retrieved_utc", now_utc())
        led["sources"][key] = entry
        LEDGER.write_text(json.dumps(led, indent=2, sort_keys=True), encoding="utf-8")
