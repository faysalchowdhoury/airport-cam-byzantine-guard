"""Read individual members of a remote ZIP archive via HTTP Range requests.

WILDTRACK is distributed as a single 6.8 GB archive. Rather than downloading it in
full, we read the central directory and fetch only the members we need
(calibration, annotations, a subsampled frame set). Every fetched member is
SHA-256 hashed and logged in ``artifacts/data_provenance.json``.
"""
from __future__ import annotations

import io
import zipfile

import requests

USER_AGENT = "airport-cam-byzantine-guard/0.1 (research data pipeline)"


class HTTPRangeFile(io.RawIOBase):
    """Seekable read-only file object backed by HTTP Range requests."""

    def __init__(self, url: str, block: int = 1 << 20, timeout: float = 120.0):
        self.url = url
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.timeout = timeout
        head = self.session.head(url, allow_redirects=True, timeout=timeout)
        head.raise_for_status()
        self.size = int(head.headers["Content-Length"])
        self.pos = 0
        self.block = block
        self._cache_start = -1
        self._cache = b""
        self.bytes_fetched = 0

    def readable(self) -> bool:  # noqa: D401
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        return self.pos

    def _fetch(self, start: int, end: int) -> bytes:
        for attempt in range(5):
            try:
                r = self.session.get(
                    self.url, headers={"Range": f"bytes={start}-{end - 1}"}, timeout=self.timeout
                )
                if r.status_code != 206:
                    raise IOError(f"server did not honour range request: HTTP {r.status_code}")
                self.bytes_fetched += len(r.content)
                return r.content
            except (requests.RequestException, IOError):
                if attempt == 4:
                    raise
        raise RuntimeError("unreachable")

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        cs, ce = self._cache_start, self._cache_start + len(self._cache)
        if cs <= self.pos and self.pos + n <= ce:
            out = self._cache[self.pos - cs : self.pos - cs + n]
        elif n >= self.block:
            out = self._fetch(self.pos, self.pos + n)
        else:
            end = min(self.size, self.pos + max(n, self.block))
            self._cache = self._fetch(self.pos, end)
            self._cache_start = self.pos
            out = self._cache[:n]
        self.pos += len(out)
        return out

    def readinto(self, b) -> int:  # pragma: no cover - used by zipfile internals
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def repair_wrapped_offsets(z: zipfile.ZipFile) -> int:
    """Repair local-header offsets that wrapped modulo 2**32.

    Some archivers (e.g. macOS Archive Utility) write archives > 4 GiB without
    proper ZIP64 offsets, so central-directory offsets are only correct modulo
    2**32. Local headers are laid out in central-directory order, so the true
    offset is the smallest value congruent to the stored one (mod 2**32) that is
    >= the previous member's offset. ``zipfile`` re-checks the local-header file
    name on open, so a wrong reconstruction fails loudly rather than silently.
    Returns the number of members whose offset changed.
    """
    prev, changed = 0, 0
    for info in z.infolist():
        low = info.header_offset % (1 << 32)
        cand = low + ((prev - low + (1 << 32) - 1) // (1 << 32)) * (1 << 32) if prev > low else low
        if cand != info.header_offset:
            info.header_offset = cand
            changed += 1
        prev = cand
    # zipfile's overlap ("zip bomb") guard uses per-member end offsets computed at
    # open time from the *unrepaired* offsets; recompute them from repaired ones.
    infos = sorted(z.infolist(), key=lambda i: i.header_offset)
    for a, b in zip(infos, infos[1:] + [None]):
        if hasattr(a, "_end_offset"):
            a._end_offset = b.header_offset if b is not None else z.start_dir
    return changed


def open_remote_zip(url: str) -> tuple[zipfile.ZipFile, HTTPRangeFile]:
    f = HTTPRangeFile(url)
    z = zipfile.ZipFile(f)
    repair_wrapped_offsets(z)
    return z, f
