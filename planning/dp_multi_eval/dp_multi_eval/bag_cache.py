"""Cache for data derived from a source bag.

Route goals and the dwell profile are pure functions of a recording, and the
recordings run to 72 GB. Every attempt re-derived both by re-reading the bag at
~30 MiB/s — three quarters of an hour of IO to recompute answers that cannot
have changed, on every retry of every pass. The cache keys on the bag's files
(path, sizes, mtimes) plus a version string per derivation, so editing the
derivation code or touching the bag invalidates cleanly.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable


def _cache_dir() -> Path:
    d = Path.home() / ".cache" / "dp_multi_eval" / "bag_derived"
    d.mkdir(parents=True, exist_ok=True)
    return d


def bag_fingerprint(bag_path: Path) -> str:
    bag_path = bag_path.expanduser().resolve()
    parts = [str(bag_path)]
    if bag_path.is_dir():
        for f in sorted(bag_path.iterdir()):
            if f.suffix in (".db3", ".mcap", ".yaml"):
                st = f.stat()
                parts.append(f"{f.name}:{st.st_size}:{int(st.st_mtime)}")
    else:
        st = bag_path.stat()
        parts.append(f"{st.st_size}:{int(st.st_mtime)}")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()


def cached_json(
    bag_path: Path,
    kind: str,
    version: str,
    compute: Callable[[], Any],
) -> Any:
    """The cached derivation, or compute() stored for next time.

    A failed compute is never cached; a cache file that will not parse is
    recomputed rather than trusted.
    """
    key = hashlib.sha1(
        f"{bag_fingerprint(bag_path)}|{kind}|{version}".encode()
    ).hexdigest()
    path = _cache_dir() / f"{key}.json"
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            print(f"[info] {kind} loaded from cache ({path.name})")
            return payload["data"]
        except (json.JSONDecodeError, OSError, KeyError):
            pass
    data = compute()
    try:
        path.write_text(
            json.dumps({"kind": kind, "version": version, "data": data}),
            encoding="utf-8",
        )
    except OSError:
        pass
    return data
