# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GUI results server (embedded dashboard): files come back whole or as one byte
range (206, Content-Range), "-N" means the last N bytes, a range past the end is
416, and the dashboard's ?v= cache-buster is ignored.
Run: python3 tests/test_results_server.py"""
import functools
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from dp_multi_eval.app import ResultsHandler  # noqa: E402
except (ImportError, SystemExit):
    print("SKIP: needs streamlit")
    raise SystemExit(0)


def get(url: str, rng: str | None = None):
    req = urllib.request.Request(url, headers={"Range": rng} if rng else {})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.headers, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers, b""


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        data = bytes(range(256)) * 4  # 1024 bytes
        (Path(d) / "videos").mkdir()
        (Path(d) / "videos" / "p.mp4").write_bytes(data)
        server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(ResultsHandler, directory=d))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{server.server_address[1]}/videos/p.mp4?v=123"
        try:
            st, h, body = get(url)
            assert st == 200 and body == data and h.get("Accept-Ranges") == "bytes", (st, h)
            assert h.get("Content-Type") == "video/mp4", h

            st, h, body = get(url, "bytes=10-19")
            assert st == 206 and body == data[10:20], (st, body)
            assert h["Content-Range"] == "bytes 10-19/1024" and h["Content-Length"] == "10", h

            st, h, body = get(url, "bytes=1000-")
            assert st == 206 and body == data[1000:], (st, len(body))

            st, h, body = get(url, "bytes=-8")
            assert st == 206 and body == data[-8:] and h["Content-Range"] == "bytes 1016-1023/1024", h

            st, h, _ = get(url, "bytes=5000-")
            assert st == 416 and h["Content-Range"] == "bytes */1024", (st, h)
        finally:
            server.shutdown()
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
