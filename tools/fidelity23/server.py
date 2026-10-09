#!/usr/bin/env python3
"""Fidelity23 local web service (stdlib only).

Serves the Chinese comparison workbench at http://localhost:8230:
  /                web/index.html (+ app.js, style.css, vendor/echarts)
  /api/overview    per-version medians, representative runs, segment tables
  /api/run/<V>/<t> full metrics.json (series included) for one run

Reads data/runs/<VERSION>/*.metrics.json produced by evaluator.py — this
service never writes; run the evaluator after each acceptance round and the
page picks the new runs up on refresh.
"""
import json
import os
import re
import statistics
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(ROOT, "web")
RUNS = os.path.join(ROOT, "data", "runs")
PORT = int(os.environ.get("FIDELITY23_PORT", "8230"))

MIME = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8", ".png": "image/png", ".svg": "image/svg+xml"}


def load_overview():
    versions = {}
    for ver in sorted(os.listdir(RUNS)):
        vdir = os.path.join(RUNS, ver)
        if not os.path.isdir(vdir) or not os.path.exists(os.path.join(vdir, "meta.json")):
            continue
        meta = json.load(open(os.path.join(vdir, "meta.json")))
        runs = []
        for f in sorted(os.listdir(vdir)):
            if f.endswith(".metrics.json"):
                runs.append(json.load(open(os.path.join(vdir, f))))
        if not runs:
            continue
        flat = [r["summary"] for r in runs]

        def med(path):
            vals = []
            for s in flat:
                v = s
                try:
                    for p in path:
                        v = v[p]
                except (KeyError, TypeError):
                    v = None
                if v is not None:
                    vals.append(v)
            return statistics.median(vals) if vals and all(isinstance(v, (int, float)) for v in vals) else None

        median_summary = {
            "mpjpe_world_p50": med(["mpjpe_world_p50"]), "mpjpe_world_p90": med(["mpjpe_world_p90"]),
            "fsr_ref_L": med(["fsr_ref_L"]), "fsr_ref_R": med(["fsr_ref_R"]),
            "slide_ref_p50_L": med(["slide_ref_p50_L"]), "slide_ref_p50_R": med(["slide_ref_p50_R"]),
            "float_L": med(["float", "L", "duration_frac"]), "float_R": med(["float", "R", "duration_frac"]),
            "penetration": med(["penetration", "duration_frac"]),
            "kick_apex_ratio_L": med(["kick_apex_ratio", "L"]), "kick_apex_cm_L": med(["kick_apex_cm", "L"]),
            "drift_net": med(["ending_drift_m", "net"]), "drift_path": med(["ending_drift_m", "path"]),
            "ankle_margin_rad": med(["ankle_margin_rad"]),
        }
        segs = {}
        for seg in runs[0]["segments"]:
            segs[seg] = {k: statistics.median([r["segments"][seg][k] for r in runs
                                               if isinstance(r["segments"][seg][k], (int, float))]
                                              or [float("nan")]) for k in runs[0]["segments"][seg]}
        joints = {"names": runs[0]["joints"]["names"],
                  "err_p90": [statistics.median(x) for x in zip(*[r["joints"]["err_p90"] for r in runs])],
                  "err_p50": [statistics.median(x) for x in zip(*[r["joints"]["err_p50"] for r in runs])],
                  "std_ratio": [statistics.median(x) for x in zip(*[r["joints"]["std_ratio"] for r in runs])]}
        m0 = median_summary["mpjpe_world_p50"]
        rep = min(runs, key=lambda r: abs(r["summary"]["mpjpe_world_p50"] - m0))
        versions[ver] = {"meta": meta, "runs": [r["tag"] for r in runs], "n": len(runs),
                         "median": median_summary, "segments": segs, "joints": joints,
                         "representative": rep["tag"],
                         "routing": rep["routing"],
                         "t_entry_source": {r["tag"]: r["t_entry_source"] for r in runs}}
    return versions


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, mime="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/" or path == "/index.html":
            self._send(200, open(os.path.join(WEB, "index.html"), "rb").read(), MIME[".html"])
            return
        if path == "/api/overview":
            self._send(200, json.dumps(load_overview(), ensure_ascii=False))
            return
        m = re.fullmatch(r"/api/run/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)", path)
        if m:
            p = os.path.join(RUNS, m.group(1), m.group(2) + ".metrics.json")
            if os.path.exists(p):
                self._send(200, open(p, "rb").read())
                return
        if ".." not in path and not path.startswith("/api"):
            p = os.path.join(WEB, path.lstrip("/"))
            if os.path.exists(p) and os.path.isfile(p):
                ext = os.path.splitext(p)[1]
                self._send(200, open(p, "rb").read(), MIME.get(ext, "application/octet-stream"))
                return
        self._send(404, '{"error":"not found"}')


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"fidelity23 serving http://localhost:{PORT} (data: {RUNS})")
    sys.stdout.flush()
    srv.serve_forever()


if __name__ == "__main__":
    main()
