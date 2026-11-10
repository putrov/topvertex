"""Local HTTP server for the graphical input page (``python -m topvertex --draw``).

Standard library only (:class:`http.server.ThreadingHTTPServer`), bound to 127.0.0.1.  Routes:

* ``GET /`` -- the page ``topvertex/draw/index.html`` (package data, read with importlib.resources);
* ``POST /api/compute`` -- body ``{corners: [[x,y],...], triangles: [[[x,y],[x,y],[x,y]],...],
  order: int, heights?: [int,...], gv: bool, classes: bool, nekrasov: bool,
  format: "text"|"latex"|"mathematica"|"sympy"|"mathjax", latex?: bool}``.  The triangulation is validated by
  :meth:`topvertex.polygon.Triangulation.validate` (its message is returned verbatim) and the result
  is computed by :func:`topvertex.cli.geometry_report` / :func:`topvertex.cli.series_report`, i.e.
  exactly what the command line prints (``nekrasov``: the comparison with the 5d Nekrasov partition
  function of the linear quiver for all classes up to the order).  Answer: ``{points,
  triangles, edges: [{name, segment, n, weight, class, tail, head}], triangles_str, basis, heights,
  weights, order, Z, F?, gv_table?, nekrasov?, command, time}`` (``triangles`` as point triples, ``triangles_str`` in the CLI's --triangles format), or
  ``{error}`` with HTTP 400 for invalid input (500 for an unexpected failure).  With format
  "mathjax" (text fields then in format "text") or ``latex: true`` also LaTeX for MathJax 3
  (:mod:`topvertex.draw.mathjax`): ``Z_latex`` (list, one line per coefficient), ``F_latex`` (list,
  with gv), ``gv_latex`` (array), ``edges_latex`` (array);
* ``POST /api/validate`` -- same ``corners``/``triangles``; answers ``{ok: true, points, edges:
  [{name, segment, n, tail, head}], regular, heights}`` or ``{ok: false, error}`` (HTTP 200).

Limits (unless ``unsafe``): order D <= 12, at most 30 lattice points.
The request handling itself lives in :mod:`topvertex.draw.compute` (shared with the Pyodide backend
of the page, which runs the same code in the browser).
Requests must carry ``Content-Type: application/json`` and a local ``Host`` header (127.0.0.1,
localhost or [::1], any port, so that ``ssh -L`` forwarding works); this keeps other web sites
from driving the server through the browser.
"""
from __future__ import annotations

import json
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources

from .. import polygon as pg
from .compute import FORMATS, SERVER_LIMITS, api_compute, api_validate, cli_command, handle

__all__ = ["MAX_ORDER", "MAX_POINTS", "FORMATS", "page_html", "make_server", "serve", "api_compute",
           "api_validate", "cli_command"]

MAX_ORDER = SERVER_LIMITS.order
MAX_POINTS = SERVER_LIMITS.points
MAX_BODY = 1 << 20
LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]", "::1"}


def page_html() -> bytes:
    """The page ``index.html`` shipped as package data."""
    return resources.files("topvertex.draw").joinpath("index.html").read_bytes()


# ----------------------------------------------------------------------------- HTTP

class _Handler(BaseHTTPRequestHandler):
    server_version = "topvertex-draw/0.1"

    def log_message(self, fmt, *args):  # noqa: D401 - quiet unless verbose
        if getattr(self.server, "verbose", False):
            super().log_message(fmt, *args)

    def _local_host(self) -> bool:
        host = (self.headers.get("Host") or "").strip().lower()
        if host.startswith("["):
            name = host[:host.find("]") + 1]
        else:
            name = host.rsplit(":", 1)[0] if ":" in host else host
        return name in LOCAL_HOSTS

    def _send(self, status: int, body: bytes, ctype: str):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, obj):
        self._send(status, json.dumps(obj).encode(), "application/json; charset=utf-8")

    def do_GET(self):
        if not self._local_host():
            return self._json(HTTPStatus.FORBIDDEN, {"error": "non-local Host header"})
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            return self._send(HTTPStatus.OK, page_html(), "text/html; charset=utf-8")
        self._json(HTTPStatus.NOT_FOUND, {"error": f"not found: {path}"})

    def do_POST(self):
        if not self._local_host():
            return self._json(HTTPStatus.FORBIDDEN, {"error": "non-local Host header"})
        if not (self.headers.get("Content-Type") or "").startswith("application/json"):
            return self._json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "expected application/json"})
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            return self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "request too large"})
        try:
            body = json.loads(self.rfile.read(length) or b"null")
        except json.JSONDecodeError as exc:
            return self._json(HTTPStatus.BAD_REQUEST, {"error": f"invalid JSON: {exc}"})
        status, answer = handle(self.path.split("?", 1)[0], body, None if self.server.unsafe else SERVER_LIMITS)
        self._json(status, answer)


def make_server(port: int = 8765, unsafe: bool = False, verbose: bool = False) -> ThreadingHTTPServer:
    """A ThreadingHTTPServer on 127.0.0.1:port (port 0: a free port) serving the page and the API."""
    httpd = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    httpd.daemon_threads = True
    httpd.unsafe = unsafe
    httpd.verbose = verbose
    return httpd


def serve(port: int = 8765, open_browser: bool = True, unsafe: bool = False, polygon: str | None = None,
          verbose: bool = True) -> None:
    """Serve the drawing page on http://127.0.0.1:port/ until Ctrl-C; print the URL and open it in
    the default browser (:mod:`webbrowser`) unless ``open_browser`` is False.  ``polygon`` (compact
    corner string) is preloaded via the URL hash ``#p=...``."""
    httpd = make_server(port, unsafe, verbose)
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    if polygon:
        url += "#" + pg.format_polygon(pg.parse_polygon(polygon))
    print(f"topvertex --draw: serving on {url}  (Ctrl-C to stop)", flush=True)
    if unsafe:
        print("  --unsafe: no limits on order / number of lattice points", flush=True)
    print(f"  remote machine? forward the port: ssh -L {httpd.server_address[1]}:127.0.0.1:"
          f"{httpd.server_address[1]} host", flush=True)
    if open_browser:
        threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("stopped", file=sys.stderr)
    finally:
        httpd.server_close()
