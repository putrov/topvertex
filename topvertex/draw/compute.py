"""Request handling shared by the local server (:mod:`topvertex.draw.server`) and the in-browser
Pyodide backend of the drawing page: parse a JSON request, run the command-line computation
(:func:`topvertex.cli.geometry_report` / :func:`topvertex.cli.series_report`) and build the JSON
answer, so that both backends return exactly the same structure (documented in
:mod:`topvertex.draw.server`).

:func:`handle` maps (path, body) to (HTTP status, answer dict); :func:`handle_json` is the
string-in / string-out wrapper called from JavaScript in Pyodide.  Limits: :data:`SERVER_LIMITS`
(lifted by ``--unsafe``) and :data:`PYODIDE_LIMITS` (the online version).
"""
from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass

from .. import polygon as pg
from ..cli import InputError, geometry_report, series_report
from ..toric import from_triangulation
from . import mathjax as mj

__all__ = ["Limits", "SERVER_LIMITS", "PYODIDE_LIMITS", "REPO_URL", "FORMATS", "api_compute", "api_validate",
           "cli_command", "handle", "handle_json"]

FORMATS = ("text", "latex", "mathematica", "sympy", "mathjax")
REPO_URL = "https://github.com/putrov/topvertex"


@dataclass(frozen=True)
class Limits:
    """Maximal order D and number of lattice points (the Nekrasov comparison runs up to the same
    order D); ``hint`` says how to lift them."""
    order: int
    points: int
    hint: str


SERVER_LIMITS = Limits(12, 30, "restart with python -m topvertex --draw --unsafe to lift it")
PYODIDE_LIMITS = Limits(6, 12, f"limit of the online version; a local installation lifts it: {REPO_URL}")


# ----------------------------------------------------------------------------- request parsing

def _int(v, what: str) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise InputError(f"{what} must be an integer, got {v!r}")
    return v


def _point(v, what: str = "point") -> tuple[int, int]:
    if not isinstance(v, (list, tuple)) or len(v) != 2:
        raise InputError(f"{what} must be [x, y], got {v!r}")
    return (_int(v[0], what), _int(v[1], what))


def _triangulation(body: dict) -> pg.Triangulation:
    """Corners and triangles of the request -> validated Triangulation (ValueError otherwise;
    messages of Triangulation.validate are passed on verbatim)."""
    if not isinstance(body, dict):
        raise InputError("request body must be a JSON object")
    corners = [_point(p, "corner") for p in body.get("corners") or []]
    tris = body.get("triangles")
    if not isinstance(tris, list) or not tris:
        raise InputError("no triangles: complete the triangulation first")
    triangles = []
    for t in tris:
        if not isinstance(t, (list, tuple)) or len(t) != 3:
            raise InputError(f"a triangle must be three points, got {t!r}")
        triangles.append(tuple(_point(p, "triangle vertex") for p in t))
    pg.lattice_points(corners)  # raises for a degenerate polygon
    T = pg.Triangulation.from_corners_and_triangles(corners, triangles)
    return T.validate()


def _jsonable(geo: dict) -> dict:
    out = {k: v for k, v in geo.items() if k not in ("graph", "coords", "triangles", "triangle_points")}
    out["triangles"] = geo["triangle_points"]
    out["triangles_str"] = geo["triangles"]
    return out


def cli_command(T: pg.Triangulation, order: int, heights=None, gv=False, classes=False, fmt="text",
                nekrasov=False) -> str:
    """The equivalent ``python -m topvertex ...`` command line."""
    from ..cli import _tri_str
    cmd = f'python -m topvertex "{pg.format_polygon(T.corners())}" --triangles "{_tri_str(T)}" --order {order}'
    if heights is not None:
        cmd += " --heights=" + ",".join(str(h) for h in heights)
    if gv:
        cmd += " --gv"
    if classes:
        cmd += " --classes"
    if fmt != "text":
        cmd += f" --format {fmt}"
    if nekrasov:
        cmd += " --nekrasov"
    return cmd


def api_validate(body: dict) -> dict:
    """Validation status, framings (KP tail labels, :func:`topvertex.toric.from_triangulation`) and
    regularity (:func:`topvertex.polygon.heights`) of the triangulation in ``body``."""
    try:
        T = _triangulation(body)
    except (ValueError, KeyError, TypeError) as exc:
        return {"ok": False, "error": str(exc)}
    g = from_triangulation(T)
    tri_pts = [[g.points[i] for i in t] for t in T.triangles]
    h = pg.heights(T)
    edges = [{"name": e.name, "segment": [g.points[i] for i in e.segment], "n": e.n,
              "tail": tri_pts[e.tail], "head": tri_pts[e.head]} for e in g.edges]
    return {"ok": True, "points": g.points, "triangles": tri_pts, "edges": edges,
            "regular": h is not None, "heights": h}


def api_compute(body: dict, limits: Limits | None = SERVER_LIMITS) -> dict:
    """Run the command-line computation for the request ``body`` (see :mod:`topvertex.draw.server`);
    ``limits=None``: no limits.  Raises InputError / ValueError for invalid input."""
    t0 = time.perf_counter()
    T = _triangulation(body)
    order = _int(body.get("order", 4), "order")
    if order < 0:
        raise InputError("order must be >= 0")
    heights = body.get("heights")
    if heights in ("", None, []):
        heights = None
    else:
        if isinstance(heights, str):
            try:
                heights = [int(h) for h in heights.replace(",", " ").split()]
            except ValueError:
                raise InputError(f"heights must be integers, got {body['heights']!r}") from None
        heights = [_int(h, "height") for h in heights]
    fmt = body.get("format", "text")
    if fmt not in FORMATS:
        raise InputError(f"format must be one of {', '.join(FORMATS)}")
    gv, classes = bool(body.get("gv", False)), bool(body.get("classes", False))
    nek = bool(body.get("nekrasov", False))  # a "nekrasov_box" key of old clients is ignored
    if limits is not None:
        if order > limits.order:
            raise InputError(f"order {order} exceeds the limit {limits.order} ({limits.hint})")
        if len(T.points) > limits.points:
            raise InputError(f"{len(T.points)} lattice points exceed the limit {limits.points} ({limits.hint})")
    want_latex = fmt == "mathjax" or bool(body.get("latex", False))
    text_fmt = "text" if fmt == "mathjax" else fmt
    geo = geometry_report(T, heights)
    out = _jsonable(geo)
    rep = series_report(geo, order, gv=gv, classes=classes, fmt=text_fmt, nekrasov=nek, free_energy=gv,
                        objects=True)
    objs = rep.pop("objects")
    out.update(rep)
    if want_latex:
        out["Z_latex"] = mj.series_latex(objs["Z"], "Z")
        out["edges_latex"] = mj.edges_latex(out["edges"])
        if objs["F"] is not None:
            out["F_latex"] = mj.series_latex(objs["F"], "F")
        if objs["gv"] is not None:
            out["gv_latex"] = mj.gv_latex(objs["gv"], objs["F"].variables)
    out["order"] = order
    out["format"] = fmt
    out["command"] = cli_command(T, order, heights, gv, classes, "latex" if fmt == "mathjax" else fmt, nek)
    out["time"] = time.perf_counter() - t0
    return out


def handle(path: str, body, limits: Limits | None = SERVER_LIMITS) -> tuple[int, dict]:
    """(status, answer) for a POST to ``path``: /api/validate always 200; /api/compute 200, 400 for
    invalid input (InputError, Triangulation.validate, bad heights, ...), 500 otherwise; 404 else."""
    if path == "/api/validate":
        return 200, api_validate(body)
    if path != "/api/compute":
        return 404, {"error": f"not found: {path}"}
    try:
        return 200, api_compute(body, limits)
    except (ValueError, KeyError, TypeError) as exc:
        return 400, {"error": str(exc)}
    except Exception as exc:  # pragma: no cover - reported to the page
        traceback.print_exc()
        return 500, {"error": f"{type(exc).__name__}: {exc}"}


def handle_json(path: str, body_json: str, limits: str = "pyodide") -> str:
    """JSON in / JSON out wrapper of :func:`handle` for the Pyodide backend: returns
    ``{"status": int, "body": answer}``.  ``limits``: "pyodide", "server" or "none"."""
    lim = {"pyodide": PYODIDE_LIMITS, "server": SERVER_LIMITS, "none": None}[limits]
    try:
        body = json.loads(body_json)
    except json.JSONDecodeError as exc:
        return json.dumps({"status": 400, "body": {"error": f"invalid JSON: {exc}"}})
    status, answer = handle(path, body, lim)
    return json.dumps({"status": status, "body": answer})
