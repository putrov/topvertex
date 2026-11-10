"""Command line interface: ``python -m topvertex POLYGON [options]``.

POLYGON is given as a compact corner string (``p=0,0_3,0_0,3``, the bare list ``0,0_3,0_0,3``, or
a URL with a ``p=`` parameter; see :func:`topvertex.polygon.parse_polygon`).  Examples::

    python -m topvertex "p=0,0_3,0_0,3" --list-triangulations
    python -m topvertex "p=1,0_0,1_-1,-1" --order 4 --gv
    python -m topvertex "p=0,0_2,0_0,2" --pick 0 --order 3 --gv --format mathematica
    python -m topvertex "p=1,0_0,1_-1,0_0,-1" --order 4 --nekrasov               # local F0: pure SU(2)
    python -m topvertex "p=0,0_2,0_2,1_1,2_0,1" --pick 0 --order 4 --nekrasov   # SU(2), 1 + 1 flavours

The triangulation is taken from ``--triangles`` ("x,y x,y x,y; ..."), else the unique unimodular
triangulation of the polygon, else ``--pick k`` (index in the ``--list-triangulations`` output).
``--nekrasov``: if the triangulation is a ladder along some lattice direction (every triangle has a
side parallel to it, :func:`topvertex.nekrasov.quiver_structure`: a linear quiver
U(N_1) x ... x U(N_r) with fundamentals and CS levels; pure SU(N) is the case of one node without
fundamentals), check Z_top = Z_pert Z_Nekrasov for all curve classes of Kahler weight <= ORDER
(:func:`topvertex.nekrasov.compare_with_nekrasov`) and print the dictionary; otherwise print that the
geometry is not recognised.

``--draw [--port P] [--no-browser] [--unsafe]``: draw the polygon and its triangulation with the
mouse in a browser page served on 127.0.0.1 (:mod:`topvertex.draw.server`); POLYGON is then
optional and, if given, preloaded in the page.

The computation behind the printed output is split into :func:`geometry_report` (graph, framings,
weights, classes) and :func:`series_report` (Z, GV table, Nekrasov), shared with the web server.
"""
from __future__ import annotations

import argparse
import sys

from . import polygon as pg
from .partition_function import class_series, default_weights, gv_invariants, gv_table, partition_function
from .toric import class_coordinates, from_triangulation, kahler_weights, relation_lattice_basis

__all__ = ["main", "parse_triangles", "InputError", "geometry_report", "series_report"]


class InputError(ValueError):
    """Invalid user input (bad heights, order, ...): the CLI turns it into a usage error,
    the web server into HTTP 400."""


def parse_triangles(s: str) -> list[tuple[tuple[int, int], ...]]:
    """'0,0 1,0 0,1; 1,0 1,1 0,1' -> [((0,0),(1,0),(0,1)), ((1,0),(1,1),(0,1))]."""
    tris = []
    for chunk in s.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        pts = [tuple(int(v) for v in p.split(",")) for p in chunk.split()]
        if len(pts) != 3 or any(len(p) != 2 for p in pts):
            raise ValueError(f"bad triangle {chunk!r}: need three points x,y")
        tris.append(tuple(pts))
    return tris


def _fmt_series(S, fmt: str) -> str:
    if fmt == "sympy":
        return str(S.to_sympy())
    if fmt == "mathematica":
        return S.to_mathematica()
    if fmt == "latex":
        return S.latex()
    return str(S)


def _tri_str(T) -> str:
    return "; ".join(" ".join(f"{x},{y}" for x, y in (T.points[i] for i in t)) for t in T.triangles)


def geometry_report(T, heights=None) -> dict:
    """GT graph data of the triangulation ``T`` (:func:`topvertex.toric.from_triangulation`):
    framings n_e (Konishi tail labels), Kahler weights w_e = <h, l_e>
    (:func:`topvertex.partition_function.default_weights`; ``heights`` in lattice_points order) and
    curve-class coordinates in :func:`topvertex.toric.relation_lattice_basis` (None if unavailable).
    ``heights`` in the result are the heights actually used (the given ones, else
    :func:`topvertex.polygon.heights`; None if the triangulation has no internal edges).

    Returns a dict with JSON-friendly entries ``points``, ``triangles`` (CLI string),
    ``triangle_points``, ``edges`` (list of {name, segment, n, weight, class, tail, head}; ``tail``
    / ``head`` are the triangles at the ends of the edge, n is the framing at the tail), ``basis``,
    ``heights``, and the objects ``graph``, ``weights``, ``coords``.  Raises InputError for bad heights."""
    g = from_triangulation(T)
    used = heights
    if used is None and g.edges:
        used = pg.heights(T)  # None if not regular: default_weights then raises
    try:
        weights = kahler_weights(g, heights) if heights is not None else default_weights(g, used)
    except ValueError as exc:
        raise InputError(f"bad heights: {exc}") from None
    coords = basis = None
    if g.edges:
        try:
            basis = relation_lattice_basis(g)
            coords = class_coordinates(g, basis)
        except ValueError:
            basis = coords = None
    tri_pts = [[g.points[i] for i in t] for t in T.triangles]
    edges = []
    for k, e in enumerate(g.edges):
        edges.append({"name": e.name, "segment": [g.points[i] for i in e.segment], "n": e.n,
                      "weight": weights[k], "class": coords[k] if coords is not None else None,
                      "tail": tri_pts[e.tail], "head": tri_pts[e.head]})
    return {"points": g.points, "triangles": _tri_str(T), "triangle_points": tri_pts, "edges": edges,
            "basis": basis, "heights": list(used) if used is not None else None, "graph": g, "weights": weights,
            "coords": coords}


def series_report(geo: dict, order: int, gv: bool = False, classes: bool = False, fmt: str = "text",
                  nekrasov: bool = False, free_energy: bool = False,
                  objects: bool = False) -> dict:
    """Z truncated at sum_e w_e |lam_e| <= order (:func:`topvertex.partition_function.partition_function`)
    for the output of :func:`geometry_report`, as text in format ``fmt``; in class variables if
    ``classes``.  With ``gv``: the GV table of F = log Z in class variables
    (:func:`topvertex.partition_function.gv_invariants`, :func:`gv_table`), key ``gv_table``
    (absent if there are no curve classes); with ``free_energy`` also the text of F, key ``F``.
    With ``nekrasov``: key ``nekrasov`` with :func:`topvertex.nekrasov.format_comparison` of
    :func:`topvertex.nekrasov.compare_with_nekrasov` (Z_top = Z_pert Z_Nekrasov for all classes of
    weight <= order, reusing Z) if the geometry is a linear quiver
    (:func:`topvertex.nekrasov.quiver_structure`), else a note that it is not recognised.
    With ``objects``: key ``objects`` = {Z: the printed Series, F: Series or None, gv: dict or None}
    (used by the web server for its LaTeX output)."""
    g, coords, basis = geo["graph"], geo["coords"], geo["basis"]
    Z = partition_function(g, order, geo["weights"])
    out_series, Zc, out = Z, None, {}
    if (classes or gv) and coords is not None:
        Zc, _, _ = class_series(g, Z, basis)
        if classes:
            out_series = Zc
    out["Z"] = _fmt_series(out_series, fmt)
    F = gvd = None
    if gv and Zc is not None:
        F = Zc.log()
        if free_energy:
            out["F"] = _fmt_series(F, fmt)
        gvd = gv_invariants(F)
        out["gv_table"] = gv_table(gvd, variables=Zc.variables)
    if objects:
        out["objects"] = {"Z": out_series, "F": F, "gv": gvd}
    if nekrasov:
        from .nekrasov import compare_with_nekrasov, format_comparison, quiver_structure
        if quiver_structure(g.triangulation) is None:
            out["nekrasov"] = ("Nekrasov comparison: geometry not recognised as a linear quiver "
                               "(no lattice direction parallel to a side of every triangle)")
        else:
            out["nekrasov"] = format_comparison(compare_with_nekrasov(g, order, heights=geo["heights"], Z=Z))
    return out


def main(argv=None) -> int:
    """Entry point of ``python -m topvertex`` (see module docstring)."""
    ap = argparse.ArgumentParser(prog="python -m topvertex",
                                 description="Topological vertex partition function and GV invariants "
                                             "of the toric CY3 of a triangulated lattice polygon "
                                             "(Konishi conventions).")
    ap.add_argument("polygon", nargs="?",
                    help="polygon corners in compact corner format, e.g. 'p=0,0_3,0_0,3' (optional with --draw)")
    ap.add_argument("--triangles", help="triangles as 'x,y x,y x,y; ...'")
    ap.add_argument("--list-triangulations", action="store_true", help="list unimodular triangulations and exit")
    ap.add_argument("--pick", type=int, help="index of the triangulation to use")
    ap.add_argument("--order", type=int, default=3, help="truncation: sum_e w_e |lam_e| <= ORDER")
    ap.add_argument("--gv", action="store_true", help="print the Gopakumar-Vafa table")
    ap.add_argument("--classes", action="store_true", help="print Z in curve-class variables t_i")
    ap.add_argument("--format", choices=["sympy", "mathematica", "latex", "text"], default="text")
    ap.add_argument("--heights",
                    help="integer heights h0,h1,... of the lattice points, in the order of the 'points:' line. "
                         "The triangles must be the projections of the lower faces of the convex hull of the "
                         "lifted points (x, y, h). The heights fix the weights w_e of the internal edges "
                         "(default: found automatically)")
    ap.add_argument("--nekrasov", action="store_true",
                    help="compare with the 5d Nekrasov partition function of the linear quiver (when every "
                         "triangle has a side parallel to one lattice direction; pure SU(N) included) for "
                         "all curve classes of weight <= ORDER")
    ap.add_argument("--draw", action="store_true",
                    help="draw the triangulation in a browser page served on 127.0.0.1 (graphical input)")
    ap.add_argument("--port", type=int, default=None,
                    help="port for --draw (default: 8765, or the next free port up to 8775)")
    ap.add_argument("--no-browser", action="store_true", help="with --draw: only print the URL")
    ap.add_argument("--unsafe", action="store_true",
                    help="with --draw: lift the server limits (order <= 12, <= 30 lattice points)")
    args = ap.parse_args(argv)

    if args.draw:
        from .draw.server import serve
        if args.polygon is not None:
            try:
                pg.lattice_points(pg.parse_polygon(args.polygon))
            except ValueError as exc:
                ap.error(f"bad polygon: {exc}")
        ports = [args.port] if args.port is not None else list(range(8765, 8776))
        for i, port in enumerate(ports):
            try:
                serve(port=port, open_browser=not args.no_browser, unsafe=args.unsafe, polygon=args.polygon)
                return 0
            except OSError as exc:
                if i == len(ports) - 1:
                    ap.error(f"cannot listen on 127.0.0.1:{port}: {exc}")
                print(f"port {port} is in use, trying {ports[i + 1]}", file=sys.stderr)
        return 0
    if args.polygon is None:
        ap.error("the polygon argument is required (unless --draw)")

    corners = pg.parse_polygon(args.polygon)
    pts = pg.lattice_points(corners)
    if args.list_triangulations:
        tris = pg.all_unimodular_triangulations(corners)
        print(f"lattice points: {pts}")
        print(f"{len(tris)} unimodular triangulation(s):")
        for k, T in enumerate(tris):
            reg = "regular" if pg.is_regular(T) else "NOT regular"
            print(f"[{k}] {_tri_str(T)}   ({reg})")
        return 0
    if args.triangles:
        T = pg.Triangulation.from_corners_and_triangles(corners, parse_triangles(args.triangles))
    else:
        tris = pg.all_unimodular_triangulations(corners)
        if args.pick is not None:
            if not 0 <= args.pick < len(tris):
                ap.error(f"--pick must be in 0..{len(tris) - 1}")
            T = tris[args.pick]
        elif len(tris) == 1:
            T = tris[0]
        else:
            ap.error(f"polygon has {len(tris)} unimodular triangulations; use --pick k or --triangles "
                     "(see --list-triangulations)")
    heights = [int(h) for h in args.heights.split(",")] if args.heights else None
    try:
        geo = geometry_report(T, heights)
    except InputError as exc:
        ap.error(str(exc))

    print(f"points: {geo['points']}")
    print(f"triangles: {geo['triangles']}")
    print("internal edges (segment: framing n_e, truncation weight w_e, H2 class):")
    for row in geo["edges"]:
        a, b = row["segment"]
        cls = f", class {row['class']}" if row["class"] is not None else ""
        print(f"  {row['name']}: {a}-{b}  n = {row['n']}, w = {row['weight']}{cls}")
    if geo["basis"] is not None:
        print(f"class basis (relation vectors): {geo['basis']}")
    res = series_report(geo, args.order, gv=args.gv, classes=args.classes, fmt=args.format,
                        nekrasov=args.nekrasov)
    print(f"Z (truncated at weight {args.order}):")
    print(res["Z"])
    if args.gv:
        if "gv_table" not in res:
            print("no curve classes: GV invariants unavailable")
        else:
            print("Gopakumar-Vafa invariants n^g_beta (physical convention):")
            print(res["gv_table"])
    if args.nekrasov:
        print(res["nekrasov"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
