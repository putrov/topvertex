"""Tests for the graphical input (python -m topvertex --draw): the HTTP API of
topvertex.draw.server and, if firefox is installed, a headless render of the page including its
JavaScript self-test (faces, completeness, framings, Auto-triangulate checked against Python)."""
import json
import shutil
import subprocess
import threading
import urllib.error
import urllib.request
import zlib
from pathlib import Path
from urllib.parse import quote

import pytest

from tests.data import LOCAL_P2_GENUS0
from topvertex import polygon as pg
from topvertex.cli import main
from topvertex.draw.server import MAX_ORDER, make_server, page_html
from topvertex.toric import from_triangulation

PAGE = Path(__file__).resolve().parents[1] / "topvertex" / "draw" / "index.html"
P2_CORNERS = [[-1, -1], [1, 0], [0, 1]]
P2_STAR = [[[-1, -1], [1, 0], [0, 0]], [[1, 0], [0, 1], [0, 0]], [[0, 1], [-1, -1], [0, 0]]]


@pytest.fixture(scope="module")
def server():
    httpd = make_server(0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def post(url, body, ctype="application/json"):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": ctype})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def test_get_page(server):
    with urllib.request.urlopen(server + "/", timeout=10) as r:
        assert r.status == 200
        assert r.headers["Content-Type"] == "text/html; charset=utf-8"
        html = r.read().decode()
    assert "topvertex-draw-page" in html and html.encode() == page_html()


def test_compute_local_p2(server):
    st, res = post(server + "/api/compute", {"corners": P2_CORNERS, "triangles": P2_STAR, "order": 3,
                                             "gv": True, "classes": True, "format": "text"})
    assert st == 200, res
    assert sorted(abs(e["n"]) for e in res["edges"]) == [2, 2, 2]
    rows = {int(l.split()[0]): int(l.split()[1]) for l in res["gv_table"].splitlines()[2:]}
    assert [rows[d] for d in (1, 2, 3)] == LOCAL_P2_GENUS0[:3] == [3, -6, 27]
    assert res["command"].startswith('python -m topvertex "p=-1,-1_1,0_0,1" --triangles "')
    assert "Z" in res and "F" in res and res["time"] >= 0


def test_compute_matches_cli(server, capsys):
    """The page's command reproduces the server's numbers on the command line."""
    st, res = post(server + "/api/compute", {"corners": P2_CORNERS, "triangles": P2_STAR, "order": 2, "gv": True})
    assert st == 200
    cmd = res["command"]
    poly = cmd.split('"')[1]
    tris = cmd.split('"')[3]
    assert main([poly, "--triangles", tris, "--order", "2", "--gv"]) == 0
    out = capsys.readouterr().out
    assert res["gv_table"] in out and res["Z"] in out


def _balanced(tex: str) -> bool:
    depth = 0
    for ch in tex.replace(r"\{", "").replace(r"\}", ""):
        depth += {"{": 1, "}": -1}.get(ch, 0)
        if depth < 0:
            return False
    return depth == 0 and tex.count(r"\begin{") == tex.count(r"\end{")


def test_compute_mathjax_local_p2(server):
    st, res = post(server + "/api/compute", {"corners": P2_CORNERS, "triangles": P2_STAR, "order": 3,
                                             "gv": True, "classes": True, "format": "mathjax"})
    assert st == 200, res
    assert res["format"] == "mathjax" and res["command"].endswith("--format latex")
    assert res["Z"].startswith("Series in t1") and res["gv_table"]          # text fallbacks
    assert len(res["Z_latex"]) == 4 and len(res["F_latex"]) == 3
    assert any(r"\frac" in l for l in res["Z_latex"])
    assert res["Z_latex"][0] == r"[t_{1}^{0}]\,Z = 1"
    assert r"\begin{aligned} {[t_{1}^{3}]\,Z}" in res["Z_latex"][3]          # 17 terms: broken
    gv = res["gv_latex"]
    assert gv.startswith(r"\begin{array}") and r"\begin{tabular}" not in gv
    for row in ("1 & 3 & 0", "2 & -6 & 0", "3 & 27 & -10"):
        assert row in gv
    assert "Q_{0} & (-1,-1)" in res["edges_latex"] and "& -2 &" in res["edges_latex"]
    for tex in res["Z_latex"] + res["F_latex"] + [gv, res["edges_latex"]]:
        assert _balanced(tex), tex
    st, res = post(server + "/api/compute", {"corners": P2_CORNERS, "triangles": P2_STAR, "order": 1})
    assert st == 200 and "Z_latex" not in res                              # only on request


def test_mathjax_helpers():
    from fractions import Fraction
    from topvertex.draw import mathjax as mj
    from topvertex.qring import QFrac
    assert mj.var_latex("t12") == "t_{12}" and mj.var_latex("QB") == "QB"
    assert mj.monomial_latex(("t1", "t2"), (0, 0)) == "t_{1}^{0} t_{2}^{0}"
    assert mj.monomial_latex(("t1", "t2"), (1, 2)) == "t_{1} t_{2}^{2}"
    assert mj.laurent_terms({-1: Fraction(-1, 2), 0: 1, 3: -1, 4: 2}) == [
        r"-\tfrac{1}{2}q^{-1/2}", "+1", "-q^{3/2}", "+2q^{2}"]
    c = QFrac.inv_qnums([1, 1], coeff=-3)
    assert mj.coefficient_latex("X", c) == r"X = \frac{-3}{[1]^{2}}"
    long = QFrac.from_laurent({n: 1 for n in range(-20, 21, 2)})          # 21 terms, no denominator
    tex = mj.coefficient_latex("[t]", long, chunk=8)
    assert tex.startswith(r"\begin{aligned} {[t]} = {}&") and tex.count(r"\\") == 2 and _balanced(tex)


def test_handle_json_pyodide_entry_point():
    """The function the Pyodide backend calls: same answer as the server, online limits."""
    from topvertex.draw.compute import PYODIDE_LIMITS, REPO_URL, handle_json
    body = {"corners": P2_CORNERS, "triangles": P2_STAR, "order": 2, "gv": True, "format": "mathjax"}
    out = json.loads(handle_json("/api/compute", json.dumps(body)))
    assert out["status"] == 200 and out["body"]["gv_latex"] and out["body"]["Z_latex"]
    out = json.loads(handle_json("/api/compute", json.dumps(dict(body, order=PYODIDE_LIMITS.order + 1))))
    assert out["status"] == 400 and REPO_URL in out["body"]["error"]
    out = json.loads(handle_json("/api/validate", json.dumps(body)))
    assert out["status"] == 200 and out["body"]["ok"]
    assert json.loads(handle_json("/api/compute", "{bad"))["status"] == 400


def test_handle_json_nekrasov_quiver():
    """A linear quiver is compared up to the order; a stray "nekrasov_box" key of an old client is
    ignored, and the returned CLI command carries a plain --nekrasov.  Pure SU(N) (local F0) goes
    through the same comparison."""
    from topvertex.draw.compute import handle_json
    corners = [[0, 0], [2, 0], [2, 1], [1, 2], [0, 1]]  # SU(2) with 1 + 1 flavours
    T = pg.all_unimodular_triangulations([tuple(c) for c in corners])[0]
    tris = [[list(T.points[i]) for i in t] for t in T.triangles]
    body = {"corners": corners, "triangles": tris, "order": 3, "nekrasov": True, "nekrasov_box": [9, 9]}
    out = json.loads(handle_json("/api/compute", json.dumps(body)))
    assert out["status"] == 200, out
    res = out["body"]
    assert "linear quiver along (0, 1): U(2)" in res["nekrasov"] and "MATCH for all" in res["nekrasov"]
    assert res["heights"] == pg.heights(T)
    assert res["command"].endswith(" --order 3 --nekrasov")
    f0 = {"corners": [[1, 0], [0, 1], [-1, 0], [0, -1]], "order": 2, "nekrasov": True, "nekrasov_box": "junk",
          "triangles": [[[1, 0], [0, 1], [0, 0]], [[0, 1], [-1, 0], [0, 0]], [[-1, 0], [0, -1], [0, 0]],
                        [[0, -1], [1, 0], [0, 0]]]}
    out = json.loads(handle_json("/api/compute", json.dumps(f0)))
    assert out["status"] == 200, out
    assert "U(2), no fundamentals, CS level kappa = 0" in out["body"]["nekrasov"]
    assert "MATCH for all" in out["body"]["nekrasov"] and out["body"]["command"].endswith(" --nekrasov")
    out = json.loads(handle_json("/api/compute", json.dumps(dict(f0, nekrasov=False))))
    assert out["status"] == 200 and "nekrasov" not in out["body"] and "--nekrasov" not in out["body"]["command"]


def test_validate(server):
    st, res = post(server + "/api/validate", {"corners": P2_CORNERS, "triangles": P2_STAR})
    assert st == 200 and res["ok"] and res["regular"]
    assert sorted(e["n"] for e in res["edges"]) == [-2, 2, 2]
    st, res = post(server + "/api/validate", {"corners": P2_CORNERS, "triangles": P2_STAR[:2]})
    assert st == 200 and not res["ok"] and "not covering" in res["error"]


def test_invalid_triangles(server):
    T = pg.Triangulation.from_corners_and_triangles([tuple(p) for p in P2_CORNERS],
                                                    [tuple(map(tuple, t)) for t in P2_STAR[:2]])
    with pytest.raises(ValueError) as exc:
        T.validate()
    st, res = post(server + "/api/compute", {"corners": P2_CORNERS, "triangles": P2_STAR[:2], "order": 2})
    assert st == 400 and res["error"] == str(exc.value)
    st, res = post(server + "/api/compute", {"corners": P2_CORNERS, "triangles": [[[-1, -1], [1, 0], [0, 1]]],
                                             "order": 2})
    assert st == 400 and "non-unimodular" in res["error"]
    st, res = post(server + "/api/compute", {"corners": P2_CORNERS, "triangles": [[[5, 5], [1, 0], [0, 1]]]})
    assert st == 400 and "not a lattice point" in res["error"]
    st, res = post(server + "/api/compute", {"corners": P2_CORNERS, "triangles": P2_STAR, "heights": [0, 1, 1, 1]})
    assert st == 400 and res["error"].startswith("bad heights")
    st, res = post(server + "/api/compute", {"corners": "nonsense", "triangles": P2_STAR})
    assert st == 400


def test_guards(server):
    st, res = post(server + "/api/compute", {"corners": P2_CORNERS, "triangles": P2_STAR, "order": MAX_ORDER + 1})
    assert st == 400 and "--unsafe" in res["error"]
    T = pg.Triangulation.from_corners_and_triangles(  # standard triangulation, 36 lattice points
        [(0, 0), (7, 0), (0, 7)],
        [((x, y), (x + 1, y), (x, y + 1)) for x in range(7) for y in range(7 - x)]
        + [((x + 1, y), (x + 1, y + 1), (x, y + 1)) for x in range(7) for y in range(6 - x)])
    tris = [[list(T.points[i]) for i in t] for t in T.triangles]
    st, res = post(server + "/api/compute", {"corners": [[0, 0], [7, 0], [0, 7]], "triangles": tris, "order": 1})
    assert st == 400 and "36 lattice points exceed the limit" in res["error"]
    st, res = post(server + "/api/compute", {"corners": P2_CORNERS, "triangles": P2_STAR}, ctype="text/plain")
    assert st == 415


def test_host_check(server):
    port = server.rsplit(":", 1)[1]
    req = urllib.request.Request(server + "/", headers={"Host": f"evil.example:{port}"})
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req, timeout=10)
    assert exc.value.code == 403


def test_cli_requires_polygon_without_draw():
    with pytest.raises(SystemExit):
        main(["--order", "2"])


# ----------------------------------------------------------------------------- headless firefox

FIREFOX = shutil.which("firefox")


def _screenshot(url: str, out: Path, tmp: Path, size="1400,1100"):
    prof = tmp / "ffprofile"
    prof.mkdir(exist_ok=True)
    subprocess.run([FIREFOX, "--headless", "--no-remote", "--profile", str(prof), f"--window-size={size}",
                    "--screenshot", str(out), url], capture_output=True, timeout=120)
    assert out.exists() and out.stat().st_size > 1000
    return out


def _first_pixel(png: Path) -> tuple[int, int, int]:
    """RGB of pixel (0, 0) of an 8-bit non-interlaced PNG.  The first pixel of the first row is
    stored unfiltered by every PNG filter type (no left / upper neighbours)."""
    data = png.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat = 8, b""
    while pos < len(data):
        n = int.from_bytes(data[pos:pos + 4], "big")
        kind = data[pos + 4:pos + 8]
        if kind == b"IHDR":
            assert data[pos + 16] == 8 and data[pos + 20] == 0  # bit depth 8, not interlaced
        if kind == b"IDAT":
            idat += data[pos + 8:pos + 8 + n]
        pos += 12 + n
    raw = zlib.decompress(idat)
    return tuple(raw[1:4])


def _selftest_cases():
    polys = [[(-1, -1), (1, 0), (0, 1)],                 # local P^2
             [(1, 0), (0, 1), (-1, 0), (0, -1)],          # local F0
             [(0, 0), (0, 2), (-1, 1), (1, 0)],           # local F1
             [(0, 0), (1, 0), (1, 1), (0, 1)],            # conifold
             [(0, 0), (4, 0), (0, 2)]]                    # several triangulations
    cases = []
    for corners in polys:
        for T in pg.all_unimodular_triangulations(corners)[:3]:
            g = from_triangulation(T)
            tri = [[list(T.points[i]) for i in t] for t in T.triangles]
            fr = [{"segment": [list(g.points[i]) for i in e.segment], "tail": tri[e.tail], "n": e.n} for e in g.edges]
            cases.append({"corners": [list(c) for c in corners], "triangles": tri, "framings": fr})
    return cases


@pytest.mark.skipif(FIREFOX is None, reason="firefox not installed")
def test_page_static_render(tmp_path):
    png = _screenshot(PAGE.as_uri() + "#p=-1,-1_1,0_0,1", tmp_path / "draw.png", tmp_path)
    assert _first_pixel(png) != (0, 0, 0)


@pytest.mark.skipif(FIREFOX is None, reason="firefox not installed")
def test_page_js_selftest(tmp_path):
    """The page's JS geometry against Python: faces, completeness, framings (Konishi rule),
    Auto-triangulate and the corner-string parser, then synthetic mouse events (add/remove corners, draw,
    flip and remove diagonals); the page paints itself green iff all checks pass."""
    cases = _selftest_cases()
    url = PAGE.as_uri() + "#selftest=" + quote(json.dumps(cases, separators=(",", ":")), safe="")
    r, g, b = _first_pixel(_screenshot(url, tmp_path / "selftest.png", tmp_path))
    assert (r, g, b) == (0, 192, 0), "JS self-test failed (red page); open the URL in a browser for details"
