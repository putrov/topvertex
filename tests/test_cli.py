"""Tests for the command line interface (python -m topvertex)."""
import subprocess
import sys
from pathlib import Path

import pytest

from tests.data import LOCAL_P2_GENUS0
from topvertex.cli import main, parse_triangles

ROOT = Path(__file__).resolve().parents[1]


def gv_rows(out: str) -> dict[int, list[int]]:
    lines = out.splitlines()
    k = next(i for i, l in enumerate(lines) if l.startswith("Gopakumar-Vafa"))
    rows = {}
    for l in lines[k + 3:]:
        parts = l.split()
        rows[int(parts[0])] = [int(x) for x in parts[1:]]
    return rows


def test_cli_subprocess_local_p2():
    r = subprocess.run([sys.executable, "-m", "topvertex", "p=1,0_0,1_-1,-1", "--order", "3", "--gv"],
                       capture_output=True, text=True, cwd=ROOT, timeout=120)
    assert r.returncode == 0, r.stderr
    rows = gv_rows(r.stdout)
    assert [rows[d][0] for d in (1, 2, 3)] == LOCAL_P2_GENUS0[:3]
    assert "n = -2" in r.stdout and "n = 2" in r.stdout


def test_cli_list_and_pick(capsys):
    assert main(["p=0,0_2,0_0,2", "--list-triangulations"]) == 0
    out = capsys.readouterr().out
    assert "4 unimodular triangulation(s)" in out and "[3]" in out
    with pytest.raises(SystemExit):
        main(["p=0,0_2,0_0,2", "--order", "1"])  # ambiguous without --pick
    capsys.readouterr()
    assert main(["p=0,0_2,0_0,2", "--pick", "2", "--order", "2", "--gv", "--classes", "--format", "mathematica"]) == 0
    out = capsys.readouterr().out
    assert "Gopakumar-Vafa" in out and "q^(1/2)" in out


def test_cli_triangles_and_formats(capsys):
    tris = "0,0 1,0 0,1; 1,0 1,1 0,1"
    assert parse_triangles(tris) == [((0, 0), (1, 0), (0, 1)), ((1, 0), (1, 1), (0, 1))]
    for fmt in ("sympy", "latex", "text"):
        assert main(["p=0,0_1,0_1,1_0,1", "--triangles", tris, "--order", "2", "--gv", "--format", fmt]) == 0
    out = capsys.readouterr().out
    assert "n = 0" in out
    # lattice points order: (-1,-1), (0,0), (0,1), (1,0); heights give w = 3 per line
    assert main(["p=1,0_0,1_-1,-1", "--order", "6", "--heights", "1,0,1,1", "--gv"]) == 0
    out = capsys.readouterr().out
    assert "w = 3" in out and gv_rows(out)[2][0] == LOCAL_P2_GENUS0[1]
    with pytest.raises(SystemExit):
        main(["p=1,0_0,1_-1,-1", "--heights", "0,1,1,1"])  # non-positive weights
