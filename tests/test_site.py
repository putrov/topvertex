"""tools/build_site.py: the static "try it online" site (page + wheel + manifest)."""
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    out = tmp_path_factory.mktemp("site")
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "build_site.py"), "--out", str(out)],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    return out


def test_site_files(site):
    page = (ROOT / "topvertex" / "draw" / "index.html").read_text(encoding="utf-8")
    assert (site / "index.html").read_text(encoding="utf-8") == page
    assert (site / ".nojekyll").exists()
    if (ROOT / "notes" / "topvertex_notes.pdf").exists():
        assert (site / "topvertex_notes.pdf").stat().st_size > 0


def test_site_manifest_and_wheel(site):
    man = json.loads((site / "manifest.json").read_text())
    version = re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(), re.M).group(1)
    assert man["version"] == version and man["wheel"] == f"topvertex-{version}-py3-none-any.whl"
    whl = site / man["wheel"]
    assert hashlib.sha256(whl.read_bytes()).hexdigest() == man["sha256"]
    page = (site / "index.html").read_text(encoding="utf-8")
    assert f'const PYODIDE_VERSION = "{man["pyodide"]}"' in page
    names = zipfile.ZipFile(whl).namelist()
    for f in ("topvertex/draw/index.html", "topvertex/draw/compute.py", "topvertex/draw/mathjax.py",
              "topvertex/cli.py"):
        assert f in names
    assert len(list(site.glob("topvertex-*.whl"))) == 1
