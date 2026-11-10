#!/usr/bin/env python3
"""Build the static "try it online" site (GitHub Pages from ``docs/``).

    python tools/build_site.py [--out docs]

Writes into the output directory (default ``docs/`` at the repository root):

* ``index.html``  -- copy of ``topvertex/draw/index.html``; served by a static host the page finds no
  topvertex server and computes in the browser with Pyodide (see the BACKENDS block of the page);
* ``topvertex-<version>-py3-none-any.whl`` -- the package, built with ``pip wheel . --no-deps``
  from a temporary copy of the sources (so no ``build/`` directory is left in the repository);
  older ``topvertex-*.whl`` files in the output directory are removed;
* ``manifest.json`` -- ``{"wheel": <file name>, "version": <version>, "sha256": <hash>, "pyodide": <version>}``,
  read by the page to install the wheel;
* ``topvertex_notes.pdf`` -- copy of ``notes/topvertex_notes.pdf`` if it exists;
* ``.nojekyll`` -- GitHub Pages: serve the files as they are.

Test locally with ``python -m http.server -d docs 8000`` and open http://127.0.0.1:8000/ (not the
topvertex server, which would take precedence over Pyodide).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ["pyproject.toml", "README.md", "LICENSE", "topvertex"]


def pyodide_version(page: str) -> str | None:
    m = re.search(r'const PYODIDE_VERSION = "([^"]+)"', page)
    return m.group(1) if m else None


def build_wheel(out: Path) -> Path:
    """pip wheel of the package (pure Python), built from a temporary copy of the sources."""
    with tempfile.TemporaryDirectory(prefix="topvertex-site-") as tmp:
        src = Path(tmp) / "src"
        src.mkdir()
        for name in SOURCES:
            p = ROOT / name
            if p.is_dir():
                shutil.copytree(p, src / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            elif p.exists():
                shutil.copy2(p, src / name)
        dist = Path(tmp) / "dist"
        r = subprocess.run([sys.executable, "-m", "pip", "wheel", str(src), "--no-deps", "-q", "-w", str(dist)],
                           capture_output=True, text=True)
        if r.returncode:
            sys.stderr.write(r.stdout + r.stderr)
            raise RuntimeError("pip wheel failed (see the output above)")
        (whl,) = dist.glob("topvertex-*-py3-none-any.whl")
        for old in out.glob("topvertex-*.whl"):
            old.unlink()
        return Path(shutil.copy2(whl, out / whl.name))


def build_site(out: Path) -> dict:
    """Build the site into ``out``; returns the manifest."""
    out.mkdir(parents=True, exist_ok=True)
    page = (ROOT / "topvertex" / "draw" / "index.html").read_text(encoding="utf-8")
    (out / "index.html").write_text(page, encoding="utf-8")
    whl = build_wheel(out)
    version = whl.name.split("-")[1]
    manifest = {"wheel": whl.name, "version": version,
                "sha256": hashlib.sha256(whl.read_bytes()).hexdigest(), "pyodide": pyodide_version(page)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    notes = ROOT / "notes" / "topvertex_notes.pdf"
    if notes.exists():
        shutil.copy2(notes, out / notes.name)
    (out / ".nojekyll").write_text("", encoding="utf-8")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "docs", help="output directory (default: docs/)")
    args = ap.parse_args(argv)
    man = build_site(args.out)
    print(f"built {args.out}: {man['wheel']} (Pyodide {man['pyodide']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
