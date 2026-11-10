# topvertex

**topvertex** computes the all-genus topological string partition function of a toric
Calabi-Yau threefold with the **topological vertex** of Aganagic-Klemm-Mariño-Vafa
([hep-th/0305132](https://arxiv.org/abs/hep-th/0305132)).

* **Input:** a lattice polygon (the toric diagram) with a unimodular regular triangulation,
  or equivalently its dual trivalent (p,q)-web. The diagram can be typed on the command line
  or drawn with the mouse in a browser page.
* **Output:** the series `Z(q, Q_e)` with exact coefficients in `Q(q^{1/2})`, the free energy
  `F = log Z`, the curve classes and framings of the edges, and the Gopakumar-Vafa integers
  `n^g_β`. There is one variable `Q_e` per internal edge, or one variable per curve class.
* **Cross-check:** for the "ladder" polygons that engineer 5d gauge theories the result is
  compared with the Nekrasov instanton partition function. These are linear quivers
  U(N_1)×…×U(N_r) with bifundamentals, fundamental flavors and Chern-Simons levels, pure SU(N)
  included.

The conventions follow Konishi ([math/0504188](https://arxiv.org/abs/math/0504188)). They are
explained from scratch, with examples worked out by hand, in the accompanying notes
[`notes/topvertex_notes.pdf`](notes/topvertex_notes.pdf) (36 pages). The notes are
self-contained, and no background in string theory or toric geometry is assumed. The file
`CONVENTIONS.md` is a short technical summary of the same conventions.

---

## Quick start

The requirements are `git` and Python 3.11 or newer. Copy the block for the operating system
into a terminal. The last command opens the drawing page in the browser.

**macOS / Linux**

```bash
git clone https://github.com/putrov/topvertex.git
cd topvertex
python3 -m venv .venv
.venv/bin/pip install -e ".[fast]"
.venv/bin/topvertex --draw
```

**Windows** (PowerShell, and the same lines work unchanged in the classic `cmd` prompt)

```bat
git clone https://github.com/putrov/topvertex.git
cd topvertex
py -3 -m venv .venv
.venv\Scripts\pip install -e ".[fast]"
.venv\Scripts\topvertex --draw
```

Then click lattice points to draw a polygon, press **Auto-triangulate** and press **Compute**.
The result is typeset with MathJax. The **format** menu offers plain text, LaTeX source and
Mathematica as alternatives. To try a ready-made example instead, replace the last line by
`.venv/bin/topvertex "p=-1,-1_1,0_0,1" --order 5 --gv` (local P²). The sections below explain
the installation in more detail and describe every option.

### Try it online (no installation)

<https://putrov.github.io/topvertex/> is the same drawing
page, running the package inside the browser with [Pyodide](https://pyodide.org) (Python
compiled to WebAssembly). Nothing is sent to a server, and the computation happens on the
machine of the visitor. The first **Compute** downloads about 15 MB (Pyodide and sympy, cached
afterwards) and takes a few seconds, while later computations are quick. The online version is
limited to order ≤ 6 and to polygons with ≤ 12 lattice points. The installed package has no
such limits and uses the faster native arithmetic.

---

## 1. Installation

One needs **Python 3.11 or newer** (CPython, not PyPy) and `git`. Everything else is installed
by `pip` from pre-built wheels. No compiler is needed on Linux, macOS (Intel and Apple
silicon) or Windows.

### 1.1 Get the code

```bash
git clone https://github.com/putrov/topvertex.git
cd topvertex
```

### 1.2 Create a virtual environment and install

A virtual environment keeps the package and its dependencies separate from the system Python.

**Linux / macOS** (bash, zsh, or csh/tcsh, since the commands below do not need `activate`)

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[full,test]"
```

If `python3` is older than 3.11 (check with `python3 --version`), install a newer one first.
For example, one can take it from [python.org](https://www.python.org/downloads/), use
Homebrew on macOS (`brew install python@3.12`), or use conda
(`conda create -n topvertex python=3.12`, and then that environment's `python` in place of
`python3`).

**Windows** (PowerShell or cmd)

```bat
py -3 -m venv .venv
.venv\Scripts\pip install -e ".[full,test]"
```

The extras have the following meaning:

| extra | adds |
|---|---|
| `fast` | python-flint (fast exact polynomial arithmetic) and scipy (linear programming, used to find the heights of Section 2) |
| `full` | the same two packages and matplotlib (plots) |
| `test` | pytest |

The only core dependency is sympy (>= 1.13). The packages python-flint and scipy are optional
but recommended. Without them the program uses slower pure-Python fallbacks with identical
results (set `TOPVERTEX_NO_FLINT=1` or `TOPVERTEX_NO_SCIPY=1` to force the fallbacks).

### 1.3 Check that it works

```bash
.venv/bin/pytest -q -m "not slow"          # Linux/macOS, about 10 s
.venv\Scripts\pytest -q -m "not slow"      # Windows
```

### 1.4 Running the program

The installation creates a command `topvertex` inside the virtual environment. One can call it
with its full path:

```bash
.venv/bin/topvertex --help            # Linux/macOS
.venv\Scripts\topvertex --help        # Windows
```

Alternatively, one can activate the environment once per terminal session, so that plain
`topvertex` and `python -m topvertex` work:

| shell | command |
|---|---|
| bash / zsh | `source .venv/bin/activate` |
| csh / tcsh | `source .venv/bin/activate.csh` |
| fish | `source .venv/bin/activate.fish` |
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |
| Windows cmd | `.venv\Scripts\activate.bat` |

A third option is a symlink to `.venv/bin/topvertex` in a directory on the `PATH` (for example
`~/.local/bin`). In the rest of this file `topvertex` means any of these, and
`python -m topvertex` is always equivalent.

---

## 2. Describing the geometry

A toric Calabi-Yau threefold is specified by a **convex lattice polygon** (with corners at
points with integer coordinates) together with a **triangulation into triangles of area 1/2**
that uses all lattice points of the polygon. Such a triangulation is called unimodular.

### 2.1 The polygon

The polygon is given by its **corners** in a compact format: the string `p=` followed by the
corners, with the two coordinates of a corner separated by `,` and the corners separated by `_`.

| geometry | corners | lattice points |
|---|---|---|
| resolved conifold | `p=0,0_1,0_1,1_0,1` | 4 |
| local P² | `p=-1,-1_1,0_0,1` | 4 (one interior point) |
| local F0 = P¹×P¹ | `p=1,0_0,1_-1,0_0,-1` | 5 |
| local F1 | `p=1,0_0,1_-1,1_0,-1` | 5 |
| local dP3 | `p=1,0_0,1_-1,1_-1,0_0,-1_1,-1` | 7 |

The order of the corners does not matter, and the other lattice points of the polygon are
found automatically. A URL that contains such a `p=...` parameter, for example the address of
the drawing page of Section 4, is accepted too.

### 2.2 The triangulation

The triangulation can be given in four ways:

1. not at all, if the polygon has exactly one unimodular triangulation (the conifold has two,
   local P² and local F0 have one), in which case it is chosen automatically,
2. with `--pick k` after looking at the output of `--list-triangulations`,
3. explicitly with `--triangles "x,y x,y x,y; x,y x,y x,y; ..."` (one triangle per
   `;`-separated group, vertices in any order),
4. graphically with `--draw` (Section 4).

### 2.3 Heights, regular triangulations and weights

The triangulation must be *regular*. To state this condition, assign an integer `h_i`, called
a **height**, to every lattice point `ω_i` of the polygon, and lift the points to
`(ω_i, h_i)` in R³. The faces of the convex hull of the lifted points that are visible from
below project to a subdivision of the polygon. A triangulation is regular if it is obtained
in this way for some choice of the heights, which are then called *heights of the
triangulation*. For example, the triangulation of the local P² polygon into three triangles is
obtained from the height 0 at the interior point and 1 at the three corners. The program
checks regularity and reports an error otherwise.

The heights are used for one more purpose. Each internal edge `e` of the dual web corresponds
to a segment shared by two triangles. It gets a positive integer **weight** `w_e`, which
measures how much the lifted surface bends along this segment. Namely, `w_e` is the height of
the lifted opposite corner of one triangle above the plane through the lift of the other
triangle. The weights are the lengths of the internal edges of the dual web, that is the
Kähler parameters of the corresponding curves. The series `Z` is computed up to a given total
weight (the option `--order`, Section 3.1).

The heights of a triangulation are not unique. By default the program finds some heights
itself, and Section 3.5 explains how to choose them. A different choice changes which terms
are inside the truncation, but not the coefficients of the series.

---

## 3. Command-line examples

Every run first prints the lattice points, the triangles, and a line for each internal edge
`Q_e` with its segment, framing `n_e`, weight `w_e` and curve class. Then it prints the
series `Z` and, on request, the table of Gopakumar-Vafa (GV) invariants.

### 3.1 Local P², GV invariants

```bash
topvertex "p=-1,-1_1,0_0,1" --order 5 --gv --classes
```

* `--order 5` truncates the series. Only the terms with total weight `Σ_e w_e |λ_e| ≤ 5` are
  computed, where `λ_e` is the partition on the edge `e` and `w_e` is its weight (Section
  2.3). Every curve class of weight at most the order is then complete. For local P² all
  weights are 1 and the order is simply the degree in `Q`.
* `--gv` extracts and prints the GV integers `n^g_β`. They are given in the physical
  convention, in which the resolved conifold has `n^0 = +1`. The output contains
  `3, −6, 27, −192, 1695` at genus 0, `−10, 231, −4452` at genus 1, and so on.
* `--classes` expresses `Z` in **curve-class variables** `t_i` instead of one `Q_e` per edge.
  The edges in the same homology class share a variable. For local P² the three edges all have
  the class `(1,)`, and `Z` becomes a series in a single variable `t_1`. Without `--classes`
  the output is the finer expansion in `Q_0, Q_1, Q_2`. The GV invariants are always computed
  per class.

### 3.2 Local F0, two Kähler parameters

```bash
topvertex "p=1,0_0,1_-1,0_0,-1" --order 6 --gv --classes
```

The two classes are the base `B` and the fiber `F` of P¹×P¹. The GV table is indexed by
`(d_B, d_F)` and reproduces the known values `−2, −4, −6, −32, −110, −756, ...`.

### 3.3 Choosing a triangulation

```bash
topvertex "p=1,0_0,1_-1,1_0,-1" --list-triangulations
topvertex "p=1,0_0,1_-1,1_0,-1" --pick 0 --order 5 --gv
topvertex "p=1,0_0,1_-1,1_0,-1" --triangles "0,0 1,0 0,1; 0,0 0,1 -1,1; 0,0 -1,1 0,-1; 0,0 0,-1 1,0" --order 5 --gv
```

The F1 polygon has two unimodular triangulations, the star and its flop. The first command
lists them with an index and says whether each of them is regular. The second and the third
command select one. Different triangulations of the same polygon are different geometries,
related by flops.

### 3.4 Output formats

```bash
topvertex "p=-1,-1_1,0_0,1" --order 3 --format latex
topvertex "p=-1,-1_1,0_0,1" --order 3 --format mathematica
topvertex "p=-1,-1_1,0_0,1" --order 3 --format sympy
```

| format | output |
|---|---|
| `text` (default) | coefficients as Laurent polynomials in `x = q^{1/2}` over products of q-numbers `[k] = q^{k/2} − q^{−k/2}`, e.g. `(-3) / ([1]^2)` |
| `latex` | a single LaTeX expression for `Z` |
| `mathematica` | an expression that can be pasted into Mathematica (`q^(1/2)` etc.) |
| `sympy` | a Python/SymPy expression |

### 3.5 Choosing the heights

```bash
topvertex "p=1,0_0,1_-1,1_0,-1" --pick 0 --order 6 --heights 1,1,0,1,1 --gv
```

The weights `w_e` are computed from heights of the triangulation (Section 2.3). By default the
program finds some heights by linear programming. The option `--heights h0,h1,...` sets them
by hand, with one integer per lattice point **in the order in which the points are printed**
on the `points:` line (sorted by `(x, y)`). In the example above the interior point `(0,0)` is
the third one and has height 0, and the four corners have height 1. The weights of the four
edges are then `2, 3, 1, 2` instead of the default `1, 2, 1, 1`. If the given numbers are not
heights of the chosen triangulation, the program stops with an error that lists the weights
that are not positive.

The curve classes are printed as coordinates in a basis of H₂. The program chooses the basis
among the edge classes, preferring edges of the smallest weight, and the `class basis` line
shows it in terms of relation vectors. For F1 the basis consists of the base `B` (the edge
with framing 0) and the fiber `F`. The section, which is the edge with the segment from
`(0,−1)` to `(0,0)`, comes out in the class `B + F = (1, 1)`.

### 3.6 Nekrasov cross-check (gauge-theory geometries)

```bash
topvertex "p=1,0_0,1_-1,0_0,-1" --order 4 --nekrasov                     # pure SU(2), kappa = 0 (local F0)
topvertex "p=0,0_2,0_2,1_1,2_0,1" --pick 0 --order 4 --nekrasov          # SU(2), 1 + 1 flavors
topvertex "p=0,0_3,0_3,1_2,2_1,2_0,1" --pick 0 --order 4 --nekrasov      # U(2) x U(2), 1 + 1 flavors
```

The option `--nekrasov` recognizes every triangulation in which all triangles have a side
parallel to one lattice direction `d` (a "ladder" along `d`). The lattice lines parallel to
`d` cut the polygon into columns 0, 1, …, r+1 with r ≥ 1. The dictionary with the gauge theory
is the following:

* An inner column `i` with `N_i + 1` points is a gauge node U(N_i). Its `N_i` segments are the
  horizontal edges of the web, with one Coulomb branch parameter each.
* The two outer columns give `N_0` and `N_{r+1}` fundamental flavors.
* Neighboring inner columns are joined by bifundamentals.
* The Chern-Simons levels are `κ_i = b_{i+1} − 2b_i + b_{i−1} − N_i + (N_{i−1} + N_{i+1})/2`,
  where `b_i` is the position of the lowest point of the column `i` along `d`.
* The instanton counting parameters are `φ_i = ±Q_{T_i} Π_e Q_e^{c_e}`. Here `T_i` is the
  lowest horizontal edge of the node `i`, and `c_e` are half-integers on the diagonal edges
  of the two neighboring strips.

This dictionary was derived from the web and checked numerically, and it was not taken from
the literature. The program computes the unrefined 5d Nekrasov partition function of the
quiver and checks `Z_top = Z_pert · Z_Nekrasov` coefficient by coefficient for **all curve
classes of weight ≤ `--order`**. Here `Z_pert` is the part of `Z_top` that consists of the
classes of instanton number 0.

The simplest case is pure SU(N), with one node and no fundamentals. The polygons
`{(0,j), 0 ≤ j ≤ N} ∪ {(−1,1), (1,N−m−1)}` (`topvertex.examples_geometry.su_n_ladder(N, m)`)
give κ = −m and `φ = (−1)^m Q_B Π_j Q_{F_j}^{−(N−j)}`, where `B` is the lowest horizontal edge
and `F_j` are the fibers. The cases N = 2 and m = 0, 1, 2 are local F0, F1, F2. The three
commands above print the following, each followed by the time:

```
linear quiver along (0, 1): U(2), no fundamentals, CS level kappa = 0
  node 1: branes (bottom to top) Q1, Q2; phi_1 = +Q1 Q0^-1
Nekrasov comparison: MATCH for all 10 instanton classes of weight <= 4 (instanton numbers up to 4)
also a linear quiver along (1, 0)

linear quiver along (0, 1): U(2), fundamentals 1 + 1, CS level kappa = -1
  node 1: branes (bottom to top) Q2, Q3; phi_1 = -Q2 Q0^(1/2) Q1^-1 Q4^(1/2) Q5^(1/2)
Nekrasov comparison: MATCH for all 35 instanton classes of weight <= 4 (instanton numbers up to 4)

linear quiver along (0, 1): U(2) x U(2), fundamentals 1 + 1, CS levels kappa = (-1/2, -1/2)
  node 1: branes (bottom to top) Q2, Q3; phi_1 = -Q2 Q0^(1/2) Q1^-1 Q4 Q5 Q6^(1/2)
  node 2: branes (bottom to top) Q7, Q8; phi_2 = +Q7 Q4^(1/2) Q5 Q6^-1 Q9^(1/2) Q10^(1/2)
Nekrasov comparison: MATCH for all 204 instanton classes of weight <= 4 (instanton numbers up to (4, 4))
```

Local F1 and F2 are `"p=1,0_0,1_-1,1_0,-1" --pick 0` and `"p=1,0_-1,2_0,-1"`. They come out
with κ = +1 and κ = +2. The reason is that the direction (0, 1) in these polygons is the image
of the direction (0, −1) of `su_n_ladder(2, m)`, and reversing the direction flips κ → −κ. If
the triangulation is a ladder along several directions, the one with the fewest gauge nodes is
used and the others are listed. For polygons that are not ladders `--nekrasov` prints "not
recognised". From Python the comparison is
`topvertex.nekrasov.compare_with_nekrasov(tri, 6)`. It returns a dict with the dictionary, the
flag `match` and the mismatching classes, and `topvertex.nekrasov.format_comparison` prints it
as above.

---

## 4. Graphical input (drawing the diagram)

```bash
topvertex --draw                       # starts a local web server and opens the browser
topvertex --draw "p=-1,-1_1,0_0,1"     # same, with this polygon preloaded and triangulated
topvertex --draw --no-browser          # only print the address (e.g. over ssh)
topvertex --draw --port 9000           # fix the port (by default 8765, or the next free one)
```

The page is served at `http://127.0.0.1:8765` and is reachable only from the same machine. It
has the following functions:

* **Polygon** mode: a click on a lattice point adds a corner, and a shift-click or a
  right-click removes one. The convex hull and its lattice points are shown, and the `p=...`
  string is editable.
* **Triangulate** mode: a click on two points adds or removes a diagonal. **Auto-triangulate**
  completes any partial set of diagonals, and the **Flip** mode flips a diagonal.
* The **dual web** is updated live, with the (p,q) label of every leg and the framing at both
  ends of every internal edge.
* **Compute**: choose the order, the options and the output format, and run the same
  computation as the command line. The **truncation heights** field is optional and has the same meaning
  as `--heights` (Section 3.5). The default format **rendered (MathJax)** typesets the edge
  table, the GV table and `Z` as mathematics. MathJax is fetched from a CDN the first time,
  and without an internet connection the page shows the raw LaTeX instead. The formats
  **text**, **latex** and **mathematica** give the output of the command line. The button
  **Copy LaTeX** copies the result for a paper.
* **Export**: the equivalent command line (`--triangles ...`) and a JSON description of the
  diagram, so that a drawing can be reproduced or shared.

The state of the page is kept in the address bar. For example,
`http://127.0.0.1:8765/#p=-1,-1_1,0_0,1&order=3&format=mathjax&gv=1&classes=1&auto=compute`
computes local P² on load. Opening `topvertex/draw/index.html` directly as a file also works
for drawing and exporting. Only the Compute button needs the server, and the page then links
to the online version. When the page is served by any other web server (e.g. GitHub Pages),
it computes in the browser with Pyodide instead, as described in "Try it online" above.

**Working on a remote machine.** Suppose that the program runs on a machine other than the one
with the browser. Start it there with `--no-browser`, run
`ssh -L 8765:127.0.0.1:8765 remote-host` on the local machine, and open
`http://127.0.0.1:8765` locally.

The server refuses orders above 12 and polygons with more than 30 lattice points unless the
option `--unsafe` is passed. These limits are only guards against accidentally huge
computations.

---

## 5. Using the package from Python

```python
from topvertex.examples_geometry import local_p2, local_f1          # ready-made geometries
from topvertex.polygon import Triangulation, parse_polygon
from topvertex.toric import from_triangulation
from topvertex.partition_function import (partition_function, class_series,
                                          free_energy, gv_invariants, gv_table)

g = local_f1()                                   # GTGraph: vertices, directed edges, framings, classes
Z = partition_function(g, max_weight=6)          # Series in Q0..Q3 with exact QFrac coefficients
Zc, basis, coords = class_series(g, Z)           # Series in class variables t1 (base), t2 (fiber)
print(gv_table(gv_invariants(free_energy(Zc))))

# a custom diagram
tri = Triangulation.from_corners_and_triangles(parse_polygon("p=0,0_4,0_0,2"),
        [((0,0),(1,0),(0,1)), ((0,1),(1,0),(0,2)), ((0,2),(1,0),(1,1)), ((0,2),(1,1),(2,1)),
         ((1,0),(2,0),(1,1)), ((1,1),(2,0),(2,1)), ((2,0),(3,0),(2,1)), ((2,1),(3,0),(4,0))])
g = from_triangulation(tri)
print(g.framings(), g.curve_classes())
```

Other entry points are the following:

| function | purpose |
|---|---|
| `topvertex.polygon.parse_polygon(s)`, `format_polygon(corners)` | the compact corner format of Section 2.1 |
| `topvertex.polygon.all_unimodular_triangulations(corners)` | all unimodular triangulations of a polygon |
| `topvertex.polygon.heights(tri)` | heights of a triangulation (Section 2.3), or `None` if it is not regular |
| `topvertex.polygon.plot(tri)` | picture of the triangulation and the dual web (matplotlib) |
| `topvertex.toric.from_web(vertex_vectors, gluings)` | a (p,q)-web given directly |
| `topvertex.vertex.vertex(l1, l2, l3)` | a single vertex `C_{λμν}(q)` |
| `topvertex.nekrasov.compare_with_nekrasov(tri, max_weight)` | Nekrasov comparison for linear quivers, pure SU(N) included |

The coefficients are `QFrac` objects with the methods `.to_sympy()`, `.latex()`,
`.to_mathematica()` and `.evaluate(x)`. The script `examples/basic_examples.py` runs the
standard examples, and `notes/make_tables.py` regenerates the tables in the notes from the
package.

---

## 6. What is verified (`pytest`)

* **Vertex:** the values of Konishi, the cyclic symmetry, the hook formula, the transposition
  and Cauchy identities.
* **Framings:** local P², F0, F1, F2, O(1)⊕O(−3)→P¹ and the flop of F1. The web input agrees
  with the triangulation input, and the results are invariant under SL(2,Z), relabelling and
  reflection.
* **Partition functions and GV invariants:** the closed forms for the resolved conifold and
  O(0)⊕O(−2)→P¹. The GV invariants of local P² at all genera to degree 6 (Katz-Klemm-Vafa) and
  at genus 0 to degree 10. The genus-0 tables of local F0, F1, F2 (Chiang-Klemm-Yau-Zaslow).
  The three-parameter Example VI of AKMV at genera 0 to 4 (the row ℓ = 1 of AKMV has the
  opposite sign to the physical convention).
* **Nekrasov:** agreement for all classes up to a given weight. The pure cases are SU(2),
  SU(3), SU(4) at all Chern-Simons levels 0 ≤ m ≤ N (instanton numbers up to 8, 7, 6), U(1)
  and the conifold. There are also 20 linear quivers (SU(2) with 0 to 4 flavors, SU(3), U(1)²,
  U(1)³, SU(2)², SU(2)×SU(3), T₃, SU(2)³, …) in several chambers, also after GL(2,Z)
  transformations. The arm/leg form of the Nekrasov factor is compared with the product form
  of Iqbal-Kashani-Poor.
* **Drawing page:** the server API, validation, limits, and a headless-Firefox self-test of
  the geometry code of the page against the Python code.

Run everything with `.venv/bin/pytest -q` (about 35 s, and `-m "not slow"` skips two long
tests).

## 7. Layout

| path | content |
|---|---|
| `topvertex/qring.py` | exact arithmetic in `q^{1/2}` |
| `topvertex/partitions.py` | partitions |
| `topvertex/schur.py` | specialized skew Schur functions |
| `topvertex/vertex.py` | the vertex `C_{λμν}` |
| `topvertex/polygon.py` | lattice polygons, triangulations, regularity and heights, corner strings |
| `topvertex/toric.py` | dual graph, framings, curve classes, weights |
| `topvertex/series.py` | multivariate series |
| `topvertex/partition_function.py` | gluing, free energy, GV extraction |
| `topvertex/nekrasov.py` | Nekrasov comparison |
| `topvertex/cli.py` | command line |
| `topvertex/draw/` | browser editor, local server and the request handling shared with the online version |
| `notes/` | LaTeX notes |
| `examples/`, `tests/`, `tools/` | examples, tests, site builder |

The online version is the static site in `docs/`, served by GitHub Pages. Rebuild it with
`python tools/build_site.py` after changing the package or the page. The script copies the
page, builds the wheel `docs/topvertex-<version>-py3-none-any.whl` and writes
`docs/manifest.json`. Test the site locally with `python -m http.server -d docs 8000`.

License: MIT.
