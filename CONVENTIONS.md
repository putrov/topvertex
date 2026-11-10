# topvertex — project conventions

Python package computing the AKMV topological vertex partition function of a toric
Calabi–Yau threefold given by a unimodular regular triangulation of a lattice polygon.
Run everything with the project venv: `.venv/bin/python`, tests with `.venv/bin/pytest -q`.
Reference papers (PDF + pdftotext dumps) are in `refs/` (not committed):
akmv (hep-th/0305132), marino (hep-th/0406005), strip (hep-th/0410174),
konishi_peng = Konishi (math/0504188; sole author, file name kept), ckyz (hep-th/9903053), kkv (hep-th/9910181), ikp_sun (hep-th/0306032).

## Mathematical conventions — ALL code, tests and notes follow Konishi (math/0504188)

* Variable: x = q^{1/2}. q-number `[k] = q^{k/2} − q^{−k/2} = x^k − x^{−k}`.
* Partitions λ = (λ1 ≥ λ2 ≥ ...), |λ| size, λ^t conjugate,
  `κ(λ) = Σ_i λ_i (λ_i − 2i + 1)` (always even; κ(λ^t) = −κ(λ); κ = 2·Σ contents).
* Specialisations: `q^ρ = (q^{−i+1/2})_{i≥1}`, `q^{λ+ρ} = (q^{λ_i − i + 1/2})_{i≥1}`.
  `h_i(q^ρ) = q^{i(i−1)/4} / ([1][2]…[i])`, `e_i(q^ρ) = q^{−i(i−1)/4} / ([1]…[i])`.
  For q^{λ+ρ} (KP eq. 3):
  `Σ_i e_i(q^{λ+ρ}) z^i = Π_{i=1}^{l(λ)} (1 + q^{λ_i−i+1/2} z)/(1 + q^{−i+1/2} z) · Σ_k e_k(q^ρ) z^k`
  `Σ_i h_i(q^{λ+ρ}) z^i = Π_{i=1}^{l(λ)} (1 − q^{−i+1/2} z)/(1 − q^{λ_i−i+1/2} z) · Σ_k h_k(q^ρ) z^k`
  Skew Schur by Jacobi–Trudi: `s_{μ/ν} = det(h_{μ_i − ν_j − i + j})_{1≤i,j≤l(μ)} = det(e_{μ^t_i − ν^t_j − i + j})_{1≤i,j≤l(μ^t)}` (h_{<0} = e_{<0} = 0).
* Three-point function (KP Def. 2.2), cyclically symmetric in (λ1,λ2,λ3):
  `C_{λ1,λ2,λ3}(q) = q^{κ(λ3)/2} s_{λ2}(q^ρ) Σ_η s_{λ1/η}(q^{λ2^t+ρ}) s_{λ3^t/η}(q^{λ2+ρ})`
  Known values: C_{(1),∅,∅} = 1/[1]; C_{(2),∅,∅} = q²/((q−1)(q²−1)); C_{(1,1),∅,∅} = q/((q−1)(q²−1));
  C_{(1),(1),∅} = (q²−q+1)/(1−q)²; C_{(1),(1),(1)} = (q⁴−q³+q²−q+1)/(q^{1/2}(q−1)³).
  Identities: s_{λ^t}(q^ρ) = q^{−κ(λ)/2} s_λ(q^ρ); C_{λ,∅,∅} = s_λ(q^ρ).
* GT graph from a triangulation (KP §4.2). Lattice points ω = (x, y) ↦ (x, y, 1). One vertex per
  triangle; one internal edge per interior segment τ = (ω1, ω2) shared by triangles σ ∋ ω3 and
  σ' ∋ ω3'; boundary segments are external legs carrying the empty partition.
  Write `ω3' = −a1 ω1 − a2 ω2 − ω3` (a1 + a2 = −2). Framing of the flag (v_σ, e_τ):
  `n_f = (a1 − a2)/2 = a1 + 1`, where (ω1, ω2, ω3) is ordered COUNTERCLOCKWISE; `n_{−f} = −n_f`.
  Meaning: normal bundle of C_τ is O(n_f − 1) ⊕ O(−n_f − 1). Examples: conifold 0/0; local P² ±2;
  local F0 ±1; O(1)⊕O(−3)→P¹ gives ±2.
  Each internal edge gets an arbitrary direction; its label `n_e` is the framing of the flag at its TAIL.
  Cyclic order of the three legs at a vertex = counterclockwise order of the triangle's sides.
* Partition function (KP eq. 4, Def. 2.3), λ_e partitions on internal edges, Q_e formal variables:
  `Z = Σ_λ Π_e (−1)^{|λ_e|(n_e+1)} q^{n_e κ(λ_e)/2} Q_e^{|λ_e|} Π_v C_{λ_v}(q)`
  At vertex v the triple λ_v is read counterclockwise; λ_e is used UNtransposed if e points AWAY
  from v and transposed (λ_e^t) if e points TOWARD v. External legs contribute ∅.
  Single edge with framing n (O(n−1)⊕O(−n−1) → P¹): Z_d = (−1)^{(n+1)d} Σ_{λ⊢d} q^{(n−1)κ(λ)/2} s_λ(q^ρ)².
  Resolved conifold (n=0): log Z = −Σ_k Q^k/(k [k]²). O(0)⊕O(−2) (n=1): log Z = +Σ_k Q^k/(k [k]²).
* GV expansion, physical convention (q = e^{i g_s}, so (2 sin(k g_s/2))^{2g−2} = (−1)^{g−1} [k]^{2g−2}):
  `F = log Z = Σ_{β,g,k≥1} n^g_β (−1)^{g−1} (1/k) [k]^{2g−2} Q^{kβ}`
  Conifold: n^0_1 = +1. Fiber of F0 (family of (0,−2) curves): n^0_F = −2.
  Extraction: f_β = F_β − Σ_{k≥2, k|β} (1/k) Σ_g n^g_{β/k} (−1)^{g−1} [k]^{2g−2};
  P = f_β · [1]² must be a Laurent polynomial in q, symmetric under q ↔ 1/q;
  P = Σ_g n^g_β (−1)^{g−1} y^g with y = [1]² = q − 2 + q^{−1}.
* Curve classes: relation vector l_τ ∈ Z^{#points}: +1 at ω3 and ω3', a1 at ω1, a2 at ω2 (Σ l_i ω_i = 0).
  The relation lattice L ≅ H_2(X), rank = #points − 3. Kähler weights w_e = ⟨h, l_e⟩ > 0 with h the
  heights of the regular triangulation; truncate sums by Σ_e w_e |λ_e| ≤ D.
* Literature dictionary: AKMV/Mariño use q = e^{g_s} and (2 sinh)^{2g−2}; their BPS numbers differ
  from the physical ones by (−1)^{g−1} (and possibly class-dependent signs: AKMV Example VI's
  ℓ = 1 row has the opposite sign to the physical one, tested). AKMV's gluing (3.15):
  Σ_R C_{R_j R_k R} e^{−|R|t} (−1)^{(n+1)|R|} q^{−nκ_R/2} C_{R^t R'_j R'_k}, n = v_k' ∧ v_k, where
  AKMV's ∧ is normalised by v_2 ∧ v_1 = 1 for counterclockwise-ordered legs, i.e. it is MINUS the
  standard determinant. With the standard determinant (x1 y2 − y1 x2) and all leg vectors outgoing,
  the Konishi tail label is n_e = v_k' ∧ v_k with v_k (v_k') the leg preceding the edge in ccw
  order at the tail (head); see `toric.web_framing`. Importing a gluing formula from the
  literature without re-deriving the sign of n gives wrong results (local P² would give
  n^0_3 = 15 instead of 27).
* Linear quivers vs Nekrasov (`topvertex/nekrasov.py`, notes App. A; dictionary derived from the web and
  verified numerically, not taken from the literature). This is the ONLY Nekrasov pipeline: pure SU(N) is the
  case r = 1, N_0 = N_2 = 0 and has no code path of its own; `--nekrasov` is a plain flag and compares all
  classes of weight ≤ `--order`. Ladder along primitive d: EVERY triangle has a
  side parallel to d. In an orientation-preserving unimodular frame with d = (0,1) the points are in
  columns X = 0..r+1 (r ≥ 1), column i = {(i, b_i + j), 0 ≤ j ≤ N_i}. Columns 1..r: gauge nodes U(N_i);
  columns 0, r+1: N_0, N_{r+1} fundamentals; adjacent gauge columns: bifundamental. Brane (i,k), k = 1..N_i
  from the bottom = the k-th vertical unit segment of column i (for gauge nodes an internal "horizontal"
  edge of class T_{i,k} carrying Y^(i)_k). Brane position `y^(i)_k = e_top − e_bottom ∈ Z^{#points}`;
  a^(i)_k = −y^(i)_k; differences of y within the same or adjacent columns are relation vectors of
  non-zero weight; |D| := ±D of positive weight. In a strip, consecutive triangles (bottom to top) are
  separated by a diagonal τ with `y(upper) − y(lower) = l_τ`.
  `κ_i = b_{i+1} − 2b_i + b_{i−1} − N_i + (N_{i+1} + N_{i−1})/2`, `ε_i = (−1)^{b_{i+1}+b_{i−1}+N_i+N_{i−1}}`,
  `φ_i = ε_i Q^{Φ_i}`, `Φ_i = T_{i,1} − Σ_{k>1}(y_k − y_1) + ½ Σ_{l ∈ columns i±1} |y_1 − y_l|`.
  `n_{μν}(u) = Π_{s∈μ}[u + a_μ(s) + l_ν(s) + 1] Π_{s∈ν}[u − a_ν(s) − l_μ(s) − 1]`,
  `[u + c] = Q^{−D/2} x^c − Q^{D/2} x^{−c}` for u = a − a', D = y' − y (expand in Q^{|D|}).
  `Z_quiver = Σ_Y Π_{i=1..r} φ_i^{|Y^(i)|} CS_{κ_i}(Y^(i)) / Π_{k,l} n_{Y^(i)_k Y^(i)_l}(a^(i)_k − a^(i)_l)
   × Π_{i=0..r} Π_{k,l} n_{Y^(i)_k Y^(i+1)_l}(a^(i)_k − a^(i+1)_l)`,
  `CS_κ(Y^(i)) = Π_k Π_{(a,b)∈Y_k} (Q^{y_1 − y_k} q^{a−b})^κ` (frame a^(i)_1 = 0; (a,b) = (row, column) of the box, so CS_κ = Π_k Q^{−κ(y_k − y_1)|Y_k|} x^{−κ·κ(Y_k)}).
  Leading monomial of the term Y = Π Q_{T_{i,k}}^{|Y^(i)_k|}. Instanton numbers of a class l:
  `n_i(l) = Σ_ω l_ω max(0, X_ω − i)`; Z_pert = part of Z with all n_i = 0; statement: Z = Z_pert · Z_quiver.
  Pure SU(N) ladder (N, m): κ = −m, ε = (−1)^m, φ = (−1)^m Q_{B_bot} Π_j Q_{F_j}^{−(N−j)}.
  The bifundamental is oriented left to right (n_{νμ}(−u) = (−1)^{|μ|+|ν|} n_{μν}(u)), which is why ε_i is
  not left-right symmetric; the ladder along −d has reversed columns and κ → −κ.

## Code style
* Pure Python 3.11+. Exact arithmetic in q: python-flint (`flint.fmpq_poly`) when installed, else the
  pure-Python backend `_pypoly.py` (same results; `TOPVERTEX_NO_FLINT=1` forces it; `qring.BACKEND`).
  Heights LP: scipy when installed, else the exact Fraction simplex in polygon.py (`TOPVERTEX_NO_SCIPY=1`).
  The only hard dependency is sympy>=1.13 (output, lattice bases); the package must keep running in
  Pyodide (browser) with sympy alone. matplotlib is imported lazily.
* Partitions are tuples of positive ints in weakly decreasing order; the empty partition is `()`.
* Every public function has a docstring stating the formula it implements and its reference.
* Tests in `tests/`, oracle numbers in `tests/data.py` (never retype literature numbers elsewhere).
