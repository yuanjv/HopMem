# HopMem v2.3.2 — consolidated specification

**A bounded associative memory with an auditable compression channel: the mathematics, the measurements, and the claims that survived them.**

*This issue supersedes every prior v2.3.2 section-issue and all of v2.3.1 / v2.2.1. Code is unchanged at v2.3.2 (47/47 self-checks); the revision is documentation and claim structure. Section numbers are preserved so that code references (`spec 4.5`, `4.6`, `17.6`, `23`, `section 24`) remain valid.*

**Reading guide.** §1 is the claim ledger — read it first. §3–§8 are the mathematics with proofs. §17 is every measurement, including the negative ones. §19 records what does not work and why. §21 lists every withdrawn claim. §23 is the LM interface, now measured in part.

---

## 1. Scope and claim ledger

HopMem is a fixed-size associative state with a Gibbs readout, a calibrated abstention gate, and a parameter-free compression channel. It is **not** a claim of superior retrieval. Its contribution is a bounded state whose forgetting is evidence-monotone, auditable, and provably bounded in cost.

| Claim | Status | Evidence |
|---|---|---|
| State invariants; convex write; dose additivity | **THEOREM + unit** | §3, §4, checks 01–09 |
| Gibbs descent for every β>0, no step size | **THEOREM + unit** | §8.2, check 11 |
| Decide-before-descend; hysteresis witness | **THEOREM + unit** | §8.4, check 13 |
| Evidence domination (both operations) | **THEOREM + unit** | Thm A, 0/20000 |
| Commutation incl. the resolution gate | **THEOREM + unit** | Thm B, checks 43/47 |
| Bounded forced debit ≤ ½ nat | **THEOREM + unit** | Thm E′, check 46 |
| Debit dominance over dose-ordered eviction (`debit` policy) | **THEOREM + unit** | Thm E, check 40 |
| Exact incremental Gram, O(C) | **THEOREM + unit** | §4.6, drift 6.7e−16 |
| Forgetting rate recoverable from the marginal law | **MEASURED** | gap +0.0139 nats |
| Beats cue-only and centroid on present | **MEASURED** | z = 70.6 / 18.1 |
| Canonical ≡ `debit` in spread geometry | **MEASURED** | bit-identical, check 45 |
| Canonical beats `debit` at duplicate saturation | **MEASURED, 2 devices** | −3.73σ / −3.77σ |
| Compression beats forced-commit eviction in duplicate streams | **MEASURED, 2 devices** | +8.85σ / +8.44σ |
| Merging beats a tuned kNN+threshold at *moderate* duplicate density | **MEASURED** | ~4σ / ~2.7σ |
| Merging protects duplicated ideas by dose doubling | **MEASURED** | applied demo |
| LM retrieval fidelity transfers | **MEASURED** | 0.0097 vs cue 0.1586 |
| LM surprise channel works | **MEASURED** | arithmetic verified |
| **Beats a tuned kNN+threshold on present or absent** | **FALSIFIED** | z = −1.45 / +0.75 |
| **Compression beats tuned kNN+threshold at matched C, spread** | **FALSIFIED** | \|z\| ≤ 0.45 |
| **LM null prior transfers with law frozen** | **FALSIFIED** | κ window empty 4/4 |
| Canonical vs `ward` at saturation | **OPEN** | −1.06σ / −1.78σ |
| LM induction; perplexity benefit | **NOT TESTED** | §23.7–23.8 |

**Non-goals.** Exact recall below resolution; order-independence of greedy selection; a β-independent ledger; monotone max-atom gap; decision-invariance of live eviction; gradual per-target loss under eviction; Loewner-monotone W; ledger comparability across merge counts; retrieval superiority over a tuned nearest-neighbour baseline.

**Conceptual status.** The density-matrix language is a real-symmetric analogue, not a claim of quantum physics.

## 2. Notation and quickstart

| Symbol | Meaning |
|---|---|
| $d$, $C$, $T$, $N$ | embedding dim; capacity; episode length; descent steps (20) |
| $\alpha_i=e^{\ell_i}$, $S=\sum_i\alpha_i$ | dose; stored mass, $S\in[0,1)$ |
| $J=I/d$; $W=(1-S)J+M$; $M=\sum_i\alpha_i z_iz_i^\top$ | uniform state; state; memory |
| $\omega,\Gamma,\bar\varepsilon$; $\beta,\kappa$; $\tau$ | law; readout; gate threshold |
| $G_{ij}=z_i\cdot z_j$ | cached Gram, incrementally updated |
| $A=\alpha_i+\alpha_j$, $w_i=\alpha_i/A$ | pair mass; fusion weight |
| $m=w_iz_i+w_jz_j$, $r=\|m\|$, $\mu=m/r$ | resultant; length; direction |
| $h,l$, $\rho=\alpha_l/\alpha_h$ | heavier / lighter member; mass ratio |
| $\delta_{\mathrm{fuse}},\delta_{\mathrm{evict}}$; $\Delta=(A/S)\delta_{\mathrm{op}}$ | debits; expected block-nats |
| $\Lambda_c=\sum\delta_{\mathrm{fuse}}$ | per-surviving-block ledger, **nats** |
| $p_i=\alpha_i/S$; $c_{ij}=\frac{\alpha_i\alpha_j}{A}(1-G_{ij})$ | target law; Ward cost |
| $\mathcal{R}$; $\epsilon_r$ | resolution-eligible set; resultant mask |

```
python hopmem.py selftest                                     # 47 checks
python hopmem.py train --schedule probe --batch 256 --C 16 --c_min 4
python hopmem_diag.py  --ckpt hopmem_theta.pt --T_big 100 --caps 4 8 16 32 --dup_sweep
python hopmem_stats.py --ckpt hopmem_theta.pt --seeds 5 --readout --readout_only
python hopmem_stats.py --ckpt hopmem_theta.pt --seeds 5 --dup --dup_only --ref debit
python hopmem_demo.py  --verbose --ideas 6
python hopmem_lm.py    --recall --calibrate 400 --caps 8 16 32 --center
```

Deployment: `mem = HopMem(p, d, C, tau)`; `mem.write(z,s,t)`; `mem.retrieve(q,t_q)`. Default policy `resfirst`; `debit` is the ablation satisfying Theorem E.

## 3. State

### 3.1 Invariants
$W\succeq0$, $\mathrm{tr}\,W=1$, $S\in[0,1)$, $\lambda_{\min}(W)\ge(1-S)/d>0$ in exact arithmetic. Initialize $W=J$; never $W=0$. Store $(z_i,\ell_i,\Lambda_i)$, $S$, a lazy absolute-time clock, and $G$. Never materialize $M$ or $W$ outside diagnostics or the legacy readout. Measured violation $4.4\times10^{-16}$.

### 3.2 Canonical log-dose
Between compressions,
$$\ell_i(t_q)=\ln\big(a_i(1-\varepsilon_i)\big)-\omega\sum_{j>i}s_j-\Gamma(t_q-t_i),\quad a_i=1-e^{-\omega s_i},\ \varepsilon_i=\bar\varepsilon(1-a_i).$$
For a compressed block the exact dose is the recursion of §4.5. The closed form uses **per-constituent ages inside** the logaddexp, and
$$\boxed{\;\ell_c\;\ge\;\mathrm{logaddexp}(\mathrm{base})-\Lambda_c\;}$$
is **exact for one merge, conservative for chains**.

*Proof of the chain step.* $\mathrm{logaddexp}(x-a,y-b)\ge\mathrm{logaddexp}(x,y)-\max(a,b)\ge\mathrm{logaddexp}(x,y)-(a+b)$ for $a,b\ge0$. ∎

Hence $e^{-\Lambda_c}$ is a **lower bound** on retained dose fraction. Measured: single-merge error $1.8\times10^{-15}$; chain slack $\le0.27$ nats over 300 random trees.

### 3.3 Ledger (nats)
Per surviving block $\Lambda_c=\sum\delta_{\mathrm{fuse}}$; episode ledger $\sum\delta=\sum\delta_{\mathrm{fuse}}+\sum\delta_{\mathrm{evict}}$; per target retained $=e^{-\Lambda_{c(t)}}$ if surviving, **hard 0** if destroyed. A destroyed block never carries a finite $\Lambda$. Debits use the current β; recompute in-forward during training, freeze at deployment.

**Currency caveat (load-bearing).** $\Delta$ is expected *block log-dose reduction*, a proxy for loss. It charges eviction only the block's dose drop, not the evicted target's total loss of retrievability. Two measured consequences: the canonical policy exists because this currency misprices eviction in duplicate-rich streams (§22.1); and the ledger is **not comparable across runs with different merge counts** (§17.5).

### 3.4 Atom-dependence
The readout depends on the atom set, not on $W$ alone. Compression is defined on atoms; no Loewner claim.

## 4. Operation alphabet

{diffuse, write, reinforce, fuse, null-fuse}. Order per event: diffuse → write → reinforce → compress-on-overflow.

### 4.1 Diffuse
Lazy, **absolute-time** clock: $\Delta t=\max(t-t_{\mathrm{clock}},0)$, $\ell_i\leftarrow\ell_i-\Gamma\Delta t$, $S\leftarrow Se^{-\Gamma\Delta t}$.

### 4.2 Write (convex primitive)
Unit $z$, importance $s\ge0$: $a=-\mathrm{expm1}(-\omega s)$, $\varepsilon=\bar\varepsilon(1-a)$,
$$W'=(1-a)W+a\big[(1-\varepsilon)zz^\top+\varepsilon J\big].$$
Log-atom form: $\ell_j\leftarrow\ell_j-\omega s$ for existing $j$; append $\ln\max(a(1-\varepsilon),10^{-30})$; $S'=(1-a)S+a(1-\varepsilon)$. Exact impure decomposition $a[(1-\varepsilon)zz^\top+\varepsilon J]=a(1-\varepsilon)zz^\top+a\varepsilon J$ — the blur applies to the injected state, not only the target direction.

### 4.3 Reinforce
$\mu=u^\top Wu$, $\tau_{\mathrm{ret}}=1/d+(1-1/d)a_{\mathrm{ret}}$,
$$\beta_{\mathrm{ret}}=\mathrm{cf}\cdot\min\big(\bar\beta_{\mathrm{ret}},\ (\tau_{\mathrm{ret}}-\mu)_+/(1-\mu)\big)\quad\text{if }\mu<\tau_{\mathrm{ret}},\ \text{else }0,$$
masked denominator, never the raw fraction. Counts toward $C$. Purity is not monotone.

### 4.4 Null-fusion — two triggers
**(a) Gate-dead (free, decision-free).** Atom $i$ can never trigger a commit for any unit cue iff $\ell_i\le\kappa+\ln\tau$. *Proof:* $\max_q\mathrm{gap}_i(q)=\ell_i-\kappa$, attained at $q=z_i$. ∎ Deleting it leaves every gate decision unchanged. Applied first on overflow.
**(b) Debit-eviction (decision-changing).** A live lighter atom is deleted when §4.5 selects eviction. Not covered by decision-invariance; Theorem A still bounds its evidence.

### 4.5 The compression policy (canonical)

**Trigger.** Atom count $>C$.

**Step 1.** Null-fuse all gate-dead atoms.

**Step 2 — resolution gate.** With $r^2=1-2w_iw_j(1-G_{ij})$, let
$$\mathcal{R}=\big\{(i,j):\ r\ge\epsilon_r,\ \beta(1-G_{ij})\le1\big\}.$$
If $\mathcal{R}\ne\emptyset$, **merge** $\arg\min_{\mathcal{R}}\Delta_{\mathrm{fuse}}$.

**Step 3 — debit argmin.** Otherwise, with
$$\boxed{\;\delta_{\mathrm{fuse}}=\beta(1-r)=\frac{2\beta w_iw_j(1-G_{ij})}{1+r},\qquad \delta_{\mathrm{evict}}=\ln\frac{A}{\alpha_h}=\mathrm{softplus}(\ell_l-\ell_h)\;}$$
$$\boxed{\;\Delta(i,j,\mathrm{op})=\frac{A}{S}\,\delta_{\mathrm{op}},\qquad (i^{*},j^{*},\mathrm{op}^{*})=\text{the minimizing pair and operation}\;}$$
execute the winner. Since $1/S$ is constant per episode the argmin is over $A\delta$; compare in log space, $\ln\Delta=\mathrm{logaddexp}(\ell_i,\ell_j)-\ln S+\ln\delta_{\mathrm{op}}$. Repeat until count $\le C$. Never form $1-r$ by subtraction.

**The two maps.** *fuse*: $\mu=m/r$, $\ell'=\mathrm{logaddexp}(\ell_i,\ell_j)-\delta_{\mathrm{fuse}}$, deficit $A(1-e^{-\delta})$ to $(1-S)J$, $\Lambda'=\Lambda_i+\Lambda_j+\delta_{\mathrm{fuse}}$, provenance unions. *evict*: delete $l$; $h$ untouched; book $\delta_{\mathrm{evict}}$; mark $l$'s provenance destroyed.

**Debit semantics.** $\delta_{\mathrm{fuse}}$ is *sufficient* for Theorem A, tight at equal masses (binding at $u=\mu$), conservative otherwise. $\delta_{\mathrm{evict}}$ is the *exact* block log-dose reduction and the unique debit leaving the survivor's evidence identical. The smaller angle-dependent minimum $\ln\!\big(A/(\alpha_h+\alpha_le^{-\beta\sqrt{2(1-G)}})\big)$ inflates the survivor by $1+\rho e^{-\beta\sqrt{2(1-G)}}$ and is **forbidden**.

**Two derived angles, zero new constants.**
$$\theta_{\mathrm{gate}}=\arccos\!\big(1-1/\beta\big)\approx\sqrt{2/\beta},\qquad \theta^{*}=2\arccos\!\big(1-\ln2/\beta\big).$$
At β=20.70: $\theta_{\mathrm{gate}}=0.312$ (cos ≥ 0.952), $\theta^{*}=0.519$ (cos ≥ 0.868). **Merge reach shrinks as the readout sharpens** — a sharper readout distinguishes more pairs and therefore merges fewer. This tension recurs in §23.

**Representational symmetry.** $(i,j)$ and $(j,i)$ are the same unordered pair; their floating-point costs differ by ~1 ulp. Invariance claims compare **unordered-pair keys and operations**, never flat argmin indices (§17.7).

**Edge cases.** $s=0$: gate-dead, free. $r<\epsilon_r$: excluded from $\mathcal{R}$ and from merge candidates; eviction always admissible, so the $C{=}1$ antipodal fallback is subsumed. $C\ge T$: nothing fires; bit-for-bit v2.1.1 regression.

### 4.6 Incremental Gram law (mandatory)
On a merge into slot $i$, using the **old** rows, for every alive $k\ne i,j$:
$$\boxed{\;G_{ik}\leftarrow\frac{w_iG_{ik}+w_jG_{jk}}{r},\quad G_{ki}\leftarrow G_{ik},\quad G_{ii}\leftarrow1,\quad \text{row/col } j\leftarrow0\;}$$
Exact by linearity of the dot product; $O(C)$, no $d$-dimensional products. **Two guards:** the formula does not yield 1 at $k=i$ (it gives $\mu\cdot z_i$), so set it explicitly; zero row/col $j$ **after** computing the new row. On write: one new row, $O(Cd)$. Full rebuild is a drift diagnostic only; refresh if any $|G_{ij}|>1+10^{-6}$. Measured drift $6.66\times10^{-16}$ over $10^3$ mixed operations.

## 5. Diffusion: six exact laws

Generator: uniform log-drift at rate Γ. Over Δt:

- **D1 semigroup.** $W(t+\Delta)=e^{-\Gamma\Delta}W(t)+(1-e^{-\Gamma\Delta})J$; intervals compose additively.
- **D2 log-drift.** $\ell_i\leftarrow\ell_i-\Gamma\Delta$, uniform and additive.
- **D3 mass.** $S\leftarrow Se^{-\Gamma\Delta}$; $S<1$ preserved.
- **D4 spectral invariance.** Eigenvectors of $M$ preserved; eigenvalues scale by $e^{-\Gamma\Delta}$; $\mathrm{range}(M)$ invariant.
- **D5 commutation.** Diffusion commutes with write-dilution, both compression maps, the debit argmin, **and the gate predicate** $\beta(1-G)\le1$: under a uniform log-shift $A/S$, $w$, $r$, $\ell_l-\ell_h$ and $G$ are invariant. The gate-dead branch remains clock-dependent.
- **D6 forgetting times.** $t_{\mathrm{forget}}=(\ell_k(0)-\Lambda_k-\kappa)/\Gamma$; gate-dead time subtracts $\ln\tau$. Evicted blocks are gone, not faded.

**Steady state.** $S^{*}=a(1-\varepsilon)/\big(1-(1-a)e^{-\Gamma\Delta}\big)$. **Decay split:** report $\omega\mathbb{E}[s]+\Gamma\mathbb{E}[\Delta t]$, not Γ alone.

## 6. Writing details
$-\mathrm{expm1}$; clamp $s\ge0$; $\ell=\ln\max(a(1-\varepsilon),10^{-30})$ finite at $s=0$ with zero clamped-branch gradient; $\bar\varepsilon\equiv0$ via the `pure_writes` flag. **Sigmoid-parameterized scalars never reach 0 or 1**: `eps_bar=0.0` is a domain error, use the flag.

## 7. Parameters and calibration
Law $\omega,\Gamma,\bar\varepsilon$; readout $\beta,\kappa$; gate $\tau$; capacity $C$ architectural. **Compression adds zero parameters** — both debits and both angles derive from $\alpha,G,\beta,S$.

Positive scalars via softplus, $\bar\varepsilon,\chi$ via sigmoid, $\kappa$ raw. Stable inverse: $\mathrm{invsoftplus}(y)=\ln(e^y-1)$ via expm1 for $y<20$, else $y+\ln(1-e^{-y})$.

**τ protocol.** Calibrated **once** by a two-pass fixed point in $\tau_{\mathrm{dead}}$ on one fixed compressed training batch, saved, applied identically by `evaluate` and streaming `retrieve`. Grid $\tau_i=e^{-6+0.4i}$, $i=0..18$; flat sweep → argmax, not the grid edge. Never fit on eval.

**Identifiability.** Γ from the marginal (cue-free) law; β, κ from the conditional. $\partial\mathrm{CE}/\partial\Gamma=\mathrm{age}_k-\mathbb{E}_p[\mathrm{age}]$ vanishes when the posterior is one-hot: ambiguous cues are a data requirement. κ serves two jobs (marginal null-odds in P, conditional commit threshold in A/B), so prior CE rising after P is expected.

## 8. Readout

### 8.1 Gibbs free energy
$$F(u)=\frac{1}{\beta}\ln\Big(e^{\beta q\cdot u+\kappa}+\sum_i\alpha_ie^{\beta z_i\cdot u}\Big).$$
At $u=q$ the posterior is $\mathrm{softmax}\big([\beta Z^\top q+\ell,\ \beta+\kappa]\big)$, last index null.

### 8.2 Descent theorem
$F$ is convex; with $p$ the posterior, $\nabla F(u)=p_qq+\sum_ip_iz_i$ and $u^{+}=\nabla F(u)/\|\nabla F(u)\|$ satisfies $F(u^{+})\ge F(u)$ for **every** β>0, equality iff $u^{+}=u$; no step size.

*Proof.* $F$ is a log-sum-exp of affine functions, hence convex; $\nabla F(u)$ is a convex combination of $q$ and the $z_i$, so the normalized gradient is an ascent direction by convexity of $F$ on the sphere. ∎

Energy-increase fraction must be 0 up to rounding. Masked float guard; empty memory returns $q$ exactly; permutation- and log-shift-invariant.

### 8.3 Geometry
$$\theta_{\mathrm{basin}}=\arccos\!\big(1-(\ell_k-\Lambda_k-\kappa)/\beta\big),\qquad \text{floor}\approx(\bar\alpha/\alpha_k)e^{-\beta}\sqrt{T-1}.$$
Resolution: $\beta(1-z_1\cdot z_2)\gtrsim|\ell_1-\ell_2|$. Each merge shrinks the basin parameter by exactly $\delta_{\mathrm{fuse}}/\beta$.

### 8.4 Decide before descend
$$\mathrm{gap}(q)=\max_k\big(\beta q\cdot z_k+\ell_k\big)-(\beta+\kappa);\qquad\text{commit iff }\mathrm{gap}(q)>\ln\tau\text{ at }u=q.$$
The map's null basin is strictly smaller than the one-shot argmax region. Witness: β=16, ℓ=−3, κ=−6.5, one atom at 0.70 rad → atom logit **9.237**, null **9.500**; null wins at $u=q$, the map still commits. The mass gate $\ln(E_{\mathrm{mem}}/E_0)$ is the ablation that is monotone under compression.

### 8.5 Legacy quadratic (ablation only)
One stable attractor; $H_k=(\mu+\gamma)P_k-P_kM_{\mathrm{oth}}P_k\succ0$ on the tangent space. For $T$ equal masses stability needs $\gamma\gtrsim2\alpha\sqrt{(T-1)/d}$ while closure is $\alpha/(\alpha+\gamma)$, hence
$$\mathrm{closure}_{\max}\approx1/\big(1+2\sqrt{(T-1)/d}\big).$$
The witness measures the **angle-closing fraction** $(\phi-\theta)/\phi$ at nonzero cue angle; measured 0.313 against bound 0.692.

### 8.6 Compression theorems

**Theorem A (evidence domination — both operations).** For every unit $u$,
$$e^{\ell'+\beta\mu\cdot u}\le e^{\ell_i+\beta z_i\cdot u}+e^{\ell_j+\beta z_j\cdot u}\ \text{(fuse)},\qquad E_{\mathrm{mem}}'\le E_{\mathrm{mem}}\ \text{(evict)}.$$
*Proof (fuse).* $E'=Ae^{\beta(\mu\cdot u-(1-r))}\le Ae^{\beta r\mu\cdot u}=Ae^{\beta m\cdot u}\le\alpha_ie^{\beta z_i\cdot u}+\alpha_je^{\beta z_j\cdot u}$. First step: $(1-r)(\mu\cdot u-1)\le0$. Second: Jensen on the convex exponential. *Proof (evict).* $E'\equiv E_h$. ∎

**Corollaries.** Block posterior mass non-increasing; null mass non-decreasing — **compression never fabricates confidence**. The max-atom gap may rise by ≤ ln2 under fuse; block mass is the monotone object; τ recalibrates. Measured: 0 violations / 20 000 queries, max residual $-1.64\times10^{-5}$.

**Theorem B (dose additivity and commutation).** $\mathrm{logaddexp}(x+c,y+c)=\mathrm{logaddexp}(x,y)+c$ and $r$, $G$ are invariant under uniform log-shifts, so Γ-decay and ω-dilution commute with both maps, the debit argmin, and the gate. Measured $|\delta\Delta|\le3.6\times10^{-15}$ (debit) and $6.8\times10^{-14}$ (gate).

**Proposition C (Ward = expected angular damage).** Under $p_i\propto\alpha_i$,
$$\mathbb{E}\big[1-z_{\mathrm{target}}\cdot\mu\big]=\frac{c_{ij}}{S}+O\big((1-G)^2\big).$$
*Proof.* $1-z_i\cdot\mu=w_j^2(1-G)+O((1-G)^2)$; weight by $\alpha_i/S$ and symmetrize: $\alpha_iw_j^2+\alpha_jw_i^2=\alpha_i\alpha_j/A$. ∎

**Proposition D (first-order expected debits).**
$$\Delta_{\mathrm{fuse}}=\frac{\beta c_{ij}}{S}+O\big((1-G)^2\big),\qquad \Delta_{\mathrm{evict}}=p_l\Big(1+\frac{\rho}{2}+O(\rho^2)\Big).$$
*Proof.* $1-r=w_iw_j(1-G)+O((1-G)^2)$ and $(A/S)\beta w_iw_j=\beta c_{ij}/S$; $\ln(1+\rho)=\rho-\rho^2/2+\cdots$ with $A=\alpha_h(1+\rho)$. ∎

**Lemma F (angular ordering).** $\mathbb{E}[\text{damage}\mid\mathrm{fuse}]=p_lw_h(1-G)$ versus $p_l(1-G)$ for evicting $l$: merging is **always** angularly cheaper, by $w_h\in[\tfrac12,1)$. *Proof:* $c_{ij}/S=p_lw_h(1-G)$. ∎ Eviction competes only because it levies no debit on the survivor.

**Lemma G (cheapest eviction, exact).** With $f(\alpha_h)=(\alpha_h+\alpha_l)\ln(1+\alpha_l/\alpha_h)$: $f'(\alpha_h)=\ln(1+\rho)-\rho<0$ and $\partial f/\partial\alpha_l=\ln(1+\rho)+1>0$. Hence $\Delta_{\mathrm{evict}}$ is minimized **exactly** at (lightest, heaviest). ∎

**Theorem E (debit dominance — `debit` policy only).** $(\mathrm{evict},l^{*},h^{*})$ is a candidate and by Lemma G the cheapest eviction, so
$$\Delta_{\mathrm{debit}}\le\Delta_{\mathrm{evict}}(l^{*},h^{*})=p_{l^{*}}\big(1+O(\rho)\big).\qquad\blacksquare$$
**The canonical policy does not satisfy this.** Measured max excess 0.

**Theorem E′ (bounded forced debit — canonical).** Any gate-selected merge satisfies
$$1-r=\frac{t}{1+r}\le t=2w_iw_j(1-G)\le\tfrac12(1-G),$$
using $w_iw_j\le\tfrac14$; hence $\delta_{\mathrm{fuse}}\le\tfrac12\beta(1-G)\le\tfrac12$ **nat**. Sharper, from $t\le1/(2\beta)$:
$$\delta_{\mathrm{fuse}}\le\beta\Big(1-\sqrt{1-\tfrac{1}{2\beta}}\Big)\xrightarrow[\beta\ \text{large}]{}\tfrac14.\qquad\blacksquare$$
Measured: max 0.251641 ≤ sharp 0.251984 (β=16); 0.251400 ≤ 0.251529 (β=20.70).

**Corollary (bounded override).** Per overflow, $\Delta_{\mathrm{canonical}}-\Delta_{\mathrm{debit}}\le\frac{A}{S}\delta_{\mathrm{fuse}}\le\tfrac12$ expected nat, since $A/S\le1$. The canonical policy pays at most half an expected nat more than the debit-optimal policy and preserves provenance eviction destroys.

**Corollary (regime reduction).** If $\mathcal{R}=\emptyset$ the canonical policy *is* the debit policy; if additionally $\min\Delta_{\mathrm{fuse}}\ge\Delta_{\mathrm{evict}}(l^{*},h^{*})$ its trajectory equals streaming dose-ordered eviction exactly. **Measured bit-identical** at every tested capacity.

**Corollary (heavy-atom protection).** Eviction always removes the lighter member; the heavier atom's direction and dose are untouched — zero distortion and zero evidence inflation for high-probability targets.

**Proposition H (null separability — new).** A single κ achieves perfect present/absent separation iff the **separation margin**
$$\Sigma(\beta)\;=\;\inf_{\text{present}}\big(\beta c_t+\ell_t\big)\;-\;\sup_{\text{absent}}\max_j\big(\beta c_j+\ell_j\big)$$
is positive; the admissible window is $[\,-\!\sup_{\text{absent}}\!\max_j(\cdot)+\beta\ \text{lower}\,]$ — explicitly, $\kappa\in(\sup_a\max_j(\beta c_j+\ell_j)-\beta,\ \inf_p(\beta c_t+\ell_t)-\beta)$. Σ is increasing in β when present targets carry higher cosine than absent maxima, but the *perfect*-separation margin is set by the worst-case pair. **Measured Σ < 0 in 4/4 LM configurations** (§23.5), and the β needed for acceptable overlap drives $\theta_{\mathrm{gate}}\to0$: the sharpness that fixes retrieval disables merging.

## 9. Training and the transfer protocol
Posterior CE at $u=q$ with the null class; labels map to the target's fused block; **evicted targets map to null**. No-match episodes required ($p_{\mathrm{absent}}=0.2$), else κ→−∞ is a free optimum.

**Probe schedule.** P: cue-free CE, compression off, trainable $\omega,\Gamma,\kappa$. A: cue CE on compressed states, $C\sim U\{C_{\min},\ldots,T\}$, trainable $\beta,\kappa$. B: refine $\bar\varepsilon,\beta,\kappa$, law frozen.

**Gradients.** Selection of pair, operation and gate branch is hard argmin with straight-through. Gradients flow through $\delta_{\mathrm{fuse}}$ (which enters $\ell'$) and surviving doses; $\delta_{\mathrm{evict}}$, the gate predicate and all argmins are selection-only and carry **no gradient**; the evict branch has no path through the deleted atom. Ledger bookkeeping detached. Gradient checks inside `no_grad` suites must be wrapped in `enable_grad`.

**Transfer protocol — FALSIFIED as stated.** "Stage A/B on new states with the law frozen" is **insufficient** whenever the task's importance structure differs from the episodic law it was fit on. On LM states the κ window is empty and readout-only refitting cannot open it (§23.5). The law itself — Γ in particular — must be re-fit, or the task must supply an aligned importance signal.

**Gate activity differs between training and deployment.** With $C\sim U\{4..20\}$ and $T=20$, small-C draws put near-duplicate sentences below $\theta_{\mathrm{gate}}$, so the gate fires and changes the gradient trajectory; at deployment capacity it is inert. Canonical and `debit` therefore yield **different checkpoints** even where they agree at evaluation — cross-version checkpoint comparisons are unpaired.

## 10. Episodes and generators
$s_i\sim\mathrm{Exp}(1)$; arrivals $\mathrm{Exp}(1)$; $t_q=t_T+\mathrm{Exp}(1)$; target $\sim\mathrm{softmax}(\ln s-\lambda(t_q-t))$, λ=0.2; cue = random contiguous half-span; same-passage competitors; warn at $p_{\mathrm{true}}>0.99$.

**Duplicate-heavy generator.** With probability $\rho_{\mathrm{dup}}$, $z_{\mathrm{new}}=\mathrm{normalize}(z_{\mathrm{prev}}+\sigma\xi)$, $\sigma=\tan(\theta^{*}/4)$; sweep $\rho_{\mathrm{dup}}\in\{0,0.25,0.5,0.75\}$. At β=20.70 copies sit at 0.130 rad (cos 0.992), inside $\theta_{\mathrm{gate}}=0.312$.

**Document mode — two traps.** (i) *Document position is not time.* The learned Γ=0.1811 was fit on Exp(1) inter-arrivals; applied per sentence it makes the memory a recency buffer. Use Δt=0 for documents. (ii) *Uniform importance makes dose a proxy for reverse age* even at Δt=0, so non-mergeable sentences are evicted oldest-first regardless of salience. Importance must carry salience if position is not the intended prior.

## 11–13. Reinforcement, decoder, blending
Masked denominators throughout. The decoder consumes the **gated** output; normalization $\hat D=D/\sqrt{\|D\|^2+\epsilon^2}$ has Jacobian bound $\le1/\epsilon$; evaluate retrieval separately, since a strong decoder can hide weak memory. Compression row: merge debits the basin by $\delta_{\mathrm{fuse}}/\beta$ (gradual per target); evict removes the block (abrupt); the ledger audits both.

## 14. Guaranteed properties

**Exact arithmetic.** All state, readout and gate guarantees; Theorem A both operations; Theorem B including the gate; Propositions C, D, H; Lemmas F, G; **Theorem E′ and bounded override for the canonical policy**; Theorem E for `debit`; the exact incremental Gram identity; gate-dead decision-invariance; $C\ge T$ regression; strict atom bound; the conservative chain bound.

**Not guaranteed.** Exact recall; unique basins; resolution below ~1/β; Γ identification from the conditional alone; monotone S, purity or max-atom gap; Loewner monotonicity; order-independence; a β-independent ledger; decision-invariance of live eviction; gradual per-target loss under eviction; ledger dominance for the canonical policy; **null separability on tasks whose importance structure differs from the learned law**; retrieval superiority over a tuned kNN baseline; float32-safe margins at extreme β, |κ|.

## 15. Implementation rules
01 Normalize $z,q$; validate finiteness; clamp $s\ge0$.
02 Initialize $W=J$, $S=0$, empty atoms; never $W=0$.
03 Store $(z_i,\ell_i,\Lambda_i)$, $S$, lazy clock, $G$; materialize $M,W$ only for diagnostics.
04 Lazy diffusion with an **absolute-time** clock.
05 Enforce diffuse → write → reinforce → compress-on-overflow.
06 $a=-\mathrm{expm1}(-\omega s)$; stable inverse softplus.
07 $\ell=\ln\max(a(1-\varepsilon),10^{-30})$; finite at $s=0$, zero clamped-branch gradient.
08 Masked denominators wherever a branch may be non-finite.
09 Off-flags: `pure_writes`, diffusion bypass, compression bypass ($C\ge T$). Sigmoid scalars stay in (0,1).
10 The gate lives in the inference path; save state, τ, readout, C, config.
11 Calibrate τ once (two-pass) on a fixed compressed batch; never on eval.
12 Log β, κ, doses, S, purity, margins, gate decisions, hysteresis, decay split, Σδ, destroyed mass, merge/evict counts. Energy-increase fraction must be 0.
13 Report all baselines **decomposed by present/absent**, paired per episode.
14 Run the full 47-check suite.
15 **Policy:** gate-dead null-fusion; resolution gate; debit argmin. Never top-k truncate *as the policy*.
16 Debits and Δ exactly as §4.5, compared in log space; ledger in nats.
17 **Incremental Gram mandatory**, both guards; rebuild only as a drift diagnostic.
18 Evict deletes the lighter; heavier untouched; destroyed provenance → retained 0.
19 $\epsilon_r$ masks merge candidates only; eviction always admissible.
20 Debits use the current β; recompute in-forward; freeze at deployment.
21 P compression-free; A/B with compression; τ two-pass on one fixed batch.
22 Report Λ, retained, destroyed fraction, merge/evict counts, and **`agree-debit`** = fraction of merges the debit branch would also have chosen. **Convention:** with `n_fuse = 0` the ratio is 0/1 = 0%; read as *no merges*, not *total disagreement*. Overrides = `n_fuse` × (1 − `agree-debit`).
23 Self-checks compare like-for-like transcendentals (torch-vs-torch for floor constants, tolerance ≥ 1e−6); gradient checks wrapped in `enable_grad`; masked ±∞ excluded from difference statistics; **invariance checks compare unordered-pair keys and operations, never flat argmin indices**.

## 16. Complexity and storage

| Operation | cost |
|---|---:|
| store state | $C(d+2)+C^2$ |
| write incl. compression | $O(Cd+C^2)$ |
| Gram merge row | $O(C)$ |
| gate + policy scan | $O(C^2)$ (no $d$) |
| diffusion | $O(1)$ lazy |
| posterior / gate | $O(Cd)$ |
| retrieval ($N$) | $O(NCd)$ |
| corrected stability | $O(C^2d+C^3)$ |

All $O(1)$ in stream length for fixed $C,d$. Storage is $O(Cd+C^2)$; "O(1) memory" is claimed **only** in stream length. Design rule $C\le d/2$ keeps the state within $d^2$ floats for $d\ge4$.

## 17. Evaluation protocol and measurements

**17.1 Comparators.** Decomposed present/absent for every method. **kNN + one train-fit threshold** is the like-for-like gate comparator; forced-commit kNN is not (its absent penalty makes aggregate comparison invalid). Beating claims are made on **present** trials; rejection claims on **absent**; both with paired SEs over episodes. At matched capacity, baselines prune the identical subset.

**17.2–17.5 Batteries.** Retrieval battery; init arms R/A; identifiability protocol; ablations `--fuse_policy {resfirst, debit, ward, lightest, random, delta}`; capacity sweep $T_{\mathrm{big}}\in\{100,500,1000\}$, ≥5 seeds, per-C τ recalibration as a train-batch diagnostic only.

**17.6 Predictions and status.**

| # | Prediction | Status |
|---|---|---|
| 1 | Spread geometry: parity with eviction and tuned kNN | **MEASURED** (\|z\| ≤ 0.45) |
| 2 | Duplicate streams: gain over forced-commit eviction | **MEASURED** (+8.44 / +8.85σ) |
| 3 | Gate neutrality in spread, gain at saturation vs `debit` | **MEASURED** (−3.77 / −3.73σ) |
| 4 | Gain over *tuned* kNN at moderate duplicate density | **MEASURED** (~4σ / ~2.7σ); tie at saturation |
| 5 | Bounded override ≤ ½ nat | **THEOREM + unit** |
| 6 | Within-policy absent loss monotone in C | **MEASURED** |
| 7 | Prior-CE gap ≈ +0.02 nats | **MEASURED** (+0.0139) |
| 8 | Canonical vs `ward` at saturation | **OPEN** (−1.06 / −1.78σ) |
| 9 | Gate beats tuned kNN on present or absent | **FALSIFIED** |
| 10 | LM retrieval fidelity transfers | **MEASURED** |
| 11 | LM null prior transfers with law frozen | **FALSIFIED** |
| 12 | LM induction | **NOT TESTED** (out of scope, tied KV) |

**17.7 Metric hygiene (five confounds found and removed).** Each of these produced a wrong conclusion before it was caught; all are now rules.

1. **`acc` mixes present and absent** and counts correct abstention. Decompose.
2. **Block accuracy counts correct abstention on destroyed targets** (label = null index). Condition on the target's block **surviving**; report `surv` separately. At C=8 this turned an apparent 0.992 into "4.3% of targets survive."
3. **Flat argmin indices are not invariance witnesses.** The cost matrix is symmetric; $(i,j)$ and $(j,i)$ differ by ~1 ulp and the representative can flip under a shift. Compare unordered-pair keys and operations.
4. **Masked ±∞ entries poison difference statistics** (`inf − inf = NaN`). Restrict to entries finite in both tensors.
5. **Cross-library and cross-device comparisons are unpaired.** `torch.log` vs `math.log` of the same constant differ enough to fail a 1e−9 test; CPU and CUDA draw different RNG streams, which moved a train-fit threshold 0.410→0.420 and **reversed the sign** of an absent-loss comparison. Draw randomness on a CPU generator; compare like-for-like transcendentals.

**17.8 Dose-spread reporting.** The raw spread across a store is dominated by the $s=0$ first token, whose dose is clamped to $10^{-30}$ ($\ell=-69.08$). Report the **live-atom** spread separately: measured 14.79 nats live versus 86.09 raw on LM states. Using the raw figure to size β gives an answer wrong by an order of magnitude.

## 18. Core mathematical summary

**State.** $W=(1-S)J+M$; ≤C log-atoms; ledger Λ in nats.
**Compress.** Gate-dead first; merge $\arg\min_{\mathcal{R}}\Delta_{\mathrm{fuse}}$ if $\mathcal{R}\ne\emptyset$; else $\arg\min\frac{A}{S}\min\big(\beta(1-r),\mathrm{softplus}(\ell_l-\ell_h)\big)$.
**Gram.** $G_{ik}\leftarrow(w_iG_{ik}+w_jG_{jk})/r$, then $G_{ii}\leftarrow1$, then zero row/col $j$.
**Retrieve.** Gibbs $F$; decide at $u=q$ ($\mathrm{gap}>\ln\tau$), then descend.
**Limits.** $\theta_{\mathrm{gate}}=\arccos(1-1/\beta)$; $\theta^{*}=2\arccos(1-\ln2/\beta)$; $\delta_{\mathrm{fuse}}\le\tfrac12$ on $\mathcal{R}$; $\ell_c\ge\mathrm{logaddexp}(\mathrm{base})-\Lambda_c$; $\Sigma(\beta)>0$ for a working gate.
**Train.** Probe P→A→B; τ once; straight-through over pairs, operations and gate.

## 19. Measured results

**19.1 Law and readout.** Real corpus, 73 923 items, d=384, train 66 963 / eval 6 960. ω=0.01047, Γ=0.1811 (λ=0.2, undershoot ≈9%), ε̄=0.00068, β=20.70, κ=−17.41, τ=0.6703. Prior CE oracle 2.2432, learned 2.2572, **gap +0.0139 nats**. Stage P collapse as designed (commit 0.000, basin 0.12); stage B p_true 0.935, acc 0.969.

**19.2 Readout, paired, 5 seeds, n=1005 present / 275 absent, train-fit threshold 0.420.**

| policy | all | present | absent | dPres | SE | z | dAbs | SE | z |
|---|--|--|--|--|--|--|--|--|--|
| gate | 0.0263 | 0.0239 | 0.0349 | −0.004531 | 0.003120 | **−1.45** | +0.006543 | 0.008693 | **+0.75** |
| kNN+thr | 0.0284 | 0.0285 | 0.0283 | ref | | | ref | | |
| kNN forced | 0.1649 | 0.0102 | 0.7303 | −0.018262 | 0.003435 | −5.32 | +0.701934 | 0.011068 | 63.4 |
| map | 0.1023 | 0.0127 | 0.4297 | −0.015728 | 0.003915 | −4.02 | +0.401380 | 0.022126 | 18.1 |
| cue-only | 0.2417 | 0.3078 | 0.0000 | +0.279364 | 0.003957 | 70.6 | −0.028344 | 0.007283 | −3.89 |

**The gate is indistinguishable from a tuned kNN+threshold on both axes.** The absent point estimate reversed sign versus an earlier CUDA run (gate 0.0220 vs kNN 0.0413). What survives is *beats cue-only and centroid decisively*.

**19.3 Capacity, spread, paired, 5 seeds, n=1043.**

| C | canonical | kNN+thr | z | ward | ward z | oracle |
|--|--|--|--|--|--|--|
| 4 | 0.1130 | 0.1141 | 0.45 | 0.1258 | +4.07 | 0.1148 (+2.28) |
| 8 | 0.0663 | 0.0668 | 0.09 | 0.0712 | +1.92 | 0.0665 |
| 16 | 0.0258 | 0.0246 | −0.45 | 0.0269 | +1.16 | 0.0255 |

Canonical ≡ `debit` ≡ `stream-lightest` ≡ `topk-learned` to four decimals. **`topk-oracle` is worse than `topk-learned`** at C=4 (+2.28σ): the generating prior is not the best eviction score, because the readout is calibrated to the learned dose. Single-batch diagnostics: absent 0.0057/0.0136/0.0136/0.0181 and retained 0.702/0.845/0.959/1.000 for C=4/8/16/32; ledger 18.4/8.6/1.9/0.09 nats.

**19.4 Duplicate sweep, paired, 5 seeds, C=4, T=100, REF=`debit`.**

| ρ | canonical | debit | ward | topk | kNN+thr | canonical−debit | z |
|--|--|--|--|--|--|--|--|
| 0.00 | 0.1128 | 0.1128 | 0.1201 | 0.1128 | 0.1156 | 0.000000 | tie |
| 0.25 | 0.0987 | 0.0987 | 0.1138 | 0.1100 | 0.1130 | −0.000003 | −1.64 |
| 0.50 | 0.0872 | 0.0881 | 0.0939 | 0.1093 | 0.0988 | −0.000974 | −1.30 |
| 0.75 | **0.0665** | 0.0739 | 0.0714 | 0.1031 | **0.0659** | **−0.007380** | **−3.77** |

At ρ=0.75: `topk−debit` +0.029259 ± 0.003466 (+8.44σ); `ward−debit` −0.002436 (−1.06σ, ns). CUDA replication: canonical−debit −0.007848 ± 0.002104 (−3.73σ), topk−debit +8.85σ.

**Decomposition.** Forced top-k 0.1031 → thresholded kNN 0.0659 (−36%) → canonical 0.0665. **Most of the gain over forced eviction is the readout threshold, not the compression policy.** The policy's own share is −0.0074 (7%). Against the *tuned* comparator, canonical wins at ρ=0.25 (−0.0143, ~4σ) and ρ=0.5 (−0.0116, ~2.7σ) and **ties at saturation** (+0.0006). Mechanism: after eviction a surviving near-copy has low dose, so the dose-weighted readout under-commits where dose-blind kNN returns the copy.

**19.5 Applied document demo.** 18 sentences, C=8, Δt=0, uniform importance, β=20.70. Geometry: gate cos ≥ 0.952, debit cos ≥ 0.868; max pair cos 1.000, median 0.201. Outcome: 2 merges, 8 evictions, 8 atoms covering 10 sentences, 8 destroyed, ledger 3.7612 nats, `agree-debit 2/2`.

- Verbatim block {1,8}: δ=0, retained 1.000, dose 0.0183 ≈ 2× single → heaviest → survived every later eviction; its query committed at gap **+9.105**. **Merging protects an idea by doubling its dose.**
- Near-verbatim block {11,12}: retained 0.595 → Λ=0.519 → implied cos ≈ 0.901, **below the gate window**; merged via the debit branch. Retrieval still fine (gap +3.894, cos(u) 0.980) — Theorem A reduced evidence without breaking the answer.
- Paraphrase at cos ≈ 0.84 was destroyed. **The readout's resolution is not semantic equivalence.**
- **Ledger non-comparability:** identical destroyed sets, 5.2172 nats (no merges) → 3.7612 (2 merges); per-eviction cost 0.652 → 0.405 because the heavy merged atom shrinks ρ.

**19.6 LM interface.** See §23.

**19.7 Unit checks.** 47/47. Thm A 0/20000 (residual −1.64e−5); Thm B μ err 5.6e−17, ℓ err 8.9e−16; Thm E excess 0; Thm E′ 0.251641 ≤ 0.251984 and 0.251400 ≤ 0.251529; Gram drift 6.66e−16 / 10³ ops; check 45 bit-identical (25 evictions); check 47 gate fires on 8000 pairs, |δΔ| 6.75e−14; check 43 3.55e−15; Lemma F identity 1.11e−16; hysteresis 9.237 / 9.500; closure fraction 0.313 ≤ 0.692; antipodal ledger 0.6926 ≈ ln2.

**19.8 Open limitations.** Canonical vs ward (~1.9σ); Γ undershoot ≈9%; prior-CE rise after P by design; compression-in-loop cost on the uncompressed readout; every kNN comparison device-sensitive; ledger non-comparable across merge counts; dose-weighting under-commits on low-dose survivors; gate reach excludes natural paraphrases at learned β.

## 20. Central hypothesis

1. Retrieval beats cue-only and centroid decisively; abstains on off-topic. *(MEASURED.)*
2. Forgetting rate recoverable from the marginal law only. *(MEASURED.)*
3. Vagueness is basin collapse at a computable time. *(THEOREM.)*
4. Compression is evidence-monotone, parameter-free, auditable, fixed-size, with bounded override. *(THEOREM + unit.)*
5. Decision precedes descent. *(THEOREM + witness.)*
6. The canonical policy reduces to eviction in spread geometry and strictly improves on the debit policy at saturation. *(MEASURED, two devices.)*
7. Merging beats a tuned comparator at moderate duplicate density and protects duplicated ideas by dose doubling. *(MEASURED.)*
8. LM retrieval fidelity and the surprise channel transfer. *(MEASURED.)*
9. The LM null prior transfers with the law frozen. *(FALSIFIED.)*
10. The latent is a Transformer interface supporting induction. *(NOT TESTED; architecturally out of scope.)*

**Position.** The contribution is not superior retrieval. It is a bounded associative state with an auditable, evidence-monotone compression channel that **matches** a tuned nearest-neighbour baseline while costing O(1) per token, beats forced eviction decisively, and beats a tuned comparator at moderate duplicate density — together with a characterized account of where the learned prior stops transferring.

## 21. Supersession

**21.1 Inherited (v2.2.1 items 01–30).** Small-step descent; crosstalk-as-limit; spectral merging; target-only blur; guard reachability; μ rounding; naive $1-e^{-x}$; cue-only sufficiency; no-new-basins; fixed-point floor independent of η,N; legacy adequacy; W-only retrieval; the $\sqrt{(T-1)/d}$ Gibbs floor; argmax-as-decision; half-closure vagueness; decorrelated-sampling sufficiency; one-κ-two-jobs; S-as-useful-memory; eval-fit τ; every-operation-convex; top-k-as-policy; pruning-as-exception; exact block odds; monotone max-gap; β-independent ledger; Loewner monotonicity; conserved S. All remain withdrawn.

**21.2 v2.3.1 → v2.3.2 (31–51).**

| # | Withdrawn | Replacement |
|---|---|---|
| 31 | Fusion must beat top-k on present at small C | Thm E + regime predictions |
| 32 | Null-fusion is decision-free | gate-dead branch only |
| 33 | Null-fusion restricted to gate-dead atoms | extended via derived comparison |
| 34 | C=1 antipodal fallback is a special case | subsumed |
| 35 | Streaming Gram rebuild O(C²d) acceptable | incremental law mandatory |
| 36 | Ledger $D_c=\sum(1-r)$, debit outside logaddexp | nats; merge-only; per-constituent ages |
| 37 | Per-atom relevance $R_i$ with EMA rate η | dose **is** the target law |
| 38 | $c^Q=\mathrm{Ward}\times(R_i+R_j)$ | expected-debit comparison |
| 39 | softmax(sims·10) relevance; $\sum R_i\approx C$ | arbitrary temperature; sums to ≈1 |
| 40 | Thm A proof without the debit factor | correct two-step proof |
| 41 | Periodic Gram re-orthogonalization | drift check + refresh |
| 42 | Timing-slope complexity self-check | deterministic identity checks |
| 43 | Gate beats kNN+thr on present (single run) | not established → item 52 |
| 44 | Theorem E for the canonical policy | narrowed to `debit`; E′ instead |
| 45 | `ward` beats `debit` at saturation | −0.80 / −1.78σ, noise |
| 46 | `debit` beats top-k-learned in spread | parity |
| 47 | `below_cross` = below-crossover fraction | `agree-debit`; 0% with 0 merges = no merges |
| 48 | `resfirst` is an ablation | promoted to canonical |
| 49 | Ledger ranks information loss across policies | use `retained`, `destroyed_frac` |
| 50 | Flat argmin indices test invariance | unordered-pair keys + operation |
| 51 | Gate's present advantage established | → item 52 |

**21.3 This issue (52–60).**

| # | Withdrawn | Evidence |
|---|---|---|
| 52 | Gate beats tuned kNN+thr on present **or absent** | z = −1.45 / +0.75, n≈1280; absent sign flipped across devices |
| 53 | Compression beats tuned kNN+thr at matched C, spread | ties at every C, \|z\| ≤ 0.45 |
| 54 | Duplicate gain attributable to the compression policy | mostly the readout threshold; policy share ~7%; tie at saturation |
| 55 | Ledger comparable across merge counts | 5.2172 → 3.7612 nats, identical destroyed sets |
| 56 | Resolution gate fires on natural paraphrases | needs cos ≥ 0.952 at β=20.7; MiniLM paraphrases ≈ 0.84 |
| 57 | β, κ transfer to LM states with law frozen | κ window empty 4/4; law must be re-fit |
| 58 | Layer choice is a hyperparameter | precondition: layer 12 unusable (50.5% eligible) |
| 59 | Block accuracy measures retrieval under compression | counts abstention on destroyed targets; condition on `surv` |
| 60 | Compression only degrades retrieval | trades coverage for commitability; net loss can be flat |

## 22. Changes and the structural findings

**22.1 Why the gate exists.** Lemma F: merging is always angularly cheaper. Proposition D: the debit branch ranks merges correctly. The failure was the **currency** — Δ charges eviction only the block's dose drop, so destroying a low-probability target looks cheaper than an unresolvable merge that would preserve it. The resolution gate overrides that comparison exactly where the readout cannot distinguish the pair; Theorem E′ bounds the override at half a nat. Confirmed twice: neutral where the currency is right, −2.93σ then −3.73σ where it is not.

**22.2 Why the prior stops transferring.** Proposition H makes the gate's requirement explicit: Σ(β) > 0. The dose prior orders by recency × surprise. On a task whose targets are *old and unsurprising*, present-target logits fall below absent-max logits and Σ < 0 for every reachable β. Readout-only refitting moves the overlap by ~7 points and the inversion by ~0.2 nats. **The law, not the readout, is what fails to transfer.**

**22.3 Deliberately unchanged.** State class; convex write; dose maps; diffusion laws; Gibbs readout and descent; decide-before-descend; probe split; reinforcement; decoder; legacy ablation; ledger currency and its caveat; the form of the LM interface.

## 23. Language-model interface (MEASURED IN PART)

**23.1 Latent and read — algebraic, unit-verified.** $(\mu_c,\ell_c)\in\mathbb{R}^{C\times(d+1)}$; the Gibbs posterior at $u=q$ is one cross-attention head with tied key/value,
$$u=\sum_c\mathrm{softmax}_c\big(\beta q\cdot\mu_c+\ell_c\big)\mu_c+p_0q,$$
C memory tokens, keys/values $W_K\mu_c,W_V\mu_c$, logit bias $\ell_c$ (a learned per-token ALiBi), null token bias κ, gate as residual mix $g=\sigma(\mathrm{gap}-\ln\tau)$. With identity projections this equals the Gibbs gradient at $u=q$ to <1e−12.

**23.2 Geometry precondition — MEASURED.** GPT-2, 256 tokens, gate window cos ≥ 1−1/β:

| layer | raw mean / p50 | raw eligible | centered mean / p50 | centered eligible | β@p99 centered |
|--|--|--|--|--|--|
| 0 | 0.418 / 0.406 | 0.5% | −0.002 / −0.030 | **0.3%** | 5.8 |
| 3 | 0.658 / 0.666 | 1.6% | 0.061 / 0.034 | **0.7%** | 15.4 |
| 6 | 0.574 / 0.574 | 1.6% | 0.046 / 0.015 | **0.8%** | 15.8 |
| 9 | 0.461 / 0.440 | 1.7% | 0.013 / −0.028 | 1.2% | 25.3 |
| 12 | 0.650 / **0.955** | **50.5%** | 0.125 / 0.703 | **10.7%** | 431.7 |

The distribution is heavy-tailed, not shifted: the gate is *selective* at layers 0–9 and β=20.7 is usable there. **Layer 12 is unusable at any reasonable β** (raw β@p99 = 3914). Rule: tap a middle layer, center, require gate-eligible ≲ 2%.

**23.3 Retrieval fidelity — MEASURED, transfers.** Angular error on committed present trials:

| | cue-only | kNN | HopMem ∞ | C=32 | C=16 | C=8 |
|--|--|--|--|--|--|--|
| natural query | 0.1586 | 0.0026 | **0.0097** | 0.0097 | 0.0123 | 0.0444 |
| angle 0.7 | 0.2352 | 0.0000 | **0.0104** | 0.0104 | 0.0194 | 0.0526 |

16–23× better than the cue, within ~0.01 of kNN, degrading gracefully with compression.

**23.4 Surprise channel — MEASURED, works.** $s_t=-\ln p_{\mathrm{LM}}(x_t\mid x_{<t})$ into $a=1-e^{-\omega s}$ retains high-NLL content: top doses `compression` (8.84), `by` (5.86), `decide` (8.35); bottom `of` (0.85), `n` (1.19). Verified analytically: $a(8.84)e^{-2\Gamma}=0.0884\times0.696=0.0615$, matching 0.06149. The single gate-dead deletion is the $s_0=0$ token at the floor — check 10 in the wild. One merge, ledger 0.039 nats → cos ≈ 0.996, from the repeated span.

**23.5 The null prior does not transfer — MEASURED, FALSIFIED.** Proposition H's window is **empty in 4/4 configurations**:

| config | κ window | Σ (nats) | present below absent median |
|--|--|--|--|
| natural, transferred | [−9.76, −27.33] | −17.57 | 59.0% |
| natural, recalibrated | [−11.59, −28.93] | −17.34 | 51.9% |
| angle 0.7, transferred | [−11.40, −26.02] | −14.62 | 57.9% |
| angle 0.7, recalibrated | [−13.11, −26.94] | −13.83 | 49.8% |

Present targets are **old** (ℓ ≈ −17), absent-max is **recent** (ℓ ≈ −2.3); the required order is inverted. Fitted β 26.12 / 24.62 (predicted median 23.3 / 21.3), κ −11.71 / −13.55, τ 3.32 / 0.2019. The required-κ offset flips sign across calibration (+4.69 → −3.64), the signature of overshoot on an inseparable problem. Live dose spread 14.79 nats; present unreachable at any β 1.4% / 0.0%.

**23.6 Coverage/commitability trade — MEASURED.** Among *surviving* targets, ranking and commitability improve monotonically as C shrinks (`blk|surv` 0.300→0.643→1.000→1.000; `cmt|surv` 0.219→0.459→0.946→1.000) while coverage collapses (`surv` 1.000→0.467→0.176→0.043). Net present loss is roughly flat under the natural query and monotone worse under the angle probe. **Always report `surv` beside block accuracy.**

**23.7 Architectural limit.** Tied key/value reads the same μ for both projections, so with identity weights the interface returns the **matched** past state, not its continuation. Content-addressed recall, **not induction**. Untie $W_K/W_V$ and learn a shift to test induction.

**23.8 Not measured.** Perplexity benefit; learned projections; end-to-end integration; induction.

## 24. Self-check suite (47 checks)

**State and dynamics (01–10).** Invariants under write/diffuse/compress; closed form vs sequential; semigroup with **cumulative absolute times**; spectral invariance comparing the **top eigenvector only**; steady state; impure-write decomposition; empty memory returns q; permutation invariance; uniform log-shift invariance; s=0 finite log-dose with zero clamped gradient against a **torch-computed** floor reference at 1e−6.

**Readout and gate (11–18).** Energy-increase 0 and guard 0 over a β grid; N vs 10N agreement; planar hysteresis witness; gate limits; streaming vs batch gated equality with compression active; τ determinism and flat-sweep-to-argmax; basin/floor/resolution; cue-free gradient restoration.

**Legacy and numerics (19–26).** Factored vs dense corrected stability including the saddle; legacy descent for every η and W/M equivalence; capacity–closure witness via the angle-closing fraction; masked denominators with `enable_grad`; stable inverse softplus; −expm1 accuracy; decoder Jacobian bound; no-match bounds κ.

**Compression (27–36).** C≥T bit-for-bit regression; Thm A 0/20000; Thm B; merge accounting (trace 1, PSD, deficit→J); gate-dead criterion; gate-dead decision-invariance; nats ledger single-merge exactness and chain bound; streaming vs batched replay; C=1 antipodal evict branch with ledger ≈ ln2; Prop C first-order and the capacity bound.

**Policy (37–44).** Spread reduction for `debit`; duplicate free-merge; crossover identity; Thm E for `debit`; incremental Gram over 10³ mixed ops; ledger/destroyed semantics; **debit commutation on unordered-pair keys**; Lemmas F and G.

**Canonical gate (45–47).** 45: with $\mathcal{R}=\emptyset$, canonical and `debit` produce **bit-identical** Z, ℓ, Λ, alive masks, provenance and stats, zero merges. 46: Thm E′ bounds over 2×10⁴ gate-eligible draws, 0 violations, both the ½-nat and sharp forms. 47: on a state that *does* contain below-resolution pairs, $\mathcal{R}$, the selected unordered pair, the operation and Δ are invariant under a uniform log-shift (Δ tolerance 1e−11, justified by cancellation in $\ell_l-\ell_h$).

---

**End of consolidated spec v2.3.2.** One gate, one comparison, zero new parameters:
$$\text{merge if }\beta(1-G)\le1,\quad\text{else}\quad \arg\min_{(i,j)}\frac{A}{S}\min\big(\beta(1-r),\ \mathrm{softplus}(\ell_l-\ell_h)\big).$$
The override costs at most half a nat and buys back provenance. The ledger is an audit, not an objective. The compression claims are measured on two devices; the retrieval-superiority claims are withdrawn; and the LM prior's failure to transfer is characterized by a signed margin rather than a caveat.

**Next experiment worth running.** Re-fit Γ on the LM recall task and check whether Σ(β) turns positive. Proposition H says it should if the dose ordering stops fighting the task; §9's transfer protocol says the law must be re-fit for that. That is the one open question where a new positive claim is available.