# HopMem v2.3.2

**Resolution-gated compression: a provenance-preserving canonical policy with a bounded override cost, and the parity claim.**

*Measured results revised after the post-promotion confirmation run. No policy, code or theorem change from the promoted v2.3.2 — this revision updates §9, §15, §17.6, §19 and §21 with the confirmation data and marks the promotion prediction MEASURED.*

This specification is self-contained and supersedes v2.3.1. The canonical compression policy is **resolution-gated merge-first, then debit argmin**; **Theorem E is narrowed to the `debit` ablation** and replaced for the canonical policy by **Theorem E′ (forced debits ≤ ½ nat)**.

**Motivating measurement (pre-promotion, paired, 5 seeds, n≈1046).** The gate was exactly neutral in spread geometry and better at saturated duplicate density: −0.005554 ± 0.001898 (**−2.93σ**) at $\rho_{\mathrm{dup}}=0.75$, every cell ≤ 0.

**Confirmation measurement (post-promotion, paired, 5 seeds, independent batches).** The effect strengthened: −0.007848 ± 0.002104 (**−3.73σ**) at $\rho_{\mathrm{dup}}=0.75$, exact tie at $\rho_{\mathrm{dup}}=0$, monotone in between, no cell worse. Against offline top-$k$ the canonical policy's gain is −0.0399 (**36.6%**), with the debit–topk component alone at +8.85σ. Two v2.3.1 readings remain withdrawn as single-batch noise: "ward beats debit at saturation" and "debit beats top-$k$ in spread geometry."

**Design stance.** Zero new parameters. The gate threshold is $\beta(1-G_{ij})\le1$ — the readout's own resolution scale. The policy trades *unbounded ledger dominance* for a *bounded per-event override* plus provenance preservation, and that trade is a theorem.

---

## 1. Scope

**Supports.** Cue-conditioned retrieval with abstention; learned recency ($\Gamma$) and surprise ($\omega$) forgetting; sharp basin-collapse forgetting; a strictly fixed-size state of $C(d+2)+C^2$ floats; $O(1)$ read/write in stream length for fixed $C,d$; pointwise-evidence-monotone, auditable compression; a **bounded** override cost relative to the debit-optimal policy (Theorem E′); an exact $O(C)$ Gram merge law; a latent Transformer interface (§23).

**Non-goals.** Exact recall below resolution; order-independence of greedy selection; a $\beta$-independent ledger; monotone max-atom gap (block mass only); decision-invariance of live-atom eviction; gradual per-target loss under eviction (abrupt by construction); Loewner-monotone $W$; **ledger dominance for the canonical policy** (Theorem E, `debit` only); present-loss superiority over top-$k$ in spread geometry (withdrawn — the claim is parity); **a replicated present-trial advantage of the gate over kNN+threshold** (§19.5).

**Conceptual status.** The density-matrix language is a real-symmetric analogue, not a claim of quantum physics. The state of record is a trace-one PSD matrix; the canonical representation is $\le C$ log-atoms; the readout is a log-linear Gibbs model whose log-priors obey convex, dose-additive dynamics with three additive forgetting channels.

## 2. Notation and quickstart

| Symbol | Meaning |
|---|---|
| $d$, $C$, $T$, $N$ | embedding dim; architectural capacity; episode length; descent steps (20) |
| $\alpha_i=e^{\ell_i}$, $S=\sum_i\alpha_i$ | dose; stored mass, $S\in[0,1)$ |
| $J=I/d$; $W=(1-S)J+M$; $M=\sum_i\alpha_i z_iz_i^\top$ | uniform state; state; memory |
| $\omega,\Gamma,\bar\varepsilon$; $\beta,\kappa$; $\tau$ | law; readout; calibrated gate threshold |
| $G_{ij}=z_i\cdot z_j$ | cached Gram, incrementally updated (§4.6) |
| $A=\alpha_i+\alpha_j$, $w_i=\alpha_i/A$ | pair mass; fusion weight |
| $m=w_iz_i+w_jz_j$, $r=\|m\|$, $\mu=m/r$ | resultant; length; direction |
| $h$, $l$, $\rho=\alpha_l/\alpha_h$ | heavier / lighter member; mass ratio |
| $\delta_{\mathrm{fuse}},\delta_{\mathrm{evict}}$ | the two derived debits (§4.5) |
| $\Delta=(A/S)\,\delta_{\mathrm{op}}$ | expected block-nats removed |
| $\Lambda_c=\sum\delta_{\mathrm{fuse}}$ | per-surviving-block ledger, **nats** |
| $p_i=\alpha_i/S$ | identified target law |
| $c_{ij}=\frac{\alpha_i\alpha_j}{A}(1-G_{ij})$ | Ward cost |
| $\mathcal{R}$ | resolution-eligible pair set (§4.5) |
| $\epsilon_r$ | resultant mask ($r<\epsilon_r$ excludes the merge candidate) |

```
python hopmem.py selftest                                    # 47 checks (§24)
python hopmem.py train --schedule probe --batch 256 --C 16 --c_min 4
python hopmem_diag.py  --ckpt hopmem_theta.pt --T_big 100 --caps 4 8 16 32 --dup_sweep
python hopmem_stats.py --ckpt hopmem_theta.pt --seeds 5 --dup --dup_only --ref debit
```

Deployment: `mem = HopMem(p, d, C, tau)`; `mem.write(z,s,t)`; `mem.retrieve(q,t_q)`. Default policy `resfirst`; `debit` remains selectable as the ablation satisfying Theorem E.

## 3. State

### 3.1 Invariants
$W\succeq0$, $\mathrm{tr}\,W=1$, $S\in[0,1)$, $\lambda_{\min}(W)\ge(1-S)/d>0$ in exact arithmetic. Initialize $W=J$; never $W=0$. Store $(z_i,\ell_i,\Lambda_i)$, $S$, a lazy absolute-time clock, and $G$. Never materialize $M$ or $W$ outside diagnostics or the legacy readout.

### 3.2 Canonical log-dose
Between compressions,
$$\ell_i(t_q)=\ln\left(a_i(1-\varepsilon_i)\right)-\omega\sum_{j>i}s_j-\Gamma(t_q-t_i),\quad a_i=1-e^{-\omega s_i},\ \varepsilon_i=\bar\varepsilon(1-a_i).$$
For a compressed block the **exact** dose is the recursion of §4.5. The closed form uses **per-constituent ages inside** the logaddexp, and
$$\boxed{\;\ell_c\;\ge\;\mathrm{logaddexp}(\mathrm{base})-\Lambda_c\;}$$
is **exact for one merge, conservative for chains**. *Proof of the chain step:* $\mathrm{logaddexp}(x-a,y-b)\ge\mathrm{logaddexp}(x,y)-\max(a,b)\ge\mathrm{logaddexp}(x,y)-(a+b)$ for $a,b\ge0$. ∎ Hence $e^{-\Lambda_c}$ is a **lower bound** on retained dose fraction. Measured: single-merge error $1.8\times10^{-15}$; chain slack $\le0.27$ nats over 300 random trees.

### 3.3 Ledger (nats)
Per surviving block $\Lambda_c=\sum\delta_{\mathrm{fuse}}$; episode ledger $\sum\delta=\sum\delta_{\mathrm{fuse}}+\sum\delta_{\mathrm{evict}}$; per target retained $=e^{-\Lambda_{c(t)}}$ if surviving, **hard $0$** if destroyed. A destroyed block never carries a finite $\Lambda$. Debits use the current $\beta$; recompute in-forward during training, freeze at deployment.

**Currency caveat (load-bearing).** $\Delta$ is expected *block log-dose reduction*, a proxy for loss. It charges eviction only the block's dose drop, not the evicted target's total loss of retrievability. The canonical policy exists because this currency misprices eviction in duplicate-rich streams (§22.1). The ledger is an audit of nats, **not** a ranking of information loss across policies; use `retained` and `destroyed_frac` for that.

### 3.4 Atom-dependence
The Gibbs readout depends on the atom set, not on $W$ alone. Compression is defined on atoms; no Loewner claim.

## 4. Operation alphabet

{diffuse, write, reinforce, **fuse**, **null-fuse**}. Order per event: diffuse → write → reinforce → compress-on-overflow.

### 4.1–4.3 Diffuse, write, reinforce (retained)
Lazy diffuse with an **absolute-time** clock: $\Delta t=\max(t-t_{\mathrm{clock}},0)$, $\ell_i\leftarrow\ell_i-\Gamma\Delta t$, $S\leftarrow S e^{-\Gamma\Delta t}$. Write: $a=-\mathrm{expm1}(-\omega s)$, shift all $\ell_j\leftarrow\ell_j-\omega s$, append $\ln\max(a(1-\varepsilon),10^{-30})$; exact impure decomposition $a[(1-\varepsilon)zz^\top+\varepsilon J]$. Reinforce: masked denominator, confidence $\mathrm{cf}$, counts toward $C$.

### 4.4 Null-fusion — two triggers
**(a) Gate-dead (free, decision-free).** Atom $i$ can never trigger a commit for any cue iff $\ell_i\le\kappa+\ln\tau$; deleting it leaves every gate decision unchanged. Applied first on overflow.
**(b) Debit-eviction (decision-changing).** A live lighter atom is deleted when §4.5 selects eviction. Not covered by decision-invariance; Theorem A still bounds its evidence.

### 4.5 The compression policy (canonical)

**Trigger.** Atom count $>C$ after write/reinforce.

**Step 1.** Null-fuse all gate-dead atoms (free).

**Step 2 — resolution gate.** For every alive unordered pair with $r^2=1-2w_iw_j(1-G_{ij})$, define
$$\mathcal{R}=\big\{(i,j):\ r\ge\epsilon_r,\ \beta(1-G_{ij})\le1\big\}.$$
If $\mathcal{R}\ne\emptyset$, **merge** $\arg\min_{\mathcal{R}}\Delta_{\mathrm{fuse}}$. Otherwise fall through.

**Step 3 — debit argmin.** With
$$\boxed{\;\delta_{\mathrm{fuse}}=\beta(1-r)=\frac{2\beta w_iw_j(1-G_{ij})}{1+r},\qquad \delta_{\mathrm{evict}}=\ln\frac{A}{\alpha_h}=\mathrm{softplus}(\ell_l-\ell_h)\;}$$
$$\boxed{\;\Delta(i,j,\mathrm{op})=\frac{A}{S}\,\delta_{\mathrm{op}},\qquad (i^{*},j^{*},\mathrm{op}^{*})=\text{the minimizing pair and operation}\;}$$
execute the winner. Since $1/S$ is constant per episode the argmin is over $A\,\delta$; compare in log space, $\ln\Delta=\mathrm{logaddexp}(\ell_i,\ell_j)-\ln S+\ln\delta_{\mathrm{op}}$. Repeat until count $\le C$.

**The two maps.** *fuse*: $\mu=m/r$, $\ell'=\mathrm{logaddexp}(\ell_i,\ell_j)-\delta_{\mathrm{fuse}}$, deficit $A(1-e^{-\delta})$ to $(1-S)J$, $\Lambda'=\Lambda_i+\Lambda_j+\delta_{\mathrm{fuse}}$, provenance unions. *evict*: delete $l$, $h$ untouched, book $\delta_{\mathrm{evict}}$, mark $l$'s provenance destroyed.

**Debit semantics.** $\delta_{\mathrm{fuse}}$ is a *sufficient* uniform debit for Theorem A, tight at equal masses (binding at $u=\mu$), conservative otherwise. $\delta_{\mathrm{evict}}$ is the *exact* block log-dose reduction and the unique debit leaving the survivor's evidence identical; the smaller angle-dependent minimum $\ln\!\big(A/(\alpha_h+\alpha_le^{-\beta\sqrt{2(1-G)}})\big)$ inflates the survivor and is forbidden.

**Two derived angles, no new constants.** $\theta_{\mathrm{gate}}=\arccos(1-1/\beta)\approx\sqrt{2/\beta}$; the debit branch's equal-mass crossover is $\theta^{*}=2\arccos(1-\ln2/\beta)$. At $\beta=20.70$: $\theta_{\mathrm{gate}}=0.312$, $\theta^{*}=0.519$. The gate is tighter at equal masses; its overrides occur for unequal masses whose pair is nonetheless unresolvable.

**Representational symmetry.** $(i,j)$ and $(j,i)$ denote the same unordered pair and the same operation. Their floating-point costs differ by ~1 ulp, so a flat argmin may return either representative. Invariance claims (§24 checks 43, 47) are stated on the **unordered pair key and the operation**, never on the flat index.

**Edge cases.** $s=0$: gate-dead, free. $r<\epsilon_r$: excluded from $\mathcal{R}$ and from merge candidates; eviction always admissible, so the $C{=}1$ antipodal fallback is subsumed. $C\ge T$: nothing fires; bit-for-bit v2.1.1 regression.

### 4.6 Incremental Gram law (mandatory)
On a merge into slot $i$, using the **old** rows, for every alive $k\ne i,j$:
$$\boxed{\;G_{ik}\leftarrow\frac{w_iG_{ik}+w_jG_{jk}}{r},\quad G_{ki}\leftarrow G_{ik},\quad G_{ii}\leftarrow1,\quad \text{row/col } j\leftarrow0\;}$$
Exact by linearity; $O(C)$, no $d$-dimensional products. Two guards: the formula does not yield $1$ at $k=i$, so set it explicitly; zero row/col $j$ **after** computing the new row. On write: one new row, $O(Cd)$. Full rebuild is a drift diagnostic only; refresh if any $|G_{ij}|>1+10^{-6}$. Measured drift $6.7\times10^{-16}$ over $10^3$ mixed operations.

## 5. Diffusion: six exact laws
D1 semigroup; D2 uniform log-drift; D3 mass decay; D4 spectral invariance; D5 commutation; D6 forgetting times $t_{\mathrm{forget}}=(\ell_k(0)-\Lambda_k-\kappa)/\Gamma$.

**D5 extended to the canonical policy.** Under a uniform log-shift, $A/S$, $w$, $r$, $\ell_l-\ell_h$ **and the gate predicate $\beta(1-G)\le1$** are invariant, hence $\mathcal{R}$, $\Delta$ and the selected unordered pair/operation are invariant. The gate-dead branch remains clock-dependent. Steady state $S^{*}=a(1-\varepsilon)/(1-(1-a)e^{-\Gamma\Delta})$; report the decay split $\omega\mathbb{E}[s]+\Gamma\mathbb{E}[\Delta t]$.

## 6. Writing details
$-\mathrm{expm1}$; clamp $s\ge0$; $\ell=\ln\max(a(1-\varepsilon),10^{-30})$ finite at $s=0$ with zero clamped-branch gradient; $\bar\varepsilon\equiv0$ via the `pure_writes` flag (sigmoid parameters never reach 0).

## 7. Parameters
Law $\omega,\Gamma,\bar\varepsilon$; readout $\beta,\kappa$; gate $\tau$; capacity $C$ architectural. **Compression adds zero parameters**: both debits and both angles derive from $\alpha,G,\beta,S$.

$\tau$ is calibrated **once** by a two-pass fixed point in $\tau_{\mathrm{dead}}$ on one fixed compressed training batch, saved, applied identically by `evaluate` and streaming `retrieve`; grid $\tau_i=e^{-6+0.4i}$, flat sweep → argmax. $\bar\varepsilon,\chi$ are sigmoid-parameterized, strictly in $(0,1)$.

**Identifiability (measured).** $\Gamma$ from the marginal law, $\beta,\kappa$ from the conditional; ambiguous cues required; prior CE rising after P is expected (two-job $\kappa$).

## 8. Readout

### 8.1–8.4 (retained)
$F(u)=\beta^{-1}\ln(e^{\beta q\cdot u+\kappa}+\sum_i\alpha_ie^{\beta z_i\cdot u})$ convex; $u^{+}=\nabla F/\|\nabla F\|$ ascends for every $\beta>0$, no step size; energy-increase fraction 0; masked float guard; empty memory returns $q$. $\theta_{\mathrm{basin}}=\arccos(1-(\ell_k-\Lambda_k-\kappa)/\beta)$; floor $\approx(\bar\alpha/\alpha_k)e^{-\beta}\sqrt{T-1}$. **Decide before descend:** commit iff $\mathrm{gap}(q)>\ln\tau$ at $u=q$. Witness: $\beta=16,\ell=-3,\kappa=-6.5$, atom at $0.70$ rad → $9.237$ vs $9.500$, null wins at $u=q$, map commits. The mass gate $\ln(E_{\mathrm{mem}}/E_0)$ is the monotone ablation.

### 8.5 Legacy quadratic (ablation only)
One stable attractor; $\mathrm{closure}_{\max}\approx1/(1+2\sqrt{(T-1)/d})$; the witness measures the angle-closing fraction at nonzero cue angle.

### 8.6 Compression theorems

**Theorem A (evidence domination — both operations).** For every unit $u$,
$$e^{\ell'+\beta\mu\cdot u}\le e^{\ell_i+\beta z_i\cdot u}+e^{\ell_j+\beta z_j\cdot u}\ \text{(fuse)},\qquad E_{\mathrm{mem}}'\le E_{\mathrm{mem}}\ \text{(evict)}.$$
*Proof (fuse).* $E'=Ae^{\beta(\mu\cdot u-(1-r))}\le Ae^{\beta r\mu\cdot u}=Ae^{\beta m\cdot u}\le\alpha_ie^{\beta z_i\cdot u}+\alpha_je^{\beta z_j\cdot u}$; first step $(1-r)(\mu\cdot u-1)\le0$, second Jensen. *Proof (evict).* $E'\equiv E_h$. ∎
**Corollaries.** Block posterior mass non-increasing; null mass non-decreasing — compression never fabricates confidence. Max-atom gap may rise by $\le\ln2$ under fuse; $\tau$ recalibrates.

**Theorem B (commutation).** Uniform channels commute with both maps, with the debit argmin, and with the resolution gate.

**Proposition C (Ward = expected angular damage).** $\mathbb{E}[1-z_{\mathrm{target}}\cdot\mu]=c_{ij}/S+O((1-G)^2)$.

**Proposition D (first-order expected debits).** $\Delta_{\mathrm{fuse}}=\beta c_{ij}/S+O((1-G)^2)$; $\Delta_{\mathrm{evict}}=p_l(1+\rho/2+O(\rho^2))$.

**Lemma F (angular ordering).** $\mathbb{E}[\text{damage}\mid\mathrm{fuse}]=p_lw_h(1-G)$ vs $p_l(1-G)$ for evicting $l$: merging is always angularly cheaper by $w_h\in[\tfrac12,1)$. Eviction competes only because it levies no debit on the survivor.

**Lemma G (cheapest eviction, exact).** $f'(\alpha_h)=\ln(1+\rho)-\rho<0$ and $\partial f/\partial\alpha_l=\ln(1+\rho)+1>0$, so $\Delta_{\mathrm{evict}}$ is minimized exactly at (lightest, heaviest).

**Theorem E (debit dominance — `debit` policy only).** With $l^{*}$ lightest and $h^{*}$ heaviest, $(\mathrm{evict},l^{*},h^{*})$ is a candidate and by Lemma G the cheapest eviction, hence $\Delta_{\mathrm{debit}}\le\Delta_{\mathrm{evict}}(l^{*},h^{*})=p_{l^{*}}(1+O(\rho))$. **The canonical policy does not satisfy this.**

**Theorem E′ (bounded forced debit).** Any gate-selected merge satisfies
$$\delta_{\mathrm{fuse}}=\beta(1-r),\qquad 1-r=\frac{t}{1+r}\le t=2w_iw_j(1-G)\le\tfrac12(1-G),$$
using $w_iw_j\le\tfrac14$; hence $\delta_{\mathrm{fuse}}\le\tfrac12\beta(1-G)\le\tfrac12$ **nat**. Sharper, from $t\le1/(2\beta)$:
$$\delta_{\mathrm{fuse}}\le\beta\Big(1-\sqrt{1-\tfrac{1}{2\beta}}\Big)\xrightarrow[\beta\ \text{large}]{}\tfrac14 .$$
∎ At $\beta=16$ the sharp bound is $0.251984$; measured maximum $0.251641$ over $2\times10^4$ gate-eligible draws. At the deployed $\beta=20.70$: bound $0.251529$, measured $0.251400$.

**Corollary (bounded override).** Per overflow event, $\Delta_{\mathrm{canonical}}-\Delta_{\mathrm{debit}}\le\frac{A}{S}\delta_{\mathrm{fuse}}\le\tfrac12$ expected nat, since $A/S\le1$. The canonical policy pays at most half an expected nat more than the debit-optimal policy and preserves provenance that eviction destroys outright.

**Corollary (regime reduction).** If $\mathcal{R}=\emptyset$ the canonical policy *is* the debit policy; if additionally $\min\Delta_{\mathrm{fuse}}\ge\Delta_{\mathrm{evict}}(l^{*},h^{*})$ its trajectory equals streaming dose-ordered eviction exactly. **Measured: bit-identical at every tested capacity on the real corpus** (§19.2).

**Corollary (heavy-atom protection).** The evict branch always removes the lighter member; the heavier atom's direction and dose are untouched.

## 9. Training objective
Posterior CE at $u=q$ with the null class; labels map to the target's fused block; **evicted targets map to null**. No-match episodes required. Probe: P cue-free compression-off ($\omega,\Gamma,\kappa$); A cue CE on compressed states ($\beta,\kappa$), $C\sim U\{C_{\min},\ldots,T\}$; B refine ($\bar\varepsilon,\beta,\kappa$), law frozen.

**Gradients.** Selection of the pair, the operation and the gate branch is hard argmin with straight-through. Gradients flow through $\delta_{\mathrm{fuse}}$ (which enters $\ell'$) and surviving doses; $\delta_{\mathrm{evict}}$, the gate predicate and all argmins are selection-only and carry **no gradient**; the evict branch has no path through the deleted atom. Ledger bookkeeping is detached.

**Recorded observation — the gate is active in training, inert at deployment.** With $C\sim U\{4,\ldots,20\}$ and $T=20$, small-$C$ draws put near-duplicate sentences of the same passage below $\theta_{\mathrm{gate}}$, so the gate fires and changes the compressed states, hence the gradient trajectory. At the deployment capacity the gate is inert (`agree-debit 100%`, zero merges). Consequently the canonical and `debit` policies yield **different checkpoints** even though they agree at evaluation capacity. This is a real mechanism, not seed noise, and it is why cross-version checkpoint comparisons are unpaired (§19.5).

**Recorded cost.** Compression-in-loop training gave a slightly worse *uncompressed* readout than the v2.1.1 compression-free checkpoint. Kept honest; under study.

## 10. Episodes and schedules
$s_i\sim\mathrm{Exp}(1)$; arrivals $\mathrm{Exp}(1)$; $t_q=t_T+\mathrm{Exp}(1)$; target $\sim\mathrm{softmax}(\ln s-\lambda(t_q-t))$, $\lambda=0.2$; $p_{\mathrm{absent}}=0.2$; cue = random contiguous half-span; same-passage competitors; warn at $p_{\mathrm{true}}>0.99$.

**Duplicate-heavy generator.** With probability $\rho_{\mathrm{dup}}$, $z_{\mathrm{new}}=\mathrm{normalize}(z_{\mathrm{prev}}+\sigma\xi)$, $\sigma=\tan(\theta^{*}/4)$; sweep $\rho_{\mathrm{dup}}\in\{0,0.25,0.5,0.75\}$. At $\beta=20.70$ the copies sit at $0.130$ rad, well inside $\theta_{\mathrm{gate}}=0.312$.

## 11–13. Reinforcement, decoder, blending (retained)
Masked denominators; the decoder consumes the **gated** output; normalization Jacobian $\le1/\epsilon$. Compression row: merge debits the basin by $\delta_{\mathrm{fuse}}/\beta$ (gradual); evict removes the block (abrupt); ledger $\Lambda$ audits both.

## 14. Guaranteed properties

**Exact arithmetic.** All state, readout and gate guarantees; Theorem A for both operations; Theorem B including the gate; Propositions C, D; Lemmas F, G; **Theorem E′ and the bounded-override corollary for the canonical policy**; Theorem E for `debit`; the exact incremental Gram identity; gate-dead decision-invariance; $C\ge T$ regression; strict atom bound; the conservative chain bound.

**Not guaranteed.** Exact recall; unique basins; resolution below $\sim1/\beta$; $\Gamma$ identification from the conditional alone; monotone $S$, purity or max-atom gap; Loewner monotonicity; order-independence; a $\beta$-independent ledger; decision-invariance of live eviction; gradual per-target loss under eviction; ledger dominance for the canonical policy; present-loss superiority over top-$k$ in spread geometry; a replicated gate advantage over kNN+threshold on present trials.

## 15. Implementation rules
01–14 retained (normalize; absolute-time lazy clock; masked denominators; $-\mathrm{expm1}$; stable inverse softplus; gate in the inference path; $\tau$ once, saved; energy-increase counter 0; decomposed baselines).
15. **Policy:** gate-dead null-fusion; then the resolution gate; then the debit argmin. Never top-$k$ truncate *as the policy*.
16. Debits and $\Delta$ exactly as §4.5, compared in log space; ledger in nats.
17. **Incremental Gram mandatory**, both guards; rebuild only as a drift diagnostic.
18. Evict deletes the lighter; heavier untouched; destroyed provenance → retained $0$.
19. $\epsilon_r$ masks merge candidates only; eviction always admissible.
20. Debits use the current $\beta$; recompute in-forward; freeze at deployment.
21. P compression-free; A/B with compression; $\tau$ two-pass on one fixed batch.
22. Report $\Lambda$, retained, destroyed fraction, merge/evict counts, and **`agree-debit`** = fraction of merges the debit branch would also have chosen. **Convention:** when `n_fuse = 0` the ratio is $0/1=0\%$; read this as *no merges*, not *total disagreement*. The override count is `n_fuse` $\times(1-$`agree-debit`).
23. Self-checks compare like-for-like transcendentals (torch-vs-torch for floor constants, tolerance $\ge10^{-6}$); gradient checks wrapped in `enable_grad` inside `no_grad` suites; masked $\pm\infty$ entries excluded from difference statistics; **policy-invariance checks compare unordered-pair keys and operations, never flat argmin indices** (§4.5).

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

All $O(1)$ in stream length for fixed $C,d$. "O(1) memory" is claimed **only** in stream length. Design rule $C\le d/2$ keeps the state within $d^2$ floats.

## 17. Evaluation plan

**17.1** Decomposed baselines enforced; kNN + one train-fit threshold is the like-for-like gate comparator; beating claims must be made on present trials; **paired** differences with SE over episodes.
**17.2–17.5** Retrieval battery; init arms R/A; identifiability protocol; ablations `--fuse_policy {resfirst, debit, ward, lightest, random, delta}`.
**17.6** Capacity regime: $T_{\mathrm{big}}\in\{100,500,1000\}$, $\ge5$ seeds, paired SEs, query-aware keep-target upper bound, per-$C$ $\tau$ recalibration as a train-batch diagnostic only.

**Predictions and status.**
1. **Spread geometry — parity. MEASURED.** Canonical equals streaming dose-ordered eviction exactly and offline top-$k$ within noise: $|d/\mathrm{SE}|\le0.85$ at $C\in\{4,8,16,32\}$, $T_{\mathrm{big}}=100$; earlier $500/1000$ sweep $\le1.62$. *Not* a win.
2. **Duplicate-rich streams — strict gain over eviction. MEASURED.** top-$k$ − debit $=+3.59\sigma/+6.02\sigma/+8.85\sigma$ at $\rho=0.25/0.5/0.75$; canonical adds a further $-3.73\sigma$, for a total $-0.0399$ (**36.6%**) at $\rho=0.75$.
3. **Gate neutrality and saturation gain over `debit`. MEASURED.** Bit-identical when $\mathcal{R}=\emptyset$ (check 45; every spread capacity); $\Delta=0.000000\pm0.000000$ at $\rho=0$; $-0.001911\pm0.001018$ ($-1.88\sigma$) at $\rho=0.5$; $-0.007848\pm0.002104$ ($\mathbf{-3.73\sigma}$) at $\rho=0.75$; no cell worse.
4. **Bounded override. THEOREM + unit-MEASURED.** Every forced merge $\le\tfrac12$ nat; sharp bound respected with margin at two temperatures.
5. **Within-policy absent loss monotone in $C$. MEASURED.** $0.0057\to0.0136\to0.0136\to0.0181$ for $C=4/8/16/32$. The cross-policy "absent inversion" is a reporting artifact.
6. **Prior-CE gap unchanged. MEASURED.** $+0.0139$ nats.
7. **Canonical vs `ward` at saturation. OPEN.** Point estimate $-0.0038$, $\approx1.9\sigma$; needs $\ge8$ seeds or a direct `--ref ward` run.

## 18. Core mathematical summary
**State.** $W=(1-S)J+M$; $\le C$ log-atoms; ledger $\Lambda$ in nats.
**Compress.** Gate-dead first; then merge $\arg\min_{\mathcal{R}}\Delta_{\mathrm{fuse}}$ if $\mathcal{R}\ne\emptyset$; else $\arg\min\frac{A}{S}\min(\beta(1-r),\mathrm{softplus}(\ell_l-\ell_h))$.
**Gram.** $G_{ik}\leftarrow(w_iG_{ik}+w_jG_{jk})/r$, then $G_{ii}\leftarrow1$, then zero row/col $j$.
**Retrieve.** Gibbs $F$; decide at $u=q$, then descend.
**Limits.** $\theta_{\mathrm{gate}}=\arccos(1-1/\beta)$; $\theta^{*}=2\arccos(1-\ln2/\beta)$; $\delta_{\mathrm{fuse}}\le\tfrac12$ on $\mathcal{R}$; $\ell_c\ge\mathrm{logaddexp}(\mathrm{base})-\Lambda_c$.

## 19. Measured results (confirmation run)

**19.1 Law and readout.** Real corpus, 73 923 items, $d=384$, train 66 963 / eval 6 960, CUDA. $\omega=0.01047$, $\Gamma=0.1811$ (data $\lambda=0.2$; undershoot ≈ 9%), $\bar\varepsilon=0.00068$, $\beta=20.70$, $\kappa=-17.41$, $\tau=0.6703$ calibrated once. Prior-CE: oracle $2.2432$, learned $2.2572$, **gap $+0.0139$ nats**. Stage P collapse as designed (`commit 0.000`, basin $0.12$); stage B `p_true 0.935`, `acc 0.969`.

Uncompressed eval ($C\ge T$), decomposed:

| method | all | present | absent |
|---|---:|---:|---:|
| map (ungated) | 0.0994 | 0.0018 | 0.4824 |
| **gate (saved $\tau$)** | **0.0224** | **0.0184** | **0.0379** |
| cue-only | 0.2425 | 0.3043 | 0.0000 |
| kNN (forced commit) | 0.1457 | 0.0037 | 0.7026 |
| kNN + threshold ($\cos>0.410$) | 0.0255 | 0.0214 | 0.0418 |

The gate beats the like-for-like comparator on all three axes in this run. **One observation, not a replicated claim** (§19.5).

**19.2 Capacity, spread geometry ($T_{\mathrm{big}}=100$).** Canonical, `debit`, `stream-lightest` and `topk-learned` agree to four decimals at every capacity; `agree-debit 100%`, merges $0.1$–$0.3$ per episode.

| $C$ | canonical present | ward present | canonical absent | retained | ledger (nats) |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.0960 | 0.1157 | 0.0057 | 0.702 | 18.4 |
| 8 | 0.0542 | 0.0605 | 0.0136 | 0.845 | 8.6 |
| 16 | 0.0258 | 0.0263 | 0.0136 | 0.959 | 1.9 |
| 32 | 0.0161 | 0.0161 | 0.0181 | 1.000 | 0.09 |

`ward`'s ledger (210 / 202 / 183 / 146 nats) is **not** comparable to the canonical ledger: eviction under-books destroyed targets (§3.3). Use `retained` (0.702 vs ward's 0.444 at $C{=}4$) for the information-loss story.

**19.3 Duplicate sweep, paired, 5 seeds, $C{=}4$, $T{=}100$, REF = `debit`.**

| $\rho_{\mathrm{dup}}$ | canonical | debit | ward | top-$k$ | canonical − debit | SE | d/SE |
|--|--|--|--|--|--|--|--|
| 0.00 | 0.1137 | 0.1137 | 0.1278 | 0.1137 | 0.000000 | 0.000000 | exact tie |
| 0.25 | 0.1125 | 0.1125 | 0.1254 | 0.1227 | −0.000000 | 0.000001 | −0.21 |
| 0.50 | 0.0943 | 0.0962 | 0.1032 | 0.1175 | −0.001911 | 0.001018 | −1.88 |
| 0.75 | **0.0691** | 0.0770 | 0.0729 | 0.1090 | **−0.007848** | 0.002104 | **−3.73** |

Reference comparisons at $\rho=0.75$: `ward` − `debit` $=-0.004079\pm0.002289$ ($-1.78\sigma$, ns); `topk` − `debit` $=+0.032004\pm0.003618$ ($+8.85\sigma$). Merge counts $70.0$ (canonical) vs $68.4$ (debit) — the ~1.6 overrides per episode buy back the provenance. At $\rho=0$ `ward` is $+3.81\sigma$ **worse** than canonical.

**19.4 Unit checks.** 47/47. Theorem A $0/20000$ (max residual $-1.64\times10^{-5}$); Theorem B $\mu$ err $5.6\times10^{-17}$, $\ell$ err $8.9\times10^{-16}$; Theorem E (`debit`) max excess $0$; Theorem E′ max forced debit $0.251641\le0.251984$ ($\beta{=}16$) and $0.251400\le0.251529$ ($\beta{=}20.70$); Gram drift $6.66\times10^{-16}$ over $10^3$ mixed ops; check 45 bit-identical with gate off; check 47 gate fires on 8 000 pairs, $|\delta\Delta|_{\max}=6.75\times10^{-14}$; check 43 $3.55\times10^{-15}$; Lemma F identity $1.11\times10^{-16}$.

**19.5 Residual limitations (all open).**
1. **The gate's present-trial advantage is not established.** Across three runs it read tie → win → win as $\beta,\kappa$ and the train-fit threshold moved. It has never been subjected to the paired multi-seed treatment the compression claim received. This is currently the least rigorously supported headline number in the project.
2. **Absent loss is boundary-sensitive.** At $C{=}4$ it moved $0.0378\to0.0057$ between runs on $\beta:20.82\to20.70$, $\kappa:-17.08\to-17.41$ — roughly 6 of ~100 absent trials crossing the gate boundary. Never quote an absent number without a CI.
3. **Cross-version checkpoint comparisons are unpaired.** The gate is active during training at small $C$ (§9), so canonical and `debit` produce different checkpoints; and CPU vs CUDA draw different RNG streams. Only within-run columns are paired.
4. **Canonical vs `ward`** at saturation: $\approx1.9\sigma$, unresolved.
5. $\Gamma$ undershoots $\lambda$ by ~9%; prior CE rises after P by design; compression-in-loop cost on the uncompressed readout.
6. **§23 LM interface: SPECIFIED, TO MEASURE.** The last major claim with no data.

## 20. Central research hypothesis
1. Retrieval beats cue-only and centroid on present; abstains on absent. *(MEASURED.)*
2. Forgetting rate recoverable from the marginal law only. *(MEASURED.)*
3. Vagueness is basin collapse at a computable time. *(THEOREM.)*
4. Compression is evidence-monotone, parameter-free, auditable, fixed-size, with a **bounded** override cost. *(THEOREM A/B/E′; unit-MEASURED.)*
5. Decision precedes descent. *(THEOREM + witness.)*
6. The policy reduces to eviction in spread geometry and strictly dominates it in duplicate-rich streams. *(MEASURED, paired, 3.6–8.9σ.)*
7. The gate is neutral where the debit currency is right and strictly better where it is not. *(MEASURED, −3.73σ.)*
8. The latent is a Transformer interface. *(SPECIFIED; TO MEASURE.)*

**Withdrawn:** present-loss superiority over top-$k$ in spread geometry; ledger dominance for the canonical policy; the ward crossover. **Not claimed:** a replicated gate advantage over kNN+threshold on present.

## 21. Supersession

**21.1** v2.2.1 §21 items 01–30 and v2.3.1 items 31–43 remain withdrawn.

**21.2 v2.3.2.**
44. "Theorem E holds for the canonical policy." → Narrowed to `debit`; canonical satisfies E′ and the bounded-override corollary.
45. "`ward` beats `debit` at saturated duplicate density." → **FALSIFIED** (−0.80σ, replicated at −1.78σ); single-batch noise.
46. "`debit` beats top-$k$-learned on present in spread geometry." → **FALSIFIED** ($|d/\mathrm{SE}|\le0.85$); the claim is parity.
47. "`below_cross` is the fraction of merges below the debit crossover." → Renamed `agree-debit`; complement counts gate overrides; $0\%$ with zero merges means *no merges*.
48. "`resfirst` is an ablation." → Promoted to canonical; `debit` is now the ablation.
49. "The ledger ranks information loss across policies." → Never did; use `retained` and `destroyed_frac`.
50. "Policy invariance can be tested by comparing flat argmin indices." → False in floating point for any symmetric cost; compare unordered-pair keys and operations (§4.5, §15 rule 23).
51. "The gate's present-trial advantage over kNN+threshold is established." → Not established; three runs, three readings, no paired test (§19.5 item 1).

## 22. Changes from v2.3.1, and why

**22.1 The structural finding.** Lemma F: merging is always angularly cheaper. Proposition D: the debit branch ranks merges correctly. The failure was the **currency** — $\Delta$ charges eviction only the block's dose drop, so destroying a low-probability target looks cheaper than an unresolvable merge that would preserve it. The resolution gate overrides that comparison exactly where the readout cannot distinguish the pair; Theorem E′ bounds the override at half a nat. The paired data confirms the mechanism twice: neutral where the currency is right, $-2.93\sigma$ then $-3.73\sigma$ where it is not.

**22.2 This revision.** §9 records the training-time gate activity and the resulting checkpoint divergence; §15 rule 22 documents the zero-merge convention and rule 23 the unordered-pair comparison; §17.6 marks predictions 1–6 MEASURED with the confirmation $\sigma$ values and adds prediction 7 (OPEN); §19 replaced with the confirmation run; §21 adds items 50–51. No policy, code or theorem change.

**22.3 Deliberately unchanged.** State class; convex write; dose maps; diffusion laws; Gibbs readout and descent; decide-before-descend; probe split; reinforcement; decoder; legacy ablation; ledger currency and its caveat; LM interface form.

## 23. Language-model interface (SPECIFIED; TO MEASURE)
$C$ memory tokens, tied KV, log-dose attention biases, null token, gate as residual mix, surprise-driven writes. Two token-level compression operations — merge (resultant + bounded debit) and drop-with-ledger — both bounding $C$ and both evidence-monotone. Calibration does not transfer for free; stage A/B on LM states required. **This is now the only major claim in the project with no measurement.**

## 24. Self-check suite (47 checks)

**1–36** retained with the established harness fixes (absolute-time semigroup; rank-1 top-eigenvector; torch-vs-torch floor reference; angle-closing-fraction closure witness; `enable_grad` wrappers; $\omega=10^{-3}$ antipodal evict branch; nats ledger chain bound).
**37–44** retained: spread reduction for `debit`; duplicate free-merge; crossover identity; Theorem E for `debit`; incremental Gram over $10^3$ ops; ledger/destroyed semantics; **debit commutation on unordered-pair keys**; Lemmas F/G.
**45. Gate-off equivalence.** With no admissible pair below resolution, canonical and `debit` produce **bit-identical** $Z,\ell,\Lambda$, alive masks, provenance and stats, and zero merges.
**46. Theorem E′ bound.** Over $2\times10^4$ gate-eligible draws, $\delta_{\mathrm{fuse}}\le\tfrac12\beta(1-G)\le\tfrac12$ and $\delta_{\mathrm{fuse}}\le\beta(1-\sqrt{1-1/(2\beta)})$, 0 violations.
**47. Gate commutation.** On a state that *does* contain below-resolution pairs, $\mathcal{R}$, the selected **unordered pair** and operation, and $\Delta$ are invariant under a uniform log-shift ($\Delta$ tolerance $10^{-11}$, justified by cancellation in $\ell_l-\ell_h$ when the two log-doses are close).

---

**End of spec v2.3.2 (measured results revised).** One gate, one comparison, zero new parameters: merge if $\beta(1-G)\le1$, else $\arg\min\frac{A}{S}\min\big(\beta(1-r),\mathrm{softplus}(\ell_l-\ell_h)\big)$. The override costs at most half a nat and buys back provenance; the ledger stays an audit, not an objective. The compression claim is now the best-supported result in the project; the readout claim is the least, and §19.5 says so.

---

**What changed, concretely:** §19 now carries the confirmation run (47/47, prior gap +0.0139, the four-way spread parity, the −3.73σ paired table, both E′ margins); §17.6 marks predictions 1–6 MEASURED and adds the open canonical-vs-ward item; §9 documents that the gate fires during training at small $C$ and therefore produces a *different checkpoint* from `debit` — which is why cross-version comparisons were never paired; §15 rule 22 documents the `agree-debit 0%`-with-zero-merges convention so it isn't misread as total disagreement; §21 adds items 50 (flat-index invariance) and 51 (the readout claim is not established).

The next artifact worth building is the train-fit kNN+threshold row in `hopmem_stats.py`, so §19.5 item 1 can be closed rather than carried.