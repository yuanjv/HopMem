# HopMem v2.3.1

**Debit-optimal compression: one derived comparison over two evidence-monotone operations, an exact incremental Gram law, and the withdrawn superiority claim.**

This specification is self-contained and supersedes v2.2.1 **and** the interim "query-relevance" draft ($R_i$ / EMA). It incorporates a pre-release audit (§21.4) that corrected three minimality overclaims, the ledger bookkeeping, the gradient statement, and the LaTeX.

**Three design decisions.** *No relevance scores*: the identifiability result already says the target law is $p_i\propto\alpha_i$ — the doses **are** the query distribution, so a separate EMA vector double-counts the dose and adds a free parameter. *One formula*: the expected debit of a merge is $\beta\times\mathrm{Ward}/S$; the expected debit of an eviction is the lighter atom's target probability; the policy takes the cheaper. *Exact incremental Gram*: the $O(C^2d)$ rebuild is withdrawn.

**Motivating measurement (v2.2.1, both corpora).** Min-Ward fusion lost to learned top-$k$ on present trials at every tested capacity (real: $0.1299$ vs $0.1215$ at $C{=}4$; $0.0184$ vs $0.0165$ at $C{=}32$). The gate's win over kNN+threshold is **rejection, not retrieval** (present tie $0.0171/0.0169$; absent $0.0505/0.0622$). Lemma F explains the failure exactly: merging is *always* angularly cheaper than evicting the lighter atom, so eviction competes **only** because it levies no evidence debit on the survivor. v2.2.1 forced merges and paid that debit on every overflow. v2.3.1 supplies the missing option.

---

## 1. Scope

**Supports.** Cue-conditioned retrieval with abstention; learned recency ($\Gamma$) and surprise ($\omega$) forgetting; sharp basin-collapse forgetting; a strictly fixed-size state of $C(d+2)+C^2$ floats; $O(1)$ read/write in stream length for fixed $C,d$; pointwise-evidence-monotone, auditable compression; per-event debit dominance over dose-ordered eviction (Theorem E); an exact $O(C)$ Gram merge law; a latent Transformer interface (§23).

**Non-goals.** Exact recall below resolution; order-independence of greedy selection; a $\beta$-independent ledger; monotone max-atom gap (block mass only); **decision-invariance of live-atom eviction** (only gate-dead deletion is decision-free); **gradual per-target loss under eviction** (abrupt by construction); Loewner-monotone $W$; present-loss superiority over top-$k$ in spread geometry (withdrawn, §21).

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
| $h$, $l$, $\rho=\alpha_l/\alpha_h$ | heavier / lighter member of a pair; mass ratio |
| $\delta_{\mathrm{fuse}},\delta_{\mathrm{evict}}$ | the two derived debits (§4.5) |
| $\Delta=(A/S)\,\delta_{\mathrm{op}}$ | expected block-nats removed |
| $\Lambda_c=\sum\delta_{\mathrm{fuse}}$ | per-surviving-block ledger, **nats** |
| $p_i=\alpha_i/S$ | identified target law |
| $c_{ij}=\dfrac{\alpha_i\alpha_j}{A}(1-G_{ij})$ | Ward cost |
| $\epsilon_r$ | resultant mask ($r<\epsilon_r$ excludes the merge candidate) |
| $E_i(u)=e^{\ell_i+\beta z_i\cdot u}$, $E_0(u)=e^{\kappa+\beta q\cdot u}$ | atom / null evidence |

```
python hopmem.py selftest                                    # 44 checks (§24)
python hopmem.py train --schedule probe --batch 256 --C 16 --c_min 4
python hopmem_diag.py --ckpt hopmem_theta.pt --T_big 100 --caps 4 8 16 32 --dup_sweep
```

Deployment: `mem = HopMem(p, d, C, tau)`; `mem.write(z, s, t)` → diffuse → write → compress-on-overflow; `mem.retrieve(q, t_q)` → gate at $u=q$, then descend.

## 3. State

### 3.1 Invariants
$W\succeq0$, $\mathrm{tr}\,W=1$, $S\in[0,1)$, $\lambda_{\min}(W)\ge(1-S)/d>0$ in exact arithmetic. Initialize $W=J$, $S=0$, empty atom set; never $W=0$. Store $(z_i,\ell_i,\Lambda_i)$, $S$, a lazy clock, and $G$. Never materialize $M$ or $W$ outside diagnostics or the legacy readout.

### 3.2 Canonical log-dose
Between compressions,
$$\ell_i(t_q)=\ln\big(a_i(1-\varepsilon_i)\big)-\omega\sum_{j>i}s_j-\Gamma(t_q-t_i),\qquad a_i=1-e^{-\omega s_i},\quad \varepsilon_i=\bar\varepsilon(1-a_i).$$
For a compressed block the **exact** dose is the recursion of §4.5. The closed form is
$$\ell_c(t_q)=\mathrm{logaddexp}_{i\in c}\Big(\ln\big(a_i(1-\varepsilon_i)\big)-\omega\sum_{j>i}s_j-\Gamma(t_q-t_i)\Big)-\Lambda_c,$$
with **per-constituent ages inside** the logaddexp (a single-age form is exact only for equal-age blocks), and with the inequality
$$\boxed{\;\ell_c\;\ge\;\mathrm{logaddexp}(\mathrm{base})-\Lambda_c\;}$$
**exact for one merge, conservative for chains.** *Proof of the chain step:* $\mathrm{logaddexp}(x-a,y-b)\ge\mathrm{logaddexp}(x,y)-\max(a,b)\ge\mathrm{logaddexp}(x,y)-(a+b)$ for $a,b\ge0$. ∎ Hence $e^{-\Lambda_c}$ is a **lower bound** on retained dose fraction, not an identity. Measured: single-merge error $1.8\times10^{-15}$; chain slack $\le0.27$ nats over 300 random trees.

### 3.3 Ledger (nats)
- **Per surviving block:** $\Lambda_c=\sum\delta_{\mathrm{fuse}}$ only. Bound as in §3.2.
- **Episode ledger:** $\sum\delta=\sum\delta_{\mathrm{fuse}}+\sum\delta_{\mathrm{evict}}$ — the auditable nats removed by compression.
- **Per target:** retained $=e^{-\Lambda_{c(t)}}$ if the target's block survives; **hard $0$** if it was destroyed. A destroyed block is never assigned a finite $\Lambda$.

The $\beta$-gauge caveat is retained: debits use the current $\beta$; recompute in-forward during training; freeze at deployment. Diagnostic ledgers are reported at the checkpoint $\beta$.

### 3.4 Atom-dependence
The Gibbs readout depends on the atom set $\{(z_i,\ell_i)\}$, not on $W$ alone. Compression is defined on atoms; no claim that $W$ decreases in Loewner order.

## 4. Operation alphabet

{diffuse, write, reinforce, **fuse**, **null-fuse**}. Order per event: diffuse → write → reinforce → compress-on-overflow. Write and diffuse are convex mixtures; fuse is coarse-graining plus a depolarizing step on a block; null-fuse routes mass to $(1-S)J$. "Every operation is one convex primitive" remains withdrawn.

### 4.1 Diffuse
Lazy form with an **absolute-time clock**: $\Delta t=\max(t-t_{\mathrm{clock}},0)$, then $\ell_i\leftarrow\ell_i-\Gamma\Delta t$, $S\leftarrow S\,e^{-\Gamma\Delta t}$, $t_{\mathrm{clock}}\leftarrow\max(t,t_{\mathrm{clock}})$.

### 4.2 Write (convex primitive)
Given unit $z$, importance $s\ge0$: $a=-\mathrm{expm1}(-\omega s)$, $\varepsilon=\bar\varepsilon(1-a)$,
$$W'=(1-a)W+a\big[(1-\varepsilon)zz^\top+\varepsilon J\big].$$
Log-atom form: $\ell_j\leftarrow\ell_j-\omega s$ for all existing $j$; append $\ln\max(a(1-\varepsilon),10^{-30})$; $S'=(1-a)S+a(1-\varepsilon)$. Exact impure decomposition: $a[(1-\varepsilon)zz^\top+\varepsilon J]=a(1-\varepsilon)zz^\top+a\varepsilon J$ — the blur applies to the injected state, not only to the target direction.

### 4.3 Reinforce (optional)
Masked denominator; confidence named $\mathrm{cf}$; appends an atom and counts toward $C$; compress-on-overflow follows. Purity is not monotone.

### 4.4 Null-fusion — two triggers

**(a) Gate-dead (free, decision-free).** Atom $i$ can never trigger a commit for any unit cue iff
$$\ell_i\le\kappa+\ln\tau.$$
*Proof.* $\max_q\mathrm{gap}_i(q)=\ell_i-\kappa$, attained at $q=z_i$; commit requires $\mathrm{gap}>\ln\tau$. ∎ Deleting such an atom leaves every gate decision unchanged. Always applied first on overflow.

**(b) Debit-eviction (decision-changing).** A *live* lighter atom may be deleted when the comparison of §4.5 selects eviction. This deliberately changes decisions — it is the price top-$k$ pays — and is **not** covered by decision-invariance. Theorem A still bounds its evidence.

### 4.5 The compression policy

**Trigger.** Atom count $>C$ after write/reinforce.

**Step 1.** Null-fuse all gate-dead atoms (free).

**Step 2.** For every alive unordered pair $(i,j)$, with $A$, $w_i$, $h$, $l$, $\rho$, and $r^2=1-2w_iw_j(1-G_{ij})$:

$$\boxed{\;\delta_{\mathrm{fuse}}(i,j)=\beta(1-r)=\frac{2\beta\,w_iw_j(1-G_{ij})}{1+r},\qquad
\delta_{\mathrm{evict}}(i,j)=\ln\frac{A}{\alpha_h}=\mathrm{softplus}(\ell_l-\ell_h)\;}$$

Both are stable as written; never form $1-r$ by subtraction. Weight by the block's target probability:

$$\boxed{\;\Delta(i,j,\mathrm{op})=\frac{A}{S}\,\delta_{\mathrm{op}}(i,j),\qquad
(i^{*},j^{*},\mathrm{op}^{*})=\text{the minimizing pair and operation}\;}$$

Compare in log space, $\ln\Delta=\mathrm{logaddexp}(\ell_i,\ell_j)-\ln S+\ln\delta_{\mathrm{op}}$, since $A/S\in(0,1]$ can underflow. Execute the winner; repeat until count $\le C$.

**The two maps.**
- *fuse*: $\mu=m/r$, $\ell'=\mathrm{logaddexp}(\ell_i,\ell_j)-\delta_{\mathrm{fuse}}$; deficit $A(1-e^{-\delta})$ routes to $(1-S)J$; $\Lambda'=\Lambda_i+\Lambda_j+\delta_{\mathrm{fuse}}$; provenance unions.
- *evict*: delete $l$; $h$ is **untouched** (direction and dose exact); book $\delta_{\mathrm{evict}}$ to the episode ledger and mark $l$'s provenance destroyed (retained $0$).

**Debit semantics (audited).**
$\delta_{\mathrm{fuse}}$ is a **sufficient** uniform debit for Theorem A, **tight at equal masses** (binding at $u=\mu$), conservative otherwise.
$\delta_{\mathrm{evict}}$ is the **exact block log-dose reduction** ($\ell'=\ell_h$) and the unique debit leaving the survivor's evidence *identical*: $E'\equiv E_h$. It is the minimal **angle-independent** choice. For a fixed geometry the smaller debit
$$\delta_{\min}^{\mathrm{anchor}}=\ln\frac{A}{\alpha_h+\alpha_l e^{-\beta\sqrt{2(1-G)}}}$$
also satisfies Theorem A, but inflates the survivor by $1+\rho\,e^{-\beta\sqrt{2(1-G)}}$ — partial credit for a deleted atom. **Forbidden by design.**

**Currency caveat (honest).** $\Delta$ is expected *block log-dose reduction*, a proxy for retrieval loss. It charges eviction only the block's dose drop, **not** the evicted target's total loss of retrievability (loss $\approx$ the cue gap). The policy is therefore **conservative toward merging**: its merge window is narrower than a loss-calibrated comparison would give. Theorem E is dominance **in this currency only**. A loss-calibrated variant replaces $\delta_{\mathrm{evict}}$ by the measured cue-gap statistic $\bar L$; it adds one calibrated scalar and is an **ablation**, not the canonical policy.

**Derived crossover (no new constant).** The policy merges iff $\beta(1-r)<\mathrm{softplus}(\ell_l-\ell_h)$. At equal masses $r=\cos(\theta/2)$, so
$$\theta<\theta^{*}=2\arccos\!\big(1-\ln 2/\beta\big),\qquad \beta>\tfrac{\ln 2}{2}.$$
At $\beta=17$: $\theta^{*}=0.5723$ rad, i.e. $1-G<0.159$. **The memory merges what its own readout cannot resolve, and evicts otherwise.** The v2.2.1 "resolution quotient" design rule is now the operational policy.

**Edge cases.** $s=0$: gate-dead, deleted free. $r<\epsilon_r$: mask the merge candidate; eviction remains admissible, so the v2.2.1 $C{=}1$ antipodal fallback is **subsumed** (evict lighter, ledger $\ln 2$). All gate-dead: count drops to $0\le C$, loop exits. $C\ge T$: nothing fires; bit-for-bit v2.1.1 regression retained.

### 4.6 Incremental Gram law (mandatory)
On a merge into slot $i$, using the **old** rows $i$ and $j$, for every alive $k\ne i,j$:
$$\boxed{\;G_{ik}\leftarrow\frac{w_iG_{ik}+w_jG_{jk}}{r},\qquad G_{ki}\leftarrow G_{ik},\qquad G_{ii}\leftarrow1,\qquad \text{row/col } j\leftarrow0\;}$$
Exact by linearity of the dot product; $O(C)$, no $d$-dimensional products. **Two guards:** the formula does not yield $1$ at $k=i$ (it gives $\mu\cdot z_i$), so set $G_{ii}$ explicitly; zero row/column $j$ **after** computing the new row. On eviction: zero row/col $j$. On write: one new row $G_{\mathrm{new},k}=z_{\mathrm{new}}\cdot z_k$, $O(Cd)$. The streaming full rebuild ($O(C^2d)$) is withdrawn from the hot path; permitted only as a drift diagnostic. Refresh if any $|G_{ij}|>1+10^{-6}$ or the periodic check fails (self-check 41).

## 5. Diffusion: six exact laws

Generator: uniform log-drift at rate $\Gamma$. Over $\Delta t$:

- **D1 (semigroup).** $W(t+\Delta)=e^{-\Gamma\Delta}W(t)+(1-e^{-\Gamma\Delta})J$; intervals compose additively.
- **D2 (log-drift).** $\ell_i\leftarrow\ell_i-\Gamma\Delta$, uniform and additive.
- **D3 (mass).** $S\leftarrow S\,e^{-\Gamma\Delta}$; $S<1$ preserved.
- **D4 (spectral invariance).** Eigenvectors of $M$ preserved; eigenvalues scale by $e^{-\Gamma\Delta}$; $\mathrm{range}(M)$ invariant. Diffusion does not merge directions.
- **D5 (commutation, extended).** Diffusion commutes with write-dilution, with both compression maps, **and with the policy argmin**: under a uniform log-shift, $A/S$, $w$, $r$, and $\ell_l-\ell_h$ are invariant, hence $\delta_{\mathrm{op}}$, $\Delta$, and the minimizer are invariant. The gate-dead branch remains clock-dependent.
- **D6 (forgetting times).** $t_{\mathrm{forget}}=(\ell_k(0)-\Lambda_k-\kappa)/\Gamma$; gate-dead time subtracts $\ln\tau$; legacy half-closure $t_{1/2}=\Gamma^{-1}\ln(\tilde m_0/\gamma)$. Evicted blocks are gone, not faded.

**Steady state.** Periodic writes of dose $a$ at interval $\Delta$, diffuse-then-write: $S^{*}=a(1-\varepsilon)/\big(1-(1-a)e^{-\Gamma\Delta}\big)$.
**Decay split.** Report $\omega\,\mathbb{E}[s]+\Gamma\,\mathbb{E}[\Delta t]$, not $\Gamma$ alone.

## 6. Writing details
$a=-\mathrm{expm1}(-\omega s)$; clamp $s\ge0$; $\ell=\ln\max(a(1-\varepsilon),10^{-30})$ finite at $s=0$ with zero gradient through the clamped branch; $\bar\varepsilon\equiv0$ is the pure-write ablation flag.

## 7. Parameters

**Shared law:** $\omega>0$, $\Gamma>0$, $\bar\varepsilon\in(0,1)$. **Readout:** $\beta>0$, $\kappa\in\mathbb{R}$. **Legacy:** $\eta>0$, $\chi\in(0,1)$, $\gamma=\chi/\eta$. **Gate:** $\tau>0$, calibrated once. **Capacity:** $C$ architectural.

**Compression adds zero parameters.** Both debits are derived from $\alpha,G,\beta,S$; $\theta^{*}$ is derived. The interim draft's EMA rate $\eta$ and softmax temperature are withdrawn. The loss-calibrated ablation (§4.5) adds one calibrated scalar and is flagged as such.

**Parameterization.** Positive scalars via softplus; $\bar\varepsilon,\chi$ via sigmoid; $\kappa$ raw. Stable inverse: $\mathrm{invsoftplus}(y)=\ln(e^y-1)$ via $\mathrm{expm1}$ for $y<20$, else $y+\ln(1-e^{-y})$.

**$\tau$ protocol.** Calibrated **once** on a fixed fused training batch by a two-pass fixed point in $\tau_{\mathrm{dead}}$ (pass 1 at $\tau_{\mathrm{dead}}=1$, pass 2 re-simulates at $\tau_{\mathrm{dead}}=\tau_1$ and re-picks), saved with the checkpoint, applied identically by `evaluate` and streaming `retrieve`. Grid $\tau_i=e^{-6+0.4i}$, $i=0,\ldots,18$; index 15 is exactly $\tau=1$. Flat sweep → argmax, not the grid edge. Never fit on eval. The real-corpus run legitimately landed on $\tau=1.0$ (grid interior).

**Identifiability (measured).** $\Gamma$ from the marginal (cue-free) law; $\beta,\kappa$ from the conditional. $\partial\mathrm{CE}/\partial\Gamma=\mathrm{age}_k-\mathbb{E}_p[\mathrm{age}]$ vanishes when the posterior is one-hot: ambiguous cues are a data requirement. $(\beta,\kappa)$ share a flat ray; $\kappa$ serves two jobs (marginal null-odds in P, conditional commit threshold in A/B), so prior CE rising after P is expected.

## 8. Readout

### 8.1 Gibbs free energy
$$F(u)=\frac{1}{\beta}\ln\Big(e^{\beta q\cdot u+\kappa}+\sum_i\alpha_i e^{\beta z_i\cdot u}\Big).$$
At $u=q$ the posterior is the vMF-mixture likelihood $p(q)=\mathrm{softmax}\big([\beta Z^\top q+\ell,\;\beta+\kappa]\big)$, last index the null class.

### 8.2 Descent theorem
$F$ is convex; with $p$ the posterior, $\nabla F(u)=p_q q+\sum_i p_i z_i$ and $u^{+}=\nabla F(u)/\|\nabla F(u)\|$ satisfies $F(u^{+})\ge F(u)$ for every $\beta>0$, equality iff $u^{+}=u$; no step size. Energy-increase fraction must be 0 up to rounding. Masked float guard retained; empty memory returns $q$ exactly; permutation- and log-dose-shift-invariant.

### 8.3 Geometry
$\theta_{\mathrm{basin}}=\arccos\big(1-(\ell_k-\Lambda_k-\kappa)/\beta\big)$ (conservative); floor $\approx(\bar\alpha/\alpha_k)e^{-\beta}\sqrt{T-1}$ (no $1/d$); resolution $\beta(1-z_1\cdot z_2)\gtrsim|\ell_1-\ell_2|$; each merge shrinks the basin parameter by exactly $\delta_{\mathrm{fuse}}/\beta$.

### 8.4 Decide before descend
$$\mathrm{gap}(q)=\max_k\big(\beta q\cdot z_k+\ell_k\big)-(\beta+\kappa);\qquad\text{commit iff }\mathrm{gap}(q)>\ln\tau,\text{ else return }q.$$
The map's null basin is strictly smaller than the one-shot argmax region. Deterministic witness: $\beta=16$, $\ell=-3$, $\kappa=-6.5$, one atom at $0.70$ rad → atom logit $9.237$, null $9.500$; null wins at $u=q$, the map still commits. The **mass gate** $\ln(E_{\mathrm{mem}}/E_0)$ is the ablation that is monotone under compression.

### 8.5 Legacy quadratic (ablation only)
One stable attractor; $H_k=(\mu+\gamma)P_k-P_kM_{\mathrm{oth}}P_k\succ0$ on the tangent space; for $T$ equal masses stability needs $\gamma\gtrsim2\alpha\sqrt{(T-1)/d}$ while closure is $\alpha/(\alpha+\gamma)$, hence $\mathrm{closure}_{\max}\approx1/\big(1+2\sqrt{(T-1)/d}\big)$. The witness measures the **angle-closing fraction** $(\phi-\theta)/\phi$ at nonzero cue angle.

### 8.6 Compression theorems

**Theorem A (evidence domination — both operations).** For every unit $u$,
$$e^{\ell'+\beta\mu\cdot u}\le e^{\ell_i+\beta z_i\cdot u}+e^{\ell_j+\beta z_j\cdot u}\quad\text{(fuse)},\qquad E_{\mathrm{mem}}'\le E_{\mathrm{mem}}\quad\text{(evict)}.$$
*Proof (fuse).* $E'=Ae^{\beta(\mu\cdot u-(1-r))}\le Ae^{\beta r\mu\cdot u}=Ae^{\beta m\cdot u}\le\alpha_ie^{\beta z_i\cdot u}+\alpha_je^{\beta z_j\cdot u}$. First step: $(1-r)(\mu\cdot u-1)\le0$. Second: Jensen on the convex exponential. *Proof (evict).* With $\delta_{\mathrm{evict}}$, $E'\equiv E_h\le E_h+E_l$; a nonnegative term is deleted. ∎
**Corollaries.** $E_{\mathrm{mem}}$ pointwise non-increasing; block posterior mass non-increasing; null mass non-decreasing — **compression never fabricates confidence**. The max-atom gap may rise by $\le\ln2$ under fuse; block mass is the monotone object; $\tau$ recalibrates.

**Theorem B (dose additivity and commutation).** $\mathrm{logaddexp}(x+c,y+c)=\mathrm{logaddexp}(x,y)+c$ and $r$ is invariant under uniform log-shifts, so $\Gamma$-decay and $\omega$-dilution commute with both maps and with the policy argmin (§4.5, D5). The closed form of §3.2 holds with per-constituent ages; chains are conservative.

**Proposition C (Ward = expected angular damage).** Under $p_i\propto\alpha_i$,
$$\mathbb{E}\big[1-z_{\mathrm{target}}\cdot\mu\big]=\frac{\alpha_i\alpha_j}{AS}(1-G_{ij})+O\big((1-G)^2\big)=\frac{c_{ij}}{S}.$$
*Proof sketch.* $1-z_i\cdot\mu=w_j^2(1-G)+O((1-G)^2)$; weight by $\alpha_i/S$ and symmetrize: $\alpha_iw_j^2+\alpha_jw_i^2=\alpha_i\alpha_j/A$. ∎

**Proposition D (first-order expected debits).** Under the same law,
$$\Delta_{\mathrm{fuse}}=\frac{A}{S}\beta(1-r)=\frac{\beta\,c_{ij}}{S}+O\big((1-G)^2\big),\qquad
\Delta_{\mathrm{evict}}=\frac{A}{S}\ln(1+\rho)=p_l\Big(1+\frac{\rho}{2}+O(\rho^2)\Big).$$
*Proof.* $1-r=w_iw_j(1-G)+O((1-G)^2)$ and $(A/S)\beta w_iw_j=\beta c_{ij}/S$; $\ln(1+\rho)=\rho-\rho^2/2+\cdots$ with $A=\alpha_h(1+\rho)$. ∎
**Reading:** merge cost $=\beta\times\mathrm{Ward}/S$; evict cost $=$ the lighter target's probability. Ward was already the correct **merge ranking**; what v2.2.1 lacked was the **option not to merge**.

**Lemma F (angular ordering — why eviction competes).** In the angular-damage currency of Proposition C, to first order,
$$\mathbb{E}[\text{damage}\mid\mathrm{fuse}]=\frac{c_{ij}}{S}=p_l\,w_h\,(1-G),\qquad
\mathbb{E}[\text{damage}\mid\mathrm{evict}\ l]=p_l\,(1-G).$$
Hence merging is **always** angularly cheaper, by the factor $w_h=\alpha_h/A\in[\tfrac12,1)$.
*Proof.* $c_{ij}/S=\alpha_h\alpha_l(1-G)/(AS)=p_lw_h(1-G)$; eviction substitutes $z_h$ for target $l$ at damage $1-G$ and leaves target $h$ intact. ∎
**Consequence.** The *only* reason eviction is competitive is that it levies **no debit on the survivor**. The design is an angle-versus-evidence trade; the crossover is where they balance. (Deployed loss adds the cue floor and gate effects; the ordering is stated in the damage currency.)

**Lemma G (cheapest eviction is exact).** With $f(\alpha_h)=(\alpha_h+\alpha_l)\ln(1+\alpha_l/\alpha_h)$,
$$f'(\alpha_h)=\ln(1+\rho)-\rho<0,\qquad \frac{\partial f}{\partial\alpha_l}=\ln(1+\rho)+1>0.$$
Hence $\Delta_{\mathrm{evict}}=f/S$ is minimized **exactly** at (lightest, heaviest), not merely to first order. ∎

**Theorem E (per-event debit dominance).** Let $l^{*}$ be the lightest alive atom and $h^{*}$ the heaviest. Since $(\mathrm{evict},l^{*},h^{*})$ is a candidate and by Lemma G is the cheapest eviction,
$$\Delta(i^{*},j^{*},\mathrm{op}^{*})\;\le\;\Delta_{\mathrm{evict}}(l^{*},h^{*})\;=\;p_{l^{*}}\big(1+O(\rho)\big),$$
with strict inequality whenever some merge is cheaper at the minimizer. **The policy never pays more expected block-nats than dose-ordered eviction.** Dominance is in the ledger currency (§4.5 caveat). ∎

**Corollary (regime reduction).** If $\min_{(i,j)}\Delta_{\mathrm{fuse}}\ge\Delta_{\mathrm{evict}}(l^{*},h^{*})$, the policy deletes $l^{*}$ and its state trajectory is **identical** to streaming dose-ordered eviction. If some pair is below crossover, the policy merges and preserves $\ln(1+\rho)-\beta(1-r)>0$ nats that eviction destroys.

**Corollary (heavy-atom protection).** The evict branch always removes the lighter member; the heavier atom's direction and dose are untouched. Zero distortion *and zero evidence inflation* for high-probability targets — the mechanism top-$k$ exploited, now derived.

## 9. Training objective

Posterior cross-entropy at $u=q$ with the null class; labels map to the target's fused block; **evicted targets map to null** (the achievable prediction). Deployed angular loss is still measured against the original $z_{\mathrm{target}}$. No-match episodes required ($p_{\mathrm{absent}}=0.2$), else $\kappa\to-\infty$ is a free optimum.

**Probe schedule.** P: cue-free CE, **fusion off**, trainable $\omega,\Gamma,\kappa$. A: cue CE on compressed states, $C\sim U\{C_{\min},\ldots,T\}$, trainable $\beta,\kappa$. B: cue CE, trainable $\bar\varepsilon,\beta,\kappa$, law frozen.

**Gradients (audited).** Selection of the pair **and** the operation is hard argmin with straight-through. Gradients flow through $\delta_{\mathrm{fuse}}=\beta(1-r)$ (which enters $\ell'$) and through surviving doses. **$\delta_{\mathrm{evict}}$ and both argmins are selection-only and carry no gradient**; the evict branch has no gradient path through the deleted atom, exactly as gate-dead deletion. Ledger bookkeeping is detached.

**Recorded cost.** Fusion-in-loop training gave a slightly worse *unfused* readout than the v2.1.1 fusion-free checkpoint ($0.0241$ vs $0.0196$ aggregate). Kept honest; under study.

## 10. Episodes and schedules

Reset to $W=J$; per event diffuse → write → compress-on-overflow; diffuse to $t_q$; retrieve at frozen state; gate; loss; backprop. Sampler: $s_i\sim\mathrm{Exp}(1)$ i.i.d.; inter-arrivals $\mathrm{Exp}(1)$; $t_q=t_T+\mathrm{Exp}(1)$; target $\sim\mathrm{softmax}(\ln s-\lambda(t_q-t))$, $\lambda=0.2$; with probability $0.2$ the cue refers to an unstored item (label null, $z_{\mathrm{target}}=q$). Optional `same_passage`: $T$ consecutive items from one document; cue = random contiguous half-span, same encoder. Vary $s$ and $\Delta t$ independently; warn at $p_{\mathrm{true}}>0.99$.

**Duplicate-heavy generator (new, for the falsifiable win).** With probability $\rho_{\mathrm{dup}}$, $z_{\mathrm{new}}=\mathrm{normalize}(z_{\mathrm{prev}}+\sigma\xi)$ with $\sigma$ set so $\theta\approx\theta^{*}/4$; importance $\sim s_{\mathrm{prev}}\cdot\mathrm{Exp}(1)$. Sweep $\rho_{\mathrm{dup}}\in\{0,0.25,0.5,0.75\}$.

## 11. Reinforcement (optional)
Given the **gated** output, retrieved-content importance $s_{\mathrm{ret}}$, confidence $\mathrm{cf}$, cap $\bar\beta_{\mathrm{ret}}$: $\mu=u^\top Wu$, $a_{\mathrm{ret}}=1-e^{-\omega s_{\mathrm{ret}}}$, $\tau_{\mathrm{ret}}=1/d+(1-1/d)a_{\mathrm{ret}}$, and $\beta_{\mathrm{ret}}=\mathrm{cf}\cdot\min(\bar\beta_{\mathrm{ret}},(\tau_{\mathrm{ret}}-\mu)_+/(1-\mu))$ when $\mu<\tau_{\mathrm{ret}}$, else $0$ — masked denominator, never the raw fraction. Counts toward $C$.

## 12. Decoder (optional)
Must consume the **gated** output. Normalization $\hat D=D/\sqrt{\|D\|^2+\epsilon_{\mathrm{dec}}^2}$ with Jacobian bound $\le1/\epsilon_{\mathrm{dec}}$. Evaluate retrieval separately: a strong decoder can hide weak memory.

## 13. Blending and vagueness

| | legacy quadratic | core Gibbs | compression |
|---|---|---|---|
| mechanism | closure fades smoothly | basin shrinks then vanishes | merge debits basin by $\delta_{\mathrm{fuse}}/\beta$; evict removes the block |
| per-target loss | smooth | sharp collapse | **gradual on merge, abrupt on evict** |
| threshold | $t_{1/2}=\Gamma^{-1}\ln(\tilde m_0/\gamma)$ | $t_{\mathrm{forget}}=(\ell_k(0)-\Lambda_k-\kappa)/\Gamma$ | shortened by $\Lambda_k/\Gamma$ |
| audit | — | — | ledger $\sum\delta$; retained $e^{-\Lambda}$ or $0$ |

## 14. Guaranteed properties

**Exact arithmetic.** $W\succeq0$, $\mathrm{tr}\,W=1$, $\lambda_{\min}\ge(1-S)/d$; closed-form log-doses between compressions; six diffusion laws; dose additivity; exact impure-write decomposition; Gibbs descent for every $\beta>0$; empty memory returns $q$; permutation and log-shift invariance; gate limits; batch/streaming gated equality; strict atom bound $\le C$; Theorem A for **both** operations; Theorem B including policy commutation; Propositions C, D; Lemmas F, G; Theorem E; the exact incremental Gram identity; gate-dead decision-invariance; $C\ge T$ regression; the chain bound $\ell_c\ge\mathrm{logaddexp}(\mathrm{base})-\Lambda_c$ (single-merge exact).

**Not guaranteed.** Exact recall; unique basins or global optimality; angular resolution below $\sim1/\beta$; $\Gamma$ identification from the conditional alone or from compressed states; monotone $S$, purity, or max-atom gap; Loewner-monotone $W$; order-independence; a $\beta$-independent ledger; **decision-invariance of live eviction**; **gradual per-target loss under eviction**; float32-safe margins at extreme $\beta,|\kappa|$; **present-loss superiority over top-$k$ in spread geometry** (the policy *becomes* top-$k$ there).

## 15. Implementation rules

01. Normalize $z,q$; validate finiteness; clamp $s\ge0$.
02. Initialize $W=J$, $S=0$, empty atoms; never $W=0$.
03. Store $(z_i,\ell_i,\Lambda_i)$, $S$, lazy clock, $G$; materialize $M,W$ only for diagnostics/legacy.
04. Lazy diffusion with an **absolute-time** clock.
05. Enforce diffuse → write → reinforce → compress-on-overflow.
06. $a=-\mathrm{expm1}(-\omega s)$; stable softplus inverse.
07. $\ell=\ln\max(a(1-\varepsilon),10^{-30})$; finite at $s=0$, zero clamped-branch gradient.
08. Masked denominators everywhere a branch may be non-finite.
09. Off-flags: $\bar\varepsilon\equiv0$, diffusion bypass, compression bypass ($C\ge T$).
10. The gate lives in the inference path; save state, $\tau$, readout, $C$, config.
11. Calibrate $\tau$ once (two-pass) on a fixed fused batch; never on eval.
12. Log $\beta,\kappa$, doses, $S$, purity, margins, gate decisions, hysteresis, decay split, $\sum\delta$, destroyed mass, fuse/evict counts, below-crossover fraction. Energy-increase fraction must be 0.
13. Report all baselines **decomposed by present/absent**.
14. Run the full self-check suite (§24).
15. **Policy:** gate-dead null-fusion first; then the minimizing pair and operation of §4.5. Never top-$k$ truncate *as the policy* (it remains a baseline).
16. **Debits:** $\delta_{\mathrm{fuse}}=2\beta w_iw_j(1-G)/(1+r)$; $\delta_{\mathrm{evict}}=\mathrm{softplus}(\ell_l-\ell_h)$; ledger in nats; compare $\Delta$ in log space.
17. **Incremental Gram (§4.6) mandatory**, with both guards; full rebuild only as a drift diagnostic; refresh on $|G|>1+10^{-6}$.
18. Evict deletes the lighter; heavier untouched; destroyed provenance → retained $0$.
19. $\epsilon_r$ masks merge candidates only; eviction always admissible (fallback subsumed).
20. Debits use the current $\beta$; recompute in-forward; freeze at deployment.
21. P fusion-free; A/B with compression; $\tau$ two-pass on one fixed fused batch.
22. Report $\Lambda$, retained bound, destroyed fraction, and both operation counts — not "target kept."
23. Self-checks compare **like-for-like transcendentals** (torch-vs-torch for floor constants; cross-library tolerance $\ge10^{-6}$); gradient checks wrapped in `enable_grad` inside `no_grad` suites.

## 16. Complexity and storage

| Operation | v2.2.1 streaming | v2.3.1 |
|---|---:|---:|
| store state | $C(d+2)+C^2$ | $C(d+2)+C^2$ ($\Lambda$ replaces $D$) |
| write incl. compression | $O(C^2d)$ (rebuild) | $O(Cd+C^2)$ |
| Gram merge row | $O(C^2d)$ | $O(C)$ |
| policy scan | $O(C^2)$ | $O(C^2)$ (no $d$) |
| diffusion | $O(1)$ lazy | $O(1)$ lazy |
| posterior / gate | $O(Cd)$ | $O(Cd)$ |
| retrieval ($N$ steps) | $O(NCd)$ | $O(NCd)$ |
| corrected stability | $O(C^2d+C^3)$ | $O(C^2d+C^3)$ |

All $O(1)$ in stream length for fixed $C,d$. Storage is $O(Cd+C^2)$; "O(1) memory" is claimed **only** in stream length. Design rule: $C\le d/2$ keeps $C(d+2)+C^2\le d^2$ for $d\ge4$.

## 17. Evaluation plan

**17.1 Decomposed baselines (enforced).** Present/absent split for every method: cue-only, centroid/top-eigenvector, forced-commit kNN, **kNN + one train-fit threshold** (the like-for-like gate comparator), and at matched $C$: top-$k$ by learned dose / recency / importance / random / oracle prior, plus a query-aware keep-target upper bound. Aggregate comparisons across decomposed sides are invalid; beating claims must be made on present trials; report paired differences with SE over episodes.

**17.2 Retrieval battery.** Full/partial cues; absent/unrelated memory; hard neighbors; recency and importance buckets; age at fixed importance and vice versa; irregular-$\Delta t$ generalization; steady states; $\phi$ sweeps around floor and basin; $t_{\mathrm{forget}}$ vs measured collapse; hysteresis rate; gate sweep; init sensitivity; unseen sequences.

**17.3 Init calibration.** Arms R (random unit atoms) and A (shared-mean cluster / real embeddings): closure at $N$ and $10N$; legacy stability fraction and both floors; Gibbs basin and floor; capture; energy-increase counter; decomposed baselines.

**17.4 Identifiability protocol.** Run `probe`; check P recovers the generating decay in the split, A restores the basin, B does not move the law, deploy uses the saved $\tau$. If $p_{\mathrm{true}}>0.99$, change the data, not the optimizer.

**17.5 Ablations.** Forgetting on/off; dose-map variants; $\bar\varepsilon\equiv0$; Gibbs vs legacy; $\beta$ sweep; gate on/off and $\tau$ sweep; CE vs cosine vs mixed; staged vs probe; compression on/off; `--fuse_policy {debit, ward, random, delta}` where `ward` reproduces v2.2.1; the loss-calibrated eviction variant (§4.5); embedding geometry raw vs centered.

**17.6 Capacity regime.** Stream $T_{\mathrm{big}}\in\{100,500,1000\}$; sweep $C$; $\ge5$ seeds; paired SE; per-$C$ $\tau$ recalibration as a **train-batch diagnostic only**. Report decomposed loss, retained bound $e^{-\Lambda}$, destroyed fraction, ledger, fuse/evict counts, below-crossover fraction, ECE under load.

**Falsifiable predictions (replacing the withdrawn claim).**
1. **Spread geometry:** the policy's trajectory equals **streaming** dose-ordered eviction exactly (proved reduction); present loss matches offline top-$k$ within the streaming-vs-offline gap, to be measured. The v2.2.1 gap ($0.1299$ vs $0.1215$ at $C{=}4$) closes.
2. **Duplicate-heavy streams:** present loss **strictly beats** top-$k$, the gap increasing with $\rho_{\mathrm{dup}}$. By Lemma F the per-merge angular advantage is $O(\theta^2)$, so the effect is small per merge and accumulates.
3. **Within-policy** absent loss monotone in $C$. The cross-policy "absent inversion" (random eviction appearing best on absent) is a reporting artifact, not a target.
4. Theorem E holds per-event in the logs (0 violations).
5. Prior-CE gap unchanged ($\approx+0.02$ nats; law untouched, P fusion-free).

## 18. Core mathematical summary

**State.** $W=(1-S)J+M$; $\le C$ log-atoms; ledger $\Lambda$ in nats.
**Compress.** Gate-dead first; then the minimizing pair and operation of $\dfrac{A}{S}\min\big(\beta(1-r),\ \mathrm{softplus}(\ell_l-\ell_h)\big)$.
**Fuse.** $\mu=m/r$, $\ell'=\mathrm{logaddexp}(\ell_i,\ell_j)-\beta(1-r)$. **Evict.** Delete lighter; heavier exact.
**Gram.** $G_{ik}\leftarrow(w_iG_{ik}+w_jG_{jk})/r$, then $G_{ii}\leftarrow1$, then zero row/col $j$.
**Retrieve.** Gibbs $F$; decide at $u=q$ ($\mathrm{gap}>\ln\tau$), then descend.
**Limits.** $\theta^{*}=2\arccos(1-\ln2/\beta)$ at equal mass; $\ell_c\ge\mathrm{logaddexp}(\mathrm{base})-\Lambda_c$.
**Train.** Probe P → A → B; $\tau$ once; straight-through over pairs and operations.

## 19. Measured results

**19.1 v2.2.1 record (inherited; both corpora).** Law: $\omega=0.0104$, $\Gamma=0.1664$ (real); prior-CE gap $+0.0168$ (synthetic) / $+0.0259$ (real) nats; $\kappa$ re-fit $-4.50$; $\Gamma$ undershoots $\lambda$ by 10–17%. Deploy (real): gate $0.0241$ vs kNN+threshold $0.0263$; **present tie** $0.0171/0.0169$; absent win $0.0505/0.0622$; $\tau$ calibrated to $1.0$.

| $C$ | fusion present | top-$k$ learned | top-$k$ oracle | fusion absent |
|---:|---:|---:|---:|---:|
| 4 | 0.1299 | 0.1215 | 0.1189 | 0.0492 |
| 8 | 0.0757 | 0.0678 | 0.0633 | 0.0622 |
| 16 | 0.0283 | 0.0265 | 0.0257 | 0.0670 |
| 32 | 0.0184 | 0.0165 | 0.0165 | 0.0670 |

Fusion lost on present at every $C$; better on absent at $C{=}4$. Synthetic replicates the ordering. Unit checks: Theorem A $0/20000$ violations; Theorem B $\sim10^{-16}$; stable $\delta$ exact; selftest 36/36 after harness fixes.

**19.2 v2.3.1.** Theorems A, B, E, Propositions C, D, Lemmas F, G and the Gram identity are proved (§8.6, §4.6). Unit checks **TO MEASURE** in your environment. **All end-to-end claims TO MEASURE** under §17.6. No superiority claim is asserted before that run.

**19.3 Known residuals.** $\Gamma$ undershoot; prior-CE rise after P by design; fusion-in-loop cost on the unfused readout; the gate's advantage is rejection, not semantic retrieval.

## 20. Central research hypothesis

1. Retrieval beats cue-only and centroid on present; abstains on absent. *(MEASURED.)*
2. The forgetting rate is recoverable from the marginal law only. *(MEASURED.)*
3. Vagueness is basin collapse at a computable time, accelerated by merge debits. *(THEOREM.)*
4. Compression is evidence-monotone, parameter-free, auditable, fixed-size, and per-event debit-dominant over dose-ordered eviction. *(THEOREM A/B/E; unit TO MEASURE.)*
5. The decision precedes descent. *(THEOREM + witness.)*
6. The policy reduces to eviction in spread geometry and strictly dominates it below crossover. *(THEOREM corollaries; end-to-end TO MEASURE.)*
7. The latent is a Transformer interface. *(SPECIFIED; TO MEASURE.)*

**Withdrawn:** "fusion beats top-$k$ on present in spread geometry"; "the gate beats kNN+threshold on present." The claimed advantage is **rejection plus guarantees**, not semantic retrieval.

## 21. Supersession

**21.1 Inherited.** v2.2.1 §21 items 01–30 remain withdrawn (small-step descent; crosstalk-as-limit; spectral merging; target-only blur; the $10^{-8}$ guard reachability; $\mu$ rounding; naive $1-e^{-x}$; cue-only sufficiency; no-new-basins; the fixed-point floor independent of $\eta,N$; legacy adequacy; $W$-only retrieval; the $\sqrt{(T-1)/d}$ Gibbs floor; argmax-as-decision; half-closure vagueness; decorrelated-sampling sufficiency; one-$\kappa$-two-jobs; $S$-as-useful-memory; eval-fit $\tau$; every-operation-convex; top-$k$-as-policy; pruning-as-exception; exact block odds; monotone max-gap; $\beta$-independent ledger; Loewner monotonicity; conserved $S$).

**21.2 v2.2.1 → v2.3.1.**
31. "Fusion must beat top-$k$ on present at small $C$." → **FALSIFIED** on both corpora; replaced by Theorem E plus the regime predictions of §17.6.
32. "Null-fusion is decision-free." → Only the gate-dead branch; debit-eviction changes decisions by design.
33. "Null-fusion is restricted to gate-dead atoms." → Extended to live lighter atoms via the derived comparison.
34. "$C{=}1$ antipodal fallback is a special case." → Subsumed; eviction is always admissible.
35. "Streaming Gram rebuild $O(C^2d)$ is acceptable." → Withdrawn; the incremental row law is mandatory.
36. "Ledger $D_c=\sum(1-r)$ with debit $\beta D_c$ outside the logaddexp." → Ledger is nats; $\Lambda_c$ tracks merge debits only; the outside form is exact for a single merge and conservative for chains; ages sit per constituent inside.

**21.3 Interim relevance draft.**
37. "Per-atom relevance $R_i$ with EMA rate $\eta$." → Withdrawn: a new free parameter; the dose already **is** the identified target law; double-counting.
38. "$c^{Q}=\mathrm{Ward}\times(R_i+R_j)$." → Withdrawn: not loss-derived; replaced by the expected-debit comparison.
39. "softmax(sims·10) relevance update; $\sum_i R_i\approx C$." → Withdrawn: arbitrary temperature; an EMA of softmax weights sums to $\approx1$, not $C$.
40. "Theorem A proof: $E'\le Ae^{\beta\mu\cdot u}$." → Incorrect without the debit factor; the correct two-step proof is restated in §8.6.
41. "Periodic re-orthogonalization of the Gram." → Withdrawn; drift check plus refresh (§4.6).
42. "Timing-slope complexity self-check." → Replaced by deterministic precision and identity checks.
43. "The gate beats kNN+threshold on present (real corpus)." → Not reproduced (tie); the advantage is narrowed to absent/rejection.

**21.4 Pre-release audit corrections applied to this document.**
(i) $\delta_{\mathrm{evict}}$ is the exact block log-dose reduction and the unique survivor-preserving debit, **not** the minimal uniform debit; the smaller angle-dependent minimum is given and forbidden because it inflates the survivor.
(ii) $\delta_{\mathrm{fuse}}$ is sufficient and tight at equal masses, **not** minimal in general.
(iii) Ledger split: $\Lambda_c$ merge-only per surviving block; episode ledger sums both; destroyed targets report hard $0$.
(iv) Currency caveat added: the debit currency under-books eviction's retrievability loss, so the policy is conservative toward merging; Theorem E is ledger-currency dominance.
(v) Gradient statement corrected: $\delta_{\mathrm{evict}}$ and both argmins are selection-only.
(vi) "Gradual" narrowed: eviction is abrupt per target; gradualness holds on the merge branch.
(vii) Reduction condition compares the two minima; the exact match is to **streaming** eviction.
(viii) Gram guards added: old rows, explicit $G_{ii}\leftarrow1$, zero row $j$ last.
(ix) LaTeX repaired: `^{*}` replaces the invalid `^\*`; `\min` with subscripts replaces `\arg\min`; `\leftarrow` replaces `\mathrel{-}=`; `\ldots` replaces `..`; `\mathrm{}` replaces `\text{}`.

## 22. Changes from v2.2.1, and why

**22.1 The structural finding.** Lemma F shows merging is *always* angularly cheaper than evicting the lighter atom, by the factor $w_h\in[\tfrac12,1)$. Eviction competes **only** because it levies no evidence debit on the survivor. Proposition D shows the expected debit of a merge is $\beta\times\mathrm{Ward}/S$ — so Ward was already the correct merge ranking under the identified target law. The failure was the **forced merge**: v2.2.1 had no eviction option for live atoms and paid the debit on every overflow. v2.3.1 adds exactly one derived comparison, $\min\big(\beta(1-r),\ \mathrm{softplus}(\ell_l-\ell_h)\big)$, weighted by $A/S$. The policy degenerates to the empirical champion where the champion is right, and strictly dominates it where duplicates make merging nearly free. No relevance statistics, no EMA, no new constants: the crossover is the readout's own resolution angle.

**22.2 Section map.** §3.2–3.3 ledger corrected to nats with the chain bound; §4.4 two null-fusion triggers; **§4.5 new policy with debit semantics and currency caveat**; **§4.6 incremental Gram with guards**; §5 D5 extended to the argmin; §7 zero-parameter statement and two-pass $\tau$; §8.6 Proposition D, Lemmas F/G, Theorem E, reduction corollary; §9 corrected gradient paths; §10 duplicate generator; §13 gradual/abrupt row; §14 narrowed guarantees; §15 rules 15–23; §16 complexity fixed; §17.6 revised predictions; §19 record plus TO MEASURE; §20 revised claims; §21 items 31–43 plus the audit list; §24 suite 44.

**22.3 Deliberately unchanged.** State class; convex write; dose maps; diffusion generator and all six laws; impure-write decomposition; Gibbs readout and descent proof; decide-before-descend and the $\tau$ protocol; probe identifiability split; reinforcement; decoder; legacy ablation; the form of the LM interface. The revision is confined to the compression policy, the ledger currency, and the Gram update.

## 23. Language-model interface (SPECIFIED; TO MEASURE)

**Latent.** $(\mu_c,\ell_c)\in\mathbb{R}^{C\times(d+1)}$. **Read.** The Gibbs posterior at $u=q$ is one cross-attention head with tied key/value:
$$u=\sum_c\mathrm{softmax}_c\big(\beta q\cdot\mu_c+\ell_c\big)\mu_c+p_0q.$$
Prepend $C$ memory tokens; keys/values $W_K\mu_c$, $W_V\mu_c$; $\beta$ absorbs $1/\sqrt{h}$; attention-logit bias $\ell_c$ (a learned per-token ALiBi); null token with bias $\kappa$; gate as the residual mix $g=\sigma(\mathrm{gap}-\ln\tau)$. The $N$-step descent is optional recurrent refinement over the same tokens.

**Write.** $z_t=\mathrm{normalize}(W_zh_t)$; surprise is the LM's own signal $s_t=-\ln p_{\mathrm{LM}}(x_t\mid x_{<t})$; feed the existing dose map. Per token: diffuse, write, compress-on-overflow, $O(Cd+C^2)$, state never grows. The compression channel now has two token-level operations — **merge** (resultant plus debit) and **drop-with-ledger** (evict lighter) — both bounding $C$ and both evidence-monotone.

**Transfer caveat.** $\beta,\kappa$ were calibrated on MiniLM sentence geometry and $\mathrm{Exp}(1)$ importance; LM hidden states and NLL-scaled surprise require stage A/B recalibration with the law frozen. The functional form transfers; the calibration does not transfer for free. Evaluate retrieval separately (§12).

## 24. Self-check suite (44 checks)

**State and dynamics (1–10).** 1 invariants under write/diffuse/compress sequences; 2 closed form vs sequential; 3 semigroup with **cumulative absolute times**; 4 spectral invariance comparing the **top eigenvector only** (rank-1 null space is degenerate); 5 steady state; 6 impure-write decomposition; 7 empty memory returns $q$; 8 permutation invariance; 9 uniform log-shift invariance; 10 $s=0$ finite log-dose with zero clamped-branch gradient, compared against a **torch-computed** floor reference at $10^{-6}$ tolerance.

**Readout and gate (11–18).** 11 energy-increase fraction 0, guard 0 over a $\beta$ grid; 12 descent at $N$ and $10N$ agrees on a converged configuration; 13 planar hysteresis witness; 14 gate limits; 15 streaming vs batch gated equality with compression active; 16 $\tau$ determinism and flat-sweep-to-argmax; 17 basin/floor/resolution vs measured; 18 cue-free gradient restoration.

**Legacy and numerics (19–26).** 19 factored vs dense corrected stability including the saddle; 20 legacy descent for every $\eta$ and $W$/$M$ trajectory equivalence; 21 capacity–closure witness via the **angle-closing fraction** at $\phi=0.6$; 22 masked denominators (guard, reinforce branch, resultant mask) with gradient parts under `enable_grad`; 23 stable inverse softplus; 24 $-\mathrm{expm1}$ accuracy; 25 decoder-normalization Jacobian bound; 26 no-match prevents $\kappa\to-\infty$ and the $p_{\mathrm{true}}$ warning.

**Compression (27–36).** 27 $C\ge T$ bit-for-bit regression; 28 Theorem A domination, 0 violations / 20 000 queries; 29 Theorem B commutation; 30 fusion accounting (trace one, PSD, deficit to $J$); 31 gate-dead criterion exactness; 32 gate-dead deletion decision-invariance; 33 ledger single-merge exactness and the conservative chain bound; 34 streaming vs batched replay equality (provenance-keyed); 35 $C{=}1$ antipodal with $\omega=10^{-3}$: merge masked, **evict branch fires**, ledger $\ln2$; 36 Ward matches Proposition C to first order and the capacity bound holds through write/reinforce/compress sequences.

**New in v2.3.1 (37–44).**
37. **Spread reduction:** when $\min\Delta_{\mathrm{fuse}}\ge\Delta_{\mathrm{evict}}(l^{*},h^{*})$ at every step, the trajectory equals streaming lightest-first eviction exactly (identical kept sets).
38. **Duplicate reduction:** identical equal-mass atoms select merge with $\delta=0$ and $E'=E_i+E_j$ exactly; eviction would pay $\ln2$.
39. **Crossover identity:** equal masses at $\theta^{*}$ give $\delta_{\mathrm{fuse}}=\delta_{\mathrm{evict}}=\ln2$ to $10^{-12}$.
40. **Theorem E:** $\Delta_{\mathrm{policy}}\le\Delta_{\mathrm{evict}}(l^{*},h^{*})$, 0 violations over $10^4$ random states.
41. **Incremental Gram:** row law vs full rebuild $<10^{-10}$ over $10^3$ mixed operations (float64), including both guards; drift-refresh triggers.
42. **Ledger nats:** $\Lambda$ additivity across merge histories; chain bound; single-merge exactness; destroyed targets report retained $0$; episode ledger sums both operation kinds.
43. **Policy commutation:** a uniform log-shift leaves $\Delta$ and the minimizing pair/operation invariant to $10^{-13}$.
44. **Lemmas F and G:** $c_{ij}/S=p_lw_h(1-G)$ to $10^{-12}$ on random pairs; $\Delta_{\mathrm{evict}}$ minimized at (lightest, heaviest) over $10^4$ random states, 0 violations.

---

**End of spec v2.3.1.** One comparison, two operations, zero new parameters:
$$\min_{(i,j)}\ \frac{A}{S}\,\min\big(\beta(1-r),\ \mathrm{softplus}(\ell_l-\ell_h)\big).$$
The fusion theorem is untouched; the policy now knows when *not* to fuse; the Gram update is exact and incremental; the ledger is honest about what eviction destroys; and the withdrawn superiority claim is replaced by a proven per-event dominance plus a falsifiable duplicate-regime prediction. Run §17.6 before asserting any end-to-end advantage. All displays use MathJax/KaTeX-safe macros (`\mathrm{}` for upright names, no `\operatorname`, no `\text`, `^{*}` for stars).