# HopMem — Frozen Specification v2.1.1

**Status.** This document supersedes v2.0.4. It retains the v2.0.4 *state* (density matrix, single convex primitive, mandatory depolarizing diffusion, dose additivity) and replaces the v2.0.4 *readout*, *angular-limit theory*, *forgetting interpretation*, and *training/identifiability protocol*. Reference implementation: `hopmem.py` v3.6 (26 self-checks). The withdrawal list is §21; the rationale for every change is §22.

**Epistemic labels used throughout.**
- **THEOREM** — proved in the text, exact arithmetic.
- **MEASURED** — observed in the reference implementation on the setup of §19; single seed, one geometry family, not a general claim.
- **DESIGN RULE** — heuristic with a stated derivation, not enforced.
- **HYPOTHESIS** — untested.

**Numbers in §19 come from reported runs of the reference implementation, not from an independent audit.** All theorems are hand-derived; all identities flagged as verified were checked numerically in float64 against an independent formulation.

---

## 1. Objective

A semantic associative memory whose state is a trace-one positive-semidefinite matrix, whose writes are importance-dosed convex mixtures, whose forgetting is a mandatory learnable-rate depolarizing channel, and whose retrieval is a **cue-conditioned Gibbs posterior with an explicit no-match hypothesis**, followed by an energy-descending refinement that is gated by a calibrated decision.

The system supports:

- fixed-size *state* invariants ($W\succeq0$, $\operatorname{tr}W=1$) exactly, under all core operations;
- one convex update primitive for diffusion, writing, reinforcement;
- mandatory time-based forgetting with learnable rate, **additive in the log-dose**;
- priority-weighted writing with a learnable write/diffusion coupling;
- cue-conditioned retrieval with **guaranteed energy descent at every inverse temperature**;
- **abstention** (return the cue when no stored direction is supported);
- **pattern selection** among overlapping traces, with angular resolution set by $\beta$;
- predictable, **sharp** forgetting: the retrieval basin vanishes at a computable time;
- separation of the *forgetting law* from the *readout calibration*, with a protocol that identifies each.

The system does not support:

- exact episodic recall of arbitrary records;
- unlimited capacity (effective capacity is bounded by pruning, §16);
- a readout that is a function of $W$ alone (§3.6 — this is the central structural change);
- guaranteed unique basins or global optimality;
- immunity to dominant-direction capture **under the legacy quadratic readout** (§8.6);
- angular resolution below the Gibbs floor (§8.4);
- identification of $\Gamma$ from the conditional (cue) objective alone (§10.4);
- a single scalar serving both marginal and conditional calibration (§7.4).

---

## 2. Inputs

$d\ge2$; at event time $t$:

$$z_t\in\mathbb R^d,\ \|z_t\|_2=1,\qquad q\in\mathbb R^d,\ \|q\|_2=1,\qquad s_t\ge0\ \text{finite}.$$

$\Delta t\ge0$ finite is elapsed time since the previous event; use $\Delta t=1$ if timestamps are absent ($\Gamma$ absorbs the unit). No scale assumption on $s_t$: $\omega$ absorbs the importance scale. $s_t$ is a proxy, not ground truth.

A query carries **no** label. The *no-match* case is a property of the episode, not an input flag: the memory must decide it (§8.5).

---

## 3. Memory state

### 3.1 Density-matrix form (state of record)

$$W\succeq0,\qquad \operatorname{tr}W=1,\qquad W_0=J=\frac Id.$$

An empty memory is $J$, not $0$.

### 3.2 Canonical signal decomposition

Every reachable state has the exact form

$$\boxed{\;W=(1-S)J+M,\qquad M=\sum_i\alpha_i z_iz_i^{\top},\qquad \alpha_i\ge0,\qquad S=\sum_i\alpha_i\in[0,1).\;}$$

$S$ is tracked as a scalar with the same convex recursion as $W$ (§4.2). For finite scores and finitely many operations $S<1$ strictly, hence

$$\lambda_{\min}(W)\ge\frac{1-S}{d}>0,\qquad \lambda_{\max}(W)\le1-(1-S)\Bigl(1-\frac1d\Bigr)<1 .$$

Pure states are limits, not reachable states.

### 3.3 Log-atom form (preferred representation)

$$\boxed{\;\ell_i=\ln\alpha_i,\qquad \text{state}=\{(z_i,\ell_i)\}_i,\ \ S=\sum_i e^{\ell_i}.\;}$$

**THEOREM (closed form).** Under the diffuse-then-write convention,

$$\boxed{\;\ell_i(t_q)=\ln\!\bigl(a_i(1-\varepsilon_i)\bigr)\;-\;\omega\!\!\sum_{j>i}\!s_j\;-\;\Gamma\,(t_q-t_i).\;}$$

*Proof.* $\ln(1-a_j)=-\omega s_j$ exactly, and diffusion multiplies $M$ by $e^{-\Gamma\Delta t}$. Products of these factors become sums of logarithms. $\square$

**Forgetting is an additive logit drift.** This is the operational content of dose additivity, and it is what makes the Gibbs readout (§8) natural.

### 3.4 Sign-blindness

$zz^\top=(-z)(-z)^\top$: the state stores unoriented directions. The readout is not sign-blind: $u_0=q$ and the cue atom's logit pin the hemisphere.

### 3.5 State scalars

$$P=\operatorname{tr}(W^2)=\frac{1-S^2}{d}+\|M\|_F^2,\qquad \hat P=\frac{dP-1}{d-1}\in[0,1].$$

Diagnostics only. **$S$ is not an indicator of retrieval quality** (§19: $S\approx0.05$ with excellent retrieval). Only dose *ratios* and the cue comparison are pinned by the readout.

### 3.6 Atom-dependence (structural change from v2.0.4)

**THEOREM.** The legacy quadratic readout depends only on $(M,\gamma q)$. The Gibbs readout depends on the **atom set** $\{(z_i,\ell_i)\}$, not on $M$ alone.

*Proof.* Two atom sets with equal $M$ (e.g. the original atoms and the spectral decomposition of $M$) give equal $u^\top Mu$ but different $\sum_i\alpha_ie^{\beta z_i^\top u}$. $\square$

Consequences, stated plainly:

- $W$ remains the state of record: it carries the invariants, the diffusion laws, and the legacy readout.
- The **core** readout requires the atom set. Storage is $O(Td)$, not $O(d^2)$ (§16).
- "The memory is a second-moment matrix" is no longer accurate for the core path. The core is a log-linear Gibbs model over atoms whose log-priors are produced by a density-matrix-valued convex dynamics.

---

## 4. The unified convex primitive

### 4.1 Definition and invariants

$$\boxed{\;W\leftarrow(1-a)W+aT,\qquad a\in[0,1],\ T\succeq0,\ \operatorname{tr}T=1.\;}$$

PSD, trace one, and symmetry are preserved by linearity; no projection or renormalization is ever needed.

| Operation | Target $T$ | Weight $a$ |
|---|---|---|
| Diffusion (§5, mandatory) | $J$ | $p=1-e^{-\Gamma\Delta t}$ |
| Write (§6) | $(1-\varepsilon_t)z_tz_t^\top+\varepsilon_tJ$ | $a_t=1-e^{-\omega s_t}$ |
| Reinforcement (§11, optional) | $u^\ast u^{\ast\top}$ | $\beta_{\mathrm{reinf}}$ |

### 4.2 Scalar and log-atom bookkeeping

$$
\begin{aligned}
\text{diffusion:}&\quad \ell_i\leftarrow\ell_i-\Gamma\Delta t,\qquad S\leftarrow e^{-\Gamma\Delta t}S,\\
\text{write:}&\quad \ell_j\leftarrow\ell_j-\omega s_t\ (j<t),\quad \text{append }\ell_t=\ln\!\bigl(a_t(1-\varepsilon_t)\bigr),\quad S\leftarrow(1-a_t)S+a_t(1-\varepsilon_t),\\
\text{reinforce:}&\quad \ell_j\leftarrow\ell_j+\ln(1-\beta),\quad\text{append }\ln\beta,\quad S\leftarrow(1-\beta)S+\beta .
\end{aligned}
$$

These are exact. $S$ is **not monotone**.

### 4.3 Composition and ordering

Diffusion maps compose as a semigroup. Different operations do not commute; the exact commutator is $ap(T-J)$. **Ordering convention:** at each event, diffuse to the event timestamp, then write, then optionally reinforce. Retrieval runs at frozen state.

### 4.4 Optional pruning (outside the single primitive)

To bound storage (§16), atoms may be dropped:

$$\boxed{\;\text{drop atom }i\ \text{if}\ \ell_i<\max_j\ell_j-2\beta-\ln(T/\epsilon_{\mathrm{prune}}).\;}$$

**THEOREM (bound).** For all unit $u$, $\beta z_i^\top u+\ell_i\in[\ell_i-\beta,\ell_i+\beta]$, so the pruned atom's posterior share satisfies $p_i/p_{\max}\le\epsilon_{\mathrm{prune}}/T$ uniformly in $u$.

**THEOREM (invariants).** Dropping atom $i$ and returning its mass to $J$,
$W\leftarrow W-\alpha_iz_iz_i^\top+\alpha_iJ$, preserves PSD and trace one, because $M\succeq\alpha_iz_iz_i^\top$ implies $W-\alpha_iz_iz_i^\top\succeq(1-S)J\succeq0$.

**This map is not a convex mixture.** It is a specified approximation with a stated bound, and it breaks the "every operation is one primitive" property. Disabled by default; enable only with the bound logged.

---

## 5. Mandatory diffusion

Generator $\dot W=\Gamma(J-W)$, $\Gamma>0$ learnable; exact discrete map

$$W\leftarrow e^{-\Gamma\Delta t}W+(1-e^{-\Gamma\Delta t})J,\qquad \ell_i\leftarrow\ell_i-\Gamma\Delta t .$$

CPTP as a convex mixture of identity and replacer channels; real Kraus form $K_0=\sqrt{1-p}\,I$, $K_{ij}=\sqrt{p/d}\,E_{ij}$.

**Exact laws (unchanged from v2.0.4).**

1. Eigenvectors fixed; $\lambda_i(t)-\lambda_j(t)=e^{-\Gamma t}(\lambda_i(0)-\lambda_j(0))$; ordering preserved; $\operatorname{range}(M)$ unchanged.
2. $\|W(t)-J\|_F=e^{-\Gamma t}\|W(0)-J\|_F$; $P(t)-\tfrac1d=e^{-2\Gamma t}(P(0)-\tfrac1d)$.
3. Entropy non-decreasing (majorization by the uniform vector); $D(W\|J)=\ln d-S_{\mathrm{vN}}(W)\downarrow0$.
4. Signal half-life $t_{1/2}=\ln2/\Gamma$.
5. Strict contraction, no CPTP inverse.
6. **Relative spectral invariance of $M$:** eigenvalue ratios, normalized gaps, effective rank, and spectral entropy of $M/S$ are exactly invariant. Diffusion is pure scalar gain decay on the signal.

**New in v2.1.1.** In log-atom form diffusion is a *subtractive logit drift*, which is why it produces a **sharp basin collapse** under the Gibbs readout (§8.4) rather than a smooth gain fade.

Steady states under periodic writing (write-only regime, gap $g$ after the last write):

$$S^\ast_{\mathrm{post}}=\frac{a(1-\varepsilon)}{1-(1-a)e^{-\Gamma\Delta t}},\qquad S^\ast_{\mathrm{pre}}=e^{-\Gamma\Delta t}S^\ast_{\mathrm{post}},\qquad S(g)=S^\ast_{\mathrm{post}}e^{-\Gamma g}.$$

$g>\Delta t$ is valid only if writes stop. Measure at the labeled instant.

---

## 6. Writing

### 6.1 Dose

$$a_t=1-e^{-\omega s_t},\qquad \omega=\operatorname{softplus}(a_\omega)>0 .$$

$a(0)=0$, strictly increasing, $a_t\in[0,1)$ for finite $s_t$. Compute as $-\operatorname{expm1}(-\omega s_t)$.

### 6.2 Impure writes are a reparameterization

$$T_t=(1-\varepsilon_t)z_tz_t^\top+\varepsilon_tJ,\qquad \varepsilon_t=\bar\varepsilon(1-a_t),\ \ \bar\varepsilon=\sigma(a_\varepsilon).$$

**THEOREM (exact decomposition).**

$$(1-a)W+a\bigl[(1-\varepsilon)zz^\top+\varepsilon J\bigr]=(1-a_2)\bigl[(1-p')W+p'J\bigr]+a_2zz^\top,$$
$$a_2=a(1-\varepsilon),\qquad p'=\frac{a\varepsilon}{1-a+a\varepsilon}.$$

An impure write **is** a diffusion followed by a pure write. $\bar\varepsilon$ is not an independent forgetting mechanism.

**New in v2.1.1.** At small dose ($a\approx\omega s$), $\varepsilon_t\approx\bar\varepsilon$ and the injected mass is $\approx\omega s(1-\bar\varepsilon)$: $\bar\varepsilon$ is a **pure scale factor absorbed by $\kappa$**, hence unidentifiable from the marginal objective. Freeze it during law identification (§10.3).

### 6.3 Contribution retention

$$\alpha_i(t)=a_i(1-\varepsilon_i)\!\!\prod_{j>i,\ \text{writes}}\!\!(1-a_j)\!\!\prod_{\text{diffusions after }i}\!\!e^{-\Gamma\Delta t},$$

with additional $(1-\beta_k)$ factors for reinforcements. Importance sets *initial* mass; it does not slow decay.

### 6.4 Angular limits — **replaced**

The v2.0.4 claim that $\theta_{\mathrm{floor}}\approx\|x_\perp\|/(\tilde m+\gamma)$ is a fixed-point property independent of $\eta,N$ is **withdrawn**. It counts the cross-talk *force* and ignores the cross-talk *curvature*. The corrected analysis is §8.6 (legacy readout) and §8.4 (core readout).

---

## 7. Learnable parameters

### 7.1 Parameter set

$$\theta=\{\underbrace{\omega,\ \Gamma,\ \bar\varepsilon}_{\text{state law}},\ \underbrace{\beta,\ \kappa}_{\text{Gibbs readout}}\},\qquad \text{legacy: }\{\eta,\chi\},\qquad \text{optional: }\bar\beta .$$

$$\omega=\operatorname{softplus}(a_\omega),\quad \Gamma=\operatorname{softplus}(a_\Gamma),\quad \beta=\operatorname{softplus}(a_\beta),\quad \bar\varepsilon=\sigma(a_\varepsilon),\quad \kappa\in\mathbb R\ \text{free}.$$

$\kappa=\ln(\text{cue-atom mass})$ is unconstrained: no sigmoid saturation pathology. The legacy cue strength remains derived, $\gamma=\chi/\eta$ with $\chi=\sigma(a_\chi)$.

### 7.2 The decision threshold is calibrated, not learned

$$\boxed{\;\tau>0:\ \text{commit iff }\operatorname{gap}(q)>\ln\tau\quad(\S8.5).\;}$$

$\tau$ is one scalar fit by decision-theoretic risk minimization on a **fixed training calibration batch**, saved with the checkpoint, and applied identically in batch and streaming paths. It is not a gradient parameter and not a dynamics constant.

### 7.3 Suggested initialization

| Parameter | Init | Meaning |
|---|---:|---|
| $\omega$ | $-\ln0.95\approx0.0513$ | $a(s{=}1)=0.05$ |
| $\Gamma$ | $10^{-3}$ | half-life $\approx693$ steps |
| $\bar\varepsilon$ | $0.01$ | near-pure writes |
| $\beta$ | $16$ | see DESIGN RULE below |
| $\kappa$ | $\ln10^{-4}\approx-9.21$ | cue-atom log-prior |
| $\eta,\chi$ (legacy) | $1,\ 0.01$ | v2.0.4 defaults |

**DESIGN RULE (initialization).** For a fresh trace of mass $\alpha$ and expected cue overlap $c$, retrieval fires when $\beta(1-c)<\ln\alpha-\kappa$. Choose $\beta_0\approx(\ln\alpha_0-\kappa_0)/(1-c_{\text{typ}})$.

Softplus inverse: $\ln(\operatorname{expm1}(y))$ for $y<20$, else $y+\ln(1-e^{-y})$ (the naive form overflows).

### 7.4 Identifiability — **rewritten**

1. **$\Gamma$ is identified by the marginal law; $\beta,\kappa$ by the conditional law.** The cue posterior's gradient
 $$\frac{\partial\,\mathrm{CE}}{\partial\Gamma}=\mathrm{age}_k-\mathbb E_p[\mathrm{age}]$$
 vanishes when $p$ is one-hot. **MEASURED:** $|\partial\mathrm{CE}/\partial a_\Gamma|$ falls from $5.8\times10^{-3}$ (cue-free) to $6.5\times10^{-5}$ (informative cue), a factor $\approx90$.
2. **Write-dilution substitutes for time.** $-\omega\sum_{j>i}s_j$ is a *count* proxy for age. With informative cues the model buys its decay there and leaves $\Gamma\approx\Gamma_0$. Decorrelating $(s,\Delta t)$ is **necessary but not sufficient**; the prior must carry signal.
3. **$\kappa$ has two incompatible jobs.** Marginal null-odds (law identification) and conditional commit threshold (retrieval). One scalar cannot serve both; `prior CE` rising after the law stage is expected, not a fault.
4. **$(\beta,\kappa)$ share a flat ray.** Only $(\ell-\kappa)/\beta$ is pinned. $\beta$ grows without bound when the true cue likelihood ratio is large; this is correct, not a pathology.
5. **$\bar\varepsilon$ is pure scale at small dose** (§6.2), unidentifiable from the marginal.
6. **$S$ is not pinned.** Absolute signal mass is a near-gauge; small $S$ is compatible with excellent retrieval.
7. **$\omega$–$\Gamma$ split requires the marginal stage.** Report the per-event split $\omega\mathbb E[s]+\Gamma\mathbb E[\Delta t]$ against the data rate, not $\Gamma$ alone.

---

## 8. Retrieval

### 8.1 Core: Gibbs free energy

$$\boxed{\;F(u)=\frac1\beta\ln\Bigl(e^{\beta q^\top u+\kappa}+\sum_i\alpha_i e^{\beta z_i^\top u}\Bigr),\qquad E=-F,\qquad\|u\|_2=1.\;}$$

The cue enters as **one extra atom** with log-mass $\kappa$: the explicit *no-match* hypothesis. With
$$p=\operatorname{softmax}\bigl([\beta Z^\top u+\ell,\ \beta q^\top u+\kappa]\bigr),\qquad \nabla F(u)=p_q\,q+\sum_i p_i z_i,$$

$$\boxed{\;u_0=q,\qquad u_{n+1}=\frac{\nabla F(u_n)}{\|\nabla F(u_n)\|_2},\qquad u^\ast=u_N.\;}$$

**There is no step size.** $\beta$ is an inverse temperature, not a learning rate.

**Guard.** If $\|\nabla F\|<\epsilon_{\mathrm{fall}}$ or non-finite, set $u_{n+1}=q$ and count it. Use a masked denominator so the unselected branch cannot poison gradients. Recommended $N=20$.

### 8.2 Descent theorems

**THEOREM (descent for every $\beta>0$).** $F$ is convex (log-sum-exp of affine functions). $u^+=\nabla F/\|\nabla F\|$ maximizes $\langle\nabla F(u),\cdot\rangle$ over the unit sphere, so
$$F(u^+)\ \ge\ F(u)+\langle\nabla F(u),u^+-u\rangle\ \ge\ F(u),$$
with equality iff $u^+=u$. Hence $E(u^+)\le E(u)$ for **every** $\beta>0$. $\square$

This is the same argument as v2.0.4 §8.3, applied to a different convex functional. The small-step caveat for retracted gradient descent does not apply.

**THEOREM (weighted square-summability).** Using $\langle\nabla F,u^+-u\rangle=\tfrac12\|\nabla F\|\|u^+-u\|^2$,
$$\sum_{n<N}\|\nabla F_n\|\,\|\Delta u_n\|_2^2\ \le\ 2\bigl(F(u_N)-F(u_0)\bigr)\ <\ \infty,$$
since $\|\nabla F\|\le1$ (convex combination of unit vectors) bounds $F$ on the sphere. If the guard never fires, $\|\nabla F_n\|\ge\epsilon_{\mathrm{fall}}$ gives the unweighted bound $\sum\|\Delta u_n\|^2\le2\Delta F/\epsilon_{\mathrm{fall}}$. Unique-limit convergence is not claimed. $\square$

**Empty memory.** All $\alpha_i=0$ ⇒ $p_q=1$ ⇒ $\nabla F=q$ ⇒ $u^\ast=q$ **exactly**. (Implementation clamps $\ell\ge\ln10^{-30}$, giving $q$ to machine precision.)

### 8.3 Limits and invariances

- $\beta\to0$: $\nabla F\to\bigl(\sum_i\alpha_iz_i+e^\kappa q\bigr)/(S+e^\kappa)$ — the **dose-weighted first moment**. The Gibbs family is *not* a reparameterization of the quadratic readout; the quadratic uses the second moment.
- $\beta\to\infty$: hard nearest-atom with log-dose prior (soft-NN).
- **Permutation invariance** in the atoms; **shift invariance** under $\ell_i\mapsto\ell_i+c$, $\kappa\mapsto\kappa+c$. Only ratios matter.
- Blending and selection are one continuous knob.

### 8.4 Basin, floor, resolution, forgetting

Let $c_k=q^\top z_k$.

**Basin (DESIGN RULE, conservative).** At $u=q$ the atom beats the cue iff $\beta(1-c_k)<\ell_k-\kappa$, so
$$\boxed{\;\theta_{\mathrm{basin}}=\arccos\Bigl(1-\frac{\ell_k-\kappa}{\beta}\Bigr)\ \text{if }\ell_k>\kappa,\ \text{else }0.\;}$$
Conservative because moving toward the atom only widens its lead.

**Floor (first order).** At the atom-$k$ fixed point, with $p$ evaluated at $z_k$ and $P_k=I-z_kz_k^\top$,
$$\boxed{\;\theta_{\mathrm{floor}}\approx\Bigl\|\sum_{i\ne k}\frac{p_i}{p_k}P_kz_i+\frac{p_q}{p_k}P_kq\Bigr\|,\qquad \frac{p_i}{p_k}=e^{\beta(z_i^\top z_k-1)+\ell_i-\ell_k}.\;}$$
For near-orthogonal atoms $\|P_kz_i\|\approx1$, so
$$\theta_{\mathrm{floor}}\approx\frac{\bar\alpha}{\alpha_k}e^{-\beta}\sqrt{T-1}.$$

**The prefactor is $\sqrt{T-1}$, not $\sqrt{(T-1)/d}$.** The legacy force carries $c_j\sim1/\sqrt d$; the Gibbs force does not. Ratio to the legacy floor $\approx e^{-\beta}\sqrt d\,(\alpha+\gamma)/\alpha$.

**MEASURED:** $\beta=6$, $d=384$, $T=20$, equal masses: predicted $1.32\times10^{-2}$, measured $1.18\times10^{-2}$ rad. Doubling $\beta$ to 12 divided the floor by 320 ($e^6\approx403$).

**Resolution.** Two atoms are separated iff $\beta(1-z_1^\top z_2)\gtrsim|\ell_1-\ell_2|$. Angular resolution is $\sim1/\beta$.

**Forgetting (sharp).** $\ell_k(t)=\ell_k(0)-\Gamma t$, so the basin shrinks and **vanishes** at
$$\boxed{\;t_{\mathrm{forget}}=\max\Bigl(0,\ \frac{\ell_k(0)-\kappa}{\Gamma}\Bigr).\;}$$
This is v2.0.4's $t_{1/2\mathrm{cl}}$ with $\gamma\to e^\kappa$, now a *threshold* rather than 50 % closure. Past it, retrieval returns the cue.

### 8.5 Hysteresis and the gate — **new, mandatory**

**THEOREM (the null basin is strictly smaller than the argmax region).** Moving toward an atom raises its logit and lowers the cue's, so a marginal null win at $u=q$ can run away to that atom.

**Deterministic witness (verified).** $d=2$, $T=1$, $\beta=16$, $\ell=-3$, $\kappa=-6.5$, atom at $0.70$ rad from $q$: logits $9.237$ (atom) vs $9.500$ (null) — null wins at $u=q$ — and the map converges to the atom ($\cos\to1.0000$).

**MEASURED:** on real embeddings, `abstain` $=0.898$–$0.980$ while the raw map's absent-trial loss is $0.46$–$0.52$; the fraction of absent trials where argmax said *no match* but the map left the cue is $0.56$–$0.76$.

**Therefore the decision is taken before descent:**

$$\boxed{\;\operatorname{gap}(q)=\max_k\bigl(\beta q^\top z_k+\ell_k\bigr)-\bigl(\beta+\kappa\bigr);\qquad u^{\mathrm{out}}=\begin{cases}u^\ast,&\operatorname{gap}(q)>\ln\tau,\\ q,&\text{otherwise.}\end{cases}\;}$$

- $\tau\to0$ reproduces the raw map; $\tau\to\infty$ reproduces cue-only.
- Costs are asymmetric (abstaining on a present trial $\approx1-\cos(q,z)\approx0.25$; committing wrongly on an absent trial $\approx0.4$–$0.9$), so **argmax ($\tau=1$) is not generally optimal**.
- $\tau$ is calibrated on a fixed training batch over a log grid; a flat sweep ties to $\tau=1$. It is saved with the checkpoint and applied by the streaming path.
- The gate is a *decision*, not a step: the descent theorem is unaffected.

**The raw map is a diagnostic, not a deployable output.**

### 8.6 Legacy quadratic readout (retained for ablation)

$$E(u)=-\tfrac12u^\top Wu-\gamma u^\top q,\quad v_n=u_n+\eta(Wu_n+\gamma q),\quad u_{n+1}=v_n/\|v_n\|.$$

Descent for every $\eta>0$, $\|v_n\|\ge1-\chi>0$, and the exact $W/M$ reparameterization $\eta'=\eta/(1+\eta(1-S)/d)$ all remain **THEOREMS** unchanged from v2.0.4.

**Corrected angular limit (replaces v2.0.4 §6.4).** With $\mu=z_k^\top Mz_k$, $P=I-z_kz_k^\top$, $M_{\mathrm{oth}}=M-\alpha_kz_kz_k^\top$:

$$\boxed{\;H=(\mu+\gamma)P-PM_{\mathrm{oth}}P\ \text{on the tangent space};\qquad \text{stable}\iff H\succ0.\;}$$

If stable, the residual is $\|H^{-1}x_\perp\|$ with $x_\perp=PM_{\mathrm{oth}}z_k$; if not, **no floor exists** — the target is a saddle and the iterate leaves it.

**Factored $T\times T$ computation (verified against dense $d\times d$).** With $B=[\sqrt{\alpha_j}Pz_j]_{j\ne k}$, $G=B^\top B$, $v_j=\sqrt{\alpha_j}(z_j^\top z_k)$, $m=\mu+\gamma$:

$$\text{margin}=m-\lambda_{\max}(G),\quad \text{floor}_{\mathrm{spec}}=\frac{\sqrt{v^\top Gv}}{\alpha_k+\gamma},\quad \text{floor}_{H}=\sqrt{w^\top Gw},\ \ w=(mI-G)^{-1}v.$$

When the dense form is used, lift the $z$-mode to $m$ so the smallest eigenvalue is the tangent minimum **whether or not the target is stable**.

**Capacity–closure tradeoff (THEOREM for equal masses).** $\lambda_{\max}(PM_{\mathrm{oth}}P)\approx\alpha(1+\sqrt{(T-1)/d})^2$, so stability needs $\gamma\gtrsim2\alpha\sqrt{(T-1)/d}$, while converged closure is $\alpha/(\alpha+\gamma)$. Hence
$$\text{closure}_{\max}\approx\frac1{1+2\sqrt{(T-1)/d}}.$$

**MEASURED** ($d=384$, $T=20$, $\alpha=0.05$, $\gamma=0.01$): margin $-0.0123$ (saddle); v2.0.4 formula $0.170$ rad; converged angle $0.795$ rad. At $\gamma=0.15$: $\|H^{-1}x_\perp\|=0.071$ vs measured $0.070$ rad. **DESIGN RULE:** $\gamma=3(S/T)\sqrt{(T-1)/d}$ restores stability (margin $+0.0111$) at the cost of closure.

**This tradeoff is a property of the quadratic functional, not of the parameters.** It is the reason the core readout changed.

### 8.7 Diagnostics

Per retrieval: $E(u_n)$, $\|\Delta u_n\|$, $\|\nabla F_n\|$, guard count, posterior $p$, $\operatorname{gap}(q)$, gate decision. Globally: $\beta,\kappa$, doses, $S$, $\hat P$, half-life, $\theta_{\mathrm{basin}}$, $\theta_{\mathrm{floor}}$, hysteresis rate, capture rate.

**The energy-increase fraction is a bug detector and must be $0$ up to rounding.** A systematic increase indicates non-PSD state, a sign error, or a fallback event.

---

## 9. Training objective

### 9.1 Primary: posterior cross-entropy at $u=q$

$$\boxed{\;L_{\mathrm{CE}}=-\ln p_{k^\ast}(q),\qquad p(q)=\operatorname{softmax}\bigl([\beta Z^\top q+\ell,\ \beta+\kappa]\bigr),\;}$$

with $k^\ast$ the target index, or the **cue atom index $T$** for a no-match episode.

This is the von Mises–Fisher mixture likelihood with log-priors $\ell_i$. It is evaluated at $u=q$, **not** at $u^\ast$: iterating double-counts the cue evidence and biases $\beta$.

### 9.2 Why not the cosine of the map

Once the map commits, the softmax saturates and $\partial u^\ast/\partial\kappa\approx0$: the cosine term becomes **inert** while CE keeps pushing. **MEASURED:** with a mixed objective, the map's cosine loss drifted $0.054\to0.18$ during training while CE fell monotonically. Default: CE alone. The cosine loss is retained as an ablation.

For ablation use, $L=1-|c|$ with the Clarke subgradient $\operatorname{sign}(0):=+1$ and $c$ clamped to $[-1,1]$; the smooth surrogate $\sqrt{1+\epsilon_L}-\sqrt{c^2+\epsilon_L}$ remains available. For no-match episodes the cosine target is $q$ itself.

### 9.3 No-match episodes are required

With probability $p_{\mathrm{absent}}$ (default $0.2$) the cue refers to an item **not stored**. Without these, $\kappa\to-\infty$ ("never abstain") is a free optimum. **MEASURED:** cue mass drifted to $5\times10^{-6}$ and below in the absence of no-match trials.

---

## 10. Episodes and schedules

### 10.1 Episode construction

Reset to $W=J$, $S=0$, empty atom set. Per event: diffuse by $\Delta t_i$, write. Diffuse to $t_q$; retrieve at frozen state; gate; compute loss; backpropagate.

Vary $s_t$ and $\Delta t_t$ **independently**. Vary target position, cue completeness, recency, importance, competing inputs, and **cue ambiguity**. Use separate train/eval episodes and a **fixed** calibration batch for $\tau$.

### 10.2 Cue ambiguity is a data requirement

$\partial\mathrm{CE}/\partial\Gamma=\mathrm{age}_k-\mathbb E_p[\mathrm{age}]$ is zero when the posterior is one-hot. If $p_{\text{true}}\approx1$, the prior has no gradient and $\Gamma$ is not identified. **The reference implementation emits a warning at $p_{\text{true}}>0.99$.** Use same-passage competitors, wider cue noise, or larger $T$.

### 10.3 Schedules

**`staged`** (legacy-compatible): A = readout $\{\beta,\kappa\}$; B = all.

**`probe`** (recommended): the law and the readout are identified from different distributions, then the law is locked.

| Stage | Objective | Trainable | Purpose |
|---|---|---|---|
| **P** | cue-free CE ($\beta$ term removed) | $\{\omega,\Gamma,\kappa\}$ | identify the log-dose law from the **marginal** |
| **A** | cue CE | $\{\beta,\kappa\}$ | recalibrate the readout from the **conditional** |
| **B** | cue CE | $\{\bar\varepsilon,\beta,\kappa\}$ | refine; $\omega,\Gamma$ **frozen** |

**Stage P alone is a diagnostic, not a recipe.** Its optimum puts $\alpha\ll e^\kappa$ (it only has to match null odds), so retrieval collapses until A restores the balance. **MEASURED:** after P, $S\approx0.05$, basin $0.07$ rad, commit rate $0.000$; after A, basin $1.0$–$1.2$ rad and gate loss $0.025$–$0.054$.

Freezing $\omega$ in B matters: **MEASURED**, leaving it free overshot the per-event decay to $0.228$ against a data rate of $0.206$; freezing it gave $0.197$.

### 10.4 What is and is not identified

$\Gamma$ and $\beta$ are recoverable. $\omega$ is bounded above rather than pinned (flat direction with $\kappa$). $\bar\varepsilon$ is pure scale at small dose. Report the **split** $\omega\mathbb E[s]+\Gamma\mathbb E[\Delta t]$, not $\Gamma$ alone.

---

## 11. Optional reinforcement

Given $u^\ast$ (the **gated** output), $\mu=u^{\ast\top}Wu^\ast\in[0,1)$, retrieved-content importance $s_{\mathrm{ret}}$, confidence $C\in[0,1]$:

$$a_{\mathrm{ret}}=1-e^{-\omega s_{\mathrm{ret}}},\qquad \tau_{\mathrm{reinf}}=\tfrac1d+(1-\tfrac1d)a_{\mathrm{ret}},$$
$$\beta_{\mathrm{reinf}}=\begin{cases}C\min\bigl(\bar\beta,\frac{\tau_{\mathrm{reinf}}-\mu}{1-\mu}\bigr),&\mu<\tau_{\mathrm{reinf}},\\0,&\mu\ge\tau_{\mathrm{reinf}}.\end{cases}$$

Use the masked-denominator form: $(\tau-\mu)_+$ divided by $d_\mu=1-\mu$ if $\mu<\tau$ else $1$. Never form the raw fraction and select afterward. In log-atom form: shift all $\ell_j$ by $\ln(1-\beta)$, append $\ln\beta$. Purity is **not** monotone: $P'-P=2\beta(\mu-P)+\beta^2(1+P-2\mu)$. Log the reinforcement share of $S$.

---

## 12. Optional decoder

Features $r=[u^{\mathrm{out}},q,u^{\mathrm{out}}\odot q,E(u^\ast),\mu,S,\hat P,\text{gate flag},\text{guard flag}]$. **The decoder must consume the gated output**, not the raw map.

Normalization: max-floor $\hat z=D/\max(\|D\|,\epsilon_{\mathrm{dec}})$ or smooth $\hat z=D/\sqrt{\|D\|^2+\epsilon_{\mathrm{dec}}^2}$, $\epsilon_{\mathrm{dec}}=10^{-6}$. Jacobian bounds $1/\max(\|D\|,\epsilon_{\mathrm{dec}})$ and $1/\sqrt{\|D\|^2+\epsilon_{\mathrm{dec}}^2}$; both $\le1/\epsilon_{\mathrm{dec}}$. Weight decay is not a remedy for large gradients near zero output. The float32 naive-loss absorption band around $\alpha\approx2^{12}\epsilon_{\mathrm{dec}}$ and the stable form $\epsilon_{\mathrm{dec}}^2/(s(s+\alpha))$ are unchanged from v2.0.4 §12.3.

Evaluate retrieval separately; a strong decoder can hide weak memory.

---

## 13. Blending and vagueness

**Retained.** $u^\top Wu=\frac{1-S}{d}+\sum_i\alpha_i(z_i^\top u)^2$: overlapping traces jointly support blend directions. Diffusion leaves the **relative spectral shape of $M$ exactly invariant** — no spectral reshaping, no literal merging of directions.

**Replaced.** "Vagueness = gain loss relative to the cue, with half-closure time" is now readout-specific:

| | legacy quadratic | core Gibbs |
|---|---|---|
| mechanism | closure $\tilde m(t)/(\tilde m(t)+\gamma)$ fades smoothly | basin radius shrinks, then **vanishes** |
| threshold | $t_{1/2\mathrm{cl}}=\Gamma^{-1}\ln(\tilde m_0/\gamma)$ (50 % closure) | $t_{\mathrm{forget}}=\Gamma^{-1}(\ell_k(0)-\kappa)$ (basin collapse) |
| angular limit | $\alpha\sqrt{(T-1)/d}/(\alpha+\gamma)$, needs stability | $\approx(\bar\alpha/\alpha_k)e^{-\beta}\sqrt{T-1}$ |
| blending | forced (one attractor) | tunable via $\beta$ |

**What diffusion does not fix.** Eigenvectors are preserved and all gaps contract by the same factor, so a dominant shared-mean direction stays dominant *relative to everything else*. Under the legacy readout this is capture; the Gibbs readout suppresses it by exponential down-weighting rather than by integration. **MEASURED:** anisotropic-arm capture $1.00\to0.00$–$0.14$.

---

## 14. Guaranteed properties

Given unit $z_t,q$, finite $s_t\ge0$, finite $\Delta t\ge0$, and the parameterizations of §7, in exact arithmetic:

**State.** $W\succeq0$, $\operatorname{tr}W=1$, $W=(1-S)J+M$, $M\succeq0$, $S\in[0,1)$, $\lambda_{\min}(W)\ge(1-S)/d>0$; closed-form log-doses (§3.3); all diffusion laws (§5) including relative spectral invariance; steady states; dose additivity; exact impure-write decomposition; pruning bound and invariants (§4.4) when enabled.

**Core readout (guard-free).** $E(u_{n+1})\le E(u_n)$ for every $\beta>0$; weighted square-summability; empty memory returns $q$ exactly; permutation and log-dose-shift invariance; basin, floor, resolution and forgetting formulas of §8.4 as first-order/design statements.

**Legacy readout.** Descent for every $\eta$; $\|v_n\|\ge1-\chi>0$; exact $W/M$ reparameterization; corrected stability criterion and factored diagnostics (§8.6).

**Decision.** Gate limits: $\tau\to0$ = raw map, $\tau\to\infty$ = cue-only. Batch and streaming gated paths are identical.

**Not guaranteed.** Exact recall; unique basins or global optimality; which critical point is selected; a floor when the legacy margin is non-positive; angular resolution below $\sim1/\beta$; identification of $\Gamma$ from the conditional alone; a pinned $\omega$, $\bar\varepsilon$, or $S$; float32-safe margins when $\beta$ or $|\kappa|$ become extreme; monotone $S$ or purity.

---

## 15. Implementation rules

1. Normalize $z_t,q$; validate finiteness; clamp $s_t\ge0$.
2. Initialize $W=J$, $S=0$, empty atom set. Never $W=0$.
3. **Store log-atoms** $\{(z_i,\ell_i)\}$ plus $S$. Materialize $M$ or $W$ only for diagnostics or the legacy readout.
4. **Lazy diffusion clock:** $\ell_i\mathrel{-}=\Gamma\Delta t$, $S\mathrel{\ast}=e^{-\Gamma\Delta t}$. Never materialize $J$.
5. Enforce diffuse → write → reinforce ordering.
6. Doses: $a=-\operatorname{expm1}(-\omega s)$, $p=-\operatorname{expm1}(-\Gamma\Delta t)$; retention $\exp(-(\omega s+\Gamma\Delta t))$; stable softplus inverse (§7.3).
7. $\ell_i=\ln\bigl(\max(a_i(1-\varepsilon_i),10^{-30})\bigr)$ — finite at $s=0$, zero gradient through the clamped branch.
8. Masked denominators everywhere a branch may be non-finite (retrieval guard, reinforcement).
9. Off-flags for ablations: $\bar\varepsilon\equiv0$, diffusion bypass. $\sigma$ and softplus never reach $0$.
10. **The gate lives in the inference path.** `retrieve` must apply $\tau$ exactly as evaluation does. Save $\{\text{state},\tau,\text{readout},\text{config}\}$; the reported deploy metric must use the saved $\tau$.
11. **Calibrate $\tau$ once on a fixed training batch.** Do not resample per report; do not fit on eval.
12. Log $\beta,\kappa$, doses, $S$, $\hat P$, half-life, margin, $\|\nabla F\|$, gate decisions, hysteresis rate, and the per-event decay split. Energy-increase fraction must be $0$.
13. Run both calibration arms and report cue-only, centroid, and nearest-neighbour baselines **decomposed by present/absent** (§17.1).
14. Self-checks must include: factored-vs-dense corrected diagnostics (including the saddle case); the planar hysteresis witness; gate limits; streaming-vs-batch gated equality; the cue-free gradient restoration.

---

## 16. Complexity and storage

| Operation | Dense | Log-atom (core) |
|---|---:|---:|
| Store state | $O(d^2)$ | $O(Td+T)$ |
| Write | $O(d^2)$ | $O(T)$ logit shift $+$ append |
| Diffusion | $O(d^2)$ | $O(1)$ lazy, $O(T)$ applied |
| Posterior / matvec | $O(d^2)$ | $O(Td)$ |
| Retrieval ($N$ steps) | $O(Nd^2)$ | $O(NTd)$ |
| Gate decision | — | $O(Td)$ (one posterior at $q$) |
| Corrected stability | $O(d^3)$ | $O(T^2d+T^3)$ |
| Purity $\hat P$ | $O(d^2)$ | $O(T^2)$ with cached Gram |

**Storage grows with $T$.** This is the cost of §3.6. With pruning (§4.4) the effective capacity in steady state is bounded: since $\ell$ drifts at rate $\Gamma$, atoms older than $\bigl(2\beta+\ln(T/\epsilon_{\mathrm{prune}})\bigr)/\Gamma$ are droppable, giving

$$T_{\mathrm{eff}}\ \approx\ \frac{2\beta+\ln(T/\epsilon_{\mathrm{prune}})}{\Gamma\,\mathbb E[\Delta t]}.$$

Backprop: $\partial\ell_i/\partial\Gamma=-\mathrm{age}_i$, $\partial a_t/\partial\omega=s_te^{-\omega s_t}$; activation memory $O(Td+Nd)$ with recomputation.

---

## 17. Evaluation plan

### 17.1 Baselines, decomposed

Report **every** baseline on present trials and absent trials separately, alongside the gated and raw-map numbers:

0. cue-only ($=$ exact empty-memory output);
1. top-eigenvector / centroid;
2. nearest-neighbour over stored embeddings;
3. diffusion-off at matched init;
4. constant dose vs importance-dosed;
5. legacy quadratic readout, with and without the auto-$\gamma$ rule.

**Aggregate comparisons are invalid** when one side is decomposed: nearest-neighbour cannot abstain, so its aggregate carries a forced-commit penalty on absent trials. The claim "beats cue-only and centroid" must be made on present trials.

### 17.2 Retrieval experiments

Full and partial cues; absent/unrelated memory (must return $q$ exactly at $W=J$); hard neighbors; recency and importance buckets; **age at fixed importance and importance at fixed age**; irregular-$\Delta t$ generalization; retention vs $\ln2/\Gamma$; steady states at labeled instants; $\varphi$ sweeps below/near/above $\theta_{\mathrm{floor}}$ and $\theta_{\mathrm{basin}}$; $t_{\mathrm{forget}}$ vs measured basin collapse; **hysteresis rate**; gate-threshold sweep; sensitivity to init; unseen sequences.

### 17.3 Calibration before training

Two arms at init, $d=384$: **R** (random unit atoms) and **A** (shared-mean cluster or real embeddings). Report per arm: closure at $N$ and $10N$; legacy stability fraction and both floors; Gibbs basin and floor; capture rate; energy-increase counter; and all baselines decomposed.

If closure is negative at small $\varphi$ under the legacy readout, the cause is curvature, not gain — do not raise $\eta$. If Arm A captures under the legacy readout, do not expect diffusion to fix it.

### 17.4 Identifiability protocol

Run `probe`. Check: (i) stage P recovers the generating decay in the **split**; (ii) stage A restores the basin; (iii) stage B does not move the law; (iv) the deploy metric uses the saved $\tau$. If $p_{\text{true}}>0.99$, the prior is unidentifiable — change the data, not the optimizer.

### 17.5 Ablations

Forgetting on/off; dose map variants; $\bar\varepsilon\equiv0$ vs coupled; Gibbs vs legacy readout; $\beta$ sweep (floor and resolution); gate on/off and $\tau$ sweep; CE vs cosine vs mixed; `staged` vs `probe`; pruning on/off; embedding geometry raw vs centered.

---

## 18. Core mathematical summary

**State.** $W=(1-S)J+M\succeq0$, $\operatorname{tr}W=1$; log-atoms $\ell_i=\ln\alpha_i$ with
$$\ell_i=\ln\bigl(a_i(1-\varepsilon_i)\bigr)-\omega\textstyle\sum_{j>i}s_j-\Gamma(t_q-t_i).$$

**Diffuse (first at each event).** $\ell_i\mathrel{-}=\Gamma\Delta t$; $S\mathrel{\ast}=e^{-\Gamma\Delta t}$.

**Write.** $a_t=1-e^{-\omega s_t}$, $\varepsilon_t=\bar\varepsilon(1-a_t)$; $\ell_j\mathrel{-}=\omega s_t$ for $j<t$; append $\ln(a_t(1-\varepsilon_t))$.

**Retrieve.** $F(u)=\beta^{-1}\ln\bigl(e^{\beta q^\top u+\kappa}+\sum_i\alpha_ie^{\beta z_i^\top u}\bigr)$; $u_{n+1}=\nabla F(u_n)/\|\nabla F(u_n)\|$; $E=-F$ descends for every $\beta>0$.

**Decide.** $\operatorname{gap}(q)=\max_k(\beta q^\top z_k+\ell_k)-(\beta+\kappa)$; commit iff $\operatorname{gap}>\ln\tau$, else return $q$.

**Limits.** $\theta_{\mathrm{basin}}=\arccos(1-(\ell_k-\kappa)/\beta)$; $\theta_{\mathrm{floor}}\approx(\bar\alpha/\alpha_k)e^{-\beta}\sqrt{T-1}$; $t_{\mathrm{forget}}=(\ell_k(0)-\kappa)/\Gamma$.

**Train.** $L=-\ln p_{k^\ast}(q)$ with a cue-atom no-match class; `probe` = P(marginal law) → A(readout) → B(readout, law frozen); calibrate $\tau$ on a fixed training batch.

**Legacy.** $u\leftarrow\operatorname{normalize}(u+\eta(Wu+\gamma q))$; stable iff $(\mu+\gamma)P-PM_{\mathrm{oth}}P\succ0$; closure $\le1/(1+2\sqrt{(T-1)/d})$.

---

## 19. Measured results (reference implementation v3.6)

**Setup.** $d=384$, $T=20$, $N=20$, $p_{\mathrm{absent}}=0.2$, generating decay $\lambda=0.2$, batch $256$, calibration batch $1024$, `probe` schedule, CE loss, CPU, single seed. *Synthetic*: 10 000 clustered unit vectors ($\rho=0.3$), cues at $\varphi\sim U(0.4,1.3)$ rad. *Real*: 100 Gutenberg books → 73 923 sentences, `all-MiniLM-L6-v2`, same-passage competitors, cue = random contiguous half-span.

| | synthetic | real |
|---|---:|---:|
| **gate (deployed)** | **0.0081** | **0.0267** |
| raw map | 0.1744 | 0.1072 |
| cue-only | 0.2924 | 0.2388 |
| nearest-neighbour | 0.2132 | 0.1658 |
| centroid | 0.7206 | 0.4829 |
| posterior CE | 0.134 | 0.147 |
| argmax accuracy | 0.980 | 0.953 |
| commit / abstain | 0.985 / 0.968 | 0.970 / 0.898 |
| gate present / absent | 0.0037 / 0.0220 | 0.0177 / 0.0570 |
| hysteresis rate | 0.758 | 0.559 |
| $\Gamma$ vs $\lambda=0.2$ | 0.1840 | 0.1664 |
| decay split (dilution + time) | 0.0102 + 0.1870 = 0.1973 | 0.0104 + 0.1659 = 0.1763 |
| data rate | 0.2033 | 0.1994 |
| Gibbs floor (rad) | $\sim10^{-7}$ | $\sim8\times10^{-4}$ |
| $S$ | 0.05 | 0.06 |
| energy increases / guards | 0 / 0 | 0 / 0 |

26/26 self-checks pass, including the dense-vs-factored corrected diagnostics, the planar hysteresis witness, gate limits, and streaming-vs-batch gated equality.

**Known residual limitations.**

1. **$\Gamma$ undershoots by 12–17 %.** Cause: $\omega$ stalls near $0.010$, so write-dilution still supplies a noisy count proxy for age. The cost of that noise is $O(\omega^2\mathrm{Var})\approx10^{-3}$ nats, so the gradient driving $\omega\to0$ is weak. Retrieval is insensitive to this (gate $\approx0.027$ either way). **HYPOTHESIS:** larger batch and longer stage P drive $\omega\to0$ and $\Gamma\to\lambda$; if not, the residual bias lies in the $\kappa$/null-odds coupling.
2. **`prior CE` rises after stage P** (3.9 → 5.1). By design (§7.4.3).
3. **Aggregate baseline comparison is not like-for-like** (§17.1); the present-only decomposition is required before quoting fold-improvements over nearest-neighbour.

---

## 20. Central research hypothesis

HopMem tests whether a trace-one PSD state, whose every operation is one convex mixture and whose forgetting is a mandatory learnable-rate depolarizing channel, can support useful cue-conditioned semantic retrieval **when the readout is a Gibbs posterior over the induced atoms rather than a quadratic form**:

1. Retrieval beats cue-only, centroid, and nearest-neighbour **on present trials**, and abstains correctly on absent trials. *(MEASURED, §19; decomposition required.)*
2. The forgetting rate is recoverable from the **marginal** relevance law, and only there. *(MEASURED.)*
3. Vagueness is **basin collapse** at a computable time, not spectral reshaping. *(THEOREM + MEASURED.)*
4. Angular resolution is set by $\beta$ and the angular gap, with an exponentially small floor. *(THEOREM + MEASURED.)*
5. The decision must precede descent. *(THEOREM, deterministic witness.)*

**Conceptual status.** The density-matrix and channel language is a real-symmetric analogue and a mathematical convenience, not a claim of quantum physics. In v2.1.1 the state of record is still a density matrix, but the core readout is a **log-linear Gibbs model over atoms** — a modern-Hopfield/attention readout whose log-priors are produced by a convex, dose-additive, uniformly-decaying dynamics. The distinctive claims are: single convex primitive for the state; dose additivity as additive log-priors; provable descent for both readouts; sharp basin-collapse forgetting; decision-before-descent; and the marginal/conditional identifiability split. The costs are stated: atom-dependence (§3.6), growing storage (§16), a calibrated threshold outside the gradient loop (§7.2), and one scalar serving two calibration roles (§7.4.3).

---

## 21. Supersession: withdrawn claims

The following must not be used. Items 1–11 are inherited from v2.0.4's cleared list and remain cleared; items 12–22 are new in v2.1.1.

1. "Energy descent holds only for sufficiently small steps." → Theorem for every $\eta$ (legacy) and every $\beta$ (core).
2. "Crosstalk $\approx(S-\tilde\alpha)/d$ is the retrieval limit." → That is a support shift.
3. "Vagueness = spectral gap closure / merging," and arbitrary-tolerance $t_{\mathrm{merge}}$.
4. "Impure blur applies only to the injected target." → Exact decomposition (§6.2).
5. "$W$-form vs $M$-form differ by a bounded $h$ decrease." → Exact trajectory reparameterization.
6. "The $10^{-8}$ guard becomes reachable at margin $\sim10^{-4}$."
7. "$\mu$ reliably rounds to $1$ after 25 writes."
8. "Compute doses as $1-e^{-x}$." → Use $-\operatorname{expm1}(-x)$.
9. "The $g=2.5$ query-gap example applies under continued periodic writing."
10. "Gain over cue-only suffices." → Centroid and nearest-neighbour too, decomposed.
11. "Diffusion creates no new basins / no spurious minima." → Range of $M$ unchanged; critical-point counts not claimed.
12. **"§6.4's $\theta_{\mathrm{floor}}$ is a fixed-point property independent of $\eta,N$."** → It omits curvature; when the margin is non-positive there is no floor at all (§8.6).
13. **"The legacy quadratic readout is adequate for retrieval."** → One stable attractor; capacity–closure bound; 5–6 % stable at the v2.0.4 init.
14. **"Retrieval depends only on $(M,\gamma q)$."** → True for the legacy readout only; the core readout depends on the atom set (§3.6).
15. **"Storage is fixed-size $O(d^2)$."** → Core is $O(Td)$, growing; bounded only via pruning (§4.4, §16).
16. **"The Gibbs floor prefactor is $\sqrt{(T-1)/d}$."** → $\sqrt{T-1}$; the Gibbs force carries no $1/\sqrt d$ (§8.4).
17. **"The argmax of the posterior is the retrieval decision."** → Hysteresis; the null basin is strictly smaller (§8.5).
18. **"Vagueness is smooth gain loss with a half-closure time."** → Core: sharp basin collapse (§13).
19. **"Decorrelated $(s,\Delta t)$ suffices to identify $\Gamma$."** → Also needs the marginal objective and ambiguous cues (§7.4, §10.2).
20. **"One $\kappa$ can calibrate both marginal null-odds and the conditional commit threshold."** → §7.4.3.
21. **"$S$ indicates useful memory."** → Near-gauge; small $S$ compatible with excellent retrieval (§3.5).
22. **"$\tau$ may be fit on eval or resampled per report."** → Fixed training calibration batch, saved with the checkpoint (§7.2, §15.10–11).
23. **"Every operation is one convex primitive."** → Holds for the core; pruning (§4.4) is a specified, bounded exception.
24. **"The cosine loss of the map is a useful training signal."** → Inert once the map saturates (§9.2).

**Unchanged from the v2.0 line:** the primitive, dose maps, state parameterization, diffusion laws and steady states, impure-write decomposition, $W/M$ reparameterization, reinforcement branch, decoder normalization and float32 analysis, and the legacy defaults.

---

## 22. Changes from v2.0.4, and why

### 22.1 The four structural findings

**(a) The angular limit was mis-derived, and the failure was structural.**
v2.0.4 §6.4 divided a cross-talk *force* by a curvature term that omitted the cross-talk's own contribution to the Hessian. The correct object is $H=(\mu+\gamma)P-PM_{\mathrm{oth}}P$. Once written down, the consequence is unavoidable: a quadratic form on a sphere has exactly one stable attractor, and every other stored direction is a saddle unless the cue supplies enough curvature. For $T$ equal masses this needs $\gamma\gtrsim2\alpha\sqrt{(T-1)/d}$, while closure is $\alpha/(\alpha+\gamma)$ — capacity and closure fight each other. At the v2.0.4 init the target was a saddle in 94–95 % of draws, and the "measured closures" in the old spec were transients: the iterate kept walking away at $10N$. No parameter setting fixes this. **Change:** §6.4 deleted, §8.6 added with the corrected criterion, a verified factored $T\times T$ computation, and the tradeoff stated as a property of the functional.

**(b) So the functional changed, not the state.**
The state, primitive, diffusion, dose additivity and invariants were all sound and are retained verbatim. Dose additivity says the memory *is* a set of atoms with additive log-doses, which makes the log-sum-exp free energy the natural readout: log-doses become log-priors, forgetting becomes additive logit drift, and the cue becomes an explicit no-match atom. Because $F$ is convex, v2.0.4's own descent proof transfers verbatim — and improves, since there is no step size at all. **Change:** §8 rewritten; the legacy readout kept as an ablation with its theorems intact. **Cost, stated openly:** the core readout is no longer a function of $W$ (§3.6), and storage grows with $T$ (§16), bounded only by the specified pruning map, which is itself outside the single primitive (§4.4).

**(c) The decision must precede descent.**
With a no-match hypothesis available, the map's positive feedback becomes a liability: moving toward an atom raises its logit and lowers the cue's, so a marginal null win at $u=q$ runs away. This was proved with a two-dimensional witness and measured at 56–76 % of absent trials. Argmax gating is also not optimal, because the costs are asymmetric. **Change:** §8.5 added; the gate is part of the model, $\tau$ is a calibrated scalar saved with the checkpoint, and the streaming path must apply it identically. The raw map is demoted to a diagnostic.

**(d) Identifiability is a two-distribution problem.**
v2.0.4 warned that $\omega$ and $\Gamma$ need decorrelated $(s,\Delta t)$. That is necessary but not sufficient. The gradient $\partial\mathrm{CE}/\partial\Gamma=\mathrm{age}_k-\mathbb E_p[\mathrm{age}]$ vanishes when the cue already explains the answer — measured as a 90× reduction — and the model then buys its decay from write-dilution, a *count* proxy for age. $\Gamma$ is identified by the **marginal** law, $\beta,\kappa$ by the **conditional** law. **Change:** §7.4 rewritten, §10.3 adds the `probe` schedule (P → A → B with the law frozen), §10.2 makes cue ambiguity a data requirement, and §17.4 adds the recovery protocol. Two honest corollaries are recorded rather than hidden: $\kappa$ cannot serve both calibrations, and $(\beta,\kappa)$ share a flat ray.

### 22.2 Section-by-section map

| v2.0.4 | v2.1.1 | Change |
|---|---|---|
| §1 supports / non-goals | §1 | Adds abstention, pattern selection, sharp forgetting; removes "no pattern separation"; adds atom-dependence and growing storage as non-goals |
| §3 state | §3.1–3.5 | Retained; adds log-atom canonical form and closed form |
| — | §3.6 | **New:** atom-dependence theorem |
| §4 primitive | §4.1–4.3 | Retained; log-atom bookkeeping added |
| — | §4.4 | **New:** bounded pruning map, flagged as outside the primitive |
| §5 diffusion | §5 | Retained; log-domain statement added |
| §6.1–6.3 writing | §6.1–6.3 | Retained; $\bar\varepsilon$ as pure scale at small dose |
| §6.4 angular floor | — | **Deleted**, replaced by §8.4 / §8.6 |
| §7 parameters | §7 | $\{\eta,\chi\}\to\{\beta,\kappa\}$; $\tau$ calibrated; identifiability rewritten |
| §8 retrieval | §8.1–8.5 | **Replaced** by Gibbs free energy, descent, basin/floor/resolution/forgetting, hysteresis and gate |
| §8.3, §8.2 legacy results | §8.6 | Retained as theorems; corrected stability added |
| §9 loss | §9 | CE primary; cosine demoted with the saturation reason; no-match class required |
| §10 episodes | §10 | Adds cue-ambiguity requirement and `probe` schedule |
| §11 reinforcement | §11 | Retained; log-atom form; consumes gated output |
| §12 decoder | §12 | Retained; must consume gated output |
| §13 vagueness | §13 | Gain-loss → basin collapse; spectral invariance retained |
| §14 guarantees | §14 | Updated for both readouts and the gate |
| §15 implementation | §15 | Log-atoms, lazy logit clock, stable softplus inverse, gate in inference, fixed $\tau$ batch |
| §16 complexity | §16 | $O(Td)$ core; $T_{\mathrm{eff}}$ bound |
| §17 evaluation | §17 | Decomposed baselines, hysteresis metric, identifiability protocol, deploy contract |
| §19 hypothesis | §20 | Restated; conceptual status updated |
| §20 withdrawals | §21 | 11 inherited + 13 new |
| — | §19 | **New:** measured results with setup and residual limitations |

### 22.3 What was deliberately not changed

The state class, the convex primitive, the dose maps, the diffusion generator and all six of its exact laws, the impure-write decomposition, the $W/M$ reparameterization, the reinforcement branch and its masked denominator, the decoder normalization analysis, and the float32 guidance. These were correct in v2.0.4 and survived every check. The revision is confined to the readout, the angular-limit theory, the forgetting interpretation, and the training/identifiability protocol — plus honest accounting of what the readout change costs.