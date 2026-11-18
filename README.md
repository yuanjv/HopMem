# HopMem v2.2.1

**Dose-weighted associative memory with a Gibbs readout, a calibrated abstention gate, and a parameter-free fusion channel.**

This specification is self-contained. It supersedes v2.1.1. The state class, convex write primitive, dose maps, diffusion laws, Gibbs readout, descent theorem, decide-before-descend gate, and marginal/conditional identifiability split are retained. The revision adds one operation — **fusion** — which bounds the atom set by an architectural constant $C$, absorbs pruning as a special case, and makes capacity pressure a derived, monotone, auditable forgetting channel with **zero new free parameters**. A Transformer interface is specified in §23.

**Motivating measurement.** On the v2.1.1 checkpoint (real corpus, $d=384$, $T=20$), the log-dose law is near-optimal (cue-free prior CE gap $+0.0168$ nats vs the generating oracle) and the gated readout beats the like-for-like kNN+threshold baseline ($0.0196$ vs $0.0256$ aggregate; present $0.0139$ vs $0.0202$). The only failing object is the capacity map: top-$k$ truncation produces a present-loss cliff at $C=4$ ($0.1007$ vs $0.0139$ full) and an **absent-loss inversion** (random eviction $0.0000$ vs learned/oracle $0.0114$–$0.0167$). Both are properties of truncation, not of the law (§19.3, §22.1).

---

## 1. Scope

**Supports.** Cue-conditioned semantic retrieval with abstention; learned recency ($\Gamma$) and surprise ($\omega$, dose map) forgetting; sharp basin-collapse forgetting; a **strictly fixed-size state** of $C(d+2)+C^2$ floats with $O(1)$ read/write in stream length for fixed $C$; **gradual, pointwise-monotone, auditable information loss under capacity pressure** (Theorem 8.6-A); a latent interface consumable by a Transformer (§23).

**Non-goals.** Exact recall below the readout's resolution limit; order-independence of greedy fusion; a $\beta$-independent fusion ledger; monotonicity of the max-atom gate statistic under fusion (block posterior mass only, §8.6); identification of $\Gamma$ from fused states (stage P stays fusion-free, §10.4); Loewner-monotonicity of $W$ under fusion (the monotone object is the evidence functional, consistent with atom-dependence, §3.5).

**Conceptual status.** The density-matrix and channel language is a real-symmetric analogue and a mathematical convenience, not a claim of quantum physics. The state of record is a trace-one PSD matrix; the canonical representation is a bounded set of log-atoms; the core readout is a log-linear Gibbs model over those atoms — a modern-Hopfield/attention readout whose log-priors are produced by convex, dose-additive dynamics with three additive forgetting channels.

## 2. Notation and quickstart

| Symbol | Meaning |
|---|---|
| $d$ | embedding dimension; all $z_i, q, u$ are unit vectors |
| $C$ | architectural atom capacity (fixed, not learned) |
| $T$ | episode length / current atom count before compression |
| $N$ | descent steps (default 20) |
| $\alpha_i = e^{\ell_i}$ | dose (mass) of atom $i$; $\ell_i$ its log-dose |
| $S=\sum_i \alpha_i$ | stored mass, $S\in[0,1)$ |
| $J = I/d$ | depolarizing (uniform) state |
| $W=(1-S)J+M$, $M=\sum_i \alpha_i z_i z_i^\top$ | state; $\mathrm{tr}\,W=1$, $W\succeq 0$ |
| $\omega,\Gamma,\bar\varepsilon$ | shared law parameters (write dilution, time decay, impurity) |
| $\beta,\kappa$ | Gibbs readout (inverse temperature, null log-prior) |
| $\tau$ | gate threshold, calibrated once, saved with checkpoint |
| $\eta,\chi$, $\gamma=\chi/\eta$ | legacy quadratic readout (ablation only) |
| $G_{ij}=z_i\cdot z_j$ | cached Gram matrix |
| $A,w_i,m,r,\mu,\delta_{ij},D_c$ | fusion quantities (§4.5) |
| $\hat P=\mathrm{tr}(W^2)$ | purity |
| $E_i(u)=e^{\ell_i+\beta z_i\cdot u}$, $E_0(u)=e^{\kappa+\beta q\cdot u}$ | atom / null evidence |

Quickstart:

```
python hopmem.py selftest
python hopmem.py calibrate --read gibbs
python hopmem.py train --schedule probe --batch 256
python hopmem.py train --same_passage --schedule probe --batch 256
python hopmem_diag.py --ckpt hopmem_theta.pt          # decomposed baselines, prior CE, capacity sweep
```

Deployment:

```
p, tau, cfg = load_checkpoint("hopmem_theta.pt")
mem = HopMem(p, d=cfg.get("d", 384), C=cfg.get("C", 64), tau=tau)
mem.write(z, s, t)          # diffuse -> write -> fuse-on-overflow
u = mem.retrieve(q, t_q)    # gate at u=q, then descend; same rule as evaluate
```

## 3. State

### 3.1 Definition and invariants

$$W = (1-S)J + M,\qquad M=\sum_i \alpha_i z_i z_i^\top,\qquad S=\sum_i \alpha_i .$$

Every operation below preserves $W\succeq 0$, $\mathrm{tr}\,W=1$, $S\in[0,1)$, and $\lambda_{\min}(W)\ge (1-S)/d>0$ in exact arithmetic. Initialize $W=J$, $S=0$, empty atom set; never $W=0$.

### 3.2 Canonical log-atom form

Store $(z_i,\ell_i)$ with $\alpha_i=e^{\ell_i}$, plus $S$ and a lazy clock. Never materialize $M$ or $W$ except for diagnostics or the legacy readout. Between fusion events the closed form is

$$\ell_i = \ln\big(a_i(1-\varepsilon_i)\big) - \omega\sum_{j>i}s_j - \Gamma\,(t_q-t_i),\qquad a_i = 1-e^{-\omega s_i},\quad \varepsilon_i=\bar\varepsilon(1-a_i).$$

Forgetting is **additive in the log-dose**. With fusion (§4.5) a third additive term appears:

$$\boxed{\;\ell_c(t_q) = \mathrm{logaddexp}_{i\in c}\Big(\ln\big(a_i(1-\varepsilon_i)\big)-\omega\sum_{j>i}s_j\Big) \;-\; \Gamma\cdot\mathrm{age} \;-\; \beta D_c\;}$$

where $D_c=\sum(1-r)$ is the cumulative resultant deficit along the atom's fusion history and $\mathrm{logaddexp}(x,y)=\ln(e^x+e^y)$. The three channels are time ($\Gamma$), interference/dilution ($\omega$), and compression ($\beta D$). Theorem 8.6-B proves the uniform channels commute with fusion, so this ledger is exact for fixed $\beta$.

### 3.3 Dose additivity

Write-dilution shifts every existing log-dose by $-\omega s$; diffusion shifts every log-dose by $-\Gamma\Delta t$; fusion debits $\delta=\beta(1-r)$. All three are additive, so no sequential loop is needed for the uniform channels, and the fusion loop is $O(C^2+Cd)$ per write (§16).

### 3.4 Purity and the near-gauge of $S$

$\hat P=\mathrm{tr}(W^2)$. Under reinforcement (§11), $\hat P' - \hat P = 2\beta_{\mathrm{ret}}(\mu-\hat P)+\beta_{\mathrm{ret}}^2(1+\hat P-2\mu)$ with $\mu=u^\top W u$; purity is **not** monotone. $S$ is near-gauge: small $S$ is compatible with excellent retrieval (§19). Do not read $S$ as "useful memory."

### 3.5 Atom-dependence theorem

The Gibbs readout depends on the atom set $\{(z_i,\ell_i)\}$, not on $W$ alone: distinct atom sets can induce the same $M$ but different posteriors. This is the stated cost of the readout change from v2.0.4. The legacy quadratic readout is a function of $(M,\gamma q)$ only (§8.5). Fusion is defined on atoms, and its evidence accounting (§8.6) is atom-level; no claim is made that $W$ decreases in Loewner order under fusion.

## 4. Operation alphabet

The alphabet is {diffuse, write, reinforce, fuse, null-fuse}. Each has an exact law and preserves the invariants. Ordering per event: **diffuse → write → reinforce → fuse-on-overflow**. "Every operation is one convex primitive" remains withdrawn (§21, item 23): write and diffuse are convex mixtures; fusion is a coarse-graining plus a depolarizing step on a block, specified exactly below.

### 4.1 Diffuse

See §5 for the six laws. Lazy form: $\ell_i \mathrel{-}= \Gamma\Delta t$, $S \mathrel{*}= e^{-\Gamma\Delta t}$.

### 4.2 Write (convex primitive, retained)

Given unit $z$, importance $s\ge 0$:

$$a = 1-e^{-\omega s},\qquad \varepsilon=\bar\varepsilon(1-a),$$

$$\boxed{\;W' = (1-a)\,W + a\big[(1-\varepsilon)zz^\top + \varepsilon J\big]\;}$$

The bracket is trace-one PSD, so this is one convex mixture; invariants hold exactly. In log-atom form: $\ell_j \mathrel{-}= \omega s$ for all existing $j$ (since $1-a=e^{-\omega s}$), append $\ln(a(1-\varepsilon))$, and $S'=(1-a)S+a(1-\varepsilon)$.

**Impure-write decomposition (exact).** $a[(1-\varepsilon)zz^\top+\varepsilon J] = a(1-\varepsilon)zz^\top + a\varepsilon J$: the blur applies to the injected state, not only to the target direction. $\varepsilon=\bar\varepsilon(1-a)$ couples impurity to dose; at small dose $\bar\varepsilon$ is pure scale.

### 4.3 Reinforce

Optional; see §11. Appends an atom and counts toward capacity.

### 4.4 Null-fusion (pruning, absorbed; exact gate-dead criterion)

> **Gate-dead criterion.** Atom $i$ can never trigger a commit, for any unit cue $q$, iff
> $$\ell_i \le \kappa + \ln\tau .$$
> *Proof.* $\max_q\,\mathrm{gap}_i(q)=\max_q[\beta q\cdot z_i+\ell_i-(\beta+\kappa)]=\ell_i-\kappa$. Commit requires $\mathrm{gap}>\ln\tau$. ∎

**Decision invariance.** Deleting a gate-dead atom leaves the gate decision unchanged for every $q$: if $\mathrm{gap}>\ln\tau$ the maximizer was another atom (a gate-dead atom contributes at most $\ln\tau$); otherwise removing it can only lower the gap. The descended $u$ is perturbed by at most the atom's floor contribution $\|(p_i/p_k)P_k z_i\|$ (§8.3), bounded and measured $\sim 10^{-3}$ rad on the real corpus.

Null-fusion routes the atom's mass to $(1-S)J$: $S'=S-\alpha_i$, $M'=M-\alpha_i z_i z_i^\top$. The implied age bound $t_{\mathrm{dead}}=(\ell_i(0)-\kappa-\ln\tau)/\Gamma$ is sharper than v2.1.1's $(2\beta+\ln(T/\epsilon_{\mathrm{prune}}))/\Gamma$. The "outside the primitive" flag is removed; null-fusion is a specified channel. It is always preferred over lossy fusion when available.

### 4.5 Fusion channel (new; the capacity policy)

**Trigger.** After write (and optional reinforce), if the atom count exceeds $C$.

**Policy, in order.**
1. Null-fuse any gate-dead atom (§4.4). Decision-free.
2. Otherwise fuse the pair minimizing the **Ward cost**
$$c_{ij} = \frac{\alpha_i\alpha_j}{\alpha_i+\alpha_j}\,(1-G_{ij}),$$
computed from the cached Gram. Pairs with resultant $r<\epsilon_r$ are masked (cost $+\infty$). At $C=1$ with no valid pair, null-fuse the lower-dose atom and record the full evidence loss in the ledger.

**The fusion map.** With $A=\alpha_i+\alpha_j$, $w_i=\alpha_i/A$, $w_j=\alpha_j/A$:

$$m=w_i z_i+w_j z_j,\qquad r=\|m\|\in(0,1],\qquad \mu=m/r,$$

$$\boxed{\;\ell' = \mathrm{logaddexp}(\ell_i,\ell_j) - \delta_{ij},\qquad \delta_{ij}=\beta(1-r)\;}$$

The debit $\delta$ is the von Mises–Fisher resultant-length deficit. It is **not a parameter**; it is forced by Theorem 8.6-A.

**Stable forms.** From the Gram,

$$1-r^2 = 2w_i w_j(1-G_{ij}),\qquad \delta_{ij}=\frac{2\beta\,w_i w_j(1-G_{ij})}{1+r},$$

cancellation-free near $r=1$ and manifestly quadratic in angle. Never form $1-r$ by subtraction near $r=1$.

**State accounting.** $\alpha'=A e^{-\delta}$; $S'=S-A(1-e^{-\delta})$; the deficit moves to $(1-S)J$; $M'=M-\alpha_i z_i z_i^\top-\alpha_j z_j z_j^\top+\alpha'\mu\mu^\top$. Then $W'\succeq 0$, $\mathrm{tr}\,W'=1$ exactly. $S$ is **not** conserved (withdrawn claim 30); $W'\preceq W$ is **not** claimed.

**Ledger.** Store per-atom cumulative deficit $D_c=\sum(1-r)$; the current debit is $\beta D_c$. The episode ledger $\sum\delta$ is the auditable **nats of evidence forgotten by compression**. Per-target evidence retained is $e^{-\sum\delta_{\mathrm{target}}}$.

**Emergent recency/surprise protection.** $c_{ij}$ scales with $\alpha_i\alpha_j/A$, so high-dose atoms (recent, surprising) are expensive to fuse and are automatically protected; old low-dose atoms fuse cheaply. No separate recency or surprise heuristic exists — both enter only through the learned law.

**Edge cases.** $s=0\Rightarrow \ell=\ln(10^{-30})$: gate-dead, deleted free. $r\to 0$ (antipodal, equal mass): masked; fallback above. $C\ge T$: fusion never fires; v2.2.1 $\equiv$ v2.1.1 exactly (self-check 27). Fusion of an already-fused atom uses current debited masses and current direction; the ledger accumulates.

## 5. Diffusion: six exact laws

Generator: uniform log-drift at rate $\Gamma$. Over $\Delta t$:

- **D1 (semigroup).** $W(t+\Delta)=e^{-\Gamma\Delta}W(t)+(1-e^{-\Gamma\Delta})J$. Intervals compose additively; closed form, no sequential loop.
- **D2 (log-drift).** $\ell_i(t+\Delta)=\ell_i(t)-\Gamma\Delta$, uniform and additive.
- **D3 (mass).** $S(t+\Delta)=e^{-\Gamma\Delta}S(t)$; $S<1$ preserved.
- **D4 (spectral invariance).** Eigenvectors of $M$ are preserved; every eigenvalue scales by $e^{-\Gamma\Delta}$; relative spectral shape and $\mathrm{range}(M)$ are invariant. Diffusion does **not** merge directions.
- **D5 (commutation).** Diffusion commutes with write-dilution (both uniform log-shifts) and with fusion (Theorem 8.6-B). The diffuse→write ordering is a convention; the uniform ledger is order-independent.
- **D6 (forgetting times).** Gibbs basin collapse $t_{\mathrm{forget}}=(\ell_k(0)-\beta D_k-\kappa)/\Gamma$; gate-dead time $t_{\mathrm{dead}}=(\ell_k(0)-\beta D_k-\kappa-\ln\tau)/\Gamma$; legacy half-closure $t_{1/2}=\Gamma^{-1}\ln(\tilde m_0/\gamma)$.

**Steady state.** Under periodic writes of equal dose $a$ at interval $\Delta$, diffuse-then-write:

$$S^* = \frac{a(1-\varepsilon)}{1-(1-a)e^{-\Gamma\Delta}} .$$

**Decay split.** Report $\omega\,\mathbb{E}[s]+\Gamma\,\mathbb{E}[\Delta t]$, not $\Gamma$ alone (§19).

## 6. Writing details

Dose map $a=1-e^{-\omega s}$ computed as $-\mathrm{expm1}(-\omega s)$ for float stability; clamp $s\ge 0$; $\ell=\ln(\max(a(1-\varepsilon),10^{-30}))$ is finite at $s=0$ with zero gradient through the clamped branch. $\bar\varepsilon\equiv 0$ is the pure-write ablation flag. The impure decomposition (§4.2) is exact; no "target-only blur" claim.

## 7. Parameters

**Shared law:** $\omega>0$, $\Gamma>0$, $\bar\varepsilon\in(0,1)$. **Gibbs readout:** $\beta>0$, $\kappa\in\mathbb{R}$. **Legacy:** $\eta>0$, $\chi\in(0,1)$, $\gamma=\chi/\eta$. **Gate:** $\tau>0$, calibrated once (§8.4). **Capacity:** $C$ architectural, not learned. Fusion adds no parameters: $\delta$ uses learned $\beta$; Ward uses $\alpha$ (frozen law) and data Gram; null-fusion uses $\kappa,\tau$.

**Parameterization.** Positive scalars via softplus; $\bar\varepsilon,\chi$ via sigmoid; $\kappa$ raw. Stable inverse: $\mathrm{invsoftplus}(y)=\ln(e^y-1)$ via $\mathrm{expm1}$ for $y<20$, else $y+\ln(1-e^{-y})$; $\mathrm{logit}(p)=\ln(p/(1-p))$. Defaults: $\omega=-\ln(0.95)$, $\Gamma=10^{-3}$, $\bar\varepsilon=0.01$, $\beta=16$, $\kappa=\ln(10^{-4})$, $\eta=1$, $\chi=0.01$.

**Identifiability (measured).** $\Gamma$ is identified by the **marginal** relevance law (cue-free CE); $\beta,\kappa$ by the **conditional** (cue) law. Fitting both on the conditional alone leaves $\Gamma$ unidentified: $|\partial\mathrm{CE}/\partial\Gamma_{\mathrm{logit}}|$ drops from $7\times10^{-3}$ to $2\times10^{-4}$, and the required decay gets supplied by write-dilution (a count proxy for age) instead of time. The gradient is

$$\frac{\partial\,\mathrm{CE}}{\partial\Gamma}=\mathrm{age}_k-\mathbb{E}_p[\mathrm{age}],$$

which vanishes when the posterior is one-hot; ambiguous cues are a data requirement (§10.2).

**Flat ray and two-job $\kappa$.** $(\beta,\kappa)$ share a flat ray: only $(\ell-\kappa)/\beta$ is pinned by the conditional; $\beta$ grows because near-orthogonal atoms give an enormous true cue likelihood ratio. $\kappa$ serves marginal null-odds in stage P and the conditional commit threshold in A/B; one scalar cannot do both, so prior CE rising after P is expected (§19). $\omega$ is bounded above rather than pinned (flat direction with $\kappa$); $\bar\varepsilon$ is pure scale at small dose.

**$\beta$-gauge caveat (honest cost).** $\delta=\beta(1-r)$ and fusion weights use debited masses, so the ledger does not commute with $\beta$ updates. During training, fusion is recomputed in the forward pass (consistency automatic); at deployment $\beta$ is frozen. Diagnostic ledgers are reported at the checkpoint $\beta$.

## 8. Readout

### 8.1 Gibbs free energy

$$\boxed{\;F(u)=\frac{1}{\beta}\ln\Big(e^{\beta q\cdot u+\kappa}+\sum_i \alpha_i e^{\beta z_i\cdot u}\Big)\;}$$

The cue is an explicit **no-match atom** with log-prior $\kappa$. At $u=q$ the posterior is the vMF-mixture likelihood

$$p(q)=\mathrm{softmax}\big([\beta Z^\top q+\ell,\;\beta+\kappa]\big),$$

with the last index the null class. Log-doses are log-priors; forgetting is additive logit drift.

### 8.2 Descent theorem (retained, guard-free in exact arithmetic)

$F$ is convex (log-sum-exp of affine functions). With $p=\mathrm{softmax}([\beta Z^\top u+\ell,\;\beta q\cdot u+\kappa])$,

$$\nabla F(u)=p_q\,q+\sum_i p_i z_i,\qquad u^+=\frac{\nabla F(u)}{\|\nabla F(u)\|}.$$

> **Theorem.** $F(u^+)\ge F(u)$ for every $\beta>0$, equality iff $u^+=u$; no step size.
> *Proof.* $u^+$ maximizes the linearization $v\mapsto\nabla F(u)\cdot v$ over the unit sphere (Cauchy–Schwarz). Convexity gives $F(u^+)\ge F(u)+\nabla F(u)\cdot(u^+-u)$, and $\nabla F(u)\cdot u^+=\|\nabla F(u)\|\ge\nabla F(u)\cdot u$ since $u$ is unit. ∎

$E=-F$ descends; the energy-increase fraction is a bug detector and must be 0 up to rounding. A float guard ($\|v\|<\epsilon_{\mathrm{fall}}$ or non-finite $\Rightarrow u\leftarrow q$) is retained with masked denominators; measured guard events: 0. Empty memory returns $q$ exactly. The map is permutation- and log-dose-shift-invariant.

### 8.3 Geometry: basin, floor, resolution, forgetting

- **Basin (conservative).** Atom $k$ beats the cue at $u=q$ iff $\beta(1-q\cdot z_k)<\ell_k-\kappa$, so
$$\theta_{\mathrm{basin}}=\arccos\Big(1-\frac{\ell_k-\kappa}{\beta}\Big).$$
Descent toward an atom raises its logit and lowers the cue's, so the true basin is larger.
- **Floor.** First-order residual angle at the atom-$k$ fixed point:
$$\theta_{\mathrm{floor}}\approx\Big\|\sum_{i\ne k}\frac{p_i}{p_k}P_k z_i+\frac{p_0}{p_k}P_k q\Big\|\sim\frac{\bar\alpha}{\alpha_k}e^{-\beta}\sqrt{T-1},\qquad P_k=I-z_k z_k^\top.$$
The Gibbs force carries **no** $1/d$ (unlike the quadratic force, whose $c_j\sim 1/\sqrt d$).
- **Resolution (design statement).** Atoms separated iff $\beta(1-z_1\cdot z_2)\gtrsim|\ell_1-\ell_2|$.
- **Forgetting.** $\ell_k$ drifts down linearly; the basin vanishes at $t_{\mathrm{forget}}=(\ell_k(0)-\beta D_k-\kappa)/\Gamma$ — sharp, not a half-life. Each fusion decreases the basin parameter $(\ell_k-\kappa)/\beta$ by exactly $\delta/\beta$; $\theta_{\mathrm{basin}}$ follows via $\arccos$.

### 8.4 Decision rule: decide before descend; $\tau$ protocol

The map's null basin is **strictly smaller** than the one-shot argmax region: moving toward an atom raises its logit and lowers the cue's, so a marginal null win at $u=q$ runs away.

> **Deterministic witness (self-check).** $\beta=16$, $\ell=-3$, $\kappa=-6.5$, one atom at $0.70$ rad from $q$: atom logit $16\cos(0.70)-3=9.237$, null logit $16-6.5=9.500$. Null wins at $u=q$; the map still commits. Measured hysteresis rate: $0.758$ synthetic / $0.559$ real.

Therefore the system **decides at $u=q$** and only then descends:

$$\boxed{\;\mathrm{gap}(q)=\max_k\big(\beta q\cdot z_k+\ell_k\big)-(\beta+\kappa);\qquad \text{commit iff }\mathrm{gap}(q)>\ln\tau,\ \text{else return }q.\;}$$

Costs are asymmetric (present abstention $\sim0.25$, absent commitment $\sim0.4$–$0.9$), so argmax ($\tau=1$) is not always optimal. $\tau$ is one scalar, calibrated **once** on a fixed training calibration batch, reused for every report, saved with the checkpoint, and applied identically by `evaluate` and streaming `retrieve`. Limits: $\tau\to0$ = raw map; $\tau\to\infty$ = cue-only. Decision grid: $\tau_i=e^{-6+0.4i}$, $i=0,\dots,18$; index 15 is exactly $\tau=1$. If the sweep is flat (map $\equiv$ cue), report argmax, not the grid edge. Never fit $\tau$ on eval or resample per report.

### 8.5 Legacy quadratic readout (ablation only; corrected theory retained)

Energy $E(u)=-\tfrac12 u^\top M u-\gamma u^\top q$; update $u^+=\mathrm{normalize}(u+\eta(Mu+\gamma q))$; descent for every $\eta$. A quadratic on a sphere has **one stable attractor**; any other stored direction is a saddle unless cue curvature beats cross-talk curvature. With $\mu=z_k^\top M z_k$, $P_k=I-z_k z_k^\top$, $M_{\mathrm{oth}}=M-\alpha_k z_k z_k^\top$:

$$H_k=(\mu+\gamma)P_k-P_k M_{\mathrm{oth}}P_k\succ 0\quad\text{on the tangent space.}$$

For $T$ equal masses, $\lambda_{\max}(P_k M_{\mathrm{oth}}P_k)\approx\alpha(1+(T-1)/d)^2$, so stability needs $\gamma\gtrsim 2\alpha\sqrt{(T-1)/d}$ while converged closure is $\alpha/(\alpha+\gamma)$; hence

$$\mathrm{closure}_{\max}\approx\frac{1}{1+2\sqrt{(T-1)/d}} .$$

**Capacity fights closure; no tuning fixes it.** This is a property of the functional, and the reason the core readout is Gibbs. Design rule: $\gamma=3(S/T)\sqrt{(T-1)/d}$ restores stability at the cost of closure. Factored $T\times T$ computation (verified against dense $d\times d$): $B=[\sqrt{\alpha_j}P_k z_j]_{j\ne k}$, $G=B^\top B$, $v_j=\sqrt{\alpha_j}(z_j\cdot z_k)$, $m=\mu+\gamma$; margin $=m-\lambda_{\max}(G)$; $\mathrm{floor}_{\mathrm{spec}}=\sqrt{v^\top G v}/(\alpha_k+\gamma)$; $\mathrm{floor}_{\mathrm{hess}}=\sqrt{w^\top G w}$ with $w=(mI-G)^{-1}v$. When margin $\le0$ there is **no floor** (saddle). Measured ($d=384,T=20,\alpha=0.05,\gamma=0.01$): margin $-0.0123$; at $\gamma=0.15$, $\|H^{-1}x_\perp\|=0.071$ vs measured $0.070$ rad.

### 8.6 Compression theorems (new)

**Theorem A (evidence domination — gradual forgetting).** For every unit $u$,

$$e^{\ell'+\beta\mu\cdot u}\;\le\;e^{\ell_i+\beta z_i\cdot u}+e^{\ell_j+\beta z_j\cdot u}.$$

*Proof.* $e^{\ell'+\beta\mu\cdot u}=A e^{\beta(\mu\cdot u-(1-r))}\le A e^{\beta r\,\mu\cdot u}=A e^{\beta m\cdot u}\le \alpha_i e^{\beta z_i\cdot u}+\alpha_j e^{\beta z_j\cdot u}$. First step: $(1-r)(\mu\cdot u-1)\le0$ since $r\le1$ and $\mu\cdot u\le1$. Second: Jensen on the convex exponential with $m=w_i z_i+w_j z_j$. ∎

**Corollaries.**
1. $E_{\mathrm{mem}}(u)=\sum_c \alpha_c e^{\beta\mu_c\cdot u}$ is pointwise non-increasing under fusion and null-fusion; $F(u)$ is monotone under compression for every $u$. Writes are the only evidence-increasing operation; diffusion is uniform decay.
2. For any fused block, posterior mass $x/(E_0+E_{\mathrm{rest}}+x)$ is non-increasing in $x$; hence block commit probability never increases, and null posterior mass is non-decreasing. **Compression can never fabricate confidence.**
3. This is the mathematical realization of the project goal: recency and surprise set the doses; capacity pressure triggers a derived, monotone, ledgered contraction.

**Theorem B (dose additivity extends; commutation).** $\mathrm{logaddexp}(x+c,y+c)=\mathrm{logaddexp}(x,y)+c$, and $r$ (hence $\delta$) is invariant under uniform log-shifts because the weights $w$ are unchanged. Therefore $\Gamma$-decay and $\omega$-dilution commute with fusion exactly, and the closed form of §3.2 holds. The three forgetting channels are independent and additive.

**Proposition C (Ward = expected marginal angular distortion).** Under the cue-free target law $p_i\propto\alpha_i$, the expected first-order angular damage of fusing $i,j$ is

$$\mathbb{E}[\mathrm{damage}]=\frac{\alpha_i\alpha_j}{A\,S}(1-G_{ij})+O\big((1-G_{ij})^2\big).$$

*Proof sketch.* $z_i\cdot\mu=(w_i+w_j G)/r$ with $r^2=1-2w_iw_j(1-G)$; expanding to first order in $1-G$ gives $1-z_i\cdot\mu=w_j^2(1-G)+O((1-G)^2)$. Weighting by $\alpha_i/S$ and symmetrizing: $\alpha_i w_j^2+\alpha_j w_i^2=\alpha_i\alpha_j/A$. ∎

Hence min-Ward greedy minimizes expected marginal angular distortion. The exact per-query evidence loss $\Delta(u)=\ln(E_i+E_j)-\ln E'\ge0$ is closed-form; its small-angle limit is $\delta\approx\beta w_i w_j(1-G)$. Ward and $\delta$ differ by the factor $2\beta/(A(1+r))$: Ward weights heavy pairs (correct for expected task loss); $\delta$ is the evidence ledger. Both are parameter-free.

**Design rule (resolution quotient).** Prefer pairs below the readout's own resolution limit $\beta(1-G_{ij})\lesssim|\ell_i-\ell_j|$: the memory forgets precisely what its own readout could not distinguish at temperature $\beta$. When no pair is below the limit, fusion is genuinely lossy and the ledger records it.

**Caveat (max-gap non-monotonicity).** Two equal constituent logits can aggregate to $+\ln 2$ in the max-atom statistic; block mass, not the max gap, is the monotone object. $\tau$ is recalibrated on fused states (§10.4), as required for any state change.

**MEASURED (unit-level; $d=64$, $\beta=21.31$, 20 000 random unit queries).** Domination violations: 0 (max residual $-2.4\times10^{-5}$; bound tight near $u=\mu$). Commutation residual $\le 9\times10^{-16}$. $\delta$ vs $\beta w_iw_j(1-\cos)$: $0.2183$ vs $0.2174$ at $\cos 0.95$; quadratic growth confirmed ($\cos 0.99\to0.0405$; $0.50\to2.5082$). Stable-form identity verified: $\delta=4.2084=2\beta w_iw_j(1-G)/(1+r)$ at $G=0.1279$, $r=0.8025$. End-to-end fusion-regime results: **TO MEASURE** (§17.6, §19.4).

## 9. Training objective

**Primary: posterior cross-entropy at $u=q$.**

$$\boxed{\;L_{\mathrm{CE}}=-\ln p_{k^*}(q),\qquad p(q)=\mathrm{softmax}\big([\beta Z^\top q+\ell,\;\beta+\kappa]\big)\;}$$

$k^*$ is the target index, or the null index for a no-match episode. Evaluated at $u=q$, **not** at $u^*$: iterating double-counts cue evidence and biases $\beta$. Under fusion, labels map to the target's fused block (the achievable prediction); the deployed angular loss is still measured against the original $z_{\mathrm{target}}$.

**Why not cosine of the map.** Once the map commits, the softmax saturates and $\partial u^*/\partial\kappa\approx0$: the cosine term is inert while CE keeps pushing. Measured: under a mixed objective the map's cosine loss drifted $0.054\to0.18$ while CE fell monotonically. Default: CE alone. Cosine retained as ablation: $L=1-|c|$ with Clarke subgradient $\mathrm{sign}(0):=+1$, $c$ clamped to $[-1,1]$; smooth surrogate $1+\epsilon_L-\sqrt{c^2+\epsilon_L^2}$. For no-match episodes the cosine target is $q$.

**No-match episodes are required.** With probability $p_{\mathrm{absent}}$ (default $0.2$) the cue refers to an item not stored; otherwise $\kappa\to-\infty$ ("never abstain") is a free optimum. Measured: cue mass drifted to $5\times10^{-6}$ and below without them.

**Fusion in the loop (new).** Episodes simulate writes + fusion at capacity $C$ (sampled per episode, e.g. $C\sim U\{C_{\min},\dots,T\}$); CE is evaluated on the fused state. Fusion is differentiable ($\mathrm{logaddexp}$, normalize, Gram); with the law frozen, pair selection is parameter-free and $\beta$ gradients flow through $\delta=\beta(1-r)$, $\partial\delta/\partial\beta=1-r$. Hard argmin selection with straight-through is canonical; a soft-fusion temperature is an ablation, not a deployment knob.

## 10. Episodes and schedules

### 10.1 Construction

Reset to $W=J$, $S=0$, empty atoms. Per event: diffuse by $\Delta t_i$, write, fuse-on-overflow. Diffuse to $t_q$; retrieve at frozen state; gate; loss; backprop. Sampler: $s_i\sim\mathrm{Exp}(1)$ i.i.d.; inter-arrivals $\sim\mathrm{Exp}(1)$, $t_i$ cumulative; $t_q=t_T+\mathrm{Exp}(1)$; target $\sim\mathrm{softmax}(\ln s-\lambda(t_q-t))$ with $\lambda=0.2$; with prob $p_{\mathrm{absent}}=0.2$ the cue is a random other item's cue, label = null, $z_{\mathrm{target}}=q$. Optional `same_passage`: $T$ consecutive items from one document. Cue = random contiguous half-span of the sentence (`cue_frac` $0.5$), embedded by the same model. Vary $s$ and $\Delta t$ **independently**; vary target position, cue completeness, recency, importance, competitors, and cue ambiguity. Separate train/eval episodes; fixed calibration batch for $\tau$.

### 10.2 Cue ambiguity is a data requirement

$\partial\mathrm{CE}/\partial\Gamma=\mathrm{age}_k-\mathbb{E}_p[\mathrm{age}]$ is zero when the posterior is one-hot. If $p_{\mathrm{true}}\approx1$ the prior has no gradient and $\Gamma$ is unidentified. The reference implementation warns at $p_{\mathrm{true}}>0.99$. Use same-passage competitors, wider cue noise, or larger $T$. Decorrelated $(s,\Delta t)$ is necessary but not sufficient.

### 10.3 Schedules

`staged` (legacy-compatible): A = readout $\beta,\kappa$; B = all. `probe` (recommended):

| Stage | Objective | Trainable | Fusion | Purpose |
|---|---|---|---|---|
| P | cue-free CE ($\beta$ term removed) | $\omega,\Gamma,\kappa$ | **off** | identify the law from the marginal |
| A | cue CE | $\beta,\kappa$ | on, $C$ randomized | recalibrate readout on fused states |
| B | cue CE | $\bar\varepsilon,\beta,\kappa$ | on | refine; $\omega,\Gamma$ frozen |

### 10.4 Honest corollaries

**P must stay fusion-free:** the debit is age-correlated through the policy and would contaminate $\Gamma$ if fitted jointly. The law is identified in the uncompressed regime and frozen; fusion adds no parameters to it. **Stage P alone is a diagnostic, not a recipe:** its optimum puts $\alpha\ll e^\kappa$, so retrieval collapses until A restores the balance. Measured: after P, $S\approx0.05$, basin $0.07$ rad, commit $0.000$; after A, basin $1.0$–$1.2$ rad, gate $0.025$–$0.054$. Freezing $\omega$ in B matters: leaving it free overshot per-event decay to $0.228$ vs data $0.206$; frozen gave $0.197$. $\tau$ is recalibrated once on a fixed **fused** training batch, saved, and applied identically in batch and streaming paths.

## 11. Reinforcement (optional, retained)

Given the **gated** output $u^*$, retrieved-content importance $s_{\mathrm{ret}}$, confidence $\mathrm{cf}\in[0,1]$, cap $\bar\beta_{\mathrm{ret}}$:

$$\mu=u^{*\top}Wu^*,\qquad a_{\mathrm{ret}}=1-e^{-\omega s_{\mathrm{ret}}},\qquad \tau_{\mathrm{ret}}=\frac1d+\Big(1-\frac1d\Big)a_{\mathrm{ret}},$$

$$\beta_{\mathrm{ret}}=\begin{cases}\mathrm{cf}\cdot\min\Big(\bar\beta_{\mathrm{ret}},\ \dfrac{\tau_{\mathrm{ret}}-\mu}{1-\mu}\Big),&\mu<\tau_{\mathrm{ret}},\\[4pt]0,&\mu\ge\tau_{\mathrm{ret}}.\end{cases}$$

**Masked denominator:** use $(\tau-\mu)_+$ divided by $1-\mu$ when $\mu<\tau$, else denominator 1; never form the raw fraction and select afterward. Log-atom form: shift all $\ell_j$ by $\ln(1-\beta_{\mathrm{ret}})$, append $\ln\beta_{\mathrm{ret}}$; $S'=(1-\beta_{\mathrm{ret}})S+\beta_{\mathrm{ret}}$. Reinforcement counts toward capacity; fuse-on-overflow follows. Purity is not monotone (§3.4); log the reinforcement share of $S$. (Confidence is renamed $\mathrm{cf}$ from v2.1.1's $C$ to avoid collision with capacity $C$.)

## 12. Decoder (optional, retained)

Features $r=[u_{\mathrm{out}},\,q,\,u_{\mathrm{out}}\odot q,\,E(u^*),\,\mu,\,S,\,\hat P,\,\mathrm{gate},\,\mathrm{guard}]$. The decoder **must consume the gated output**, not the raw map. Normalization: $\hat D=D/\max(\|D\|,\epsilon_{\mathrm{dec}})$ or smooth $\hat D=D/\sqrt{\|D\|^2+\epsilon_{\mathrm{dec}}^2}$, $\epsilon_{\mathrm{dec}}=10^{-6}$. Jacobian bounds $1/\max(\|D\|,\epsilon_{\mathrm{dec}})$ and $1/\sqrt{\|D\|^2+\epsilon_{\mathrm{dec}}^2}$, both $\le1/\epsilon_{\mathrm{dec}}$; weight decay is not a remedy for large gradients near zero output. The float32 naive-loss absorption band and the stable form $\epsilon_{\mathrm{dec}}^2/(s(s+\alpha))$ are retained. Evaluate retrieval separately: a strong decoder can hide weak memory.

## 13. Blending and vagueness

$u^\top W u=(1-S)/d+\sum_i\alpha_i(z_i\cdot u)^2$: overlapping traces jointly support blend directions. Diffusion leaves relative spectral shape invariant — no literal merging by diffusion. **Fusion is the separate, capacity-triggered directional contraction**, with its own exact law.

| | legacy quadratic | core Gibbs | compression (new) |
|---|---|---|---|
| mechanism | closure $\tilde m/(\tilde m+\gamma)$ fades smoothly | basin shrinks, then **vanishes** | basin parameter debited by $\delta/\beta$ per fusion |
| threshold | $t_{1/2}=\Gamma^{-1}\ln(\tilde m_0/\gamma)$ | $t_{\mathrm{forget}}=(\ell_k(0)-\beta D_k-\kappa)/\Gamma$ | $t_{\mathrm{forget}}$ shortens by $\beta D_k/\Gamma$ |
| angular limit | needs stability; capacity–closure bound | $\approx(\bar\alpha/\alpha_k)e^{-\beta}\sqrt{T-1}$ | ledger $\sum\delta$ auditable |
| blending | forced (one attractor) | tunable via $\beta$ | resolution quotient (§8.6) |

Anisotropic-arm capture under legacy: $1.00\to0.00$–$0.14$ under Gibbs (exponential down-weighting, not integration).

## 14. Guaranteed properties

**Retained (exact arithmetic).** $W\succeq0$, $\mathrm{tr}\,W=1$, $\lambda_{\min}(W)\ge(1-S)/d>0$; closed-form log-doses between fusions; all six diffusion laws; dose additivity; exact impure-write decomposition; Gibbs descent for every $\beta>0$; empty memory returns $q$ exactly; permutation and log-dose-shift invariance; basin/floor/resolution/forgetting formulas as first-order/design statements; legacy descent for every $\eta$ and corrected stability; gate limits; batch/streaming gated equality.

**New.** Strict atom bound $\le C$; evidence domination and free-energy monotonicity under compression (Thm A); commutation of fusion with diffusion/dilution (Thm B); trace-one and PSD after fusion accounting; gate-dead criterion exactness and decision invariance of gate-dead deletion; $C\ge T$ regression identity with v2.1.1; ledger additivity; Ward = expected marginal angular distortion to first order (Prop C).

**Not guaranteed.** Exact recall; unique basins or global optimality; which critical point is selected; a floor when legacy margin $\le0$; angular resolution below $\sim1/\beta$; $\Gamma$ identification from the conditional alone or from fused states; a pinned $\omega,\bar\varepsilon,S$; float32-safe margins at extreme $\beta,|\kappa|$; monotone $S$ or purity; order-independence of greedy fusion; $\beta$-independent ledger; monotone max-atom gap; Loewner-monotone $W$; a valid fusion pair at $C=1$ with antipodal atoms (specified fallback).

## 15. Implementation rules

01. Normalize $z,q$; validate finiteness; clamp $s\ge0$.
02. Initialize $W=J$, $S=0$, empty atoms; never $W=0$.
03. Store log-atoms $(z_i,\ell_i,D_i)$ plus $S$ and a lazy clock; materialize $M,W$ only for diagnostics/legacy.
04. Lazy diffusion: $\ell_i\mathrel{-}=\Gamma\Delta t$, $S\mathrel{*}=e^{-\Gamma\Delta t}$; never materialize $J$.
05. Enforce diffuse → write → reinforce → fuse-on-overflow.
06. Doses: $a=-\mathrm{expm1}(-\omega s)$; retention $e^{-(\omega s+\Gamma\Delta t)}$; stable softplus inverse.
07. $\ell_i=\ln(\max(a_i(1-\varepsilon_i),10^{-30}))$; finite at $s=0$, zero gradient through the clamped branch.
08. Masked denominators everywhere a branch may be non-finite (retrieval guard, reinforcement, fusion mask).
09. Off-flags for ablations: $\bar\varepsilon\equiv0$, diffusion bypass, fusion bypass ($C\ge T$). Softplus/sigmoid never reach 0.
10. The gate lives in the inference path: `retrieve` applies saved $\tau$ exactly as `evaluate`. Save state, $\tau$, readout, $C$, config.
11. Calibrate $\tau$ once on a fixed training batch (fused states); never on eval; never resample per report.
12. Log $\beta,\kappa$, doses, $S$, $\hat P$, half-life, margin, $\|\nabla F\|$, gate decisions, hysteresis, decay split, ledger $\sum\delta$, gate-dead deletions, fusion count. Energy-increase fraction must be 0.
13. Run both calibration arms; report cue-only, centroid, kNN baselines **decomposed by present/absent** (§17.1).
14. Self-checks must include §24.
15. Capacity policy: null-fuse gate-dead atoms first ($\ell\le\kappa+\ln\tau$), else min-Ward fusion; **never top-$k$ truncate**.
16. Fuse with $\ell'=\mathrm{logaddexp}(\ell_i,\ell_j)-\beta(1-r)$; route deficit $A(1-e^{-\delta})$ to $(1-S)J$; update $S$.
17. Compute $\delta$ via $2\beta w_iw_j(1-G)/(1+r)$; never $1-r$ by subtraction near $r=1$.
18. Mask pairs with $r<\epsilon_r$ (cost $+\infty$); $C=1$ antipodal fallback: null-fuse lower dose, ledger full evidence.
19. Cache the $C\times C$ Gram; update one row per fusion, $O(Cd)$. Optional: recompute overlaps to trade storage for compute.
20. Fusion debits use current $\beta$; recompute in-forward during training; $\beta$ frozen at deployment.
21. Stage P fusion-free; A/B with fusion; $\tau$ on fixed fused batch.
22. Report per-target evidence retained $e^{-\sum\delta_{\mathrm{target}}}$ and block angular distortion, not "target kept."

## 16. Complexity and storage

| Operation | dense | v2.1.1 log-atom | v2.2.1 (fixed $C$) |
|---|---:|---:|---:|
| store state | $O(d^2)$ | $O(Td+T)$, growing | $C(d+2)+C^2$, **strictly fixed** |
| write | $O(d^2)$ | $O(T)$ shift + append | $O(C^2+Cd)$ incl. fusion |
| diffusion | $O(d^2)$ | $O(1)$ lazy / $O(T)$ | $O(1)$ lazy / $O(C)$ |
| posterior / gate | $O(d^2)$ | $O(Td)$ | $O(Cd)$ |
| retrieval ($N$ steps) | $O(Nd^2)$ | $O(NTd)$ | $O(NCd)$ |
| corrected stability | $O(d^3)$ | $O(T^2d+T^3)$ | $O(C^2d+C^3)$ |
| purity | $O(d^2)$ | $O(T^2)$ cached Gram | $O(C^2)$ |

All $O(1)$ in stream length for fixed $C$. **Design rule:** $C\le d/2$ keeps $C(d+2)+C^2\le d^2$ for $d\ge4$ — the matrix-memory budget quoted in diagnostics. Backprop through fusion: activation memory $O(Cd+Nd)$ with recomputation; $\partial\delta/\partial\beta=1-r$; $\partial\ell_i/\partial\Gamma=-\mathrm{age}_i$; $\partial a/\partial\omega=s e^{-\omega s}$.

## 17. Evaluation plan

### 17.1 Baselines, decomposed

Report every baseline on present and absent trials separately, alongside gated and raw-map numbers: cue-only (exact empty-memory output); top-eigenvector/centroid; kNN over stored embeddings; **kNN + one threshold on max cosine fitted on train** (the like-for-like comparator for a gate that can abstain); diffusion-off at matched init; constant dose vs importance-dosed; legacy quadratic with/without auto-$\gamma$; and at matched $C$: top-$k$ learned dose (v2.1.1 policy), recency, importance-only, random, oracle-prior top-$k$, query-aware keep-target upper bound. **Aggregate comparisons are invalid** when one side is decomposed: forced-commit kNN carries an absent penalty that says nothing about present trials. Claims of beating baselines must be made on present trials; report the paired gate-minus-comparator difference with SE over episodes.

### 17.2 Retrieval experiments

Full/partial cues; absent/unrelated memory (must return $q$ exactly at $W=J$); hard neighbors; recency and importance buckets; age at fixed importance and importance at fixed age; irregular-$\Delta t$ generalization; retention vs $\ln2/\Gamma$; steady states at labeled instants; $\phi$ sweeps below/near/above $\theta_{\mathrm{floor}},\theta_{\mathrm{basin}}$; $t_{\mathrm{forget}}$ vs measured basin collapse; hysteresis rate; gate-threshold sweep; sensitivity to init; unseen sequences.

### 17.3 Calibration before training

Two arms at init, $d=384$: **R** (random unit atoms) and **A** (shared-mean cluster or real embeddings). Per arm: closure at $N$ and $10N$; legacy stability fraction and both floors; Gibbs basin and floor; capture rate; energy-increase counter; all baselines decomposed. If closure is negative at small $\phi$ under legacy, the cause is curvature — do not raise $\eta$. If arm A captures under legacy, do not expect diffusion to fix it.

### 17.4 Identifiability protocol

Run `probe`. Check: (i) P recovers the generating decay in the **split**; (ii) A restores the basin; (iii) B does not move the law; (iv) deploy metric uses saved $\tau$. If $p_{\mathrm{true}}>0.99$, the prior is unidentifiable — change the data, not the optimizer.

### 17.5 Ablations

Forgetting on/off; dose-map variants; $\bar\varepsilon\equiv0$ vs coupled; Gibbs vs legacy; $\beta$ sweep; gate on/off and $\tau$ sweep; CE vs cosine vs mixed; `staged` vs `probe`; fusion on/off ($C\ge T$); fusion policy (Ward vs random-pair vs top-$k$); soft-fusion temperature; embedding geometry raw vs centered.

### 17.6 Capacity regime (new; replaces top-$k$ eviction diagnostic)

Stream $T_{\mathrm{big}}$ writes through the fusion loop; sweep $C$. Report, decomposed: present loss vs $C$; absent loss vs $C$; per-target evidence retained $e^{-\sum\delta_{\mathrm{target}}}$ and block angular distortion; ledger $\sum\delta$; gate-dead deletions; fusions below/above the resolution limit; calibration under load (commit rate vs empirical accuracy, ECE). **Falsifiable predictions:** (i) present loss interpolates smoothly toward cue-only with no cliff, bounded by the target block's $\sum\delta$; (ii) the absent inversion disappears (Thm A); (iii) when all fusions are below the resolution limit, present loss $\approx$ full-memory loss at the same $C$ where top-$k$ loses $\sim30\%$ of targets; (iv) prior-CE gap stays $\approx+0.017$ nats plus the ledger-weighted term. Fusion must beat top-$k$ on **present** trials at small $C$.

## 18. Core mathematical summary

**State.** $W=(1-S)J+M\succeq0$, $\mathrm{tr}\,W=1$; $\le C$ log-atoms.
**Diffuse.** $\ell_i\mathrel{-}=\Gamma\Delta t$; $S\mathrel{*}=e^{-\Gamma\Delta t}$.
**Write.** $a=-\mathrm{expm1}(-\omega s)$; $\ell_j\mathrel{-}=\omega s$; append $\ln(a(1-\varepsilon))$.
**Fuse (on overflow).** Null-fuse if $\ell_i\le\kappa+\ln\tau$; else min-Ward pair: $\mu=m/r$, $\ell'=\mathrm{logaddexp}(\ell_i,\ell_j)-\beta(1-r)$, $1-r^2=2w_iw_j(1-G)$.
**Retrieve.** $F(u)=\beta^{-1}\ln(e^{\beta q\cdot u+\kappa}+\sum_i\alpha_i e^{\beta z_i\cdot u})$; $u^+=\nabla F/\|\nabla F\|$.
**Decide.** $\mathrm{gap}(q)>\ln\tau$ at $u=q$, then descend.
**Ledger.** $\ell=\text{write-dose}-\omega\cdot\text{dilution}-\Gamma\cdot\text{age}-\beta D$; compression is evidence-monotone (Thm A) and commutes with the uniform channels (Thm B).
**Limits.** $\theta_{\mathrm{basin}}=\arccos(1-(\ell_k-\kappa)/\beta)$; $\theta_{\mathrm{floor}}\approx(\bar\alpha/\alpha_k)e^{-\beta}\sqrt{T-1}$; $t_{\mathrm{forget}}=(\ell_k(0)-\beta D_k-\kappa)/\Gamma$.
**Train.** CE at $u=q$ with null class; probe P (fusion-free) → A → B (law frozen); $\tau$ on fixed fused batch.

## 19. Measured results

### 19.1 Setup (inherited, $C\ge T$ regime)

$d=384$, $T=20$, $N=20$, $p_{\mathrm{absent}}=0.2$, $\lambda=0.2$, batch 256, calibration batch 1024, `probe`, CE, CPU, single seed. Synthetic: 10 000 clustered unit vectors ($\rho=0.3$), cues at $\phi\sim U(0.4,1.3)$ rad. Real: 100 Gutenberg books $\to$ 73 923 sentences, `all-MiniLM-L6-v2`, same-passage competitors, cue = random contiguous half-span.

| | synthetic | real |
|---|---:|---:|
| gate (deployed) | **0.0081** | **0.0267** |
| raw map | 0.1744 | 0.1072 |
| cue-only | 0.2924 | 0.2388 |
| kNN | 0.2132 | 0.1658 |
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

These numbers are unchanged in v2.2.1 when fusion never fires ($C\ge T$).

### 19.2 Diagnostic run (v2.1.1 checkpoint, real eval split)

Checkpoint: $\tau=0.6703$, $\beta=21.31$, $\kappa=-17.96$, $\omega=0.0104$, $\Gamma=0.1664$; train 66 963 / eval 6 960.

| deployed vs baselines | all | present | absent |
|---|---:|---:|---:|
| map (ungated) | 0.1082 | 0.0022 | 0.5117 |
| gate (saved $\tau$) | **0.0196** | **0.0139** | **0.0409** |
| cue-only | 0.2415 | 0.3049 | 0.0000 |
| kNN (forced commit) | 0.1472 | 0.0026 | 0.6977 |
| kNN + threshold ($\cos>0.410$, train-fit) | 0.0256 | 0.0202 | 0.0462 |

Cue-free prior CE: oracle $2.2284$; learned log-doses with $\kappa$ re-fit to $-4.50$: $2.2452$; gap $+0.0168$ nats. The saved $\kappa=-17.96$ vs re-fit $-4.50$ illustrates the two-job $\kappa$ (§7); deployment uses the saved value.

### 19.3 Top-$k$ eviction (the broken capacity map; $T_{\mathrm{big}}=100$)

| $C$ | policy | target kept | gate present | gate absent |
|---:|---|---:|---:|---:|
| 4 | learned | 0.701 | 0.1007 | 0.0114 |
| 4 | recency | 0.545 | 0.1478 | 0.0114 |
| 4 | random | 0.046 | 0.2853 | 0.0000 |
| 4 | oracle | 0.704 | 0.1011 | 0.0114 |
| 8 | learned | 0.877 | 0.0510 | 0.0167 |
| 8 | recency | 0.786 | 0.0821 | 0.0114 |
| 8 | random | 0.084 | 0.2750 | 0.0000 |
| 8 | oracle | 0.870 | 0.0527 | 0.0167 |
| 16 | learned | 0.971 | 0.0218 | 0.0167 |
| 16 | recency | 0.952 | 0.0258 | 0.0167 |
| 16 | random | 0.164 | 0.2540 | 0.0000 |
| 16 | oracle | 0.964 | 0.0228 | 0.0167 |
| 32 | learned | 1.000 | 0.0146 | 0.0167 |
| 32 | recency | 0.995 | 0.0147 | 0.0167 |
| 32 | random | 0.340 | 0.2025 | 0.0087 |
| 32 | oracle | 1.000 | 0.0146 | 0.0167 |

Learned $\approx$ oracle because both use the same near-optimal prior; the failure is truncation itself (present cliff; absent inversion where random "wins" by destroying evidence indiscriminately).

### 19.4 Fusion channel

**MEASURED (unit-level):** §8.6. **TO MEASURE (end-to-end):** all §17.6 quantities. No fusion-regime end-to-end numbers are claimed yet.

### 19.5 Known residual limitations (inherited)

1. $\Gamma$ undershoots by 12–17%; $\omega$ stalls near $0.010$, so write-dilution supplies a noisy count proxy for age; cost $O(\omega^2\mathrm{Var})\approx10^{-3}$ nats, weak gradient. Hypothesis: larger batch and longer P drive $\omega\to0$, $\Gamma\to\lambda$; else the bias lies in the $\kappa$/null-odds coupling. Retrieval is insensitive (gate $\approx0.027$ either way).
2. Prior CE rises after P ($3.9\to5.1$) by design (§10.4).
3. Aggregate baseline comparison is not like-for-like (§17.1).

The fusion channel neither fixes nor worsens these by construction (law frozen; P fusion-free).

## 20. Central research hypothesis

HopMem tests whether a trace-one PSD state with dose-additive, uniformly-decaying log-priors and a Gibbs readout can support useful cue-conditioned retrieval with abstention **and** a strictly fixed-size realization in which capacity pressure is a derived, parameter-free, evidence-monotone contraction:

1. Retrieval beats cue-only, centroid, kNN on present trials; abstains on absent. *(MEASURED, §19.1–19.2.)*
2. The forgetting rate is recoverable from the marginal law only. *(MEASURED.)*
3. Vagueness is basin collapse at a computable time, accelerated by compression via the debit $\delta/\beta$. *(THEOREM.)*
4. Compression is gradual forgetting: fusion is pointwise evidence-dominated, commutes with the two learned channels, adds zero parameters, and its cost is an auditable ledger. *(THEOREM + unit-MEASURED; end-to-end TO MEASURE.)*
5. The decision precedes descent. *(THEOREM + deterministic witness.)*
6. The latent is a Transformer interface: $C$ atoms with log-dose biases are $C$ KV tokens; the readout is one attention head. *(SPECIFIED; TO MEASURE.)*

Distinctive claims: single convex write primitive; dose additivity as additive log-priors with three channels; provable descent; sharp basin-collapse forgetting; decision-before-descent; marginal/conditional identifiability split; the fusion channel and its theorems. Stated costs: atom-dependence; greedy order-dependence; $\beta$-gauge of the ledger; max-gap caveat; architectural $C$; a calibrated threshold outside the gradient loop; one scalar serving two calibration roles.

## 21. Supersession: withdrawn claims

The following must not be used.

**Inherited from v2.0.4 (cleared in v2.1.1).**
01. "Energy descent holds only for sufficiently small steps." → Theorem for every $\eta$ (legacy) and every $\beta$ (Gibbs).
02. "Crosstalk $\approx(S-\tilde\alpha)/d$ is the retrieval limit." → That is a support shift.
03. "Vagueness = spectral gap closure / merging," and arbitrary-tolerance $t_{\mathrm{merge}}$.
04. "Impure blur applies only to the injected target." → Exact decomposition (§4.2).
05. "$W$-form vs $M$-form differ by a bounded $h$ decrease." → Exact trajectory reparameterization.
06. "The $10^{-8}$ guard becomes reachable at margin $\sim10^{-4}$."
07. "$\mu$ reliably rounds to 1 after 25 writes."
08. "Compute doses as $1-e^{-x}$." → Use $-\mathrm{expm1}(-x)$.
09. "The $g=2.5$ query-gap example applies under continued periodic writing."
10. "Gain over cue-only suffices." → Centroid and kNN too, decomposed.
11. "Diffusion creates no new basins / no spurious minima." → Range of $M$ unchanged; critical-point counts not claimed.

**v2.1.1 structural withdrawals.**
12. "§6.4's $\theta_{\mathrm{floor}}$ is a fixed-point property independent of $\eta,N$." → Omits curvature; no floor when margin $\le0$.
13. "The legacy quadratic readout is adequate for retrieval." → One stable attractor; capacity–closure bound; 5–6% stable at init.
14. "Retrieval depends only on $(M,\gamma q)$." → Legacy only; Gibbs depends on the atom set.
15. "Storage is fixed-size $O(d^2)$." → Was $O(Td)$ growing. **Restored in modified form in v2.2.1:** strictly fixed $C(d+2)+C^2$ at the stated price of the fusion ledger.
16. "The Gibbs floor prefactor is $\sqrt{(T-1)/d}$." → $\sqrt{T-1}$; no $1/d$.
17. "The argmax of the posterior is the retrieval decision." → Hysteresis; null basin strictly smaller.
18. "Vagueness is smooth gain loss with a half-closure time." → Sharp basin collapse (Gibbs).
19. "Decorrelated $(s,\Delta t)$ suffices to identify $\Gamma$." → Also needs the marginal objective and ambiguous cues.
20. "One $\kappa$ can calibrate both marginal null-odds and the conditional commit threshold." → Cannot.
21. "$S$ indicates useful memory." → Near-gauge.
22. "$\tau$ may be fit on eval or resampled per report." → Fixed training calibration batch, saved.
23. "Every operation is one convex primitive." → **Stays withdrawn.** Alphabet {write, diffuse, fuse, null-fuse}; fusion is a coarse-graining plus depolarizing step, not a single convex mixture of $W$. The pruning exception is absorbed (§4.4).

**New in v2.2.1.**
24. "Top-$k$ truncation is the capacity policy." → Zeroth-order lossy for an angular loss; pays full mass where fusion pays mass $\times$ angle$^2$. Replaced by §4.5.
25. "Pruning is an exception outside the primitive." → Absorbed as null-fusion with the exact gate-dead criterion.
26. "Block prior odds against the null are exactly preserved under fusion." → Debited by exactly the block's ledger $\sum\delta$; no other distortion.
27. "The max-atom gate gap is monotone under fusion." → Only block posterior mass is (Thm A); equal logits can aggregate to $+\ln2$. $\tau$ recalibrates.
28. "The fusion ledger is $\beta$-independent." → $\delta=\beta(1-r)$; recompute in-forward during training.
29. "$W$ is Loewner-monotone under compression." → The evidence functional is monotone, not the matrix order.
30. "Fusion conserves total stored mass $S$." → $S'=S-A(1-e^{-\delta})$; deficit routed to $(1-S)J$; trace-one preserved.

## 22. Changes from v2.1.1, and why

### 22.1 The structural finding

The diagnostics isolated the capacity map as the only unidentified object: the law is near-optimal ($+0.0168$ nats) and the gate wins like-for-like, but truncation produces a present cliff and an absent inversion because it is zeroth-order lossy for a second-order (angular) distortion. The readout's own free energy defines the correct compression: among two-atoms→one-atom maps preserving the Gibbs form, the resultant-debit fusion is the unique choice that is pointwise evidence-dominated, commutes with both existing forgetting channels, and adds no parameters. Pruning is its null special case; the gate-dead criterion is exact and sharper than the age bound. Fixed size is restored with an auditable price, and the same functional form is a Transformer attention interface.

### 22.2 Section map

§1 updated; §2 notation/quickstart (new); §3.5 atom-dependence retained; §4.4 absorbed; **§4.5 new**; §5 D5 commutation extended; §7 $\beta$-gauge caveat; **§8.6 new**; §9 fusion-in-loop; §10.4 P fusion-free; §11 confidence renamed $\mathrm{cf}$; §13 compression row; §14 new guarantees; §15 rules 15–22; §16 fixed-size table; **§17.6 new**; §18 third channel; §19 diagnostic tables + unit-measured + TO MEASURE; §20 claims 4, 6; §21 items 24–30, #15 restored, #23 stays; **§23 new**; **§24 self-check suite**.

### 22.3 Deliberately not changed

State class; convex write primitive; dose maps; diffusion generator and all six laws; impure-write decomposition; $W/M$ reparameterization; Gibbs readout and descent proof; decide-before-descend and the $\tau$ protocol; probe identifiability split; reinforcement branch and masked denominator; decoder normalization and float32 analysis; legacy readout ablation with its theorems. The revision is confined to the capacity channel, its theorems, the training/evaluation protocol around it, and the LM interface.

## 23. Language-model interface (SPECIFIED; not yet measured)

**Latent.** $(\mu_c,\ell_c)\in\mathbb{R}^{C\times(d+1)}$ — $C$ unit directions with log-doses.

**Read.** The Gibbs posterior at $u=q$ is exactly one cross-attention head with tied key/value:

$$u=\sum_c \mathrm{softmax}_c\big(\beta\,q\cdot\mu_c+\ell_c\big)\,\mu_c \;+\; p_0\,q .$$

For a decoder LM: prepend $C$ memory tokens, keys/values $W_K\mu_c$, $W_V\mu_c$ (fixed projections to LM width; $\beta$ absorbs $1/\sqrt{h}$), attention-logit bias $\ell_c$ (a learned per-token ALiBi), null token with bias $\kappa$. The gate becomes the residual mix $g=\sigma(\mathrm{gap}-\ln\tau)$ — decide-before-descend, unchanged. The $N$-step descent is optional recurrent refinement over the same $C$ tokens.

**Write.** $z_t=\mathrm{normalize}(W_z h_t)$; **surprise is the LM's own signal**, $s_t=-\ln p_{\mathrm{LM}}(x_t\mid x_{<t})$ (or residual surprise $\|z_t-Mz_t\|$), fed to the existing dose map $a=1-e^{-\omega s}$. Diffuse, write, fuse-on-overflow per token: $O(Cd)$, state never grows.

**Transfer caveat (honest).** $\beta,\kappa$ were calibrated on MiniLM sentence geometry and $\mathrm{Exp}(1)$ importance; LM hidden states and NLL-scaled surprise require stage A/B recalibration (law frozen) on LM states. The functional form is identical, so the readout/gate interface transfers; the calibration does not transfer for free. Evaluate retrieval separately (§12).

## 24. Self-check suite (36 checks)

**State and dynamics (1–10).** 1: $W\succeq0$, $\mathrm{tr}\,W=1$, $S<1$ after random write/diffuse sequences. 2: closed-form log-doses vs sequential simulation. 3: diffusion semigroup composition. 4: spectral invariance under diffusion. 5: steady-state $S^*$ under periodic writes. 6: impure-write decomposition exact. 7: empty memory returns $q$ exactly. 8: permutation invariance. 9: uniform log-shift invariance of posterior. 10: $s=0$ finite log-dose, zero clamped-branch gradient.

**Readout and gate (11–18).** 11: energy-increase fraction 0; guard count 0. 12: descent at $N$ and $10N$ agrees. 13: planar hysteresis witness (§8.4 numbers). 14: gate limits $\tau\to0$, $\tau\to\infty$. 15: streaming-vs-batch gated equality. 16: $\tau$ protocol (fixed batch, saved, applied identically). 17: basin/floor/resolution formulas vs measured. 18: cue-free gradient restoration ($\partial\mathrm{CE}/\partial\Gamma$ nonzero under ambiguity, $\approx0$ when one-hot).

**Legacy and numerics (19–26).** 19: factored-vs-dense corrected stability, including the saddle case. 20: legacy descent for every $\eta$; $\|v_n\|\ge1-\chi$. 21: capacity–closure bound witness. 22: masked denominators (retrieval guard, reinforcement, fusion mask). 23: stable softplus inverse. 24: $-\mathrm{expm1}$ dose form. 25: decoder normalization Jacobian bounds. 26: no-match episodes prevent $\kappa\to-\infty$; $p_{\mathrm{true}}>0.99$ warning fires.

**Fusion (27–36, new).** 27: $C\ge T$ $\equiv$ v2.1.1 bit-for-bit. 28: domination inequality on random queries, zero violations. 29: commutation residual 0 to machine precision. 30: trace-one and PSD after fusion accounting; $S$ deficit routed to $J$. 31: gate-dead criterion exactness (no commit for any $q$ when $\ell\le\kappa+\ln\tau$). 32: gate-dead deletion leaves gate decisions invariant. 33: ledger additivity; stable-$\delta$ form agreement with $\beta(1-r)$. 34: streaming-vs-replayed fusion equality at fixed order. 35: $C=1$ antipodal fallback fires and ledgers correctly; $r<\epsilon_r$ mask. 36: Ward cost matches Prop C first-order prediction on synthetic clusters; capacity invariant $\le C$ maintained through write/reinforce/fuse sequences.
