# HopMem — Frozen Specification v2.0.4

HopMem is a fixed-size semantic associative memory whose state is a real symmetric **density matrix**: positive semidefinite with trace exactly one. All memory operations are instances of a single **convex mixing primitive** $W \leftarrow (1-a)W + aT$ with only two target types: pure states $zz^{\top}$ and the maximally mixed state $J = I/d$. Forgetting is a **mandatory depolarizing diffusion channel** at learnable rate $\Gamma$. Writes inject importance-dosed mass; an impure write is exactly a diffusion followed by a pure write (§6.2), not a separate mechanism. Retrieval is a cue-injected normalized gradient step — the generalized power method — and **provably descends the energy for every step size** (§8.3). The core is dense, with **no sparsity and no pattern separation**. "Vagueness" is not spectral reshaping: diffusion leaves the relative spectral shape of the signal exactly invariant and acts as pure gain loss relative to the cue (§13); angular resolution is limited by a tangential cross-talk floor (§6.4). Every scalar that shapes the dynamics is learnable; there are no fixed model constants. For finite inputs and finitely many operations the state is strictly positive definite ($S<1$); pure states are limits, not reachable states.

This document supersedes v1.10 through v2.0.3. Prior claims that conflict with it are withdrawn; the cleared list is §20. All identities, bounds, and table entries were checked by hand derivation and numerical simulation (including a 61-check external audit and independent re-verification of the v2.0.4 corrections).

---

## 1. Objective

The system maintains a fixed-size semantic memory as a trace-one positive-semidefinite matrix. Incoming normalized embeddings are written with importance-dosed weight. Time erodes all structure isotropically toward maximum entropy. Retrieval uses a normalized cue and an energy-descending recurrence to produce a unit-norm semantic state.

The system supports:

- fixed-size storage with exact invariants ($W \succeq 0$, $\mathrm{tr}\,W = 1$),
- one update primitive for all memory operations,
- mandatory time-based forgetting with a learnable rate,
- priority-weighted writing with a learnable write/diffusion coupling,
- cue-conditioned retrieval with guaranteed energy descent,
- blending through shared eigenstructure, and predictable gain-loss "vagueness",
- learned write/read/forget parameters,
- optional reinforcement and decoding.

The system does not support:

- exact episodic recall,
- unlimited capacity,
- individually addressable records,
- guaranteed unique basins or global optimality,
- pattern separation of overlapping traces,
- sparse retrieval states,
- immunity to dominant-direction capture on anisotropic embeddings (diffusion preserves eigenvectors; §13, §17),
- angular resolution below the cross-talk floor (§6.4).

---

## 2. Inputs

Let $d \ge 2$ be the embedding dimension. At event time $t$:

$$
z_t \in \mathbb R^d,\quad \|z_t\|_2 = 1,
\qquad
q \in \mathbb R^d,\quad \|q\|_2 = 1,
\qquad
s_t \ge 0 \text{ finite}.
$$

Each event carries a timestamp; $\Delta t \ge 0$ is the elapsed time since the previous event. If no timestamps are available, use unit steps $\Delta t = 1$; the learnable rate $\Gamma$ absorbs the time unit either way.

**No scale assumption on $s_t$.** The write map of §6 is self-bounding for every $s_t \ge 0$: the learnable gain $\omega$ absorbs the importance scale, exactly as $\Gamma$ absorbs the time scale.

The importance score is a proxy, not ground truth. All guarantees require only $s_t \ge 0$ finite and $\Delta t \ge 0$ finite.

---

## 3. Memory state

The memory is a real symmetric matrix satisfying

$$
W \succeq 0,
\qquad
\mathrm{tr}(W) = 1 .
$$

Initialize

$$
W_0 = J = \frac{I}{d}.
$$

An empty memory is the maximally mixed state, **not** the zero matrix; trace one holds at all times, including initialization.

### 3.1 Canonical signal decomposition

Every state reachable by the operations of §4–§6 and §11 has the exact form

$$
\boxed{\;
W \;=\; (1-S)\,J \;+\; M,
\qquad
M = \sum_i \tilde\alpha_i\, z_i z_i^{\top},
\qquad
\tilde\alpha_i \ge 0,
\qquad
S = \sum_i \tilde\alpha_i \in [0,1].
\;}
$$

$M \succeq 0$ is the **signal matrix**; $S$ is the **signal mass**. The coefficients $\tilde\alpha_i$ are mathematical contributions, not records; reinforced directions (§11) enter the sum as additional terms. $S$ is tracked as a scalar alongside $W$ with exact recursions (§4.2); it is bookkeeping, not an intrinsic function of $W$. The intrinsic bound $S \ge 1 - d\,\lambda_{\min}(W)$ holds but is not needed operationally.

**Strict positivity.** For finite scores and finitely many operations, $S < 1$ strictly: each operation is a convex mixture with weight $<1$ toward a target of signal mass $\le 1$, starting from $S=0$. Hence

$$
\lambda_{\min}(W) \ge \frac{1-S}{d} > 0,
\qquad
\lambda_{\max}(W) \le \frac{1-S}{d} + S
= 1-(1-S)\left(1-\frac1d\right) < 1,
$$

so $W \succ 0$ and $\mu = u^{\top}Wu < 1$ for every unit $u$. The pure-state case $S=1$ is a limit, not a reachable state. This is why the $\mu=1$ division hazard of §11 is purely a floating-point artifact.

Because $J$ acts as a constant on the unit sphere, **the retrieval landscape depends only on $M$** (§8.2).

### 3.2 Sign-blindness

$zz^{\top} = (-z)(-z)^{\top}$: the memory operator stores unoriented directions. The retrieval map is **not** sign-blind: $u_0 = q$ and the cue term $\gamma u^{\top}q > 0$ pin the hemisphere (§8).

### 3.3 State scalars

$$
S \in [0,1],
\qquad
P = \mathrm{tr}(W^2) = \frac{1-S^2}{d} + \|M\|_F^2 \in \left[\tfrac1d, 1\right],
\qquad
\hat P = \frac{dP - 1}{d-1} \in [0,1].
$$

$P$ is purity; $\hat P$ is normalized purity ($1$ = pure state, $0$ = $J$), computable in $O(d^2)$ without eigendecomposition. Von Neumann entropy $S_{\mathrm{vN}}(W) = -\mathrm{tr}(W \ln W)$ is available at $O(d^3)$ for diagnostics only.

---

## 4. The unified convex primitive

### 4.1 Definition and invariants

Every memory operation is

$$
\boxed{\; W \leftarrow (1-a)\,W + a\,T, \qquad a \in [0,1], \quad T \succeq 0, \quad \mathrm{tr}(T) = 1. \;}
$$

**Invariance proof.** For any $x$: $x^{\top}W_{\mathrm{new}}x = (1-a)\,x^{\top}Wx + a\,x^{\top}Tx \ge 0$. Trace: $(1-a)\cdot 1 + a \cdot 1 = 1$. Symmetry is preserved by linearity. No projection or renormalization is ever needed.

The operations differ only in target and weight:

| Operation | Target $T$ | Weight $a$ |
|---|---|---|
| Diffusion (§5, mandatory) | $J$ | $p = 1 - e^{-\Gamma \Delta t}$ |
| Write (§6) | $(1-\varepsilon_t)z_tz_t^{\top} + \varepsilon_t J$ | $a_t = 1 - e^{-\omega s_t}$ |
| Reinforcement (§11, optional) | $u^{\ast}u^{\ast\top}$ | $\beta$ |

Both nontrivial weights are **exponential saturation of a dose**:

$$
a_{\mathrm{write}} = 1 - e^{-\omega s_t},
\qquad
a_{\mathrm{diff}} = 1 - e^{-\Gamma \Delta t}.
$$

The impure write target is a convenience parameterization: by §6.2 it is exactly a diffusion followed by a pure write, so only the two target types $zz^{\top}$ and $J$ are primitive.

### 4.2 Scalar bookkeeping

Each target carries a signal mass ($J \mapsto 0$; the write target $\mapsto 1-\varepsilon_t$; a pure state $\mapsto 1$), and $S$ obeys the **same** convex recursion as $W$:

$$
\text{diffusion: } S \leftarrow e^{-\Gamma\Delta t} S,
\qquad
\text{write: } S \leftarrow (1-a_t)S + a_t(1-\varepsilon_t),
\qquad
\text{reinforce: } S \leftarrow (1-\beta)S + \beta .
$$

These are exact: applying the operation to $W = (1-S)J + M$ and re-collecting terms gives $M \leftarrow e^{-\Gamma\Delta t}M$, $M \leftarrow (1-a_t)M + a_t(1-\varepsilon_t)z_tz_t^{\top}$, $M \leftarrow (1-\beta)M + \beta u^{\ast}u^{\ast\top}$ respectively, with $S = \mathrm{tr}(M)$ throughout. $S \in [0,1]$ is preserved by induction, and $S<1$ strictly for finite reachable states (§3.1).

**$S$ is not monotone.** Diffusion strictly decreases $S$; a write moves $S$ toward $1-\varepsilon_t$ (up if $S < 1-\varepsilon_t$, down if $S > 1-\varepsilon_t$); reinforcement moves $S$ toward $1$.

### 4.3 Composition and ordering

Diffusion maps compose as **maps** (semigroup):

$$
\Phi_{t_1} \circ \Phi_{t_2} = \Phi_{t_1 + t_2},
\qquad
\Phi_t(W) = e^{-\Gamma t} W + (1 - e^{-\Gamma t}) J.
$$

Different operations do **not** commute. For a write of weight $a$ and a diffusion of weight $p$, the exact commutator is

$$
(\text{write} \circ \text{diffuse})(W) - (\text{diffuse} \circ \text{write})(W)
= a\,p\,(T - J),
$$

vanishing only if $ap = 0$ or $T = J$.

**Ordering convention.** At each event, first advance the memory to the event's timestamp (diffuse by $\Delta t$), then write, then optionally reinforce. A fresh write is undecayed at its own timestamp. Retrieval runs at frozen $W$ (§8).

---

## 5. Mandatory diffusion

### 5.1 Continuous generator and exact discrete map

Between events the state obeys the depolarizing generator on the trace-one slice:

$$
\frac{dW}{dt} = \mathcal L(W) = \Gamma\,(J - W),
\qquad \Gamma > 0 \text{ learnable},
$$

with exact solution

$$
W(t) = e^{-\Gamma t}\,W(0) + \left(1 - e^{-\Gamma t}\right) J .
$$

The discrete update over a gap $\Delta t$ is exact, not an Euler approximation:

$$
\boxed{\; W \leftarrow (1-p)\,W + p\,J, \qquad p = 1 - e^{-\Gamma \Delta t} \in [0,1). \;}
$$

**Channel validity, self-contained.** $\Phi_p$ is a convex mixture of the identity channel and the replacer channel $W \mapsto J$, both completely positive; hence $\Phi_p$ is CPTP. An explicit Kraus form on real matrices: $K_0 = \sqrt{1-p}\,I$ and $K_{ij} = \sqrt{p/d}\,E_{ij}$, since $\sum_{ij} E_{ij} W E_{ij}^{\top} = \mathrm{tr}(W)\,I$ and $K_0^{\top}K_0 + \sum_{ij} K_{ij}^{\top}K_{ij} = I$. No complex structure is required; real symmetric PSD trace-one matrices are closed under all operations of this specification.

### 5.2 Exact forgetting laws

Let $W(0) = \sum_i \lambda_i(0)\, v_i v_i^{\top}$. Then:

1. **Spectral contraction.** Eigenvectors are fixed; eigenvalues obey
$$
\lambda_i(t) = \frac1d + e^{-\Gamma t}\left(\lambda_i(0) - \frac1d\right),
\qquad
\lambda_i(t) - \lambda_j(t) = e^{-\Gamma t}\left(\lambda_i(0) - \lambda_j(0)\right).
$$
Ordering is preserved; no eigenvector rotates; the range of $M$ is unchanged, so diffusion creates no new stored direction.
2. **Distance contraction.** $\|W(t) - J\|_F = e^{-\Gamma t}\,\|W(0) - J\|_F$, and the purity contrast obeys $P(t) - \tfrac1d = e^{-2\Gamma t}\left(P(0) - \tfrac1d\right)$.
3. **Entropy monotonicity.** $\lambda(t) = (1-p)\lambda(0) + p\,(\tfrac1d,\dots,\tfrac1d)$ is a convex mixture with the uniform vector, which is majorized by every probability vector; hence $\lambda(t) \prec \lambda(0)$ and, by Schur-concavity, $S_{\mathrm{vN}}(W(t)) \ge S_{\mathrm{vN}}(W(0))$. Equivalently
$$
D(W \,\|\, J) = \ln d - S_{\mathrm{vN}}(W)
$$
decreases monotonically to $0$.
4. **Half-life.** A trace's signal mass halves every $t_{1/2} = \ln 2 / \Gamma$ of elapsed time (in the absence of writes).
5. **Irreversibility.** For $\Gamma > 0$ the evolution is a strict contraction: entropy strictly increases unless $W = J$, and no CPTP inverse exists. The trajectory is a straight segment in mixture-affine coordinates, reparameterized by $e^{-\Gamma t}$; it is not claimed to be a Bures or exponential geodesic.
6. **Relative spectral invariance.** Since $M \leftarrow e^{-\Gamma t}M$, every relative spectral quantity of $M$ (eigenvalue ratios, normalized gaps, effective rank, spectral entropy of $M/S$) is exactly invariant. Diffusion is pure scalar gain decay on the signal; see §13.

### 5.3 Steady state under periodic writing

With a write of weight $a$ and impurity $\varepsilon$ every $\Delta t > 0$, under the diffuse-then-write convention, the cycle map on the post-write value is

$$
S_{\mathrm{post}} \leftarrow (1-a)\,e^{-\Gamma\Delta t}\,S_{\mathrm{post}} + a(1-\varepsilon),
$$

so the fixed point has two values at two instants, related by one diffusion step:

$$
\boxed{\;
S^{\ast}_{\mathrm{post}} = \frac{a(1-\varepsilon)}{1-(1-a)e^{-\Gamma\Delta t}},
\qquad
S^{\ast}_{\mathrm{pre}} = e^{-\Gamma\Delta t}\,S^{\ast}_{\mathrm{post}}
= \frac{a(1-\varepsilon)\,e^{-\Gamma\Delta t}}{1-(1-a)e^{-\Gamma\Delta t}}.
\;}
$$

$S^{\ast}_{\mathrm{post}}$ is the value just after a write; $S^{\ast}_{\mathrm{pre}}$ is the value just after diffusion, before the next write. In dose form ($1-a = e^{-\omega s}$):

$$
S^{\ast}_{\mathrm{post}} = \frac{(1-e^{-\omega s})(1-\varepsilon)}{1-e^{-(\omega s+\Gamma\Delta t)}},
\qquad
S^{\ast}_{\mathrm{pre}} = e^{-\Gamma\Delta t}\,S^{\ast}_{\mathrm{post}}.
$$

Sanity limits, per instant (pure writes, $\varepsilon=0$):

| Limit | $S^{\ast}_{\mathrm{pre}}$ | $S^{\ast}_{\mathrm{post}}$ |
|---|---:|---:|
| $\Gamma \to 0$ | $1$ | $1$ |
| $a \to 1$ | $e^{-\Gamma\Delta t}$ | $1$ |
| $\Gamma \to \infty$ | $0$ | $a$ |

For impure writes the $\Gamma\to 0$ limit is $1-\varepsilon$ for both instants, and the $\Gamma\to\infty$ post-write floor is $a(1-\varepsilon)$, not $0$. If $a=0$ there are no writes and $S^{\ast}=0$.

**Worked check** ($\omega=0.5$, $\Gamma=0.2$, $s=\Delta t=1$, pure): $a \approx 0.393469$, $e^{-\Gamma\Delta t}\approx 0.818731$, denominator $1-e^{-0.7}\approx 0.503415$. Then $S^{\ast}_{\mathrm{pre}} \approx 0.63992$ and $S^{\ast}_{\mathrm{post}} \approx 0.78160$.

**Measurement note.** Retrieval advances the memory to the query time and runs at frozen $W$ with no write at the query. In the **write-only periodic regime** of this subsection, for a query a gap $g \ge 0$ after the last write,

$$
S = S^{\ast}_{\mathrm{post}}\,e^{-\Gamma g}.
$$

At $g=0$ this is the post-write value; at $g=\Delta t$ (a query just before the next periodic write) it equals $S^{\ast}_{\mathrm{pre}}$. Under continued periodic writing $g \in [0,\Delta t)$; a gap $g > \Delta t$ (e.g. $g=2.5$ with $\Delta t=1$, giving $S \approx 0.4741$) is valid only if writes stop. With reinforcement enabled or aperiodic writes, replace $S^{\ast}_{\mathrm{post}}$ by the actual post-event value. For $g=0.25$ in the worked check, $S \approx 0.7435$. §17.2 must compare empirical $S$ against the formula at the labeled instant or gap.

### 5.4 Dose additivity

A contribution written with importance $s$ and then aged by $\Delta t$ retains, per event cycle, the factor

$$
(1-a)\,e^{-\Gamma \Delta t} = e^{-(\omega s + \Gamma \Delta t)} .
$$

Write-driven dilution and time-driven diffusion **add in the exponent**. This is exact, and it creates an identifiability constraint on training data (§7.3).

---

## 6. Writing

### 6.1 Write dose

$$
\boxed{\; a_t = 1 - e^{-\omega s_t}, \qquad \omega = \mathrm{softplus}(a_\omega) > 0 \text{ learnable}. \;}
$$

Properties: $a(0) = 0$; $a$ is strictly increasing in $s$; $a_t \in [0,1)$ for every finite $s_t$; $a \approx \omega s$ for small dose; $a \to 1$ only as $s \to \infty$. Gradient: $\partial a_t / \partial \omega = s_t e^{-\omega s_t} \ge 0$. **Implementation:** compute $a_t = -\mathrm{expm1}(-\omega s_t)$ (§15).

### 6.2 Impure writes are a reparameterization

The write target is

$$
T_t = (1-\varepsilon_t)\, z_t z_t^{\top} + \varepsilon_t\, J,
\qquad
\boxed{\; \varepsilon_t = \bar\varepsilon\,(1 - a_t), \qquad \bar\varepsilon = \sigma(a_\varepsilon) \in (0,1) \text{ learnable}. \;}
$$

Strong writes are sharp ($a \to 1 \Rightarrow \varepsilon \to 0$); weak writes are blurred up to $\bar\varepsilon$. The injected signal mass $a_t(1-\varepsilon_t) = a_t(1-\bar\varepsilon) + \bar\varepsilon a_t^2$ is strictly increasing in $a_t$.

**Exact decomposition.** For any $a \in [0,1)$, $\varepsilon \in [0,1]$:

$$
\boxed{\;
(1-a)W + a\bigl[(1-\varepsilon)zz^{\top} + \varepsilon J\bigr]
= (1-a_2)\bigl[(1-p')W + p'J\bigr] + a_2\,zz^{\top},
\;}
$$

$$
a_2 = a(1-\varepsilon),
\qquad
p' = \frac{a\varepsilon}{1-a+a\varepsilon} \in [0,1).
$$

An impure write **is** a diffusion of the old state by $p'$ followed by a pure write of weight $a_2$. At fixed $a$, $\varepsilon$ reduces injected mass; at fixed injected mass $a_2$, it adds write-triggered diffusion. $\varepsilon$ is therefore a reparameterization of extra mixing, not a separate mechanism; $\bar\varepsilon$ is identifiable only from the coupling between fresh-trace mass and old-trace dilution (§7.3).

**Pure-write ablation.** $\sigma$ never reaches $0$; keep an explicit flag for $\varepsilon \equiv 0$.

**Optional variant (specified, deferred): purity-coupled coupling.** Replace the coupling by

$$
\varepsilon_t = \sigma\!\left(a_\varepsilon + b_\varepsilon\,(1 - \hat P(W))\right),
$$

with $b_\varepsilon$ learnable and $\hat P$ from §3.3 ($O(d^2)$, no eigendecomposition). This is state-dependent write-triggered diffusion. A von Neumann-entropy version costs $O(d^3)$ per write; prefer the purity proxy. Enable only after the core is validated.

### 6.3 Write update

$$
W \leftarrow (1-a_t)\,W + a_t\,T_t,
\qquad
M \leftarrow (1-a_t)\,M + a_t(1-\varepsilon_t)\,z_tz_t^{\top},
\qquad
S \leftarrow (1-a_t)\,S + a_t(1-\varepsilon_t).
$$

All invariants follow from §4.1–4.2.

### 6.4 Contribution retention, support, and the angular floor

For any sequence of writes and diffusions, the decomposition of §3.1 holds exactly with

$$
\tilde\alpha_{i,t}
= a_i\,(1-\varepsilon_i)
\prod_{\substack{j > i \\ \text{writes}}} (1-a_j)
\prod_{\substack{\text{diffusions} \\ \text{after } i}} e^{-\Gamma \Delta t},
\qquad
S_t = \sum_{i \le t} \tilde\alpha_{i,t},
$$

generalized by additional factors $(1-\beta_k)$ for reinforcements (§11), with reinforced directions entering the sum with initial mass $\beta_k$ and decaying by all later factors. Importance sets initial mass; later writes and elapsed time set decay. A high-importance trace starts with more mass; it does not decay slower.

**Two-sided mass bound:**

$$
\tilde\alpha_{i,t}
\;\le\;
z_i^{\top} M_t z_i
\;\le\;
\lambda_{\max}(M_t)
\;\le\;
S_t
\;\le\; 1,
$$

with the lower bound tight when all other stored directions are orthogonal to $z_i$, and the **support identity**

$$
z_i^{\top} M_t z_i = \tilde\alpha_{i,t} + \sum_{j \ne i} \tilde\alpha_{j,t}\,(z_j^{\top} z_i)^2 .
$$

For random unit vectors the support shift is $\approx (S_t - \tilde\alpha_{i,t})/d$ (about $0.0025$ at $d=384$, $T=20$, $\alpha=0.05$). **This is an energy shift, not the quantity that limits retrieval accuracy.**

**Tangential cross-talk floor.** The tangential force at $u = z_i$ from the other traces is

$$
x_\perp = \sum_{j\ne i}\tilde\alpha_j\,(z_j^{\top}z_i)\,(I - z_iz_i^{\top})\,z_j,
$$

and for random unit vectors, using $\mathbb E[t^2(1-t^2)] = \frac{d-1}{d(d+2)}$ for $t = z_j^{\top}z_i$,

$$
\mathbb E\|x_\perp\|_2^2 = \frac{d-1}{d(d+2)}\sum_{j\ne i}\tilde\alpha_j^2
\;\approx\; \frac{\sum_{j\ne i}\tilde\alpha_j^2}{d}.
$$

The equilibrium residual angle from the target is

$$
\boxed{\;\theta_{\mathrm{floor}} \approx \frac{\|x_\perp\|_2}{\tilde m + \gamma}
\;\approx\; \frac{\sqrt{\sum_{j\ne i}\tilde\alpha_j^2/d}}{\tilde m+\gamma},\;}
$$

a **fixed-point property, independent of $\eta$ and $N$**. For $T$ equal masses $\alpha$: $\theta_{\mathrm{floor}} \approx \alpha\sqrt{(T-1)/d}/(\alpha+\gamma)$. At $\alpha=0.05$, $\gamma=0.01$, $T=20$, $d=384$: $\theta_{\mathrm{floor}} \approx 0.185$ rad (measured $\|x_\perp\| = 0.01098$ vs predicted $0.01108$). Cues with $\varphi < \theta_{\mathrm{floor}}$ are **made worse** (measured closed fraction $-0.83$ at $\varphi=0.1$); $\varphi=0.6$ improves ($+0.44$). Raising $\eta$ cannot beat the floor; only changing $M$ (fewer/sharper traces, centering) or $\gamma$ can. Negative closure at small $\varphi$ diagnoses the floor, not weak gain.

For anisotropic embeddings with a large shared mean component, cross-talk can dominate $\tilde\alpha$; diffusion does not remove this (§13, §17.3).

---

## 7. Learnable parameters

### 7.1 Parameter set

$$
\theta = \{\omega,\ \Gamma,\ \eta,\ \chi,\ \bar\varepsilon\},
\qquad
\text{optional: } \bar\beta \text{ (§11)}.
$$

$$
\omega = \mathrm{softplus}(a_\omega), \quad
\Gamma = \mathrm{softplus}(a_\Gamma), \quad
\eta = \mathrm{softplus}(a_\eta), \quad
\chi = \sigma(a_\chi), \quad
\bar\varepsilon = \sigma(a_\varepsilon).
$$

The cue strength is **derived**, not free:

$$
\boxed{\; \gamma = \frac{\chi}{\eta}, \qquad \eta\gamma = \chi \in (0,1). \;}
$$

The stability condition of §8.3 holds by construction. There are no caps $\eta_{\max}, \gamma_{\max}$, no half-saturation constant, no $\kappa_{\max}$, no importance-scale assumption. The dynamics contain **no fixed model constants**: $d$ is given by the encoder, $N$ is a compute budget, and $\epsilon_{\mathrm{fall}}, \epsilon_L, \epsilon_{\mathrm{dec}}$ are floating-point guards, not model parameters. $\gamma > 0$ for all finite logits, so hemisphere pinning is always active.

Interpretation of $\chi$: it is the per-step cue-hold dose. With read gain $g = \eta \tilde m$ (§8.5), retrieval closes the fraction $g/(g+\chi)$ of the cue-to-target angle at equilibrium; $\chi \to 0$ gives full closure, $\chi \gg g$ holds the cue.

### 7.2 Suggested initialization

Chosen so that initialization reproduces v1.10's default retrieval behavior with slow diffusion added:

| Parameter | Initial value | Logit | Meaning at init |
|---|---:|---:|---|
| $\omega$ | $-\ln 0.95 \approx 0.0513$ | $a_\omega \approx -2.94$ | $a(s{=}1) = 0.05$ |
| $\Gamma$ | $10^{-3}$ | $a_\Gamma \approx -6.91$ | half-life $\approx 693$ steps |
| $\eta$ | $1$ | $a_\eta \approx 0.54$ | v1.10 default |
| $\chi$ | $0.01$ | $a_\chi \approx -4.60$ | $\gamma = 0.01$ |
| $\bar\varepsilon$ | $0.01$ | $a_\varepsilon \approx -4.60$ | near-pure writes |

For $\mathrm{softplus}$, the inverse is $a = \ln(\mathrm{expm1}(y))$; for $\sigma$, the logit. These are initial conditions, not frozen constants.

### 7.3 Identifiability warnings

1. **$\eta$–$\omega$ degeneracy through retrieval.** Retrieval depends on $(\eta\tilde m, \chi, N)$ only (§8.5), and $\tilde m \propto \omega s$ at small dose. Retrieval loss alone identifies the product $\eta\omega$, not the factors. The split is identified by forgetting: retention $e^{-(\omega s + \Gamma\Delta t)}$ depends on $\omega$ but not $\eta$.
2. **$\omega$–$\Gamma$ degeneracy through dose additivity.** Since the two doses add (§5.4), $\omega$ and $\Gamma$ are separately identifiable **only if $s_t$ and $\Delta t$ vary independently in training data**. If importance correlates with inter-arrival time, the model cannot distinguish "forgotten because unimportant" from "forgotten because old." Episode construction must decorrelate them (§10).
3. **Compensation.** Joint training can inflate $\omega$ or shrink $\Gamma$ to compensate a weak $\eta$. The staged schedule of §10 exists for this reason.
4. **$\bar\varepsilon$ is a reparameterization.** By §6.2, $\varepsilon$ is equivalent to write-triggered diffusion plus a reduced pure write. $\bar\varepsilon$ is identifiable only from the specific coupling between fresh-trace signal mass $a(1-\varepsilon)$ and old-trace retention $(1-a)$; it is not an independent forgetting mechanism.

Log all five parameters and the doses.

---

## 8. Retrieval

### 8.1 Energy and recurrence

$$
E(u; q, W) = -\tfrac12\, u^{\top} W u - \gamma\, u^{\top} q,
\qquad \|u\|_2 = 1 .
$$

Since $\mathrm{tr}\,W = 1$, $W \ne 0$ always, so the smooth part is concave with Hessian $-W \preceq 0$ and is unbounded below off the sphere in every case. Normalization is what makes retrieval well-posed.

$$
u_0 = q;
\qquad
v_n = u_n + \eta\,(W u_n + \gamma q);
\qquad
u_{n+1} = \frac{v_n}{\|v_n\|_2};
\qquad
u^{\ast} = u_N .
$$

**Guard.** If $\|v_n\|_2 < \epsilon_{\mathrm{fall}}$ or $v_n$ is non-finite, set $u_{n+1} = q$ and increment a counter; $\epsilon_{\mathrm{fall}} = 10^{-8}$. The explicit finiteness test is required (NaN defeats the norm comparison). Inputs $q, W$ must be finite and are validated before retrieval. Recommended $N = 20$ (range $[10,30]$); reduce only after the calibration run shows saturation.

Retrieval runs at **frozen $W$**: diffusion is the slow dynamics of the state, retrieval the fast dynamics at a fixed state. No diffusion is applied between the $N$ iterations.

### 8.2 Isotropy lemma and exact W/M equivalence

Write $W = (1-S)J + M$ and $c = (1-S)/d$. For unit $u$:

$$
(I - uu^{\top})\,W u = (I - uu^{\top})\,M u,
\qquad
E(u) = -\frac{1-S}{2d} - \tfrac12 u^{\top} M u - \gamma u^{\top} q .
$$

- The tangential gradient field, the critical points of $E$ on the sphere, and the fixed points of the recurrence depend only on $(M, \gamma q)$.
- **Empty memory returns the cue exactly.** If $W = J$, then $v_n = (1 + \eta(\tfrac1d + \gamma))\,u_n$, so $u_n = q$ for all $n$; for $\gamma > 0$, $u = q$ is the unique minimizer of $E$.
- **Exact reparameterization.** Since $v = (1+\eta c)u + \eta(Mu + \gamma q)$ and the positive scalar cancels in normalization,

$$
\boxed{\;
u^+_{W}(\eta,\gamma) = u^+_{M}(\eta',\gamma),
\qquad
\eta' = \frac{\eta}{1+\eta(1-S)/d},
\qquad
\chi' = \eta'\gamma = \frac{\chi}{1+\eta(1-S)/d} < \chi,
\;}
$$

**trajectory-wise exact**. The entire $W$-form run equals the $M$-form run at $\eta'$ with the same $\gamma$. Both forms are legal; the $M$-form is the recommended implementation (§15). $W$-form calibration rows equal $M$-form rows at $\eta'$ exactly.

### 8.3 Stability and guaranteed energy descent

With $\mu_n = u_n^{\top} W u_n \ge 0$ and $u_n^{\top} q \ge -1$, $r_n = \mu_n + \gamma u_n^{\top}q \ge -\gamma$, so

$$
1 + \eta r_n \ge 1 - \chi > 0,
\qquad
\|v_n\|_2 \ge u_n^{\top} v_n = 1 + \eta r_n \ge 1 - \chi,
\qquad
0 < h_n \le \frac{\eta}{1-\chi},
$$

where $h_n = \eta/(1+\eta r_n)$ is the effective Riemannian step. The guard's norm test is unreachable in exact arithmetic.

**Theorem (energy descent).** Let $W \succeq 0$, $\eta > 0$, $\|u\|_2 = 1$, and $v = u + \eta(Wu + \gamma q) \ne 0$. Then $u^+ = v/\|v\|_2$ satisfies

$$
\boxed{\; E(u^+) \le E(u), \;}
$$

with equality iff $u^+ = u$. This holds for **every** $\eta$, not just small steps.

*Proof.* $f(x) = \tfrac12\|x\|^2 - \eta E(x) = \tfrac12\|x\|^2 + \eta(\tfrac12 x^{\top}Wx + \gamma x^{\top}q)$ is convex (Hessian $I + \eta W \succeq 0$) with $\nabla f(u) = v$, and $u^+ = \arg\max_{\|y\|=1}\langle v, y\rangle$. Convexity gives $f(u^+) \ge f(u) + \langle v, u^+ - u\rangle \ge f(u) + \langle v, u - u\rangle = f(u)$. On the sphere $f = \tfrac12 - \eta E$. Equality forces $u^+ = u$. $\square$

This is the generalized power method (convex maximization on the sphere); the usual small-step caveat for retracted gradient descent does not apply to this map. In the core, $v \ne 0$ is guaranteed by $\|v\| \ge 1-\chi > 0$. The guard fallback $u^+ = q$ is exempt from the theorem and is logged.

**Theorem (square-summability).** Using $\langle v, u^+ - u\rangle = \tfrac12\|v\|\,\|u^+ - u\|^2$ and strong convexity modulus $\lambda_{\min}(I+\eta W) \ge 1$:

$$
f(u_{n+1}) - f(u_n) \;\ge\; \tfrac12\bigl(1 + \|v_n\|\bigr)\,\|\Delta u_n\|^2
\;\ge\; \tfrac{2-\chi}{2}\,\|\Delta u_n\|^2,
$$

$$
\boxed{\;\sum_{n=0}^{N-1}\|\Delta u_n\|_2^2 \;\le\; \frac{2\eta\,\bigl(E(u_0) - E(u_N)\bigr)}{2-\chi} \;\le\; \frac{2\eta\,(E_0 - E_{\min})}{2-\chi} < \infty. \;}
$$

Hence $E$ converges, $\|\Delta u_n\| \to 0$, and every limit point is a critical point of $E$ on the sphere. (The looser $/(1-\chi)$ form also holds.) Unique-limit convergence is not claimed.

**Corollary.** $f(u_n) \ge f(q)$ and $f(q) - f(-q) = 2\eta\gamma > 0$, so $u = -q$ is unreachable: the worst-case margin $1-\chi$ is never saturated.

**Numerical margin note.** With no caps, $\chi = \sigma(a_\chi)$ can approach $1$. In float32 the $10^{-8}$ guard becomes reachable only when $\chi$ rounds to exactly $1.0$ (logit $\approx 17$); at margin $10^{-4}$, $\|v\| \ge 10^{-4} \gg 10^{-8}$ and nothing fires. Log $\chi$ and the margin; if $\chi$ saturates near $1$, treat it as a training pathology ($\chi \gg g$ means "ignore memory," which sits at the cue-only loss floor).

### 8.4 Diagnostics

Monitor per retrieval: $E(u_n)$, $\|u_{n+1}-u_n\|_2$, $\mu_n$, the margin $1+\eta r_n$, $h_n$, and the guard count. **The fraction of steps with $E(u_{n+1}) > E(u_n)$ is a bug detector, not a diagnostic of expected behavior**: it must be $0$ up to float rounding for the guard-free recurrence; a systematic increase indicates non-PSD $W$, a sign error, or a fallback event. Monitor globally: $\chi = \eta\gamma$, the doses $\omega s_t$ and $\Gamma \Delta t$, $S$, $\hat P$, and the half-life $\ln 2/\Gamma$.

### 8.5 Retrieval-gain calibration

Single-trace model: $M \approx \tilde m\, zz^{\top}$, cue at small angle $\varphi$ from $z$, $\psi_n$ the angle closed toward $z$. Define

$$
g = \eta\,\tilde m \quad (\text{read gain}),
\qquad
\chi = \eta\gamma \quad (\text{cue hold}).
$$

Then $h(\tilde m + \gamma) = (g+\chi)/(1+g+\chi) =: \rho_{\mathrm{step}} \in (0,1)$ (monotone, no overshoot), and the exact discrete solution of the linearized recurrence is

$$
\boxed{\;
\frac{\psi_N}{\varphi}
= \frac{g}{g+\chi}\left[\,1 - \left(1 + g + \chi\right)^{-N}\,\right].
\;}
$$

Use this form, not $1 - e^{-\rho_{\mathrm{step}} N}$; the exact decay exponent is $N\ln(1+g+\chi) \ge \rho_{\mathrm{step}} N$. The exact nonlinear planar fixed point obeys $\gamma \sin(\varphi - \theta^{\ast}) = \tfrac{\tilde m}{2}\sin 2\theta^{\ast}$, reducing to $\psi^{\ast} = \tfrac{g}{g+\chi}\varphi$ with $O(\varphi^3)$ error (below $1\%$ at $\varphi = 0.1$ rad).

Worked values at $d = 384$, $\varphi = 0.1$ rad, **one stored direction**, $M$-form ($W$-form equals $M$-form at $\eta'$ exactly, §8.2):

| Setting | $\eta$ | $\chi$ | $N$ | $\tilde m$ | $g$ | $\rho_{\mathrm{step}}N$ | $N\ln(1{+}g{+}\chi)$ | $\psi_N/\varphi$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v1.2 legacy | 0.1 | 0.005 | 20 | 0.05 | 0.005 | 0.20 | 0.20 | 0.09 |
| $\chi \to 0$ | 1 | $\approx 0$ | 20 | 0.05 | 0.05 | 0.95 | 0.98 | 0.62 |
| Init default (§7.2) | 1 | 0.01 | 20 | 0.05 | 0.05 | 1.13 | 1.17 | 0.57 |
| Init, older item | 1 | 0.01 | 20 | 0.01 | 0.01 | 0.39 | 0.40 | 0.16 |
| $\eta = 10$ | 10 | 0.05 | 20 | 0.01 | 0.10 | 2.61 | 2.80 | 0.63 |

**Multi-trace scope.** The table is the $T=1$ case only. With $T$ comparable-mass traces, the equilibrium angle from the target is $\theta_{\mathrm{floor}}$ (§6.4); calibration must sweep $\varphi$ below, near, and above the floor and report negative-closure cases. Arm R is not floor-free: its floor is $\alpha\sqrt{(T-1)/d}/(\alpha+\gamma)$. Gain requires $N \ln(1+g+\chi) \gtrsim 1$ at the smallest mass to be retrieved; diffusion lowers $\tilde m$ with age as $e^{-\Gamma \cdot \mathrm{age}}$, so calibration must sweep age as well as importance. The two-arm calibration run (§17.3) is mandatory before training.

---

## 9. Training objective

Primary loss, for unit target $z_{\mathrm{target}}$:

$$
L_{\mathrm{retrieval}} = 1 - \left| u^{\ast\top} z_{\mathrm{target}} \right| .
$$

Sign-blind, matching the memory operator (not the retrieval map; §3.2). $L = 1-|c|$, with $c = u^{\ast\top}z_{\mathrm{target}}$, has a concave kink at $c = 0$: its derivative with respect to $c$ is $+1$ just left of $0$ and $-1$ just right of $0$, and its Clarke (generalized) subdifferential at $0$ is $[-1,1]$. (The convex-analysis subdifferential of a concave kink is empty, so "subgradient" here means Clarke.) The loss retains full-strength gradient signal arbitrarily close to orthogonality — unlike $1-c^2$, whose gradient $-2c$ vanishes there. That claim rests on the one-sided slopes, not on the set containing $0$. Autograd of $|c|$ returns $0$ at exactly $c=0$, a valid Clarke element but carrying no signal; use a custom subgradient or the smooth surrogate if exact zeros occur.

Smooth surrogate:

$$
L_{\mathrm{smooth\text{-}abs}} = \sqrt{1+\epsilon_L} - \sqrt{c^2 + \epsilon_L},
\qquad
c = \mathrm{clip}(u^{\ast\top} z_{\mathrm{target}}, -1, 1),
\qquad
\epsilon_L = 10^{-8},
$$

with dead zone $|c| \lesssim \sqrt{\epsilon_L} = 10^{-4}$ and gradient $-c/\sqrt{c^2+\epsilon_L}$. Clamping prevents tiny numerical overshoots from making the surrogate negative.

Ablations: $1-|\cos|$, smooth surrogate, $1-\cos^2$, oriented $1-\cos$ (only if cues reliably carry sign). Reference floor: the cue-only loss $1 - |q^{\top} z_{\mathrm{target}}|$, which by §8.2 is also the exact output of retrieval against an empty memory. Learned retrieval must beat it **and the centroid baseline** (§17.1).

---

## 10. Training episodes

Each episode contains inputs $z_{1:T}$, scores $s_{1:T}$, timestamps $t_{1:T}$ (gaps $\Delta t_i$), a cue $q$ at time $t_q$, and a target.

Procedure:

1. Reset $W = J$, $S = 0$ at episode start.
2. For each event: diffuse by $\Delta t_i$ (§5), then write (§6).
3. Diffuse to $t_q$; run retrieval at frozen $W$ (§8).
4. Compute loss; backpropagate through diffusion, writing, and retrieval; update $\theta$.

**Data requirement (from §7.3):** vary $s_t$ and $\Delta t$ **independently**, or $\omega$ and $\Gamma$ are not identifiable. Also vary target position, cue completeness, recency, importance, and competing inputs; use separate train/eval episodes.

**Staged schedule:**

- **Stage A:** freeze $\{\omega, \Gamma, \bar\varepsilon\}$ at init; learn $\{\eta, \chi\}$.
- **Stage B:** unfreeze all five. Compare against Stage A to test whether learning the write/forget parameters helps.

Start with short sequences and small $N$; increase after training is stable.

---

## 11. Optional reinforcement

Enable only after basic retrieval is reliable. Given $u^{\ast}$, let $\mu = u^{\ast\top} W u^{\ast} \in [0,1)$ (baseline $1/d$; strict upper bound by §3.1). With retrieved-content importance $s_{\mathrm{ret}} \ge 0$ and confidence $C \in [0,1]$, define the target support as the support a fresh pure write of importance $s_{\mathrm{ret}}$ would have against an empty memory:

$$
a_{\mathrm{ret}} = 1 - e^{-\omega s_{\mathrm{ret}}},
\qquad
\tau = \frac1d + \left(1 - \frac1d\right) a_{\mathrm{ret}} \in \left[\tfrac1d,\,1\right),
$$

with $\tau = 1/d$ attained exactly at $s_{\mathrm{ret}}=0$. (At $s_{\mathrm{ret}}=0$, $\beta>0$ only if $\mu < 1/d$.)

$$
\boxed{\;
\beta =
\begin{cases}
C\,\min\!\left(\bar\beta,\ \dfrac{\tau-\mu}{1-\mu}\right), & \mu < \tau,\\[8pt]
0, & \mu \ge \tau,
\end{cases}
\qquad \bar\beta = \sigma(a_\beta) \text{ learnable},
\;}
$$

then $W \leftarrow (1-\beta)W + \beta\, u^{\ast}u^{\ast\top}$, $S \leftarrow (1-\beta)S + \beta$.

Verified properties: $\beta \in [0, \bar\beta] \subset [0,1)$; the first branch has $\mu < \tau < 1$, so the denominator is safe; the second branch never divides. **In exact arithmetic $\mu < 1$ for finite scores and finitely many operations** ($a_t < 1$, $\tau < 1$, and reinforcement raises $\mu$ toward $\tau$ but never above it). $\mu = 1$ is a float32 artifact: the exact $1-\mu$ crosses $2^{-25}$ at write 25 for $a=0.5$, but the *computed* $\mu$ lands within float noise of $1$ (below, equal, or above). It occurs in two cases: (i) repeated pure writes of one direction make $W$ round toward a pure state; (ii) $a_{\mathrm{ret}}$ (hence $\tau$) rounds to $1$ for very large dose $\omega s_{\mathrm{ret}}$, and repeated reinforcement then drives $\mu$ toward $1$. In all cases the correct value is $\beta = 0$ whenever $\mu \ge \tau$, which the branch returns. $\mu' = (1-\beta)\mu + \beta$, reaching $\tau$ exactly at $C = 1$ with the cap slack; PSD and trace one are preserved. The hard threshold $C_{\min}$ of v1.10 is removed — confidence scales $\beta$ smoothly. **Implementation: use the masked-denominator form of §15.**

**Purity is not monotone under reinforcement.** The exact change is

$$
P' - P = 2\beta(\mu - P) + \beta^2\,(1 + P - 2\mu),
$$

which can be negative (reinforcing a direction orthogonal to a near-pure memory mixes it). Reinforcement raises support along $u^{\ast}$, not global sharpness. Repeated confident retrievals concentrate mass on frequently retrieved directions in competition with diffusion; log the reinforcement share of $S$ (track $S_{\mathrm{write}}$ and $S_{\mathrm{reinf}}$ with the same linear recursions; report $S_{\mathrm{reinf}}/S$). Reinforcement reinforces a direction, not necessarily a correct memory; confidence gating matters.

---

## 12. Optional decoder

Retrieval features:

$$
r = [\, u^{\ast},\ q,\ u^{\ast} \odot q,\ E(u^{\ast}),\ \mu,\ S,\ \hat P,\ \text{guard flag} \,],
$$

optionally $\lambda_{\max}(M)$, eigengap, effective rank, spectral entropy, and trace age.

### 12.1 Normalization

Let $D = D_\psi(r)$ be the raw decoder output. Max-floor normalization:

$$
\hat z = \frac{D}{\max(\|D\|_2,\ \epsilon_{\mathrm{dec}})},
\qquad
\epsilon_{\mathrm{dec}} = 10^{-6} \quad (\text{range } [10^{-6}, 10^{-4}]).
$$

$\|\hat z\|_2 \le 1$, with exact unit norm iff $\|D\|_2 \ge \epsilon_{\mathrm{dec}}$. The map is continuous but not differentiable at the floor (the radial derivative jumps from $1/\epsilon_{\mathrm{dec}}$ to $0$). Smooth alternative:

$$
\hat z_{\mathrm{smooth}} = \frac{D}{\sqrt{\|D\|_2^2 + \epsilon_{\mathrm{dec}}^2}},
\qquad
\|\hat z_{\mathrm{smooth}}\|_2 < 1 \text{ for finite } D.
$$

Normalization of either kind is required: for unnormalized output, $1-(\hat z^{\top}z_{\mathrm{target}})^2$ is unbounded below by growing $\|\hat z\|$.

### 12.2 Jacobian bounds

**Max-floor, away from the kink.** For $\|D\| > \epsilon_{\mathrm{dec}}$: $\hat z = D/\|D\|$, so

$$
J = \frac{I - \hat z \hat z^{\top}}{\|D\|},
\qquad
\text{eigenvalues } \{0,\ 1/\|D\|\},
\qquad
\|J\|_2 = \frac{1}{\|D\|}.
$$

For $\|D\| < \epsilon_{\mathrm{dec}}$: $J = I/\epsilon_{\mathrm{dec}}$, $\|J\|_2 = 1/\epsilon_{\mathrm{dec}}$. Hence $\|J\|_2 = 1/\max(\|D\|, \epsilon_{\mathrm{dec}})$ away from the kink.

**Smooth.** With $s = \sqrt{\|D\|^2 + \epsilon_{\mathrm{dec}}^2}$:

$$
J = \frac{I}{s} - \frac{D D^{\top}}{s^3},
\qquad
\text{eigenvalues } \frac{1}{s}\ (\perp D),\quad \frac{\epsilon_{\mathrm{dec}}^2}{s^3}\ (\parallel D).
$$

Since $\epsilon_{\mathrm{dec}}^2/s^3 \le 1/s$, $\|J\|_2 = 1/s$. Both forms are bounded by $1/\epsilon_{\mathrm{dec}}$: $10^6$ at the default, $10^4$ at $\epsilon_{\mathrm{dec}} = 10^{-4}$. Monitor decoder gradient norms near zero output; if unstable, increase $\epsilon_{\mathrm{dec}}$, or use gradient clipping or learning-rate warmup. **Weight decay is not a remedy:** for a final linear layer, rescaling its weights by $c$ leaves $\hat z$ unchanged for max-floor normalization above the floor (and approximately for the smooth form when $\|D\| \gg \epsilon_{\mathrm{dec}}$), while multiplying the weight gradient by $1/c$; shrinking weights increases gradients and the effective step size. The smooth denominator removes the kink but not the worst-case gradient scale.

### 12.3 Loss and float32 behavior

Default semantic reconstruction loss (oriented on purpose; the decoder sees $q$):

$$
L_{\mathrm{meaning}} = 1 - \hat z^{\top} z_{\mathrm{target}}.
$$

If cues do not reliably carry orientation, use $1 - |\hat z^{\top} z_{\mathrm{target}}|$ or the §9 surrogate with $c = \mathrm{clip}(\hat z^{\top}z_{\mathrm{target}}, -1, 1)$. Because $\|\hat z\|_2$ may be below $1$, this $c$ is a clamped inner product, not necessarily a cosine; subunit outputs cannot reach exactly zero surrogate loss.

For the smooth form with $D = \alpha z_{\mathrm{target}}$, $\alpha > 0$:

$$
L_{\mathrm{meaning}} = 1 - \frac{\alpha}{\sqrt{\alpha^2+\epsilon_{\mathrm{dec}}^2}}
= \frac{\epsilon_{\mathrm{dec}}^2}{s\,(s+\alpha)},
\qquad s = \sqrt{\alpha^2+\epsilon_{\mathrm{dec}}^2},
$$

strictly positive for finite $\alpha$; the right-hand form is the numerically stable evaluation.

**Float32 behavior (a transition band, not a sharp cutoff).** For $\alpha \gg \epsilon_{\mathrm{dec}}$, $L \approx \epsilon_{\mathrm{dec}}^2/(2\alpha^2)$. Floats just below $1$ are spaced $2^{-24}$, so a ratio $\alpha/s$ whose true value lies within $2^{-25}$ of $1$ rounds to $1$ (ties-to-even), and the naive evaluation $1-\alpha/s$ returns exactly $0$. With an exactly rounded ratio this happens when

$$
\frac{\epsilon_{\mathrm{dec}}^2}{2\alpha^2} < 2^{-25}
\iff
\alpha > 2^{12}\,\epsilon_{\mathrm{dec}} = 4096\,\epsilon_{\mathrm{dec}} \approx 4.1\times 10^{-3}.
$$

Intermediate roundings of $\alpha^2$, the sum, the square root, and the division widen this into a band.

Hard guarantees (round-to-nearest, no overflow/underflow):

- **Absorption** of $\epsilon_{\mathrm{dec}}^2$ into $\alpha^2$ requires $\epsilon_{\mathrm{dec}}^2/\alpha^2$ below the relative half-ulp of $\alpha^2$, which lies in $(2^{-25}, 2^{-24}]$ depending on the mantissa. Hence absorption is impossible for $\alpha < 2^{12}\epsilon_{\mathrm{dec}}$, mantissa/tie-dependent at $\alpha = 2^{12}\epsilon_{\mathrm{dec}}$, and guaranteed for $\alpha \ge 2^{12.5}\epsilon_{\mathrm{dec}} \approx 1.414\cdot 2^{12}\epsilon_{\mathrm{dec}}$.
- Once absorption occurs, $\mathrm{fl}(\alpha^2+\epsilon_{\mathrm{dec}}^2) = \mathrm{fl}(\alpha^2)$ and $\sqrt{\mathrm{fl}(\alpha^2)}$ rounds back to $\alpha$, so the computed ratio is exactly $1$ and the naive loss is exactly $0$. Thus naive zero is guaranteed for $\alpha \ge 2^{12.5}\epsilon_{\mathrm{dec}}$.
- Below that, zero loss can occur **without** absorption, so the zero-loss threshold is not the same condition as absorption.

Illustrative simulated float32 fractions at $\epsilon_{\mathrm{dec}} = 10^{-6}$ (one log-uniform grid, round-to-nearest; $\rho = \alpha/(2^{12}\epsilon_{\mathrm{dec}})$; exact fractions depend on the grid, `libm`, and rounding mode):

| $\rho$ | naive loss exactly $0$ | $\alpha^2+\epsilon_{\mathrm{dec}}^2 \to \alpha^2$ |
|---|---:|---:|
| $[0.25,\,0.7)$ | 1% | 0% |
| $[0.7,\,0.95)$ | 36% | 0% |
| $[0.95,\,1.05)$ | 54% | 0% |
| $[1.05,\,1.45)$ | 81% | 22% |
| $[1.45,\,4)$ | 100% | 100% |

Treat $2^{12}\epsilon_{\mathrm{dec}}$ as accurate to about a factor $\sqrt2$ in $\alpha$. The stable form retains the tiny positive loss throughout. If exact zero loss or exact unit-norm outputs are required for finite outputs, prefer max-floor normalization.

Evaluate memory retrieval separately with §9; a strong decoder can hide weak memory.

---

## 13. Blending and vagueness as gain loss

For any unit $u$: $u^{\top} W u = \frac{1-S}{d} + \sum_i \tilde\alpha_i (z_i^{\top} u)^2$. Overlapping traces jointly support blend directions. The equal-weight example (an idealized $S=1$ state; reachable states have $S<1$, §3.1): for $W = \tfrac12 z_1z_1^{\top} + \tfrac12 z_2z_2^{\top}$ with $\omega_{12} = z_1^{\top}z_2$, $|\omega_{12}| < 1$, the in-span eigenvalues are $\nu_\pm = (1 \pm \omega_{12})/2$ with eigenvectors $\propto z_1 \pm z_2$; at $\omega_{12} = \pm 1$ the span collapses and the formulas give $\{1, 0\}$. Blending, not separation.

**Theorem (pure gain decay).** Diffusion acts on the signal matrix as $M \leftarrow e^{-\Gamma t}M$. Therefore **every relative spectral quantity of $M$ is invariant**: eigenvalue ratios, normalized gaps $(\nu_+-\nu_-)/(\nu_+-1/d)$, effective rank, spectral entropy of $M/S$. Absolute gaps decay by $e^{-\Gamma t}$; the relative shape never changes. There is no spectral reshaping and no literal merging of directions.

**What actually fades.** Fixed points depend on $(M/\gamma, q)$. For an isolated trace, equilibrium closure is $\tilde m(t)/(\tilde m(t)+\gamma)$ with $\tilde m(t)=\tilde m_0 e^{-\Gamma t}$, decreasing continuously to $0$ (cue-return). The cue-referenced **half-closure time** is

$$
\boxed{\;
t_{1/2\mathrm{cl}} =
\begin{cases}
0, & \tilde m_0 \le \gamma,\\[4pt]
\dfrac{1}{\Gamma}\ln\dfrac{\tilde m_0}{\gamma}, & \tilde m_0 > \gamma,
\end{cases}
\;}
$$

the time at which closure equals $\tfrac12$ (i.e. $\tilde m = \gamma$). No arbitrary tolerance is involved. (Worked: $\tilde m_0=0.05$, $\gamma=0.01$, $\Gamma=10^{-3}$ → 1609 steps, closure $0.500$.)

"Vagueness" is this gain loss relative to the cue, plus the cross-talk floor of §6.4 — not a change in stored spectral shape. As $W \to J$, retrieval $\to$ cue-return exactly (§8.2).

**What diffusion does not fix.** Because eigenvectors are preserved and all gaps contract by the same factor, a dominant shared-mean direction on anisotropic embeddings stays dominant relative to everything else. Diffusion removes capture only by removing all gain; it is not a cure for capture. Centering is the first variant to try if Arm A fails (§17.6), and Arm A of the calibration remains mandatory.

---

## 14. Guaranteed properties

Given unit $z_t, q$, finite $s_t \ge 0$, finite $\Delta t \ge 0$, and the parameterizations of §7 (so $a_t, p, \varepsilon_t, \beta \in [0,1)$, $\chi \in (0,1)$, $\eta, \omega, \Gamma > 0$), in exact arithmetic:

$$
W \succeq 0,\quad \mathrm{tr}(W) = 1,\quad
W = (1-S)J + M,\quad M \succeq 0,\quad S = \mathrm{tr}(M) \in [0,1),
$$

with $\lambda_{\min}(W) \ge (1-S)/d > 0$ and $\lambda_{\max}(W) \le 1-(1-S)(1-1/d) < 1$; the contribution formulas of §6.4; the mass bounds $\tilde\alpha \le z^{\top}Mz \le \lambda_{\max}(M) \le S < 1$; the diffusion laws of §5.2 (spectral contraction, $e^{-\Gamma t}$ Frobenius, $e^{-2\Gamma t}$ purity contrast, entropy monotonicity, semigroup composition, irreversibility, **relative spectral invariance**); the steady states $S^{\ast}_{\mathrm{pre}}, S^{\ast}_{\mathrm{post}}$; dose additivity $e^{-(\omega s + \Gamma\Delta t)}$; the exact impure-write decomposition (§6.2); the exact $W/M$ reparameterization (§8.2); and

$$
\|v_n\|_2 \ge 1-\chi > 0,
\qquad
1 + \eta r_n \ge 1-\chi > 0,
\qquad
0 < h_n \le \frac{\eta}{1-\chi}.
$$

**Retrieval guarantees (guard-free recurrence):** $E(u_{n+1}) \le E(u_n)$ for every $\eta > 0$; $\sum_n \|\Delta u_n\|^2 \le 2\eta(E_0 - E_N)/(2-\chi) < \infty$; $E$ converges, $\|\Delta u_n\| \to 0$, and every limit point is a critical point of $E$ on the sphere; $u = -q$ is unreachable; fixed points depend only on $(M, \gamma q)$; empty memory returns $q$ exactly. The guard fallback $u^+ = q$ is exempt and is logged.

Reinforcement preserves all state invariants, is computed by the branch of §11 (no division at $\mu \ge \tau$), and attains $\mu' = \tau$ at $C=1$ with cap slack.

**Not guaranteed:** exact recall; unique basins, global optimality, or which critical point is selected; pattern separation; meaningful blending for every combination; useful gain at the initial parameters for a given memory; angular resolution below the cross-talk floor; avoidance of dominant-direction capture on anisotropic embeddings; separate identifiability of $\omega$ and $\Gamma$ without decorrelated $(s, \Delta t)$ data; a float32-safe margin when $\chi$ rounds to $1$; monotone $S$ or monotone purity under writes/reinforcement. These must be checked empirically.

---

## 15. Implementation rules

1. Normalize all $z_t, q$; validate finiteness of $z_t, q, s_t, W$ before use. Clamp $s_t \leftarrow \max(s_t, 0)$.
2. Initialize $W = J$, $S = 0$. Never initialize $W = 0$.
3. **Preferred representation:** store $(M, S)$, not $W$. All updates act on $M$ (§4.2); retrieval force $Mu + \gamma q$ gives identical fixed points and, by the exact reparameterization of §8.2, the same trajectory as the $W$-form at $\eta'$. If the $W$-form is used, the isotropic term is included automatically.
4. **Lazy diffusion clock.** In the signal representation, diffusion is scalar multiplication: $M \leftarrow e^{-\Gamma\Delta t}M$, $S \leftarrow e^{-\Gamma\Delta t}S$. Accumulate elapsed time and apply the factor lazily at the next read/write; never materialize $J$.
5. Enforce the ordering convention: diffuse-to-now, then write, then reinforce (§4.3). The commutator $ap(T-J)$ is nonzero; order is part of the model.
6. Symmetrize after updates; recompute $S \leftarrow \mathrm{tr}(M)$ if drift appears; verify $\mathrm{tr}(W) = 1$ in the $W$-form.
7. Explicit off-flags for ablations: $\varepsilon \equiv 0$ (pure writes) and diffusion-off ($\Gamma$-bypass), since $\sigma$ and $\mathrm{softplus}$ never reach $0$.
8. Use separate numerical guards $\epsilon_{\mathrm{fall}} = 10^{-8}$ (retrieval) and $\epsilon_{\mathrm{dec}} = 10^{-6}$ (decoder). Core guard count must be $0$ unless non-finite values appear.
9. **Dose weights:** compute $a = -\mathrm{expm1}(-\omega s)$ and $p = -\mathrm{expm1}(-\Gamma\Delta t)$; compute retention as $\exp(-(\omega s + \Gamma\Delta t))$; use $\ln(\mathrm{expm1}(y))$ for the softplus inverse at init. Float32 $1-e^{-x}$ returns $0$ at $x=10^{-9}$ and has ~40% error at $10^{-7}$.
10. Log $\chi$, margin $1-\chi$, $h_n$, doses $\omega s_t$ and $\Gamma\Delta t$, $S$, $\hat P$, half-life $\ln 2/\Gamma$, and all five parameters. **The energy-increase fraction must be $0$ up to float rounding** (§8.4); treat any systematic increase as a bug.
11. Run both calibration arms (§17.3) and report the cue-only **and centroid** baselines before the first training run.
12. Keep $\gamma = \chi/\eta$ at or below the scale of isolated-trace masses as a sanity heuristic ($\gamma \lesssim a(1-\varepsilon)$ at typical $s$); the parameterization does not enforce this — monitor it.
13. Use the sign-blind loss for memory-only evaluation; decoder rules per §12.
14. When testing steady states, measure $S$ at the labeled instant (just after diffusion or just after write) and compare against the matching formula of §5.3; for a query a gap $g$ after the last write in the write-only periodic regime, compare against $S^{\ast}_{\mathrm{post}}e^{-\Gamma g}$.
15. **Reinforcement (autodiff-safe):** use the branch of §11. In frameworks that evaluate both branches (e.g. `torch.where`), do not form the raw fraction $(\tau-\mu)/(1-\mu)$ and select afterward; the unselected division at $\mu=1$ produces NaN gradients. Mask the denominator before dividing:
$$
(\tau-\mu)_+ = \max(0,\ \tau-\mu),
\qquad
d_\mu = \begin{cases} 1-\mu, & \mu < \tau,\\ 1, & \mu \ge \tau, \end{cases}
\qquad
\beta = C\,\min\!\left(\bar\beta,\ \frac{(\tau-\mu)_+}{d_\mu}\right).
$$
This is exactly equivalent to the piecewise definition, never divides by zero, and is safe under both-branch evaluation.

---

## 16. Computational complexity

| Operation | Dense | Factored / signal form |
|---|---:|---:|
| Store state | $O(d^2)$ | $O(Td + T)$ |
| Write one item | $O(d^2)$ | $O(d+T)$: append $z$, rescale $T$ coefficients; $O(Td)$ if a Gram cache is maintained for purity |
| Diffusion step | $O(d^2)$ | $O(1)$ lazy (scalar clock), $O(T)$ applied to coefficients |
| Matvec $Mu$ | $O(d^2)$ | $O(Td)$, preferred when $T < d$ |
| Retrieval ($N$ steps) | $O(Nd^2)$ | $O(NTd)$ |
| Purity $\hat P$ | $O(d^2)$ | $O(T^2)$ with cached Gram; $O(T^2 d)$ from scratch; incremental Gram maintenance $O(Td)$ per write |
| Full eigendecomposition / entropy | $O(d^3)$ | diagnostics only |

Purity in factored form uses $\|M\|_F^2 = \sum_{ij}\tilde\alpha_i\tilde\alpha_j (z_i^{\top}z_j)^2$.

Backprop: gradients through diffusion scale by $e^{-\Gamma\Delta t}$, with

$$
\frac{\partial W_{\mathrm{new}}}{\partial\Gamma} = -\Delta t\, e^{-\Gamma\Delta t}\,(W_{\mathrm{old}} - J),
\qquad
\frac{\partial M_{\mathrm{new}}}{\partial\Gamma} = -\Delta t\, e^{-\Gamma\Delta t}\,M_{\mathrm{old}};
$$

through the write dose, $\partial a_t/\partial\omega = s_t e^{-\omega s_t}$; and $\partial\gamma/\partial\chi = 1/\eta$, $\partial\gamma/\partial\eta = -\chi/\eta^2$. The state sequence is determined by scalars $\{a_i, \varepsilon_i, p_i, \beta_k\}$ and stored $\{z_i\}$; with recomputation, activation memory is $O(Td + Nd)$, not $O((T{+}N)d^2)$. At $d = 384$, $d^2 = 147{,}456$: a dense first implementation is practical.

---

## 17. Evaluation plan

### 17.1 Baselines

0. **Cue-only** $u^{\ast} = q$ (also the exact empty-memory output).
1. **Top-eigenvector / centroid readout:** $u = v_1$ sign-aligned to the target; report $|v_1^{\top}z_{\mathrm{target}}|$. Cue-conditioned retrieval must beat **both** cue-only and centroid; gain over cue-only that merely matches the centroid baseline is capture, not retrieval.
2. **Diffusion-off at matched init** — write-driven dilution with the same dose map; isolates the contribution of time-based forgetting.
3. Nearest-neighbor over stored embeddings (explicit records).
4. Simple linear-memory baseline.
5. Constant dose vs importance-dosed writes; fixed vs learned parameters.

### 17.2 Retrieval experiments

Full and partial cues; absent/unrelated memory (must return $q$ exactly at $W=J$); hard neighbors; recency and importance buckets; **age buckets at fixed importance and importance buckets at fixed age** (tests dose separability); irregular-$\Delta t$ generalization; empirical retention curves vs the predicted half-life $\ln 2/\Gamma$; steady states measured at the labeled instants or query gaps (§5.3); **$\varphi$ sweeps below, near, and above $\theta_{\mathrm{floor}}$ (§6.4), reporting negative-closure cases**; half-closure times vs $t_{1/2\mathrm{cl}}$ (§13); sensitivity to initialization; unseen sequences. Separate train/eval episodes.

### 17.3 Calibration run before training

Fixed parameters at §7.2 init, $d = 384$, cues at known angles, varied recency/importance/age. Two arms:

- **Arm R (random):** random unit $z_i$; support shift $\approx (S-\tilde\alpha)/d$, **and** tangential floor $\theta_{\mathrm{floor}} \approx \alpha\sqrt{(T-1)/d}/(\alpha+\gamma)$. Not floor-free.
- **Arm A (anisotropic):** clustered embeddings with a shared mean component (or real model embeddings); tests dominant-direction capture.

Report per bucket, per arm: closed angle fraction $\psi_N/\varphi$; loss improvement over cue-only **and over centroid**; retrieved $|\cos|$ vs centroid $|\cos|$; $\rho_{\mathrm{step}}N$ and $N\ln(1+g+\chi)$; $h_n$ and the energy-increase bug counter; $\lambda_{\max}(M)$, $\lambda_{\max}(M)/S$, effective rank; $\theta_{\mathrm{floor}}$; **capture rate** $\Pr[\,|u^{\ast\top}v_1| > |u^{\ast\top}z_{\mathrm{target}}|\,]$ with $v_1$ the top eigenvector. If retrieval is near-identity everywhere, raise $\eta$ or $N$ or lower $\chi$ before training. If closure is negative at small $\varphi$, the cause is the cross-talk floor, not gain — do not raise $\eta$. If Arm A captures while Arm R is healthy, the failure is anisotropy; do not expect diffusion to fix it (§13). Diffusion removes capture only by removing all gain.

### 17.4 Ablations

| Component | Variants |
|---|---|
| Forgetting | Diffusion on / off; write-dilution only |
| Write dose | $1-e^{-\omega s}$; rational $\kappa_{\max} s/(s+1)$; constant |
| Write/diffusion coupling | $\varepsilon\equiv 0$; $\bar\varepsilon(1-a)$; purity-coupled (§6.2) — a reparameterization ablation |
| Cue parameterization | $\gamma = \chi/\eta$ (core); free capped $\gamma$ |
| Schedule | Staged vs joint |
| Retrieval loss | $1-\lvert\cos\rvert$; smooth surrogate; $1-\cos^2$; oriented |
| Reinforcement | Off / on |
| Decoder normalization | Max-floor / smooth |
| Embedding geometry | Raw / mean-centered (§17.6) |

### 17.5 Diagnostics

Energy and the energy-increase bug counter; state change; $\mu_n$; guard count; $S$ and its write/reinforcement split; $\hat P$; all five parameters and saturation of $\chi$; doses and half-life; $\lambda_{\max}$, eigengap, effective rank, spectral entropy; $\theta_{\mathrm{floor}}$; capture rate; centroid-baseline comparison. Spectral quantities are hypotheses, not established measures of semantic ambiguity.

### 17.6 Deferred extensions

- **Centering/whitening — promoted to the first variant to try if Arm A fails.** Preserves all invariants (centered renormalized vectors still give PSD trace-one targets) but changes what a stored direction means and breaks direct compatibility with the external embedding space unless the decoder compensates. Still not core.
- **Delta-rule-style selective writes** — the principled answer to pattern separation. A PSD-preserving congruence form exists but is not specified here.
- **Log-dose softmax readout (untested):** $u \leftarrow \sum_i \mathrm{softmax}(\beta\, z_i^{\top}u + \ln\tilde\alpha_i)\,z_i$ at $O(Td)$, exploiting dose additivity ($\ln\tilde\alpha_i$ is additive in doses). Modern-Hopfield-style; not core before validation.
- Contrastive objectives (use $|\cos|$), learned importance functions, reinforcement confidence models.

---

## 18. Core mathematical summary

**State.** $W \succeq 0$, $\mathrm{tr}\,W = 1$; $W = (1-S)J + M$, $J = I/d$; init $W = J$, $S = 0$; finite reachable states have $S<1$, $W \succ 0$.

**Primitive.** $W \leftarrow (1-a)W + aT$; $\;S \leftarrow (1-a)S + a\,s(T)$.

**Diffuse (mandatory, first at each event):**

$$
W \leftarrow e^{-\Gamma\Delta t} W + (1 - e^{-\Gamma\Delta t})\,J,
\qquad
S \leftarrow e^{-\Gamma\Delta t} S .
$$

**Write:**

$$
a_t = 1 - e^{-\omega s_t},
\qquad
\varepsilon_t = \bar\varepsilon\,(1-a_t),
\qquad
W \leftarrow (1-a_t)W + a_t\left[(1-\varepsilon_t) z_tz_t^{\top} + \varepsilon_t J\right],
$$

equivalently diffuse by $p'=\frac{a_t\varepsilon_t}{1-a_t+a_t\varepsilon_t}$ then pure-write with weight $a_t(1-\varepsilon_t)$.

**Retrieve** (frozen $W$, $\gamma = \chi/\eta$):

$$
u_0 = q,
\qquad
v_n = u_n + \eta(Wu_n + \gamma q),
\qquad
u_{n+1} = v_n / \|v_n\|_2,
\qquad
u^{\ast} = u_N,
$$

guard $\|v_n\|_2 < \epsilon_{\mathrm{fall}}$ or non-finite $\Rightarrow u_{n+1} = q$; norm test unreachable since $\|v_n\|_2 \ge 1-\chi > 0$. **Energy descent:** $E(u_{n+1}) \le E(u_n)$ for every $\eta$; $\sum\|\Delta u_n\|^2 \le 2\eta(E_0-E_N)/(2-\chi)$.

**Train.** $L = 1 - |u^{\ast\top} z_{\mathrm{target}}|$; learn $\theta = \{\omega, \Gamma, \eta, \chi, \bar\varepsilon\}$ in two stages: $\{\eta,\chi\}$ first, then all.

**Calibration.** $\dfrac{\psi_N}{\varphi} = \dfrac{g}{g+\chi}\left[1 - (1+g+\chi)^{-N}\right]$, $g = \eta\tilde m$ ($T=1$); multi-trace floor $\theta_{\mathrm{floor}} \approx \sqrt{\sum_{j\ne i}\tilde\alpha_j^2/d}/(\tilde m+\gamma)$.

**Vagueness.** Relative spectral shape of $M$ is invariant; closure $=\tilde m(t)/(\tilde m(t)+\gamma)$; $t_{1/2\mathrm{cl}} = \max(0, \Gamma^{-1}\ln(\tilde m_0/\gamma))$ (piecewise, §13).

**Optional reinforce.** $\tau = \tfrac1d + (1-\tfrac1d)(1-e^{-\omega s_{\mathrm{ret}}})$; $\beta = C\min(\bar\beta, \tfrac{\tau-\mu}{1-\mu})$ if $\mu<\tau$, else $0$ (masked-denominator form in §15); $W \leftarrow (1-\beta)W + \beta u^{\ast}u^{\ast\top}$.

---

## 19. Central research hypothesis

HopMem tests whether a trace-one PSD memory, whose every operation is one convex mixture and whose forgetting is a mandatory learnable-rate depolarizing channel, preserves useful semantic structure better and more interpretably than write-driven dilution:

1. The memory retains useful semantic direction (beats cue-only **and centroid** on partial cues).
2. Learned $\{\omega,\Gamma,\eta,\chi,\bar\varepsilon\}$ improve over fixed values.
3. Retrieval does not collapse to the dominant/mean direction on anisotropic embeddings (Arm A); apparent gain equal to the centroid baseline is counted as failure.
4. Time-based forgetting is identifiable and interpretable — empirical retention matches the half-life $\ln 2/\Gamma$ and the steady states at the labeled instants or query gaps (§5.3), with $\omega$ and $\Gamma$ separately recovered when $(s,\Delta t)$ are decorrelated.
5. Vagueness emerges as predictable gain loss: equilibrium closure follows $\tilde m(t)/(\tilde m(t)+\gamma)$ with half-closure time $t_{1/2\mathrm{cl}}$; relative spectral shape is invariant (theorem), so any observed "merging" beyond this is a cross-talk or capture effect, to be diagnosed separately.

**Conceptual status.** The density-matrix and channel language is a real-symmetric analogue and a mathematical convenience, not a claim of quantum physics. The model is a classical second-moment associative memory with isotropic decay — the state class of gated linear-attention associative memories with uniform decay. The missing ingredient relative to delta-rule memories is selective replacement, which is exactly the pattern-separation gap. Its distinctive claims are the single convex primitive with two target types, dose additivity, provable energy descent, and predictable gain-loss vagueness. Its limits — no pattern separation, the cross-talk floor, anisotropic capture, and $\omega$–$\Gamma$ identifiability only under decorrelated $(s,\Delta t)$ — are non-guarantees and are tested in §17. Only after the core is validated should the system add centering, delta-rule writes, softmax readouts, contrastive objectives, learned importance, reinforcement, or decoders.

---

## 20. Supersession: cleared prior claims

This document supersedes v1.10 through v2.0.3. Where any earlier text conflicts, this document governs. The following prior claims are **withdrawn** and must not be used:

1. "Monotone energy decrease holds only for sufficiently small steps / is not guaranteed." → Energy descent is a theorem for every $\eta$ (§8.3); the energy-increase fraction is a bug detector.
2. "Crosstalk $\approx (S-\tilde\alpha)/d$ is the retrieval limit." → That is the support shift; the angular limit is the tangential floor $\theta_{\mathrm{floor}}$ (§6.4), independent of $\eta, N$.
3. "Vagueness = spectral gap closure / spectral merging," and the arbitrary-tolerance $t_{\mathrm{merge}}$. → Relative spectral shape is exactly invariant; vagueness is gain loss relative to the cue, with half-closure time $t_{1/2\mathrm{cl}}$ (§13).
4. "Impure blur applies only to the injected target, not the old state, and is distinct from diffusion." → Exact decomposition: impure write = diffusion($p'$) + pure write($a_2$) (§6.2).
5. "$W$-form vs $M$-form differ only by a bounded $h$ decrease." → Exact trajectory-level reparameterization $\eta' = \eta/(1+\eta(1-S)/d)$ (§8.2).
6. "The $10^{-8}$ guard becomes reachable at margin $\sim10^{-4}$." → It fires only when $\chi$ rounds to exactly $1.0$ in float32 (logit $\approx 17$); the worst case $u=-q$ is unreachable (§8.3).
7. "$\mu$ reliably rounds to $1$ after 25 writes." → Computed $\mu$ lands within float noise of $1$ (below, equal, or above); the masked-denominator form is safe in all cases (§11, §15).
8. "Compute doses as $1-e^{-x}$." → Use $-\mathrm{expm1}(-x)$ (§15).
9. "The $g=2.5$ query-gap example applies under continued periodic writing." → Valid only if writes stop; otherwise $g \in [0,\Delta t)$ (§5.3).
10. "Gain over cue-only suffices to show cue-conditioned retrieval." → Must beat the centroid baseline too (§17.1, §17.3).
11. "Diffusion creates no new basins / no spurious minima." → Precise statement: the range of $M$ is unchanged and no new stored direction is created; critical-point counts of the cue-conditioned landscape are not claimed (§5.2, §13).

**Unchanged from the v2.0 line:** the primitive, dose maps, parameterization, recurrence, defaults ($\omega_0 \approx 0.0513$, $\Gamma_0 = 10^{-3}$, $\eta_0 = 1$, $\chi_0 = 0.01$, $\bar\varepsilon_0 = 0.01$, $N = 20$, $\epsilon_{\mathrm{fall}} = 10^{-8}$, $\epsilon_L = 10^{-8}$, $\epsilon_{\mathrm{dec}} = 10^{-6}$), and the single-trace calibration table. The qualitative claims (dose identifiability, anisotropic capture, training benefits) remain empirical hypotheses for §17.
