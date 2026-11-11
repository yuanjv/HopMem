HopMem is a fixed-size semantic associative memory module for language models. It maintains a symmetric positive-semidefinite fast-weight matrix with bounded trace, writing normalized semantic embeddings using importance-weighted mass and multiplicative forgetting. Retrieval is performed via sparse, energy-inspired attractor dynamics on the unit sphere, which resolve overlapping traces into distinct semantic states. A confidence-gated reinforcement mechanism consolidates successfully retrieved directions, allowing recent or frequently cued memories to remain detailed while uncued older traces compress into aggregate background structure.

# Frozen Specification v1.9  
## Simple Fixed-Size Priority-Weighted Semantic Associative Memory

---

## 1. Objective

The system maintains a fixed-size semantic memory as a positive semidefinite matrix. Incoming normalized semantic embeddings are written with importance-weighted mass. New writes multiplicatively dilute old contributions. Retrieval uses a normalized cue and an energy-inspired recurrence to produce a unit-norm semantic state.

The system supports:

- fixed-size storage,
- priority-weighted writing,
- write-driven forgetting,
- cue-conditioned retrieval,
- blending through shared eigenstructure,
- learned write/read parameters,
- optional reinforcement,
- optional decoding.

The system does not support:

- exact episodic recall,
- unlimited capacity,
- individually addressable records,
- guaranteed attractor basins.

---

## 2. Inputs

Let $d$ be the embedding dimension.

At time $t$, the full semantic embedding is:

$$
z_t \in \mathbb R^{d},
\qquad
\|z_t\|_2 = 1.
$$

A retrieval cue is:

$$
q \in \mathbb R^{d},
\qquad
\|q\|_2 = 1.
$$

An importance score is:

$$
s_t \ge 0.
$$

The importance score is a proxy, not ground-truth semantic importance. The write rule and the guaranteed properties require only that $s_t$ is a nonnegative finite scalar. The specification does not assume a particular source for it.

**Scale assumption (A1).** The calibration statements in §4.1, §5, §6.5, and §15 assume the scores are scaled to

$$
s_t \in [0,1].
$$

If they are not, either rescale them to $[0,1]$, or replace $s_t=1$ in those statements by the typical high-importance score $s_{\mathrm{hi}}$, and use the mass scale

$$
\kappa_{\max}
\frac{s_{\mathrm{hi}}}{s_{\mathrm{hi}}+1}
$$

in place of $\kappa_{\max}/2$.

---

## 3. Memory state

The memory is a symmetric matrix:

$$
W \in \mathbb R^{d \times d}.
$$

It must satisfy:

$$
W \succeq 0,
\qquad
F = \mathrm{tr}(W) \in [0,1].
$$

Here $F$ is memory fill.

Conceptually:

$$
W = \sum_i \alpha_i z_i z_i^{\top},
$$

with:

$$
\alpha_i \ge 0,
\qquad
\sum_i \alpha_i = F.
$$

The coefficients $\alpha_i$ are mathematical contributions, not explicit memory records.

Because:

$$
zz^{\top} = (-z)(-z)^{\top},
$$

the memory is sign-blind. It stores unoriented directions.

Initialize:

$$
W_0 = 0,
\qquad
F_0 = 0.
$$

---

## 4. Writing

### 4.1 Write rate

The half-saturation constant is fixed:

$$
c = 1.
$$

The write rate is:

$$
\kappa_t =
\kappa_{\max}
\frac{s_t}{s_t + 1}.
$$

Constraints:

$$
0 < \kappa_{\max} < 1,
\qquad
s_t \ge 0.
$$

Therefore:

$$
0 \le \kappa_t < \kappa_{\max}.
$$

Under (A1), $s_t\in[0,1]$, so

$$
\frac{s_t}{s_t+1} \le \frac12,
$$

and therefore:

$$
\kappa_t \le \frac{\kappa_{\max}}{2}.
$$

So under (A1), a freshly written item has mass at most $\kappa_{\max}/2$, attained at $s_t=1$. This mass scale matters for retrieval calibration; see §6.5.

Initial default:

$$
\kappa_{\max} = 0.1.
$$

So if $s_t=1$,

$$
\kappa_t = 0.1 \cdot \frac{1}{2} = 0.05.
$$

If (A1) does not hold, see §2 for the replacement mass scale.

---

### 4.2 Memory update

For each input $z_t$:

$$
W_t =
(1-\kappa_t)W_{t-1} +
\kappa_t z_t z_t^{\top}.
$$

Fill update:

$$
F_t =
(1-\kappa_t)F_{t-1} +
\kappa_t.
$$

Because $\|z_t\|_2=1$,

$$
\mathrm{tr}(z_t z_t^{\top})=1.
$$

Therefore the fill update is exact.

This update preserves:

$$
W_t \succeq 0,
\qquad
0 \le F_t \le 1.
$$

---

### 4.3 Fill is monotone

The fill update can be rewritten as:

$$
F_t =
F_{t-1} +
\kappa_t(1-F_{t-1}).
$$

Since $\kappa_t \ge 0$ and $F_{t-1}\le 1$,

$$
F_t \ge F_{t-1}.
$$

So total fill does not decrease during writing.

Forgetting in this system means:

- old contributions are multiplied by $1-\kappa_t$,
- old relative mass is diluted,
- total fill increases toward 1.

This is write-driven forgetting, not time-based decay.

---

### 4.4 Contribution retention

The contribution of an input written at time $i$, after later writes through time $t$, is:

$$
\alpha_{i,t} =
\kappa_i
\prod_{j=i+1}^{t}
(1-\kappa_j).
$$

These contributions satisfy:

$$
F_t = \sum_{i\le t} \alpha_{i,t}.
$$

Interpretation:

- importance determines initial mass,
- later writes determine decay,
- a high-importance memory does not have a slower decay rate; it starts with more mass.

The directional mass of a stored direction is bounded below by its contribution:

$$
z_i^{\top} W_t z_i \ge \alpha_{i,t},
$$

with equality when all other stored directions with nonzero mass are orthogonal to $z_i$.

---

## 5. Learnable parameters

The initial trainable parameter set is:

$$
\theta = \{\kappa_{\max}, \eta, \gamma\}.
$$

The half-saturation constant $c$ is fixed at 1 in the frozen core. Learning $c$ should be treated as a later ablation.

Use bounded parameterization:

$$
\kappa_{\max} = \sigma(a_\kappa)
$$

$$
\eta = \eta_{\max}\sigma(a_\eta)
$$

$$
\gamma = \gamma_{\max}\sigma(a_\gamma)
$$

where $\sigma$ is the sigmoid.

Recommended initial bounds:

$$
\eta_{\max} = 10,
\qquad
\gamma_{\max} = 0.05.
$$

This guarantees:

$$
\eta\gamma \le \eta_{\max}\gamma_{\max} = 0.5 < 1,
$$

which is the stability condition of §6.3.

Why these bounds:

- Item masses are small under (A1). A fresh item with $s_t=1$ has mass $\kappa_{\max}/2 = 0.05$, the largest single-item mass under (A1), and older items are smaller. The pull toward memory per step is therefore roughly $\eta m$ with $m \lesssim 0.05$. With $\eta_{\max}=1$ and $N\le 20$, the recurrence barely leaves the cue.
- $\gamma$ competes directly with $m$. When $\gamma \gg m$, retrieval stays near $q$. $\gamma_{\max}=0.5$ would be ten times the largest single-item mass under (A1) and the default $\kappa_{\max}=0.1$, so most of its range would mean “ignore memory.”

Suggested initial values:

$$
\kappa_{\max}=0.1,
\qquad
\eta=1,
\qquad
\gamma=0.01.
$$

The corresponding sigmoid outputs are $0.1$, $0.1$ of $\eta_{\max}=10$, and $0.2$ of $\gamma_{\max}=0.05$. The initial logits are therefore approximately:

$$
a_\kappa \approx -2.20,
\qquad
a_\eta \approx -2.20,
\qquad
a_\gamma \approx -1.39.
$$

Coupling warning. The read gain scales roughly as $\eta m$, and $m$ is set by $\kappa_{\max}$, which also sets the forgetting rate. Training can therefore raise $\kappa_{\max}$ to compensate for a weak $\eta$, which distorts forgetting. §8 freezes $\kappa_{\max}$ in the first training stage, and §6.4 logs how close $\eta$ and $\gamma$ are to their caps.

---

## 6. Retrieval

### 6.1 Retrieval energy

The core retrieval energy has no regularizer:

$$
E(u;q,W) =
-\frac12 u^{\top} W u -
\gamma u^{\top} q,
\qquad
\|u\|_2=1.
$$

The smooth gradient is:

$$
\nabla E(u) =
-Wu - \gamma q.
$$

This assumes $W$ is symmetric. In implementation, symmetrize $W$ if needed.

---

### 6.2 Retrieval recurrence

Initialize:

$$
u_0 = q.
$$

For $n=0,\dots,N-1$:

$$
v_n =
u_n +
\eta(Wu_n + \gamma q).
$$

Normalize:

$$
u_{n+1} =
\frac{v_n}{\|v_n\|_2}.
$$

Return:

$$
u^{*} = u_N.
$$

Numerical guard. If

$$
\|v_n\|_2 < \epsilon_{\mathrm{fall}}
$$

or $v_n$ contains a non-finite value, set $u_{n+1}=q$ and increment a counter. The explicit finiteness test is needed because a NaN makes the comparison $\|v_n\|_2<\epsilon_{\mathrm{fall}}$ false.

Recommended threshold:

$$
\epsilon_{\mathrm{fall}} = 10^{-8}.
$$

The guard assumes that $q$ and $W$ are finite. If $q$ or $W$ is non-finite, this is an invalid input or memory state and should be logged or rejected before retrieval. The guard is a safety mechanism inside the recurrence, not a substitute for finite inputs.

In the core recurrence, the norm-based part of the guard is unreachable in exact finite arithmetic. If $\|u_n\|_2=1$ and $\eta\gamma<1$, then

$$
\|v_n\|_2 \ge u_n^{\top} v_n = 1+\eta r_n \ge 1-\eta\gamma,
$$

where $r_n$ is defined in §6.3. Under the recommended bounds,

$$
\|v_n\|_2 \ge \frac12.
$$

A nonzero guard count in the core indicates a bug, a non-unit $u_n$, a violated bound, or a non-finite value. The norm test becomes a real branch only with shrinkage (§9).

Recommended iteration count:

$$
N = 20
\quad
(\text{range } [10,30]).
$$

Reduce $N$ only after the calibration run (§17.3) shows that retrieval has saturated.

---

### 6.3 Stability note

Define directional support:

$$
\mu_n = u_n^{\top} W u_n.
$$

Define:

$$
r_n =
\mu_n + \gamma u_n^{\top} q.
$$

Then:

$$
v_n =
(1+\eta r_n)u_n +
\eta(I-u_nu_n^{\top})(Wu_n+\gamma q).
$$

If:

$$
1+\eta r_n > 0,
$$

then the normalized update is a Riemannian gradient step on the sphere, with normalization as the retraction, and with effective step size:

$$
h_n =
\frac{\eta}{1+\eta r_n}.
$$

Because:

$$
\mu_n \ge 0,
\qquad
u_n^{\top} q \ge -1,
$$

we have:

$$
r_n \ge -\gamma.
$$

Therefore, if:

$$
\eta\gamma < 1,
$$

then:

$$
1+\eta r_n \ge 1-\eta\gamma > 0,
$$

and:

$$
0 < h_n \le \frac{\eta}{1-\eta\gamma}.
$$

So the recommended bound $\eta_{\max}\gamma_{\max}<1$ prevents radial sign flips in the retrieval recurrence.

Remarks:

- $\eta\gamma<1$ is sufficient, not necessary. What the recurrence needs is $1+\eta r_n>0$. Since $r_n \ge \gamma u_n^{\top} q$, if $\eta\gamma>0$, this condition holds whenever

$$
  u_n^{\top} q > -\frac{1}{\eta\gamma}.
$$

  If $\gamma=0$, the condition is automatic because $r_n=\mu_n\ge0$. The bound $\eta\gamma<1$ is conservative and ties $\gamma_{\max}$ to $\eta_{\max}$.

- For fixed $r_n>0$, $h_n \to 1/r_n$ as $\eta\to\infty$, and the update tends to

$$
  u \leftarrow \frac{Wu+\gamma q}{\|Wu+\gamma q\|_2},
$$

  which is power iteration with cue injection. Large $\eta$ is therefore a well-defined regime, not automatically a divergence risk. This is why $\eta_{\max}=10$ is acceptable.

- The smooth part of the energy,

$$
  -\frac12 u^{\top} W u-\gamma u^{\top} q,
$$

  is concave in $u$, because its Hessian is $-W \preceq 0$. Without the sphere constraint, minimizing it would be unbounded below, and the plain step would grow like a power iteration. Normalization is what makes retrieval well-posed, and standard proximal-gradient theory does not apply.

This does not guarantee global convergence or monotone energy decrease. Monotone decrease holds for sufficiently small $h_n$; $h_n$ can be large at large $\eta$, so monitor it (§6.4).

---

### 6.4 Retrieval diagnostics

During retrieval, monitor:

$$
E(u_n;q,W),
$$

$$
\|u_{n+1}-u_n\|_2,
$$

$$
\mu_n = u_n^{\top} W u_n,
$$

$$
1+\eta r_n \quad (\text{margin}),
$$

$$
h_n,
$$

and the guard counter, which should be exactly 0 in the core unless a non-finite value appears.

Also monitor:

- $\eta\gamma$,
- $\eta/\eta_{\max}$,
- $\gamma/\gamma_{\max}$.

If $\eta/\eta_{\max}$ sits near 1 during training, the cap or $N$ is limiting retrieval gain; revisit §6.5.

---

### 6.5 Retrieval-gain calibration

This section explains the current bounds and gives the check to run before training.

Setup. Let $z$ be a stored direction with mass

$$
m = z^{\top} W z,
$$

and suppose $W \approx m\,zz^{\top}$ over the relevant subspace. Let the cue $q$ sit at a small angle $\varphi$ from $z$, and let $\psi_n$ be the angle $u_n$ has moved from $q$ toward $z$. To first order in small angles:

$$
\psi_{n+1}
\approx
\psi_n +
h\left[m(\varphi-\psi_n)-\gamma\psi_n\right],
\qquad
h=\frac{\eta}{1+\eta r}.
$$

Hence:

$$
\psi^{*}=\frac{m}{m+\gamma}\,\varphi,
\qquad
\rho = h(m+\gamma),
\qquad
\psi_N \approx \left(1-e^{-\rho N}\right)\psi^{*}.
$$

Here $\psi^{*}$ is the equilibrium and $\rho$ is the per-step relaxation rate.

Consequences:

1. **$\gamma$ competes with $m$.** If $\gamma\gg m$, then $\psi^{*}\approx (m/\gamma)\varphi$ and retrieval stays near $q$. $\gamma$ should live at the scale of the item masses it is meant to retrieve.
2. **Gain requires $\rho N \gtrsim 1$.** Use the smallest mass you expect to retrieve, for example $m\approx 0.01$ for older items.
3. **Mass scale.** $m \ge \alpha_{i,t}$ (§4.4). Under (A1), the largest contribution of a freshly written item is $\alpha=\kappa_{\max}/2$, which is $0.05$ at the default $\kappa_{\max}=0.1$.

Worked values, with $d=384$, $\varphi=0.1$ rad, and one stored direction, from a direct simulation of the recurrence:

| Setting | $\eta$ | $\gamma$ | $N$ | $m$ | $\rho N$ | $\psi_N/\varphi$ |
|---|---:|---:|---:|---:|---:|---:|
| v1.2 default | 0.1 | 0.05 | 20 | 0.05 | 0.20 | 0.09 |
| v1.2 at $\eta$ cap, $\gamma\to0$ | 1 | ≈0 | 20 | 0.05 | 0.95 | 0.62 |
| Current default | 1 | 0.01 | 20 | 0.05 | 1.13 | 0.57 |
| Current default, older item | 1 | 0.01 | 20 | 0.01 | 0.39 | 0.16 |
| Current bounds, at $\eta$ cap | 10 | 0.005 | 20 | 0.01 | 2.6 | 0.63 |

With the v1.2 defaults, retrieval moved only about 9% of the way from the cue to the stored direction. With the current defaults, unchanged since v1.3, it moves about 57% for a fresh item. Learning is expected to raise $\eta$ toward its cap for older items.

Caveat. These are small-angle, single-dominant-direction estimates. Real memories blend several directions, and mass shared along the cue direction raises the effective $m$. Treat the table as a sanity check, not a prediction; the calibration run (§17.3) is the real test.

---

## 7. Training objective

Because the memory is sign-blind, the primary loss is sign-blind.

For a target embedding:

$$
z_{\mathrm{target}} \in \mathbb R^{d},
\qquad
\|z_{\mathrm{target}}\|_2=1,
$$

define:

$$
L_{\mathrm{retrieval}} =
1 - \left|u^{*\top} z_{\mathrm{target}}\right|.
$$

Equivalently:

$$
L_{\mathrm{retrieval}} =
1 - |\cos(u^{*}, z_{\mathrm{target}})|.
$$

This loss treats $u^{*}$ and $-u^{*}$ as equivalent, which matches the memory operator. Unlike $1-\cos^{2}$, its gradient does not vanish when $u^{*}\perp z_{\mathrm{target}}$: the gradient of $1-\cos^{2}$ with respect to $u$ is proportional to $\cos(u^{*},z_{\mathrm{target}})$, so it gives little signal exactly when retrieval is far from the target.

The absolute-value loss is nondifferentiable at $\cos=0$. This is usually acceptable as a measure-zero event, and a subgradient can be used.

If a differentiable surrogate is required, first clamp the cosine:

$$
c = \mathrm{clip}\left(u^{*\top}z_{\mathrm{target}}, -1, 1\right).
$$

Then use:

$$
L_{\mathrm{smooth\text{-}abs}} =
\sqrt{1+\epsilon_L} -
\sqrt{c^{2}+\epsilon_L}.
$$

Recommended initial value:

$$
\epsilon_L = 10^{-8}.
$$

This surrogate is nonnegative for $|c|\le1$, and it is exactly zero at perfect retrieval, $|c|=1$. Its gradient with respect to $c$ is:

$$
-\frac{c}{\sqrt{c^{2}+\epsilon_L}}.
$$

Therefore it approximates the subgradient of $1-|c|$ when $|c|\gg \sqrt{\epsilon_L}$, but its gradient vanishes at $c=0$. The signal is weakened for

$$
|c| \lesssim \sqrt{\epsilon_L}.
$$

In implementation, clamping $c$ prevents tiny numerical overshoots beyond $\pm1$ from making the surrogate slightly negative.

Ablations:

- absolute sign-blind loss:

$$
  1-|\cos(u^{*},z_{\mathrm{target}})|,
$$

- smooth sign-blind surrogate:

$$
  \sqrt{1+\epsilon_L} -
  \sqrt{(u^{*\top}z_{\mathrm{target}})^{2}+\epsilon_L},
$$

- smooth quadratic sign-blind loss:

$$
  1-\cos^{2}(u^{*},z_{\mathrm{target}}),
$$

- oriented loss:

$$
  1-u^{*\top}z_{\mathrm{target}},
$$

  only if the cue reliably determines the target sign.

Reference value. The cue-only baseline $u^{*}=q$, equivalent to $N=0$, has loss

$$
1-|q^{\top} z_{\mathrm{target}}|.
$$

Learned retrieval must beat it; see §17.1.

---

## 8. Training episodes

Each episode contains:

1. a sequence of semantic inputs $z_{1:T}$,
2. importance scores $s_{1:T}$,
3. a cue $q$,
4. a target embedding $z_{\mathrm{target}}$.

Procedure:

1. Reset memory:

$$
   W_0=0,
   \qquad
   F_0=0.
$$

2. Write the sequence using the write rule.
3. Construct the cue.
4. Run retrieval.
5. Compute the retrieval loss.
6. Backpropagate through writing and retrieval.
7. Update $\theta$.

Targets should not always be the most recent item. Vary:

- target position,
- cue completeness,
- recency,
- importance,
- competing semantic inputs.

Start with short sequences and few retrieval iterations. Increase after training is stable.

Staged schedule:

- **Stage A:** freeze $\kappa_{\max}=0.1$; learn $\{\eta,\gamma\}$.
- **Stage B:** unfreeze $\kappa_{\max}$ from its Stage A value and learn $\{\kappa_{\max},\eta,\gamma\}$.

Reason: retrieval gain depends on $\eta m$, and $m$ depends on $\kappa_{\max}$. Joint training from the start can use $\kappa_{\max}$ to compensate for a weak $\eta$, which distorts the forgetting rate. Compare Stage B against Stage A to see whether learning $\kappa_{\max}$ helps.

---

## 9. Optional shrinkage

Shrinkage is not part of the frozen v1.9 core. It should be introduced only after the $\lambda=0$ system works.

If enabled, the energy becomes:

$$
E_\lambda(u;q,W) =
-\frac12 u^{\top} W u -
\gamma u^{\top} q +
\lambda \|u\|_1,
$$

with:

$$
\lambda \ge 0.
$$

The retrieval step becomes:

$$
v_n =
u_n + \eta(Wu_n+\gamma q),
$$

$$
\widetilde u_{n+1} =
S_{\eta\lambda,\tau}(v_n),
$$

$$
u_{n+1} =
\begin{cases}
q,
&
\|\widetilde u_{n+1}\|_2 < \epsilon_{\mathrm{fall}}
\ \text{or}\
\widetilde u_{n+1}\ \text{non-finite},
\\
\dfrac{\widetilde u_{n+1}}{\|\widetilde u_{n+1}\|_2},
&
\text{otherwise}.
\end{cases}
$$

With shrinkage the fallback is a real branch: $\widetilde u_{n+1}$ can vanish. Count and report fallback events. The non-finite check of §6.2 applies here too. The analysis of §6.3 does not carry over, since the update is no longer simply $v_n/\|v_n\|$.

When:

$$
\lambda=0,
$$

skip shrinkage entirely:

$$
\widetilde u_{n+1}=v_n.
$$

Do not approximate $\lambda=0$ by applying the smooth operator with a very small $\lambda$. Use an explicit off-switch. If $\lambda=\mathrm{softplus}(a_\lambda)$ is learned, it is strictly positive for finite parameters, so keep a separate shrinkage flag.

Scale of the threshold. A typical coordinate of a dense unit vector in $\mathbb R^{384}$ has magnitude about

$$
\frac{1}{\sqrt d} \approx 0.051.
$$

Choose

$$
a=\eta\lambda
$$

well below that, for example

$$
a \lesssim \frac{0.2}{\sqrt d} \approx 0.01.
$$

At $\eta=1$, this means $\lambda\lesssim 0.01$. At larger $\eta$, scale $\lambda$ down accordingly. A larger $a$ drives $\widetilde u$ toward zero and makes the guard fire.

---

### 9.1 Exact soft-threshold

The exact soft-threshold operator is:

$$
S_a(v)_j =
\mathrm{sign}(v_j)
\max(|v_j|-a,0).
$$

This is soft-thresholding, not hard-thresholding. Hard-thresholding would be:

$$
v_j\mathbf 1\{|v_j|>a\}.
$$

The exact soft-threshold operator is useful as a non-differentiable baseline.

---

### 9.2 Centered smooth soft-threshold

For differentiable training, use the centered smooth soft-threshold:

$$
S_{a,\tau}(v)_j =
\mathrm{sign}(v_j)
\tau
\left[
\mathrm{softplus}
\left(
\frac{|v_j|-a}{\tau}
\right) -
\mathrm{softplus}
\left(
-\frac{a}{\tau}
\right)
\right],
$$

where:

$$
a = \eta\lambda,
\qquad
\tau > 0.
$$

This operator satisfies:

$$
S_{a,\tau}(0)=0,
$$

is continuous, is $C^{1}$, and has derivative at zero

$$
\sigma(-a/\tau)
$$

from both sides. It is not $C^{2}$.

Implementation note: autograd through

$$
\mathrm{sign}(v_j)\,g(|v_j|)
$$

usually returns gradient 0 at exactly $v_j=0$, not $\sigma(-a/\tau)$. Use a custom derivative or a safe factorized form if exact zeros can occur.

As $\tau\to0$, it approaches exact soft-thresholding:

$$
S_{a,\tau}(v)_j
\to
\mathrm{sign}(v_j)
\max(|v_j|-a,0).
$$

At $a=0$ it is not the identity, which is why $\lambda=0$ skips the operator.

Initially, keep $\tau$ fixed. If shrinkage is learned, learn only $\lambda$ first.

---

## 10. Optional reinforcement

Reinforcement writes a retrieved direction back into memory. It should be enabled only after basic retrieval is reliable.

Given a retrieved state $u^{*}$, compute directional support:

$$
\mu = u^{*\top} W u^{*}.
$$

This satisfies:

$$
0 \le \mu \le F \le 1.
$$

Let the retrieved content have importance:

$$
s_{\mathrm{ret}} \ge 0.
$$

Define its target write rate:

$$
\kappa_{\mathrm{ret}} =
\kappa_{\max}
\frac{s_{\mathrm{ret}}}{s_{\mathrm{ret}}+1}.
$$

Let:

$$
C \in [0,1]
$$

be confidence, with threshold:

$$
C_{\min}\in[0,1].
$$

Choose:

$$
0 < \beta_{\max} \le 1.
$$

A simple initial value is:

$$
\beta_{\max}=0.1.
$$

If:

$$
C \ge C_{\min}
$$

and:

$$
\mu < \kappa_{\mathrm{ret}},
$$

set:

$$
\beta =
C
\cdot
\min\left(
\beta_{\max},
\frac{\kappa_{\mathrm{ret}}-\mu}{1-\mu}
\right).
$$

The denominator is safe because:

$$
\mu<\kappa_{\mathrm{ret}}<\kappa_{\max}<1.
$$

Otherwise:

$$
\beta=0.
$$

Then update:

$$
W
\leftarrow
(1-\beta)W +
\beta\, u^{*} u^{*\top}.
$$

Fill update:

$$
F
\leftarrow
(1-\beta)F +
\beta.
$$

The updated support is:

$$
\mu' =
(1-\beta)\mu + \beta.
$$

If $C=1$ and the cap $\beta_{\max}$ does not bind, this raises $\mu$ exactly to $\kappa_{\mathrm{ret}}$.

Reinforcement preserves:

$$
W \succeq 0,
\qquad
0 \le F \le 1.
$$

It reinforces a direction, not necessarily a correct episodic memory. Confidence gating is important.

---

## 11. Optional decoder

The retrieved state may be passed to a decoder.

Define retrieval features:

$$
r =
[
u^{*},
q,
u^{*} \odot q,
E(u^{*}),
\mu,
F,
\text{fallback flag}
].
$$

Optional additional features:

- largest eigenvalue of $W$,
- eigengap,
- effective rank,
- spectral entropy.

Let the raw decoder output be $D_\psi(r)$. Apply norm-clipped normalization:

$$
\hat z =
\frac{D_\psi(r)}
{\max\left(\|D_\psi(r)\|_2,\ \epsilon_{\mathrm{dec}}\right)},
\qquad
\epsilon_{\mathrm{dec}}=10^{-6}.
$$

Recommended range:

$$
10^{-6} \le \epsilon_{\mathrm{dec}} \le 10^{-4}.
$$

This guarantees:

$$
\|\hat z\|_2 =
\frac{\|D_\psi(r)\|_2}
{\max(\|D_\psi(r)\|_2,\epsilon_{\mathrm{dec}})}
\le 1.
$$

The output has exact unit norm only when:

$$
\|D_\psi(r)\|_2 \ge \epsilon_{\mathrm{dec}}.
$$

When the raw output norm is below the floor, $\hat z$ has norm below 1. This is intentional: it avoids division by zero and keeps the loss bounded.

Normalization, or norm-clipped normalization, is required. For an unnormalized output, a squared-overlap loss such as

$$
1-(\hat z^{\top} z_{\mathrm{target}})^{2}
$$

is unbounded below, because it can be driven down by growing $\|\hat z\|$.

The max-floor denominator is continuous. It is differentiable below the floor and above the floor, but not exactly differentiable at the boundary

$$
\|D_\psi(r)\|_2=\epsilon_{\mathrm{dec}}.
$$

If a fully smooth decoder normalization is desired, use instead:

$$
\hat z_{\mathrm{smooth}} =
\frac{D_\psi(r)}
{\sqrt{\|D_\psi(r)\|_2^{2}+\epsilon_{\mathrm{dec}}^{2}}}.
$$

This variant satisfies:

$$
\|\hat z_{\mathrm{smooth}}\|_2 < 1
$$

for finite raw outputs, is smooth, and approaches unit norm when

$$
\|D_\psi(r)\|_2 \gg \epsilon_{\mathrm{dec}}.
$$

Because the smooth decoder output is strictly subunit for finite raw outputs, it can never produce an exactly zero oriented reconstruction loss: for any target, $L_{\mathrm{meaning}}=1-\hat z^{\top} z_{\mathrm{target}}\ge 1-\|\hat z\|_2>0$. For example, if

$$
D_\psi(r)=\alpha z_{\mathrm{target}},
\qquad
\alpha>0,
$$

then max-floor normalization gives exact zero loss whenever $\alpha\ge\epsilon_{\mathrm{dec}}$, while the smooth denominator gives

$$
L_{\mathrm{meaning}} =
1-\frac{\alpha}{\sqrt{\alpha^{2}+\epsilon_{\mathrm{dec}}^{2}}},
$$

which approaches zero only as $\alpha\to\infty$. In practice, with $\epsilon_{\mathrm{dec}}=10^{-6}$ and decoder outputs of ordinary magnitude, this difference is tiny. When evaluated naively as $1-\hat z^{\top} z_{\mathrm{target}}$ in float32, the smooth loss rounds to zero once $\|D_\psi(r)\|_2$ exceeds roughly $10^{4}\epsilon_{\mathrm{dec}}$, which is $10^{-2}$ at the default. A numerically stable evaluation can retain a tiny positive loss. Under naive evaluation, therefore, the choice between the two normalizations matters only in exact arithmetic or near the floor. If exact zero loss or exact unit-norm outputs are required for finite decoder outputs, prefer max-floor normalization.

Gradient scale. The Jacobian of the decoder normalization has spectral norm

$$
\frac{1}{\max\left(\|D_\psi(r)\|_2,\ \epsilon_{\mathrm{dec}}\right)}
\quad\text{(max-floor)},
$$

away from the kink, and

$$
\frac{1}{\sqrt{\|D_\psi(r)\|_2^{2}+\epsilon_{\mathrm{dec}}^{2}}}
\quad\text{(smooth)}.
$$

Both are bounded by

$$
\frac{1}{\epsilon_{\mathrm{dec}}}.
$$

At the default $\epsilon_{\mathrm{dec}}=10^{-6}$, this bound is $10^{6}$. At $\epsilon_{\mathrm{dec}}=10^{-4}$, it is $10^{4}$.

The decoder floor is a separate constant from $\epsilon_{\mathrm{fall}}$. The retrieval guard can use $10^{-8}$ because $\|v_n\|_2\ge 1/2$ in the core recurrence. A decoder's raw output is not so constrained, and a floor of $10^{-6}$ to $10^{-4}$ costs little when trained outputs are typically much larger.

Monitor decoder gradient norms, especially when raw outputs are near zero. If outputs collapse near zero or gradients are unstable, increase $\epsilon_{\mathrm{dec}}$, for example to $10^{-5}$ or $10^{-4}$, or use gradient clipping or learning-rate warmup. Weight decay is not a remedy: for a final linear layer, rescaling its weights by $c$ leaves $\hat z$ unchanged away from the floor and multiplies the weight gradient by $1/c$, so shrinking the weights increases gradients and the effective step size. The smooth denominator removes the kink at the floor but does not reduce the worst-case gradient scale; only increasing $\epsilon_{\mathrm{dec}}$ reduces that bound.

Default semantic reconstruction loss:

$$
L_{\mathrm{meaning}} =
1 - \hat z^{\top} z_{\mathrm{target}}.
$$

This is oriented on purpose. The decoder sees $q$, which can supply orientation, and downstream use of embeddings is often oriented. Discarding sign would lose information that the decoder can in principle recover from the cue.

If the cues do not reliably carry orientation, use the sign-blind variant:

$$
L_{\mathrm{meaning}}^{\mathrm{blind}} =
1 - \left|\hat z^{\top} z_{\mathrm{target}}\right|.
$$

A smooth sign-blind decoder loss can use the surrogate of §7 with

$$
c=\mathrm{clip}\left(\hat z^{\top} z_{\mathrm{target}},-1,1\right).
$$

Because $\|\hat z\|_2$ may be below 1, this $c$ is a clamped inner product, not necessarily a true cosine. If $\|\hat z\|_2<1$, then $|c|<1$ even when the direction is perfect, so the smooth surrogate cannot reach exactly zero. This happens under the smooth denominator for finite outputs, and under max-floor normalization when $\|D_\psi(r)\|_2<\epsilon_{\mathrm{dec}}$. Under max-floor normalization with $\|D_\psi(r)\|_2\ge\epsilon_{\mathrm{dec}}$, $\|\hat z\|_2=1$ and the surrogate reaches exactly zero at perfect direction. The loss therefore penalizes subunit decoder outputs.

This loss can optionally be weighted by target importance.

Evaluate memory retrieval separately from the decoder, using the sign-blind retrieval loss of §7. A strong decoder can hide weak memory retrieval.

---

## 12. Optional trace-one representation

An isotropic background can be added:

$$
\widetilde W =
W + \frac{1-F}{d}I.
$$

Then:

$$
\mathrm{tr}(\widetilde W)=1.
$$

For unit $u$,

$$
u^{\top} \widetilde W u =
u^{\top} W u +
\frac{1-F}{d}.
$$

The added term is constant on the unit sphere. Therefore the energy landscape, its fixed points, and its Riemannian gradient direction are unchanged; the energy shifts by the constant

$$
-\frac{1-F}{2d}.
$$

The recurrence itself is not exactly unchanged. The tangential part of the update is the same, but the radial coefficient becomes

$$
1+\eta\widetilde r_n,
$$

with

$$
\widetilde r_n = r_n + \frac{1-F}{d}.
$$

So the effective step becomes:

$$
\widetilde h_n =
\frac{\eta}{1+\eta r_n+\eta(1-F)/d}.
$$

This is smaller than $h_n$. If $r_n\ge0$, the relative decrease is at most approximately

$$
\frac{\eta}{d}.
$$

For $d=384$, that is about 0.3% at $\eta=1$ and about 2.6% at $\eta=10$. Under the stability bound alone, if $r_n$ is allowed to be negative, the looser bound is

$$
\frac{\eta}{(1-\eta\gamma)d}.
$$

With the recommended $\eta\gamma\le0.5$, this is at most $2\eta/d$, about 0.5% at $\eta=1$ and 5.2% at $\eta=10$.

Use $W$, not $\widetilde W$, for reinforcement and directional support unless you intentionally want the isotropic background included.

---

## 13. Blending

For any unit state $u$,

$$
u^{\top} W u =
\sum_i \alpha_i (z_i^{\top} u)^{2}.
$$

If several stored directions overlap with $u$, they can jointly support it.

For example:

$$
W =
\frac12 z_1z_1^{\top} +
\frac12 z_2z_2^{\top}.
$$

Let:

$$
\omega = z_1^{\top} z_2.
$$

In the span of $z_1,z_2$, the eigenvalues are:

$$
\nu_\pm =
\frac{1\pm \omega}{2},
$$

with eigenvectors proportional to $z_1\pm z_2$. Outside that span the eigenvalue is 0.

If $\omega>0$, the top eigenvector is proportional to:

$$
z_1+z_2.
$$

If $\omega<0$, the top eigenvector is proportional to:

$$
z_1-z_2.
$$

The top eigenvalue is:

$$
\frac{1+|\omega|}{2}.
$$

This means blend-like directions can be spectrally recoverable, even though the memory does not explicitly store an averaged vector. This is exact only for equal weights; with unequal weights the top eigenvector is not generally the weighted mean.

---

## 14. Guaranteed properties

Given the assumptions:

$$
\|z_t\|_2=1,
\qquad
s_t\ge0,
\qquad
0<\kappa_{\max}<1,
\qquad
0\le\beta\le1,
$$

and finite arithmetic, the following are guaranteed:

$$
W \succeq 0,
$$

$$
0 \le F \le 1,
$$

$$
F_t =
(1-\kappa_t)F_{t-1} +
\kappa_t,
$$

$$
\alpha_{i,t} =
\kappa_i
\prod_{j=i+1}^{t}(1-\kappa_j),
$$

and reinforcement preserves PSD and fill bounds.

Given in addition:

$$
\|u_n\|_2=1,
\qquad
\eta\gamma<1,
$$

the core recurrence satisfies:

$$
\|v_n\|_2 \ge 1-\eta\gamma > 0,
$$

and:

$$
1+\eta r_n \ge 1-\eta\gamma > 0.
$$

So the norm test of the fallback guard never triggers in exact finite arithmetic, and the update is a Riemannian gradient step with:

$$
0 < h_n \le \frac{\eta}{1-\eta\gamma}.
$$

Not guaranteed:

- exact recall,
- stable attractors,
- unique retrieval basins,
- monotone energy decrease for arbitrary $\eta,\gamma,N$,
- meaningful blending for every input combination,
- that the default bounds give useful retrieval gain for a given memory; check this with §6.5 and §17.3.

These must be checked empirically.

---

## 15. Implementation rules

For a clean implementation:

1. Normalize all $z_t$ and $q$.
2. Check that $z_t$, $q$, $s_t$, and $W$ are finite. Non-finite inputs or memory states should be logged or rejected; the retrieval guard is not a substitute for valid inputs.
3. Clamp importance scores:

$$
   s_t \leftarrow \max(s_t,0).
$$

4. Symmetrize $W$ after updates:

$$
   W \leftarrow \frac{W+W^{\top}}{2}.
$$

5. Keep $F$ consistent:

$$
   F = \mathrm{tr}(W).
$$

   If numerical drift appears, recompute $F$ from the trace.

6. Use $\epsilon_{\mathrm{fall}}$ for the retrieval guard and the separate constant $\epsilon_{\mathrm{dec}}$ for the decoder floor (§11). In the core recurrence, the guard count should be 0 unless a non-finite value appears.
7. If $\lambda=0$, skip shrinkage completely, using an explicit flag.
8. Log $\eta\gamma$, $\eta/\eta_{\max}$, and $\gamma/\gamma_{\max}$.
9. Log $1+\eta r_n$ and $h_n$ if diagnosing retrieval stability.
10. Keep $\gamma$ at or below the scale of the item masses, roughly

$$
    \gamma \lesssim \frac{\kappa_{\max}}{2},
$$

    under (A1). A larger $\gamma$ makes retrieval follow the cue and ignore memory.

11. Use the sign-blind loss for memory-only evaluation.
12. Run the calibration (§17.3) and report the cue-only baseline before the first training run.
13. If using the decoder, monitor decoder gradient norms, especially when raw decoder outputs are near zero. If gradients are unstable, increase $\epsilon_{\mathrm{dec}}$, for example to $10^{-5}$ or $10^{-4}$, or use gradient clipping or learning-rate warmup. Weight decay is not a remedy; it increases decoder gradients (§11). The smooth decoder denominator removes the kink at the floor but does not reduce the worst-case gradient scale. If exact zero loss or exact unit-norm decoder outputs are required for finite outputs, use max-floor normalization rather than the smooth denominator.

---

## 16. Computational complexity

For dense $W\in\mathbb R^{d\times d}$:

| Operation | Complexity |
|---|---:|
| Store $W$ | $O(d^{2})$ |
| Write one item | $O(d^{2})$ |
| Compute $Wu$ | $O(d^{2})$ |
| One retrieval iteration | $O(d^{2})$ |
| Full eigendecomposition | $O(d^{3})$ |

For $d=384$:

$$
d^{2} = 147{,}456.
$$

This is practical for a dense first implementation. Backpropagation through $N$ retrieval iterations and the write sequence adds memory proportional to the number of stored intermediate states.

---

## 17. Evaluation plan

The first experiment should focus on writing, retrieval, and learning the global parameters.

### 17.1 Baselines

Compare the proposed memory against:

0. **Cue-only retrieval, $u^{*}=q$, equivalent to $N=0$.** This is the floor: if the memory does not beat it, retrieval is not using the memory.
1. Nearest-neighbor retrieval over stored embeddings. This baseline stores explicit records.
2. A simple linear-memory baseline.
3. Constant write rate versus importance-dependent write rate.
4. Manually tuned parameters versus learned parameters.

Comparison 4 determines whether learning actually improves the system rather than merely adding complexity.

### 17.2 Retrieval experiments

Evaluate:

- full-cue and partial-cue retrieval,
- retrieval with absent or unrelated memory,
- hard-neighbor retrieval cases,
- retrieval across recency buckets,
- retrieval across importance-score buckets,
- sensitivity to initialization and perturbations,
- generalization to unseen input sequences.

Use separate training and evaluation episodes to prevent sequence leakage.

### 17.3 Calibration run before training

With fixed parameters, using the §5 initial values, run synthetic episodes:

- random unit vectors $z_i$,
- $d=384$,
- cues at known angles from a target,
- varied recency and importance.

Report per recency bucket:

- the fraction of the cue-to-target angle that retrieval closes, $\psi_N/\varphi$,
- the loss improvement over the cue-only baseline,
- $\rho N$,
- $h_n$.

If retrieval is close to the identity in every bucket, raise $\eta_{\max}$ or $N$, or lower $\gamma_{\max}$, before training. Compare against the §6.5 table.

### 17.4 Ablations

| Component | Variants |
|---|---|
| Write rate | Constant $\kappa$, importance-score-based $\kappa$, learned importance function |
| Half-saturation constant | $c=1$ fixed, $c$ learned |
| Parameter optimization | Fixed parameters, learned global parameters |
| Training schedule | $\kappa_{\max}$ frozen then unfrozen (§8), joint from the start |
| Retrieval loss | $1-\lvert\cos\rvert$, smooth $\lvert\cos\rvert$ surrogate, $1-\cos^{2}$, oriented $1-\cos$ |
| Shrinkage | None, exact soft-threshold, centered smooth soft-threshold, sigmoid gate |
| Reinforcement | Disabled, enabled |
| Decoder normalization | Max-floor norm clipping, smooth denominator |

### 17.5 Diagnostics

Record:

- retrieval energy $E(u_n;q,W)$,
- state change $\|u_{n+1}-u_n\|_2$,
- directional support $u_n^{\top} W u_n$,
- guard count in the core, or fallback frequency with shrinkage,
- memory fill $F$,
- learned parameter values and cap saturation,
- largest eigenvalue,
- eigengap,
- effective rank,
- spectral entropy.

Spectral quantities are diagnostic hypotheses, not established measures of semantic ambiguity. Their relationship to retrieval quality must be tested.

### 17.6 Deferred extensions

Not specified in v1.9:

- a contrastive objective,
- a learned importance function,
- a reinforcement confidence model.

If a contrastive objective is added later, use $|\cos|$ as the similarity so that it stays sign-blind.

---

## 18. Core mathematical summary

### Write

$$
\kappa_t =
\kappa_{\max}
\frac{s_t}{s_t+1}
$$

$$
W_t =
(1-\kappa_t)W_{t-1} +
\kappa_t z_tz_t^{\top}
$$

$$
F_t =
(1-\kappa_t)F_{t-1} +
\kappa_t
$$

---

### Retrieve

$$
u_0 = q
$$

$$
v_n =
u_n +
\eta(Wu_n+\gamma q)
$$

$$
u_{n+1} =
\frac{v_n}{\|v_n\|_2}
$$

$$
u^{*} = u_N
$$

with the numerical guard

$$
\|v_n\|_2<\epsilon_{\mathrm{fall}}
\ \text{or}\ v_n\ \text{non-finite}
\quad\Rightarrow\quad
u_{n+1}=q,
$$

whose norm test is unreachable in exact finite arithmetic when $\eta\gamma<1$.

Recommended bounds:

$$
\eta\gamma < 1,
\qquad
\eta_{\max}=10,
\quad
\gamma_{\max}=0.05.
$$

---

### Train

$$
L =
1 - \left|u^{*\top}z_{\mathrm{target}}\right|
$$

Learn:

$$
\theta = \{\kappa_{\max},\eta,\gamma\}
$$

in two stages:

1. learn $\{\eta,\gamma\}$ with $\kappa_{\max}$ frozen;
2. learn all three.

---

### Optional reinforce

$$
\mu = u^{*\top} W u^{*}
$$

$$
\kappa_{\mathrm{ret}} =
\kappa_{\max}
\frac{s_{\mathrm{ret}}}{s_{\mathrm{ret}}+1}
$$

$$
\beta =
C
\cdot
\min\left(
\beta_{\max},
\frac{\kappa_{\mathrm{ret}}-\mu}{1-\mu}
\right)
$$

when active, then:

$$
W
\leftarrow
(1-\beta)W +
\beta\, u^{*} u^{*\top}
$$

$$
F
\leftarrow
(1-\beta)F +
\beta
$$

---

## 19. Central research hypothesis

Version 1.9 tests whether a fixed-size PSD second-moment memory, with simple importance-weighted writes and cue-conditioned normalized retrieval, can preserve useful semantic structure while supporting forgetting and blending.

The first experiment should show:

1. The second-moment memory retains useful semantic direction.
2. Partial cues retrieve useful directions beyond the cue-only baseline $u^{*}=q$.
3. Learned $\{\kappa_{\max},\eta,\gamma\}$ improve over fixed manually chosen values.

Only after that should the system add:

- shrinkage,
- contrastive objectives,
- learned importance functions,
- reinforcement,
- semantic decoders.
