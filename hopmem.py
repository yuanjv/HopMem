#!/usr/bin/env python
"""
HopMem v2.1.1 -- dose-weighted associative memory with a Gibbs (log-sum-exp) readout.

    python hopmem.py selftest
    python hopmem.py calibrate --read gibbs
    python hopmem.py calibrate --read power
    python hopmem.py train --synthetic --schedule probe --batch 256
    python hopmem.py train --same_passage --schedule probe --batch 256

STATE (unchanged from v2.0.4).  W = (1-S) J + M,  M = sum_i alpha_i z_i z_i^T.
Every operation is a convex mixture, so W >= 0 and tr W = 1 hold exactly.  Dose
additivity + the diffusion semigroup give a closed form

    ln alpha_i = ln(a_i (1 - eps_i)) - omega * sum_{j>i} s_j - Gamma * (t_q - t_i),
    a_i = 1 - exp(-omega s_i),   eps_i = eps_bar (1 - a_i).

Forgetting is ADDITIVE IN THE LOG-DOSE.  The state is a set of atoms (z_i, ln alpha_i);
no d x d matrix is ever materialised.  Reinforcement appends an atom.

READOUT.  v2.0.4 maximises E(u) = -1/2 u^T M u - gamma u^T q on the sphere.  A quadratic
on a sphere has ONE stable attractor; any other stored direction is a saddle unless cue
curvature beats cross-talk curvature:

    H = (mu + gamma) P - P M_oth P > 0,  mu = z_k^T M z_k,  P = I - z_k z_k^T,

which for T equal masses needs gamma >~ 2 alpha sqrt((T-1)/d), while converged closure is
alpha/(alpha+gamma).  Capacity fights closure; no tuning fixes it.  Maximise the free
energy instead:

    F(u) = (1/beta) log( e^{beta q.u + kappa} + sum_i alpha_i e^{beta z_i.u} ).

F is convex, so the generalized power method applies verbatim: with
p = softmax([beta Z^T u + ln alpha, beta q.u + kappa]) and nabla F = p_q q + sum p_i z_i,
u+ = nabla F / |nabla F| maximises the linearisation of F over the sphere, hence

    F(u+) >= F(u) for EVERY beta > 0, equality iff u+ = u,

by exactly the spec's 8.3 argument -- and with no step size at all.

  basin       atom k beats the cue at u=q iff beta(1 - q.z_k) < ln alpha_k - kappa,
              so theta_basin = arccos(1 - (ln alpha_k - kappa)/beta)  (conservative).
  floor       theta_floor ~ sqrt( sum_{i!=k} (p_i/p_k)^2 ) ~ (a_bar/a_k) e^{-beta} sqrt(T-1).
              The quadratic force carries c_j ~ 1/sqrt(d); the Gibbs force does not.
  vagueness   ln alpha_k drifts down linearly in time, so the basin shrinks and VANISHES
              at t = (ln alpha_k(0) - kappa)/Gamma  -- the spec's t_1/2cl, now sharp.
  resolution  atoms separated iff beta(1 - z_1.z_2) >~ |ln alpha_1 - ln alpha_2|.

DECISION RULE (part of the model).  The map's null basin is STRICTLY SMALLER than the
one-shot argmax region: moving toward an atom raises its logit and lowers the cue's, so a
marginal null win at u=q runs away.  Deterministic witness (selftest 12): beta=16,
ln alpha=-3, kappa=-6.5, one atom at 0.70 rad -> logits 9.237 (atom) vs 9.500 (null),
null wins, map still commits.  So the system DECIDES at u=q and only then descends:

    commit to the map iff max_k logit_k - logit_null > ln tau, else return q.

tau is one scalar, calibrated ONCE on a fixed training calibration batch (--cal_batch),
reused for every report, saved with the checkpoint, and applied by BOTH `evaluate` and
the streaming `HopMem.retrieve`.  The final [deploy] line is evaluated with the exact
saved tau.  Costs are asymmetric (present abstention ~0.25, absent commitment ~0.4-0.9),
so argmax (tau=1) is not always optimal.

IDENTIFIABILITY (measured).  Gamma is identified by the MARGINAL relevance law; beta and
kappa by the CONDITIONAL (cue) law.  Fitting both on the conditional alone leaves Gamma
unidentified: |dCE/dGamma_logit| drops from 7e-3 to 2e-4, and the required decay gets
supplied by write-dilution (-omega sum_{j>i} s_j, a count proxy for age) instead of time.
`--schedule probe` separates them, and then LOCKS the law:

    P: cue-free CE -> {omega, Gamma, kappa}   identifies the log-dose law
    A: cue CE      -> {beta, kappa}           recalibrates the readout
    B: cue CE      -> {eps_bar, beta, kappa}  refines it; omega and Gamma stay frozen

P alone is a diagnostic, NOT a recipe: its optimum puts alpha << e^kappa, so retrieval
collapses until A restores the balance.  Note kappa has two jobs -- marginal null-odds in
P, conditional commit threshold in A/B -- and one scalar cannot do both, so `prior ce`
rising after P is expected; Gamma survives because it is frozen.  (beta, kappa) share a
flat ray: only (ln alpha - kappa)/beta is pinned, and beta grows because with
near-orthogonal atoms the true cue likelihood ratio really is enormous.

Default loss is `ce`: once the map saturates, d(u_map)/d(kappa) ~ 0, so an ungated cosine
term is inert while CE keeps pushing.

Deployment:
    p, tau, cfg = load_checkpoint("hopmem_theta.pt")
    mem = HopMem(p, cfg.get("d", d), tau=tau)
    mem.write(z, s, t)
    u = mem.retrieve(q, t_q)
"""
import argparse
import math
import os
import random
import re

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------- parameters
def inv_softplus(y):
    """log(e^y - 1), stable for large y (math.expm1 overflows above ~709)."""
    y = float(y)
    return math.log(math.expm1(y)) if y < 20.0 else y + math.log1p(-math.exp(-y))


def logit(p):
    return math.log(p / (1.0 - p))


class Params(nn.Module):
    """Shared {omega, Gamma, eps_bar}; gibbs readout {beta, kappa}; power readout {eta, chi}."""

    def __init__(self, read="gibbs", omega=-math.log(0.95), Gamma=1e-3, eps_bar=0.01,
                 beta=16.0, kappa=math.log(1e-4), eta=1.0, chi=0.01,
                 diffusion=True, pure_writes=False, dtype=torch.float32):
        super().__init__()
        t = lambda v: nn.Parameter(torch.tensor(float(v), dtype=dtype))
        self.read, self.diffusion, self.pure_writes = read, diffusion, pure_writes
        self.a_omega = t(inv_softplus(omega))
        self.a_Gamma = t(inv_softplus(Gamma))
        self.a_eps = t(logit(eps_bar))
        self.a_beta, self.a_kappa = t(inv_softplus(beta)), t(kappa)
        self.a_eta, self.a_chi = t(inv_softplus(eta)), t(logit(chi))

    @property
    def omega(self): return F.softplus(self.a_omega)

    @property
    def Gamma(self): return F.softplus(self.a_Gamma) if self.diffusion else self.a_Gamma * 0

    @property
    def eps_bar(self): return self.a_eps * 0 if self.pure_writes else torch.sigmoid(self.a_eps)

    @property
    def beta(self): return F.softplus(self.a_beta)

    @property
    def kappa(self): return self.a_kappa

    @property
    def eta(self): return F.softplus(self.a_eta)

    @property
    def chi(self): return torch.sigmoid(self.a_chi)

    def shared(self): return {"a_omega", "a_Gamma", "a_eps"}

    def law_names(self): return {"a_omega", "a_Gamma"}

    def read_names(self): return {"a_beta", "a_kappa"} if self.read == "gibbs" else {"a_eta", "a_chi"}

    def theta(self):
        d = {k: float(getattr(self, k).detach()) for k in ("omega", "Gamma", "eps_bar")}
        if self.read == "gibbs":
            kap = float(self.kappa.detach())
            d.update(beta=float(self.beta.detach()), kappa=kap,
                     cue_mass=math.exp(kap) if kap < 700 else float("inf"))
        else:
            d.update(eta=float(self.eta.detach()), chi=float(self.chi.detach()),
                     gamma=float((self.chi / self.eta).detach()))
        return d


def load_checkpoint(path, device="cpu"):
    """Load a v3.6 checkpoint.  Returns (Params, gate_tau, config)."""
    try:
        ck = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ck = torch.load(path, map_location=device)
    cfg = ck.get("config", {}) if isinstance(ck, dict) else {}
    read = ck.get("read", "gibbs") if isinstance(ck, dict) else "gibbs"
    p = Params(read=read, diffusion=cfg.get("diffusion", True),
               pure_writes=cfg.get("pure_writes", False)).to(device)
    state = ck.get("state", ck) if isinstance(ck, dict) else ck
    p.load_state_dict(state)
    tau = ck.get("gate_tau") if isinstance(ck, dict) else None
    return p, tau, cfg


# ----------------------------------------------------------------- log-doses (closed form)
def log_injected(p, s):
    """ln[a(1-eps)]; clamp_min keeps s=0 finite (gradient 0 through the clamped branch)."""
    a = -torch.expm1(-p.omega * s.clamp_min(0))
    return torch.log((a * (1 - p.eps_bar * (1 - a))).clamp_min(1e-30))


def log_doses(p, s, t, t_q):
    """(B,T) log-masses at query time.  Diffuse-then-write: later writes dilute by
    exp(-omega s_j), elapsed time by exp(-Gamma dt).  Additive, so no sequential loop."""
    later = (s.flip(-1).cumsum(-1).flip(-1) - s).clamp_min(0)
    age = (t_q.unsqueeze(-1) - t).clamp_min(0)
    return log_injected(p, s) - (p.omega * later + p.Gamma * age)


# ----------------------------------------------------------------- readouts
def make_step(Z, log_alpha, q, p, auto_gamma=False):
    """Returns step(u) -> (v, E).  u+ = v/|v|; E never increases (both readouts)."""
    if p.read == "gibbs":
        beta, kappa = p.beta, p.kappa

        def step(u):
            ov = torch.einsum("btd,bd->bt", Z, u)
            cq = (q * u).sum(-1, keepdim=True)
            logits = torch.cat([beta * ov + log_alpha, beta * cq + kappa], -1)
            w = logits.softmax(-1)
            v = torch.einsum("bt,btd->bd", w[..., :-1], Z) + w[..., -1:] * q
            E = -torch.logsumexp(logits, -1) / beta
            return v, E
        return step

    alpha = log_alpha.exp()
    eta = p.eta
    if auto_gamma:  # corrected 6.4 as a design rule: gamma above the cross-talk edge
        T, d = Z.shape[1], Z.shape[2]
        gamma = (3.0 * alpha.sum(-1) / T * math.sqrt(max(T - 1, 1) / d)).reshape(-1, 1)
    else:
        gamma = (p.chi / p.eta).reshape(-1, 1)

    def step(u):
        ov = torch.einsum("btd,bd->bt", Z, u)
        Mu = torch.einsum("btd,bt->bd", Z, alpha * ov)
        E = (-0.5 * (alpha * ov * ov).sum(-1, keepdim=True)
             - gamma * (u * q).sum(-1, keepdim=True)).squeeze(-1)
        return u + eta * (Mu + gamma.reshape(-1, 1) * q), E
    return step


def retrieve(step, q, N=20, eps_fall=1e-8, diagnostics=False):
    """Masked denominator: a non-finite unselected branch must not poison grads."""
    u, Es, guard, gmin = q, [], 0, float("inf")
    for _ in range(N):
        v, E = step(u)
        n = v.norm(dim=-1, keepdim=True)
        bad = (~torch.isfinite(n)) | (n < eps_fall)
        if diagnostics:
            Es.append(E.detach())
            guard += int(bad.sum())
            gmin = min(gmin, float(n.detach().min()))
        n_safe = torch.where(bad, torch.ones_like(n), n).clamp_min(eps_fall)
        u = torch.where(bad, q, v / n_safe)
    if not diagnostics:
        return u
    Es.append(step(u)[1].detach())
    E = torch.stack(Es)
    tol = 100 * torch.finfo(E.dtype).eps * E.abs().max().clamp_min(1.0)
    return u, {"E": E, "E_up": float((E[1:] > E[:-1] + tol).float().mean()),
               "guard": guard, "gmin": gmin}


def read(Z, log_alpha, q, p, N=20, diagnostics=False, auto_gamma=False):
    return retrieve(make_step(Z, log_alpha, q, p, auto_gamma), q, N, diagnostics=diagnostics)


# ----------------------------------------------------------------- geometry / diagnostics
def top_eigvec(Z, alpha):
    """Top eigenpair of M via the T x T Gram matrix (centroid baseline)."""
    A = Z * alpha.clamp_min(0).sqrt().unsqueeze(-1)
    lam, w = torch.linalg.eigh(A @ A.transpose(1, 2))
    return F.normalize(torch.einsum("bt,btd->bd", w[..., -1], A), dim=-1), lam[..., -1]


def quad_stability(Z, alpha, k, gamma):
    """Corrected 6.4, entirely in factored T x T form (no d x d matrix).

    B = [sqrt(alpha_j) P z_j]_{j != k},  G = B^T B,  v_j = sqrt(alpha_j) (z_j.z_k).
    Then P M_oth P = B B^T, x_perp = B v, H B = B (m I - G) with m = mu + gamma, so
      margin      = m - lam_max(G)                (>0 iff the target is a strict minimum)
      floor_spec  = sqrt(v^T G v) / (alpha_k + gamma)      (the v2.0.4 formula)
      floor_hess  = sqrt(w^T G w),  w = (m I - G)^{-1} v   (the corrected floor)
    """
    B, T, d = Z.shape
    onehot = F.one_hot(k, T).to(alpha)
    a_oth = alpha * (1 - onehot)
    sa = a_oth.clamp_min(0).sqrt()
    z_k = Z.gather(1, k[:, None, None].expand(-1, 1, d)).squeeze(1)
    c = torch.einsum("btd,bd->bt", Z, z_k)
    G0 = Z @ Z.transpose(1, 2)
    G = sa[..., :, None] * sa[..., None, :] * (G0 - c[..., :, None] * c[..., None, :])
    v = sa * c
    mu = (alpha * c * c).sum(-1)
    g = torch.as_tensor(gamma, dtype=alpha.dtype, device=alpha.device).reshape(-1)
    m = mu + g
    lam = torch.linalg.eigvalsh(G)[..., -1]
    margin = m - lam
    a_k = alpha.gather(1, k[:, None]).squeeze(1)
    vGv = torch.einsum("bi,bij,bj->b", v, G, v)
    fs = vGv.clamp_min(0).sqrt() / (a_k + g)
    eye = torch.eye(T, dtype=G.dtype, device=G.device).expand_as(G)
    w = torch.linalg.solve(m[:, None, None] * eye - G, v.unsqueeze(-1)).squeeze(-1)
    wGw = torch.einsum("bi,bij,bj->b", w, G, w)
    fh = wGw.clamp_min(0).sqrt()
    fh = torch.where(margin > 0, fh, torch.full_like(fh, float("nan")))
    return margin, fs, fh


def gibbs_basin(log_alpha, k, beta, kappa):
    """Conservative basin radius: cue-vs-atom logit tie at u = q."""
    d = log_alpha.gather(1, k[:, None]).squeeze(1) - kappa
    x = (1 - d / beta).clamp(-1, 1)
    return torch.where(d > 0, torch.acos(x), torch.zeros_like(x))


def gibbs_floor(Z, log_alpha, q, k, beta, kappa):
    """First-order residual angle at the atom-k fixed point: |sum_{i != k} (p_i/p_k) P_k z_i|.
    ||P_k z_i|| ~ 1, so this scales as e^{-beta} sqrt(T-1), NOT e^{-beta} sqrt((T-1)/d)."""
    B, T, d = Z.shape
    z_k = Z.gather(1, k[:, None, None].expand(-1, 1, d)).squeeze(1)
    c = torch.einsum("btd,bd->bt", Z, z_k)
    cq = (q * z_k).sum(-1, keepdim=True)
    logits = torch.cat([beta * c + log_alpha, beta * cq + kappa], -1)
    lk = (beta + log_alpha.gather(1, k[:, None])).squeeze(1)
    r = torch.exp(logits - lk[:, None])
    r = r * (1 - F.one_hot(k, T + 1).to(r))
    PkZ = Z - c.unsqueeze(-1) * z_k[:, None, :]
    Pkq = q - cq * z_k
    x = torch.einsum("bt,btd->bd", r[..., :T], PkZ) + r[..., T:] * Pkq
    return x.norm(dim=-1)


def nn_read(Z, q):
    idx = torch.einsum("btd,bd->bt", Z, q).argmax(-1)
    return Z.gather(1, idx[:, None, None].expand(-1, 1, Z.shape[-1])).squeeze(1)


# decision grid: 19 log-spaced thresholds, index 15 is exactly tau = 1 (argmax)
GATE_TAUS = tuple(math.exp(-6.0 + 0.4 * i) for i in range(19))
GATE_ARGMAX_I = 15


def gate_gap(Z, log_alpha, q, p):
    """max_k logit_k - logit_null at u = q.  The decision statistic."""
    lg = gibbs_logits(Z, log_alpha, q, p)
    return lg[..., :Z.shape[1]].max(-1).values - lg[..., -1]


def gate_loss(gap, loss_map, loss_q, tau):
    """Commit to the map iff gap > ln tau; else return the cue.
    tau -> 0 reproduces the ungated map, tau -> inf reproduces cue-only."""
    return torch.where(gap > math.log(tau), loss_map, loss_q)


# ----------------------------------------------------------------- loss / posterior
def retrieval_loss(u, z_target):
    """1 - |c| with the Clarke subgradient, sign(0) := +1.  Clamp kills float32 c > 1."""
    c = (u * z_target).sum(-1).clamp(-1, 1)
    sign = torch.where(c.detach() >= 0, torch.ones_like(c), -torch.ones_like(c))
    return 1 - sign * c


def gibbs_logits(Z, log_alpha, q, p, u=None, cue_free=False):
    """Posterior logits over [atoms..., cue].  The cue atom is the 'no match' hypothesis,
    so kappa is its log-prior.  At u = q this is the vMF-mixture likelihood
    P(k | q) ~ alpha_k exp(beta z_k.q).  cue_free=True leaves the prior alone (Gamma probe)."""
    u = q if u is None else u
    b = 0.0 if cue_free else p.beta
    return torch.cat([b * torch.einsum("btd,bd->bt", Z, u) + log_alpha,
                      b * (q * u).sum(-1, keepdim=True) + p.kappa], -1)


# ----------------------------------------------------------------- streaming path
class HopMem:
    """Literal state machine in log-atom form.  Diffusion is a subtractive logit drift;
    a write appends an atom and shifts the rest by ln(1-a) = -omega*s exactly.

    `retrieve` applies the SAME gated decision as `evaluate`: decide at u=q, then descend.
    Pass tau=None to get the raw map (diagnostic only -- see the hysteresis note)."""

    def __init__(self, p, d, dtype=torch.float64, device="cpu", tau=None):
        self.p, self.d, self.dtype, self.device, self.tau = p, d, dtype, device, tau
        self.Z, self.la, self.S, self.t = [], [], 0.0, 0.0

    def _f(self, name):
        return float(getattr(self.p, name).detach())

    def _advance(self, t):
        dt = max(t - self.t, 0.0)
        if dt:
            g = self._f("Gamma") * dt
            self.la = [x - g for x in self.la]
            self.S *= math.exp(-g)
        self.t = max(t, self.t)

    def write(self, z, s, t):
        self._advance(t)
        z = F.normalize(z.to(dtype=self.dtype, device=self.device), dim=-1)
        om = self._f("omega")
        s = max(s, 0.0)
        a = -math.expm1(-om * s)
        eps = self._f("eps_bar") * (1 - a)
        self.la = [x - om * s for x in self.la] + [math.log(max(a * (1 - eps), 1e-30))]
        self.Z.append(z)
        self.S = (1 - a) * self.S + a * (1 - eps)

    def reinforce(self, u, s_ret, C, beta_bar):
        """Spec 11, masked denominator.  Appends u as an atom of mass beta."""
        mu = float(u @ self.W @ u)
        tau_r = 1 / self.d + (1 - 1 / self.d) * (-math.expm1(-self._f("omega") * max(s_ret, 0.0)))
        den = 1.0 - mu if mu < tau_r else 1.0
        beta = C * min(beta_bar, max(0.0, tau_r - mu) / den)
        if beta > 0:
            self.la = [x + math.log1p(-beta) for x in self.la] + [math.log(beta)]
            self.Z.append(F.normalize(u.to(dtype=self.dtype, device=self.device), dim=-1))
            self.S = (1 - beta) * self.S + beta
        return beta

    @property
    def Zs(self):
        return torch.stack(self.Z) if self.Z else torch.zeros(0, self.d, dtype=self.dtype, device=self.device)

    @property
    def log_alpha(self):
        return torch.tensor(self.la, dtype=self.dtype, device=self.device)

    @property
    def M(self):
        Z, a = self.Zs, self.log_alpha.exp()
        return torch.einsum("td,t,te->de", Z, a, Z) if len(a) else torch.zeros(
            self.d, self.d, dtype=self.dtype, device=self.device)

    @property
    def W(self):
        return (1 - self.S) / self.d * torch.eye(self.d, dtype=self.dtype, device=self.device) + self.M

    def retrieve(self, q, t, N=20, tau=None):
        self._advance(t)
        q = F.normalize(q.to(dtype=self.dtype, device=self.device), dim=-1)[None]
        Z, la = self.Zs[None], self.log_alpha[None]
        thr = self.tau if tau is None else tau
        if self.p.read == "gibbs" and thr is not None and Z.shape[1] > 0:
            if float(gate_gap(Z, la, q, self.p)[0]) <= math.log(thr):
                return q[0]                      # abstain: no match, return the cue
        return read(Z, la, q, self.p, N)[0]


# ----------------------------------------------------------------- data
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")


def _explode(batch, indices, a):
    out = {"sentence": [], "cue": [], "doc": []}
    for did, text in zip(indices, batch["text"]):
        rng = random.Random(a.seed * 1_000_003 + did)
        sents = [x for x in _SENT.split(re.sub(r"\s+", " ", text))
                 if a.min_words <= len(x.split()) <= a.max_words]
        if len(sents) > a.max_sent:
            off = rng.randrange(len(sents) - a.max_sent + 1)
            sents = sents[off:off + a.max_sent]
        for sent in sents:
            w = sent.split()
            k = max(1, round(a.cue_frac * len(w)))
            st = rng.randrange(len(w) - k + 1)
            out["sentence"].append(sent)
            out["cue"].append(" ".join(w[st:st + k]))
            out["doc"].append(did)
    return out


def build_corpus(a):
    import numpy as np
    from datasets import Dataset, load_dataset, load_from_disk

    tag = (f"{a.model.split('/')[-1]}_{a.dataset.split('/')[-1]}_b{a.n_books}_k{a.max_sent}"
           f"_w{a.min_words}-{a.max_words}_c{a.cue_frac}_s{a.seed}")
    path = os.path.join(a.cache_dir, tag)
    if os.path.isdir(path):
        ds = load_from_disk(path)
    else:
        rows, col = [], None
        for ex in load_dataset(a.dataset, split="train", streaming=True).take(a.n_books):
            col = col or next(k for k in ex if k.lower() == "text")
            rows.append({"text": ex[col]})
        ds = Dataset.from_list(rows).map(_explode, batched=True, batch_size=8, with_indices=True,
                                         remove_columns=["text"], fn_kwargs={"a": a})
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(a.model)
        enc = lambda xs: model.encode(xs, batch_size=256, normalize_embeddings=True, convert_to_numpy=True)
        ds = ds.map(lambda b: {"z": enc(b["sentence"]), "q": enc(b["cue"])}, batched=True, batch_size=1024)
        ds.save_to_disk(path)
    ds = ds.with_format("numpy")
    f = lambda c: torch.from_numpy(np.stack(ds[c])).float()
    return f("z"), f("q"), torch.from_numpy(np.asarray(ds["doc"]))


def synthetic_corpus(a):
    """No network: clustered unit vectors, cues at a known angle.  rho = shared-mean cosine."""
    g = torch.Generator().manual_seed(a.seed)
    nd, per, d = max(4, a.n_books // 2), 200, a.d
    mean = F.normalize(torch.randn(nd, d, generator=g), dim=-1)
    Z = F.normalize(math.sqrt(a.rho) * mean.repeat_interleave(per, 0)
                    + math.sqrt(1 - a.rho) * torch.randn(nd * per, d, generator=g), dim=-1)
    phi = 0.4 + 0.9 * torch.rand(nd * per, 1, generator=g)
    e = torch.randn(nd * per, d, generator=g)
    e = F.normalize(e - (e * Z).sum(-1, keepdim=True) * Z, dim=-1)
    Q = phi.cos() * Z + phi.sin() * e
    return Z, Q, torch.arange(nd).repeat_interleave(per)


class EpisodeSampler:
    """s and dt independent.  Target ~ s_i exp(-query_decay * age_i).  With prob p_absent the
    cue refers to an item that is NOT stored: the correct output is the cue itself, and the
    label is the cue atom (index T).  Without these, kappa -> -inf is optimal."""

    def __init__(self, Zs, Qs, doc, T, same_passage=False, query_decay=0.2, uniform_target=False,
                 p_absent=0.2):
        self.Zs, self.Qs, self.T, self.P = Zs, Qs, T, len(Zs)
        self.lam, self.uniform, self.p_absent = query_decay, uniform_target, p_absent
        self.starts = None
        if same_passage and self.P >= T:
            self.starts = (doc[:self.P - T + 1] == doc[T - 1:]).nonzero().squeeze(1)

    def sample(self, B):
        T, dev = self.T, self.Zs.device
        if self.starts is not None and len(self.starts):
            ri = torch.randint(len(self.starts), (B,), device=self.starts.device)
            idx = self.starts[ri].to(dev)[:, None] + torch.arange(T, device=dev)
        else:
            idx = torch.randint(self.P, (B, T), device=dev)
        s = torch.empty(B, T, device=dev).exponential_()
        t = torch.empty(B, T, device=dev).exponential_().cumsum(-1)
        t_q = t[:, -1] + torch.empty(B, device=dev).exponential_()
        logits = torch.zeros(B, T, device=dev) if self.uniform else torch.log(s) - self.lam * (t_q[:, None] - t)
        tgt = torch.multinomial(logits.softmax(-1), 1).squeeze(1)
        ar = torch.arange(B, device=dev)
        Z = self.Zs[idx]
        q, z_t = self.Qs[idx[ar, tgt]], Z[ar, tgt]
        absent = torch.rand(B, device=dev) < self.p_absent
        if self.p_absent > 0:
            j = torch.randint(self.P, (B,), device=dev)
            qo = F.normalize(self.Qs[j], dim=-1)
            q = torch.where(absent[:, None], qo, q)
            z_t = torch.where(absent[:, None], qo, z_t)
            tgt = torch.where(absent, torch.full_like(tgt, T), tgt)
        return {"Z": Z, "s": s, "t": t, "t_q": t_q, "tgt": tgt, "z_t": z_t, "q": q, "absent": absent}


# ----------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(p, ep, N=20, auto_gamma=False, tau=None):
    """tau=None sweeps the grid on THIS batch (labelled 'eval'); pass a train-fitted tau
    for an honest number."""
    la = log_doses(p, ep["s"], ep["t"], ep["t_q"])
    Z, z, q, lab = ep["Z"], ep["z_t"], ep["q"], ep["tgt"]
    T = Z.shape[1]
    u, info = read(Z, la, q, p, N, diagnostics=True, auto_gamma=auto_gamma)
    uL = read(Z, la, q, p, 10 * N, auto_gamma=auto_gamma)
    v1, _ = top_eigvec(Z, la.exp())
    cos = lambda x, y: (x * y).sum(-1).abs().clamp(max=1.0)
    S = la.exp().sum(-1)
    per = 1 - cos(u, z)
    per_q = 1 - cos(q, z)
    pres, absm = lab < T, lab >= T
    out = {"loss": per.mean().item(), "loss10N": (1 - cos(uL, z)).mean().item(),
           "cue": per_q.mean().item(), "centroid": (1 - cos(v1, z)).mean().item(),
           "nn": (1 - cos(nn_read(Z, q), z)).mean().item(),
           "capture": (cos(u, v1) > cos(u, z)).float().mean().item(),
           "E_up": info["E_up"], "guard": info["guard"], "S": S.mean().item()}
    if bool(pres.any()):
        out["map_pres"] = per[pres].mean().item()
    if bool(absm.any()):
        out["map_abs"] = per[absm].mean().item()
    if p.read == "gibbs":
        lg = gibbs_logits(Z, la, q, p)
        am = lg.argmax(-1)
        out["ce"] = F.cross_entropy(lg, lab).item()
        out["p_true"] = lg.softmax(-1).gather(1, lab[:, None]).mean().item()
        out["acc"] = (am == lab).float().mean().item()
        lg0 = gibbs_logits(Z, la, q, p, cue_free=True)
        out["ce_prior"] = F.cross_entropy(lg0, lab).item()
        out["p_prior"] = lg0.softmax(-1).gather(1, lab[:, None]).mean().item()
        null = am == T
        if bool(pres.any()):
            out["commit"] = (am[pres] == lab[pres]).float().mean().item()
            out["basin"] = gibbs_basin(la[pres], lab[pres], p.beta, p.kappa).mean().item()
            out["floor"] = gibbs_floor(Z[pres], la[pres], q[pres], lab[pres], p.beta, p.kappa).mean().item()
            out["phi"] = torch.acos((q[pres] * z[pres]).sum(-1).clamp(-1, 1)).mean().item()
        if bool(absm.any()):
            out["abstain"] = null[absm].float().mean().item()
            out["hyst"] = (null & (cos(u, q) < 0.9))[absm].float().mean().item()
        gap = lg[..., :T].max(-1).values - lg[..., T]
        sw = [float(gate_loss(gap, per, per_q, t).mean()) for t in GATE_TAUS]
        if tau is None:
            # a flat sweep means the map == cue-return: report argmax, not the grid edge
            bi = (GATE_ARGMAX_I if (max(sw) - min(sw)) < 1e-9
                  else min(range(len(sw)), key=lambda i: sw[i]))
            tsel = GATE_TAUS[bi]
        else:
            tsel = tau
        out["gate_argmax"] = sw[GATE_ARGMAX_I]
        out["gate_best"] = float(gate_loss(gap, per, per_q, tsel).mean())
        out["gate_tau"], out["gate_src"] = tsel, ("eval" if tau is None else "train")
        pg = 1 - cos(torch.where((gap > math.log(tsel))[:, None], u, q), z)
        if bool(pres.any()):
            out["gate_pres"] = pg[pres].mean().item()
        if bool(absm.any()):
            out["gate_abs"] = pg[absm].mean().item()
    return out


@torch.no_grad()
def calibrate_tau(p, ep, N=20, auto_gamma=False):
    """Calibrate the single decision threshold on a FIXED batch.  Same rule as evaluate."""
    if p.read != "gibbs":
        return None
    la = log_doses(p, ep["s"], ep["t"], ep["t_q"])
    Z, z, q = ep["Z"], ep["z_t"], ep["q"]
    u = read(Z, la, q, p, N, auto_gamma=auto_gamma)
    cos = lambda x, y: (x * y).sum(-1).abs().clamp(max=1.0)
    per = 1 - cos(u, z)
    per_q = 1 - cos(q, z)
    gap = gate_gap(Z, la, q, p)
    sw = [float(gate_loss(gap, per, per_q, t).mean()) for t in GATE_TAUS]
    if (max(sw) - min(sw)) < 1e-9:
        return GATE_TAUS[GATE_ARGMAX_I]
    return GATE_TAUS[min(range(len(sw)), key=lambda i: sw[i])]


def report(tag, p, ep, N, auto_gamma=False, tau=None):
    r = evaluate(p, ep, N, auto_gamma, tau=tau)
    g = lambda k: r.get(k, float("nan"))
    th = " ".join(f"{k}={v:.4g}" for k, v in p.theta().items())
    head = (f"[{tag}] gate {g('gate_best'):.4f} (tau {g('gate_tau'):.3g}, {r.get('gate_src','-')}) | "
            if p.read == "gibbs" else f"[{tag}] ")
    print(head + f"map {r['loss']:.4f} @10N {r['loss10N']:.4f} | cue {r['cue']:.4f} "
          f"centroid {r['centroid']:.4f} nn {r['nn']:.4f} | capture {r['capture']:.2f} | "
          f"E_up {r['E_up']:.0e} guard {r['guard']} | S {r['S']:.2f}")
    if p.read == "gibbs":
        print(f"      ce {g('ce'):.3f} p_true {g('p_true'):.3f} acc {g('acc'):.3f} | "
              f"prior ce {g('ce_prior'):.3f} p {g('p_prior'):.3f} | "
              f"commit {g('commit'):.3f} abstain {g('abstain'):.3f} hyst {g('hyst'):.3f} | "
              f"phi {g('phi'):.2f} basin {g('basin'):.2f} floor {g('floor'):.1e}")
        print(f"      map pres/abs {g('map_pres'):.4f}/{g('map_abs'):.4f} | "
              f"gate pres/abs {g('gate_pres'):.4f}/{g('gate_abs'):.4f} | "
              f"gate argmax {g('gate_argmax'):.4f}")
    print(f"      {th}")


@torch.no_grad()
def calibrate(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    dt = torch.float64
    p = Params(read=a.read, dtype=dt, diffusion=not a.no_diffusion, pure_writes=a.pure_writes).to(dev)
    d, T, B, N = a.d, a.T, a.batch, a.N
    ar = torch.arange(B, device=dev)
    print("read:", a.read, "theta:", {k: round(v, 6) for k, v in p.theta().items()}, f"d={d} T={T} N={N}")
    for arm in ("R", "A"):
        Z = F.normalize(torch.randn(B, T, d, generator=torch.Generator(device="cpu").manual_seed(a.seed),
                                    dtype=dt).to(dev), dim=-1)
        if arm == "A":
            mu = F.normalize(torch.randn(B, 1, d, dtype=dt, device=dev), dim=-1)
            Z = F.normalize(math.sqrt(a.rho) * mu + math.sqrt(1 - a.rho) * Z, dim=-1)
        s = torch.empty(B, T, dtype=dt, device=dev).exponential_()
        t = torch.empty(B, T, dtype=dt, device=dev).exponential_().cumsum(-1)
        t_q = t[:, -1] + torch.empty(B, dtype=dt, device=dev).exponential_()
        tgt = torch.randint(T, (B,), device=dev)
        z_t = Z[ar, tgt]
        la = log_doses(p, s, t, t_q)
        alpha = la.exp()
        v1, lam1 = top_eigvec(Z, alpha)
        margin, fs, fh = quad_stability(Z, alpha, tgt, 0.01)
        print(f"\nArm {arm}: S={float(alpha.sum(-1).mean()):.3f} "
              f"lam1/S={float((lam1 / alpha.sum(-1)).mean()):.2f} "
              f"target mass {float(alpha.gather(1, tgt[:, None]).mean()):.4f}")
        fhm = float(torch.nanmean(fh)) if bool(torch.isfinite(fh).any()) else float("nan")
        fsm = float(fs.mean())
        note = " (linearisation invalid)" if fsm > 0.5 else ""
        print(f"   quadratic readout: target stable in {float((margin > 0).float().mean()):.0%} of draws; "
              f"floor(spec) {fsm:.3f}{note} / floor(Hessian, stable) {fhm:.3f} rad")
        if a.read == "gibbs":
            print(f"   gibbs: basin {float(gibbs_basin(la, tgt, p.beta, p.kappa).mean()):.3f} rad, "
                  f"floor {float(gibbs_floor(Z, la, z_t, tgt, p.beta, p.kappa).mean()):.2e} rad")
        for phi in a.phis:
            e = torch.randn(B, d, dtype=dt, device=dev)
            e = F.normalize(e - (e * z_t).sum(-1, keepdim=True) * z_t, dim=-1)
            q = math.cos(phi) * z_t + math.sin(phi) * e
            u, info = read(Z, la, q, p, N, diagnostics=True, auto_gamma=a.auto_gamma)
            uL = read(Z, la, q, p, 10 * N, auto_gamma=a.auto_gamma)
            ang = lambda x: torch.acos((x * z_t).sum(-1).clamp(-1, 1))
            cos = lambda x: (x * z_t).sum(-1).abs().clamp(max=1.0)
            print(f"   phi={phi:.2f} closure@N {float(((phi - ang(u)) / phi).mean()):+.2f} "
                  f"@10N {float(((phi - ang(uL)) / phi).mean()):+.2f} | "
                  f"loss {float((1 - cos(u)).mean()):.4f} "
                  f"(cue {1 - math.cos(phi):.4f}, centroid {float((1 - cos(v1)).mean()):.3f}, "
                  f"nn {float((1 - cos(nn_read(Z, q))).mean()):.3f}) | capture "
                  f"{float(((u * v1).sum(-1).abs() > cos(u)).float().mean()):.2f} | E_up {info['E_up']:.0e}")


def train(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    Zs, Qs, doc = synthetic_corpus(a) if a.synthetic else build_corpus(a)
    Zs, Qs = Zs.to(dev), Qs.to(dev)
    P = len(Zs)
    b = (doc[1:] != doc[:-1]).nonzero().squeeze(1) + 1
    b = b[b >= int(0.9 * P)]
    cut = int(b[0]) if len(b) else int(0.9 * P)
    kw = dict(T=a.T, same_passage=a.same_passage, query_decay=a.query_decay,
              uniform_target=a.uniform_target, p_absent=a.p_absent)
    tr = EpisodeSampler(Zs[:cut], Qs[:cut], doc[:cut], **kw)
    ev = EpisodeSampler(Zs[cut:], Qs[cut:], doc[cut:], **kw)
    print(f"{P} items, d={Zs.shape[1]}, train {cut} / eval {P-cut}, device {dev}, "
          f"read {a.read}, schedule {a.schedule}, loss {a.loss}, p_absent {a.p_absent}, "
          f"cal_batch {a.cal_batch if a.read == 'gibbs' and a.gate_tau is None else '-'}")
    p = Params(read=a.read, diffusion=not a.no_diffusion, pure_writes=a.pure_writes).to(dev)

    # One fixed calibration batch for tau.  Do not resample per report: the saved tau must
    # be the same scalar used by the reported gate and by HopMem.retrieve.
    if p.read != "gibbs":
        get_tau = lambda: None
    elif a.gate_tau is not None:
        get_tau = lambda: float(a.gate_tau)
    else:
        cal_ep = tr.sample(a.cal_batch)
        get_tau = lambda: calibrate_tau(p, cal_ep, a.N, a.auto_gamma)

    mode = a.loss if a.loss in ("cos", "ce", "both") else "ce"
    if a.read != "gibbs":
        if a.loss != "cos":
            print("note: CE requires the gibbs readout; using cosine loss for power.")
        mode = "cos"
        if a.schedule == "probe":
            print("note: the probe needs the gibbs posterior; falling back to --schedule staged.")
            a.schedule = "staged"
    if a.schedule == "probe":
        if a.loss == "prior":
            print("note: --loss prior alone kills retrieval; probe uses it for stage P only.")
        stages = [("P", {"a_omega", "a_Gamma", "a_kappa"}, "prior", a.steps_p),
                  ("A", p.read_names(), mode, a.steps_a),
                  ("B", (p.shared() | p.read_names()) - p.law_names(), mode, a.steps_b)]
    else:
        stages = [("A", p.read_names(), mode, a.steps_a), ("B", None, mode, a.steps_b)]

    ev_ep = ev.sample(a.eval_batch)
    report("init  ", p, ev_ep, a.N, a.auto_gamma, tau=get_tau())

    for stage, names, smode, steps in stages:
        if stage == "P":
            print("  stage P fits the MARGINAL relevance law (cue-free). Retrieval metrics are "
                  "not meaningful until stage A recalibrates (beta, kappa).")
        if stage == "B" and a.schedule == "probe":
            print("  stage B refines the readout only: omega and Gamma stay frozen at P's values.")
        for n, prm in p.named_parameters():
            prm.requires_grad_(names is None or n in names)
        opt = torch.optim.Adam([x for x in p.parameters() if x.requires_grad], lr=a.lr)
        for it in range(steps):
            ep = tr.sample(a.batch)
            la = log_doses(p, ep["s"], ep["t"], ep["t_q"])
            u = read(ep["Z"], la, ep["q"], p, a.N, auto_gamma=a.auto_gamma)
            l_cos = retrieval_loss(u, ep["z_t"]).mean()
            l_ce = torch.zeros((), device=dev)
            if p.read == "gibbs" and smode in ("ce", "both", "prior"):
                l_ce = F.cross_entropy(gibbs_logits(ep["Z"], la, ep["q"], p,
                                                    cue_free=(smode == "prior")), ep["tgt"])
            if p.read != "gibbs" or smode == "cos":
                loss = l_cos
            elif smode in ("ce", "prior"):
                loss = l_ce
            else:
                loss = l_ce + a.cos_w * l_cos
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([x for x in p.parameters() if x.requires_grad], 5.0)
            opt.step()
            if a.log_every > 0 and it % a.log_every == 0:
                th = " ".join(f"{k}={v:.4g}" for k, v in p.theta().items())
                print(f"  stage {stage} step {it:4d} ce {l_ce.item():.4f} cos {l_cos.item():.4f}  {th}")
        report(f"stage {stage}", p, ev_ep, a.N, a.auto_gamma, tau=get_tau())

    with torch.no_grad():
        sub = {k: v[:64] for k, v in ev_ep.items()}
        la = log_doses(p, sub["s"], sub["t"], sub["t_q"])
        m = sub["tgt"] < sub["Z"].shape[1]
        if bool(m.any()):
            margin, fs, fh = quad_stability(sub["Z"][m], la[m].exp(), sub["tgt"][m], 0.01)
            fhm = float(torch.nanmean(fh)) if bool(torch.isfinite(fh).any()) else float("nan")
            print(f"quadratic-target stable in {float((margin > 0).float().mean()):.0%} of eval draws "
                  f"(floor spec {float(fs.mean()):.3f} / Hessian {fhm:.3f} rad)")
            if p.read == "gibbs":
                gf = gibbs_floor(sub["Z"][m], la[m], sub["q"][m], sub["tgt"][m], p.beta, p.kappa).mean()
                print(f"gibbs floor {float(gf):.2e} rad")
        r = evaluate(p, ev_ep, a.N, a.auto_gamma)
        if p.read == "gibbs" and r.get("hyst", 0) > 0.1:
            print(f"note: hysteresis {r['hyst']:.0%} -- the map leaves the cue on absent trials "
                  "where the posterior said 'no match'. HopMem.retrieve applies the gate.")
        if p.read == "gibbs" and r.get("p_true", 0) > 0.99:
            print("WARNING p_true ~ 1: cues are unambiguous, so the prior has no gradient. "
                  "Use --schedule probe, --same_passage, wider cue noise, or larger T.")
    if not a.uniform_target:
        dts = torch.cat([ev_ep["t"][:, :1], ev_ep["t"][:, 1:] - ev_ep["t"][:, :-1]], -1)
        Es, Edt = float(ev_ep["s"].mean()), float(dts.mean())
        om, G = float(p.omega.detach()), float(p.Gamma.detach())
        print(f"data decay {a.query_decay} vs learned Gamma {G:.4f} | per-event decay: "
              f"write-dilution {om*Es:.4f} + time {G*Edt:.4f} = {om*Es + G*Edt:.4f} "
              f"(data {a.query_decay*Edt:.4f})")
        if G * Edt < 0.5 * a.query_decay * Edt:
            print("note: decay is carried by write-dilution (a count proxy), not by time. "
                  "Run --schedule probe to identify Gamma.")

    tau_final = get_tau()
    if p.read == "gibbs":
        rd = evaluate(p, ev_ep, a.N, a.auto_gamma, tau=tau_final)
        print(f"[deploy] gate {rd['gate_best']:.4f} (tau {tau_final:.3g}) | "
              f"map {rd['loss']:.4f} | cue {rd['cue']:.4f} nn {rd['nn']:.4f} | "
              f"pres/abs {rd.get('gate_pres', float('nan')):.4f}/{rd.get('gate_abs', float('nan')):.4f}")
    ck = {"state": p.state_dict(), "gate_tau": tau_final, "read": a.read,
          "config": {"diffusion": not a.no_diffusion, "pure_writes": a.pure_writes,
                     "d": int(Zs.shape[1]), "N": a.N, "schedule": a.schedule,
                     "query_decay": a.query_decay, "p_absent": a.p_absent,
                     "gate_taus": GATE_TAUS},
          "theta": p.theta()}
    torch.save(ck, a.out)
    print(f"saved {a.out} (gate_tau={tau_final})")


# ----------------------------------------------------------------- selftest
def selftest():
    torch.manual_seed(0)
    dt, res = torch.float64, []

    def check(name, cond, extra=""):
        print(("PASS " if cond else "FAIL ") + name + ("  " + extra if extra else ""))
        res.append(bool(cond))

    with torch.no_grad():
        d, T, B = 24, 7, 3
        p = Params(omega=0.7, Gamma=0.3, eps_bar=0.3, beta=6.0, kappa=-5.0, eta=2.0, chi=0.2, dtype=dt)
        s = torch.rand(B, T, dtype=dt) * 2
        t = torch.cumsum(torch.rand(B, T, dtype=dt) + 0.1, -1)
        t_q = t[:, -1] + 0.5
        Z = F.normalize(torch.randn(B, T, d, dtype=dt), dim=-1)
        la = log_doses(p, s, t, t_q)
        om, eb = float(p.omega.detach()), float(p.eps_bar.detach())

        # 1. invariants + closed form, streaming vs batch
        mem, eye = HopMem(p, d, dt), torch.eye(d, dtype=dt)
        prim = inv = 0.0
        for i in range(T):
            mem._advance(float(t[0, i]))
            W0 = mem.W.clone()
            mem.write(Z[0, i], float(s[0, i]), float(t[0, i]))
            a = -math.expm1(-om * float(s[0, i]))
            eps = eb * (1 - a)
            Wexp = (1 - a) * W0 + a * ((1 - eps) * torch.outer(Z[0, i], Z[0, i]) + eps * eye / d)
            prim = max(prim, float((mem.W - Wexp).abs().max()))
            ev = torch.linalg.eigvalsh(mem.W)
            inv = max(inv, abs(float(mem.W.trace()) - 1), max(0.0, (1 - mem.S) / d - float(ev.min())))
        check("write == convex primitive; PSD, tr=1, lam_min >= (1-S)/d", prim < 1e-12 and inv < 1e-12)
        mem._advance(float(t_q[0]))
        err_la = float((mem.log_alpha - la[0]).abs().max())
        err_M = float((mem.M - torch.einsum("td,t,te->de", Z[0], la[0].exp(), Z[0])).abs().max())
        check("closed-form log-doses == streaming", err_la < 1e-12 and err_M < 1e-12,
              f"dla={err_la:.1e} dM={err_M:.1e}")

        q = F.normalize(torch.randn(B, d, dtype=dt), dim=-1)
        k = torch.zeros(B, dtype=torch.long)

        # 2. gibbs energy descent for every beta, and empty memory -> cue
        worst = 0.0
        for b_ in (0.05, 1.0, 6.0, 50.0, 1000.0):
            pb = Params(beta=b_, kappa=-5.0, dtype=dt)
            worst = max(worst, read(Z, la, q, pb, 30, diagnostics=True)[1]["E_up"])
        check("gibbs: energy never increases (beta 0.05 .. 1000)", worst == 0.0)
        empty = torch.full_like(la, -69.0)
        u0 = read(Z, empty, q, p, 20)
        check("gibbs: empty memory returns the cue", float((u0 - q).abs().max()) < 1e-12)

        # 3. permutation and scale invariance
        perm = torch.randperm(T)
        up = read(Z[:, perm], la[:, perm], q, p, 20)
        ps = Params(beta=6.0, kappa=-5.0 + 3.0, dtype=dt)
        us = read(Z, la + 3.0, q, ps, 20)
        u = read(Z, la, q, p, 20)
        check("gibbs: atom-order and log-dose shift invariance",
              float((up - u).abs().max()) < 1e-12 and float((us - u).abs().max()) < 1e-12)

        # 4. corrected 6.4: factored T x T diagnostics == dense d x d brute force
        alpha = la.exp()
        gam = 0.01
        margin, fs, fh = quad_stability(Z, alpha, k, gam)
        mm, ff, hh = [], [], []
        for b in range(B):
            zk = Z[b, 0]
            M = torch.einsum("td,t,te->de", Z[b], alpha[b], Z[b])
            Moth = M - alpha[b, 0] * torch.outer(zk, zk)
            P = eye - torch.outer(zk, zk)
            mu = float(zk @ M @ zk)
            Hd = (mu + gam) * P - P @ Moth @ P + (mu + gam) * torch.outer(zk, zk)
            ev = torch.linalg.eigvalsh(Hd)
            mm.append(float(ev.min()))
            x = P @ Moth @ zk
            ff.append(float(x.norm()) / (float(alpha[b, 0]) + gam))
            hh.append(float(torch.linalg.solve(Hd, x).norm()) if float(ev.min()) > 0 else float("nan"))
        ok_m = float((margin - torch.tensor(mm, dtype=dt)).abs().max()) < 1e-10
        ok_f = float((fs - torch.tensor(ff, dtype=dt)).abs().max()) < 1e-10
        ok_h = bool(torch.allclose(fh, torch.tensor(hh, dtype=dt), atol=1e-9, equal_nan=True))
        check("corrected 6.4 factored == dense (margin, both floors)", ok_m and ok_f and ok_h)

        # 5. quadratic floor O(0.1) rad and a saddle; gibbs floor exponentially small
        d2, T2, al = 384, 20, 0.05
        Z2 = F.normalize(torch.randn(1, T2, d2, dtype=dt), dim=-1)
        la2 = torch.full((1, T2), math.log(al), dtype=dt)
        k2 = torch.zeros(1, dtype=torch.long)
        mg, fsp, _ = quad_stability(Z2, la2.exp(), k2, 0.01)
        pq = Params(read="power", eta=1.0, chi=0.01, dtype=dt)
        uq = read(Z2, la2, Z2[:, 0], pq, 3000)
        ang_q = float((uq - Z2[:, 0]).norm())
        pg = Params(beta=6.0, kappa=math.log(al) - 2.0, dtype=dt)
        ug = read(Z2, la2, Z2[:, 0], pg, 300)
        ang_g = float((ug - Z2[:, 0]).norm())
        pred = float(gibbs_floor(Z2, la2, Z2[:, 0], k2, pg.beta, pg.kappa))
        check("quadratic: target is a saddle at the spec init", bool(mg[0] < 0), f"margin={float(mg[0]):+.4f}")
        check("quadratic converged angle >> gibbs converged angle", ang_q > 0.1 and ang_g < 0.05 * ang_q,
              f"power {ang_q:.3f} rad vs gibbs {ang_g:.2e} rad (spec floor {float(fsp[0]):.3f})")
        check("gibbs floor formula predicts the measured residual", abs(ang_g - pred) < 0.25 * pred + 1e-6,
              f"measured {ang_g:.3e} predicted {pred:.3e}")
        pg2 = Params(beta=12.0, kappa=math.log(al) - 2.0, dtype=dt)
        ug2 = read(Z2, la2, Z2[:, 0], pg2, 300)
        ang_g2 = float((ug2 - Z2[:, 0]).norm())
        check("gibbs floor decays exponentially in beta (x2 beta -> /50 at least)",
              ang_g2 < ang_g / 50.0,
              f"beta=6 {ang_g:.3e} -> beta=12 {ang_g2:.3e} (ratio {ang_g/max(ang_g2,1e-30):.0f}, "
              f"e^6={math.exp(6):.0f})")

        # 6. gibbs basin radius: commit inside, cue-return outside
        pb = Params(beta=16.0, kappa=math.log(1e-4), dtype=dt)
        tb = float(gibbs_basin(la2, k2, pb.beta, pb.kappa)[0])
        ins = outs = True
        for phi in (0.3, 0.6, 0.8):
            e = F.normalize(torch.randn(1, d2, dtype=dt), dim=-1)
            e = F.normalize(e - (e * Z2[:, 0]).sum(-1, keepdim=True) * Z2[:, 0], dim=-1)
            qq = math.cos(phi) * Z2[:, 0] + math.sin(phi) * e
            uu = read(Z2, la2, qq, pb, 50)
            ins &= float((uu * Z2[:, 0]).sum()) > 0.99
        for phi in (1.3, 1.5):
            e = F.normalize(torch.randn(1, d2, dtype=dt), dim=-1)
            e = F.normalize(e - (e * Z2[:, 0]).sum(-1, keepdim=True) * Z2[:, 0], dim=-1)
            qq = math.cos(phi) * Z2[:, 0] + math.sin(phi) * e
            uu = read(Z2, la2, qq, pb, 50)
            outs &= float((uu * qq).sum()) > 0.999
        check("gibbs basin: commit inside, cue-return outside", ins and outs, f"theta_basin={tb:.3f} rad")

        # 7. forgetting: basin vanishes at t = (ln alpha_0 - kappa)/Gamma
        G = 0.3
        pf = Params(beta=16.0, kappa=math.log(1e-4), Gamma=G, dtype=dt)
        la_t = lambda age: la2 - G * age
        tstar = (math.log(al) - float(pf.kappa.detach())) / G
        b_in = float(gibbs_basin(la_t(tstar - 0.5), k2, pf.beta, pf.kappa)[0])
        b_out = float(gibbs_basin(la_t(tstar + 0.5), k2, pf.beta, pf.kappa)[0])
        check("basin vanishes at (ln alpha_0 - kappa)/Gamma", b_in > 0.1 and b_out == 0.0,
              f"t*={tstar:.2f}, basin {b_in:.3f} -> {b_out:.3f}")

        # 8. power readout: descent, W/M equivalence, corrected floor in its valid regime
        A = torch.randn(d, 5, dtype=dt)
        Wr = A @ A.T / (A @ A.T).trace()
        worst = 0.0
        for e_ in (0.1, 10.0, 1e4):
            ge = 0.5 / e_
            st = lambda u, e_=e_, ge=ge: (u + e_ * (u @ Wr + ge * q),
                                          -0.5 * (u * (u @ Wr)).sum(-1) - ge * (u * q).sum(-1))
            worst = max(worst, retrieve(st, q, 30, diagnostics=True)[1]["E_up"])
        check("power: energy never increases (eta up to 1e4)", worst == 0.0)
        S, M = mem.S, mem.M
        W = mem.W
        eta, chi = 2.0, 0.2
        gam_, etap = chi / eta, eta / (1 + eta * (1 - S) / d)
        stW = lambda u: (u + eta * (u @ W + gam_ * q), -0.5 * (u * (u @ W)).sum(-1) - gam_ * (u * q).sum(-1))
        stM = lambda u: (u + etap * (u @ M + gam_ * q), -0.5 * (u * (u @ M)).sum(-1) - gam_ * (u * q).sum(-1))
        check("W-form == M-form at eta' (trajectory)",
              float((retrieve(stW, q, 20) - retrieve(stM, q, 20)).abs().max()) < 1e-12)
        gam_t = 0.15
        mg_t, _, fh_t = quad_stability(Z2, la2.exp(), k2, gam_t)
        pt = Params(read="power", eta=1.0, chi=gam_t, dtype=dt)
        ut = read(Z2, la2, Z2[:, 0], pt, 5000)
        ang_t = float((ut - Z2[:, 0]).norm())
        check("corrected Hessian floor predicts the converged angle (stable regime)",
              bool((mg_t > 0).all()) and abs(ang_t - float(fh_t[0])) < 0.2 * float(fh_t[0]) + 0.02,
              f"measured {ang_t:.3f} vs Hessian floor {float(fh_t[0]):.3f} rad")
        mg_a, _, _ = quad_stability(Z2, la2.exp(), k2, 3 * al * math.sqrt((T2 - 1) / d2))
        check("auto-gamma puts the target in the stable regime", bool((mg_a > 0).all()),
              f"margin={float(mg_a[0]):+.4f}")

        # 9. steady state, reinforcement
        m2 = HopMem(Params(omega=0.5, Gamma=0.2, pure_writes=True, dtype=dt), d, dt)
        for kk in range(1, 400):
            m2.write(Z[0, 0], 1.0, float(kk))
        post = m2.S
        m2._advance(400.0)
        check("steady state S*_post / S*_pre", abs(post - 0.78160) < 1e-5 and abs(m2.S - 0.63992) < 1e-5)
        m3 = HopMem(p, d, dt)
        m3.write(Z[0, 0], 1.0, 1.0)
        uu = F.normalize(torch.randn(d, dtype=dt), dim=0)
        mu = float(uu @ m3.W @ uu)
        tau_r = 1 / d + (1 - 1 / d) * (-math.expm1(-om * 5.0))
        m3.reinforce(uu, 5.0, 1.0, 0.999)
        check("reinforcement hits tau, keeps invariants",
              (mu >= tau_r or abs(float(uu @ m3.W @ uu) - tau_r) < 1e-12)
              and abs(float(m3.W.trace()) - 1) < 1e-12 and float(torch.linalg.eigvalsh(m3.W).min()) > 0)

        # 10. prior probe: cue-free logits are exactly the log-doses plus the cue prior
        pp = Params(omega=0.4, Gamma=0.25, dtype=dt)
        lgp = gibbs_logits(Z, la, q, pp, cue_free=True)
        ok_p = float((lgp[..., :T] - la).abs().max()) < 1e-12 and \
            float((lgp[..., T] - pp.kappa).abs().max()) < 1e-12
        check("prior probe: cue-free logits == [ln alpha, kappa]", ok_p)

        # 11. gate limits: tau -> 0 is the map, tau -> inf is cue-only
        gap = torch.randn(64, dtype=dt)
        lmap, lq = torch.rand(64, dtype=dt), torch.rand(64, dtype=dt)
        g0 = float(gate_loss(gap, lmap, lq, 1e-12).mean())
        gi = float(gate_loss(gap, lmap, lq, 1e12).mean())
        check("gate limits: tau->0 == map loss, tau->inf == cue loss",
              abs(g0 - float(lmap.mean())) < 1e-12 and abs(gi - float(lq.mean())) < 1e-12)

        # 12. the gate is NECESSARY: argmax says null at u=q, the map commits anyway.
        Zh = torch.tensor([[[math.cos(0.7), math.sin(0.7)]]], dtype=dt)
        qh = torch.tensor([[1.0, 0.0]], dtype=dt)
        lah = torch.full((1, 1), -3.0, dtype=dt)
        ph = Params(beta=16.0, kappa=-6.5, dtype=dt)
        lgh = gibbs_logits(Zh, lah, qh, ph)
        uh = read(Zh, lah, qh, ph, 50)
        check("gate necessary: argmax=null at u=q but the map commits (hysteresis)",
              int(lgh.argmax(-1)[0]) == 1 and float((uh * Zh[:, 0]).sum()) > 0.99,
              f"logits atom {float(lgh[0, 0]):.3f} vs null {float(lgh[0, 1]):.3f}, "
              f"cos(u_map, z)={float((uh * Zh[:, 0]).sum()):.4f}")

        # 13. streaming gated retrieve == batch gated read (deployed == evaluated)
        mgate = HopMem(Params(beta=16.0, kappa=-8.0, dtype=dt), d, dt, tau=0.5)
        for i in range(T):
            mgate.write(Z[0, i], float(s[0, i]), float(t[0, i]))
        qg = F.normalize(torch.randn(d, dtype=dt), dim=0)
        ustream = mgate.retrieve(qg, float(t_q[0]))
        Zb, lb, qb = mgate.Zs[None], mgate.log_alpha[None], qg[None]
        ub = read(Zb, lb, qb, mgate.p, 20)
        gb = gate_gap(Zb, lb, qb, mgate.p)
        ubg = torch.where((gb > math.log(0.5))[:, None], ub, qb)[0]
        check("streaming gated retrieve == batch gated read",
              float((ustream - ubg).abs().max()) < 1e-12,
              f"gap={float(gb[0]):+.3f}, committed={bool(gb[0] > math.log(0.5))}")
        u_abst = mgate.retrieve(qg, float(t_q[0]), tau=1e12)
        u_comm = mgate.retrieve(qg, float(t_q[0]), tau=1e-12)
        check("streaming gate limits: tau->inf returns q, tau->0 returns the map",
              float((u_abst - qg).abs().max()) < 1e-12 and float((u_comm - ub[0]).abs().max()) < 1e-12)

    # 14. gradients, both readouts, float32
    for rd in ("gibbs", "power"):
        p32 = Params(read=rd)
        Zf = F.normalize(torch.randn(4, 10, 32), dim=-1)
        sf = torch.empty(4, 10).exponential_()
        tf = torch.empty(4, 10).exponential_().cumsum(-1)
        lf = log_doses(p32, sf, tf, tf[:, -1] + 0.3)
        qf = F.normalize(torch.randn(4, 32), dim=-1)
        retrieval_loss(read(Zf, lf, qf, p32, 20), Zf[:, 0]).mean().backward()
        act = p32.shared() | p32.read_names()
        ok = all(x.grad is not None and bool(torch.isfinite(x.grad))
                 for n, x in p32.named_parameters() if n in act)
        check(f"finite gradients for all active params ({rd})", ok)

    # 15. prior probe restores the Gamma gradient that the cue posterior kills
    p32 = Params(read="gibbs", beta=60.0)
    Zf = F.normalize(torch.randn(8, 12, 32), dim=-1)
    sf = torch.empty(8, 12).exponential_()
    tf = torch.empty(8, 12).exponential_().cumsum(-1)
    tq = tf[:, -1] + 0.3
    lf = log_doses(p32, sf, tf, tq)
    lab = torch.zeros(8, dtype=torch.long)
    qf = Zf[:, 0]
    lg = gibbs_logits(Zf, lf, qf, p32)
    g_cue = torch.autograd.grad(F.cross_entropy(lg, lab), p32.a_Gamma, retain_graph=True)[0]
    lg0 = gibbs_logits(Zf, lf, qf, p32, cue_free=True)
    g_pri = torch.autograd.grad(F.cross_entropy(lg0, lab), p32.a_Gamma)[0]
    check("prior probe restores the Gamma gradient that the cue posterior kills",
          float(g_pri.abs()) > 10.0 * float(g_cue.abs()) + 1e-9,
          f"|dCE/dGamma_logit| cue {float(g_cue.abs()):.2e} -> prior {float(g_pri.abs()):.2e}")

    assert all(res), "selftest failed"
    print(f"all {len(res)} passed")


# ----------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    for name in ("calibrate", "train"):
        c = sub.add_parser(name)
        c.add_argument("--read", choices=["gibbs", "power"], default="gibbs")
        c.add_argument("--auto_gamma", action="store_true")
        c.add_argument("--d", type=int, default=384)
        c.add_argument("--T", type=int, default=20)
        c.add_argument("--N", type=int, default=20)
        c.add_argument("--batch", type=int, default=64)
        c.add_argument("--rho", type=float, default=0.3)
        c.add_argument("--seed", type=int, default=0)
        c.add_argument("--no_diffusion", action="store_true")
        c.add_argument("--pure_writes", action="store_true")
        if name == "calibrate":
            c.add_argument("--phis", type=float, nargs="+", default=[0.05, 0.1, 0.3, 0.6, 1.0])
        else:
            c.add_argument("--synthetic", action="store_true")
            c.add_argument("--dataset", default="sedthh/gutenberg_english")
            c.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
            c.add_argument("--n_books", type=int, default=100)
            c.add_argument("--max_sent", type=int, default=1000)
            c.add_argument("--min_words", type=int, default=8)
            c.add_argument("--max_words", type=int, default=40)
            c.add_argument("--cue_frac", type=float, default=0.5)
            c.add_argument("--cache_dir", default="hopmem_cache")
            c.add_argument("--eval_batch", type=int, default=256)
            c.add_argument("--cal_batch", type=int, default=1024,
                           help="fixed training batch size for gate tau calibration")
            c.add_argument("--lr", type=float, default=3e-2)
            c.add_argument("--schedule", choices=["staged", "probe"], default="staged",
                           help="probe = P(cue-free law) -> A(readout) -> B(readout, law frozen)")
            c.add_argument("--steps_p", type=int, default=800)
            c.add_argument("--steps_a", type=int, default=300)
            c.add_argument("--steps_b", type=int, default=700)
            c.add_argument("--log_every", type=int, default=50)
            c.add_argument("--same_passage", action="store_true")
            c.add_argument("--query_decay", type=float, default=0.2)
            c.add_argument("--uniform_target", action="store_true")
            c.add_argument("--p_absent", type=float, default=0.2)
            c.add_argument("--loss", choices=["cos", "ce", "both", "prior"], default="ce")
            c.add_argument("--cos_w", type=float, default=1.0)
            c.add_argument("--gate_tau", type=float, default=None,
                           help="fix the decision threshold instead of calibrating it")
            c.add_argument("--out", default="hopmem_theta.pt")
    a = ap.parse_args()
    {"selftest": selftest, "calibrate": lambda: calibrate(a), "train": lambda: train(a)}[a.cmd]()


if __name__ == "__main__":
    main()