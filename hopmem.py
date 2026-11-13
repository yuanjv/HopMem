#!/usr/bin/env python
"""HopMem v2.0.4 in PyTorch (single file).

    python hopmem.py selftest     # spec identities in float64 (no data, no network)
    python hopmem.py calibrate    # spec 17.3 two-arm calibration at the 7.2 init (synthetic, no data)
    python hopmem.py train        # HF dataset -> sentence embeddings -> episodes -> two-stage training

Design note. Retrieval depends only on the signal matrix M = sum_i alpha_i z_i z_i^T (isotropy lemma, 8.2),
and for an episode alpha has a closed form (6.4 + semigroup property of 5.1):

    alpha_i = a_i (1 - eps_i) * exp(-(omega * sum_{j>i} s_j + Gamma * (t_q - t_i)))

so training needs neither a sequential write loop nor a d x d state: memory is O(T d), matvec is O(T d).
`HopMem` below is the literal (M, S) state machine of the spec; `selftest` checks the two agree.

Loss: 1 - |c| with the exact Clarke subgradient (sign(0) := +1), so there is no smoothing constant (9).
"""
import argparse
import math
import os
import random
import re

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- parameters (spec 7)
def inv_softplus(y):  # ln(expm1(y)), 15.9
    return math.log(math.expm1(y))


def logit(p):
    return math.log(p / (1.0 - p))


class HopMemParams(nn.Module):
    """theta = {omega, Gamma, eta, chi, eps_bar} through unconstrained logits; gamma = chi / eta is derived (7.1).
    Defaults are the 7.2 init. `diffusion=False` / `pure_writes=True` are the explicit ablation off-flags (15.7)."""

    def __init__(self, omega=-math.log(0.95), Gamma=1e-3, eta=1.0, chi=0.01, eps_bar=0.01,
                 diffusion=True, pure_writes=False):
        super().__init__()
        self.a_omega = nn.Parameter(torch.tensor(inv_softplus(omega)))
        self.a_Gamma = nn.Parameter(torch.tensor(inv_softplus(Gamma)))
        self.a_eta = nn.Parameter(torch.tensor(inv_softplus(eta)))
        self.a_chi = nn.Parameter(torch.tensor(logit(chi)))
        self.a_eps = nn.Parameter(torch.tensor(logit(eps_bar)))
        self.diffusion, self.pure_writes = diffusion, pure_writes

    @property
    def omega(self): return F.softplus(self.a_omega)

    @property
    def Gamma(self): return F.softplus(self.a_Gamma) if self.diffusion else torch.zeros_like(self.a_Gamma)

    @property
    def eta(self): return F.softplus(self.a_eta)

    @property
    def chi(self): return torch.sigmoid(self.a_chi)

    @property
    def eps_bar(self): return torch.zeros_like(self.a_eps) if self.pure_writes else torch.sigmoid(self.a_eps)

    @property
    def gamma(self): return self.chi / self.eta

    def theta(self):
        return {k: float(getattr(self, k)) for k in ("omega", "Gamma", "eta", "chi", "eps_bar", "gamma")}


# --------------------------------------------------------------------------- doses and closed-form masses
def write_weight(p, s):
    """a = 1 - exp(-omega s) (6.1), expm1 form (15.9)."""
    return -torch.expm1(-p.omega * s)


def injected_mass(p, s):
    """a (1 - eps) with eps = eps_bar (1 - a) (6.2)."""
    a = write_weight(p, s)
    return a * (1 - p.eps_bar * (1 - a))


def coefficients(p, s, t, t_q):
    """Mass of each trace at query time. s, t: (B,T) importance and ascending event times; t_q: (B,) >= t[:, -1].
    Diffuse-then-write ordering (4.3): later writes dilute by exp(-omega s_j), elapsed time by exp(-Gamma dt) (5.4)."""
    later = (s.flip(-1).cumsum(-1).flip(-1) - s).clamp_min(0)  # sum_{j>i} s_j
    age = t_q.unsqueeze(-1) - t
    return injected_mass(p, s) * torch.exp(-(p.omega * later + p.Gamma * age))


# --------------------------------------------------------------------------- retrieval (spec 8)
def factored_matvec(Z, alpha):
    """u -> M u with M = sum_i alpha_i z_i z_i^T, O(T d).  Z: (B,T,d), alpha: (B,T), u: (B,d)."""
    return lambda u: torch.einsum("btd,bt->bd", Z, alpha * torch.einsum("btd,bd->bt", Z, u))


def retrieve(matvec, q, eta, chi, N=20, eps_fall=1e-8, diagnostics=False):
    """u_0 = q; u_{n+1} = normalize(u_n + eta (A u_n + gamma q)), gamma = chi / eta.
    A = M (preferred, 15.3) or W; E(u) = -1/2 u.Au - gamma u.q never increases, for any eta (8.3).
    With diagnostics=True returns (u, {'E', 'E_up', 'guard'}); E_up must be 0 up to rounding (a bug detector, 8.4)."""
    gamma = chi / eta
    energy = lambda x, Ax: (-0.5 * (x * Ax).sum(-1) - gamma * (x * q).sum(-1)).detach()
    u, E, guard = q, [], 0
    for _ in range(N):
        Au = matvec(u)
        if diagnostics:
            E.append(energy(u, Au))
        v = u + eta * (Au + gamma * q)
        n = v.norm(dim=-1, keepdim=True)
        fell = n < eps_fall  # unreachable: |v| >= 1 - chi > 0 (8.3)
        if diagnostics:
            guard += int(fell.sum())
        u = torch.where(fell, q, v / n.clamp_min(eps_fall))
    if not diagnostics:
        return u
    E.append(energy(u, matvec(u)))
    E = torch.stack(E)
    tol = 100 * torch.finfo(E.dtype).eps
    return u, {"E": E, "E_up": (E[1:] > E[:-1] + tol).float().mean().item(), "guard": guard}


# --------------------------------------------------------------------------- loss and baselines (9, 17.1)
def retrieval_loss(u, z_target):
    """1 - |u.z| with the Clarke subgradient at c = 0 (sign(0) := +1): full-strength gradient, no epsilon."""
    c = (u * z_target).sum(-1)
    sign = torch.where(c.detach() >= 0, torch.ones_like(c), -torch.ones_like(c))
    return 1 - sign * c


def top_eigvec(Z, alpha):
    """Top eigenvector / eigenvalue of M = A^T A (A = sqrt(alpha) Z) through the T x T Gram matrix (centroid baseline)."""
    A = Z * alpha.clamp_min(0).sqrt().unsqueeze(-1)
    lam, w = torch.linalg.eigh(A @ A.transpose(1, 2))
    return F.normalize(torch.einsum("bt,btd->bd", w[..., -1], A), dim=-1), lam[..., -1]


def tangent_margin(Z, alpha, z_t, tgt, gamma):
    """Local stability of the target as a fixed point for a cue exactly on it (see report: corrects 6.4).
    Riemannian Hessian of E at u = z_t:  H = (mu + gamma) P - P M_oth P,  P = I - z z^T,  mu = z^T M z.
    Returns lam_min(H) (<= 0: the target is a saddle, no stable floor exists), the corrected floor |H^-1 x_perp|
    (NaN where unstable) and the spec's floor |x_perp| / (alpha_t + gamma)."""
    B, T, d = Z.shape
    M = torch.einsum("btd,bt,bte->bde", Z, alpha, Z)
    a_t = alpha.gather(1, tgt[:, None]).squeeze(1)
    zz = z_t[:, :, None] * z_t[:, None, :]
    Moth = M - a_t[:, None, None] * zz
    P = torch.eye(d, device=Z.device, dtype=Z.dtype) - zz
    mu = torch.einsum("bd,bde,be->b", z_t, M, z_t)
    c = (mu + gamma)[:, None, None]
    H = c * P - P @ Moth @ P + c * zz  # z-direction set to +c so lam_min is the tangent minimum
    x = (P @ (Moth @ z_t.unsqueeze(-1))).squeeze(-1)
    lam_min = torch.linalg.eigvalsh(H)[:, 0]
    floor_H = torch.linalg.solve(H, x.unsqueeze(-1)).squeeze(-1).norm(dim=-1)
    floor_H = torch.where(lam_min > 0, floor_H, torch.full_like(floor_H, float("nan")))
    return lam_min, floor_H, x.norm(dim=-1) / (a_t + gamma)


# --------------------------------------------------------------------------- literal (M, S) state machine (3-6, 11)
class HopMem:
    """Streaming reference/inference path on the signal form: W = (1-S) I/d + M. Not differentiable (training uses
    `coefficients`). Order per event: advance clock (diffuse), write, optionally reinforce (4.3)."""

    def __init__(self, params, d, dtype=torch.float64, device="cpu"):
        self.p, self.d = params, d
        self.M = torch.zeros(d, d, dtype=dtype, device=device)
        self.S, self.t = 0.0, 0.0

    def _f(self, name):
        return float(getattr(self.p, name))

    def _advance(self, t):  # diffusion is scalar decay on (M, S): lazy clock (15.4)
        f = math.exp(-self._f("Gamma") * max(t - self.t, 0.0))
        self.M, self.S, self.t = self.M * f, self.S * f, max(t, self.t)

    @property
    def W(self):
        eye = torch.eye(self.d, dtype=self.M.dtype, device=self.M.device)
        return (1 - self.S) / self.d * eye + self.M

    def write(self, z, s, t):
        self._advance(t)
        z = F.normalize(z.to(self.M), dim=-1)
        a = -math.expm1(-self._f("omega") * max(s, 0.0))
        eps = self._f("eps_bar") * (1 - a)
        self.M = (1 - a) * self.M + a * (1 - eps) * torch.outer(z, z)
        self.S = (1 - a) * self.S + a * (1 - eps)

    def retrieve(self, q, t, N=20):
        self._advance(t)
        q = F.normalize(q.to(self.M), dim=-1)[None]
        eta, chi = self.p.eta.detach().to(q), self.p.chi.detach().to(q)
        return retrieve(lambda x: x @ self.M, q, eta, chi, N)[0]

    def reinforce(self, u, s_ret, C, beta_bar):
        """Optional (11), masked-denominator form (15.15). beta_bar in [0,1) is learnable in the spec."""
        mu = float(u @ self.W @ u)
        tau = 1 / self.d + (1 - 1 / self.d) * (-math.expm1(-self._f("omega") * s_ret))
        den = 1.0 - mu if mu < tau else 1.0
        beta = C * min(beta_bar, max(0.0, tau - mu) / den)
        self.M = (1 - beta) * self.M + beta * torch.outer(u, u)
        self.S = (1 - beta) * self.S + beta
        return beta

    def diagnostics(self):
        d, P = self.d, (1 - self.S ** 2) / self.d + float((self.M * self.M).sum())
        return {"S": self.S, "P_hat": (d * P - 1) / (d - 1), "half_life": math.log(2) / max(self._f("Gamma"), 1e-30)}


# --------------------------------------------------------------------------- data: HF datasets + sentence embeddings
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")  # crude splitter; swap in your own from the mergemem pipeline


def _explode(batch, indices, a):
    """books -> rows (sentence, cue, doc). cue = random contiguous span of cue_frac of the words (partial cue)."""
    out = {"sentence": [], "cue": [], "doc": []}
    for did, text in zip(indices, batch["text"]):
        rng = random.Random(a.seed * 1_000_003 + did)
        sents = [x for x in _SENT.split(re.sub(r"\s+", " ", text)) if a.min_words <= len(x.split()) <= a.max_words]
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
    """Returns unit-norm store embeddings Zs (P,d), cue embeddings Qs (P,d), doc ids (P,); contiguous per doc."""
    import numpy as np
    from datasets import Dataset, load_dataset, load_from_disk

    tag = f"{a.model.split('/')[-1]}_{a.dataset.split('/')[-1]}_b{a.n_books}_k{a.max_sent}_w{a.min_words}-{a.max_words}_c{a.cue_frac}_s{a.seed}"
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
    f = lambda col: torch.from_numpy(np.stack(ds[col])).float()
    return f("z"), f("q"), torch.from_numpy(np.asarray(ds["doc"]))


class EpisodeSampler:
    """Episodes per spec 10. Importance s and gaps dt are drawn independently (identifiability, 7.3). The target is
    drawn with prob ~ s_i * exp(-query_decay * age_i): importance is query frequency, relevance decays at a known
    data rate (so learned Gamma can be compared with it). `uniform_target` removes both effects."""

    def __init__(self, Zs, Qs, doc, T, same_passage=False, query_decay=0.2, uniform_target=False):
        self.Zs, self.Qs, self.T = Zs, Qs, T
        self.P, self.lam, self.uniform = len(Zs), query_decay, uniform_target
        self.starts = None
        if same_passage:  # T consecutive sentences of one book: topically related competitors
            self.starts = (doc[: self.P - T + 1] == doc[T - 1:]).nonzero().squeeze(1).to(Zs.device)

    def sample(self, B):
        T, dev = self.T, self.Zs.device
        if self.starts is not None:
            idx = self.starts[torch.randint(len(self.starts), (B,), device=dev)][:, None] + torch.arange(T, device=dev)
        else:
            idx = torch.multinomial(torch.ones(B, self.P, device=dev), T)
        s = torch.empty(B, T, device=dev).exponential_()
        t = torch.empty(B, T, device=dev).exponential_().cumsum(-1)
        t_q = t[:, -1] + torch.empty(B, device=dev).exponential_()
        logits = torch.zeros(B, T, device=dev) if self.uniform else torch.log(s) - self.lam * (t_q[:, None] - t)
        tgt = torch.multinomial(logits.softmax(-1), 1).squeeze(1)
        ar = torch.arange(B, device=dev)
        Z = self.Zs[idx]
        return {"Z": Z, "s": s, "t": t, "t_q": t_q, "tgt": tgt, "z_t": Z[ar, tgt], "q": self.Qs[idx[ar, tgt]]}


# --------------------------------------------------------------------------- evaluation, calibration, training
@torch.no_grad()
def evaluate(p, ep, N=20):
    alpha = coefficients(p, ep["s"], ep["t"], ep["t_q"])
    Z, z, q = ep["Z"], ep["z_t"], ep["q"]
    u, info = retrieve(factored_matvec(Z, alpha), q, p.eta, p.chi, N, diagnostics=True)
    v1, lam1 = top_eigvec(Z, alpha)
    cos = lambda x, y: (x * y).sum(-1).abs()
    nn_vec = Z[torch.arange(len(Z), device=Z.device), torch.einsum("btd,bd->bt", Z, q).argmax(-1)]
    S = alpha.sum(-1)
    return {"loss": (1 - cos(u, z)).mean().item(), "cue": (1 - cos(q, z)).mean().item(),
            "centroid": (1 - cos(v1, z)).mean().item(), "nn": (1 - cos(nn_vec, z)).mean().item(),
            "capture": (cos(u, v1) > cos(u, z)).float().mean().item(), "E_up": info["E_up"],
            "guard": info["guard"], "S": S.mean().item(), "lam1_over_S": (lam1 / S).mean().item()}


def report(tag, p, ep, N):
    r = evaluate(p, ep, N)
    th = " ".join(f"{k}={v:.4g}" for k, v in p.theta().items())
    print(f"[{tag}] loss {r['loss']:.4f} | cue-only {r['cue']:.4f} centroid {r['centroid']:.4f} nn {r['nn']:.4f} | "
          f"capture {r['capture']:.2f} | E_up {r['E_up']:.0e} guard {r['guard']} | S {r['S']:.2f} lam1/S {r['lam1_over_S']:.2f}\n"
          f"      {th}")


@torch.no_grad()
def calibrate(a):
    """Spec 17.3 at the 7.2 init. Arm R: random unit z. Arm A: shared mean direction with pairwise cosine ~ rho.
    Adds two columns the spec lacks: closure after 10N steps (is the target an attractor?) and the Hessian margin."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    p = HopMemParams().to(dev)
    d, T, B, N = a.d, a.T, a.batch, a.N
    ar = torch.arange(B, device=dev)
    print("theta:", {k: round(v, 5) for k, v in p.theta().items()}, f" d={d} T={T} N={N}")
    for arm in ("R", "A"):
        Zr = F.normalize(torch.randn(B, T, d, device=dev), dim=-1)
        if arm == "A":
            mu = F.normalize(torch.randn(B, 1, d, device=dev), dim=-1)
            Zr = F.normalize(math.sqrt(a.rho) * mu + math.sqrt(1 - a.rho) * Zr, dim=-1)
        s = torch.empty(B, T, device=dev).exponential_()
        t = torch.empty(B, T, device=dev).exponential_().cumsum(-1)
        t_q = t[:, -1] + torch.empty(B, device=dev).exponential_()
        tgt = torch.randint(T, (B,), device=dev)
        z_t = Zr[ar, tgt]
        alpha = coefficients(p, s, t, t_q)
        mv = factored_matvec(Zr, alpha)
        v1, lam1 = top_eigvec(Zr, alpha)
        lam_min, fH, fS = tangent_margin(Zr, alpha, z_t, tgt, p.gamma)
        print(f"\nArm {arm}: S={alpha.sum(-1).mean():.3f} lam1/S={(lam1 / alpha.sum(-1)).mean():.2f} "
              f"target mass {alpha.gather(1, tgt[:, None]).mean():.4f} gamma={float(p.gamma):.3g}\n"
              f"   target is a stable fixed point (Hessian>0) in {(lam_min > 0).float().mean():.0%} of episodes; "
              f"floor(spec) {fS.mean():.3f} rad, floor(Hessian, stable only) {torch.nanmean(fH):.3f} rad")
        for phi in a.phis:
            e = torch.randn(B, d, device=dev)
            e = F.normalize(e - (e * z_t).sum(-1, keepdim=True) * z_t, dim=-1)
            q = math.cos(phi) * z_t + math.sin(phi) * e
            u, info = retrieve(mv, q, p.eta, p.chi, N, diagnostics=True)
            uL = retrieve(mv, q, p.eta, p.chi, 10 * N)
            ang = lambda x: torch.acos((x * z_t).sum(-1).clamp(-1, 1))
            cos = lambda x: (x * z_t).sum(-1).abs()
            print(f"   phi={phi:.2f}  closure@N {((phi - ang(u)) / phi).mean():+.2f}  closure@10N {((phi - ang(uL)) / phi).mean():+.2f}  "
                  f"loss {(1 - cos(u)).mean():.3f} (cue-only {1 - math.cos(phi):.3f}, centroid {(1 - cos(v1)).mean():.3f})  "
                  f"capture {(((u * v1).sum(-1).abs()) > cos(u)).float().mean():.2f}  E_up {info['E_up']:.0e}")


def train(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    Zs, Qs, doc = build_corpus(a)
    Zs, Qs = Zs.to(dev), Qs.to(dev)
    P = len(Zs)
    b = (doc[1:] != doc[:-1]).nonzero().squeeze(1) + 1  # split at a book boundary: eval books are unseen
    b = b[b >= int(0.9 * P)]
    cut = int(b[0]) if len(b) else int(0.9 * P)
    kw = dict(T=a.T, same_passage=a.same_passage, query_decay=a.query_decay, uniform_target=a.uniform_target)
    tr, ev = EpisodeSampler(Zs[:cut], Qs[:cut], doc[:cut], **kw), EpisodeSampler(Zs[cut:], Qs[cut:], doc[cut:], **kw)
    print(f"{P} sentences, d={Zs.shape[1]}, train {cut} / eval {P - cut}, device {dev}")
    ev_ep = ev.sample(a.eval_batch)
    p = HopMemParams(diffusion=not a.no_diffusion, pure_writes=a.pure_writes).to(dev)
    report("init  ", p, ev_ep, a.N)

    for stage, names, steps in (("A", {"a_eta", "a_chi"}, a.steps_a), ("B", None, a.steps_b)):
        for n, prm in p.named_parameters():
            prm.requires_grad_(names is None or n in names)
        opt = torch.optim.Adam([x for x in p.parameters() if x.requires_grad], lr=a.lr)
        for it in range(steps):
            ep = tr.sample(a.batch)
            alpha = coefficients(p, ep["s"], ep["t"], ep["t_q"])
            u = retrieve(factored_matvec(ep["Z"], alpha), ep["q"], p.eta, p.chi, a.N)
            loss = retrieval_loss(u, ep["z_t"]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if it % a.log_every == 0:
                print(f"  stage {stage} step {it:4d} loss {loss.item():.4f}  eta {float(p.eta):.3f} chi {float(p.chi):.4f} "
                      f"omega {float(p.omega):.4f} Gamma {float(p.Gamma):.4f} eps {float(p.eps_bar):.4f}")
        report(f"stage {stage}", p, ev_ep, a.N)

    with torch.no_grad():
        al = coefficients(p, ev_ep["s"], ev_ep["t"], ev_ep["t_q"])[:64]
        lam_min, fH, fS = tangent_margin(ev_ep["Z"][:64], al, ev_ep["z_t"][:64], ev_ep["tgt"][:64], p.gamma)
        print(f"target stable (Hessian>0) in {(lam_min > 0).float().mean():.0%} of eval episodes; "
              f"floor spec {fS.mean():.3f} / Hessian {torch.nanmean(fH):.3f} rad")
    if not a.uniform_target:
        print(f"data-generating decay {a.query_decay} vs learned Gamma {float(p.Gamma):.4f} (omega {float(p.omega):.4f})")
    torch.save(p.state_dict(), a.out)
    print("saved", a.out)


# --------------------------------------------------------------------------- selftest (float64)
def selftest():
    torch.manual_seed(0)
    dt, results = torch.float64, []

    def check(name, cond):
        print(("PASS " if cond else "FAIL ") + name)
        results.append(bool(cond))

    d, T, B = 16, 7, 2
    p = HopMemParams(omega=0.7, Gamma=0.3, eta=2.0, chi=0.2, eps_bar=0.3).to(dt)
    s = torch.rand(B, T, dtype=dt) * 2
    t = torch.cumsum(torch.rand(B, T, dtype=dt) + 0.1, -1)
    t_q = t[:, -1] + 0.5
    Z = F.normalize(torch.randn(B, T, d, dtype=dt), dim=-1)
    alpha = coefficients(p, s, t, t_q)
    om, eb = float(p.omega), float(p.eps_bar)  # logits are float32, so use the realised values

    # 1. streaming (M,S) machine: write == the convex primitive on W; PSD, trace one, W >= (1-S) I/d
    mem, eye = HopMem(p, d, dt), torch.eye(d, dtype=dt)
    prim_err = inv_err = 0.0
    for i in range(T):
        mem._advance(float(t[0, i]))
        W0 = mem.W.clone()
        mem.write(Z[0, i], float(s[0, i]), float(t[0, i]))
        a = -math.expm1(-om * float(s[0, i]))
        eps = eb * (1 - a)
        Wexp = (1 - a) * W0 + a * ((1 - eps) * torch.outer(Z[0, i], Z[0, i]) + eps * eye / d)
        prim_err = max(prim_err, float((mem.W - Wexp).abs().max()))
        ev = torch.linalg.eigvalsh(mem.W)
        inv_err = max(inv_err, abs(float(mem.W.trace()) - 1), max(0.0, (1 - mem.S) / d - float(ev.min())))
    check("write equals W <- (1-a)W + aT", prim_err < 1e-12)
    check("W PSD, trace 1, lam_min >= (1-S)/d", inv_err < 1e-12)

    # 2. closed-form masses == streaming
    mem._advance(float(t_q[0]))
    check("closed-form alpha == streaming M", float((mem.M - torch.einsum("td,t,te->de", Z[0], alpha[0], Z[0])).abs().max()) < 1e-12)

    # 3. exact W/M reparameterization: W-form(eta) == M-form(eta'), trajectory-wise (8.2)
    S, M, W = mem.S, mem.M, mem.W
    q = F.normalize(torch.randn(1, d, dtype=dt), dim=-1)
    eta, chi = 2.0, 0.2
    gam, etap = chi / eta, eta / (1 + eta * (1 - S) / d)
    uW = retrieve(lambda u: u @ W, q, eta, chi, 20)
    uM = retrieve(lambda u: u @ M, q, etap, etap * gam, 20)
    check("W-form == M-form at eta'", float((uW - uM).abs().max()) < 1e-12)

    # 4. energy descent for any eta
    A = torch.randn(d, 5, dtype=dt)
    Wr = A @ A.T / (A @ A.T).trace()
    worst = max(retrieve(lambda u: u @ Wr, q, e, 0.5, 30, diagnostics=True)[1]["E_up"] for e in (0.1, 10.0, 1e4))
    check("energy never increases (eta up to 1e4)", worst == 0.0)

    # 5. steady state (5.3): omega=.5, Gamma=.2, s=dt=1, pure writes -> post .78160, pre .63992
    m2 = HopMem(HopMemParams(omega=0.5, Gamma=0.2, pure_writes=True).to(dt), d, dt)
    for k in range(1, 300):
        m2.write(Z[0, 0], 1.0, float(k))
    post = m2.S
    m2._advance(300.0)
    check("steady state S*_post, S*_pre", abs(post - 0.78160) < 1e-5 and abs(m2.S - 0.63992) < 1e-5)

    # 6. reinforcement attains tau at C = 1 and keeps the invariants (11)
    m3 = HopMem(p, d, dt)
    m3.write(Z[0, 0], 1.0, 1.0)
    u = F.normalize(torch.randn(d, dtype=dt), dim=0)
    mu = float(u @ m3.W @ u)
    tau = 1 / d + (1 - 1 / d) * (-math.expm1(-om * 5.0))
    m3.reinforce(u, 5.0, 1.0, 0.999)
    check("reinforcement: mu' = tau, PSD, trace 1", (mu >= tau or abs(float(u @ m3.W @ u) - tau) < 1e-12)
          and abs(float(m3.W.trace()) - 1) < 1e-12 and float(torch.linalg.eigvalsh(m3.W).min()) > 0)

    # 7. single-trace calibration formula (8.5): psi/phi = g/(g+chi) [1 - (1+g+chi)^-N]
    d2, phi = 64, 0.1
    z = torch.zeros(1, d2, dtype=dt)
    z[0, 0] = 1
    q2 = torch.zeros(1, d2, dtype=dt)
    q2[0, 0], q2[0, 1] = math.cos(phi), math.sin(phi)
    good = True
    for eta_, chi_, m_ in [(1.0, 0.01, 0.05), (10.0, 0.05, 0.01), (0.1, 0.005, 0.05)]:
        u2 = retrieve(lambda x: m_ * z * (x * z).sum(-1, keepdim=True), q2, eta_, chi_, 20)
        g = eta_ * m_
        form = g / (g + chi_) * (1 - (1 + g + chi_) ** -20)
        good &= abs((phi - math.acos(float((u2 * z).sum()))) / phi - form) < 0.01
    check("calibration formula", good)

    # 8. gradients reach all five parameters and are finite (float32)
    p32 = HopMemParams()
    Zf = F.normalize(torch.randn(4, 10, 32), dim=-1)
    sf = torch.empty(4, 10).exponential_()
    tf = torch.empty(4, 10).exponential_().cumsum(-1)
    al = coefficients(p32, sf, tf, tf[:, -1] + 0.3)
    qf = F.normalize(torch.randn(4, 32), dim=-1)
    retrieval_loss(retrieve(factored_matvec(Zf, al), qf, p32.eta, p32.chi, 20), Zf[:, 0]).mean().backward()
    check("finite gradients for omega, Gamma, eta, chi, eps_bar",
          all(x.grad is not None and bool(torch.isfinite(x.grad)) for x in p32.parameters()))

    assert all(results), "selftest failed"
    print("all passed")


# --------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    c = sub.add_parser("calibrate")
    c.add_argument("--d", type=int, default=384)
    c.add_argument("--T", type=int, default=20)
    c.add_argument("--N", type=int, default=20)
    c.add_argument("--batch", type=int, default=64)
    c.add_argument("--rho", type=float, default=0.3, help="Arm A pairwise cosine")
    c.add_argument("--phis", type=float, nargs="+", default=[0.05, 0.1, 0.3, 0.6, 1.0])
    c.add_argument("--seed", type=int, default=0)
    t = sub.add_parser("train")
    t.add_argument("--dataset", default="sedthh/gutenberg_english")
    t.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    t.add_argument("--n_books", type=int, default=100)
    t.add_argument("--max_sent", type=int, default=1000, help="max sentences kept per book")
    t.add_argument("--min_words", type=int, default=8)
    t.add_argument("--max_words", type=int, default=40)
    t.add_argument("--cue_frac", type=float, default=0.5, help="cue = random contiguous span of this fraction of words")
    t.add_argument("--cache_dir", default="hopmem_cache")
    t.add_argument("--T", type=int, default=20)
    t.add_argument("--N", type=int, default=20)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--eval_batch", type=int, default=256)
    t.add_argument("--lr", type=float, default=3e-2)
    t.add_argument("--steps_a", type=int, default=300)
    t.add_argument("--steps_b", type=int, default=700)
    t.add_argument("--log_every", type=int, default=50)
    t.add_argument("--same_passage", action="store_true", help="competitors = consecutive sentences of one book")
    t.add_argument("--query_decay", type=float, default=0.2, help="data-generating decay of target relevance")
    t.add_argument("--uniform_target", action="store_true")
    t.add_argument("--no_diffusion", action="store_true")
    t.add_argument("--pure_writes", action="store_true")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", default="hopmem_theta.pt")
    a = ap.parse_args()
    {"selftest": lambda: selftest(), "calibrate": lambda: calibrate(a), "train": lambda: train(a)}[a.cmd]()


if __name__ == "__main__":
    main()