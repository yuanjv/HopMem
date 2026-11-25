#!/usr/bin/env python
"""HopMem v2.3.1 -- debit-optimal compression: one derived comparison over two
evidence-monotone operations, an exact incremental Gram law, and the withdrawn
superiority claim.

    python hopmem.py selftest        # 44 checks (spec section 24)
    python hopmem.py calibrate --read gibbs
    python hopmem.py train --schedule probe --batch 256 --C 16 --c_min 4
    python hopmem_diag.py --ckpt hopmem_theta.pt --T_big 100 --caps 4 8 16 32 --dup_sweep

State W=(1-S)J+M, tr W=1, W>=0, S in [0,1); <=C log-atoms (z, ell, Lam) + S + lazy clock.
On overflow: (1) null-fuse gate-dead atoms (free, decision-free); (2) select the
minimizing pair AND operation of

    Delta(i,j,op) = (A/S) * delta_op,
    delta_fuse  = beta(1-r) = 2 beta w_i w_j (1-G)/(1+r),
    delta_evict = ln(A/alpha_h) = softplus(ell_l - ell_h).

Since 1/S is constant per episode, the argmin is over A*delta; computed in log space.
Merge: mu=m/r, ell'=logaddexp-delta_fuse, Lam'=Lam_i+Lam_j+delta_fuse.
Evict: delete the LIGHTER atom, heavier untouched, book delta_evict, mark destroyed.
Ledger Lam is in NATS and tracks merge debits only; episode ledger sums both kinds;
retained = exp(-Lam) if surviving, hard 0 if destroyed.
Chain bound: ell_c >= logaddexp(base) - Lam_c (exact for one merge, conservative for chains).
Gram: G_ik <- (w_i G_ik + w_j G_jk)/r from OLD rows, then G_ii <- 1, then zero row/col j.
Crossover (equal mass): theta* = 2 arccos(1 - ln2/beta). No new parameters.

PARAMETERIZATION NOTE: eps_bar and chi are sigmoid-parameterized and must lie strictly
in (0,1). The pure-write ablation is the flag pure_writes=True, NOT eps_bar=0.0.
"""
import argparse
import math
import os
import random
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

__version__ = "2.3.1"
EPS_R = 1e-3
NEG = -1e30
DOSE_FLOOR = 1e-30
TINY = 1e-30


def inv_softplus(y):
    y = float(y)
    return math.log(math.expm1(y)) if y < 20.0 else y + math.log1p(-math.exp(-y))


def logit(p):
    return math.log(p / (1.0 - p))


class Params(nn.Module):
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
    cfg.setdefault("C", 64)
    cfg.setdefault("version", ck.get("version", "?"))
    return p, tau, cfg


def log_injected(p, s):
    a = -torch.expm1(-p.omega * s.clamp_min(0))
    return torch.log((a * (1 - p.eps_bar * (1 - a))).clamp_min(DOSE_FLOOR))


def log_doses(p, s, t, t_q):
    later = (s.flip(-1).cumsum(-1).flip(-1) - s).clamp_min(0)
    age = (t_q.unsqueeze(-1) - t).clamp_min(0)
    return log_injected(p, s) - (p.omega * later + p.Gamma * age)


# ----------------------------------------------------------------- fusion math
def fusion_weights(la_i, la_j):
    return torch.sigmoid(la_i - la_j)


def resultant_deficit(w_i, one_minus_G, beta):
    """delta_fuse = beta(1-r), cancellation-free."""
    t = 2.0 * w_i * (1.0 - w_i) * one_minus_G.clamp_min(0.0)
    r = (1.0 - t).clamp_min(0.0).sqrt()
    omr = t / (1.0 + r)
    return beta * omr, r, omr


def fuse_pair_math(z_i, z_j, la_i, la_j, beta):
    w = fusion_weights(la_i, la_j)
    m = w.unsqueeze(-1) * z_i + (1.0 - w).unsqueeze(-1) * z_j
    r_dir = m.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    mu = m / r_dir
    G_ij = (z_i * z_j).sum(-1).clamp(-1.0, 1.0)
    delta, r, omr = resultant_deficit(w, 1.0 - G_ij, beta)
    la_new = torch.logaddexp(la_i, la_j) - delta
    return mu, la_new, r, delta, w, G_ij, omr, r_dir.squeeze(-1)


def ward_log_cost(la, G, alive, eps_r=EPS_R):
    """v2.2.1 ablation: log-Ward cost, +inf where masked."""
    li, lj = la.unsqueeze(2), la.unsqueeze(1)
    w = torch.sigmoid(li - lj)
    omg = (1.0 - G).clamp_min(0.0)
    t = 2.0 * w * (1.0 - w) * omg
    r2 = (1.0 - t).clamp_min(0.0)
    logc = li + lj - torch.logaddexp(li, lj) + omg.clamp_min(TINY).log()
    K = la.shape[1]
    eye = torch.eye(K, dtype=torch.bool, device=la.device)
    valid = alive.unsqueeze(2) & alive.unsqueeze(1) & (~eye) & (r2 >= eps_r * eps_r)
    return torch.where(valid, logc, torch.full_like(logc, float("inf"))), r2


def debit_candidates(la, G, alive, beta, eps_r=EPS_R):
    """Spec 4.5.  Returns (logDelta, is_fuse, d_fuse, d_evict, r, fuse_ok, valid).

    logDelta = logaddexp(li,lj) - log S + log(min(d_fuse, d_evict)), +inf where masked.
    is_fuse  = merge admissible (r>=eps_r) AND strictly cheaper than eviction.
    Eviction is admissible for EVERY alive pair, so a candidate always exists.
    """
    li, lj = la.unsqueeze(2), la.unsqueeze(1)
    logA = torch.logaddexp(li, lj)
    w = torch.sigmoid(li - lj)
    omg = (1.0 - G).clamp_min(0.0)
    t = 2.0 * w * (1.0 - w) * omg
    r = (1.0 - t).clamp_min(0.0).sqrt()
    d_fuse = beta * t / (1.0 + r)
    d_evict = F.softplus(-(li - lj).abs())          # ln(1 + alpha_l/alpha_h), stable
    fuse_ok = r >= eps_r
    is_fuse = fuse_ok & (d_fuse < d_evict)
    d_min = torch.where(is_fuse, d_fuse, d_evict)
    S = la.exp().sum(-1).clamp_min(DOSE_FLOOR)
    logD = logA - S.log()[:, None, None] + d_min.clamp_min(TINY).log()
    K = la.shape[1]
    eye = torch.eye(K, dtype=torch.bool, device=la.device)
    valid = alive.unsqueeze(2) & alive.unsqueeze(1) & (~eye)
    logD = torch.where(valid, logD, torch.full_like(logD, float("inf")))
    return logD, is_fuse, d_fuse, d_evict, r, fuse_ok, valid


def _gath(x, idx):
    B = x.shape[0]
    if x.dim() == 2:
        return x.gather(1, idx[:, None]).squeeze(1)
    return x.gather(1, idx[:, None, None].expand(B, 1, *x.shape[2:])).squeeze(1)


def simulate_fusion(p, ep, C, tau_dead=1.0, eps_r=EPS_R, policy="debit"):
    """Differentiable batched event loop: diffuse -> write -> compress-on-overflow.
    policy: debit (v2.3.1) | ward (v2.2.1) | lightest | random | delta."""
    Z0, s, t, t_q = ep["Z"], ep["s"], ep["t"], ep["t_q"]
    B, T, d = Z0.shape
    dev, dtype = Z0.device, Z0.dtype
    Cb = torch.full((B,), int(C), dtype=torch.long, device=dev) if isinstance(C, int) else C.to(dev).long()
    Cmax = int(Cb.max())
    statkeys = ("n_fuse", "n_evict", "n_dead", "sum_delta", "below_cross", "dead_mass")
    if int(Cb.min()) >= T:
        la = log_doses(p, s, t, t_q)
        alive = torch.ones(B, T, dtype=torch.bool, device=dev)
        prov = torch.eye(T, dtype=torch.bool, device=dev)[None].expand(B, T, T).contiguous()
        return {"Z": Z0, "la": la, "alive": alive, "Lam": torch.zeros(B, T, dtype=dtype, device=dev),
                "prov": prov, "destroyed": torch.zeros(B, T, dtype=torch.bool, device=dev),
                "label": ep["tgt"].clone(), "K": T, "S": la.exp().sum(-1), "Cb": Cb,
                "stats": {k: torch.zeros(B, dtype=dtype, device=dev) for k in statkeys}}
    K = min(T, Cmax + 1)
    ar = torch.arange(B, device=dev)
    Z = torch.zeros(B, K, d, dtype=dtype, device=dev)
    la = torch.full((B, K), NEG, dtype=dtype, device=dev)
    Lam = torch.zeros(B, K, dtype=dtype, device=dev)
    G = torch.zeros(B, K, K, dtype=dtype, device=dev)
    alive = torch.zeros(B, K, dtype=torch.bool, device=dev)
    prov = torch.zeros(B, K, T, dtype=torch.bool, device=dev)
    destroyed = torch.zeros(B, T, dtype=torch.bool, device=dev)
    st = {k: torch.zeros(B, dtype=dtype, device=dev) for k in statkeys}
    beta, kappa = p.beta, p.kappa
    thr_dead = kappa + math.log(tau_dead)
    t_prev = torch.zeros(B, dtype=dtype, device=dev)
    NEGt = torch.full((1,), NEG, dtype=dtype, device=dev)

    def kill(mask):
        nonlocal Z, la, Lam, alive, prov, destroyed, G
        if not bool(mask.any()):
            return
        destroyed = destroyed | (prov & mask[..., None]).any(1)
        st["dead_mass"] += (la.exp() * mask.to(dtype)).sum(1).detach()
        alive = alive & ~mask
        la = torch.where(alive, la, NEGt)
        Lam = Lam * alive.to(dtype)
        Z = Z * alive.to(dtype)[..., None]
        prov = prov & alive[..., None]
        keep = alive.to(dtype)
        G = G * keep[:, :, None] * keep[:, None, :]

    for i in range(T):
        g = p.Gamma * (t[:, i] - t_prev)
        la = torch.where(alive, la - g[:, None], NEGt)
        la = torch.where(alive, la - (p.omega * s[:, i])[:, None], NEGt)
        a = -torch.expm1(-p.omega * s[:, i].clamp_min(0))
        la_new = torch.log((a * (1 - p.eps_bar * (1 - a))).clamp_min(DOSE_FLOOR))
        slot = (~alive).long().argmax(1)
        oh = F.one_hot(slot, K).to(dtype)
        z_i = Z0[:, i]
        Z = Z + oh[..., None] * z_i[:, None, :]
        la = la * (1 - oh) + la_new[:, None] * oh
        Lam = Lam * (1 - oh)
        alive = alive | F.one_hot(slot, K).bool()
        prov = prov | (F.one_hot(slot, K).bool()[..., None]
                       & F.one_hot(torch.full((B,), i, device=dev), T).bool()[:, None, :])
        g_row = torch.einsum("bkd,bd->bk", Z, z_i)          # O(Cd) append row
        G = G.scatter(1, slot[:, None, None].expand(B, 1, K), g_row[:, None, :])
        G = G.scatter(2, slot[:, None, None].expand(B, K, 1), g_row[:, :, None])
        t_prev = t[:, i]
        count = alive.long().sum(1)
        if not bool((count > Cb).any()):
            continue
        dead = alive & (la <= thr_dead) & (count > Cb)[:, None]
        st["n_dead"] += dead.to(dtype).sum(1)
        kill(dead)
        count = alive.long().sum(1)

        while bool((count > Cb).any()):
            act = count > Cb
            logD, is_fuse, d_fuse, d_evict, r, fuse_ok, valid = debit_candidates(la, G, alive, beta, eps_r)
            if policy == "ward":
                logD, _ = ward_log_cost(la, G, alive, eps_r)
                logD = torch.where(torch.isfinite(logD), logD,
                                   torch.where(valid, logD.new_tensor(0.0), logD))
                is_fuse = fuse_ok
            elif policy == "delta":
                S = la.exp().sum(-1).clamp_min(DOSE_FLOOR)
                base = torch.logaddexp(la.unsqueeze(2), la.unsqueeze(1)) - S.log()[:, None, None]
                lf = base + d_fuse.clamp_min(TINY).log()
                le = base + d_evict.clamp_min(TINY).log()
                logD = torch.where(valid & fuse_ok, lf,
                                   torch.where(valid, le, torch.full_like(lf, float("inf"))))
                is_fuse = fuse_ok
            elif policy == "random":
                rnd = torch.rand_like(logD)
                logD = torch.where(valid, rnd, torch.full_like(rnd, float("inf")))
                is_fuse = fuse_ok
            if policy == "lightest":
                bj = torch.where(alive, la, torch.full_like(la, float("inf"))).argmin(1)
                bi = torch.where(alive, la, torch.full_like(la, float("-inf"))).argmax(1)
                is_fuse = torch.zeros_like(is_fuse)
            else:
                mn, idx = logD.reshape(B, K * K).min(1)
                bi, bj = idx // K, idx % K
            f_sel = act & is_fuse[ar, bi, bj]
            e_sel = act & ~is_fuse[ar, bi, bj]

            zi, zj = _gath(Z, bi), _gath(Z, bj)
            lai, laj = _gath(la, bi), _gath(la, bj)
            mu, la_f, r_st, delta_f, w, Gij, omr, r_dir = fuse_pair_math(zi, zj, lai, laj, beta)
            Lami, Lamj = _gath(Lam, bi), _gath(Lam, bj)
            pi, pj = _gath(prov, bi), _gath(prov, bj)
            vic = torch.where(lai < laj, bi, bj)        # lighter; ties -> bj
            pv = _gath(prov, vic)
            lav = _gath(la, vic)
            destroyed = destroyed | (pv & e_sel[:, None])
            st["dead_mass"] += (lav.exp() * e_sel.to(dtype)).detach()

            oh_i = F.one_hot(bi, K).to(dtype)
            mu_eff = torch.where(f_sel[:, None], mu, zi)
            la_eff = torch.where(f_sel, la_f, lai)
            Lam_eff = torch.where(f_sel, Lami + Lamj + delta_f, Lami)
            Z = Z * (1 - oh_i[..., None]) + mu_eff[:, None, :] * oh_i[..., None]
            la = la * (1 - oh_i) + la_eff[:, None] * oh_i
            Lam = Lam * (1 - oh_i) + Lam_eff[:, None] * oh_i
            prov = prov | (torch.where(f_sel[:, None], pi | pj, pi)[:, None, :]
                           & F.one_hot(bi, K).bool()[..., None])

            # exact incremental Gram row (O(C), OLD rows), only where merging
            row = (w[:, None] * G[ar, bi, :] + (1.0 - w)[:, None] * G[ar, bj, :]) \
                / r_dir.clamp_min(1e-12)[:, None]
            row = torch.where(f_sel[:, None], row, G[ar, bi, :])
            G = G.scatter(1, bi[:, None, None].expand(B, 1, K), row[:, None, :])
            G = G.scatter(2, bi[:, None, None].expand(B, K, 1), row[:, :, None])
            obi = F.one_hot(bi, K).bool()
            diag_mask = obi[:, :, None] & obi[:, None, :] & f_sel[:, None, None]
            G = torch.where(diag_mask, torch.ones_like(G), G)

            dslot = torch.where(f_sel, bj, vic)
            oh_d = F.one_hot(dslot, K).to(dtype) * act.to(dtype)[:, None]
            Z = Z * (1 - oh_d[..., None])
            la = la * (1 - oh_d) + NEG * oh_d
            Lam = Lam * (1 - oh_d)
            prov = prov & ~oh_d.bool()[..., None]
            alive = alive & ~oh_d.bool()
            G = G * (1 - oh_d)[:, :, None] * (1 - oh_d)[:, None, :]

            df_sel = d_fuse[ar, bi, bj]
            de_sel = d_evict[ar, bi, bj]
            st["n_fuse"] += f_sel.to(dtype)
            st["n_evict"] += e_sel.to(dtype)
            st["sum_delta"] += (torch.where(f_sel, df_sel, de_sel) * act.to(dtype)).detach()
            st["below_cross"] += f_sel.to(dtype)
            count = alive.long().sum(1)

    la = torch.where(alive, la - (p.Gamma * (t_q - t_prev))[:, None], NEGt)
    tgt_c = ep["tgt"].clamp_max(T - 1)
    pt = prov.gather(2, tgt_c[:, None, None].expand(B, K, 1)).squeeze(2) & alive
    has = pt.any(1)
    slot_t = pt.long().argmax(1)
    lab = torch.where(has & (ep["tgt"] < T), slot_t, torch.full_like(slot_t, K))
    return {"Z": Z, "la": la, "alive": alive, "Lam": Lam, "prov": prov, "destroyed": destroyed,
            "label": lab, "K": K, "S": la.exp().sum(-1), "Cb": Cb, "stats": st}


class HopMem:
    """Fixed-size streaming state machine with the v2.3.1 debit policy and the
    exact incremental Gram law (spec 4.6)."""
    def __init__(self, p, d, C=None, tau=None, tau_dead=None, eps_r=EPS_R,
                 dtype=torch.float64, device="cpu", policy="debit"):
        self.p, self.d, self.C, self.tau = p, d, C, tau
        self.tau_dead, self.eps_r, self.policy = tau_dead, eps_r, policy
        self.dtype, self.device = dtype, device
        self.atoms = []
        self.G = torch.zeros(0, 0, dtype=dtype, device=device)
        self.S, self.t = 0.0, 0.0
        self.sum_delta, self.n_fuse, self.n_evict, self.n_dead = 0.0, 0, 0, 0
        self.dead_mass, self.destroyed, self.below_cross = 0.0, set(), 0

    def _f(self, name):
        return float(getattr(self.p, name).detach())

    def _advance(self, t):
        dt = max(t - self.t, 0.0)
        if dt:
            g = self._f("Gamma") * dt
            for a in self.atoms:
                a["la"] -= g
            self.S *= math.exp(-g)
        self.t = max(t, self.t)

    def _tau_dead(self):
        if self.tau_dead is not None:
            return self.tau_dead
        return self.tau if self.tau is not None else 1.0

    # ---- Gram law (spec 4.6) ----
    def _gram_append(self, z):
        n = len(self.atoms)
        if n == 1:
            self.G = torch.ones(1, 1, dtype=self.dtype, device=self.device)
            return
        Zs = torch.stack([a["z"] for a in self.atoms[:-1]])
        row = Zs @ z
        new = torch.zeros(n, n, dtype=self.dtype, device=self.device)
        new[:n - 1, :n - 1] = self.G
        new[n - 1, :n - 1] = row
        new[:n - 1, n - 1] = row
        new[n - 1, n - 1] = 1.0
        self.G = new

    def _gram_merge(self, i, j, w_i, w_j, r):
        row = (w_i * self.G[i] + w_j * self.G[j]) / max(r, 1e-12)   # OLD rows
        self.G[i] = row
        self.G[:, i] = row
        self.G[i, i] = 1.0                                          # explicit guard

    def _gram_drop(self, j):
        n = self.G.shape[0]
        keep = [k for k in range(n) if k != j]
        if not keep:
            self.G = torch.zeros(0, 0, dtype=self.dtype, device=self.device)
            return
        idx = torch.tensor(keep, dtype=torch.long, device=self.device)
        self.G = self.G[idx][:, idx]

    def _rebuild_gram(self):
        """Diagnostic only (drift refresh)."""
        if self.atoms:
            Zs = torch.stack([a["z"] for a in self.atoms])
            self.G = Zs @ Zs.t()
        else:
            self.G = torch.zeros(0, 0, dtype=self.dtype, device=self.device)

    def _gram_check(self):
        if self.G.numel() and bool((self.G.abs() > 1.0 + 1e-6).any()):
            self._rebuild_gram()

    def write(self, z, s, t, prov=None):
        self._advance(t)
        z = F.normalize(z.to(dtype=self.dtype, device=self.device), dim=-1)
        om, s = self._f("omega"), max(float(s), 0.0)
        a = -math.expm1(-om * s)
        eps = self._f("eps_bar") * (1 - a)
        for at in self.atoms:
            at["la"] -= om * s
        self.atoms.append({"z": z, "la": math.log(max(a * (1 - eps), DOSE_FLOOR)),
                           "Lam": 0.0, "prov": frozenset([prov]) if prov is not None else frozenset()})
        self.S = (1 - a) * self.S + a * (1 - eps)
        self._gram_append(z)
        self._compress_on_overflow()

    def reinforce(self, u, s_ret, cf, beta_bar):
        u = F.normalize(u.to(dtype=self.dtype, device=self.device), dim=-1)
        mu = float(u @ self.W @ u)
        tau_r = 1 / self.d + (1 - 1 / self.d) * (-math.expm1(-self._f("omega") * max(s_ret, 0.0)))
        den = 1.0 - mu if mu < tau_r else 1.0
        b = cf * min(beta_bar, max(0.0, tau_r - mu) / den) if mu < tau_r else 0.0
        if b > 0:
            for at in self.atoms:
                at["la"] += math.log1p(-b)
            self.atoms.append({"z": u, "la": math.log(b), "Lam": 0.0, "prov": frozenset()})
            self.S = (1 - b) * self.S + b
            self._gram_append(u)
            self._compress_on_overflow()
        return b

    def _null_fuse(self, mask):
        for k in range(len(self.atoms)):
            if mask[k]:
                self.dead_mass += math.exp(self.atoms[k]["la"])
                self.destroyed |= set(self.atoms[k]["prov"])
        keep = [k for k in range(len(self.atoms)) if not mask[k]]
        self.atoms = [self.atoms[k] for k in keep]
        self.S = sum(math.exp(a["la"]) for a in self.atoms)
        if keep:
            idx = torch.tensor(keep, dtype=torch.long, device=self.device)
            self.G = self.G[idx][:, idx]
        else:
            self.G = torch.zeros(0, 0, dtype=self.dtype, device=self.device)

    def _compress_on_overflow(self):
        if self.C is None or len(self.atoms) <= self.C:
            return
        thr = self._f("kappa") + math.log(self._tau_dead())
        dead = [a["la"] <= thr for a in self.atoms]
        if any(dead):
            self.n_dead += sum(dead)
            self._null_fuse(dead)
        beta = self._f("beta")
        bt = torch.tensor(beta, dtype=self.dtype, device=self.device)
        while len(self.atoms) > self.C:
            n = len(self.atoms)
            la = torch.tensor([a["la"] for a in self.atoms], dtype=self.dtype, device=self.device)
            alive = torch.ones(1, n, dtype=torch.bool, device=self.device)
            logD, is_fuse, d_fuse, d_evict, r, fuse_ok, valid = debit_candidates(
                la[None], self.G[None], alive, bt, self.eps_r)
            if self.policy == "lightest":
                j = int(la.argmin())
                i = int(la.argmax())
                fuse = False
            else:
                if self.policy == "ward":
                    logD, _ = ward_log_cost(la[None], self.G[None], alive, self.eps_r)
                elif self.policy == "delta":
                    Sv = la.exp().sum().clamp_min(DOSE_FLOOR)
                    basev = torch.logaddexp(la[:, None], la[None, :]) - Sv.log()
                    lfv = basev + d_fuse[0].clamp_min(TINY).log()
                    lev = basev + d_evict[0].clamp_min(TINY).log()
                    vv, ff = valid[0], fuse_ok[0]
                    logD = torch.where(vv & ff, lfv,
                                       torch.where(vv, lev, torch.full_like(lfv, float("inf"))))[None]
                elif self.policy == "random":
                    logD = torch.where(valid, torch.rand_like(logD),
                                       torch.full_like(logD, float("inf")))
                mn, idx = logD.reshape(1, n * n).min(1)
                i, j = int(idx[0]) // n, int(idx[0]) % n
                fuse = bool(is_fuse[0, i, j]) if self.policy == "debit" else bool(fuse_ok[0, i, j])
            if fuse:
                ai, aj = self.atoms[i], self.atoms[j]
                lai = torch.tensor([ai["la"]], dtype=self.dtype, device=self.device)
                laj = torch.tensor([aj["la"]], dtype=self.dtype, device=self.device)
                mu, la_f, r_st, dl, w, Gij, omr, r_dir = fuse_pair_math(
                    ai["z"][None], aj["z"][None], lai, laj, bt)
                wi = float(w[0])
                self._gram_merge(i, j, wi, 1.0 - wi, float(r_dir[0]))
                ai["z"] = mu[0]
                ai["la"] = float(la_f[0])
                ai["Lam"] = ai["Lam"] + aj["Lam"] + float(dl[0])
                ai["prov"] = ai["prov"] | aj["prov"]
                self.sum_delta += float(dl[0])
                self.n_fuse += 1
                self.below_cross += 1
                self.atoms.pop(j)
                self._gram_drop(j)
            else:
                v = i if float(la[i]) < float(la[j]) else j
                dl = float(d_evict[0, i, j])
                self.destroyed |= set(self.atoms[v]["prov"])
                self.dead_mass += math.exp(float(la[v]))
                self.sum_delta += dl
                self.n_evict += 1
                self.atoms.pop(v)
                self._gram_drop(v)
            self.S = sum(math.exp(a["la"]) for a in self.atoms)
        self._gram_check()

    @property
    def Zs(self):
        return torch.stack([a["z"] for a in self.atoms]) if self.atoms else \
            torch.zeros(0, self.d, dtype=self.dtype, device=self.device)
    @property
    def log_alpha(self):
        return torch.tensor([a["la"] for a in self.atoms], dtype=self.dtype, device=self.device)
    @property
    def Lam_vec(self):
        return torch.tensor([a["Lam"] for a in self.atoms], dtype=self.dtype, device=self.device)
    @property
    def M(self):
        Z, a = self.Zs, self.log_alpha.exp()
        return torch.einsum("td,t,te->de", Z, a, Z) if len(a) else torch.zeros(
            self.d, self.d, dtype=self.dtype, device=self.device)
    @property
    def W(self):
        return (1 - self.S) / self.d * torch.eye(self.d, dtype=self.dtype, device=self.device) + self.M
    def purity(self):
        W = self.W
        return float((W * W).sum())
    def retained(self, prov_id):
        if prov_id in self.destroyed:
            return 0.0
        for a in self.atoms:
            if prov_id in a["prov"]:
                return math.exp(-a["Lam"])
        return 0.0
    def retrieve(self, q, t, N=20, tau=None):
        self._advance(t)
        q = F.normalize(q.to(dtype=self.dtype, device=self.device), dim=-1)[None]
        if not self.atoms:
            return q[0]
        Z, la = self.Zs[None], self.log_alpha[None]
        thr = self.tau if tau is None else tau
        if self.p.read == "gibbs" and thr is not None:
            if float(gate_gap(Z, la, q, self.p)[0]) <= math.log(thr):
                return q[0]
        return read(Z, la, q, self.p, N)[0]


# ----------------------------------------------------------------- readouts
def make_step(Z, log_alpha, q, p, auto_gamma=False):
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
    if auto_gamma:
        T, d = Z.shape[1], Z.shape[2]
        gamma = (3.0 * alpha.sum(-1) / max(T, 1) * math.sqrt(max(T - 1, 1) / d)).reshape(-1, 1)
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


def top_eigvec(Z, alpha):
    A = Z * alpha.clamp_min(0).sqrt().unsqueeze(-1)
    lam, w = torch.linalg.eigh(A @ A.transpose(1, 2))
    return F.normalize(torch.einsum("bt,btd->bd", w[..., -1], A), dim=-1), lam[..., -1]


def quad_stability(Z, alpha, k, gamma):
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


def gibbs_basin(log_alpha, k, beta, kappa, Lam=None):
    d = log_alpha.gather(1, k[:, None]).squeeze(1) - kappa
    if Lam is not None:
        d = d - Lam.gather(1, k[:, None]).squeeze(1)
    x = (1 - d / beta).clamp(-1, 1)
    return torch.where(d > 0, torch.acos(x), torch.zeros_like(x))


def gibbs_floor(Z, log_alpha, q, k, beta, kappa):
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


def nn_read(Z, q, alive=None):
    cc = torch.einsum("btd,bd->bt", Z, q)
    if alive is not None:
        cc = torch.where(alive, cc, torch.full_like(cc, float("-inf")))
    idx = cc.argmax(-1)
    return Z.gather(1, idx[:, None, None].expand(-1, 1, Z.shape[-1])).squeeze(1)


def gibbs_logits(Z, log_alpha, q, p, u=None, cue_free=False):
    u = q if u is None else u
    b = 0.0 if cue_free else p.beta
    return torch.cat([b * torch.einsum("btd,bd->bt", Z, u) + log_alpha,
                      b * (q * u).sum(-1, keepdim=True) + p.kappa], -1)


def gate_gap(Z, log_alpha, q, p):
    lg = gibbs_logits(Z, log_alpha, q, p)
    return lg[..., :Z.shape[1]].max(-1).values - lg[..., -1]


def gate_gap_mass(Z, log_alpha, q, p):
    lg = gibbs_logits(Z, log_alpha, q, p)
    return torch.logsumexp(lg[..., :Z.shape[1]], -1) - lg[..., -1]


GATE_TAUS = tuple(math.exp(-6.0 + 0.4 * i) for i in range(19))
GATE_ARGMAX_I = 15


def gate_loss(gap, loss_map, loss_q, tau):
    return torch.where(gap > math.log(tau), loss_map, loss_q)


def retrieval_loss(u, z_target):
    c = (u * z_target).sum(-1).clamp(-1, 1)
    sign = torch.where(c.detach() >= 0, torch.ones_like(c), -torch.ones_like(c))
    return 1 - sign * c


def decoder_norm(D, eps=1e-6, smooth=True):
    n = D.norm(dim=-1, keepdim=True)
    return D / (n * n + eps * eps).sqrt() if smooth else D / n.clamp_min(eps)


def reinforce_beta(mu, tau_r, cf, beta_bar):
    below = mu < tau_r
    den = torch.where(below, 1.0 - mu, torch.ones_like(mu))
    frac = (tau_r - mu).clamp_min(0.0) / den
    return torch.where(below, cf * torch.min(beta_bar, frac), torch.zeros_like(mu))


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


def dup_episodes(Zs, Qs, B, T, rho_dup, theta, lam=0.2, p_absent=0.2, device="cpu"):
    """Spec 10 duplicate-heavy generator: with prob rho_dup the write is a near-copy
    of the previous direction at angle ~theta."""
    P, d = len(Zs), Zs.shape[1]
    idx = torch.randint(P, (B, T), device=device)
    Z, Q = Zs[idx].clone(), Qs[idx].clone()
    sig = math.tan(theta)
    for i in range(1, T):
        copy = torch.rand(B, device=device) < rho_dup
        if not bool(copy.any()):
            continue
        zp = Z[:, i - 1]
        xi = torch.randn(B, d, device=device)
        xi = F.normalize(xi - (xi * zp).sum(-1, keepdim=True) * zp, dim=-1)
        zn = F.normalize(zp + sig * xi, dim=-1)
        Z[:, i] = torch.where(copy[:, None], zn, Z[:, i])
        phi = 0.4 + 0.9 * torch.rand(B, 1, device=device)
        e = torch.randn(B, d, device=device)
        e = F.normalize(e - (e * zn).sum(-1, keepdim=True) * zn, dim=-1)
        qn = phi.cos() * zn + phi.sin() * e
        Q[:, i] = torch.where(copy[:, None], qn, Q[:, i])
    s = torch.empty(B, T, device=device).exponential_()
    t = torch.empty(B, T, device=device).exponential_().cumsum(-1)
    t_q = t[:, -1] + torch.empty(B, device=device).exponential_()
    logits = torch.log(s) - lam * (t_q[:, None] - t)
    tgt = torch.multinomial(logits.softmax(-1), 1).squeeze(1)
    ar = torch.arange(B, device=device)
    q, z_t = Q[ar, tgt], Z[ar, tgt]
    absent = torch.rand(B, device=device) < p_absent
    j = torch.randint(P, (B,), device=device)
    qo = F.normalize(Qs[j], dim=-1)
    q = torch.where(absent[:, None], qo, q)
    z_t = torch.where(absent[:, None], qo, z_t)
    tgt = torch.where(absent, torch.full_like(tgt, T), tgt)
    return {"Z": Z, "s": s, "t": t, "t_q": t_q, "tgt": tgt, "z_t": z_t, "q": q, "absent": absent}


def sample_C(B, c_min, T, device="cpu"):
    return torch.randint(c_min, T + 1, (B,), device=device)


# ----------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(p, ep, N=20, auto_gamma=False, tau=None, C=None, tau_dead=None, policy="debit"):
    T = ep["Z"].shape[1]
    Ceff = T if C is None else C
    td = 1.0 if tau_dead is None else tau_dead
    sim = simulate_fusion(p, ep, Ceff, tau_dead=td, policy=policy)
    Z, la, alive, lab, K = sim["Z"], sim["la"], sim["alive"], sim["label"], sim["K"]
    q, z, tgt = ep["q"], ep["z_t"], ep["tgt"]
    fused = int(sim["Cb"].min()) < T
    u, info = read(Z, la, q, p, N, diagnostics=True, auto_gamma=auto_gamma)
    uL = read(Z, la, q, p, 10 * N, auto_gamma=auto_gamma)
    v1, _ = top_eigvec(Z, la.exp())
    cos = lambda x, y: (x * y).sum(-1).abs().clamp(max=1.0)
    per, per_q = 1 - cos(u, z), 1 - cos(q, z)
    pres, absm = tgt < T, tgt >= T
    out = {"loss": per.mean().item(), "loss10N": (1 - cos(uL, z)).mean().item(),
           "cue": per_q.mean().item(), "centroid": (1 - cos(v1, z)).mean().item(),
           "nn": (1 - cos(nn_read(Z, q, alive), z)).mean().item(),
           "capture": (cos(u, v1) > cos(u, z)).float().mean().item(),
           "E_up": info["E_up"], "guard": info["guard"], "S": sim["S"].mean().item()}
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
        null = am == K
        surv = pres & (lab < K)
        if bool(surv.any()):
            out["commit"] = (am[surv] == lab[surv]).float().mean().item()
            out["basin"] = gibbs_basin(la[surv], lab[surv], p.beta, p.kappa, sim["Lam"][surv]).mean().item()
            out["floor"] = gibbs_floor(Z[surv], la[surv], q[surv], lab[surv], p.beta, p.kappa).mean().item()
            out["phi"] = torch.acos((q[surv] * z[surv]).sum(-1).clamp(-1, 1)).mean().item()
        if bool(absm.any()):
            out["abstain"] = null[absm].float().mean().item()
            out["hyst"] = (null & (cos(u, q) < 0.9))[absm].float().mean().item()
        gap = gate_gap(Z, la, q, p)
        sw = [float(gate_loss(gap, per, per_q, t).mean()) for t in GATE_TAUS]
        if tau is None:
            bi = (GATE_ARGMAX_I if (max(sw) - min(sw)) < 1e-9
                  else min(range(len(sw)), key=lambda i: sw[i]))
            tsel = GATE_TAUS[bi]
        else:
            tsel = tau
        out["gate_argmax"] = sw[GATE_ARGMAX_I]
        out["gate_best"] = float(gate_loss(gap, per, per_q, tsel).mean())
        out["gate_tau"], out["gate_src"] = tsel, ("eval" if tau is None else "train")
        commit = gap > math.log(tsel)
        pg = 1 - cos(torch.where(commit[:, None], u, q), z)
        if bool(pres.any()):
            out["gate_pres"] = pg[pres].mean().item()
        if bool(absm.any()):
            out["gate_abs"] = pg[absm].mean().item()
        if fused:
            stt = sim["stats"]
            out["fuse_n"] = stt["n_fuse"].mean().item()
            out["evict_n"] = stt["n_evict"].mean().item()
            out["dead_n"] = stt["n_dead"].mean().item()
            out["ledger"] = stt["sum_delta"].mean().item()
            out["below_cross"] = (stt["below_cross"].sum() / stt["n_fuse"].sum().clamp_min(1)).item()
            Lb = torch.where(lab < K, sim["Lam"].gather(1, lab.clamp_max(K - 1)[:, None]).squeeze(1),
                             torch.zeros_like(lab, dtype=la.dtype))
            ret = torch.where((lab < K) & pres, (-Lb).exp(), torch.zeros_like(Lb))
            out["retained"] = ret[pres].mean().item() if bool(pres.any()) else float("nan")
            out["destroyed_frac"] = ((lab == K) & pres).float().sum().item() / max(int(pres.sum()), 1)
            mu_b = Z.gather(1, lab.clamp_max(K - 1)[:, None, None].expand(-1, 1, Z.shape[-1])).squeeze(1)
            ang = torch.acos((mu_b * z).sum(-1).clamp(-1, 1))
            out["distort"] = ang[surv].mean().item() if bool(surv.any()) else float("nan")
            out["commit_rate"] = commit.float().mean().item()
            mg = gate_gap_mass(Z, la, q, p)
            msw = [float(gate_loss(mg, per, per_q, t).mean()) for t in GATE_TAUS]
            bmi = (GATE_ARGMAX_I if (max(msw) - min(msw)) < 1e-9
                   else min(range(len(msw)), key=lambda i: msw[i]))
            out["mass_gate_best"], out["mass_tau"] = msw[bmi], GATE_TAUS[bmi]
            sel = pres & commit & (lab < K)
            if bool(sel.any()):
                pa = lg[..., :K].softmax(-1)
                conf, amk = pa.max(-1)
                corr = (amk == lab).float()
                b = (conf * 10).long().clamp(0, 9)
                ece = 0.0
                for k in range(10):
                    m = b == k
                    if bool(m.any()):
                        ece += float(m.float().mean()) * abs(float(conf[m].mean()) - float(corr[m].mean()))
                out["ece"] = ece
    return out


@torch.no_grad()
def calibrate_tau(p, ep, N=20, auto_gamma=False, C=None, tau_dead=1.0, policy="debit"):
    if p.read != "gibbs":
        return None
    T = ep["Z"].shape[1]
    sim = simulate_fusion(p, ep, T if C is None else C, tau_dead=tau_dead, policy=policy)
    Z, la, q, z = sim["Z"], sim["la"], ep["q"], ep["z_t"]
    u = read(Z, la, q, p, N, auto_gamma=auto_gamma)
    cos = lambda x, y: (x * y).sum(-1).abs().clamp(max=1.0)
    per, per_q = 1 - cos(u, z), 1 - cos(q, z)
    gap = gate_gap(Z, la, q, p)
    sw = [float(gate_loss(gap, per, per_q, t).mean()) for t in GATE_TAUS]
    if (max(sw) - min(sw)) < 1e-9:
        return GATE_TAUS[GATE_ARGMAX_I]
    return GATE_TAUS[min(range(len(sw)), key=lambda i: sw[i])]


@torch.no_grad()
def calibrate_tau_fused(p, ep, N=20, C=None, passes=2, policy="debit"):
    tau = 1.0
    for _ in range(max(1, passes)):
        tau = calibrate_tau(p, ep, N, C=C, tau_dead=tau, policy=policy)
    return tau


def report(tag, p, ep, N, auto_gamma=False, tau=None, C=None, tau_dead=None, policy="debit"):
    r = evaluate(p, ep, N, auto_gamma, tau=tau, C=C, tau_dead=tau_dead, policy=policy)
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
        if "fuse_n" in r:
            print(f"      compress: {g('fuse_n'):.1f} merge / {g('evict_n'):.1f} evict / "
                  f"{g('dead_n'):.1f} gate-dead | ledger {g('ledger'):.3f} nats | "
                  f"retained {g('retained'):.3f} destroyed {g('destroyed_frac'):.3f} "
                  f"distort {g('distort'):.3f} | below-cross {g('below_cross'):.0%} | "
                  f"ECE {g('ece'):.3f} | mass-gate {g('mass_gate_best'):.4f}")
    print(f"      {th}")
    return r


# ----------------------------------------------------------------- training
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
    fuse_on = (a.fuse == "on")
    print(f"{P} items, d={Zs.shape[1]}, train {cut}/eval {P-cut}, device {dev}, read {a.read}, "
          f"schedule {a.schedule}, C={a.C} (train C~U{{{a.c_min}..{a.T}}}), "
          f"policy {a.fuse_policy}, fusion {'on' if fuse_on else 'off'}")
    p = Params(read=a.read, diffusion=not a.no_diffusion, pure_writes=a.pure_writes).to(dev)
    cal_ep = tr.sample(a.cal_batch) if (p.read == "gibbs" and a.gate_tau is None) else None
    saved_tau = [float(a.gate_tau) if a.gate_tau is not None else None]
    def get_tau():
        return saved_tau[0]
    mode = a.loss if a.loss in ("cos", "ce", "both") else "ce"
    if p.read != "gibbs":
        mode = "cos"
        if a.schedule == "probe":
            a.schedule = "staged"
    if a.schedule == "probe":
        stages = [("P", {"a_omega", "a_Gamma", "a_kappa"}, "prior", a.steps_p),
                  ("A", p.read_names(), mode, a.steps_a),
                  ("B", (p.shared() | p.read_names()) - p.law_names(), mode, a.steps_b)]
    else:
        stages = [("A", p.read_names(), mode, a.steps_a), ("B", None, mode, a.steps_b)]
    ev_ep = ev.sample(a.eval_batch)
    report("init  ", p, ev_ep, a.N, a.auto_gamma, tau=get_tau(), C=a.C, policy=a.fuse_policy)
    for stage, names, smode, steps in stages:
        stage_fuse = fuse_on and (stage != "P")
        if stage == "P":
            print("  stage P: MARGINAL law (cue-free), COMPRESSION OFF.")
        if stage == "B" and a.schedule == "probe":
            print("  stage B: readout only; omega/Gamma frozen.")
        for n, prm in p.named_parameters():
            prm.requires_grad_(names is None or n in names)
        opt = torch.optim.Adam([x for x in p.parameters() if x.requires_grad], lr=a.lr)
        for it in range(steps):
            ep = tr.sample(a.batch)
            if stage_fuse:
                Cb = sample_C(a.batch, a.c_min, a.T, dev)
                sim = simulate_fusion(p, ep, Cb, tau_dead=1.0, policy=a.fuse_policy)
                Zs_, la_, lab_ = sim["Z"], sim["la"], sim["label"]
            else:
                Zs_, la_, lab_ = ep["Z"], log_doses(p, ep["s"], ep["t"], ep["t_q"]), ep["tgt"]
            l_cos = torch.zeros((), device=dev)
            if mode in ("cos", "both") or p.read != "gibbs":
                u = read(Zs_, la_, ep["q"], p, a.N, auto_gamma=a.auto_gamma)
                l_cos = retrieval_loss(u, ep["z_t"]).mean()
            l_ce = torch.zeros((), device=dev)
            if p.read == "gibbs" and smode in ("ce", "both", "prior"):
                l_ce = F.cross_entropy(gibbs_logits(Zs_, la_, ep["q"], p, cue_free=(smode == "prior")), lab_)
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
                print(f"  stage {stage} step {it:4d} ce {l_ce.item():.4f}  {th}")
        if stage == "A" and p.read == "gibbs" and saved_tau[0] is None and cal_ep is not None:
            saved_tau[0] = calibrate_tau_fused(p, cal_ep, a.N, C=a.C, passes=2, policy=a.fuse_policy)
            print(f"  tau calibrated once on fixed compressed batch (C={a.C}): tau={saved_tau[0]:.4g}")
        report(f"stage {stage}", p, ev_ep, a.N, a.auto_gamma, tau=get_tau(), C=a.C, policy=a.fuse_policy)
    tau_final = get_tau()
    if p.read == "gibbs":
        rd = evaluate(p, ev_ep, a.N, a.auto_gamma, tau=tau_final, C=a.C,
                      tau_dead=(tau_final if tau_final is not None else 1.0), policy=a.fuse_policy)
        print(f"[deploy] gate {rd['gate_best']:.4f} (tau {tau_final}) | "
              f"pres/abs {rd.get('gate_pres', float('nan')):.4f}/{rd.get('gate_abs', float('nan')):.4f}")
    ck = {"state": p.state_dict(), "gate_tau": tau_final, "read": a.read, "version": __version__,
          "config": {"diffusion": not a.no_diffusion, "pure_writes": a.pure_writes,
                     "d": int(Zs.shape[1]), "N": a.N, "schedule": a.schedule,
                     "query_decay": a.query_decay, "p_absent": a.p_absent, "gate_taus": GATE_TAUS,
                     "C": int(a.C), "c_min": int(a.c_min), "fuse": a.fuse,
                     "fuse_policy": a.fuse_policy, "version": __version__},
          "theta": p.theta()}
    torch.save(ck, a.out)
    print(f"saved {a.out} (gate_tau={tau_final}, C={a.C}, policy={a.fuse_policy})")


class HopMemAttention(nn.Module):
    """Spec 23: Gibbs posterior at u=q == one cross-attention head, tied KV, null bias
    kappa, per-token log-dose bias.  SPECIFIED, not yet measured end-to-end."""
    def __init__(self, d, width=None, beta=16.0, kappa=math.log(1e-4)):
        super().__init__()
        self.d, self.width = d, width or d
        self.beta, self.kappa = beta, kappa
        self.WK = nn.Linear(d, self.width, bias=False) if width else None
        self.WV = nn.Linear(d, self.width, bias=False) if width else None
        self.Wq = nn.Linear(d, self.width, bias=False) if width else None

    def forward(self, q, mu, ell, gate_tau=None):
        if self.WK is None:
            logits = self.beta * torch.einsum("btd,bd->bt", mu, q) + ell
            null = self.beta * (q * q).sum(-1, keepdim=True) + self.kappa
            lg = torch.cat([logits, null], -1)
            w = lg.softmax(-1)
            u = torch.einsum("bt,btd->bd", w[..., :-1], mu) + w[..., -1:] * q
        else:
            k, v, qq = self.WK(mu), self.WV(mu), self.Wq(q)
            logits = (k @ qq.unsqueeze(-1)).squeeze(-1) + ell
            null = (qq * qq).sum(-1, keepdim=True) + self.kappa
            lg = torch.cat([logits, null], -1)
            w = lg.softmax(-1)
            u = torch.einsum("bt,btd->bd", w[..., :-1], v) + w[..., -1:] * qq
        if gate_tau is not None:
            gap = lg[..., :-1].max(-1).values - lg[..., -1]
            g = torch.sigmoid(gap - math.log(gate_tau)).unsqueeze(-1)
            u = g * u + (1 - g) * q
        return u, w


def lm_interface_check():
    torch.manual_seed(0)
    d, C = 16, 5
    p = Params(beta=16.0, kappa=-6.0, dtype=torch.float64)
    Z = F.normalize(torch.randn(2, C, d, dtype=torch.float64), dim=-1)
    la = torch.randn(2, C, dtype=torch.float64) - 3
    q = F.normalize(torch.randn(2, d, dtype=torch.float64), dim=-1)
    att = HopMemAttention(d, beta=16.0, kappa=-6.0).double()
    u_att, w = att(q, Z, la)
    v, _ = make_step(Z, la, q, p)(q)
    err = float((u_att - v).abs().max())
    print(f"LM interface: attention == Gibbs gradient at u=q, max err {err:.2e}")
    assert err < 1e-12
    return err


# ----------------------------------------------------------------- selftest (44)
def selftest():
    torch.manual_seed(0)
    dt, res = torch.float64, []
    def check(name, cond, extra=""):
        print(("PASS " if cond else "FAIL ") + name + ("  " + extra if extra else ""))
        res.append(bool(cond))
    d, T, B = 24, 7, 3
    p = Params(omega=0.7, Gamma=0.3, eps_bar=0.3, beta=6.0, kappa=-5.0, eta=2.0, chi=0.2, dtype=dt)
    s = torch.rand(B, T, dtype=dt) * 2
    t = torch.cumsum(torch.rand(B, T, dtype=dt) + 0.1, -1)
    t_q = t[:, -1] + 0.5
    Z = F.normalize(torch.randn(B, T, d, dtype=dt), dim=-1)
    la = log_doses(p, s, t, t_q)
    q = F.normalize(torch.randn(B, d, dtype=dt), dim=-1)
    om, eb = float(p.omega.detach()), float(p.eps_bar.detach())
    eye = torch.eye(d, dtype=dt)
    ep0 = {"Z": Z, "s": s, "t": t, "t_q": t_q, "tgt": torch.zeros(B, dtype=torch.long),
           "z_t": Z[:, 0], "q": q, "absent": torch.zeros(B, dtype=torch.bool)}
    with torch.no_grad():
        mem = HopMem(p, d, C=4, dtype=dt)
        inv, dec = 0.0, 0.0
        for i in range(40):
            mem.write(F.normalize(torch.randn(d, dtype=dt), dim=-1), float(torch.rand(1)) * 2, float(i) * 0.3, prov=i)
            W = mem.W
            ev = torch.linalg.eigvalsh(W)
            inv = max(inv, abs(float(W.trace()) - 1), max(0.0, (1 - mem.S) / d - float(ev.min())),
                      max(0.0, mem.S - 1 + 1e-12))
            dec = max(dec, float((W - ((1 - mem.S) / d * eye + mem.M)).abs().max()))
        check("01 invariants under write/diffuse/compress", inv < 1e-10 and dec < 1e-10,
              f"inv={inv:.1e} atoms={len(mem.atoms)}<=4")
        m2 = HopMem(p, d, C=None, dtype=dt)
        for i in range(T):
            m2.write(Z[0, i], float(s[0, i]), float(t[0, i]))
        m2._advance(float(t_q[0]))
        e_la = float((m2.log_alpha - la[0]).abs().max())
        e_M = float((m2.M - torch.einsum("td,t,te->de", Z[0], la[0].exp(), Z[0])).abs().max())
        check("02 closed form == sequential", e_la < 1e-12 and e_M < 1e-12, f"dla={e_la:.1e}")
        m3 = HopMem(p, d, C=None, dtype=dt); m3.write(Z[0, 0], 1.0, 0.0)
        W0 = m3.W.clone(); m3._advance(0.7); m3._advance(1.2); W12 = m3.W.clone()
        m3b = HopMem(p, d, C=None, dtype=dt); m3b.write(Z[0, 0], 1.0, 0.0); m3b._advance(1.2)
        g = float(p.Gamma.detach())
        Wexp = math.exp(-g * 1.2) * W0 + (1 - math.exp(-g * 1.2)) / d * eye
        check("03 diffusion semigroup (cumulative absolute time)",
              float((W12 - m3b.W).abs().max()) < 1e-12 and float((W12 - Wexp).abs().max()) < 1e-12)
        M0 = m3b.M.clone(); m3b._advance(2.1); M1 = m3b.M
        l0, v0 = torch.linalg.eigh(M0); l1, v1 = torch.linalg.eigh(M1)
        f = math.exp(-g * 0.9)
        sgn = float((v0[:, -1] * v1[:, -1]).sum().sign())
        check("04 spectral invariance (top eigenvector)",
              float((l1 - f * l0).abs().max()) < 1e-12
              and float((v1[:, -1] * sgn - v0[:, -1]).abs().max()) < 1e-9)
        m5 = HopMem(Params(omega=0.5, Gamma=0.2, pure_writes=True, dtype=dt), d, C=None, dtype=dt)
        for kk in range(1, 400):
            m5.write(Z[0, 0], 1.0, float(kk))
        post = m5.S; m5._advance(400.0)
        check("05 steady state", abs(post - 0.78160) < 1e-5 and abs(m5.S - 0.63992) < 1e-5)
        m6 = HopMem(p, d, C=None, dtype=dt); m6.write(Z[0, 0], float(s[0, 0]), float(t[0, 0]))
        W0 = (1.0 / d) * eye
        a = -math.expm1(-om * float(s[0, 0])); eps = eb * (1 - a)
        Wexp = (1 - a) * W0 + a * ((1 - eps) * torch.outer(Z[0, 0], Z[0, 0]) + eps * eye / d)
        check("06 impure-write decomposition", float((m6.W - Wexp).abs().max()) < 1e-12)
        m7 = HopMem(p, d, C=4, dtype=dt); u7 = m7.retrieve(q[0], 0.0)
        u7b = read(torch.zeros(1, 0, d, dtype=dt), torch.zeros(1, 0, dtype=dt), q[:1], p, 20)[0]
        check("07 empty memory returns q", float((u7 - q[0]).abs().max()) < 1e-15
              and float((u7b - q[0]).abs().max()) < 1e-15)
        perm = torch.randperm(T); u8 = read(Z, la, q, p, 20); u8p = read(Z[:, perm], la[:, perm], q, p, 20)
        check("08 permutation invariance", float((u8 - u8p).abs().max()) < 1e-12)
        c = 3.0
        lg_a = gibbs_logits(Z, la, q, p).softmax(-1)
        ps = Params(beta=6.0, kappa=-5.0 + c, dtype=dt)
        lg_b = gibbs_logits(Z, la + c, q, ps).softmax(-1)
        u9 = read(Z, la + c, q, ps, 20)
        check("09 uniform log-shift invariance",
              float((lg_a - lg_b).abs().max()) < 1e-12 and float((u8 - u9).abs().max()) < 1e-12)
        p10 = Params(omega=0.7, dtype=dt)
    s0 = torch.zeros(3, requires_grad=True)
    l0 = log_injected(p10, s0); l0.sum().backward()
    l0v = l0.detach(); gmax = float(s0.grad.abs().max())
    floor_ref = float(torch.log(torch.tensor(DOSE_FLOOR, dtype=l0v.dtype)))
    check("10 s=0 finite log-dose, zero clamped gradient",
          bool(torch.isfinite(l0v).all()) and abs(float(l0v[0]) - floor_ref) < 1e-6 and gmax < 1e-9,
          f"ell0={float(l0v[0]):.6f} grad={gmax:.1e}")
    with torch.no_grad():
        worst, guards = 0.0, 0
        for b_ in (0.05, 1.0, 6.0, 50.0, 1000.0):
            pb = Params(beta=b_, kappa=-5.0, dtype=dt)
            _, inf = read(Z, la, q, pb, 30, diagnostics=True)
            worst = max(worst, inf["E_up"]); guards += inf["guard"]
        check("11 Gibbs descent, guard 0", worst == 0.0 and guards == 0)
        Z12 = F.normalize(torch.randn(1, 8, 32, dtype=dt), dim=-1)
        la12 = torch.full((1, 8), math.log(0.05), dtype=dt)
        p12 = Params(beta=6.0, kappa=math.log(0.05) - 2.0, dtype=dt)
        uN = read(Z12, la12, Z12[:, 0], p12, 20); u10N = read(Z12, la12, Z12[:, 0], p12, 200)
        check("12 N vs 10N agrees", float((uN - u10N).abs().max()) < 1e-6)
        Zh = torch.tensor([[[math.cos(0.7), math.sin(0.7)]]], dtype=dt)
        qh = torch.tensor([[1.0, 0.0]], dtype=dt); lah = torch.full((1, 1), -3.0, dtype=dt)
        ph = Params(beta=16.0, kappa=-6.5, dtype=dt)
        lgh = gibbs_logits(Zh, lah, qh, ph); uh = read(Zh, lah, qh, ph, 50)
        check("13 hysteresis witness", int(lgh.argmax(-1)[0]) == 1 and float((uh * Zh[:, 0]).sum()) > 0.99,
              f"atom {float(lgh[0,0]):.3f} null {float(lgh[0,1]):.3f}")
        gap = torch.randn(64, dtype=dt); lmap, lq = torch.rand(64, dtype=dt), torch.rand(64, dtype=dt)
        check("14 gate limits",
              abs(float(gate_loss(gap, lmap, lq, 1e-12).mean()) - float(lmap.mean())) < 1e-12
              and abs(float(gate_loss(gap, lmap, lq, 1e12).mean()) - float(lq.mean())) < 1e-12)
        pg = Params(beta=16.0, kappa=-8.0, omega=0.7, Gamma=0.3, eps_bar=0.3, dtype=dt)
        mg = HopMem(pg, d, C=3, tau=0.5, tau_dead=0.5, dtype=dt)
        for i in range(T):
            mg.write(Z[0, i], float(s[0, i]), float(t[0, i]), prov=i)
        qg = F.normalize(torch.randn(d, dtype=dt), dim=0); us = mg.retrieve(qg, float(t_q[0]))
        epg = {"Z": Z[:1], "s": s[:1], "t": t[:1], "t_q": t_q[:1], "tgt": torch.zeros(1, dtype=torch.long),
               "z_t": Z[:1, 0], "q": qg[None], "absent": torch.zeros(1, dtype=torch.bool)}
        simg = simulate_fusion(pg, epg, 3, tau_dead=0.5)
        ub = read(simg["Z"], simg["la"], qg[None], pg, 20)[0]
        gb = gate_gap(simg["Z"], simg["la"], qg[None], pg)
        ubg = torch.where((gb > math.log(0.5))[:, None], ub, qg[None])[0]
        check("15 streaming == batch gated (compression on)", float((us - ubg).abs().max()) < 1e-9)
        t16a = calibrate_tau(pg, epg, 20, C=3, tau_dead=0.5)
        t16b = calibrate_tau(pg, epg, 20, C=3, tau_dead=0.5)
        empty_ep = dict(epg); empty_ep["t_q"] = epg["t_q"] + 1e4
        tflat = calibrate_tau(pg, empty_ep, 20, C=3, tau_dead=0.5)
        check("16 tau determinism; flat -> argmax", t16a == t16b and abs(tflat - 1.0) < 1e-12)
        d2, T2, al = 384, 20, 0.05
        Z2 = F.normalize(torch.randn(1, T2, d2, dtype=dt), dim=-1)
        la2 = torch.full((1, T2), math.log(al), dtype=dt); k2 = torch.zeros(1, dtype=torch.long)
        pb17 = Params(beta=16.0, kappa=math.log(1e-4), dtype=dt)
        ins = outs = True
        for phi, want in ((0.3, True), (0.6, True), (1.3, False), (1.5, False)):
            e = F.normalize(torch.randn(1, d2, dtype=dt), dim=-1)
            e = F.normalize(e - (e * Z2[:, 0]).sum(-1, keepdim=True) * Z2[:, 0], dim=-1)
            qq = math.cos(phi) * Z2[:, 0] + math.sin(phi) * e
            gg = float(gate_gap(Z2, la2, qq, pb17)[0])
            if want: ins &= gg > 0
            else: outs &= gg <= 0
        pg17 = Params(beta=6.0, kappa=math.log(al) - 2.0, dtype=dt)
        ug = read(Z2, la2, Z2[:, 0], pg17, 300); ang_g = float((ug - Z2[:, 0]).norm())
        pred = float(gibbs_floor(Z2, la2, Z2[:, 0], k2, pg17.beta, pg17.kappa))
        Zr = F.normalize(torch.stack([Z2[0, 0], Z2[0, 0] + 0.01 * F.normalize(torch.randn(d2, dtype=dt), dim=-1)]), dim=-1)[None]
        lar = torch.tensor([[math.log(al), math.log(al) - 5.0]], dtype=dt)
        qr = F.normalize(Zr[0, 0] + 0.05 * F.normalize(torch.randn(d2, dtype=dt), dim=-1), dim=-1)[None]
        lgr = gibbs_logits(Zr, lar, qr, pg17)
        check("17 basin / floor / resolution",
              ins and outs and abs(ang_g - pred) < 0.25 * pred + 1e-6 and int(lgr.argmax(-1)[0]) == 0)
    p18 = Params(read="gibbs", beta=60.0)
    Zf = F.normalize(torch.randn(8, 12, 32), dim=-1)
    sf = torch.empty(8, 12).exponential_(); tf = torch.empty(8, 12).exponential_().cumsum(-1)
    tq = tf[:, -1] + 0.3; lf = log_doses(p18, sf, tf, tq)
    lab18 = torch.zeros(8, dtype=torch.long); qf = Zf[:, 0]
    lg = gibbs_logits(Zf, lf, qf, p18)
    g_cue = torch.autograd.grad(F.cross_entropy(lg, lab18), p18.a_Gamma, retain_graph=True)[0]
    lg0 = gibbs_logits(Zf, lf, qf, p18, cue_free=True)
    g_pri = torch.autograd.grad(F.cross_entropy(lg0, lab18), p18.a_Gamma)[0]
    check("18 prior probe restores Gamma gradient", float(g_pri.abs()) > 10.0 * float(g_cue.abs()) + 1e-9)
    with torch.no_grad():
        alpha = la.exp(); gam = 0.01; k = torch.zeros(B, dtype=torch.long)
        margin, fs, fh = quad_stability(Z, alpha, k, gam)
        mm, ff, hh = [], [], []
        for b in range(B):
            zk = Z[b, 0]; M = torch.einsum("td,t,te->de", Z[b], alpha[b], Z[b])
            Moth = M - alpha[b, 0] * torch.outer(zk, zk); P = eye - torch.outer(zk, zk)
            mu = float(zk @ M @ zk)
            Hd = (mu + gam) * P - P @ Moth @ P + (mu + gam) * torch.outer(zk, zk)
            ev = torch.linalg.eigvalsh(Hd); mm.append(float(ev.min()))
            x = P @ Moth @ zk; ff.append(float(x.norm()) / (float(alpha[b, 0]) + gam))
            hh.append(float(torch.linalg.solve(Hd, x).norm()) if float(ev.min()) > 0 else float("nan"))
        check("19 factored == dense stability",
              float((margin - torch.tensor(mm, dtype=dt)).abs().max()) < 1e-9
              and float((fs - torch.tensor(ff, dtype=dt)).abs().max()) < 1e-9
              and bool(torch.allclose(fh, torch.tensor(hh, dtype=dt), atol=1e-8, equal_nan=True)))
        A = torch.randn(d, 5, dtype=dt); Wr = A @ A.t() / (A @ A.t()).trace()
        worst = 0.0
        for e_ in (0.1, 10.0, 1e4):
            ge = 0.5 / e_
            st = lambda u, e_=e_, ge=ge: (u + e_ * (u @ Wr + ge * q),
                                          -0.5 * (u * (u @ Wr)).sum(-1) - ge * (u * q).sum(-1))
            worst = max(worst, retrieve(st, q, 30, diagnostics=True)[1]["E_up"])
        S_, M_ = m2.S, m2.M; eta, chi = 2.0, 0.2
        gam_, etap = chi / eta, eta / (1 + eta * (1 - S_) / d)
        stW = lambda u: (u + eta * (u @ m2.W + gam_ * q), -0.5 * (u * (u @ m2.W)).sum(-1) - gam_ * (u * q).sum(-1))
        stM = lambda u: (u + etap * (u @ M_ + gam_ * q), -0.5 * (u * (u @ M_)).sum(-1) - gam_ * (u * q).sum(-1))
        check("20 legacy descent; W-form == M-form", worst == 0.0
              and float((retrieve(stW, q, 20) - retrieve(stM, q, 20)).abs().max()) < 1e-11)
        mg_a, _, _ = quad_stability(Z2, la2.exp(), k2, 3 * al * math.sqrt((T2 - 1) / d2))
        gc = 2 * al * math.sqrt((T2 - 1) / d2); closure_max = 1 / (1 + gc / al)
        mg_e, _, _ = quad_stability(Z2, la2.exp(), k2, gc * 1.05)
        phi21 = 0.6
        e21 = F.normalize(torch.randn(1, d2, dtype=dt), dim=-1)
        e21 = F.normalize(e21 - (e21 * Z2[:, 0]).sum(-1, keepdim=True) * Z2[:, 0], dim=-1)
        q21 = math.cos(phi21) * Z2[:, 0] + math.sin(phi21) * e21
        pt = Params(read="power", eta=1.0, chi=gc * 1.05, dtype=dt)
        ut = read(Z2, la2, q21, pt, 4000)
        theta21 = float(torch.acos((ut * Z2[:, 0]).sum(-1).clamp(-1, 1))[0])
        cf21 = (phi21 - theta21) / phi21
        check("21 capacity-closure witness", bool((mg_a > 0).all()) and bool((mg_e > 0).all())
              and cf21 <= closure_max + 0.06, f"closure_frac {cf21:.3f}")
        Zg = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]], dtype=dt)
        lg22 = torch.zeros(1, 2, dtype=dt); qg22 = torch.tensor([[0.0, 1.0]], dtype=dt)
        pg22 = Params(beta=6.0, kappa=NEG, dtype=dt)
        ug22, ig22 = read(Zg, lg22, qg22, pg22, 5, diagnostics=True)
        with torch.enable_grad():
            mu_t = torch.tensor([0.9, 0.2], dtype=dt, requires_grad=True)
            rb = reinforce_beta(mu_t, torch.tensor([0.5, 0.5], dtype=dt),
                                torch.ones(2, dtype=dt), torch.tensor([0.999, 0.999], dtype=dt))
            rb.sum().backward()
        rbv = rb.detach()
        lam22 = torch.zeros(1, 2, dtype=dt); Gm = torch.tensor([[[1.0, -1.0], [-1.0, 1.0]]], dtype=dt)
        cm, _ = ward_log_cost(lam22, Gm, torch.ones(1, 2, dtype=torch.bool), EPS_R)
        check("22 masked denominators", bool(torch.isfinite(ug22).all())
              and float((ug22 - qg22).abs().max()) < 1e-12 and ig22["guard"] > 0
              and float(rbv[0]) == 0.0 and abs(float(rbv[1]) - 0.375) < 1e-12
              and bool(torch.isfinite(mu_t.grad).all()) and bool(torch.isinf(cm[0, 0, 1]).all()))
        err = 0.0
        for y in (1e-6, 1e-3, 0.1, 1.0, 5.0, 19.9, 20.0, 50.0, 100.0, 700.0):
            x = inv_softplus(y)
            yb = math.log1p(math.exp(x)) if x < 20 else x + math.log1p(math.exp(-x))
            err = max(err, abs(yb - y) / max(y, 1e-12))
        check("23 stable inverse softplus", err < 1e-9)
        x = torch.tensor([1e-9, 1e-8, 1e-7, 1e-6], dtype=torch.float32)
        good = -torch.expm1(-x); naive = 1 - torch.exp(-x)
        check("24 -expm1 accuracy", float(((good - x).abs() / x).max()) < 1e-6
              and float(((naive - x).abs() / x).max()) > 1e-3)
        eps = 1e-6; D0 = torch.zeros(4, dtype=dt)
        with torch.enable_grad():
            J = torch.autograd.functional.jacobian(lambda x: decoder_norm(x[None], eps)[0], D0)
        sn = float(torch.linalg.eigvalsh(J.t() @ J).max().sqrt())
        check("25 decoder Jacobian bound", sn <= 1 / eps * (1 + 1e-6))
        with torch.enable_grad():
            kap = torch.tensor(-20.0, dtype=dt, requires_grad=True)
            laf = torch.randn(1000, 5, dtype=dt) - 3
            labp = torch.randint(0, 5, (1000,)); labm = labp.clone(); labm[:200] = 5
            ce_p = F.cross_entropy(torch.cat([laf, kap.reshape(1, 1).expand(1000, 1)], -1), labp)
            gp = torch.autograd.grad(ce_p, kap, retain_graph=True)[0]
            ce_m = F.cross_entropy(torch.cat([laf, kap.reshape(1, 1).expand(1000, 1)], -1), labm)
            gm = torch.autograd.grad(ce_m, kap)[0]
        check("26 no-match bounds kappa", float(gp) > 0 and float(gm) < 0)
        sim27 = simulate_fusion(p, ep0, T); la27 = log_doses(p, s, t, t_q)
        r_none = evaluate(p, ep0, 20, C=None); r_T = evaluate(p, ep0, 20, C=T)
        check("27 C>=T bit-for-bit regression", sim27["Z"] is Z
              and float((sim27["la"] - la27).abs().max()) == 0.0
              and bool((sim27["label"] == ep0["tgt"]).all())
              and all(abs(r_none[kk] - r_T[kk]) < 1e-15 for kk in ("loss", "ce", "gate_best")))
        nq, dq, bq = 20000, 64, 21.31
        zi = F.normalize(torch.randn(nq, dq, dtype=dt), dim=-1)
        zj = F.normalize(torch.randn(nq, dq, dtype=dt), dim=-1)
        u = F.normalize(torch.randn(nq, dq, dtype=dt), dim=-1)
        lai = torch.rand(nq, dtype=dt) * 5 - 6; laj = torch.rand(nq, dtype=dt) * 5 - 6
        bt = torch.tensor(bq, dtype=dt)
        mu, laf_, r_, delta_, w_, Gij_, omr_, rdir_ = fuse_pair_math(zi, zj, lai, laj, bt)
        Ep = (laf_ + bq * (mu * u).sum(-1)).exp()
        Eij = (lai + bq * (zi * u).sum(-1)).exp() + (laj + bq * (zj * u).sum(-1)).exp()
        check("28 Thm A domination", int(((Ep - Eij) > 1e-9).sum()) == 0,
              f"max resid {float((Ep-Eij).max()):.2e}")
        cc = -1.37
        m1, l1, *_ = fuse_pair_math(zi[:5], zj[:5], lai[:5], laj[:5], bt)
        m2_, l2_, *_ = fuse_pair_math(zi[:5], zj[:5], lai[:5] + cc, laj[:5] + cc, bt)
        check("29 Thm B commutation", float((m1 - m2_).abs().max()) < 1e-14
              and float((l2_ - (l1 + cc)).abs().max()) < 1e-13)
        pd30 = Params(omega=0.7, Gamma=0.3, eps_bar=0.3, beta=16.0, kappa=-8.0, dtype=dt)
        zA = F.normalize(torch.randn(d, dtype=dt), dim=-1)
        zB = F.normalize(zA + 0.01 * F.normalize(torch.randn(d, dtype=dt), dim=-1), dim=-1)
        m30 = HopMem(pd30, d, C=None, dtype=dt)
        m30.write(zA, 1.0, 0.0, prov=0); m30.write(zB, 1.0, 0.0, prov=1)
        S0 = m30.S
        a0, a1 = [math.exp(at["la"]) for at in m30.atoms]
        mu30, la30, r30, dl30, *_ = fuse_pair_math(zA[None], zB[None],
            torch.tensor([m30.atoms[0]["la"]], dtype=dt), torch.tensor([m30.atoms[1]["la"]], dtype=dt),
            torch.tensor(16.0, dtype=dt))
        A30 = a0 + a1; Sexp = S0 - A30 * (1 - math.exp(-float(dl30[0])))
        Mexp = (m30.M - a0 * torch.outer(zA, zA) - a1 * torch.outer(zB, zB)
                + A30 * math.exp(-float(dl30[0])) * torch.outer(mu30[0], mu30[0]))
        m30.C = 1; m30._compress_on_overflow()
        ev30 = torch.linalg.eigvalsh(m30.W)
        check("30 merge accounting (tr=1, PSD, deficit->J)", m30.n_fuse == 1
              and float((m30.W - ((1 - Sexp) / d * eye + Mexp)).abs().max()) < 1e-10
              and abs(m30.S - Sexp) < 1e-11 and float(ev30.min()) > 0,
              f"n_fuse={m30.n_fuse}")
        tau31, kap31, bet31 = 0.5, -5.0, 16.0; thr31 = kap31 + math.log(tau31)
        zd = F.normalize(torch.randn(10000, d, dtype=dt), dim=-1)
        la_dead = torch.full((10000,), thr31 - 0.1, dtype=dt)
        gap_dead = bet31 * (zd * zd[:1]).sum(-1) + la_dead - (bet31 + kap31)
        check("31 gate-dead criterion", float(gap_dead.max()) <= math.log(tau31) + 1e-12
              and bet31 + (thr31 + 0.01) - (bet31 + kap31) > math.log(tau31))
        Z32 = F.normalize(torch.randn(1, 6, d, dtype=dt), dim=-1)
        la32 = torch.rand(1, 6, dtype=dt) * 6 - 8
        p32 = Params(beta=16.0, kappa=-5.0, dtype=dt)
        dead32 = la32[0] <= -5.0 + math.log(0.5); keep = (~dead32).nonzero().squeeze(1)
        q32 = F.normalize(torch.randn(5000, d, dtype=dt), dim=-1)
        gf = gate_gap(Z32.expand(5000, -1, -1), la32.expand(5000, -1), q32, p32)
        gk = gate_gap(Z32[:, keep].expand(5000, -1, -1), la32[:, keep].expand(5000, -1), q32, p32)
        check("32 gate-dead deletion decision-invariant",
              bool(((gf > math.log(0.5)) == (gk > math.log(0.5))).all()) and int(dead32.sum()) > 0)
        pd33 = Params(omega=0.7, Gamma=1e-9, eps_bar=0.01, beta=16.0, kappa=-8.0, dtype=dt)
        m33 = HopMem(pd33, d, C=None, dtype=dt)
        zbase = F.normalize(torch.randn(d, dtype=dt), dim=-1)
        for i in range(4):
            m33.write(F.normalize(zbase + 0.01 * F.normalize(torch.randn(d, dtype=dt), dim=-1), dim=-1),
                      1.0, 0.0, prov=i)
        base33 = float(torch.logsumexp(m33.log_alpha, 0))
        m33.C = 1; m33._compress_on_overflow()
        la_fin = float(m33.log_alpha[0]); Lam_fin = float(m33.Lam_vec[0])
        wr = torch.rand(1000, dtype=dt); Gr = torch.rand(1000, dtype=dt) * 2 - 1
        dl_s, r_s, omr_s = resultant_deficit(wr, 1 - Gr, bt)
        r_dir = (1 - 2 * wr * (1 - wr) * (1 - Gr)).clamp_min(0).sqrt()
        check("33 nats ledger: chain bound + stable delta", m33.n_fuse == 3
              and la_fin >= base33 - Lam_fin - 1e-9
              and float((dl_s - bq * (1 - r_dir)).abs().max()) < 1e-11,
              f"n_fuse={m33.n_fuse} Lam={Lam_fin:.4f}")
        m34 = HopMem(pg, d, C=3, tau=0.5, tau_dead=0.5, dtype=dt)
        for i in range(T):
            m34.write(Z[0, i], float(s[0, i]), float(t[0, i]), prov=i)
        m34._advance(float(t_q[0]))
        sim34 = simulate_fusion(pg, epg, 3, tau_dead=0.5)
        ks = {a["prov"]: (a["z"], a["la"], a["Lam"]) for a in m34.atoms}
        kb = {}
        for kk in range(sim34["K"]):
            if bool(sim34["alive"][0, kk]):
                pr = frozenset(sim34["prov"][0, kk].nonzero().squeeze(1).tolist())
                kb[pr] = (sim34["Z"][0, kk], float(sim34["la"][0, kk]), float(sim34["Lam"][0, kk]))
        ok34 = set(ks) == set(kb); err34 = 0.0
        if ok34:
            for pr in ks:
                err34 = max(err34, float((ks[pr][0] - kb[pr][0]).abs().max()),
                            abs(ks[pr][1] - kb[pr][1]), abs(ks[pr][2] - kb[pr][2]))
        check("34 streaming == batched replay", ok34 and err34 < 1e-9, f"err={err34:.1e}")
        p35 = Params(omega=1e-3, Gamma=1e-9, beta=16.0, kappa=-8.0, eps_bar=0.01, dtype=dt)
        m35 = HopMem(p35, 2, C=1, tau=0.5, dtype=dt)
        e1 = torch.tensor([1.0, 0.0], dtype=dt)
        m35.write(e1, 1.0, 0.0, prov=0); m35.write(-e1, 1.0, 0.0, prov=1)
        check("35 C=1 antipodal: merge masked, EVICT fires, ledger ~ln2",
              m35.n_evict == 1 and m35.n_fuse == 0 and len(m35.atoms) == 1
              and abs(m35.sum_delta - math.log(2)) < 1e-3
              and (m35.retained(0) == 0.0 or m35.retained(1) == 0.0),
              f"n_evict={m35.n_evict} ledger={m35.sum_delta:.4f}")
        ai, aj, G36 = 0.3, 0.1, 0.99
        zi36 = F.normalize(torch.randn(1, d, dtype=dt), dim=-1)[0]
        e36 = F.normalize(torch.randn(d, dtype=dt), dim=-1)
        e36 = F.normalize(e36 - (e36 * zi36).sum() * zi36, dim=-1)
        zj36 = G36 * zi36 + math.sqrt(1 - G36 * G36) * e36
        wi = ai / (ai + aj); m36 = wi * zi36 + (1 - wi) * zj36; mu36 = m36 / m36.norm(); S36 = ai + aj
        exact = (ai / S36) * (1 - float(zi36 @ mu36)) + (aj / S36) * (1 - float(zj36 @ mu36))
        predC = (ai * aj / ((ai + aj) * S36)) * (1 - G36)
        m36b = HopMem(p, d, C=5, dtype=dt); cap_ok = True
        for i in range(60):
            m36b.write(F.normalize(torch.randn(d, dtype=dt), dim=-1), float(torch.rand(1)) * 2, float(i) * 0.2, prov=i)
            if i % 7 == 0:
                m36b.reinforce(F.normalize(torch.randn(d, dtype=dt), dim=-1), 1.0, 0.5, 0.3)
            cap_ok &= len(m36b.atoms) <= 5
        check("36 Prop C first-order; capacity <= C", abs(exact - predC) < 0.05 * predC + 1e-9 and cap_ok)

        # ---------------- new in v2.3.1 (37-44) ----------------
        pd = Params(omega=0.3, Gamma=1e-6, eps_bar=0.01, beta=16.0, kappa=-30.0, dtype=dt)
        Zs37 = F.normalize(torch.randn(40, 64, dtype=dt), dim=-1)
        ep37 = {"Z": Zs37[None], "s": torch.rand(1, 40, dtype=dt) * 3 + 0.05,
                "t": torch.zeros(1, 40, dtype=dt), "t_q": torch.zeros(1, dtype=dt),
                "tgt": torch.zeros(1, dtype=torch.long), "z_t": Zs37[None, 0],
                "q": Zs37[None, 0], "absent": torch.zeros(1, dtype=torch.bool)}
        sA = simulate_fusion(pd, ep37, 6, tau_dead=1e-12, policy="debit")
        sB = simulate_fusion(pd, ep37, 6, tau_dead=1e-12, policy="lightest")
        setA = {frozenset(sA["prov"][0, k].nonzero().squeeze(1).tolist())
                for k in range(sA["K"]) if bool(sA["alive"][0, k])}
        setB = {frozenset(sB["prov"][0, k].nonzero().squeeze(1).tolist())
                for k in range(sB["K"]) if bool(sB["alive"][0, k])}
        check("37 spread reduction == streaming dose-ordered eviction",
              setA == setB and float(sA["stats"]["n_fuse"][0]) == 0.0,
              f"evict={float(sA['stats']['n_evict'][0]):.0f}")
        zdup = F.normalize(torch.randn(1, 32, dtype=dt), dim=-1)
        ep38 = {"Z": zdup.expand(1, 2, 32).contiguous(), "s": torch.ones(1, 2, dtype=dt),
                "t": torch.zeros(1, 2, dtype=dt), "t_q": torch.zeros(1, dtype=dt),
                "tgt": torch.zeros(1, dtype=torch.long), "z_t": zdup,
                "q": zdup, "absent": torch.zeros(1, dtype=torch.bool)}
        pw = Params(omega=1e-9, Gamma=1e-9, pure_writes=True, beta=16.0, kappa=-30.0, dtype=dt)
        s38 = simulate_fusion(pw, ep38, 1, tau_dead=1e-12, policy="debit")
        l0_38 = float(log_injected(pw, torch.ones(1, dtype=dt))[0])
        check("38 duplicate reduction: free merge, evidence preserved",
              float(s38["stats"]["n_fuse"][0]) == 1.0 and float(s38["stats"]["n_evict"][0]) == 0.0
              and abs(float(s38["la"][0, 0]) - (l0_38 + math.log(2))) < 1e-6
              and float(s38["Lam"][0, 0]) < 1e-12)
        b39 = 17.0
        th_star = 2 * math.acos(1 - math.log(2) / b39)
        za = torch.tensor([[1.0, 0.0]], dtype=dt)
        zb = torch.tensor([[math.cos(th_star), math.sin(th_star)]], dtype=dt)
        Z39 = torch.stack([za, zb], 1)
        G39 = Z39 @ Z39.transpose(1, 2)
        la39 = torch.zeros(1, 2, dtype=dt)
        _, isf, df, de, _, _, _ = debit_candidates(la39, G39, torch.ones(1, 2, dtype=torch.bool),
                                                   torch.tensor(b39, dtype=dt))
        check("39 crossover identity delta_fuse == delta_evict == ln2",
              abs(float(df[0, 0, 1]) - math.log(2)) < 1e-12
              and abs(float(de[0, 0, 1]) - math.log(2)) < 1e-12,
              f"theta*={th_star:.4f} df={float(df[0,0,1]):.6f}")
        n40, K40, d40 = 10000, 6, 16
        Z40 = F.normalize(torch.randn(n40, K40, d40, dtype=dt), dim=-1)
        G40 = Z40 @ Z40.transpose(1, 2)
        la40 = torch.rand(n40, K40, dtype=dt) * 8 - 9
        al40 = torch.ones(n40, K40, dtype=torch.bool)
        b40 = torch.tensor(16.0, dtype=dt)
        logD40, isf40, df40, de40, _, _, _ = debit_candidates(la40, G40, al40, b40)
        best = logD40.reshape(n40, K40 * K40).min(1).values
        ar40 = torch.arange(n40)
        lo = la40.argmin(1); hi = la40.argmax(1)
        A_lh = torch.logaddexp(la40[ar40, lo], la40[ar40, hi])
        S40 = la40.exp().sum(-1)
        bound = A_lh - S40.log() + de40[ar40, lo, hi].clamp_min(TINY).log()
        check("40 Thm E per-event debit dominance", int((best > bound + 1e-9).sum()) == 0,
              f"max excess {float((best-bound).max()):.2e}")
        m41 = HopMem(pd, 32, C=5, dtype=dt)
        gerr = 0.0
        for i in range(1000):
            if i % 3:
                zi41 = F.normalize(torch.randn(32, dtype=dt), dim=-1)
            elif m41.atoms:
                zi41 = F.normalize(m41.atoms[0]["z"]
                                   + 0.02 * F.normalize(torch.randn(32, dtype=dt), dim=-1), dim=-1)
            else:
                zi41 = F.normalize(torch.randn(32, dtype=dt), dim=-1)
            m41.write(zi41, float(torch.rand(1)) * 2 + 0.1, float(i) * 0.01, prov=i)
            if m41.atoms:
                Zt = m41.Zs
                gerr = max(gerr, float((m41.G - Zt @ Zt.t()).abs().max()))
        check("41 incremental Gram == rebuild (1000 mixed ops)", gerr < 1e-10, f"max err {gerr:.2e}")
        m42 = HopMem(pd33, d, C=None, dtype=dt)
        for i in range(3):
            m42.write(F.normalize(zbase + 0.01 * F.normalize(torch.randn(d, dtype=dt), dim=-1), dim=-1),
                      1.0, 0.0, prov=i)
        base42 = float(torch.logsumexp(m42.log_alpha, 0))
        m42.C = 1; m42._compress_on_overflow()
        m42e = HopMem(pd, 32, C=2, dtype=dt)
        for i in range(12):
            m42e.write(F.normalize(torch.randn(32, dtype=dt), dim=-1), 1.0, float(i) * 0.01, prov=i)
        check("42 ledger nats: chain bound, destroyed -> 0",
              float(m42.log_alpha[0]) >= base42 - float(m42.Lam_vec[0]) - 1e-9
              and len(m42e.destroyed) > 0 and all(m42e.retained(k) == 0.0 for k in m42e.destroyed)
              and m42e.sum_delta > 0)
        sh = -1.37
        logD43, isf43, *_ = debit_candidates(la40 + sh, G40, al40, b40)
        fin43 = torch.isfinite(logD40) & torch.isfinite(logD43)
        d43 = (logD43 - logD40)[fin43]
        i_a = logD40.reshape(n40, K40 * K40).argmin(1)
        i_b = logD43.reshape(n40, K40 * K40).argmin(1)
        check("43 policy commutation (Delta and argmin invariant)",
              float(d43.abs().max()) < 1e-13 and bool((i_a == i_b).all())
              and bool((isf43[fin43] == isf40[fin43]).all()),
              f"max |dDelta|={float(d43.abs().max()):.2e}")
        al_ = la40.exp(); A_ = al_.unsqueeze(2) + al_.unsqueeze(1)
        S_ = al_.sum(-1)[:, None, None]
        omg_ = (1 - G40).clamp_min(0)
        c_ = al_.unsqueeze(2) * al_.unsqueeze(1) / A_ * omg_
        w_h = torch.maximum(al_.unsqueeze(2), al_.unsqueeze(1)) / A_
        p_l = torch.minimum(al_.unsqueeze(2), al_.unsqueeze(1)) / S_
        errF = float((c_ / S_ - p_l * w_h * omg_).abs().max())
        logA_ = torch.logaddexp(la40.unsqueeze(2), la40.unsqueeze(1))
        logS_ = al_.sum(-1).log()[:, None, None]
        le40 = logA_ - logS_ + de40.clamp_min(TINY).log()
        valid40 = ~torch.eye(K40, dtype=torch.bool)[None]
        le40 = torch.where(valid40, le40, torch.full_like(le40, float("inf")))
        ie = le40.reshape(n40, K40 * K40).argmin(1)
        ei, ej = ie // K40, ie % K40
        pair_ok = ((ei == lo) & (ej == hi)) | ((ei == hi) & (ej == lo))
        check("44 Lemma F identity; Lemma G cheapest eviction",
              errF < 1e-12 and bool(pair_ok.all()), f"errF={errF:.2e}")
    assert all(res), "selftest failed: %d/%d" % (sum(res), len(res))
    print(f"all {len(res)} passed (HopMem v{__version__}, spec section 24 suite)")


@torch.no_grad()
def calibrate(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    dt = torch.float64
    p = Params(read=a.read, dtype=dt, diffusion=not a.no_diffusion, pure_writes=a.pure_writes).to(dev)
    d, T, B, N = a.d, a.T, a.batch, a.N
    ar = torch.arange(B, device=dev)
    print("read:", a.read, "theta:", {k: round(v, 6) for k, v in p.theta().items()}, f"d={d} T={T}")
    for arm in ("R", "A"):
        Z = F.normalize(torch.randn(B, T, d, generator=torch.Generator(device="cpu").manual_seed(a.seed),
                                    dtype=dt).to(dev), dim=-1)
        if arm == "A":
            mu = F.normalize(torch.randn(B, 1, d, dtype=dt, device=dev), dim=-1)
            Z = F.normalize(math.sqrt(a.rho) * mu + math.sqrt(1 - a.rho) * Z, dim=-1)
        s = torch.empty(B, T, dtype=dt, device=dev).exponential_()
        t = torch.empty(B, T, dtype=dt, device=dev).exponential_().cumsum(-1)
        t_q = t[:, -1] + torch.empty(B, dtype=dt, device=dev).exponential_()
        tgt = torch.randint(T, (B,), device=dev); z_t = Z[ar, tgt]
        la = log_doses(p, s, t, t_q); alpha = la.exp()
        v1, _ = top_eigvec(Z, alpha)
        margin, fs, fh = quad_stability(Z, alpha, tgt, 0.01)
        fhm = float(torch.nanmean(fh)) if bool(torch.isfinite(fh).any()) else float("nan")
        print(f"\nArm {arm}: S={float(alpha.sum(-1).mean()):.3f} stable {float((margin>0).float().mean()):.0%} "
              f"floor(Hessian) {fhm:.3f}")
        for phi in a.phis:
            e = torch.randn(B, d, dtype=dt, device=dev)
            e = F.normalize(e - (e * z_t).sum(-1, keepdim=True) * z_t, dim=-1)
            q = math.cos(phi) * z_t + math.sin(phi) * e
            u, info = read(Z, la, q, p, N, diagnostics=True, auto_gamma=a.auto_gamma)
            ang = lambda x: torch.acos((x * z_t).sum(-1).clamp(-1, 1))
            print(f"   phi={phi:.2f} closure@N {float(((phi-ang(u))/phi).mean()):+.2f} "
                  f"E_up {info['E_up']:.0e}")


def main():
    ap = argparse.ArgumentParser(description=f"HopMem v{__version__} (spec v2.3.1)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    sub.add_parser("lm_check")
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
            c.add_argument("--cal_batch", type=int, default=1024)
            c.add_argument("--lr", type=float, default=3e-2)
            c.add_argument("--schedule", choices=["staged", "probe"], default="probe")
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
            c.add_argument("--gate_tau", type=float, default=None)
            c.add_argument("--C", type=int, default=64)
            c.add_argument("--c_min", type=int, default=4)
            c.add_argument("--fuse", choices=["on", "off"], default="on")
            c.add_argument("--fuse_policy", choices=["debit", "ward", "lightest", "random", "delta"],
                           default="debit")
            c.add_argument("--out", default="hopmem_theta.pt")
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "lm_check":
        lm_interface_check()
    elif a.cmd == "calibrate":
        calibrate(a)
    else:
        train(a)


if __name__ == "__main__":
    main()