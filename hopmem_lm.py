#!/usr/bin/env python
"""hopmem_lm.py -- measurement of spec section 23 (LM interface).

    python hopmem_lm.py --geometry
    python hopmem_lm.py --recall --caps 8 16 32 --center
    python hopmem_lm.py --recall --calibrate 400 --caps 8 16 32 --center
    python hopmem_lm.py --recall --query_angle 0.7 --calibrate 400 --center

MEASURED (GPT-2, layer 6 centered):
  * RETRIEVAL FIDELITY TRANSFERS.  On committed present trials the angular error is
    0.0097 (natural query) / 0.0104 (angle 0.7) against cue-only 0.1586 / 0.2352 and
    kNN 0.0026 / 0.0000.  The Gibbs readout works on LM states.
  * THE GATE IS THE BOTTLENECK, and it is a SEPARABILITY failure, not a tuning one.
    kappa must exceed the absent-max logit and sit below the present-target logit, but
    the dose prior makes present targets OLD (ell ~ -17) and absent-max RECENT
    (ell ~ -2.3), inverting the order.  The required-kappa offset flips sign across
    calibration (+4.69 -> -3.64), which is the signature of overshoot on an
    inseparable problem.  The null-separability diagnostic below tests this directly.
  * layers 0-9 centered are gate-selective (0.3-1.2%); layer 12 is unusable (50.5%).
  * the raw dose SPREAD is misleading: the s=0 first token sits at ln(1e-30) = -69.08.
    Live-atom spread is reported separately.

METRIC WARNING (two confounds, both now handled):
  1. block accuracy counts correct ABSTENTION when the target's block was DESTROYED
     (label = null index).  blk|surv therefore conditions on the block surviving, and
     `surv` is reported so the irrecoverable fraction is visible.
  2. `acc` in hopmem.evaluate mixes present and absent.  Everything here is decomposed.

ARCHITECTURAL LIMIT: tied key/value returns the MATCHED past state, not its continuation.
Content-addressed recall, NOT induction.

DEVICE-INDEPENDENT EPISODES: all randomness on a CPU generator, moved to device.
"""
import argparse
import math

import torch
import torch.nn.functional as F

import hopmem as H

SAMPLE = ("The ledger records how many nats of evidence were removed. "
          "A bounded memory must decide what to forget. "
          "The ledger records how many nats of evidence were removed by compression.")


def load_lm(name, device):
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        print("pip install transformers"); raise
    tok = AutoTokenizer.from_pretrained(name)
    lm = AutoModelForCausalLM.from_pretrained(name).to(device).eval()
    return tok, lm


@torch.no_grad()
def states(ids, lm, layer):
    out = lm(ids.unsqueeze(0), output_hidden_states=True)
    h = out.hidden_states[layer][0]
    lp = torch.log_softmax(out.logits[0, :-1].float(), -1)
    s = -lp.gather(1, ids[1:, None]).squeeze(1)
    s = torch.cat([s.new_zeros(1), s.clamp_min(0)])
    return h, s


def offdiag(Z):
    G = Z @ Z.t()
    return G[~torch.eye(Z.shape[0], dtype=torch.bool, device=Z.device)]


def cosabs(x, y):
    return (x * y).sum(-1).abs().clamp(max=1.0)


def cos_stats(Z):
    off = offdiag(Z)
    q = torch.quantile(off.float(), torch.tensor([0.5, 0.9, 0.99], device=off.device))
    return float(off.mean()), float(q[0]), float(q[1]), float(q[2]), float(off.max())


def nz(v, w, pc):
    return f"{v:>{w}.{pc}f}" if v == v else " " * (w - 1) + "-"


@torch.no_grad()
def geometry(a, tok, lm, dev):
    text = open(a.text, encoding="utf-8").read() if a.text else SAMPLE * 8
    ids = tok(text, return_tensors="pt", truncation=True, max_length=a.max_len).input_ids[0].to(dev)
    if len(ids) < 8:
        print("need more text"); return
    beta = a.beta
    g_gate = 1.0 - 1.0 / beta
    print(f"\nGEOMETRY TRANSFER  ({len(ids)} tokens, model {a.model})")
    print(f"transferred beta={beta:.2f}: gate needs cos>={g_gate:.3f} "
          f"(MiniLM median was 0.201)")
    print(f"   {'layer':>6} {'mode':>9} {'mean':>7} {'p50':>7} {'p90':>7} {'p99':>7} {'max':>7} "
          f"{'gate-eligible':>14} {'beta@p99':>9}")
    for layer in a.layers:
        h, _ = states(ids, lm, layer)
        for mode in ("raw", "centered"):
            hh = h if mode == "raw" else h - h.mean(0, keepdim=True)
            Z = F.normalize(hh.double(), dim=-1)
            mu, p50, p90, p99, mx = cos_stats(Z)
            frac = float((offdiag(Z) >= g_gate).double().mean())
            print(f"   {layer:>6} {mode:>9} {mu:>7.3f} {p50:>7.3f} {p90:>7.3f} {p99:>7.3f} "
                  f"{mx:>7.3f} {frac:>13.1%} {1.0/max(1e-6,1.0-p99):>9.1f}")


@torch.no_grad()
def guard(a, tok, lm, dev, beta):
    text = open(a.text, encoding="utf-8").read() if a.text else SAMPLE * 8
    ids = tok(text, return_tensors="pt", truncation=True, max_length=a.max_len).input_ids[0].to(dev)
    h, _ = states(ids, lm, a.layer)
    if a.center:
        h = h - h.mean(0, keepdim=True)
    Z = F.normalize(h.double(), dim=-1)
    frac = float((offdiag(Z) >= 1.0 - 1.0 / beta).double().mean())
    print(f"geometry guard: layer {a.layer} {'centered' if a.center else 'raw'} -> "
          f"gate-eligible {frac:.1%} at beta={beta:.2f}")
    if frac > 0.05:
        print("   WARNING: >5% of pairs mergeable; use a lower --layer, --center, or refit beta.")
    return frac


def recall_batch(a, tok, lm, dev, n, gen):
    Zs, Ss, Ts, TQ, TG, ZT, Q, AB = [], [], [], [], [], [], [], []
    Ttot = a.T + a.span
    for _ in range(n):
        base = torch.randint(a.vlo, a.vhi, (a.T,), generator=gen)
        i = int(torch.randint(0, a.T - a.span + 1, (1,), generator=gen).item())
        present = bool(torch.rand(1, generator=gen).item() >= a.p_absent)
        app = base[i:i + a.span] if present else torch.randint(
            a.vlo, a.vhi, (a.span,), generator=gen)
        ids = torch.cat([base, app]).to(dev)
        h, s = states(ids, lm, a.layer)
        if a.center:
            h = h - h.mean(0, keepdim=True)
        Zall = F.normalize(h.double(), dim=-1)
        tgt = i + a.span - 1 if present else Ttot - 1
        z_t = Zall[tgt] if present else Zall[-1]
        q = Zall[-1]
        if a.query_angle > 0:
            e = torch.randn(Zall.shape[1], generator=gen).to(dev).double()
            e = F.normalize(e - (e @ z_t) * z_t, dim=-1)
            q = math.cos(a.query_angle) * z_t + math.sin(a.query_angle) * e
        Zs.append(Zall[:-1])
        Ss.append(s[:-1].double())
        Ts.append(torch.arange(Ttot - 1, device=dev, dtype=torch.float64))
        TQ.append(torch.tensor(float(Ttot - 1), dtype=torch.float64, device=dev))
        TG.append(torch.tensor(tgt, dtype=torch.long, device=dev))
        Q.append(q); ZT.append(z_t)
        AB.append(torch.tensor(not present, device=dev))
    return {"Z": torch.stack(Zs), "s": torch.stack(Ss), "t": torch.stack(Ts),
            "t_q": torch.stack(TQ), "tgt": torch.stack(TG), "z_t": torch.stack(ZT),
            "q": torch.stack(Q), "absent": torch.stack(AB)}


def slice_ep(ep, idx):
    return {k: v[idx] for k, v in ep.items()}


@torch.no_grad()
def probe(p, ep, C, tau, N, policy):
    """Present trials split by whether the target's BLOCK SURVIVES.
    blk|surv and cmt|surv condition on survival; loss|cmt is over all committed present."""
    T = ep["Z"].shape[1]
    sim = H.simulate_fusion(p, ep, T if C is None else C,
                            tau_dead=(tau if tau is not None else 1.0), policy=policy)
    Z, la, lab, K = sim["Z"], sim["la"], sim["label"], sim["K"]
    lg = H.gibbs_logits(Z, la, ep["q"], p)
    am = lg.argmax(-1)
    gap = lg[..., :K].max(-1).values - lg[..., -1]
    commit = gap > math.log(tau if tau is not None else 1.0)
    u = H.read(Z, la, ep["q"], p, N)
    ang = 1 - cosabs(u, ep["z_t"])
    pres = ep["tgt"] < T
    surv = pres & (lab < K)
    cm = commit & pres
    gated = torch.where(commit[:, None], u, ep["q"])
    gp = 1 - cosabs(gated, ep["z_t"])
    nan = float("nan")
    return {
        "surv": float(surv[pres].double().mean()),
        "blk": float((am == lab)[surv].double().mean()) if bool(surv.any()) else nan,
        "cmt": float(commit[surv].double().mean()) if bool(surv.any()) else nan,
        "lcmt": float(ang[cm].mean()) if bool(cm.any()) else nan,
        "pres": float(gp[pres].mean()),
        "abs": float(gp[~pres].mean()) if bool((~pres).any()) else nan,
        "absn": float((~commit)[~pres].double().mean()) if bool((~pres).any()) else nan,
        "led": float(sim["stats"]["sum_delta"].mean()) if C is not None else nan,
    }


@torch.no_grad()
def diagnose(a, dev, p, tau, ep):
    B, T, _ = ep["Z"].shape
    ar = torch.arange(B, device=dev)
    pres = ep["tgt"] < T
    tgt = ep["tgt"].clamp_max(T - 1)
    la = H.log_doses(p, ep["s"], ep["t"], ep["t_q"])
    cc = torch.einsum("btd,bd->bt", ep["Z"], ep["q"])
    lt, ct = la[ar, tgt], cc[ar, tgt]
    dct, dl = ct[:, None] - cc, la - lt[:, None]
    pos = dct > 1e-9
    ratio = torch.where(pos, dl / dct.clamp_min(1e-9), torch.full_like(dl, float("-inf")))
    ratio[ar, tgt] = float("-inf")
    need = ratio.max(-1).values
    unreachable = ((dct <= 1e-9) & (dl > 0)).any(-1)
    beta, kappa = float(p.beta), float(p.kappa)
    liv = la > kappa + math.log(max(tau if tau is not None else 1.0, 1e-12))
    hi = torch.where(liv, la, torch.full_like(la, float("-inf"))).max(-1).values
    lo = torch.where(liv, la, torch.full_like(la, float("inf"))).min(-1).values
    print("\nMISMATCH DIAGNOSTIC")
    print(f"   dose spread LIVE {float((hi-lo).median()):.2f} nats; raw "
          f"{float((la.max(-1).values-la.min(-1).values).median()):.2f} "
          f"({int((~liv).sum())} floor/gate-dead entries incl. the s=0 token at -69.08)")
    good = pres & ~unreachable
    if bool(good.any()):
        q = torch.quantile(need[good].float(), torch.tensor([0.5, 0.9], device=dev))
        print(f"   beta to beat all ATOMS: median {float(q[0]):.1f}, p90 {float(q[1]):.1f} "
              f"(current {beta:.2f}); gate window there cos>={1.0-1.0/max(float(q[0]),1e-6):.4f}")
    if bool(pres.any()):
        print(f"   present unreachable at ANY beta (distractor better on both axes): "
              f"{float(unreachable[pres].double().mean()):.1%}")
    # --- null separability: the decisive test ---
    Lp = (beta * ct + lt)[pres]
    La = (beta * cc + la).max(-1).values[~pres]
    if Lp.numel() and La.numel():
        qa = torch.quantile(La.float(), torch.tensor([0.1, 0.5, 0.9], device=dev))
        print(f"   NULL SEPARABILITY  present-target logit median {float(Lp.median()):.2f}; "
              f"absent-max p10/50/90 {float(qa[0]):.2f}/{float(qa[1]):.2f}/{float(qa[2]):.2f}")
        below = float((Lp < qa[1]).double().mean())
        print(f"   {below:.1%} of present targets fall BELOW the absent median -> no single "
              f"kappa accepts them while rejecting typical absent queries")
        lo_k, hi_k = float(La.max()) - beta, float(Lp.min()) - beta
        print(f"   perfect-separation kappa window [{lo_k:.2f}, {hi_k:.2f}] : "
              f"{'EMPTY' if lo_k >= hi_k else 'nonempty'}   (current {kappa:.2f})")


@torch.no_grad()
def recall(a, tok, lm, dev, p, tau, tag=""):
    gen = torch.Generator().manual_seed(a.seed)
    ep = recall_batch(a, tok, lm, dev, a.n, gen)
    Tstore = ep["Z"].shape[1]
    pres = ep["tgt"] < Tstore
    cue = float((1 - cosabs(ep["q"], ep["z_t"]))[pres].mean())
    print(f"\nCONTENT RECALL{tag}  (layer {a.layer}, {'centered' if a.center else 'raw'}, "
          f"n={a.n}, query_angle={a.query_angle})")
    print(f"   beta={float(p.beta):.2f} kappa={float(p.kappa):.2f} tau={tau:.4g}  "
          f"store T={Tstore}")
    print(f"   CUE-ONLY present loss {cue:.4f}  <- headroom reference")
    cc = torch.einsum("btd,bd->bt", ep["Z"], ep["q"])
    am = cc.argmax(-1)
    kn = ep["Z"][torch.arange(len(am), device=dev), am]
    kl = float((1 - cosabs(kn, ep["z_t"]))[pres].mean())
    print(f"   {'policy':>18} {'surv':>6} {'blk|surv':>9} {'cmt|surv':>9} {'loss|cmt':>9} "
          f"{'present':>8} {'absent':>8} {'abstain':>8} {'ledger':>7}")
    print(f"   {'kNN (unbounded)':>18} {1.0:>6.3f} "
          f"{float(((am == ep['tgt']) & pres)[pres].double().mean()):>9.3f} "
          f"{1.0:>9.3f} {kl:>9.4f} {kl:>8.4f} {'-':>8} {'-':>8} {'-':>7}")
    for C in [None] + list(a.caps):
        if C is not None and C >= Tstore:
            continue
        r = probe(p, ep, C, tau, a.N, a.policy)
        name = "HopMem unbounded" if C is None else f"HopMem C={C}"
        print(f"   {name:>18} {r['surv']:>6.3f} {nz(r['blk'],9,3)} {nz(r['cmt'],9,3)} "
              f"{nz(r['lcmt'],9,4)} {r['pres']:>8.4f} {nz(r['abs'],8,4)} "
              f"{nz(r['absn'],8,3)} {nz(r['led'],7,2)}")
    print("   surv = target's block still exists; blk|surv / cmt|surv condition on it.")
    print("   loss|cmt = angular error on committed present trials (retrieval fidelity).")
    if a.diagnose:
        diagnose(a, dev, p, tau, ep)
    return ep


def calibrate_readout(a, tok, lm, dev, p, tau):
    pool_n = max(a.pool, a.batch * 4)
    print(f"\nRECALIBRATION on LM states ({a.calibrate} steps, pool {pool_n}, law frozen)")
    gen = torch.Generator().manual_seed(a.seed + 777)
    with torch.no_grad():
        pool = recall_batch(a, tok, lm, dev, pool_n, gen)
    for n_, prm in p.named_parameters():
        prm.requires_grad_(n_ in ("a_beta", "a_kappa"))
    opt = torch.optim.Adam([x for x in p.parameters() if x.requires_grad], lr=a.lr)
    with torch.enable_grad():
        for it in range(a.calibrate):
            idx = torch.randint(0, pool_n, (a.batch,), generator=gen).to(dev)
            ep = slice_ep(pool, idx)
            sim = H.simulate_fusion(p, ep, a.cal_C, tau_dead=1.0, policy=a.policy)
            lg = H.gibbs_logits(sim["Z"], sim["la"], ep["q"], p)
            loss = torch.nn.functional.cross_entropy(lg, sim["label"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([x for x in p.parameters() if x.requires_grad], 5.0)
            opt.step()
            if it % max(1, a.calibrate // 8) == 0:
                print(f"   step {it:4d} ce {loss.item():.4f} "
                      f"beta={float(p.beta.detach()):.2f} kappa={float(p.kappa.detach()):.2f}")
    for prm in p.parameters():
        prm.requires_grad_(False)
    with torch.no_grad():
        idx = torch.arange(min(a.batch * 8, pool_n), device=dev)
        tau = H.calibrate_tau_fused(p, slice_ep(pool, idx), a.N, C=a.cal_C,
                                    passes=2, policy=a.policy)
    print(f"   fitted beta={float(p.beta):.2f} kappa={float(p.kappa):.2f} tau={tau:.4g}")
    del pool
    return tau


@torch.no_grad()
def stream(a, tok, lm, dev, p, tau):
    text = open(a.text, encoding="utf-8").read() if a.text else SAMPLE
    ids = tok(text, return_tensors="pt", truncation=True, max_length=a.max_len).input_ids[0].to(dev)
    h, s = states(ids, lm, a.layer)
    if a.center:
        h = h - h.mean(0, keepdim=True)
    Z = F.normalize(h.double(), dim=-1)
    mem = H.HopMem(p, Z.shape[1], C=a.C, tau=tau, dtype=torch.float64, device=dev, policy=a.policy)
    print(f"\nSURPRISE-DRIVEN STREAM  ({len(ids)} tokens, C={a.C}, layer {a.layer})")
    for i in range(len(ids) - 1):
        mem.write(Z[i], float(s[i]), float(i), prov=i)
    mem._advance(float(len(ids)))
    print(f"   {len(mem.atoms)} atoms, {len(mem.destroyed)} destroyed, S={mem.S:.4f}; "
          f"{mem.n_fuse} merge / {mem.n_evict} evict / {mem.n_dead} gate-dead; "
          f"ledger {mem.sum_delta:.3f} nats")
    order = sorted(range(len(mem.atoms)), key=lambda k: -mem.atoms[k]["la"])
    for k in order[:a.topk]:
        at = mem.atoms[k]
        ids_k = sorted(at["prov"])
        toks = " | ".join(tok.decode([int(ids[j])]).strip() for j in ids_k[:6])
        print(f"      dose {math.exp(at['la']):.5f} retained {math.exp(-at['Lam']):.3f} "
              f"nll {float(s[ids_k[0]]):.2f} covers {len(ids_k)}: {toks}")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--text")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 3, 6, 9, 12])
    ap.add_argument("--center", action="store_true")
    ap.add_argument("--geometry", action="store_true")
    ap.add_argument("--recall", action="store_true")
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--diagnose", action="store_true", default=True)
    ap.add_argument("--no_diagnose", dest="diagnose", action="store_false")
    ap.add_argument("--query_angle", type=float, default=0.0)
    ap.add_argument("--calibrate", type=int, default=0)
    ap.add_argument("--cal_C", type=int, default=16)
    ap.add_argument("--pool", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-2)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--span", type=int, default=4)
    ap.add_argument("--caps", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--p_absent", type=float, default=0.2)
    ap.add_argument("--vlo", type=int, default=1000)
    ap.add_argument("--vhi", type=int, default=5000)
    ap.add_argument("--C", type=int, default=32)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--N", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy", choices=list(H.POLICIES), default=H.DEFAULT_POLICY)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--omega", type=float, default=0.01047)
    ap.add_argument("--Gamma", type=float, default=0.1811)
    ap.add_argument("--eps_bar", type=float, default=0.00068)
    ap.add_argument("--beta", type=float, default=20.70)
    ap.add_argument("--kappa", type=float, default=-17.41)
    ap.add_argument("--tau", type=float, default=0.6703)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    tok, lm = load_lm(a.model, dev)
    if a.ckpt:
        p, tau, cfg = H.load_checkpoint(a.ckpt, dev)
        p = p.double()
        tau = tau if tau is not None else a.tau
    else:
        p = H.Params(omega=a.omega, Gamma=a.Gamma, eps_bar=a.eps_bar,
                     beta=a.beta, kappa=a.kappa, dtype=torch.float64).to(dev)
        tau = a.tau
    for prm in p.parameters():
        prm.requires_grad_(False)

    if not (a.geometry or a.recall or a.stream):
        a.geometry = a.recall = True
    if a.geometry:
        geometry(a, tok, lm, dev)
    if a.recall:
        guard(a, tok, lm, dev, float(p.beta))
        recall(a, tok, lm, dev, p, tau)
        if a.calibrate:
            tau = calibrate_readout(a, tok, lm, dev, p, tau)
            guard(a, tok, lm, dev, float(p.beta))
            recall(a, tok, lm, dev, p, tau, tag=" (recalibrated)")
    if a.stream:
        guard(a, tok, lm, dev, float(p.beta))
        stream(a, tok, lm, dev, p, tau)


if __name__ == "__main__":
    main()