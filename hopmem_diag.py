#!/usr/bin/env python
"""hopmem_diag.py -- add-on for hopmem.py (spec v2.1.1; imports it, changes nothing in it).

    python hopmem_diag.py --ckpt hopmem_theta.pt                    # real data (uses the cached corpus)
    python hopmem_diag.py --ckpt hopmem_theta.pt --synthetic        # no network
    python hopmem_diag.py --ckpt hopmem_theta.pt --same_passage     # if you trained with it
    python hopmem_diag.py --ckpt hopmem_theta.pt --hard_k 4         # ambiguous cues (needs hopmem_hard.py)

Three questions the training log cannot answer:

 1. DECOMPOSED BASELINES, incl. kNN + one threshold on the max cosine (the like-for-like comparator
    for a gate that can abstain).  Plain nn is forced to commit on absent trials, so its aggregate
    number is dominated by that penalty and says nothing about present trials.
 2. PRIOR QUALITY: cue-free CE of the learned log-doses vs the data-generating (oracle) prior.  This,
    not the value of Gamma, is the right check that the forgetting law was recovered: CE is nearly
    flat in Gamma, so a 15% error in Gamma costs only a few thousandths of a nat.
 3. EVICTION: at T_big >> budget, keep the C atoms with the highest (a) learned log-dose, (b) recency,
    (c) random, (d) oracle prior, and measure how often the target survives and the deployed loss.
    This is a capacity-limited test of what the importance/decay law is worth as a priority.
"""
import argparse
import math

import torch

import hopmem as H


def cosabs(x, y):
    return (x * y).sum(-1).abs().clamp(max=1.0)


def row(name, loss, absent):
    print(f"   {name:38s} all {loss.mean():.4f}   present {loss[~absent].mean():.4f}   absent {loss[absent].mean():.4f}")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="hopmem_theta.pt")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--same_passage", action="store_true")
    ap.add_argument("--hard_k", type=int, default=0, help="competitors replaced by target neighbours (hopmem_hard.py)")
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--T_big", type=int, default=100)
    ap.add_argument("--caps", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--cal_batch", type=int, default=2048)
    ap.add_argument("--evict_batch", type=int, default=512)
    # corpus args: keep identical to the training run so the cached dataset is found
    ap.add_argument("--dataset", default="sedthh/gutenberg_english")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--n_books", type=int, default=100)
    ap.add_argument("--max_sent", type=int, default=1000)
    ap.add_argument("--min_words", type=int, default=8)
    ap.add_argument("--max_words", type=int, default=40)
    ap.add_argument("--cue_frac", type=float, default=0.5)
    ap.add_argument("--cache_dir", default="hopmem_cache")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--rho", type=float, default=0.3)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    p, tau, cfg = H.load_checkpoint(a.ckpt, dev)
    assert p.read == "gibbs", "diagnostics need the gibbs readout"
    N, lam, pa = cfg.get("N", 20), cfg.get("query_decay", 0.2), cfg.get("p_absent", 0.2)
    thr = math.log(tau if tau is not None else 1.0)

    Zs, Qs, doc = H.synthetic_corpus(a) if a.synthetic else H.build_corpus(a)
    Zs, Qs = Zs.to(dev), Qs.to(dev)
    P = len(Zs)
    b = (doc[1:] != doc[:-1]).nonzero().squeeze(1) + 1
    b = b[b >= int(0.9 * P)]
    cut = int(b[0]) if len(b) else int(0.9 * P)
    kw = dict(same_passage=a.same_passage, query_decay=lam, p_absent=pa)
    Sampler = H.EpisodeSampler
    if a.hard_k > 0:
        from hopmem_hard import HardSampler
        HardSampler.k_hard, Sampler = a.hard_k, HardSampler
    tr = Sampler(Zs[:cut], Qs[:cut], doc[:cut], T=a.T, **kw)
    ev = Sampler(Zs[cut:], Qs[cut:], doc[cut:], T=a.T, **kw)
    print(f"items {P}, d={Zs.shape[1]}, train {cut} / eval {P - cut}; checkpoint tau={tau}, "
          f"beta={float(p.beta):.2f} kappa={float(p.kappa):.2f} omega={float(p.omega):.4f} Gamma={float(p.Gamma):.4f}")

    def parts(ep):
        la = H.log_doses(p, ep["s"], ep["t"], ep["t_q"])
        Z, q, z = ep["Z"], ep["q"], ep["z_t"]
        u = H.read(Z, la, q, p, N)
        cc = torch.einsum("btd,bd->bt", Z, q)
        cmax, am = cc.max(-1)
        nn = Z[torch.arange(len(q), device=q.device), am]
        return {"la": la, "lmap": 1 - cosabs(u, z), "lq": 1 - cosabs(q, z), "gap": H.gate_gap(Z, la, q, p),
                "cmax": cmax, "lnn": 1 - cosabs(nn, z)}

    # ---- 1. decomposed baselines ------------------------------------------------------------
    cal, e = parts(tr.sample(a.cal_batch)), ev.sample(a.batch)
    grid = torch.linspace(0, 1, 201, device=dev)
    th = float(grid[torch.stack([torch.where(cal["cmax"] > g, cal["lnn"], cal["lq"]).mean() for g in grid]).argmin()])
    r, ab = parts(e), e["absent"]
    print("\n1. deployed vs baselines (eval), split by present / absent")
    row("map (ungated)", r["lmap"], ab)
    row(f"gate (saved tau={tau})", torch.where(r["gap"] > thr, r["lmap"], r["lq"]), ab)
    row("cue-only", r["lq"], ab)
    row("nearest-neighbour (forced commit)", r["lnn"], ab)
    row(f"kNN + threshold (cos > {th:.3f}, fit on train)", torch.where(r["cmax"] > th, r["lnn"], r["lq"]), ab)

    # ---- 2. prior quality -------------------------------------------------------------------
    c = tr.sample(a.cal_batch)
    la, tgt, B = H.log_doses(p, c["s"], c["t"], c["t_q"]), c["tgt"], len(c["tgt"])
    lo = torch.log(c["s"]) - lam * (c["t_q"][:, None] - c["t"])
    logp = torch.cat([math.log(1 - pa) + lo.log_softmax(-1), torch.full((B, 1), math.log(pa), device=dev)], -1)
    ce_or = float(-logp.gather(1, tgt[:, None]).mean())
    kg = torch.linspace(-14, 0, 57, device=dev)
    ces = torch.stack([torch.nn.functional.cross_entropy(
        torch.cat([la, torch.full((B, 1), float(k), device=dev)], -1), tgt) for k in kg])
    print(f"\n2. cue-free prior CE: oracle {ce_or:.4f} | learned log-doses (kappa re-fit to {float(kg[ces.argmin()]):.2f}) "
          f"{float(ces.min()):.4f} | gap {float(ces.min()) - ce_or:+.4f} nats")

    # ---- 3. eviction under a budget ---------------------------------------------------------
    big = Sampler(Zs[:cut], Qs[:cut], doc[:cut], T=a.T_big, same_passage=False, query_decay=lam, p_absent=pa)
    g = big.sample(a.evict_batch)
    lag, pres, d = H.log_doses(p, g["s"], g["t"], g["t_q"]), g["tgt"] < a.T_big, Zs.shape[1]
    lq = 1 - cosabs(g["q"], g["z_t"])
    scores = {"learned": lag, "recency": g["t"], "random": torch.rand_like(lag),
              "oracle": torch.log(g["s"]) - lam * (g["t_q"][:, None] - g["t"])}
    print(f"\n3. eviction: {a.T_big} items written, keep C (matrix memory = d^2 = {d * d} floats = {d} atoms)")
    for C in a.caps:
        print(f"   C={C} ({C * d} floats)")
        for name, sc in scores.items():
            keep = sc.topk(C, -1).indices
            ret = (keep == g["tgt"][:, None]).any(-1)[pres].float().mean()
            Zk, lak = g["Z"].gather(1, keep[..., None].expand(-1, -1, d)), lag.gather(1, keep)
            u, gap = H.read(Zk, lak, g["q"], p, N), H.gate_gap(Zk, lak, g["q"], p)
            lg = torch.where(gap > thr, 1 - cosabs(u, g["z_t"]), lq)
            print(f"      {name:8s} target kept {ret:.3f} | gate loss present {lg[pres].mean():.4f} "
                  f"absent {lg[~pres].mean():.4f}")


if __name__ == "__main__":
    main()