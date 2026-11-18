#!/usr/bin/env python
"""hopmem_diag.py v2.2.1 -- add-on for hopmem.py (imports it, changes nothing in it).

    python hopmem_diag.py --ckpt hopmem_theta.pt                 # real cached corpus
    python hopmem_diag.py --ckpt hopmem_theta.pt --synthetic     # no network
    python hopmem_diag.py --ckpt hopmem_theta.pt --same_passage

Sections:
 1. DECOMPOSED BASELINES incl. kNN + one train-fit threshold (like-for-like gate comparator).
 2. PRIOR QUALITY: cue-free CE of learned log-doses vs the generating oracle (kappa re-fit).
 3. TOP-k EVICTION (the v2.1.1 capacity map, retained as the broken comparator).
 4. FUSION CAPACITY REGIME (spec 17.6): stream T_big writes through the fusion loop, sweep C,
    report decomposed present/absent loss, per-target retained evidence e^{-beta D}, block
    angular distortion, ledger sum(delta), gate-dead deletions, below-resolution fraction,
    ECE under load, the monotone mass-gate ablation, and a per-C tau recalibration diagnostic
    (train batch only -- never deployed).
 5. UNIT THEOREM spot-checks: evidence domination (Thm A), commutation (Thm B), stable delta.
"""
import argparse
import math
import torch
import hopmem as H


def cosabs(x, y):
    return (x * y).sum(-1).abs().clamp(max=1.0)


def row(name, loss, absent):
    print(f"   {name:40s} all {loss.mean():.4f}   present {loss[~absent].mean():.4f}   "
          f"absent {loss[absent].mean():.4f}")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="hopmem_theta.pt")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--same_passage", action="store_true")
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--T_big", type=int, default=100)
    ap.add_argument("--caps", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--cal_batch", type=int, default=2048)
    ap.add_argument("--evict_batch", type=int, default=512)
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
    tr = H.EpisodeSampler(Zs[:cut], Qs[:cut], doc[:cut], T=a.T, **kw)
    ev = H.EpisodeSampler(Zs[cut:], Qs[cut:], doc[cut:], T=a.T, **kw)
    print(f"items {P}, d={Zs.shape[1]}, train {cut}/eval {P-cut}; ckpt v{cfg.get('version','?')} "
          f"tau={tau} C={cfg.get('C')} beta={float(p.beta):.2f} kappa={float(p.kappa):.2f} "
          f"omega={float(p.omega):.4f} Gamma={float(p.Gamma):.4f}")

    def parts(ep):
        la = H.log_doses(p, ep["s"], ep["t"], ep["t_q"])
        Z, q, z = ep["Z"], ep["q"], ep["z_t"]
        u = H.read(Z, la, q, p, N)
        cc = torch.einsum("btd,bd->bt", Z, q)
        cmax, am = cc.max(-1)
        nn = Z[torch.arange(len(q), device=q.device), am]
        return {"lmap": 1 - cosabs(u, z), "lq": 1 - cosabs(q, z), "gap": H.gate_gap(Z, la, q, p),
                "cmax": cmax, "lnn": 1 - cosabs(nn, z)}

    cal, e = parts(tr.sample(a.cal_batch)), ev.sample(a.batch)
    grid = torch.linspace(0, 1, 201, device=dev)
    th = float(grid[torch.stack([torch.where(cal["cmax"] > g, cal["lnn"], cal["lq"]).mean()
                                 for g in grid]).argmin()])
    r, ab = parts(e), e["absent"]
    print("\n1. deployed vs baselines (eval, C>=T), split present / absent")
    row("map (ungated)", r["lmap"], ab)
    row(f"gate (saved tau={tau})", torch.where(r["gap"] > thr, r["lmap"], r["lq"]), ab)
    row("cue-only", r["lq"], ab)
    row("nearest-neighbour (forced commit)", r["lnn"], ab)
    row(f"kNN + threshold (cos > {th:.3f}, train-fit)", torch.where(r["cmax"] > th, r["lnn"], r["lq"]), ab)

    c = tr.sample(a.cal_batch)
    la, tgt, B = H.log_doses(p, c["s"], c["t"], c["t_q"]), c["tgt"], len(c["tgt"])
    lo = torch.log(c["s"]) - lam * (c["t_q"][:, None] - c["t"])
    logp = torch.cat([math.log(1 - pa) + lo.log_softmax(-1),
                      torch.full((B, 1), math.log(pa), device=dev)], -1)
    ce_or = float(-logp.gather(1, tgt[:, None]).mean())
    kg = torch.linspace(-14, 0, 57, device=dev)
    ces = torch.stack([torch.nn.functional.cross_entropy(
        torch.cat([la, torch.full((B, 1), float(k), device=dev)], -1), tgt) for k in kg])
    print(f"\n2. cue-free prior CE: oracle {ce_or:.4f} | learned (kappa re-fit "
          f"{float(kg[ces.argmin()]):.2f}) {float(ces.min()):.4f} | gap {float(ces.min())-ce_or:+.4f} nats")

    big_tr = H.EpisodeSampler(Zs[:cut], Qs[:cut], doc[:cut], T=a.T_big, same_passage=False,
                              query_decay=lam, p_absent=pa)
    big_ev = H.EpisodeSampler(Zs[cut:], Qs[cut:], doc[cut:], T=a.T_big, same_passage=False,
                              query_decay=lam, p_absent=pa)
    g = big_ev.sample(a.evict_batch)
    lag, pres, d = H.log_doses(p, g["s"], g["t"], g["t_q"]), g["tgt"] < a.T_big, Zs.shape[1]
    lq = 1 - cosabs(g["q"], g["z_t"])
    scores = {"learned": lag, "recency": g["t"], "random": torch.rand_like(lag),
              "oracle": torch.log(g["s"]) - lam * (g["t_q"][:, None] - g["t"])}
    print(f"\n3/4. capacity: T_big={a.T_big} writes, budget C (matrix memory d^2={d*d} floats)")
    print(f"   {'C':>4} {'policy':>14} {'present':>8} {'absent':>8} {'kept/ret':>8} {'ledger':>8} {'ECE':>6}")
    for C in a.caps:
        rf = H.evaluate(p, g, N, tau=tau, C=C, tau_dead=(tau if tau is not None else 1.0))
        print(f"   {C:>4} {'FUSION':>14} {rf.get('gate_pres',float('nan')):>8.4f} "
              f"{rf.get('gate_abs',float('nan')):>8.4f} {rf.get('retained',float('nan')):>8.3f} "
              f"{rf.get('ledger',float('nan')):>8.3f} {rf.get('ece',float('nan')):>6.3f}  "
              f"(surv {1-rf.get('destroyed_frac',0):.3f}, below-res {rf.get('below_frac',0):.0%}, "
              f"mass-gate {rf.get('mass_gate_best',float('nan')):.4f})")
        for name, sc in scores.items():
            keep = sc.topk(C, -1).indices
            ret = (keep == g["tgt"][:, None]).any(-1)[pres].float().mean()
            Zk = g["Z"].gather(1, keep[..., None].expand(-1, -1, d))
            lak = lag.gather(1, keep)
            u, gap = H.read(Zk, lak, g["q"], p, N), H.gate_gap(Zk, lak, g["q"], p)
            lg = torch.where(gap > thr, 1 - cosabs(u, g["z_t"]), lq)
            print(f"   {C:>4} {'topk-'+name:>14} {lg[pres].mean():>8.4f} {lg[~pres].mean():>8.4f} "
                  f"{ret:>8.3f}")
        gt = big_tr.sample(a.cal_batch)
        tau_C = H.calibrate_tau(p, gt, N, C=C, tau_dead=(tau if tau is not None else 1.0))
        rf2 = H.evaluate(p, g, N, tau=tau_C, C=C, tau_dead=(tau if tau is not None else 1.0))
        print(f"        per-C train tau={tau_C:.3g} -> present {rf2.get('gate_pres',float('nan')):.4f} "
              f"absent {rf2.get('gate_abs',float('nan')):.4f} (diagnostic; deployed tau={tau})")

    print("\n5. unit theorem spot-checks (float64)")
    dt = torch.float64
    nq, dq, bq = 20000, 64, float(p.beta)
    zi = torch.nn.functional.normalize(torch.randn(nq, dq, dtype=dt), dim=-1)
    zj = torch.nn.functional.normalize(torch.randn(nq, dq, dtype=dt), dim=-1)
    u = torch.nn.functional.normalize(torch.randn(nq, dq, dtype=dt), dim=-1)
    lai = torch.rand(nq, dtype=dt) * 5 - 6
    laj = torch.rand(nq, dtype=dt) * 5 - 6
    mu, laf, r, delta, w, Gij, omr = H.fuse_pair_math(zi, zj, lai, laj, torch.tensor(bq, dtype=dt))
    Ep = (laf + bq * (mu * u).sum(-1)).exp()
    Eij = (lai + bq * (zi * u).sum(-1)).exp() + (laj + bq * (zj * u).sum(-1)).exp()
    print(f"   Thm A domination: violations {int(((Ep-Eij)>1e-9).sum())}/{nq}, max resid {float((Ep-Eij).max()):.2e}")
    cc = -1.37
    m1, l1, *_ = H.fuse_pair_math(zi[:5], zj[:5], lai[:5], laj[:5], torch.tensor(bq, dtype=dt))
    m2, l2, *_ = H.fuse_pair_math(zi[:5], zj[:5], lai[:5] + cc, laj[:5] + cc, torch.tensor(bq, dtype=dt))
    print(f"   Thm B commutation: mu err {float((m1-m2).abs().max()):.1e}, ell err {float((l2-(l1+cc)).abs().max()):.1e}")
    print(f"   stable delta == beta(1-r): err {float((delta-bq*omr).abs().max()):.1e}")


if __name__ == "__main__":
    main()