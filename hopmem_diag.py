#!/usr/bin/env python
"""hopmem_diag.py v2.3.1 -- add-on for hopmem.py.

 1 decomposed baselines (incl. kNN + train-fit threshold)
 2 cue-free prior CE vs oracle
 3/4 capacity sweep: debit / resfirst / ward / streaming-lightest vs offline top-k
 5 duplicate-heavy sweep (spec 17.6 prediction ii), now with resfirst
 6 unit theorem spot-checks
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
    ap.add_argument("--dup_sweep", action="store_true")
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
    N, lam, pa = cfg.get("N", 20), cfg.get("query_decay", 0.2), cfg.get("p_absent", 0.2)
    thr = math.log(tau if tau is not None else 1.0)
    pol = cfg.get("fuse_policy", "debit")
    Zs, Qs, doc = H.synthetic_corpus(a) if a.synthetic else H.build_corpus(a)
    Zs, Qs = Zs.to(dev), Qs.to(dev)
    P = len(Zs)
    b = (doc[1:] != doc[:-1]).nonzero().squeeze(1) + 1
    b = b[b >= int(0.9 * P)]
    cut = int(b[0]) if len(b) else int(0.9 * P)
    kw = dict(same_passage=a.same_passage, query_decay=lam, p_absent=pa)
    tr = H.EpisodeSampler(Zs[:cut], Qs[:cut], doc[:cut], T=a.T, **kw)
    ev = H.EpisodeSampler(Zs[cut:], Qs[cut:], doc[cut:], T=a.T, **kw)
    print(f"items {P}, d={Zs.shape[1]}; ckpt v{cfg.get('version','?')} tau={tau} C={cfg.get('C')} "
          f"policy={pol} beta={float(p.beta):.2f} kappa={float(p.kappa):.2f}")

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
    print("\n1. deployed vs baselines (eval, C>=T), present / absent")
    row("map (ungated)", r["lmap"], ab)
    row(f"gate (saved tau={tau})", torch.where(r["gap"] > thr, r["lmap"], r["lq"]), ab)
    row("cue-only", r["lq"], ab)
    row("kNN (forced commit)", r["lnn"], ab)
    row(f"kNN + threshold (cos > {th:.3f})", torch.where(r["cmax"] > th, r["lnn"], r["lq"]), ab)

    c = tr.sample(a.cal_batch)
    la, tgt, B = H.log_doses(p, c["s"], c["t"], c["t_q"]), c["tgt"], len(c["tgt"])
    lo = torch.log(c["s"]) - lam * (c["t_q"][:, None] - c["t"])
    logp = torch.cat([math.log(1 - pa) + lo.log_softmax(-1),
                      torch.full((B, 1), math.log(pa), device=dev)], -1)
    ce_or = float(-logp.gather(1, tgt[:, None]).mean())
    kg = torch.linspace(-14, 0, 57, device=dev)
    ces = torch.stack([torch.nn.functional.cross_entropy(
        torch.cat([la, torch.full((B, 1), float(k), device=dev)], -1), tgt) for k in kg])
    print(f"\n2. prior CE: oracle {ce_or:.4f} | learned {float(ces.min()):.4f} | "
          f"gap {float(ces.min())-ce_or:+.4f} nats")

    big_ev = H.EpisodeSampler(Zs[cut:], Qs[cut:], doc[cut:], T=a.T_big, same_passage=False,
                              query_decay=lam, p_absent=pa)
    g = big_ev.sample(a.evict_batch)
    lag, pres, d = H.log_doses(p, g["s"], g["t"], g["t_q"]), g["tgt"] < a.T_big, Zs.shape[1]
    lq = 1 - cosabs(g["q"], g["z_t"])
    scores = {"learned": lag, "recency": g["t"], "random": torch.rand_like(lag),
              "oracle": torch.log(g["s"]) - lam * (g["t_q"][:, None] - g["t"])}
    print(f"\n3/4. capacity: T_big={a.T_big}, budget C")
    print(f"   {'C':>4} {'policy':>16} {'present':>8} {'absent':>8} {'ret/kept':>8} "
          f"{'ledger':>8} {'merge':>6} {'evict':>6}")
    for C in a.caps:
        for name, pp in (("DEBIT", "debit"), ("resfirst", "resfirst"),
                         ("ward(v2.2.1)", "ward"), ("stream-lightest", "lightest")):
            rf = H.evaluate(p, g, N, tau=tau, C=C, tau_dead=(tau if tau is not None else 1.0), policy=pp)
            print(f"   {C:>4} {name:>16} {rf.get('gate_pres',float('nan')):>8.4f} "
                  f"{rf.get('gate_abs',float('nan')):>8.4f} {rf.get('retained',float('nan')):>8.3f} "
                  f"{rf.get('ledger',float('nan')):>8.3f} {rf.get('fuse_n',float('nan')):>6.1f} "
                  f"{rf.get('evict_n',float('nan')):>6.1f}")
        for name, sc in scores.items():
            keep = sc.topk(C, -1).indices
            ret = (keep == g["tgt"][:, None]).any(-1)[pres].float().mean()
            Zk = g["Z"].gather(1, keep[..., None].expand(-1, -1, d))
            lak = lag.gather(1, keep)
            u, gap = H.read(Zk, lak, g["q"], p, N), H.gate_gap(Zk, lak, g["q"], p)
            lg = torch.where(gap > thr, 1 - cosabs(u, g["z_t"]), lq)
            print(f"   {C:>4} {'topk-'+name:>16} {lg[pres].mean():>8.4f} {lg[~pres].mean():>8.4f} "
                  f"{ret:>8.3f}")

    if a.dup_sweep:
        b_ = float(p.beta)
        th_star = 2 * math.acos(max(-1.0, 1 - math.log(2) / b_)) if b_ > math.log(2) / 2 else math.pi
        C = max(4, min(a.caps) if a.caps else 8)
        print(f"\n5. duplicate-heavy sweep (theta*={th_star:.3f}, copies at theta*/4), C={C}")
        print(f"   {'rho_dup':>8} {'debit':>9} {'resfirst':>9} {'ward':>9} {'topk':>9} "
              f"{'merges(debit)':>14} {'merges(resf)':>13}")
        for rd in (0.0, 0.25, 0.5, 0.75):
            epd = H.dup_episodes(Zs[cut:], Qs[cut:], 256, a.T_big, rd, th_star / 4,
                                 lam=lam, p_absent=pa, device=dev)
            prd = epd["tgt"] < a.T_big
            rr = {}
            for pp in ("debit", "resfirst", "ward"):
                rr[pp] = H.evaluate(p, epd, N, tau=tau, C=C,
                                    tau_dead=(tau or 1.0), policy=pp)
            lad = H.log_doses(p, epd["s"], epd["t"], epd["t_q"])
            keep = lad.topk(C, -1).indices
            Zk = epd["Z"].gather(1, keep[..., None].expand(-1, -1, d))
            lak = lad.gather(1, keep)
            u, gap = H.read(Zk, lak, epd["q"], p, N), H.gate_gap(Zk, lak, epd["q"], p)
            lg = torch.where(gap > thr, 1 - cosabs(u, epd["z_t"]), 1 - cosabs(epd["q"], epd["z_t"]))
            print(f"   {rd:>8.2f} {rr['debit'].get('gate_pres',float('nan')):>9.4f} "
                  f"{rr['resfirst'].get('gate_pres',float('nan')):>9.4f} "
                  f"{rr['ward'].get('gate_pres',float('nan')):>9.4f} {lg[prd].mean():>9.4f} "
                  f"{rr['debit'].get('fuse_n',float('nan')):>14.1f} "
                  f"{rr['resfirst'].get('fuse_n',float('nan')):>13.1f}")

    print("\n6. unit theorem spot-checks (float64)")
    dt = torch.float64
    nq, dq, bq = 20000, 64, float(p.beta)
    zi = torch.nn.functional.normalize(torch.randn(nq, dq, dtype=dt), dim=-1)
    zj = torch.nn.functional.normalize(torch.randn(nq, dq, dtype=dt), dim=-1)
    u = torch.nn.functional.normalize(torch.randn(nq, dq, dtype=dt), dim=-1)
    lai = torch.rand(nq, dtype=dt) * 5 - 6
    laj = torch.rand(nq, dtype=dt) * 5 - 6
    bt = torch.tensor(bq, dtype=dt)
    mu, laf, r, delta, w, Gij, omr, rdir = H.fuse_pair_math(zi, zj, lai, laj, bt)
    Ep = (laf + bq * (mu * u).sum(-1)).exp()
    Eij = (lai + bq * (zi * u).sum(-1)).exp() + (laj + bq * (zj * u).sum(-1)).exp()
    print(f"   Thm A: violations {int(((Ep-Eij)>1e-9).sum())}/{nq}")
    m1, l1, *_ = H.fuse_pair_math(zi[:5], zj[:5], lai[:5], laj[:5], bt)
    m2, l2, *_ = H.fuse_pair_math(zi[:5], zj[:5], lai[:5] - 1.37, laj[:5] - 1.37, bt)
    print(f"   Thm B: mu err {float((m1-m2).abs().max()):.1e}, ell err {float((l2-(l1-1.37)).abs().max()):.1e}")
    print(f"   stable delta: err {float((delta-bq*omr).abs().max()):.1e}")


if __name__ == "__main__":
    main()