#!/usr/bin/env python
"""hopmem_stats.py v2.3.1 -- paired multi-seed capacity statistics (spec 17.6).

    python hopmem_stats.py --ckpt hopmem_theta.pt --same_passage \
        --seeds 5 --T_big 100 500 1000 --caps 4 8 16 --batch 256 --dup --ref topk-learned
    python hopmem_stats.py --ckpt hopmem_theta.pt --same_passage \
        --seeds 5 --dup --ref debit          # decisive resfirst-vs-debit test

Every policy scores the SAME episode batch within a (T_big, C, seed) cell, so all
differences are PAIRED per episode.  --ref is honored in BOTH the capacity and the
duplicate loop.  Differences and SEs print at 6 decimals so sub-1e-4 effects are
readable.  Pooling across seeds is valid: each seed draws a fresh independent batch.
"""
import argparse
import math

import torch

import hopmem as H


def stats(diff):
    n = diff.numel()
    m = float(diff.mean())
    se = float(diff.std(unbiased=True) / math.sqrt(n)) if n > 1 else float("nan")
    z = m / se if (se == se and se > 0) else float("nan")
    return m, se, z


def fmt(x, w, prec):
    return f"{x:>{w}.{prec}f}" if x == x else " " * (w - 3) + "ref"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="hopmem_theta.pt")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--same_passage", action="store_true")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--T_big", type=int, nargs="+", default=[100, 500, 1000])
    ap.add_argument("--caps", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--policies", nargs="+", default=["debit", "resfirst", "ward", "lightest"])
    ap.add_argument("--ref", default="topk-learned")
    ap.add_argument("--dup", action="store_true")
    ap.add_argument("--dup_T", type=int, default=100)
    ap.add_argument("--dup_C", type=int, default=4)
    ap.add_argument("--dup_only", action="store_true")
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
    td = tau if tau is not None else 1.0
    Zs, Qs, doc = H.synthetic_corpus(a) if a.synthetic else H.build_corpus(a)
    Zs, Qs = Zs.to(dev), Qs.to(dev)
    P = len(Zs)
    b = (doc[1:] != doc[:-1]).nonzero().squeeze(1) + 1
    b = b[b >= int(0.9 * P)]
    cut = int(b[0]) if len(b) else int(0.9 * P)
    Ze, Qe = Zs[cut:], Qs[cut:]
    names = list(a.policies) + ["topk-learned", "topk-oracle"]
    ref = a.ref if a.ref in names else "topk-learned"
    print(f"items {P}, d={Zs.shape[1]}, device {dev}; ckpt v{cfg.get('version','?')} "
          f"tau={tau} beta={float(p.beta):.2f}; seeds={a.seeds} batch={a.batch} REF={ref}")

    if not a.dup_only:
        for T_big in a.T_big:
            big = H.EpisodeSampler(Ze, Qe, doc[cut:], T=T_big, same_passage=False,
                                   query_decay=lam, p_absent=pa)
            for C in a.caps:
                if C >= T_big:
                    continue
                acc = {nm: {"p": [], "a": [], "seed": []} for nm in names}
                for sd in range(a.seeds):
                    torch.manual_seed(a.seed0 + sd)
                    ep = big.sample(a.batch)
                    la = H.log_doses(p, ep["s"], ep["t"], ep["t_q"])
                    sc = {"topk-learned": la,
                          "topk-oracle": torch.log(ep["s"]) - lam * (ep["t_q"][:, None] - ep["t"])}
                    pres = ep["tgt"] < T_big
                    for nm in names:
                        if nm in sc:
                            loss, _ = H.topk_losses(p, ep, C, sc[nm], N=N, tau=tau)
                        else:
                            loss, _ = H.gated_losses(p, ep, N=N, tau=tau, C=C, tau_dead=td, policy=nm)
                        acc[nm]["p"].append(loss[pres])
                        acc[nm]["a"].append(loss[~pres])
                        acc[nm]["seed"].append(float(loss[pres].mean()))
                cat = {nm: (torch.cat(v["p"]), torch.cat(v["a"])) for nm, v in acc.items()}
                npr = cat[names[0]][0].numel()
                print(f"\nT_big={T_big}  C={C}  paired present episodes n={npr} "
                      f"({a.seeds} seeds x {a.batch})  ref={ref}")
                print(f"   {'policy':>16} {'present':>9} {'absent':>9} {'dPres':>11} {'SE':>10} "
                      f"{'d/SE':>8} {'seed-sd':>8}")
                refp = cat[ref][0]
                for nm in names:
                    pp, aa = cat[nm]
                    m, se, z = stats(pp - refp)
                    ssd = float(torch.tensor(acc[nm]["seed"]).std(unbiased=True)) if a.seeds > 1 else 0.0
                    print(f"   {nm:>16} {float(pp.mean()):>9.4f} {float(aa.mean()):>9.4f} "
                          f"{m:>+11.6f} {se:>10.6f} {fmt(z, 8, 2)} {ssd:>8.4f}")

    if a.dup:
        b_ = float(p.beta)
        th_star = 2 * math.acos(max(-1.0, 1 - math.log(2) / b_)) if b_ > math.log(2) / 2 else math.pi
        C, T_big = a.dup_C, a.dup_T
        pols = [nm for nm in a.policies if nm != "lightest"]
        for extra in ("topk-learned", ref):
            if extra not in pols:
                pols.append(extra)
        dref = ref if ref in pols else "topk-learned"
        print(f"\nduplicate-heavy paired sweep: theta*={th_star:.3f}, copies at theta*/4, "
              f"C={C} T={T_big} seeds={a.seeds}  REF={dref}")
        print(f"   {'rho_dup':>8} {'policy':>16} {'present':>9} {'d vs ref':>11} {'SE':>10} {'d/SE':>8}")
        for rd in (0.0, 0.25, 0.5, 0.75):
            acc = {nm: [] for nm in pols}
            for sd in range(a.seeds):
                torch.manual_seed(a.seed0 + sd)
                ep = H.dup_episodes(Ze, Qe, a.batch, T_big, rd, th_star / 4,
                                    lam=lam, p_absent=pa, device=dev)
                pres = ep["tgt"] < T_big
                la = H.log_doses(p, ep["s"], ep["t"], ep["t_q"])
                for nm in pols:
                    if nm == "topk-learned":
                        loss, _ = H.topk_losses(p, ep, C, la, N=N, tau=tau)
                    elif nm == "topk-oracle":
                        sc = torch.log(ep["s"]) - lam * (ep["t_q"][:, None] - ep["t"])
                        loss, _ = H.topk_losses(p, ep, C, sc, N=N, tau=tau)
                    else:
                        loss, _ = H.gated_losses(p, ep, N=N, tau=tau, C=C, tau_dead=td, policy=nm)
                    acc[nm].append(loss[pres])
            cat = {nm: torch.cat(v) for nm, v in acc.items()}
            refp = cat[dref]
            for nm in pols:
                m, se, z = stats(cat[nm] - refp)
                print(f"   {rd:>8.2f} {nm:>16} {float(cat[nm].mean()):>9.4f} "
                      f"{m:>+11.6f} {se:>10.6f} {fmt(z, 8, 2)}")


if __name__ == "__main__":
    main()