#!/usr/bin/env python
"""hopmem_stats.py v2.3.2 -- paired multi-seed statistics (spec 17.1 / 17.6).

    # compression at budget C, now with a train-fit kNN+threshold row
    python hopmem_stats.py --ckpt hopmem_theta.pt --same_passage --seeds 5 \
        --T_big 100 500 1000 --caps 4 8 16 --batch 256

    # duplicate-heavy sweep, canonical vs debit
    python hopmem_stats.py --ckpt hopmem_theta.pt --same_passage --seeds 5 \
        --dup --dup_only --ref debit

    # READOUT claim (spec 19.5 item 1): uncompressed gate vs train-fit kNN+threshold
    python hopmem_stats.py --ckpt hopmem_theta.pt --same_passage --seeds 5 \
        --readout --readout_only

The kNN threshold is fit ONCE on a TRAIN calibration batch per (T, C) cell, never on
eval -- exactly the like-for-like comparator spec 17.1 requires.  Every policy scores
the SAME episodes within a cell, so all differences are PAIRED per episode.  --ref is
honored in every section.  Pooling across seeds is valid: each seed draws a fresh batch.
"""
import argparse
import math

import torch

import hopmem as H


def cosabs(x, y):
    return (x * y).sum(-1).abs().clamp(max=1.0)


def stats(diff):
    n = diff.numel()
    m = float(diff.mean())
    se = float(diff.std(unbiased=True) / math.sqrt(n)) if n > 1 else float("nan")
    z = m / se if (se == se and se > 0) else float("nan")
    return m, se, z


def fmt(x, w, prec):
    return f"{x:>{w}.{prec}f}" if x == x else " " * (w - 3) + "ref"


def knn_parts(ep, C=None, score=None):
    """Per-episode kNN quantities on the (optionally top-k-pruned) store.
    Returns (max cosine, kNN angular loss, cue-only loss)."""
    Z, q, z = ep["Z"], ep["q"], ep["z_t"]
    if C is not None and C < Z.shape[1]:
        keep = score.topk(C, -1).indices
        Z = Z.gather(1, keep[..., None].expand(-1, -1, Z.shape[-1]))
    cc = torch.einsum("btd,bd->bt", Z, q)
    cmax, am = cc.max(-1)
    nn = Z[torch.arange(len(q), device=q.device), am]
    return cmax, 1 - cosabs(nn, z), 1 - cosabs(q, z)


@torch.no_grad()
def fit_knn_threshold(p, sampler, cal_batch, dev, C=None, score_kind="learned", lam=0.2):
    """Train-fit one threshold on a TRAIN batch (spec 17.1).  Never touches eval."""
    ep = sampler.sample(cal_batch)
    sc = H.log_doses(p, ep["s"], ep["t"], ep["t_q"]) if score_kind == "learned" else None
    cmax, lnn, lq = knn_parts(ep, C, sc)
    grid = torch.linspace(0, 1, 201, device=dev)
    losses = torch.stack([torch.where(cmax > g, lnn, lq).mean() for g in grid])
    return float(grid[losses.argmin()])


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="hopmem_theta.pt")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--same_passage", action="store_true")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--T_big", type=int, nargs="+", default=[100, 500, 1000])
    ap.add_argument("--caps", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--cal_batch", type=int, default=2048)
    ap.add_argument("--policies", nargs="+", default=["resfirst", "debit", "ward", "lightest"])
    ap.add_argument("--ref", default="topk-learned")
    ap.add_argument("--dup", action="store_true")
    ap.add_argument("--dup_T", type=int, default=100)
    ap.add_argument("--dup_C", type=int, default=4)
    ap.add_argument("--dup_only", action="store_true")
    ap.add_argument("--readout", action="store_true")
    ap.add_argument("--readout_only", action="store_true")
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
    Zt, Qt, dt_ = Zs[:cut], Qs[:cut], doc[:cut]
    Ze, Qe, de_ = Zs[cut:], Qs[cut:], doc[cut:]
    print(f"items {P}, d={Zs.shape[1]}, device {dev}; ckpt v{cfg.get('version','?')} "
          f"tau={tau} beta={float(p.beta):.2f}; seeds={a.seeds} batch={a.batch}")

    # ---------------- section 1: capacity at budget C ----------------
    if not (a.dup_only or a.readout_only):
        names = list(a.policies) + ["topk-learned", "topk-oracle", "knn-threshold"]
        ref = a.ref if a.ref in names else "topk-learned"
        for T_big in a.T_big:
            bigE = H.EpisodeSampler(Ze, Qe, de_, T=T_big, same_passage=False,
                                    query_decay=lam, p_absent=pa)
            bigT = H.EpisodeSampler(Zt, Qt, dt_, T=T_big, same_passage=False,
                                    query_decay=lam, p_absent=pa)
            for C in a.caps:
                if C >= T_big:
                    continue
                torch.manual_seed(a.seed0 - 1)
                th = fit_knn_threshold(p, bigT, a.cal_batch, dev, C=C, lam=lam)
                acc = {nm: {"p": [], "a": [], "seed": []} for nm in names}
                for sd in range(a.seeds):
                    torch.manual_seed(a.seed0 + sd)
                    ep = bigE.sample(a.batch)
                    la = H.log_doses(p, ep["s"], ep["t"], ep["t_q"])
                    sc = {"topk-learned": la, "knn-threshold": la,
                          "topk-oracle": torch.log(ep["s"]) - lam * (ep["t_q"][:, None] - ep["t"])}
                    pres = ep["tgt"] < T_big
                    for nm in names:
                        if nm == "knn-threshold":
                            cmax, lnn, lq = knn_parts(ep, C, sc[nm])
                            loss = torch.where(cmax > th, lnn, lq)
                        elif nm in sc:
                            loss, _ = H.topk_losses(p, ep, C, sc[nm], N=N, tau=tau)
                        else:
                            loss, _ = H.gated_losses(p, ep, N=N, tau=tau, C=C, tau_dead=td, policy=nm)
                        acc[nm]["p"].append(loss[pres])
                        acc[nm]["a"].append(loss[~pres])
                        acc[nm]["seed"].append(float(loss[pres].mean()))
                cat = {nm: (torch.cat(v["p"]), torch.cat(v["a"])) for nm, v in acc.items()}
                npr = cat[names[0]][0].numel()
                print(f"\nT_big={T_big}  C={C}  paired present n={npr}  ref={ref}  "
                      f"kNN threshold (train-fit) = {th:.3f}")
                print(f"   {'policy':>16} {'present':>9} {'absent':>9} {'dPres':>11} {'SE':>10} "
                      f"{'d/SE':>8} {'seed-sd':>8}")
                refp = cat[ref][0]
                for nm in names:
                    pp, aa = cat[nm]
                    m, se, z = stats(pp - refp)
                    ssd = float(torch.tensor(acc[nm]["seed"]).std(unbiased=True)) if a.seeds > 1 else 0.0
                    print(f"   {nm:>16} {float(pp.mean()):>9.4f} {float(aa.mean()):>9.4f} "
                          f"{m:>+11.6f} {se:>10.6f} {fmt(z, 8, 2)} {ssd:>8.4f}")

    # ---------------- section 2: READOUT claim, uncompressed ----------------
    if a.readout:
        T = a.T
        trR = H.EpisodeSampler(Zt, Qt, dt_, T=T, same_passage=a.same_passage,
                               query_decay=lam, p_absent=pa)
        evR = H.EpisodeSampler(Ze, Qe, de_, T=T, same_passage=a.same_passage,
                               query_decay=lam, p_absent=pa)
        torch.manual_seed(a.seed0 - 1)
        th = fit_knn_threshold(p, trR, a.cal_batch, dev, C=None)
        names = ["gate", "knn-threshold", "knn-forced", "map", "cue-only"]
        acc = {nm: {"p": [], "a": [], "all": []} for nm in names}
        for sd in range(a.seeds):
            torch.manual_seed(a.seed0 + sd)
            ep = evR.sample(a.batch)
            pres = ep["tgt"] < T
            la = H.log_doses(p, ep["s"], ep["t"], ep["t_q"])
            u = H.read(ep["Z"], la, ep["q"], p, N)
            cmax, lnn, lq = knn_parts(ep)
            losses = {
                "gate": H.gated_losses(p, ep, N=N, tau=tau, C=None)[0],
                "knn-threshold": torch.where(cmax > th, lnn, lq),
                "knn-forced": lnn,
                "map": 1 - cosabs(u, ep["z_t"]),
                "cue-only": lq,
            }
            for nm in names:
                acc[nm]["p"].append(losses[nm][pres])
                acc[nm]["a"].append(losses[nm][~pres])
                acc[nm]["all"].append(losses[nm])
        cat = {nm: {k: torch.cat(v) for k, v in d.items()} for nm, d in acc.items()}
        npr = cat["gate"]["p"].numel()
        print(f"\nREADOUT (uncompressed, C>=T={T})  paired present n={npr}  "
              f"kNN threshold (train-fit) = {th:.3f}  saved tau = {tau}")
        print(f"   {'policy':>16} {'all':>9} {'present':>9} {'absent':>9} "
              f"{'dPres':>11} {'SE':>10} {'d/SE':>8} {'dAll':>11}")
        rp, ra = cat["knn-threshold"]["p"], cat["knn-threshold"]["all"]
        for nm in names:
            pp, aa, al = cat[nm]["p"], cat[nm]["a"], cat[nm]["all"]
            m, se, z = stats(pp - rp)
            ma, _, _ = stats(al - ra)
            print(f"   {nm:>16} {float(al.mean()):>9.4f} {float(pp.mean()):>9.4f} "
                  f"{float(aa.mean()):>9.4f} {m:>+11.6f} {se:>10.6f} {fmt(z, 8, 2)} {ma:>+11.6f}")
        print("   (spec 17.1: the beating claim is the PRESENT column, d/SE vs kNN-threshold)")

    # ---------------- section 3: duplicate-heavy sweep ----------------
    if a.dup:
        b_ = float(p.beta)
        th_star = 2 * math.acos(max(-1.0, 1 - math.log(2) / b_)) if b_ > math.log(2) / 2 else math.pi
        C, T_big = a.dup_C, a.dup_T
        pols = [nm for nm in a.policies if nm != "lightest"]
        for extra in ("topk-learned", a.ref):
            if extra not in pols:
                pols.append(extra)
        dref = a.ref if a.ref in pols else "topk-learned"
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