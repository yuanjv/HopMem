#!/usr/bin/env python
"""hopmem_demo.py -- the applied artifact: sentences in, sentences out.

    python hopmem_demo.py --verbose --ideas 6
    python hopmem_demo.py --beta 6 --verbose        # wider merge window
    python hopmem_demo.py --text doc.txt --C 8 --ideas 6
    python hopmem_demo.py --text doc.txt --extract 5
    python hopmem_demo.py --ckpt hopmem_theta.pt --text doc.txt

TWO REGIME TRAPS, both documented by the geometry line this script prints.

1. DOCUMENT POSITION IS NOT TIME.  The learned Gamma=0.1811 was fit on Exp(1)
   inter-arrivals, where recency IS the right prior.  Applied per sentence it makes the
   memory a recency buffer.  Default --dt 0.0 removes time decay across a document.

2. THE MERGE WINDOW IS SET BY beta.  The gate needs cos >= 1 - 1/beta; the debit branch
   needs cos >= cos(2*arccos(1 - ln2/beta)).  At the learned beta=20.7 those are 0.952
   and 0.868 -- but MiniLM paraphrases sit near 0.84, so NOTHING merges and eviction
   takes the oldest.  A sharper readout merges less, because it distinguishes more.
   Lower --beta widens the window; near-verbatim repeats merge at any beta.

The sample deliberately contains a verbatim repeat (free merge, delta=0), a near-verbatim
edit (cheap merge), and a paraphrase pair (outside resolution -> eviction), so all three
behaviours are visible.  Merged blocks carry roughly double dose and are therefore
protected from later eviction -- which is the mechanism's payoff.
"""
import argparse
import math
import re
import sys

import torch
import torch.nn.functional as F

import hopmem as H

SAMPLE = """
Associative memory stores directions rather than tables of records.
A bounded memory must decide what to forget.
Recency decays the importance of old items over time.
Surprise raises the importance of an unexpected item.
A memory with fixed capacity has to decide what to forget.
Retrieval can be written as the gradient of a log-sum-exp free energy.
The null option lets the system abstain when nothing matches.
Abstention is decided before the retrieval direction is refined.
A bounded memory must decide what to forget.
Merging two nearby directions preserves most of their evidence.
Evicting an item destroys its evidence completely.
The ledger records how many nats of evidence were removed.
The ledger records how many nats of evidence were removed by compression.
Compression should never manufacture confidence that was not there.
Attention over a fixed set of memory tokens is the same computation.
A language model can supply the surprise signal from its own prediction error.
Fixed capacity means the cost per token does not grow with the stream.
The ledger records how much evidence compression removed.
"""

DEFAULT_QUERIES = [
    "how does the memory decide what to forget",
    "can the system refuse to answer",
    "what does the ledger measure",
    "what is the capital of France",
]

_ENC = {}


def encoder(model):
    if model not in _ENC:
        from sentence_transformers import SentenceTransformer
        _ENC[model] = SentenceTransformer(model)
    return _ENC[model]


def encode(sents, model):
    Z = encoder(model).encode(sents, batch_size=64, normalize_embeddings=True,
                              convert_to_numpy=True)
    return torch.from_numpy(Z).float()


def split_sentences(text, min_words=4):
    text = re.sub(r"\s+", " ", text)
    return [s.strip() for s in H._SENT.split(text)
            if len(s.strip().split()) >= min_words]


def importance(kind, Zall, i, sents):
    if kind == "uniform":
        return 1.0
    if kind == "length":
        return math.log(1.0 + len(sents[i].split()))
    if i == 0:
        return 1.0
    return float((1.0 - (Zall[:i] @ Zall[i]).max()).clamp(0.0, 2.0)) + 1e-3


def geometry_report(Zall, beta, sents, max_n=1024):
    """Print the merge windows and the document's actual pair similarities."""
    g_gate = 1.0 - 1.0 / beta
    th_gate = math.acos(max(-1.0, min(1.0, g_gate)))
    th_star = 2.0 * math.acos(max(-1.0, min(1.0, 1.0 - math.log(2.0) / beta)))
    g_star = math.cos(th_star)
    print(f"geometry: beta={beta:.2f} -> gate needs cos>={g_gate:.3f} (theta<={th_gate:.3f}); "
          f"debit-merge needs cos>={g_star:.3f} (theta<={th_star:.3f})")
    Z = Zall[:max_n]
    G = Z @ Z.t()
    n = G.shape[0]
    G.fill_diagonal_(-2.0)
    flat = G.argmax()
    i, j = int(flat) // n, int(flat) % n
    cmax = float(G[i, j])
    off = G[G > -1.5]
    med = float(off.median()) if off.numel() else float("nan")
    verdict = ("gate can fire" if cmax >= g_gate else
               ("debit-merge can fire" if cmax >= g_star else "NO pair is mergeable"))
    print(f"document: max pair cos {cmax:.3f} (sentences {i},{j}), median {med:.3f} -> {verdict}")
    if cmax < g_star:
        print(f"          paraphrases at this similarity need beta <~ "
              f"{math.log(2.0) / (1.0 - math.cos(0.5 * math.acos(max(-1.0, cmax)))):.1f} to merge")


def representative(block, mu, Zall):
    ids = sorted(block)
    return ids[int((Zall[ids] @ mu).argmax())]


def show_block(mem, k, sents, Zall, indent="   ", also=True):
    at = mem.atoms[k]
    blk = sorted(at["prov"])
    rep = representative(blk, at["z"], Zall)
    print(f'{indent}dose {math.exp(at["la"]):.4f}  retained {math.exp(-at["Lam"]):.3f}  '
          f'covers {len(blk)} sentence(s)')
    print(f'{indent}"{sents[rep]}"')
    if also and len(blk) > 1:
        for j in blk:
            if j != rep:
                print(f'{indent}  + [{j}] "{sents[j]}"')


@torch.no_grad()
def ask(mem, sents, Zall, q_text, Zq, p, tau, N, t_q, topk=2):
    mem._advance(t_q)
    print(f'\nQ: "{q_text}"')
    if not mem.atoms:
        print("   memory empty -> ABSTAIN, no sentence emitted")
        return
    Z, la = mem.Zs[None], mem.log_alpha[None]
    q = F.normalize(Zq.to(dtype=mem.dtype, device=mem.device), dim=-1)[None]
    lg = H.gibbs_logits(Z, la, q, p)
    K = Z.shape[1]
    gaps = lg[0, :K] - lg[0, K]
    gap = float(gaps.max())
    thr = math.log(tau)
    commit = gap > thr
    print(f'   gate {"COMMIT" if commit else "ABSTAIN"}   gap {gap:+.3f}   ln tau {thr:+.3f}')
    if not commit:
        print("   -> cue returned unchanged; no sentence emitted")
        return
    u = H.read(Z, la, q, p, N)[0]
    sims = mem.Zs @ u
    order = sims.argsort(descending=True)
    for rank in range(min(topk, len(order))):
        k = int(order[rank])
        print(f'   answer #{rank + 1}  atom {k}  cos(u) {float(sims[k]):.3f}')
        show_block(mem, k, sents, Zall, indent="      ", also=(rank == 0))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--text")
    src.add_argument("--stdin", action="store_true")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--C", type=int, default=8)
    ap.add_argument("--dt", type=float, default=0.0,
                    help="time per sentence; 0 = document mode, 1 = episodic/recency mode")
    ap.add_argument("--importance", choices=["uniform", "novelty", "length"], default="uniform")
    ap.add_argument("--queries", nargs="*", default=None)
    ap.add_argument("--ideas", type=int, default=0)
    ap.add_argument("--extract", type=int, default=0)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--min_words", type=int, default=4)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--policy", choices=list(H.POLICIES), default=H.DEFAULT_POLICY)
    ap.add_argument("--omega", type=float, default=0.01047)
    ap.add_argument("--Gamma", type=float, default=0.1811)
    ap.add_argument("--eps_bar", type=float, default=0.00068)
    ap.add_argument("--beta", type=float, default=20.70,
                    help="readout sharpness; ALSO sets the merge window (lower = merges more)")
    ap.add_argument("--kappa", type=float, default=-17.41)
    ap.add_argument("--tau", type=float, default=0.6703)
    a = ap.parse_args()

    text = open(a.text, encoding="utf-8").read() if a.text else (
        sys.stdin.read() if a.stdin else SAMPLE)
    sents = split_sentences(text, a.min_words)
    if len(sents) < 2:
        print("need at least two sentences"); return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float64
    Zall = encode(sents, a.model).to(dtype=dt, device=dev)

    if a.ckpt:
        p, tau, cfg = H.load_checkpoint(a.ckpt, dev)
        p = p.double()
        tau = tau if tau is not None else a.tau
        N = cfg.get("N", 20)
    else:
        p = H.Params(omega=a.omega, Gamma=a.Gamma, eps_bar=a.eps_bar,
                     beta=a.beta, kappa=a.kappa, dtype=dt).to(dev)
        tau, N = a.tau, 20

    mem = H.HopMem(p, Zall.shape[1], C=a.C, tau=tau, dtype=dt, device=dev, policy=a.policy)
    mode = "document (no time decay)" if a.dt == 0 else f"episodic (dt={a.dt})"
    print(f"{len(sents)} sentences -> C={a.C}, policy={a.policy}, importance={a.importance}, "
          f"{mode}, beta={float(p.beta):.2f}, tau={tau:.4g}")
    geometry_report(Zall, float(p.beta), sents)
    print("-" * 78)
    for i in range(len(sents)):
        s_i = importance(a.importance, Zall, i, sents)
        f0, e0 = mem.n_fuse, mem.n_evict
        mem.write(Zall[i], s_i, float(i) * a.dt, prov=i)
        if a.verbose:
            ev = (["MERGE"] if mem.n_fuse > f0 else []) + (["EVICT"] if mem.n_evict > e0 else [])
            print(f"  [{i:3d}] s={s_i:.3f} atoms={len(mem.atoms)} S={mem.S:.4f}"
                  + ("  " + "+".join(ev) if ev else ""))

    t_q = float(len(sents)) * a.dt + a.dt
    mem._advance(t_q)
    covered = sum(len(at["prov"]) for at in mem.atoms)
    agree = f"{mem.agree_debit}/{mem.n_fuse}" if mem.n_fuse else "n/a (no merges)"
    print("-" * 78)
    print(f"state: {len(mem.atoms)} atoms covering {covered} sentences; "
          f"{len(mem.destroyed)} destroyed; S={mem.S:.4f}")
    print(f"compression: {mem.n_fuse} merge / {mem.n_evict} evict / {mem.n_dead} gate-dead; "
          f"ledger {mem.sum_delta:.4f} nats; agree-debit {agree}")

    queries = a.queries if a.queries is not None else (
        DEFAULT_QUERIES if not a.text and not a.stdin else [])
    if queries:
        Zq = encode(queries, a.model).to(dtype=dt, device=dev)
        for j, qt in enumerate(queries):
            ask(mem, sents, Zall, qt, Zq[j], p, tau, N, t_q, a.topk)

    n_ideas = a.ideas or (6 if not (a.ideas or a.extract or queries) else 0)
    if n_ideas:
        order = sorted(range(len(mem.atoms)), key=lambda k: -mem.atoms[k]["la"])
        print("\n" + "=" * 78)
        print(f"WHAT THE BOUNDED MEMORY KEPT  (top {min(n_ideas, len(order))} ideas by dose)")
        print("=" * 78)
        for rank, k in enumerate(order[:n_ideas]):
            print(f"\nidea {rank + 1}  [atom {k}]")
            show_block(mem, k, sents, Zall)
        if mem.destroyed:
            print("\n" + "-" * 78)
            print(f"WHAT IT FORGOT  ({len(mem.destroyed)} sentences destroyed by eviction)")
            print("-" * 78)
            for j in sorted(mem.destroyed):
                print(f'  [{j}] "{sents[j]}"')

    if a.extract:
        order = sorted(range(len(mem.atoms)), key=lambda k: -mem.atoms[k]["la"])
        reps = [representative(mem.atoms[k]["prov"], mem.atoms[k]["z"], Zall)
                for k in order[:a.extract]]
        print("\n--- extractive summary (memory-ranked) ---")
        for j in sorted(reps):
            print(sents[j])


if __name__ == "__main__":
    main()