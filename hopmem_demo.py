#!/usr/bin/env python
"""hopmem_demo.py -- the applied artifact: sentences in, sentences out.

    python hopmem_demo.py                                   # built-in sample, queries + ideas
    python hopmem_demo.py --text doc.txt --C 8 --ideas 6
    python hopmem_demo.py --text doc.txt --extract 5        # plain extractive summary
    cat doc.txt | python hopmem_demo.py --stdin --C 6 --importance novelty
    python hopmem_demo.py --ckpt hopmem_theta.pt --text doc.txt \
        --queries "what is forgotten" "who won the match"

What it shows that the loss tables cannot: WHICH sentences survived a bounded memory,
which merged into a single idea, and which were destroyed.  Every original sentence ends
up either inside exactly one surviving atom's provenance block or in the destroyed set,
so the printout is a readable partition of the document at capacity C.

Importance:
  uniform   equal dose per sentence (default; makes near-duplicate merges visible)
  novelty   surprise = 1 - max cosine to earlier sentences (the learned-law channel;
            paraphrases get low dose and tend to be evicted rather than merged)
  length    log word count, a crude salience proxy
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
Merging two nearby directions preserves most of their evidence.
Evicting an item destroys its evidence completely.
Compression should never manufacture confidence that was not there.
The ledger records how many nats of evidence were removed.
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


def split_sentences(text, min_words=4):
    text = re.sub(r"\s+", " ", text)
    out = [s.strip() for s in H._SENT.split(text)]
    return [s for s in out if len(s.split()) >= min_words]


def encode(sents, model):
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(model)
    Z = enc.encode(sents, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
    return torch.from_numpy(Z).float()


def importance(kind, Zall, i, sents):
    if kind == "uniform":
        return 1.0
    if kind == "length":
        return math.log(1.0 + len(sents[i].split()))
    if i == 0:
        return 1.0
    prev = Zall[:i]
    return float((1.0 - (prev @ Zall[i]).max()).clamp(0.0, 2.0)) + 1e-3


def representative(block, mu, Zall):
    """The sentence most central to the merged direction."""
    ids = sorted(block)
    sims = Zall[ids] @ mu
    return ids[int(sims.argmax())]


def show_block(mem, k, sents, Zall, indent="   ", also=True):
    at = mem.atoms[k]
    blk = sorted(at["prov"])
    rep = representative(blk, at["z"], Zall)
    dose = math.exp(at["la"])
    ret = math.exp(-at["Lam"])
    print(f'{indent}dose {dose:.4f}  retained {ret:.3f}  covers {len(blk)} sentence(s)')
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
    best = int(gaps.argmax())
    gap = float(gaps[best])
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
    src.add_argument("--text", help="path to a plain-text document")
    src.add_argument("--stdin", action="store_true")
    ap.add_argument("--ckpt", default=None, help="use trained law/readout/tau from a checkpoint")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--C", type=int, default=8)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--importance", choices=["uniform", "novelty", "length"], default="uniform")
    ap.add_argument("--queries", nargs="*", default=None)
    ap.add_argument("--ideas", type=int, default=0, help="annotated digest of surviving ideas")
    ap.add_argument("--extract", type=int, default=0, help="plain extractive summary")
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--min_words", type=int, default=4)
    ap.add_argument("--verbose", action="store_true", help="per-write event log")
    ap.add_argument("--policy", choices=list(H.POLICIES), default=H.DEFAULT_POLICY)
    # measured law/readout from the v2.3.2 confirmation run (overridden by --ckpt)
    ap.add_argument("--omega", type=float, default=0.01047)
    ap.add_argument("--Gamma", type=float, default=0.1811)
    ap.add_argument("--eps_bar", type=float, default=0.00068)
    ap.add_argument("--beta", type=float, default=20.70)
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
        p = Params = H.Params(omega=a.omega, Gamma=a.Gamma, eps_bar=a.eps_bar,
                              beta=a.beta, kappa=a.kappa, dtype=dt).to(dev)
        tau, N = a.tau, 20

    mem = H.HopMem(p, Zall.shape[1], C=a.C, tau=tau, dtype=dt, device=dev, policy=a.policy)
    print(f"{len(sents)} sentences -> capacity C={a.C}, policy={a.policy}, "
          f"importance={a.importance}, beta={float(p.beta):.2f}, tau={tau:.4g}")
    print("-" * 78)
    for i, s in enumerate(sents):
        s_i = importance(a.importance, Zall, i, sents)
        f0, e0 = mem.n_fuse, mem.n_evict
        mem.write(Zall[i], s_i, float(i) * a.dt, prov=i)
        if a.verbose:
            ev = []
            if mem.n_fuse > f0:
                ev.append("MERGE")
            if mem.n_evict > e0:
                ev.append("EVICT")
            if mem.n_dead:
                pass
            tag = ("  " + "+".join(ev)) if ev else ""
            print(f"  [{i:3d}] s={s_i:.3f} atoms={len(mem.atoms)} S={mem.S:.4f}{tag}")

    t_q = float(len(sents)) * a.dt + a.dt
    mem._advance(t_q)
    covered = sum(len(at["prov"]) for at in mem.atoms)
    print("-" * 78)
    print(f"state: {len(mem.atoms)} atoms covering {covered} sentences; "
          f"{len(mem.destroyed)} destroyed; S={mem.S:.4f}")
    print(f"compression: {mem.n_fuse} merge / {mem.n_evict} evict / {mem.n_dead} gate-dead; "
          f"ledger {mem.sum_delta:.4f} nats; agree-debit {mem.agree_debit}/{max(mem.n_fuse,1)}")

    queries = a.queries if a.queries is not None else (DEFAULT_QUERIES if not a.text and not a.stdin else [])
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