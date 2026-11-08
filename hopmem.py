"""hopmem.py - a fixed-size memory that is ONE d x d matrix. No learning yet: just the math and a toy test.

    write:   M <- rho * M + s * z z^T          z = unit vector of an event, s = surprise (how strongly to write)
    read:    u <- normalize(q + beta * M u)    repeated a few steps, starting from u = q (q = the cue)
             beta = b / (largest eigenvalue of M), so the repeated read can never run away
    energy:  E(u) = 1/2 |u - q|^2 - beta/2 u^T M u      (the read walks downhill on it)

    rho  in (0,1): recency. An event written a steps ago has weight rho**a.
    s    >= 0    : surprise. Twice the surprise = twice the weight.
    The memory has d*d numbers however many events were written.

Run the toy (needs only torch):
    python hopmem.py                      # capacity, recency and surprise tables
    python hopmem.py --sparse topk        # try a sparse read (keeps the largest 25% of coordinates)

How to read the toy: a stored vector z_j is asked for with a NOISY cue q (cosine to z_j = --cue).
  overlap = u . z_j     how close the read is to the stored vector (cue alone = --cue)
  acc     = does u point at z_j more than at any other stored vector
The memory helps if overlap(read) > overlap(cue). acc can NOT beat the cue much: a linear memory
cannot rank items better than the cue already does, except through the weights (recency, surprise).
"""
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

unit = lambda x: F.normalize(x, dim=-1)


class HopfieldMemory(nn.Module):
    def __init__(self, rho=0.95, b=0.9, steps=5, sparse="none", keep=0.25):
        super().__init__()
        self.rho, self.b, self.steps, self.sparse, self.keep = rho, b, steps, sparse, keep

    def empty(self, B, d, device="cpu"):
        return torch.zeros(B, d, d, device=device)

    def write(self, M, z, s=None):
        """M (B,d,d), z (B,d) events, s (B,) surprise -> new M."""
        z = unit(z)
        s = torch.ones(z.shape[0], device=z.device) if s is None else s
        return self.rho * M + s[:, None, None] * z[:, :, None] * z[:, None, :]

    def sparsify(self, x):
        if self.sparse == "none":
            return x
        k = max(1, int(self.keep * x.shape[-1]))                    # keep the k largest coordinates
        thr = x.abs().topk(k, dim=-1).values[..., -1:]
        return x * (x.abs() >= thr)

    def read(self, M, q):
        """M (B,d,d), cues q (B,K,d) -> cleaned-up vectors u (B,K,d)."""
        q = unit(q)
        beta = self.b / torch.linalg.eigvalsh(M)[:, -1].clamp_min(1e-6)[:, None, None]
        u = q
        for _ in range(self.steps):
            u = unit(self.sparsify(q + beta * (u @ M)))              # M is symmetric, so u @ M = (M u)^T
        return u


# ----------------------------------------------------------------------------------------------
# toy test on random vectors
# ----------------------------------------------------------------------------------------------
def build(mem, B, n, d, s_hi, dev):
    """B memories of n random events, written one at a time (oldest first). 20% of events are 'surprising'."""
    z = unit(torch.randn(B, n, d, device=dev))
    s = 1.0 + (s_hi - 1.0) * (torch.rand(B, n, device=dev) < 0.2).float()
    M = mem.empty(B, d, dev)
    for i in range(n):
        M = mem.write(M, z[:, i], s[:, i])
    return z, s, M


def ask(mem, z, s, M, cue, K, use_memory=True):
    """Ask for K random stored events per memory with a noisy cue. Returns overlap, accuracy, age, surprising?"""
    B, n, d = z.shape
    j = torch.randint(0, n, (B, K), device=z.device)
    zj = z.gather(1, j[..., None].expand(-1, -1, d))
    q = unit(cue * zj + (1 - cue ** 2) ** 0.5 * unit(torch.randn(B, K, d, device=z.device)))
    u = mem.read(M, q) if use_memory else q
    overlap = (u * zj).sum(-1)
    acc = ((u @ z.transpose(1, 2)).argmax(-1) == j).float()
    return overlap, acc, n - 1 - j, s.gather(1, j) > 1


def by_age(x, age):
    edges = [(0, 0), (1, 3), (4, 15), (16, 10 ** 9)]
    return " ".join(f"{x[(age >= lo) & (age <= hi)].mean().item():.2f}" if ((age >= lo) & (age <= hi)).any() else "  - "
                    for lo, hi in edges)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=256)
    p.add_argument("--cue", type=float, default=0.2, help="cosine between the noisy cue and the stored vector")
    p.add_argument("--rho", type=float, default=0.95)
    p.add_argument("--b", type=float, default=0.9, help="read strength, must be below 1")
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--sparse", default="none", choices=["none", "topk"])
    p.add_argument("--keep", type=float, default=0.25)
    p.add_argument("--surprise", type=float, default=3.0, help="weight of a surprising event (others get 1)")
    p.add_argument("--batch", type=int, default=100)
    p.add_argument("--queries", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mem = HopfieldMemory(a.rho, a.b, a.steps, a.sparse, a.keep).to(dev)
    print(f"d={a.d} cue cosine={a.cue} rho={a.rho} b={a.b} steps={a.steps} sparse={a.sparse}")

    print("\nCAPACITY (no decay, no surprise): overlap with the stored vector")
    print("events | cue only | after read | gain | acc cue only | acc after read")
    m0 = HopfieldMemory(1.0, a.b, a.steps, a.sparse, a.keep).to(dev)
    for n in (8, 16, 32, 64, 128):
        z, s, M = build(m0, a.batch, n, a.d, 1.0, dev)
        o0, a0, _, _ = ask(m0, z, s, M, a.cue, a.queries, use_memory=False)
        o1, a1, _, _ = ask(m0, z, s, M, a.cue, a.queries)
        print(f"{n:6d} | {o0.mean():8.2f} | {o1.mean():10.2f} | {o1.mean() - o0.mean():+.2f} | {a0.mean():12.2f} | {a1.mean():14.2f}")

    print("\nRECENCY (32 events): overlap by age of the asked event (cue only = %.2f)" % a.cue)
    print("rho   | newest 1-3  4-15  16+")
    for rho in (1.0, 0.95, 0.9, 0.8):
        mr = HopfieldMemory(rho, a.b, a.steps, a.sparse, a.keep).to(dev)
        z, s, M = build(mr, a.batch, 32, a.d, 1.0, dev)
        o, _, age, _ = ask(mr, z, s, M, a.cue, a.queries)
        print(f"{rho:5.2f} | {by_age(o, age)}")

    print(f"\nSURPRISE (32 events, rho=1): 20% of events written {a.surprise}x stronger")
    z, s, M = build(m0, a.batch, 32, a.d, a.surprise, dev)
    o, ac, _, hi = ask(m0, z, s, M, a.cue, a.queries)
    print(f"ordinary   : overlap {o[~hi].mean():.2f}  acc {ac[~hi].mean():.2f}")
    print(f"surprising : overlap {o[hi].mean():.2f}  acc {ac[hi].mean():.2f}")

    print(f"\nSURPRISE + RECENCY (32 events, rho={a.rho}): overlap by age")
    z, s, M = build(mem, a.batch, 32, a.d, a.surprise, dev)
    o, _, age, hi = ask(mem, z, s, M, a.cue, a.queries)
    print("age        | newest 1-3  4-15  16+")
    print(f"ordinary   | {by_age(o[~hi], age[~hi])}")
    print(f"surprising | {by_age(o[hi], age[hi])}")


if __name__ == "__main__":
    main()
