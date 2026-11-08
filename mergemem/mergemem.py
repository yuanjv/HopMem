"""mergemem.py - additive sentence memory in ONE file, Hugging Face `datasets` end to end.

THE IDEA
    write:  state = sum_i  w_i * fade_i * bind(code_i, key_i)     ONE vector holds n sentences
    read :  r     = unbind(state, key(cue))                       sentence i comes back, the others are noise
    code_i = meaning_i + wording_i     SUPERPOSED in the same d numbers (no separate slices)
    fade_i = lam ** age_i              age = how many sentences were stored after sentence i. lam in (0,1) differs
                                       PER COORDINATE: a fixed bank of timescales from 2 to 200 sentences
                                       (--learn_fade lets the network move them; when it could, it made all slow).
                                       Nothing says where wording goes: if the network puts it in fast coordinates,
                                       old wording fades first while the meaning survives.
    read-out: ONE flow from noise to the joint target (embedding, characters): x_t = (1-t) x0 + t x1, the network
        denoises, i.e. predicts x1 from x_t. That is the whole loss: no weights between meaning and wording, no
        budget term. A sentence counts in proportion to its information content (mean IDF, average 1).

    bind(x, k)   = rotate each (even, odd) pair of x by an angle   (complex number times e^{i*angle})
    unbind       = rotate back by the same angle, so unbind(bind(x, k), k) == x exactly
    key(cue)     = (embedding of the cue) @ R, with R a fixed random matrix
                   -> similar meaning gives similar angles: key similarity ~ exp(-sigma^2 * |e - e'|^2 / 2)
    w_i          = learned importance, a softmax share of the batch's energy (mean w = 1 by construction,
                   no penalty term), so making one sentence louder means making others quieter.
    noise        = the other n-1 sentences come back rotated by random angles; a coordinate that fades old
                   sentences has less noise, a slow one keeps everything (noise ~ sqrt(n-1)).

DATA (all through `datasets`, nothing else is written to disk)
    books --map--> sentences --> importance level (mean IDF, 3 equal bins) --map--> embeddings
    --> save_to_disk. A second run just loads the cache folder.
    Held-out = whole books never seen in training (the validation split, or the last 5% of sampled books).

    pip install torch datasets sentence-transformers
    python mergemem.py
"""
import argparse
import math
import os
import random
import re
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

LEN, CUE, NI = 64, 16, 3     # chars per sentence, cue chars shown to the wording head, importance levels
CHARS = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'.,-0123456789\"?!;:()"
V = len(CHARS)               # index 0 (space) doubles as padding
C2I = {c: i for i, c in enumerate(CHARS)}
OK_CHARS = set(CHARS)
STARTS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ\"'("
FIX = {"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"', "\u2014": "-", "\u2013": "-",
       "\u2026": "...", "_": "", "\u00a0": " "}
SPLIT = re.compile(r"(?<=[.!?])(?<!Mr\.)(?<!Mrs\.)(?<!Ms\.)(?<!Dr\.)(?<!St\.)\s+(?=[A-Z\"'(])")


# ----------------------------------------------------------------------------------------------
# text helpers
# ----------------------------------------------------------------------------------------------
def to_ids(s):
    ids = [C2I.get(c, 0) for c in s[:LEN]]
    return ids + [0] * (LEN - len(ids))


def lengths(ids):
    """True sentence length: last non-space char + 1."""
    return ids.shape[-1] - (ids != 0).flip(-1).float().argmax(-1)


def halves(t):
    w = t.split()
    k = max(1, len(w) // 2)
    return " ".join(w[:k]), " ".join(w[k:] or w)


def gather(x, idx):
    """x (B,n,D), idx (B,Q) -> (B,Q,D)."""
    return x.gather(1, idx[..., None].expand(-1, -1, x.shape[-1]))


def clean_book(text):
    m = re.search(r"\*\*\* ?START OF.*?\*\*\*", text, re.S)       # Project Gutenberg header / footer
    if m:
        text = text[m.end():]
    m = re.search(r"\*\*\* ?END OF", text)
    if m:
        text = text[:m.start()]
    for k, v in FIX.items():
        text = text.replace(k, v)
    return text.replace("\r", "")


def split_sentences(text, min_len):
    out = []
    for para in re.split(r"\n\s*\n", text):
        para = re.sub(r"\s+", " ", para).strip()
        for s in SPLIT.split(para):
            s = s.strip()
            if s and min_len <= len(s) <= LEN and s[0] in STARTS and s[-1] in ".!?\"'" and set(s) <= OK_CHARS:
                out.append(s)
    return out


def idf_scores(train_texts, other_texts):
    """Information content of a sentence = mean IDF of its words (mean surprisal in nats). A PROXY, not human
    judgment. Document frequencies come from the training sentences only, so held-out text gets no peek."""
    words = lambda s: set(re.findall(r"[a-z']+", s.lower()))
    N = len(train_texts)
    df = Counter(w for s in train_texts for w in words(s))

    def score(s):
        ws = words(s)
        return sum(math.log(N / df.get(w, 1)) for w in ws) / max(len(ws), 1)

    return [score(s) for s in train_texts], [score(s) for s in other_texts]


def importance_levels(train_texts, other_texts):
    """Only for the L0/L1/L2 columns of the report: the score cut into 3 equal bins."""
    tr, ot = idf_scores(train_texts, other_texts)
    srt = sorted(tr)
    t1, t2 = srt[len(tr) // 3], srt[2 * len(tr) // 3]
    level = lambda x: 0 if x < t1 else (1 if x < t2 else 2)
    return [level(x) for x in tr], [level(x) for x in ot]


# ----------------------------------------------------------------------------------------------
# data: Hugging Face datasets, books -> sentences -> embeddings, cached with save_to_disk
# ----------------------------------------------------------------------------------------------
def build_dataset(a):
    from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

    local = os.path.isdir(a.dataset)                            # offline: a folder made with save_to_disk
    src = os.path.basename(os.path.normpath(a.dataset)) + ("" if local else f"_{a.n_books}books")
    path = os.path.join(a.cache, f"{src}_{a.n_train}_{a.n_val}_{os.path.basename(os.path.normpath(a.model))}")
    if os.path.isdir(path):
        print(f"loading cached dataset from {path}")
        return load_from_disk(path)

    if local:
        raw = load_from_disk(a.dataset)
    else:                                                       # ~10 GB on the Hub: stream a random sample of whole books
        stream = load_dataset(a.dataset, split="train", streaming=True).shuffle(seed=0, buffer_size=100)
        raw = {"train": Dataset.from_list([{"text": ex[a.text_col]} for ex in stream.take(a.n_books)])}
    if "validation" in raw:
        books = {"train": raw["train"], "val": raw["validation"]}
    else:                                                       # no validation split: hold out last 5% of books
        n = len(raw["train"])
        k = int(0.95 * n)
        books = {"train": raw["train"].select(range(k)), "val": raw["train"].select(range(k, n))}

    def to_sentences(batch):                                    # many sentences out per book in
        return {"text": [s for t in batch["text"] for s in split_sentences(clean_book(t), a.min_len)]}

    seen, texts = set(), {}
    for part, keep in (("train", a.n_train), ("val", a.n_val)):
        sents = books[part].map(to_sentences, batched=True, batch_size=4,
                                remove_columns=books[part].column_names)
        uniq = []
        for t in sents["text"]:                                 # drop duplicates, also across train/val
            if t.lower() not in seen:
                seen.add(t.lower())
                uniq.append(t)
        random.Random(0).shuffle(uniq)
        texts[part] = uniq[:keep]
        print(f"{part}: {len(uniq)} unique sentences, keeping {len(texts[part])}")
    assert texts["train"] and texts["val"], "no sentences survived the filters"
    if len(texts["train"]) < 20000:
        print("note: fewer than ~20k training sentences; the model may memorize them. Raise --n_books.")

    lv_train, lv_val = importance_levels(texts["train"], texts["val"])

    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(a.model, device="cuda" if torch.cuda.is_available() else "cpu")
    enc = lambda xs: st.encode(xs, batch_size=256, normalize_embeddings=True)

    def embed(batch):                                           # full sentence, first half, second half
        pre, suf = zip(*(halves(t) for t in batch["text"]))
        return {"emb_full": enc(batch["text"]), "emb_pre": enc(list(pre)), "emb_suf": enc(list(suf))}

    out = DatasetDict()
    for part, lv in (("train", lv_train), ("val", lv_val)):
        ds = Dataset.from_dict({"text": texts[part], "level": lv})
        out[part] = ds.map(embed, batched=True, batch_size=1024, new_fingerprint=f"emb_{part}")
    out.save_to_disk(path)
    return out


def as_tensor(col):
    return col if torch.is_tensor(col) else torch.stack(list(col))


class Split:
    """One split as tensors on the device: char ids, importance level, three frozen embeddings."""

    def __init__(self, ds, dev, name, weight):
        self.ids = torch.tensor([to_ids(t) for t in ds["text"]], device=dev)
        self.lvl = torch.tensor(list(ds["level"]), device=dev)
        self.wt = torch.tensor(weight, device=dev, dtype=torch.float32)       # information content, mean 1
        self.cue = self.ids[:, :CUE]                             # first CUE chars
        t = ds.with_format("torch")
        self.full, self.pre, self.suf = (as_tensor(t[k]).to(dev) for k in ("emb_full", "emb_pre", "emb_suf"))
        print(f"{name}: {len(self.ids)} sentences, embedding dim {self.full.shape[1]}")

    def sample(self, B, n, hard=0.0):
        """hard = fraction of the B memories filled with the n nearest neighbours (by meaning) of one anchor
        sentence, so the cue alone cannot tell the stored sentences apart; the rest are random."""
        j = torch.randint(0, len(self.ids), (B, n), device=self.ids.device)      # with replacement
        h = int(B * hard)
        if h > 0 and n > 1:
            anchor = j[:h, 0]
            j[:h] = (self.full[anchor] @ self.full.T).topk(n, dim=1).indices      # the anchor itself comes first
        return j, self.ids[j], self.lvl[j], self.cue[j], self.full[j], self.pre[j], self.suf[j]


# ----------------------------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------------------------
def rotate(x, theta):
    """Rotate each consecutive (even, odd) pair of x by its own angle. rotate(rotate(x, a), -a) == x."""
    xr, xi = x[..., 0::2], x[..., 1::2]
    c, s = torch.cos(theta), torch.sin(theta)
    return torch.stack([xr * c - xi * s, xr * s + xi * c], -1).flatten(-2)


def dilated_stack(h, dils):
    return nn.ModuleList([nn.Conv1d(h, h, 3, padding=r, dilation=r) for r in dils])


class Key(nn.Module):
    """Frozen random map: embedding -> key angles. Frozen on purpose (a learned key can collapse to a constant)."""

    def __init__(self, E, d, sigma):
        super().__init__()
        self.register_buffer("R", torch.randn(E, d // 2) * sigma)

    def forward(self, e):
        return e @ self.R


class Writer(nn.Module):
    """sentence (chars + embedding) -> (code, importance w). Meaning and wording are SUPERPOSED in one code of
    d numbers; every coordinate forgets at its own rate lam (a fixed bank, or learned with learn_fade)."""

    def __init__(self, d, E, h=128, learn_fade=False):
        super().__init__()
        self.emb = nn.Embedding(V, h)
        self.pos = nn.Parameter(torch.randn(LEN, h) * 0.02)
        self.convs = dilated_stack(h, (1, 2, 4, 8))
        self.lex = nn.Linear(LEN * h, d)                         # wording: keeps WHERE each char is (no pooling)
        self.sem = nn.Sequential(nn.Linear(E, 256), nn.GELU(), nn.Linear(256, d))
        self.amp = nn.Sequential(nn.Linear(2 * h + E, 128), nn.GELU(), nn.Linear(128, 1))
        tau = torch.logspace(math.log10(2), math.log10(200), d // 2)      # start with timescales from 2 to 200 sentences
        self.a = nn.Parameter(torch.logit(1 - 1 / tau), requires_grad=learn_fade)   # lam = sigmoid(a): 0.5 .. 0.995

    def parts(self, ids, e):
        """-> meaning direction, wording direction (unit vectors in the SAME d numbers), importance share."""
        x, e = ids.reshape(-1, LEN), e.reshape(-1, e.shape[-1])
        h = (self.emb(x) + self.pos).transpose(1, 2)
        for c in self.convs:
            h = h + F.gelu(c(h))
        h = h.transpose(1, 2)
        p = torch.cat([h.mean(1), h.amax(1)], -1)                # pooled summary, only used for the importance w
        vs, vl = F.normalize(self.sem(e), dim=-1), F.normalize(self.lex(h.flatten(1)), dim=-1)
        s = self.amp(torch.cat([p, e], -1)).squeeze(-1)           # log-importance of every sentence in the batch
        return vs, vl, torch.softmax(s, 0) * s.numel()            # an energy SHARE: mean w = 1 by construction

    def forward(self, ids, e, theta):
        sh = ids.shape[:-1]
        vs, vl, w = self.parts(ids, e)
        code = rotate(vs + vl, theta.reshape(-1, theta.shape[-1]))   # superposition: one vector per sentence
        return code.view(*sh, -1), w.view(*sh)

    def fade(self, n):
        """(n, d): a sentence of age a is scaled by lam**a in every coordinate; the newest (age 0) is untouched."""
        age = torch.arange(n - 1, -1, -1, device=self.a.device, dtype=torch.float32)
        f = torch.exp(age[:, None] * F.logsigmoid(self.a)[None, :])            # lam**age, (n, d/2)
        return f.repeat_interleave(2, dim=-1)                                  # both parts of a complex pair fade alike


class Reader(nn.Module):
    """Unbind with a cue's key, then read meaning (embedding) and wording (characters) with one flow:
    conditional flow matching, x_t = (1-t) x0 + t x1 from noise x0 to the target x1. The network predicts the clean
    target x1 (the velocity is then (x1_hat - x_t) / (1 - t), so this is the same flow; its error is the velocity
    error times (1-t)^2, which keeps the irreducible noise x0 from drowning the learning signal)."""

    def __init__(self, d, E, key, h=192, e=16, hs=512, steps=16):
        super().__init__()
        self.d, self.E, self.e, self.h, self.steps, self.key = d, E, e, h, steps, key
        self.mvel = nn.Sequential(nn.Linear(E + 1 + d + 1, hs), nn.GELU(), nn.Linear(hs, hs), nn.GELU(),
                                  nn.Linear(hs, hs), nn.GELU(), nn.Linear(hs, E))
        self.register_buffer("Ec", F.normalize(torch.randn(V, e), dim=-1) * e ** 0.5)   # fixed char embeddings
        self.cue_emb = nn.Embedding(V, 16)
        self.cond = nn.Sequential(nn.Linear(d + 1 + CUE * 16, 2 * h), nn.GELU(), nn.Linear(2 * h, h))
        self.spread = nn.Linear(h, LEN * h)                  # one read-out vector -> its own features per position
        self.inp = nn.Linear(e + 1, h)
        self.pos = nn.Parameter(torch.randn(LEN, h) * 0.02)
        self.convs = dilated_stack(h, (1, 2, 4, 8, 1, 2))
        self.out = nn.Linear(h, e)

    def unbind(self, state, cue_e):
        """state (B,d), cue embeddings (B,Q,E) -> unbound unit vector r (B,Q,d) and log energy ln (B,Q,1)."""
        Q = cue_e.shape[1]
        n = state.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        r = rotate((state / n)[:, None].expand(-1, Q, -1), -self.key(cue_e))
        return r, n.log()[:, None].expand(-1, Q, -1)

    def den_m(self, x, t, r, ln):                            # meaning denoiser: x (N,E), t (N,1) -> x1_hat
        return self.mvel(torch.cat([x, t, r, ln], -1))

    def den_c(self, x, t, r, ln, cue):                       # wording denoiser: x (N,LEN,e), t (N,1) -> x1_hat
        c = self.cond(torch.cat([r, ln, self.cue_emb(cue).flatten(1)], -1))
        h = self.inp(torch.cat([x, t[:, :, None].expand(-1, LEN, 1)], -1)) + self.pos + self.spread(c).view(-1, LEN, self.h)
        h = h.transpose(1, 2)
        for conv in self.convs:
            h = h + F.gelu(conv(h))
        return self.out(h.transpose(1, 2))

    def loss(self, r, ln, cue, emb, ids, wt):
        """ONE flow-matching loss on the joint target (meaning, wording): squared error of the denoised estimate,
        averaged over every coordinate. No weights between the two parts; wt is only the sentence's information."""
        B, Q, _ = ids.shape
        f = lambda z: z.reshape(B * Q, -1)
        m1, c1 = f(emb) * self.E ** 0.5, self.Ec[f(ids)]     # targets, unit scale per coordinate
        m0, c0 = torch.randn_like(m1), torch.randn_like(c1)
        t = torch.rand(B * Q, 1, device=m1.device)
        t3 = t[:, :, None]
        err_m = (self.den_m((1 - t) * m0 + t * m1, t, f(r), f(ln)) - m1) ** 2
        err_c = (self.den_c((1 - t3) * c0 + t3 * c1, t, f(r), f(ln), f(cue)) - c1) ** 2
        err = torch.cat([err_m, err_c.flatten(1)], -1).mean(-1)
        return (err * wt.reshape(-1)).mean()

    @torch.no_grad()
    def sample(self, r, ln, cue, vote=1):
        """Euler steps along the learned flow: at step k, move 1/(steps-k) of the way to the current clean estimate
        (x <- x + v dt with v = (x1_hat - x)/(1-t)); the last step lands on it. Returns embedding and char ids.
        vote=K runs K independent trajectories and averages them before rounding: the posterior mean, i.e. the
        flow counterpart of an argmax (a single sample guesses at random where the answer is uncertain)."""
        B, Q, _ = cue.shape
        f = lambda z: z.reshape(B * Q, -1)
        r_, l_, cu = f(r), f(ln), f(cue)
        m_sum, c_sum = 0.0, 0.0
        for _ in range(vote):
            m = torch.randn(B * Q, self.E, device=r_.device)
            c = torch.randn(B * Q, LEN, self.e, device=r_.device)
            for k in range(self.steps):
                t = torch.full((B * Q, 1), k / self.steps, device=r_.device)
                frac = 1.0 / (self.steps - k)
                m, c = m + (self.den_m(m, t, r_, l_) - m) * frac, c + (self.den_c(c, t, r_, l_, cu) - c) * frac
            m_sum, c_sum = m_sum + m, c_sum + c
        pred = (F.normalize(c_sum, dim=-1) @ F.normalize(self.Ec, dim=-1).T).argmax(-1)
        return F.normalize(m_sum, dim=-1).view(B, Q, -1), pred.view(B, Q, LEN)


def write(enc, key, ids, ef, ep, flat=False):
    """The whole memory write: encode each sentence (meaning + wording superposed), weight it by importance,
    let old sentences fade with the learned per-coordinate forgetting, add everything up. Key = first-half embedding."""
    code, w = enc(ids, ef, key(ep))                                           # code (B,n,d), w (B,n)
    ww = torch.ones_like(w) if flat else w                                    # flat=True: ablate importance
    return (ww[..., None] * code * enc.fade(code.shape[1])).sum(1), w


# ----------------------------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def evaluate(enc, dec, key, data, n, other_half, flat=False, absent=False, hard=0.0, blind=False,
             vote=1, batches=6, B=128, Q=8):
    dev = data.ids.device
    ctx, pool, cacc, cnt, amp, amp_n = (torch.zeros(NI) for _ in range(6))
    ca_age, pool_age, n_age = (torch.zeros(4) for _ in range(3))
    edges = torch.tensor([1, 4, 16], device=dev)                 # age buckets: 0 | 1-3 | 4-15 | 16+
    cos_sum, n_q = 0.0, 0
    pos = torch.arange(LEN, device=dev)
    for _ in range(batches):
        idx, ids, lvl, cue, ef, ep, es = data.sample(B, n, hard)
        state, w = write(enc, key, ids, ef, ep, flat)
        for k in range(NI):                                      # mean importance of the stored sentences
            amp[k] += w[lvl == k].sum().cpu()
            amp_n[k] += (lvl == k).sum().cpu()
        if absent:                                               # leak control: ask about sentences NOT in the memory
            idx, ids, lvl, cue, ef, ep, es = data.sample(B, n, hard)
        j = torch.randint(0, n, (B, Q), device=dev)
        cue_e = gather(es if other_half else ep, j)              # which half is used as the cue
        r, ln = dec.unbind(state, cue_e)

        if blind:                                                # control: hide the memory's loudness from the reader
            ln = torch.zeros_like(ln)
        pe, pred = dec.sample(r, ln, gather(cue, j), vote)   # meaning and wording from the same flow
        ok_ctx = (pe @ ef.transpose(1, 2)).argmax(-1) == j                                # among the n stored
        ok_pool = (pe.reshape(-1, pe.shape[-1]) @ data.full.T).argmax(-1).view(B, Q) == idx.gather(1, j)
        cos_sum += (pe * gather(ef, j)).sum().item()
        n_q += B * Q

        tgt = gather(ids, j)
        m = (pos >= CUE) & (pos < lengths(tgt)[..., None])                                # chars after the cue
        ca = ((pred == tgt) & m).sum(-1).float() / m.sum(-1).clamp_min(1)
        bucket = torch.bucketize(n - 1 - j, edges, right=True)   # age of the asked sentence (0 = newest)
        for b in range(4):
            mb = bucket == b
            ca_age[b] += ca[mb].sum().cpu()
            pool_age[b] += ok_pool[mb].sum().cpu()
            n_age[b] += mb.sum().cpu()
        l = lvl.gather(1, j)
        for k in range(NI):
            mk = l == k
            ctx[k] += ok_ctx[mk].sum().cpu()
            pool[k] += ok_pool[mk].sum().cpu()
            cacc[k] += ca[mk].sum().cpu()
            cnt[k] += mk.sum().cpu()
    c = cnt.clamp_min(1)
    nan, seen = torch.full((4,), float("nan")), n_age > 0
    return (ctx / c, pool / c, cos_sum / n_q, cacc / c, amp / amp_n.clamp_min(1),
            torch.where(seen, ca_age / n_age.clamp_min(1), nan), torch.where(seen, pool_age / n_age.clamp_min(1), nan))


# ----------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="sedthh/gutenberg_english",
                   help="Hub name, or a local folder made with DatasetDict.save_to_disk (works offline)")
    p.add_argument("--text_col", default="TEXT", help="column holding the book text (Hub datasets only)")
    p.add_argument("--n_books", type=int, default=200, help="books to sample from the Hub (streamed, not fully downloaded)")
    p.add_argument("--cache", default="mergemem_cache", help="folder for the processed dataset (save_to_disk)")
    p.add_argument("--n_train", type=int, default=100000, help="training sentences to keep")
    p.add_argument("--n_val", type=int, default=5000, help="held-out sentences to keep")
    p.add_argument("--min_len", type=int, default=24, help="shortest sentence in chars (longest is LEN=64)")
    p.add_argument("--model", default="all-MiniLM-L6-v2", help="frozen sentence-embedding model")
    p.add_argument("--d", type=int, default=640, help="size of the ONE memory vector (meaning + wording superposed)")
    p.add_argument("--learn_fade", action="store_true", help="let the network move the per-coordinate timescales")
    p.add_argument("--recency", type=float, default=1.0,
                   help="training queries ask about a sentence of age a with probability proportional to recency**a "
                        "(1 = every stored sentence equally often); makes recent sentences matter more")
    p.add_argument("--init", default="",
                   help="weights file to START training from, then train as usual (--ckpt would skip training)")
    p.add_argument("--euler", type=int, default=16, help="Euler steps used to sample from the learned flow")
    p.add_argument("--vote", type=int, default=8,
                   help="evaluation only: average this many flow samples before rounding (posterior mean); 1 = one sample")
    p.add_argument("--sigma", type=float, default=2.0, help="key sharpness: larger = less paraphrase-tolerant")
    p.add_argument("--alt_p", type=float, default=0.3, help="fraction of training cues taken from the other half")
    p.add_argument("--min_sents", type=int, default=2, help="fewest sentences stored in one memory (1 = no crowding)")
    p.add_argument("--max_sents", type=int, default=32, help="most sentences stored in one memory")
    p.add_argument("--ramp", type=int, default=0,
                   help="steps over which the memory size grows from 1 to max_sents (0 = off, always 2..max_sents)")
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--queries", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hard", type=float, default=0.0,
                   help="fraction of training memories made of nearest neighbours of each other (0 = all random)")
    p.add_argument("--ckpt", default="", help="weights file: load it if it exists (skips training), else save to it after training")
    p.add_argument("--hf_token", default=None, help="Hugging Face token (or just set the HF_TOKEN env var)")
    a = p.parse_args()

    if a.hf_token:                                   # datasets, huggingface_hub and sentence-transformers all read this
        os.environ["HF_TOKEN"] = a.hf_token
    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = build_dataset(a)
    tr_score, va_score = idf_scores(list(ds["train"]["text"]), list(ds["val"]["text"]))
    mean = sum(tr_score) / len(tr_score)                      # weights average 1 over the training set
    data = Split(ds["train"], dev, "train", [x / mean for x in tr_score])
    val = Split(ds["val"], dev, "val", [x / mean for x in va_score])
    E, d = data.full.shape[1], a.d

    key = Key(E, d, a.sigma).to(dev)
    enc = Writer(d, E, learn_fade=a.learn_fade).to(dev)
    dec = Reader(d, E, key, steps=a.euler).to(dev)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    if a.init:
        sd = torch.load(a.init, map_location=dev)
        enc.load_state_dict(sd["enc"])
        dec.load_state_dict(sd["dec"])
        print(f"starting from the weights in {a.init}")
    loaded = bool(a.ckpt) and os.path.exists(a.ckpt)
    if loaded:
        sd = torch.load(a.ckpt, map_location=dev)
        enc.load_state_dict(sd["enc"])
        dec.load_state_dict(sd["dec"])
        print(f"loaded weights from {a.ckpt}, skipping training")

    for step in range(1, (0 if loaded else a.steps) + 1):
        hi = a.max_sents if a.ramp == 0 else max(1, round(1 + (a.max_sents - 1) * min(1.0, step / a.ramp)))
        n = int(torch.randint(min(a.min_sents, hi), hi + 1, ()))       # memory crowding varies per step
        idx, ids, lvl, cue, ef, ep, es = data.sample(a.batch, n, a.hard)
        state, _ = write(enc, key, ids, ef, ep)
        if a.recency < 1:                                               # ask more about recent sentences
            pr = a.recency ** torch.arange(n - 1, -1, -1, device=dev, dtype=torch.float32)   # position j has age n-1-j
            j = torch.multinomial(pr, a.batch * a.queries, replacement=True).view(a.batch, a.queries)
        else:
            j = torch.randint(0, n, (a.batch, a.queries), device=dev)
        other = torch.rand(a.batch, a.queries, 1, device=dev) < a.alt_p
        cue_e = torch.where(other, gather(es, j), gather(ep, j))        # first half or second half as the cue
        r, ln = dec.unbind(state, cue_e)
        wt = data.wt[idx.gather(1, j)]                                  # a sentence counts by its information content
        loss = dec.loss(r, ln, gather(cue, j), gather(ef, j), gather(ids, j), wt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0:
            print(f"step {step:5d}  flow loss {loss.item():.4f}")

    if a.ckpt and not loaded:
        torch.save({"enc": enc.state_dict(), "dec": dec.state_dict()}, a.ckpt)
    enc.eval()
    dec.eval()
    f3 = lambda t: " ".join(f"{x:.2f}" for x in t.tolist())
    print(f"\nd={a.d} sigma={a.sigma} (held-out sentences: {len(val.ids)})")

    with torch.no_grad():                                        # where did the network put meaning and wording?
        vs, vl, _ = enc.parts(val.ids[:2000], val.full[:2000])
        pair = lambda v: (v ** 2).view(len(v), -1, 2).sum(-1).mean(0)       # energy per complex coordinate
        tau = 1 / (1 - torch.sigmoid(enc.a))                               # learned memory timescale, in sentences
        print("\nPLACEMENT: share of each part's energy by coordinate timescale tau (sentences remembered)")
        print("            tau<5   5-50   >50")
        for name, v in (("meaning", vs), ("wording", vl)):
            en = pair(v)
            sel = (tau < 5, (tau >= 5) & (tau <= 50), tau > 50)
            print(f"{name:10s} " + "   ".join(f"{(en[m].sum() / en.sum()).item():5.2f}" for m in sel))
        print(f"timescales of all coordinates: min {tau.min().item():.1f}  median {tau.median().item():.1f}  "
              f"max {tau.max().item():.1f}")

    def report(name, split, other_half, flat=False, absent=False, hard=0.0, blind=False, sizes=(1, 4, 8, 16, 32)):
        print(f"\n{name}")
        print("sents | MEANING: in-memory retrieval L0 L1 L2 | pool retrieval L0 L1 L2 | cosine "
              "| WORDING: char-acc L0 L1 L2 | mean importance w L0 L1 L2 | chance (in-memory / pool)")
        for n in sizes:                              # n=1: no crowding noise at all
            if n > a.max_sents:
                continue
            ctx, pool, cos, ca, amp, _, _ = evaluate(enc, dec, key, split, n, other_half, flat, absent, hard, blind, a.vote)
            print(f"{n:5d} | {f3(ctx)} | {f3(pool)} | {cos:.2f} | {f3(ca)} | {f3(amp)} "
                  f"| {1 / n:.2f} / {1 / len(split.ids):.4f}")

    report("cue = first half (same as write key)", val, False)
    report("cue = SECOND half (key must generalize by meaning)", val, True)
    report("ABLATION: importance forced to 1, cue = first half", val, False, flat=True)

    fa = lambda t: " ".join("  - " if x != x else f"{x:.2f}" for x in t.tolist())

    def age_report(name, hard=0.0, sizes=(8, 32)):
        print(f"\n{name}")
        print("sents | WORDING char-acc by age: newest 1-3 4-15 16+ | MEANING pool retrieval by age: newest 1-3 4-15 16+")
        for n in sizes:
            if n > a.max_sents:
                continue
            *_, ca_age, pool_age = evaluate(enc, dec, key, val, n, False, False, False, hard, vote=a.vote, batches=20)
            print(f"{n:5d} | {fa(ca_age)} | {fa(pool_age)}")

    age_report("RECENCY: results by how recent the asked sentence is, cue = first half")
    age_report("RECENCY on HARD memories", hard=1.0)
    report("CHECK memorization: first-half cue on TRAINING sentences (pool = training set)", data, False,
           sizes=(1, 8, 32))
    report("CONTROL energy: the reader is told a constant loudness (does it rely on it?)",
           val, False, blind=True, sizes=(1, 8, 32))
    for half, tag in ((False, "first"), (True, "SECOND")):
        report(f"CONTROL leak: the asked sentence is NOT stored, cue = {tag} half "
               "(in-memory column = among the n candidates)", val, half, absent=True, sizes=(1, 8, 32))
    report("HARD memory: the stored sentences are nearest neighbours of each other, cue = first half",
           val, False, hard=1.0, sizes=(4, 8, 32))
    report("HARD memory: cue = SECOND half", val, True, hard=1.0, sizes=(4, 8, 32))
    report("HARD memory CONTROL: asked sentence NOT stored, cue = SECOND half",
           val, True, absent=True, hard=1.0, sizes=(4, 8, 32))


    def gain_table(title, hard, sizes):
        print(f"\n{title}")
        print("sents | first-half cue: pool retrieval, cosine | second-half cue: pool retrieval, cosine")
        for n in sizes:
            if n > a.max_sents:
                continue
            row = []
            for half in (False, True):
                st = evaluate(enc, dec, key, val, n, half, hard=hard, vote=a.vote)
                ab = evaluate(enc, dec, key, val, n, half, absent=True, hard=hard, vote=a.vote)
                row.append(f"{st[1].mean().item() - ab[1].mean().item():+.2f}  {st[2] - ab[2]:+.2f}")
            print(f"{n:5d} | {row[0]}  | {row[1]}")

    gain_table("MEMORY GAIN = stored minus not stored: what the memory itself adds (random memories)", 0.0, (1, 4, 8, 16, 32))
    gain_table("MEMORY GAIN on HARD memories (stored sentences are neighbours)", 1.0, (4, 8, 32))

if __name__ == "__main__":
    main()
