"""Baseline artifact for the lm task — a small byte-level transformer.

This file is the STARTING POINT that evolution edits. The contract the
canonical scorer (score.py) enforces on every artifact:

    def train(train_bytes, budget_seconds, seed):
        '''train_bytes: np.uint8 array. Train any model you like within
        budget_seconds of wall clock (the scorer measures; overruns
        fail). Return model_fn: a callable taking a np.uint8 array of
        length L and returning np.float32 log-probabilities of shape
        (L-1, 256), row i = log P(byte[i+1] | bytes[:i+1]), each row a
        normalized distribution (the scorer spot-checks logsumexp).'''

Everything inside train() is fair game for evolution: architecture,
optimizer, schedule, batch shapes, data ordering, precision — anything
that lowers validation bits per byte within the same wall-clock budget.
"""
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CTX = 128
D = 192
LAYERS = 4
HEADS = 4
BATCH = 48
LR = 3e-3


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D)
        self.attn = nn.MultiheadAttention(D, HEADS, batch_first=True)
        self.ln2 = nn.LayerNorm(D)
        self.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(),
                                 nn.Linear(4 * D, D))

    def forward(self, x, mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class ByteLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(256, D)
        self.pos = nn.Embedding(CTX, D)
        self.blocks = nn.ModuleList([Block() for _ in range(LAYERS)])
        self.ln = nn.LayerNorm(D)
        self.head = nn.Linear(D, 256)

    def forward(self, idx):
        L = idx.shape[1]
        mask = torch.triu(torch.full((L, L), float("-inf"),
                                     device=idx.device), diagonal=1)
        x = self.emb(idx) + self.pos(torch.arange(L, device=idx.device))
        for b in self.blocks:
            x = b(x, mask)
        return self.head(self.ln(x))


def train(train_bytes, budget_seconds, seed):
    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = ("mps" if torch.backends.mps.is_available() else "cpu")
    model = ByteLM().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    data = torch.from_numpy(train_bytes.astype(np.int64))
    n = len(data) - CTX - 1
    rng = np.random.default_rng(seed)
    step = 0
    model.train()
    while time.time() - t0 < budget_seconds * 0.92:
        ix = torch.from_numpy(rng.integers(0, n, BATCH))
        xb = torch.stack([data[i:i + CTX] for i in ix]).to(device)
        yb = torch.stack([data[i + 1:i + CTX + 1] for i in ix]).to(device)
        # simple cosine decay against the time budget
        frac = (time.time() - t0) / budget_seconds
        for g in opt.param_groups:
            g["lr"] = LR * 0.5 * (1 + math.cos(math.pi * min(frac, 1.0)))
        logits = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, 256), yb.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
    model.eval()

    def model_fn(byte_array):
        arr = torch.from_numpy(byte_array.astype(np.int64)).to(device)
        outs = []
        with torch.no_grad():
            # sliding chunks; each position conditioned on up to CTX bytes
            start = 0
            while start < len(arr) - 1:
                end = min(start + CTX, len(arr))
                logits = model(arr[start:end][None, :])[0]
                lp = F.log_softmax(logits.float(), dim=-1)
                take = (end - start - 1) if start == 0 else (end - start
                                                            - CTX // 2)
                offset = 0 if start == 0 else CTX // 2 - 1
                outs.append(lp[offset:offset + take].cpu().numpy())
                if end == len(arr):
                    break
                start = end - CTX // 2
        return np.concatenate(outs, axis=0)[:len(byte_array) - 1]

    return model_fn
