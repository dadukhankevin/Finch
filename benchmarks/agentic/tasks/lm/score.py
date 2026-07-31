"""Canonical fitness for the lm task — train-script optimization.

The artifact is a TRAINING SCRIPT (see train.py, the baseline): it must
define train(train_bytes, budget_seconds, seed) -> model_fn, where
model_fn(byte_array) returns np.float32 log-probs of shape (L-1, 256),
row i a normalized distribution over byte i+1 given bytes[:i+1].

Usage:  python3 score.py artifact.py [--holdout]

Score is NEGATIVE validation bits per byte (higher is better, engine
convention). Canonical: train on the train split with seed 0 for
BUDGET_SECONDS of wall clock, evaluate on the val split. --holdout:
retrain with seed 7, evaluate on the held-back holdout split.

Enforcement, since artifacts are competitors:
- a global file lock serializes every scoring run on this machine (the
  budget is wall clock; two trainings sharing the GPU would corrupt it)
- train() overrunning its budget by >15% + 5s fails the run
- model_fn's rows are spot-checked for normalization (a distribution
  must sum to 1 — returning mass>1 on true bytes would fake a score)
- evaluation itself is capped at EVAL_CAP seconds

This file is CANONICAL: agents run it, never edit it. Audits re-run it
(and --holdout) on the shipped artifact; the numbers must reproduce
within SEED_TOL (MPS training is not bit-deterministic across runs;
the audit tolerance is the honest price — see FINDINGS round 38).
"""
import fcntl
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data.txt")
LOCK = os.path.join(HERE, ".gpu_lock")
BUDGET_SECONDS = 60.0
EVAL_CAP = 120.0
VAL_BYTES = 32768
SEED_TOL = 0.05          # audit tolerance in bpb, MPS nondeterminism


def splits():
    raw = np.frombuffer(open(DATA, "rb").read(), dtype=np.uint8)
    n = len(raw)
    train = raw[: int(n * 0.90)]
    val = raw[int(n * 0.90): int(n * 0.90) + VAL_BYTES]
    holdout = raw[int(n * 0.95): int(n * 0.95) + VAL_BYTES]
    return train, val, holdout


def bits_per_byte(model_fn, val):
    t0 = time.time()
    lp = np.asarray(model_fn(val), dtype=np.float64)
    if time.time() - t0 > EVAL_CAP:
        raise RuntimeError(f"evaluation exceeded {EVAL_CAP}s")
    if lp.shape != (len(val) - 1, 256):
        raise ValueError(f"model_fn shape {lp.shape}, expected "
                         f"({len(val) - 1}, 256)")
    # spot-check normalization on 64 random rows
    rows = np.random.default_rng(0).integers(0, lp.shape[0], 64)
    sums = np.exp(lp[rows]).sum(axis=1)
    if np.any(np.abs(sums - 1.0) > 2e-2):
        raise ValueError("model_fn rows are not normalized distributions "
                         f"(row sums {sums.min():.4f}..{sums.max():.4f})")
    nll = -lp[np.arange(len(val) - 1), val[1:].astype(np.int64)]
    return float(nll.mean() / np.log(2.0))


def main():
    artifact = os.path.abspath(sys.argv[1])
    holdout_mode = "--holdout" in sys.argv
    train_b, val, hold = splits()
    seed, eval_split = (7, hold) if holdout_mode else (0, val)
    with open(LOCK, "w") as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)      # one training at a time
        ns = {}
        try:
            with open(artifact) as f:
                exec(compile(f.read(), artifact, "exec"), ns)
            t0 = time.time()
            model_fn = ns["train"](train_b, BUDGET_SECONDS, seed)
            train_time = time.time() - t0
            if train_time > BUDGET_SECONDS * 1.15 + 5:
                raise RuntimeError(f"train() ran {train_time:.1f}s, "
                                   f"budget {BUDGET_SECONDS}s")
            bpb = bits_per_byte(model_fn, eval_split)
            print(json.dumps({
                "task": "lm", "score": -bpb, "bpb": round(bpb, 5),
                "train_seconds": round(train_time, 1),
                "tolerance": SEED_TOL,
                "holdout": holdout_mode, "errors": []}))
        except Exception as e:
            print(json.dumps({"task": "lm", "score": -99.0, "bpb": None,
                              "holdout": holdout_mode,
                              "errors": [repr(e)]}))


if __name__ == "__main__":
    main()
