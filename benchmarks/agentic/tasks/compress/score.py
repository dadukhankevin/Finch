"""Canonical fitness for the compress task — lossless text compression.

The artifact must define:

    def compress(data: bytes) -> bytes
    def decompress(blob: bytes) -> bytes

decompress(compress(data)) must equal data EXACTLY. Score is NEGATIVE
compressed bits per byte: -(len(blob) * 8) / len(data) — higher is
better (engine convention), and directly comparable to the lm task's
bits-per-byte since both model the same corpus.

THE GAME IS THE MODELING: importing any ready-made compression library
is forbidden (zlib, gzip, bz2, lzma, zstandard, brotli, zipfile,
tarfile) — enforced by a source scan AND a blocked import hook. Pure
python + numpy. Both directions combined must finish within TIME_CAP
CPU-seconds on the 64KB slice (SIGALRM enforced).

Canonical slice and the held-out audit slice are disjoint regions of
the same corpus the lm task trains on (tasks/lm/data.txt). Everything
is CPU and bit-deterministic: audits must reproduce EXACTLY (no
tolerance field), and scoring runs need no machine-wide lock — agents
can genuinely run in parallel on this task.

A correct reference implementation lives beside this file
(baseline.py — adaptive order-0 arithmetic coding, ~5.0 bpb on the
canonical slice): a starting point, not a ceiling.

Usage:  python3 score.py artifact.py [--holdout]
"""
import json
import os
import random
import signal
import sys

FORBIDDEN = ("zlib", "gzip", "bz2", "lzma", "zstandard", "zstd",
             "brotli", "zipfile", "tarfile")
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, os.pardir, "lm", "data.txt")
SLICE = 65536
CANON_OFF = 100_000
HOLD_OFF = 1_200_000
TIME_CAP = 60
N_SUBS = 128


def load_slice(holdout):
    """The scored bytes are NOT byte-identical to anything on disk: a
    seeded 0.2% sprinkle of substitutions is applied in memory. An
    honest compressor barely notices; an artifact that embeds or
    fetches corpus content instead of modeling its input cannot
    round-trip (learned from the run where a founder base64-embedded
    the whole canonical slice and reported 0.0 bpb). compress() and
    decompress() must be pure functions of their arguments — any
    dependence on the corpus file, the environment, or hardcoded
    scored content is an automatic audit failure regardless of score."""
    raw = open(DATA, "rb").read()
    off = HOLD_OFF if holdout else CANON_OFF
    s = bytearray(raw[off: off + SLICE])
    rng = random.Random(7 if holdout else 3)
    for _ in range(N_SUBS):
        s[rng.randrange(SLICE)] ^= rng.randrange(1, 256)
    return bytes(s)


def guarded_import(name, *args, **kwargs):
    if name.split(".")[0] in FORBIDDEN:
        raise ImportError(f"compression module {name!r} is forbidden — "
                          "the game is building the model yourself")
    return __import__(name, *args, **kwargs)


def main():
    artifact = os.path.abspath(sys.argv[1])
    holdout = "--holdout" in sys.argv
    data = load_slice(holdout)
    src = open(artifact).read()
    lowered = src.lower()
    hits = [m for m in FORBIDDEN
            if f"import {m}" in lowered or f"from {m}" in lowered]
    if hits:
        print(json.dumps({"task": "compress", "score": -99.0, "bpb": None,
                          "holdout": holdout,
                          "errors": [f"forbidden import(s): {hits}"]}))
        return
    builtins = dict(__builtins__.__dict__
                    if hasattr(__builtins__, "__dict__") else __builtins__)
    builtins["__import__"] = guarded_import
    ns = {"__builtins__": builtins}
    signal.alarm(TIME_CAP + 30)     # hard backstop for load+both directions
    try:
        exec(compile(src, artifact, "exec"), ns)
        import time
        t0 = time.time()
        blob = ns["compress"](data)
        back = ns["decompress"](blob)
        elapsed = time.time() - t0
        signal.alarm(0)
        if not isinstance(blob, (bytes, bytearray)):
            raise TypeError("compress() must return bytes")
        if bytes(back) != data:
            raise ValueError("round-trip failed: decompress(compress(x))"
                             " != x")
        if elapsed > TIME_CAP:
            raise RuntimeError(f"compress+decompress took {elapsed:.1f}s,"
                               f" cap {TIME_CAP}s")
        bpb = len(blob) * 8.0 / len(data)
        print(json.dumps({"task": "compress", "score": -bpb,
                          "bpb": round(bpb, 5),
                          "seconds": round(elapsed, 2),
                          "holdout": holdout, "errors": []}))
    except BaseException as e:
        signal.alarm(0)
        print(json.dumps({"task": "compress", "score": -99.0, "bpb": None,
                          "holdout": holdout, "errors": [repr(e)]}))


if __name__ == "__main__":
    main()
