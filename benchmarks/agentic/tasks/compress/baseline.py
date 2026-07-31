"""Baseline artifact for the compress task: adaptive order-0 arithmetic
coding. No context modeling at all — it learns only the running byte
frequencies, so it approaches the corpus's unigram entropy (~4.7 bits
per byte). Everything smarter — context mixing, match models, learned
structure — is headroom for evolution. Pure python + numpy, and it
round-trips exactly (the scorer verifies)."""
import numpy as np

INC = 32
LIMIT = 1 << 16
FULL = 0xFFFFFFFF
HALF = 1 << 31
Q1 = 1 << 30
Q3 = HALF + Q1


class Bits:
    def __init__(self, data=b""):
        self.bytes_ = bytearray(data)
        self.acc = 0
        self.n = 0
        self.pos = 0

    def write(self, bit):
        self.acc = (self.acc << 1) | bit
        self.n += 1
        if self.n == 8:
            self.bytes_.append(self.acc)
            self.acc = self.n = 0

    def flush(self):
        while self.n:
            self.write(0)

    def read(self):
        if self.pos < len(self.bytes_) * 8:
            byte = self.bytes_[self.pos >> 3]
            bit = (byte >> (7 - (self.pos & 7))) & 1
            self.pos += 1
            return bit
        return 0


class Model:
    def __init__(self):
        self.freq = np.ones(256, dtype=np.int64)

    def spans(self, s):
        cum = int(self.freq[:s].sum())
        return cum, int(self.freq[s]), int(self.freq.sum())

    def find(self, target):
        c = np.cumsum(self.freq)
        s = int(np.searchsorted(c, target, side="right"))
        cum = int(c[s - 1]) if s else 0
        return s, cum, int(self.freq[s]), int(c[-1])

    def update(self, s):
        self.freq[s] += INC
        if self.freq.sum() > LIMIT:
            self.freq = (self.freq + 1) // 2


def compress(data: bytes) -> bytes:
    out = Bits()
    model = Model()
    low, high, pending = 0, FULL, 0

    def emit(bit):
        nonlocal pending
        out.write(bit)
        while pending:
            out.write(1 - bit)
            pending -= 1

    for byte in data:
        cum, freq, tot = model.spans(byte)
        span = high - low + 1
        high = low + span * (cum + freq) // tot - 1
        low = low + span * cum // tot
        while True:
            if high < HALF:
                emit(0)
            elif low >= HALF:
                emit(1)
                low -= HALF
                high -= HALF
            elif low >= Q1 and high < Q3:
                pending += 1
                low -= Q1
                high -= Q1
            else:
                break
            low <<= 1
            high = (high << 1) | 1
        model.update(byte)
    pending += 1
    emit(0 if low < Q1 else 1)
    out.flush()
    return len(data).to_bytes(4, "big") + bytes(out.bytes_)


def decompress(blob: bytes) -> bytes:
    n = int.from_bytes(blob[:4], "big")
    bits = Bits(blob[4:])
    model = Model()
    low, high = 0, FULL
    value = 0
    for _ in range(32):
        value = (value << 1) | bits.read()
    out = bytearray()
    for _ in range(n):
        span = high - low + 1
        tot = int(model.freq.sum())
        target = ((value - low + 1) * tot - 1) // span
        s, cum, freq, tot = model.find(target)
        high = low + span * (cum + freq) // tot - 1
        low = low + span * cum // tot
        while True:
            if high < HALF:
                pass
            elif low >= HALF:
                low -= HALF
                high -= HALF
                value -= HALF
            elif low >= Q1 and high < Q3:
                low -= Q1
                high -= Q1
                value -= Q1
            else:
                break
            low <<= 1
            high = (high << 1) | 1
            value = (value << 1) | bits.read()
        out.append(s)
        model.update(s)
    return bytes(out)
