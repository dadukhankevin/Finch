"""Starter: flag a length ratio outside the corpus 5th–95th percentiles."""

_CACHE = {}


def _band(corpus):
    key = id(corpus)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    vals = []
    for row in corpus or ():
        src = max(len(row.get("source") or ""), 1)
        gold = row.get("gold") or ""
        if gold.strip():
            vals.append(len(gold) / src)
    if len(vals) < 4:
        _CACHE[key] = (0.25, 2.5)
        return _CACHE[key]
    vals.sort()
    lo = vals[int(len(vals) * 0.05)]
    hi = vals[min(len(vals) - 1, int(len(vals) * 0.95))]
    _CACHE[key] = (lo, hi)
    return lo, hi


def flag(source, candidate, examples, corpus) -> bool:
    if not (candidate or "").strip():
        return True
    src = max(len(source or ""), 1)
    ratio = len(candidate) / src
    lo, hi = _band(corpus)
    if lo < hi:
        return ratio < lo or ratio > hi
    if not examples:
        return ratio < 0.25 or ratio > 2.5
    typical = [len(ex["gold"]) / max(len(ex["source"]), 1) for ex in examples]
    center = sum(typical) / len(typical)
    return abs(ratio - center) > 0.7
