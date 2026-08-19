"""Starter: flag a candidate whose length ratio is far from the examples."""


def flag(source, candidate, examples) -> bool:
    if not (candidate or "").strip():
        return True
    src = max(len(source), 1)
    ratio = len(candidate) / src
    if not examples:
        return ratio < 0.25 or ratio > 2.5
    typical = [len(ex["gold"]) / max(len(ex["source"]), 1) for ex in examples]
    center = sum(typical) / len(typical)
    return abs(ratio - center) > 0.7
