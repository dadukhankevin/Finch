"""Frozen v1 baseline: v0 champion's position-aware transport copy rule."""

import json
import re
import sys
from collections import Counter


def tokens(text):
    return re.findall(r"\w+|[^\w\s]", text.casefold(), flags=re.UNICODE)


def choose(query, training):
    q = tokens(query)
    q_counts = Counter(q)
    best = None
    for index, example in enumerate(training):
        source = tokens(example["source"])
        counts = Counter(source)
        flow = sum(min(q_counts[token], counts[token])
                   for token in q_counts if token in counts)
        q_len = len(q) or 1
        s_len = len(source) or 1
        position = sum(
            1 / (1 + abs((q.index(token) + .5) / q_len
                          - (source.index(token) + .5) / s_len))
            for token in set(q) & set(source))
        row = (2 * flow + position, -index, example["target"])
        if best is None or row > best:
            best = row
    return "" if best is None else best[2]


def main():
    if len(sys.argv) != 2 or sys.argv[1] != "translate":
        raise SystemExit("usage: candidate.py translate")
    payload = json.load(sys.stdin)
    training = payload.get("training", [])
    translations = [choose(query, training)
                    for query in payload.get("queries", [])]
    print(json.dumps({"translations": translations}, ensure_ascii=False))


if __name__ == "__main__":
    main()
