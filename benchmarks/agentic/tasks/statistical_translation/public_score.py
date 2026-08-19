#!/usr/bin/env python3
"""Score a candidate on the visible practice episodes.

Usage:
    python public_score.py --dataset PRACTICE.json candidate.py

This score is for worker iteration only. GAR selection uses a protected common
episode set that is not written into worker workspaces.
"""

import argparse
import json
import sys

from evaluator import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("candidate")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args()
    with open(args.dataset, encoding="utf-8") as handle:
        dataset = json.load(handle)
    ordinary = evaluate(args.candidate, dataset, python=args.python,
                        diagnostics=args.diagnostics)
    composition = evaluate(
        args.candidate, dataset, python=args.python,
        diagnostics=args.diagnostics, query_group="composition_queries")
    ordinary.pop("_rows", None)
    composition.pop("_rows", None)
    print(json.dumps({"score": ordinary["score"], "ordinary": ordinary,
                      "composition": composition},
                     ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
