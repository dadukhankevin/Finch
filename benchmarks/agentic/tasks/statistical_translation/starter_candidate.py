"""Neutral v2 protocol scaffold. Research workers must implement a model."""

import json
import sys


def main():
    if len(sys.argv) != 2 or sys.argv[1] != "translate":
        raise SystemExit("usage: candidate.py translate")
    payload = json.load(sys.stdin)
    print(json.dumps({"translations": ["" for _ in payload.get("queries", [])]}))


if __name__ == "__main__":
    main()
