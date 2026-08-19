"""Small agent-facing client for Finch's shared Markdown research record.

The content stays one ordinary ``Decoder.md``.  This client only gives
parallel workers race-free read and append operations through the campaign
server; Finch validates citations and versions the file but never authors or
interprets a finding.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import urllib.error
import urllib.parse
import urllib.request


def _server_url(run: Path) -> str:
    value = json.loads((run / "server.json").read_text(encoding="utf-8"))
    port = int(value["port"])
    if not 1 <= port <= 65535:
        raise ValueError("invalid Finch server port")
    return f"http://127.0.0.1:{port}"


def _request(run: Path, route: str, body=None):
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        f"{_server_url(run)}/{route}", data=data,
        method="GET" if body is None else "POST",
        headers={} if body is None else {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Finch rejected {route!r} ({error.code}): {detail}") from error


def read_research(args) -> None:
    run = Path(args.run).resolve()
    query = "research"
    if args.lineage:
        query += "?" + urllib.parse.urlencode({"lineage": args.lineage})
    payload = _request(run, query)
    if args.markdown:
        sys.stdout.write(payload["decoder"]["artifact"])
        if not payload["decoder"]["artifact"].endswith("\n"):
            sys.stdout.write("\n")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


def append_finding(args) -> None:
    run = Path(args.run).resolve()
    path = Path(args.file).resolve()
    fragment = path.read_text(encoding="utf-8")
    result = _request(run, "finding", {
        "fragment": fragment,
        "rationale": args.rationale,
    })
    print(json.dumps({
        "version": result["version"],
        "artifact_sha256": result["artifact_sha256"],
        "idempotent": result.get("idempotent", False),
    }, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read or append cited Markdown in Finch shared research")
    commands = parser.add_subparsers(dest="command", required=True)

    command = commands.add_parser("read")
    command.add_argument("--run", required=True)
    command.add_argument("--lineage")
    command.add_argument("--markdown", action="store_true")
    command.set_defaults(function=read_research)

    command = commands.add_parser("append")
    command.add_argument("--run", required=True)
    command.add_argument("--file", required=True)
    command.add_argument("--rationale", required=True)
    command.set_defaults(function=append_finding)

    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
