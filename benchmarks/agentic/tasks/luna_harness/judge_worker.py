#!/usr/bin/env python3
"""Run one named native-Sol judge for one exact Finch match assignment.

Finch supplies the two frozen trajectories and criteria, then records the
verdict and updates Elo mechanically.  It does not participate in judging.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


MODEL = "gpt-5.6-sol"
EXPECTED_RESULT_KEYS = {"winner", "rationale", "evidence", "confidence"}
FORBIDDEN_ASSIGNMENT_KEYS = {"elo", "rating", "ratings", "standings"}


def _server_url(run: Path) -> str:
    info = json.loads((run / "server.json").read_text(encoding="utf-8"))
    port = int(info["port"])
    if not 1 <= port <= 65535:
        raise ValueError("invalid Finch server port")
    return f"http://127.0.0.1:{port}"


def _request_json(url: str, body=None):
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, method="GET" if body is None else "POST",
        headers={} if body is None else {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Finch rejected request ({error.code}): {detail}") from error


def fetch_assignment(run: Path, match_id: str) -> dict:
    query = urllib.parse.urlencode({"id": match_id})
    value = _request_json(f"{_server_url(run)}/match?{query}")
    validate_assignment(value, match_id)
    return value


def _contains_forbidden_key(value) -> bool:
    if isinstance(value, dict):
        return any(str(key).lower() in FORBIDDEN_ASSIGNMENT_KEYS
                   or _contains_forbidden_key(item)
                   for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_forbidden_key(item) for item in value)
    return False


def validate_assignment(assignment: dict, match_id: str) -> None:
    if not isinstance(assignment, dict) or assignment.get("id") != match_id:
        raise ValueError("Finch returned the wrong match assignment")
    if assignment.get("status") != "open" or assignment.get("voided"):
        raise ValueError("match is not open for judging")
    if assignment.get("verdict") is not None:
        raise ValueError("match already has a verdict")
    individuals = assignment.get("individuals")
    if not isinstance(individuals, list) or len(individuals) != 2:
        raise ValueError("judge assignment must contain exactly two trajectories")
    lineage_ids = [item.get("lineage") if isinstance(item, dict) else None
                   for item in individuals]
    if any(not isinstance(item, str) or not item for item in lineage_ids):
        raise ValueError("each trajectory must name its lineage")
    if lineage_ids[0] == lineage_ids[1]:
        raise ValueError("judge assignment must compare different lineages")
    if not isinstance(assignment.get("criteria"), (dict, list, str)):
        raise ValueError("judge assignment lacks frozen criteria")
    if _contains_forbidden_key(assignment):
        raise ValueError("judge assignment leaked population ratings")


def judge_prompt(assignment: dict) -> str:
    return """You are the named independent GPT-5.6 Sol judge for one Finch
Luna-harness match. Compare exactly the two supplied research trajectories
under the supplied frozen criteria and protected evaluation evidence. Judge
trajectory quality and robustness, including reversions and cross-language
evidence; do not invent measurements. You have no population ratings, must not
seek any other files or data, and must not call tools. Finch only records your
decision and never judges.

Return exactly one JSON object and no markdown or surrounding text. It must
have exactly these fields:
{"winner": "LINEAGE_ID or tie", "rationale": "nonempty explanation",
 "evidence": {"specific": "support from the assignment"},
 "confidence": 0.0}
winner must be one of the two supplied lineage IDs or the literal string
"tie". evidence must be a JSON object. confidence must be between 0 and 1.

Exact Finch match assignment:
""" + json.dumps(assignment, ensure_ascii=False, indent=2)


def codex_command(codex: Path, cwd: Path) -> list[str]:
    return [
        str(codex), "exec", "-m", MODEL,
        "-c", 'model_reasoning_effort="low"',
        "-c", 'web_search="disabled"',
        "--disable", "multi_agent", "--sandbox", "read-only",
        "--ephemeral", "--skip-git-repo-check", "-C", str(cwd), "-",
    ]


def invoke_judge(assignment: dict, *, codex: Path, cwd: Path,
                 timeout: float, runner=subprocess.run) -> dict:
    completed = runner(
        codex_command(codex, cwd), input=judge_prompt(assignment), text=True,
        capture_output=True, check=False, timeout=timeout)
    if completed.returncode != 0:
        raise RuntimeError(
            f"native Codex judge failed ({completed.returncode}): "
            f"{completed.stderr.strip()}")
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ValueError("judge output is not one strict JSON object") from error
    validate_result(result, assignment)
    return result


def validate_result(result: dict, assignment: dict) -> None:
    if not isinstance(result, dict) or set(result) != EXPECTED_RESULT_KEYS:
        raise ValueError("judge verdict must contain exactly winner, rationale, evidence, confidence")
    allowed = {item["lineage"] for item in assignment["individuals"]} | {"tie"}
    if result["winner"] not in allowed:
        raise ValueError("judge winner is not one of the assigned lineages or tie")
    if not isinstance(result["rationale"], str) or not result["rationale"].strip():
        raise ValueError("judge rationale must be nonempty")
    if not isinstance(result["evidence"], dict):
        raise ValueError("judge evidence must be a JSON object")
    confidence = result["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("judge confidence must be numeric")
    if not 0 <= float(confidence) <= 1:
        raise ValueError("judge confidence must be between 0 and 1")


def post_verdict(run: Path, assignment: dict, judge: str, result: dict) -> dict:
    body = {
        "match": assignment["id"], "winner": result["winner"],
        "judge": judge, "rationale": result["rationale"].strip(),
        "evidence": result["evidence"], "confidence": float(result["confidence"]),
    }
    return _request_json(f"{_server_url(run)}/verdict", body)


def run_once(*, run: Path, match_id: str, judge: str, codex: Path,
             timeout: float = 300, runner=subprocess.run) -> dict:
    assignment = fetch_assignment(run, match_id)
    assigned = assignment.get("judge")
    if assigned is not None and assigned != judge:
        raise ValueError(f"match is assigned to {assigned!r}, not {judge!r}")
    result = invoke_judge(
        assignment, codex=codex, cwd=run, timeout=timeout, runner=runner)
    return post_verdict(run, assignment, judge, result)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run one named Sol judge worker")
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--match", required=True)
    parser.add_argument("--judge", required=True)
    parser.add_argument("--codex", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args(argv)
    if not args.judge.strip():
        parser.error("--judge cannot be blank")
    result = run_once(
        run=args.run.resolve(), match_id=args.match, judge=args.judge.strip(),
        codex=args.codex.resolve(), timeout=args.timeout)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
