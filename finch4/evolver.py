"""Native evolver: a population of auto-research workers on one Decoder.md.

    python3 -m finch4.evolver --run <campaign> --task gold_corruption \
        --tasks-dir benchmarks/agentic/tasks \
        --agent-cmd 'claude -p "$(cat {promptfile})"' \
        --population 6 --experiments 16 --serve-port 51234

Each member is `finch4.solo`. All share the campaign Decoder.md. The
evolver can start, stop, or replace a member. It does not author their
research.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

from .prompts import EVOLVER_PROMPT


STATE_NAME = "evolver.json"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _state_path(run):
    return os.path.join(run, STATE_NAME)


def load_state(run):
    path = _state_path(run)
    if not os.path.exists(path):
        return {"members": {}}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def save_state(run, state):
    with open(_state_path(run), "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)
        handle.write("\n")


def _post(port, route, body):
    payload = json.dumps(body).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/{route}", data=payload, method="POST",
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def _alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _solo_argv(args, member_dir, lineage, decoder):
    argv = [
        args.python, "-m", "finch4.solo",
        "--run", member_dir,
        "--task", args.task,
        "--tasks-dir", args.tasks_dir,
        "--agent-cmd", args.agent_cmd,
        "--experiments", str(args.experiments),
        "--agent-timeout", str(args.agent_timeout),
        "--decoder", decoder,
    ]
    if args.idea:
        argv.extend(["--idea", args.idea])
    if args.baseline:
        argv.extend(["--baseline", args.baseline])
    if lineage:
        argv.extend(["--lineage", lineage])
    if args.serve_port:
        argv.extend(["--serve-port", str(args.serve_port)])
    return argv


def start_member(args, run, decoder, lineage=None, idea=None):
    if args.serve_port and lineage is None:
        found = _post(args.serve_port, "found", {
            "task": args.task,
            "rationale": idea or "native evolver population member",
        })
        lineage = found["id"]
    state = load_state(run)
    if lineage is None:
        lineage = f"W{len(state['members']) + 1:04d}"
    member_dir = os.path.join(run, lineage)
    os.makedirs(member_dir, exist_ok=True)
    log = open(os.path.join(member_dir, "evolver.log"), "ab")
    proc = subprocess.Popen(
        _solo_argv(args, member_dir, lineage, decoder),
        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True)
    state["members"][lineage] = {
        "pid": proc.pid,
        "dir": member_dir,
        "idea": idea or args.idea,
        "started": time.time(),
    }
    save_state(run, state)
    print(f"[evolver] started {lineage} pid={proc.pid}", flush=True)
    return lineage, proc


def stop_member(run, lineage):
    state = load_state(run)
    member = state["members"].get(lineage)
    if not member:
        raise SystemExit(f"unknown member {lineage}")
    pid = member.get("pid")
    if _alive(pid):
        os.killpg(pid, signal.SIGTERM)
        print(f"[evolver] stopped {lineage} pid={pid}", flush=True)
    member["pid"] = None
    member["stopped"] = time.time()
    save_state(run, state)


def main():
    parser = argparse.ArgumentParser(description=EVOLVER_PROMPT)
    parser.add_argument("--run", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--tasks-dir", required=True)
    parser.add_argument("--agent-cmd", required=True)
    parser.add_argument("--population", type=int, default=4)
    parser.add_argument("--experiments", type=int, default=16)
    parser.add_argument("--agent-timeout", type=float, default=2400)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--idea", default=None)
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--serve-port", type=int, default=None)
    parser.add_argument("--kill", default=None, help="stop one member by id")
    parser.add_argument("--insert", action="store_true",
                       help="start one additional member")
    parser.add_argument("--replace", default=None,
                       help="stop a member and start a new one")
    parser.add_argument("--prompt", action="store_true",
                       help="print the Finch evolver prompt and exit")
    args = parser.parse_args()
    if args.prompt:
        print(EVOLVER_PROMPT)
        return
    run = os.path.abspath(args.run)
    os.makedirs(run, exist_ok=True)
    decoder = os.path.join(run, "Decoder.md")
    if not os.path.exists(decoder):
        open(decoder, "w", encoding="utf-8").close()
    if args.kill:
        stop_member(run, args.kill)
        return
    if args.insert:
        start_member(args, run, decoder)
        return
    if args.replace:
        stop_member(run, args.replace)
        start_member(args, run, decoder)
        return

    print(EVOLVER_PROMPT, flush=True)
    procs = []
    state = load_state(run)
    for name, member in list(state["members"].items()):
        if not _alive(member.get("pid")):
            _, proc = start_member(args, run, decoder, lineage=name)
            procs.append(proc)
    live = [name for name, member in load_state(run)["members"].items()
            if _alive(member.get("pid"))]
    for _ in range(max(0, args.population - len(live))):
        _, proc = start_member(args, run, decoder)
        procs.append(proc)
    codes = [proc.wait() for proc in procs]
    print(f"[evolver] population finished: {codes}", flush=True)


if __name__ == "__main__":
    main()
