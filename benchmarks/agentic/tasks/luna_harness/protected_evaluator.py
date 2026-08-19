#!/usr/bin/env python3
"""Protected native-Codex evaluator for evolved Luna translation harnesses."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import json
import multiprocessing
import os
from queue import Empty
import re
import random
import signal
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path


SELECTION_LANGUAGES = ("aii", "lif", "gaq", "usa")
GENERALIZATION_LANGUAGES = ("mpj", "kgk", "tzo", "wbi")
CONFIRMATION_LANGUAGES = ("hat", "tam", "fra", "jpn")
DEVELOPMENT_BLOCKS = {
    "development-1": SELECTION_LANGUAGES,
    "development-2": GENERALIZATION_LANGUAGES,
}
DEVELOPMENT_LANGUAGES = SELECTION_LANGUAGES + GENERALIZATION_LANGUAGES
LARGE_PANEL_NAME = "development-large"
LARGE_PANEL_VERSION = "large-96-v1"
LARGE_PANEL_SEED = 20260804
LARGE_SHARDS = 4
TASKS_PER_SHARD = 3
LARGE_TRANSLATION_ITEMS = (
    len(DEVELOPMENT_LANGUAGES) * LARGE_SHARDS * TASKS_PER_SHARD)
REQUIRED_KEYS = {
    "schema_version", "name", "hypothesis", "activation_condition",
    "fallback", "falsifier", "agent_instructions", "research_prompt",
    "translation_prompt", "tools",
}
OPTIONAL_KEYS = {"causal_prediction"}
V2_KEYS = OPTIONAL_KEYS | {"skills", "orchestration"}
ORCHESTRATION_KEYS = {"topology", "research_phase_prompt", "final_phase_prompt",
                      "compute_envelope"}
FIXED_COMPUTE_ENVELOPE = {
    "model": "gpt-5.6-luna",
    "sessions_per_language": 1,
    "turns_per_session": 2,
    "tasks_per_language": 3,
}
SUPPORTED_AGENT_MODELS = ("gpt-5.6-luna", "gpt-5.6-terra")
ALLOWED_IMPORTS = {
    "argparse", "collections", "csv", "difflib", "functools", "hashlib",
    "heapq", "itertools", "json", "math", "pathlib", "re", "statistics",
    "os", "string", "sys", "unicodedata",
}
FORBIDDEN_TEXT = (
    "http://", "https://", "openai", "anthropic", "openrouter", "curl ",
    "wget ", "pip install", "npm install", "socket", "subprocess",
    "os.system", "requests", "urllib", "/users/", "/private/", "../",
)
WATCHDOG_GRACE_SECONDS = 120.0
DEFAULT_SESSION_WATCHDOG_SECONDS = 20 * 60.0


def _child_session(result_queue, callback_for, code: str) -> None:
    """Run one pinned session in a process the evaluator can terminate."""
    # A native Codex CLI child must die with this controller child on timeout.
    # The evaluator itself remains outside this process group.
    os.setsid()
    try:
        result_queue.put(("result", callback_for(code)))
    except BaseException as error:  # report a worker fault as evaluator evidence
        result_queue.put(("exception", f"{type(error).__name__}: {error}"))


def _stop_session_process(process) -> None:
    """Stop a watchdog child and its native-Codex descendants."""
    if not process.is_alive():
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    process.join(timeout=1)
    if process.is_alive():
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.join(timeout=1)


def _watchdog_timeout(plan: dict) -> float:
    """Give the pinned session its configured time plus a controller grace."""
    execution = plan.get("execution") or {}
    for key in ("timeout_seconds", "timeout"):
        value = execution.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return float(value) + WATCHDOG_GRACE_SECONDS
    return DEFAULT_SESSION_WATCHDOG_SECONDS


def _run_sessions_with_watchdogs(codes, callback_for, *, timeout_seconds: float,
                                 workers: int) -> list[dict]:
    """Run at most ``workers`` sessions, terminating an overrun without retrying.

    ``run_session`` can itself be blocked after its native-Codex timeout.  A
    thread cannot be safely cancelled, so each session is a forked child.  The
    upstream project builder and fixed two-turn plan are inherited unchanged.
    """
    codes = tuple(codes)
    try:
        context = multiprocessing.get_context("fork")
    except ValueError as error:  # the pinned native evaluator requires POSIX
        raise RuntimeError("native Luna watchdog requires multiprocessing fork") from error
    pending = iter(codes)
    active = {}
    results = []

    def launch(code: str) -> None:
        result_queue = context.Queue()
        process = context.Process(
            target=_child_session, args=(result_queue, callback_for, code))
        started = time.monotonic()
        process.start()
        active[process.pid] = (code, process, result_queue, started)

    def failure(code: str, message: str, *, timed_out: bool,
                elapsed: float, exitcode) -> dict:
        return {
            "language_code": code,
            "failure": message,
            "translations": [],
            "timed_out": timed_out,
            "controller_timeout": timed_out,
            "controller_elapsed_seconds": elapsed,
            "controller_exitcode": exitcode,
        }

    try:
        for _ in range(max(1, min(int(workers), len(codes)))):
            try:
                launch(next(pending))
            except StopIteration:
                break
        while active:
            progressed = False
            now = time.monotonic()
            for pid, (code, process, result_queue, started) in list(active.items()):
                elapsed = now - started
                try:
                    kind, value = result_queue.get_nowait()
                except Empty:
                    kind = value = None
                if kind is not None:
                    process.join(timeout=1)
                    result_queue.close()
                    active.pop(pid)
                    if kind == "result" and isinstance(value, dict):
                        value.setdefault("language_code", code)
                        value["controller_timeout"] = False
                        value["controller_elapsed_seconds"] = elapsed
                        value["controller_exitcode"] = process.exitcode
                        results.append(value)
                    else:
                        results.append(failure(
                            code, f"runner exception: {value}", timed_out=False,
                            elapsed=elapsed, exitcode=process.exitcode))
                    try:
                        launch(next(pending))
                    except StopIteration:
                        pass
                    progressed = True
                elif elapsed >= timeout_seconds:
                    _stop_session_process(process)
                    result_queue.close()
                    active.pop(pid)
                    results.append(failure(
                        code,
                        f"controller timeout after {timeout_seconds:.1f} seconds",
                        timed_out=True, elapsed=elapsed, exitcode=process.exitcode))
                    try:
                        launch(next(pending))
                    except StopIteration:
                        pass
                    progressed = True
                elif not process.is_alive():
                    # A child can exit before its queue feeder is observed.
                    process.join(timeout=0)
                    try:
                        kind, value = result_queue.get(timeout=0.05)
                    except Empty:
                        kind = value = None
                    result_queue.close()
                    active.pop(pid)
                    if kind == "result" and isinstance(value, dict):
                        value.setdefault("language_code", code)
                        value["controller_timeout"] = False
                        value["controller_elapsed_seconds"] = elapsed
                        value["controller_exitcode"] = process.exitcode
                        results.append(value)
                    else:
                        detail = value if kind == "exception" else (
                            f"child exited without a result (exitcode {process.exitcode})")
                        results.append(failure(
                            code, f"runner exception: {detail}", timed_out=False,
                            elapsed=elapsed, exitcode=process.exitcode))
                    try:
                        launch(next(pending))
                    except StopIteration:
                        pass
                    progressed = True
            if not progressed:
                time.sleep(0.02)
    finally:
        for _, process, result_queue, _ in active.values():
            _stop_session_process(process)
            result_queue.close()
    return results


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_harness(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"candidate is not a file: {path}")
    if path.stat().st_size > 262_144:
        raise ValueError("candidate exceeds 256 KiB")
    value = json.loads(path.read_text(encoding="utf-8"))
    schema_version = value.get("schema_version") if isinstance(value, dict) else None
    allowed = REQUIRED_KEYS | (OPTIONAL_KEYS if schema_version == 1 else V2_KEYS)
    if (not isinstance(value, dict) or not REQUIRED_KEYS.issubset(value)
            or not set(value).issubset(allowed)):
        raise ValueError(
            f"candidate keys require {sorted(REQUIRED_KEYS)} and may also "
            f"contain {sorted(OPTIONAL_KEYS)}")
    if schema_version not in (1, 2):
        raise ValueError("unsupported candidate schema")
    for key in REQUIRED_KEYS - {"schema_version", "tools"}:
        text = value[key]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{key} must be a non-empty string")
        lowered = text.casefold()
        if any(term in lowered for term in FORBIDDEN_TEXT):
            raise ValueError(f"{key} contains a forbidden external-access term")
    if "causal_prediction" in value:
        prediction = value["causal_prediction"]
        if not isinstance(prediction, str) or not prediction.strip():
            raise ValueError("causal_prediction must be a non-empty string")
    tools = value["tools"]
    if not isinstance(tools, dict) or len(tools) > 6:
        raise ValueError("tools must be an object with at most six files")
    for relative, source in tools.items():
        if not re.fullmatch(r"tools/[a-z][a-z0-9_]{0,47}\.py", relative):
            raise ValueError(f"invalid tool path: {relative!r}")
        if not isinstance(source, str) or len(source.encode()) > 32_768:
            raise ValueError(f"invalid tool source size: {relative}")
        lowered = source.casefold()
        if any(term in lowered for term in FORBIDDEN_TEXT):
            raise ValueError(f"forbidden operation in {relative}")
        tree = ast.parse(source, filename=relative)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = ([alias.name for alias in node.names]
                         if isinstance(node, ast.Import)
                         else [node.module or ""])
                for name in names:
                    root = name.split(".", 1)[0]
                    if root not in ALLOWED_IMPORTS:
                        raise ValueError(f"forbidden import {name!r} in {relative}")
    if schema_version == 2:
        skills = value.get("skills", {})
        if not isinstance(skills, dict) or len(skills) > 6:
            raise ValueError("skills must be an object with at most six files")
        for relative, source in skills.items():
            if not re.fullmatch(
                    r"\.agents/skills/[a-z][a-z0-9_-]{0,47}/SKILL\.md", relative):
                raise ValueError(f"invalid skill path: {relative!r}")
            if not isinstance(source, str) or not source.strip() or len(source.encode()) > 32_768:
                raise ValueError(f"invalid skill source: {relative}")
            if any(term in source.casefold() for term in FORBIDDEN_TEXT):
                raise ValueError(f"forbidden operation in {relative}")
        orchestration = value.get("orchestration")
        if not isinstance(orchestration, dict) or set(orchestration) != ORCHESTRATION_KEYS:
            raise ValueError(f"orchestration must contain exactly {sorted(ORCHESTRATION_KEYS)}")
        if orchestration["topology"] != "single_session_two_turn":
            raise ValueError("only single_session_two_turn orchestration is supported")
        for key in ("research_phase_prompt", "final_phase_prompt"):
            text = orchestration[key]
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"orchestration.{key} must be a non-empty string")
            if any(term in text.casefold() for term in FORBIDDEN_TEXT):
                raise ValueError(f"orchestration.{key} contains a forbidden term")
        if orchestration["compute_envelope"] != FIXED_COMPUTE_ENVELOPE:
            raise ValueError(
                f"orchestration.compute_envelope must equal {FIXED_COMPUTE_ENVELOPE}")
    return value


def _load_upstream(factory_root: Path, corpus_dir: Path):
    source = factory_root / "src"
    harness = factory_root / "benchmarks" / "agentic-translation-harness"
    confirmation = factory_root / "benchmarks" / "luna-prompt-vs-agent-confirmation"
    for required in (source, harness, confirmation):
        if not required.is_dir():
            raise ValueError(f"missing pinned benchmark directory: {required}")
    os.environ["EBIBLE_CORPUS_DIR"] = str(corpus_dir)
    for entry in (str(confirmation), str(harness), str(source)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    study = importlib.import_module("study")
    runner = importlib.import_module("run_codex")
    agent_harness = importlib.import_module("ebiblebench.agent_harness")
    return study, runner, agent_harness


def _plan_for_agent_model(plan: dict, model: str) -> dict:
    """Override only the native Codex model for a controlled model study.

    The pinned study must still name Luna before the override.  This prevents a
    mutable upstream plan from silently changing ordinary Finch fitness while
    allowing an explicit, separately recorded Terra counterfactual.
    """
    if plan["execution"]["agent_model"] != "gpt-5.6-luna":
        raise ValueError("pinned benchmark no longer requests GPT-5.6 Luna")
    if model not in SUPPORTED_AGENT_MODELS:
        raise ValueError(f"unsupported native agent model: {model}")
    result = copy.deepcopy(plan)
    result["execution"]["agent_model"] = model
    result["execution"]["model"] = model
    return result


def _read_index_file(path: Path) -> set[int]:
    if not path.is_file():
        return set()
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return {int(item) for item in value}
    if isinstance(value, dict):
        indices = set()
        for item in value.values():
            if isinstance(item, list):
                indices.update(int(index) for index in item)
        return indices
    raise ValueError(f"unsupported protected-index file: {path}")


def create_large_development_panel(*, factory_root: Path, corpus_dir: Path,
                                   destination: Path,
                                   seed: int = LARGE_PANEL_SEED) -> dict:
    """Freeze 12 unseen shared verses without reading confirmation targets."""
    study, _, _ = _load_upstream(factory_root, corpus_dir)
    plan = study.load_plan()
    source_path = study.resolve_path(plan["corpus"]["source_file"])
    vref_path = study.resolve_path(plan["corpus"]["verse_reference_file"])
    source = source_path.read_text(encoding="utf-8").splitlines()
    vrefs = vref_path.read_text(encoding="utf-8").splitlines()
    targets = {}
    for code in DEVELOPMENT_LANGUAGES:
        path = study.resolve_path(plan["corpus"]["target_files"][code])
        rows = path.read_text(encoding="utf-8").splitlines()
        if len(rows) != len(source):
            raise ValueError(f"parallel line count changed for {code}")
        targets[code] = rows

    excluded = set(int(index) for index in plan["frozen_panel"]["target_indices"])
    profile = factory_root / "benchmarks" / "luna-language-profile" / "data"
    excluded.update(_read_index_file(profile / "historical_evaluation_indices.json"))
    excluded.update(_read_index_file(profile / "evaluation_indices.json"))
    lab = Path(__file__).resolve().parent / "laboratory_panel.json"
    if lab.is_file():
        excluded.update(int(index) for index in json.loads(
            lab.read_text(encoding="utf-8"))["target_indices"])

    try:
        nt_start = vrefs.index("MAT 1:1")
    except ValueError as error:
        raise ValueError("corpus no longer contains MAT 1:1") from error
    bands = ((6, 9), (10, 13), (14, 19), (20, 32))
    candidates = {band: [] for band in bands}
    for index in range(nt_start, len(source)):
        text = source[index].strip()
        if index in excluded or not text or not vrefs[index].strip():
            continue
        if any(not rows[index].strip() or rows[index].lstrip().startswith("<")
               for rows in targets.values()):
            continue
        words = len(text.split())
        for band in bands:
            if band[0] <= words <= band[1]:
                candidates[band].append(index)
                break

    rng = random.Random(int(seed))
    selected = []
    for band in bands:
        rows = list(candidates[band])
        rng.shuffle(rows)
        if len(rows) < 3:
            raise ValueError(f"not enough unseen rows in length band {band}")
        selected.extend(rows[:3])
    rng.shuffle(selected)
    shards = []
    for number in range(LARGE_SHARDS):
        indices = selected[
            number * TASKS_PER_SHARD:(number + 1) * TASKS_PER_SHARD]
        shards.append({
            "id": f"large-{number + 1}",
            "target_indices": indices,
            "verse_references": [vrefs[index] for index in indices],
            "lengths": [len(source[index].split()) for index in indices],
        })
    panel = {
        "schema_version": 1,
        "name": LARGE_PANEL_VERSION,
        "selection_seed": int(seed),
        "languages": list(DEVELOPMENT_LANGUAGES),
        "shards": shards,
        "tasks_per_shard": TASKS_PER_SHARD,
        "translation_items": LARGE_TRANSLATION_ITEMS,
        "excluded_indices_count": len(excluded),
        "selection_rule": (
            "Twelve shared New Testament rows, three from each frozen English "
            "length band, nonempty in all eight development languages, excluding "
            "recorded historical, laboratory, and prior protected indices."),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(panel, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(destination)
    return panel


def _load_shared_development_panel(path: Path, *,
                                   expected_name: str | None = None,
                                   expected_shards: int | None = None) -> dict:
    """Load a private shared-verse panel without touching confirmation data.

    Every shard intentionally remains three verses so the established Luna
    two-turn compute envelope is unchanged.  The number of shards may be
    larger for a separately named audit panel.
    """
    panel = json.loads(path.read_text(encoding="utf-8"))
    shards = panel.get("shards") or []
    if (panel.get("schema_version") != 1
            or not isinstance(panel.get("name"), str)
            or not panel["name"].strip()
            or panel.get("languages") != list(DEVELOPMENT_LANGUAGES)
            or not shards
            or any(not isinstance(row, dict)
                   or not isinstance(row.get("id"), str)
                   or len(row.get("target_indices") or []) != TASKS_PER_SHARD
                   or len(row.get("verse_references") or []) != TASKS_PER_SHARD
                   or len(row.get("lengths") or []) != TASKS_PER_SHARD
                   for row in shards)
            or len({row["id"] for row in shards}) != len(shards)
            or panel.get("tasks_per_shard") != TASKS_PER_SHARD
            or panel.get("translation_items") != (
                len(DEVELOPMENT_LANGUAGES) * len(shards) * TASKS_PER_SHARD)):
        raise ValueError("invalid shared development panel")
    if expected_name is not None and panel["name"] != expected_name:
        raise ValueError("shared development panel name changed")
    if expected_shards is not None and len(shards) != int(expected_shards):
        raise ValueError("shared development panel shard count changed")
    return panel


def _load_large_panel(path: Path) -> dict:
    return _load_shared_development_panel(
        path, expected_name=LARGE_PANEL_VERSION, expected_shards=LARGE_SHARDS)


def _tasks_with_example_count(study, agent_harness, plan: dict,
                              example_count: int) -> dict[str, list[dict]]:
    """Build launch prompts with a frozen number of source-only examples."""
    if not 1 <= int(example_count) <= 100:
        raise ValueError("few-shot example count must be between 1 and 100")
    tasks = study.frozen_tasks(plan)
    source_path = study.resolve_path(plan["corpus"]["source_file"])
    vref_path = study.resolve_path(plan["corpus"]["verse_reference_file"])
    source = source_path.read_text(encoding="utf-8").splitlines()
    vrefs = vref_path.read_text(encoding="utf-8").splitlines()
    for language in plan["language_panel"]:
        code = language["code"]
        language_tasks = tasks[code]
        target_path = study.resolve_path(plan["corpus"]["target_files"][code])
        target = target_path.read_text(encoding="utf-8").splitlines()
        masked = agent_harness.mask_target_lines(target, language_tasks)
        masked, _ = agent_harness.mask_reference_duplicates(
            masked, [task["reference_text"] for task in language_tasks])
        corpus = study.CorpusPair(source_path, target_path, source, masked, vrefs)
        retriever = study.BranchingBM25(
            corpus, coverage_weight=0.5,
            max_restarts=max(10, int(example_count)))
        for task in language_tasks:
            examples = retriever.search_by_index(
                task["target_index"], int(example_count))
            if (len(examples) != int(example_count)
                    or any(example.target.startswith("<") for example in examples)):
                raise ValueError(f"invalid evidence for {task['task_id']}")
            messages = study.raw_messages(
                language["name"], examples, task["source_text"])
            visible = "\n".join(message.content for message in messages)
            if task["reference_text"] in visible:
                raise ValueError(f"held-out reference leaked for {task['task_id']}")
            task["static_messages"] = [
                {"role": message.role, "content": message.content}
                for message in messages
            ]
            task["retrieved_example_count"] = int(example_count)
            task["example_indices"] = [example.index for example in examples]
    return tasks


def _large_shard_tasks(study, agent_harness, work_root: Path, plan: dict,
                       panel_path: Path, shard_id: str,
                       example_count: int = 10):
    work_root.mkdir(parents=True, exist_ok=True)
    panel = _load_shared_development_panel(panel_path)
    try:
        shard = next(row for row in panel["shards"] if row["id"] == shard_id)
    except StopIteration as error:
        raise ValueError(f"unknown large-panel shard: {shard_id}") from error
    panel_hash = sha256(panel_path)
    cache = work_root / (
        f"_protected_{panel_hash[:16]}_{shard_id}_k{int(example_count)}_tasks.json")
    shard_plan = copy.deepcopy(plan)
    shard_plan["language_panel"] = [
        row for row in plan["language_panel"]
        if row["code"] in DEVELOPMENT_LANGUAGES]
    shard_plan["corpus"]["target_files"] = {
        code: plan["corpus"]["target_files"][code]
        for code in DEVELOPMENT_LANGUAGES}
    shard_plan["frozen_panel"] = {
        "selection_seed": panel["selection_seed"],
        "target_indices": list(shard["target_indices"]),
        "line_numbers": [int(index) + 1 for index in shard["target_indices"]],
        "verse_references": list(shard["verse_references"]),
        "verse_length_bands": list(shard["lengths"]),
        "selection_rule": panel["selection_rule"],
    }
    shard_plan["execution"]["raw_calls"] = (
        len(DEVELOPMENT_LANGUAGES) * TASKS_PER_SHARD)
    shard_plan["execution"]["agent_sessions"] = len(DEVELOPMENT_LANGUAGES)
    shard_plan["execution"]["translations_per_language"] = TASKS_PER_SHARD
    if cache.is_file():
        tasks = json.loads(cache.read_text(encoding="utf-8"))
    else:
        tasks = _tasks_with_example_count(
            study, agent_harness, shard_plan, int(example_count))
        temporary = cache.with_suffix(".tmp")
        temporary.write_text(json.dumps(tasks, ensure_ascii=False),
                             encoding="utf-8")
        temporary.chmod(0o600)
        temporary.replace(cache)
    study.cached_tasks_with_prompts = lambda: tasks
    return tasks, shard_plan, panel_hash


def _cardinality_text(text: str, example_count: int) -> str:
    """Keep inherited top-ten wording aligned with the experimental condition."""
    if int(example_count) == 10:
        return text
    return (text
            .replace("top-ten", f"top-{int(example_count)}")
            .replace("Top-ten", f"Top-{int(example_count)}")
            .replace("top ten", f"top {int(example_count)}")
            .replace("Top ten", f"Top {int(example_count)}")
            .replace("exact ten approved", f"exact {int(example_count)} approved")
            .replace("--top-k 10", f"--top-k {int(example_count)}"))


def _materializer(study, agent_harness, candidate: dict, candidate_hash: str,
                  example_count: int = 10):
    original = study.build_project

    def build(root, arm, code, *, plan):
        manifest = original(root, arm, code, plan=plan, preload_baseline=True)
        agents = root / "AGENTS.md"
        agents.write_text(
            agents.read_text(encoding="utf-8")
            + "\n\n# Evolved harness instructions\n\n"
            + _cardinality_text(
                candidate["agent_instructions"].strip(), example_count) + "\n",
            encoding="utf-8")
        for relative, source in candidate["tools"].items():
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise ValueError(f"candidate tool collides with controller file: {relative}")
            destination.write_text(source, encoding="utf-8")
        skill_paths = []
        for relative, source in candidate.get("skills", {}).items():
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise ValueError(f"candidate skill collides with controller file: {relative}")
            destination.write_text(source, encoding="utf-8")
            skill_paths.append(relative)
        if skill_paths:
            agents.write_text(
                agents.read_text(encoding="utf-8")
                + "\n\n# Candidate local skills\n\n"
                + "Read and apply these candidate-provided local methods when relevant: "
                + ", ".join(sorted(skill_paths)) + ".\n",
                encoding="utf-8")
        if int(example_count) != 10:
            for relative in ("TASK.md", "EXPLORATION_PROTOCOL.md"):
                path = root / relative
                path.write_text(
                    _cardinality_text(
                        path.read_text(encoding="utf-8"), example_count),
                    encoding="utf-8")
            search_tool = root / "tools" / "search_examples.py"
            source = search_tool.read_text(encoding="utf-8")
            source = source.replace(
                "args.top_k > 50", f"args.top_k > {int(example_count)}")
            source = source.replace(
                "between 1 and 50", f"between 1 and {int(example_count)}")
            source = source.replace(
                "max_restarts=10", f"max_restarts={int(example_count)}")
            search_tool.write_text(source, encoding="utf-8")
        manifest["candidate_sha256"] = candidate_hash
        manifest["candidate_name"] = candidate["name"]
        manifest["few_shot_examples_per_task"] = int(example_count)
        manifest["files"] = agent_harness.project_file_manifest(
            root, ignored_parts=(".runner", "__pycache__", "project_manifest.json"))
        (root / "project_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest

    return build


def _cached_tasks(study, work_root: Path) -> dict:
    """Build the frozen retrieval prompts once, then reuse them verbatim.

    Branching-BM25 construction is deterministic but relatively expensive.
    This private cache removes evaluator startup friction without exposing or
    changing a single candidate input.
    """
    cache = work_root / "_protected_frozen_tasks_with_prompts.json"
    if cache.is_file():
        tasks = json.loads(cache.read_text(encoding="utf-8"))
    else:
        tasks = study.cached_tasks_with_prompts()
        temporary = cache.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(tasks, ensure_ascii=False), encoding="utf-8")
        temporary.chmod(0o600)
        temporary.replace(cache)
    # The pinned project builder calls this module hook for each language.
    study.cached_tasks_with_prompts = lambda: tasks
    return tasks


def _score_session(result: dict, *, retain_translations: bool = False) -> dict:
    translations = result.get("translations") or []
    values = [float(row.get("metrics", {}).get("chrf", 0.0))
              for row in translations]
    structurally_complete = len(values) == TASKS_PER_SHARD
    score = sum(values) / len(values) if structurally_complete else 0.0
    policy = list(result.get("command_policy_violations") or [])
    failure = str(result.get("failure") or "")
    if policy or failure:
        score = 0.0
        values = [0.0] * TASKS_PER_SHARD
    elif not structurally_complete:
        values = [0.0] * TASKS_PER_SHARD
    trace_summary = result.get("trace_summary") or {}
    scored = {
        "language": result.get("language_code"),
        "score": score,
        "task_scores": values,
        "failure": failure,
        "policy_violations": policy,
        "returncode": result.get("returncode"),
        "timed_out": bool(result.get("timed_out")),
        "controller_timeout": bool(result.get("controller_timeout")),
        "controller_elapsed_seconds": result.get("controller_elapsed_seconds"),
        "controller_exitcode": result.get("controller_exitcode"),
        "elapsed_seconds": result.get("elapsed_seconds"),
        "tool_calls": len(trace_summary.get("tool_calls") or []),
        "search_invocations": int(result.get("search_invocations") or 0),
        "trace_sha256": result.get("trace_sha256"),
        "input_mutations": list(result.get("input_mutations") or []),
    }
    if retain_translations:
        # This option is only for a chmod-0600 audit cache.  Normal Finch
        # fitness continues to expose aggregate scores and hashes only.
        scored["translations"] = copy.deepcopy(translations)
    return scored


def evaluate(candidate_path: Path, *, factory_root: Path, corpus_dir: Path,
             codex: Path, work_root: Path, panel: str = "selection",
             workers: int = 2, panel_spec: Path | None = None,
             shard_id: str | None = None,
             model: str = "gpt-5.6-luna",
             few_shot_examples: int = 10,
             retain_translations: bool = False) -> dict:
    candidate = validate_harness(candidate_path)
    candidate_hash = sha256(candidate_path)
    study, runner, agent_harness = _load_upstream(factory_root, corpus_dir)
    plan = _plan_for_agent_model(study.load_plan(), model)
    panel_hash = None
    if panel == LARGE_PANEL_NAME:
        if panel_spec is None or shard_id is None:
            raise ValueError("large development evaluation needs panel_spec and shard_id")
        languages = DEVELOPMENT_LANGUAGES
        tasks, plan, panel_hash = _large_shard_tasks(
            study, agent_harness, work_root, plan, Path(panel_spec), shard_id,
            example_count=int(few_shot_examples))
    else:
        panels = {
            "selection": SELECTION_LANGUAGES,
            "generalization": GENERALIZATION_LANGUAGES,
            "confirmation": CONFIRMATION_LANGUAGES,
            **DEVELOPMENT_BLOCKS,
        }
        languages = panels[panel]
        tasks = _cached_tasks(study, work_root)
    original_build = runner.build_project
    runner.build_project = _materializer(
        study, agent_harness, candidate, candidate_hash,
        example_count=int(few_shot_examples))
    short_model = model.removeprefix("gpt-5.6-")
    evaluation_root = Path(tempfile.mkdtemp(
        prefix=f"finch-{short_model}-{panel}-", dir=work_root))

    def one(code: str):
        run_dir = evaluation_root / "records"
        run_dir.mkdir(parents=True, exist_ok=True)
        project_root = evaluation_root / "projects"
        initial = study.initial_agent_prompt(
            tasks[code][0]["language_name"], tasks[code],
            enforce_exploration=True, research_only=True)
        initial = _cardinality_text(initial, int(few_shot_examples))
        initial += "\n\n# Evolved research method\n\n" + _cardinality_text(
            candidate["research_prompt"], int(few_shot_examples))
        orchestration = candidate.get("orchestration") or {}
        if orchestration.get("research_phase_prompt"):
            initial += "\n\n# Evolved research-phase orchestration\n\n" + orchestration["research_phase_prompt"]
        continuation = (
            "The research phase is complete in this same session. Use only the "
            "supplied project, exact examples, local tool results, and evidence "
            "record. Translate all three assigned verses and return only the "
            "required structured JSON.\n\n# Evolved generation method\n\n"
            + _cardinality_text(
                candidate["translation_prompt"], int(few_shot_examples)))
        if orchestration.get("final_phase_prompt"):
            continuation += ("\n\n# Evolved final-phase orchestration\n\n"
                             + orchestration["final_phase_prompt"])
        return runner.run_session(
            arm=f"agent_{short_model}", code=code, run_dir=run_dir,
            work_root=project_root, codex=codex, plan=plan,
            tasks_by_language=tasks, initial_prompt=initial,
            continuation_prompt=continuation,
            research_gate_path=".runner/evidence.json")

    try:
        raw_results = _run_sessions_with_watchdogs(
            languages, one, timeout_seconds=_watchdog_timeout(plan),
            workers=workers)
    finally:
        runner.build_project = original_build
    sessions = sorted((_score_session(
        row, retain_translations=retain_translations) for row in raw_results),
                      key=lambda row: row["language"] or "")
    score = sum(row["score"] for row in sessions) / len(sessions)
    result = {
        "panel": panel,
        "shard": shard_id,
        "panel_sha256": panel_hash,
        "score": score,
        "metric": "equal-language mean sentence chrF++",
        "candidate_sha256": candidate_hash,
        "candidate_name": candidate["name"],
        "few_shot_examples_per_task": int(few_shot_examples),
        "native_cli": True,
        "model": model,
        "session_budget": "one persistent two-turn Codex session per language",
        "languages": list(languages),
        "sessions": sessions,
        "failures": sum(bool(row["failure"] or row["policy_violations"])
                        for row in sessions),
        "private_record_root": str(evaluation_root),
    }
    (evaluation_root / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def paired_summary(candidate: dict, baseline: dict) -> dict:
    """Return paired evidence; pairing is by language, never list position."""
    candidate_rows = {row["language"]: row for row in candidate["sessions"]}
    baseline_rows = {row["language"]: row for row in baseline["sessions"]}
    if set(candidate_rows) != set(baseline_rows):
        raise ValueError("candidate and baseline did not evaluate the same languages")
    pairs = []
    for language in sorted(candidate_rows):
        candidate_score = float(candidate_rows[language]["score"])
        baseline_score = float(baseline_rows[language]["score"])
        pairs.append({
            "language": language,
            "candidate_score": candidate_score,
            "baseline_score": baseline_score,
            "delta": candidate_score - baseline_score,
        })
    deltas = [row["delta"] for row in pairs]
    wins = sum(delta > 0 for delta in deltas)
    ties = sum(delta == 0 for delta in deltas)
    return {
        "panel": candidate["panel"],
        "candidate_sha256": candidate["candidate_sha256"],
        "baseline_sha256": baseline["candidate_sha256"],
        "candidate_score": candidate["score"],
        "baseline_score": baseline["score"],
        "mean_delta": sum(deltas) / len(deltas),
        "win_rate": wins / len(deltas),
        "wins": wins,
        "ties": ties,
        "pairs": pairs,
        "candidate_failures": candidate["failures"],
        "baseline_failures": baseline["failures"],
    }


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile needs observations")
    position = (len(ordered) - 1) * float(fraction)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _hierarchical_interval(matrix: list[list[float]], *, seed: int,
                           replicates: int = 2000) -> dict:
    """Paired bootstrap over languages and shared verse slots."""
    if not matrix or not matrix[0]:
        raise ValueError("bootstrap matrix cannot be empty")
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("bootstrap matrix is ragged")
    rng = random.Random(int(seed))
    samples = []
    for _ in range(int(replicates)):
        languages = [rng.randrange(len(matrix)) for _ in matrix]
        verses = [rng.randrange(width) for _ in range(width)]
        values = [matrix[language][verse]
                  for language in languages for verse in verses]
        samples.append(sum(values) / len(values))
    return {
        "method": "hierarchical paired bootstrap over languages and shared verse slots",
        "replicates": int(replicates),
        "lower_95": _percentile(samples, 0.025),
        "median": _percentile(samples, 0.5),
        "upper_95": _percentile(samples, 0.975),
    }


def large_paired_summary(candidate_shards: list[dict],
                         baseline_shards: list[dict],
                         *, panel_sha256: str) -> dict:
    """Aggregate 4×3×8 paired sentences without exposing protected rows."""
    if len(candidate_shards) != LARGE_SHARDS or len(baseline_shards) != LARGE_SHARDS:
        raise ValueError("large evaluation requires all four shards")
    candidates = {row["shard"]: row for row in candidate_shards}
    baselines = {row["shard"]: row for row in baseline_shards}
    if set(candidates) != set(baselines) or len(candidates) != LARGE_SHARDS:
        raise ValueError("candidate and baseline large-panel shards differ")
    candidate_hashes = {row["candidate_sha256"] for row in candidate_shards}
    baseline_hashes = {row["candidate_sha256"] for row in baseline_shards}
    if len(candidate_hashes) != 1 or len(baseline_hashes) != 1:
        raise ValueError("artifact identity changed across large-panel shards")

    by_language = {code: {"candidate": [], "baseline": []}
                   for code in DEVELOPMENT_LANGUAGES}
    shard_summaries = []
    candidate_failures = baseline_failures = 0
    for shard_id in sorted(candidates):
        candidate = candidates[shard_id]
        baseline = baselines[shard_id]
        if (candidate.get("panel_sha256") != panel_sha256
                or baseline.get("panel_sha256") != panel_sha256):
            raise ValueError("large-panel identity changed")
        candidate_rows = {row["language"]: row for row in candidate["sessions"]}
        baseline_rows = {row["language"]: row for row in baseline["sessions"]}
        if set(candidate_rows) != set(DEVELOPMENT_LANGUAGES) or set(
                baseline_rows) != set(DEVELOPMENT_LANGUAGES):
            raise ValueError("large-panel language set changed")
        shard_deltas = []
        for code in DEVELOPMENT_LANGUAGES:
            candidate_values = list(candidate_rows[code]["task_scores"])
            baseline_values = list(baseline_rows[code]["task_scores"])
            if (len(candidate_values) != TASKS_PER_SHARD
                    or len(baseline_values) != TASKS_PER_SHARD):
                raise ValueError("large-panel task count changed")
            by_language[code]["candidate"].extend(candidate_values)
            by_language[code]["baseline"].extend(baseline_values)
            shard_deltas.extend(a - b for a, b in zip(
                candidate_values, baseline_values))
        candidate_failures += int(candidate["failures"])
        baseline_failures += int(baseline["failures"])
        shard_summaries.append({
            "shard": shard_id,
            "mean_delta": sum(shard_deltas) / len(shard_deltas),
            "candidate_failures": candidate["failures"],
            "baseline_failures": baseline["failures"],
            "comparisons": len(shard_deltas),
        })

    matrix = []
    pairs = []
    candidate_values = []
    baseline_values = []
    for code in DEVELOPMENT_LANGUAGES:
        cand = by_language[code]["candidate"]
        base = by_language[code]["baseline"]
        deltas = [a - b for a, b in zip(cand, base)]
        matrix.append(deltas)
        candidate_values.extend(cand)
        baseline_values.extend(base)
        pairs.append({
            "language": code,
            "candidate_score": sum(cand) / len(cand),
            "baseline_score": sum(base) / len(base),
            "delta": sum(deltas) / len(deltas),
            "sentence_wins": sum(value > 0 for value in deltas),
        })
    deltas = [value for row in matrix for value in row]
    ordered = sorted(deltas)
    trim = max(1, int(len(ordered) * 0.1))
    trimmed = ordered[trim:-trim]
    candidate_hash = next(iter(candidate_hashes))
    bootstrap_seed = int(hashlib.sha256(
        f"{candidate_hash}:{panel_sha256}".encode()).hexdigest()[:16], 16)
    language_wins = sum(row["delta"] > 0 for row in pairs)
    return {
        "panel": LARGE_PANEL_NAME,
        "panel_version": LARGE_PANEL_VERSION,
        "panel_sha256": panel_sha256,
        "candidate_sha256": candidate_hash,
        "baseline_sha256": next(iter(baseline_hashes)),
        "candidate_score": sum(candidate_values) / len(candidate_values),
        "baseline_score": sum(baseline_values) / len(baseline_values),
        "mean_delta": sum(deltas) / len(deltas),
        "median_delta": statistics.median(deltas),
        "trimmed_mean_delta": sum(trimmed) / len(trimmed),
        "win_rate": sum(value > 0 for value in deltas) / len(deltas),
        "wins": sum(value > 0 for value in deltas),
        "ties": sum(value == 0 for value in deltas),
        "comparisons": len(deltas),
        "language_win_rate": language_wins / len(pairs),
        "language_wins": language_wins,
        "pairs": pairs,
        "paired_bootstrap": _hierarchical_interval(
            matrix, seed=bootstrap_seed),
        "candidate_failures": candidate_failures,
        "baseline_failures": baseline_failures,
        "shards": shard_summaries,
    }


def evaluate_pair(candidate_path: Path, baseline_path: Path, *, block: str,
                  factory_root: Path, corpus_dir: Path, codex: Path,
                  work_root: Path, workers: int = 2,
                  candidate_first: bool = True,
                  model: str = "gpt-5.6-luna") -> dict:
    """Evaluate both artifacts on exactly the same frozen hidden block."""
    if block not in DEVELOPMENT_BLOCKS:
        raise ValueError(f"not a development block: {block}")
    ordered = (("candidate", candidate_path), ("baseline", baseline_path))
    if not candidate_first:
        ordered = tuple(reversed(ordered))
    results = {}
    for role, path in ordered:
        results[role] = evaluate(
            path, factory_root=factory_root, corpus_dir=corpus_dir,
            codex=codex, work_root=work_root, panel=block, workers=workers,
            model=model)
    summary = paired_summary(results["candidate"], results["baseline"])
    summary["execution_order"] = [role for role, _ in ordered]
    summary["private_record_roots"] = {
        role: result["private_record_root"] for role, result in results.items()
    }
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--factory-root", required=True, type=Path)
    parser.add_argument("--corpus-dir", required=True, type=Path)
    parser.add_argument("--codex", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--panel", choices=("selection", "generalization", "confirmation",
                                             LARGE_PANEL_NAME, *DEVELOPMENT_BLOCKS),
                        default="selection")
    parser.add_argument("--panel-spec", type=Path)
    parser.add_argument("--shard")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--model", choices=SUPPORTED_AGENT_MODELS,
                        default="gpt-5.6-luna")
    args = parser.parse_args()
    print(json.dumps(evaluate(
        args.candidate, factory_root=args.factory_root,
        corpus_dir=args.corpus_dir, codex=args.codex,
        work_root=args.work_root, panel=args.panel, workers=args.workers,
        panel_spec=args.panel_spec, shard_id=args.shard, model=args.model),
        ensure_ascii=False))


if __name__ == "__main__":
    main()
