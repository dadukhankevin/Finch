"""Protected batch evaluation for non-neural statistical translation v1.

Candidates receive every approved parallel training pair and a batch of
held-out source queries. They may use as much or as little of that supplied
data as they want. No pretrained or external linguistic state is permitted.
Selection fitness is macro sentence chrF++ only. Pair-shuffle behavior,
copying, and improvement over a frozen retrieval-copy baseline are diagnostics.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import random
import re
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

from sacrebleu.metrics import CHRF


TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
CHRF = CHRF(word_order=2)
ALLOWED_IMPORTS = {
    "bisect", "collections", "difflib", "functools", "heapq",
    "itertools", "json", "math", "numpy", "operator", "random", "re",
    "statistics", "string", "sys", "unicodedata",
}
FORBIDDEN_CALLS = {"__import__", "compile", "eval", "exec", "open"}
TARGET_FILES = {
    "French": "fra-fraLSG.txt",
    "Swahili": "swh-swhulb.txt",
    "Tzotzil": "tzo-tzoZNT.txt",
    "Limbu": "lif-lifNT2.txt",
    "Assyrian Neo-Aramaic": "aii-aii.txt",
    "Gamo": "gaq-gaq.txt",
}


def normalize(text: str) -> str:
    return " ".join(TOKEN_RE.findall(text.casefold()))


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def detokenize(items: list[str]) -> str:
    text = " ".join(items)
    text = re.sub(r"\s+([,.;:!?%\]\)])", r"\1", text)
    text = re.sub(r"([\[\(])\s+", r"\1", text)
    text = re.sub(r"\s+(['’])\s+", r"\1", text)
    return text.strip()


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def admissibility(candidate: str | os.PathLike[str]) -> dict:
    path = Path(candidate)
    if not path.is_file():
        return {"passed": False, "reasons": ["candidate.py is missing"]}
    if path.stat().st_size > 128 * 1024:
        return {"passed": False, "reasons": ["candidate exceeds 128 KiB"]}
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as error:
        return {"passed": False,
                "reasons": [f"unreadable Python: {error}"]}
    reasons = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [item.name.split(".")[0] for item in node.names]
            bad = sorted(set(names) - ALLOWED_IMPORTS)
            if bad:
                reasons.append("imports outside allowlist: " + ", ".join(bad))
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in ALLOWED_IMPORTS:
                reasons.append(
                    f"import outside allowlist: {root or '<relative>'}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
                reasons.append(f"forbidden call: {node.func.id}")
            if isinstance(node.func, ast.Attribute) and node.func.attr in {
                    "connect", "download", "popen", "run", "system",
                    "urlopen"}:
                reasons.append(f"forbidden call: .{node.func.attr}")
    return {"passed": not reasons, "reasons": sorted(set(reasons))}


def _language_specs(languages):
    if isinstance(languages, dict):
        return list(languages.items())
    return [(name, TARGET_FILES[name]) for name in languages]


def _unique_valid_indices(source_lines, target_sets, exclude_sources=()):
    excluded = set(exclude_sources)
    seen = set()
    indices = []
    for index, source in enumerate(source_lines):
        key = normalize(source)
        if (not key or key in seen or key in excluded
                or len(key.split()) < 4):
            continue
        if any(index >= len(lines) or len(normalize(lines[index]).split()) < 3
               for lines in target_sets.values()):
            continue
        seen.add(key)
        indices.append(index)
    return indices


def _transport_copy_index(query: str, sources: list[str]) -> int:
    """Frozen v0 champion mechanism, now the strong copy baseline."""
    q = tokenize(query.casefold())
    q_counts = Counter(q)
    best = None
    for index, source in enumerate(sources):
        items = tokenize(source.casefold())
        counts = Counter(items)
        flow = sum(min(q_counts[token], counts[token])
                   for token in q_counts if token in counts)
        q_len = len(q) or 1
        s_len = len(items) or 1
        position = sum(
            1 / (1 + abs((q.index(token) + .5) / q_len
                          - (items.index(token) + .5) / s_len))
            for token in set(q) & set(items))
        row = (2 * flow + position, -index, index)
        if best is None or row > best:
            best = row
    return 0 if best is None else best[2]


def build_dataset(corpus_dir: str | os.PathLike[str], languages,
                  seed: int, training_size: int,
                  queries_per_language: int, exclude_sources=(),
                  composition_queries_per_language: int = 0) -> dict:
    corpus = Path(corpus_dir)
    source_path = corpus / "eng-engULB.txt"
    source_lines = source_path.read_text(encoding="utf-8").splitlines()
    specs = _language_specs(languages)
    targets = {
        language: (corpus / filename).read_text(
            encoding="utf-8").splitlines()
        for language, filename in specs
    }
    valid = _unique_valid_indices(source_lines, targets, exclude_sources)
    random.Random(seed).shuffle(valid)
    needed = int(training_size) + int(queries_per_language)
    if len(valid) < needed:
        raise ValueError(f"need {needed} unique aligned rows, found {len(valid)}")
    training_indices = valid[:training_size]
    query_indices = valid[training_size:needed]
    training_sources = [source_lines[index] for index in training_indices]
    copy_indices = [
        _transport_copy_index(source_lines[index], training_sources)
        for index in query_indices]
    composition = []
    pair_order = [(left, right) for left in range(training_size)
                  for right in range(left + 1, training_size)]
    random.Random(seed + 8_191).shuffle(pair_order)
    used = set()
    for left, right in pair_order:
        if len(composition) >= int(composition_queries_per_language):
            break
        left_tokens = set(tokenize(training_sources[left].casefold()))
        right_tokens = set(tokenize(training_sources[right].casefold()))
        union = left_tokens | right_tokens
        overlap = len(left_tokens & right_tokens) / max(1, len(union))
        if (left in used or right in used or overlap > .35
                or not 4 <= len(left_tokens) <= 24
                or not 4 <= len(right_tokens) <= 24):
            continue
        used.update((left, right))
        composition.append((left, right))
    if len(composition) < int(composition_queries_per_language):
        raise ValueError("not enough disjoint compositional training pairs")
    return {
        "schema": 2,
        "seed": int(seed),
        "training_size": int(training_size),
        "queries_per_language": int(queries_per_language),
        "composition_queries_per_language": int(
            composition_queries_per_language),
        "source_sha256": file_sha256(source_path),
        "training_indices_sha256": hashlib.sha256(
            json.dumps(training_indices).encode()).hexdigest(),
        "query_indices_sha256": hashlib.sha256(
            json.dumps(query_indices).encode()).hexdigest(),
        "languages": [
            {
                "name": language,
                "target_file": filename,
                "training": [
                    {"source": source_lines[index],
                     "target": targets[language][index]}
                    for index in training_indices
                ],
                "queries": [
                    {"source": source_lines[index],
                     "reference": targets[language][index],
                     "frozen_copy_index": copy_index}
                    for index, copy_index in zip(query_indices, copy_indices)
                ],
                "composition_queries": [
                    {
                        "source": (source_lines[training_indices[left]] + " "
                                   + source_lines[training_indices[right]]),
                        "reference": (targets[language][training_indices[left]]
                                      + " "
                                      + targets[language][training_indices[right]]),
                        "frozen_copy_index": _transport_copy_index(
                            source_lines[training_indices[left]] + " "
                            + source_lines[training_indices[right]],
                            training_sources),
                        "kind": "novel_conjunction",
                    }
                    for left, right in composition
                ],
            }
            for language, filename in specs
        ],
    }


def dataset_source_keys(dataset: dict) -> set[str]:
    keys = set()
    for language in dataset["languages"]:
        keys.update(normalize(item["source"])
                    for item in language["training"])
        keys.update(normalize(item["source"])
                    for item in language["queries"])
    return keys


def dataset_public_copy(dataset: dict) -> dict:
    return dataset


def dataset_hash(dataset: dict) -> str:
    payload = json.dumps(dataset, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _codebook(texts: list[str], prefix: str, seed: int):
    vocabulary = sorted({token for text in texts for token in tokenize(text)})
    codes = [f"{prefix}{index:06d}x" for index in range(len(vocabulary))]
    random.Random(seed).shuffle(codes)
    forward = dict(zip(vocabulary, codes))
    backward = {code: token for token, code in forward.items()}
    return forward, backward


def _encode(text: str, mapping: dict[str, str]) -> str:
    return " ".join(mapping[token] for token in tokenize(text))


def _decode(text: str, mapping: dict[str, str]) -> str:
    decoded = [mapping.get(token, mapping.get(token.casefold(), token))
               for token in tokenize(text)]
    return detokenize(decoded)


def _candidate_call(python: str, candidate: str, payload: dict,
                    timeout: float) -> tuple[list[str], float]:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="finch-statsmt-batch-") as cwd:
        completed = subprocess.run(
            [python, candidate, "translate"],
            input=json.dumps(payload, ensure_ascii=False),
            text=True, capture_output=True, cwd=cwd, env=env,
            timeout=timeout, check=False,
        )
    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"translate exited {completed.returncode}: "
            f"{completed.stderr[-1000:]}")
    if len(completed.stdout) > 2 * 1024 * 1024:
        raise RuntimeError("translate output exceeded 2 MiB")
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("translate produced no JSON")
    result = json.loads(lines[-1])
    if not isinstance(result, dict):
        raise RuntimeError("translate result is not an object")
    translations = result.get("translations")
    if not isinstance(translations, list):
        raise RuntimeError("translate.translations is not a list")
    if any(not isinstance(text, str) for text in translations):
        raise RuntimeError("every translation must be a string")
    return translations, elapsed


def _derange_training(training: list[dict], seed: int) -> list[dict]:
    if len(training) < 2:
        return [dict(item) for item in training]
    shift = 1 + seed % (len(training) - 1)
    targets = [item["target"] for item in training]
    return [
        {"source": item["source"],
         "target": targets[(index + shift) % len(targets)]}
        for index, item in enumerate(training)
    ]


def _mean(items):
    return sum(items) / len(items) if items else 0.0


def evaluate(candidate: str | os.PathLike[str], dataset: dict, python: str,
             modes: tuple[str, ...] = ("natural", "cipher"),
             timeout: float = 60.0, diagnostics: bool = False,
             query_group: str = "queries") -> dict:
    candidate = str(Path(candidate).resolve())
    check = admissibility(candidate)
    if not check["passed"]:
        return {
            "score": -100.0, "correct_chrf": 0.0,
            "admissible": False, "admissibility": check,
            "failures": 1, "episodes": 0,
            "dataset_hash": dataset_hash(dataset), "_rows": [],
        }

    rows = []
    failures = []
    total_seconds = 0.0
    base_seed = int(dataset["seed"])
    for language_number, language in enumerate(dataset["languages"]):
        raw_training = language["training"]
        raw_queries = language[query_group]
        for mode_number, mode in enumerate(modes):
            episode_seed = (base_seed * 1_000_003
                            + language_number * 10_007 + mode_number)
            if mode == "cipher":
                source_forward, _ = _codebook(
                    [item["source"] for item in raw_training]
                    + [item["source"] for item in raw_queries],
                    "s", episode_seed + 11)
                target_forward, target_backward = _codebook(
                    [item["target"] for item in raw_training]
                    + [item["reference"] for item in raw_queries],
                    "t", episode_seed + 29)
                training = [
                    {"source": _encode(item["source"], source_forward),
                     "target": _encode(item["target"], target_forward)}
                    for item in raw_training]
                queries = [_encode(item["source"], source_forward)
                           for item in raw_queries]
            elif mode == "natural":
                training = [{"source": item["source"],
                             "target": item["target"]}
                            for item in raw_training]
                queries = [item["source"] for item in raw_queries]
                target_backward = {}
            else:
                raise ValueError(f"unknown mode {mode!r}")

            translations = None
            shuffled_translations = None
            try:
                translations, elapsed = _candidate_call(
                    python, candidate,
                    {"training": training, "queries": queries}, timeout)
                total_seconds += elapsed
                if len(translations) != len(raw_queries):
                    raise RuntimeError(
                        f"expected {len(raw_queries)} translations, got "
                        f"{len(translations)}")
                if diagnostics:
                    shuffled_training = _derange_training(
                        training, episode_seed + 47)
                    shuffled_translations, elapsed = _candidate_call(
                        python, candidate,
                        {"training": shuffled_training, "queries": queries},
                        timeout)
                    total_seconds += elapsed
                    if len(shuffled_translations) != len(raw_queries):
                        raise RuntimeError("wrong number of shuffled translations")
            except Exception as error:
                failures.append({"language": language["name"],
                                 "mode": mode, "error": repr(error)})

            raw_target_set = {normalize(item["target"])
                              for item in raw_training}
            for query_number, raw_query in enumerate(raw_queries):
                if translations is None:
                    output = ""
                    shuffled_output = "" if diagnostics else None
                else:
                    output = translations[query_number]
                    shuffled_output = (shuffled_translations[query_number]
                                       if diagnostics else None)
                    if mode == "cipher":
                        output = _decode(output, target_backward)
                        if diagnostics:
                            shuffled_output = _decode(
                                shuffled_output, target_backward)
                reference = raw_query["reference"]
                correct = CHRF.sentence_score(output, [reference]).score
                shuffled = (None if not diagnostics else
                            CHRF.sentence_score(
                                shuffled_output, [reference]).score)
                copy_target = raw_training[
                    raw_query["frozen_copy_index"]]["target"]
                frozen_copy = CHRF.sentence_score(
                    copy_target, [reference]).score
                rows.append({
                    "language": language["name"],
                    "query": query_number, "mode": mode,
                    "correct_chrf": correct,
                    "shuffled_chrf": shuffled,
                    "alignment_gain": (None if shuffled is None
                                       else correct - shuffled),
                    "frozen_copy_chrf": frozen_copy,
                    "copy_uplift": correct - frozen_copy,
                    "exact_training_copy": normalize(output) in raw_target_set,
                    "output_length_ratio": (
                        len(tokenize(output)) / max(1, len(tokenize(reference)))),
                })

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["language"], row["mode"])].append(row)
    cells = {}
    for (language, mode), items in sorted(grouped.items()):
        correct = _mean([item["correct_chrf"] for item in items])
        frozen = _mean([item["frozen_copy_chrf"] for item in items])
        cell = {
            "correct_chrf": correct,
            "frozen_copy_chrf": frozen,
            "copy_uplift": correct - frozen,
            "exact_training_copy_rate": _mean([
                float(item["exact_training_copy"]) for item in items]),
            "output_length_ratio": _mean([
                item["output_length_ratio"] for item in items]),
        }
        if diagnostics:
            shuffled = _mean([item["shuffled_chrf"] for item in items])
            cell.update(shuffled_chrf=shuffled,
                        alignment_gain=correct - shuffled)
        cells[f"{language}:{mode}"] = cell

    mean_correct = _mean([cell["correct_chrf"] for cell in cells.values()])
    mean_frozen = _mean([
        cell["frozen_copy_chrf"] for cell in cells.values()])
    result = {
        "score": mean_correct,
        "correct_chrf": mean_correct,
        "frozen_copy_chrf": mean_frozen,
        "copy_uplift": mean_correct - mean_frozen,
        "exact_training_copy_rate": _mean([
            cell["exact_training_copy_rate"] for cell in cells.values()]),
        "output_length_ratio": _mean([
            cell["output_length_ratio"] for cell in cells.values()]),
        "admissible": True,
        "admissibility": check,
        "failures": len(failures),
        "failure_examples": failures[:5],
        "episodes": len(rows),
        "batch_seconds": total_seconds,
        "cells": cells,
        "dataset_hash": dataset_hash(dataset),
        "artifact_sha256": file_sha256(candidate),
        "query_group": query_group,
        "_rows": rows,
    }
    if diagnostics:
        mean_shuffled = _mean([
            cell["shuffled_chrf"] for cell in cells.values()])
        result.update(shuffled_chrf=mean_shuffled,
                      alignment_gain=mean_correct - mean_shuffled)
    return result


def public_result(result: dict) -> dict:
    """Strip query-level protected rows before persistence or worker display."""
    return {key: value for key, value in result.items() if key != "_rows"}


def paired_bootstrap_delta(candidate: dict, baseline: dict, *, seed: int,
                           replicates: int = 10_000,
                           metric: str = "correct_chrf") -> dict:
    """Paired query-cluster bootstrap, retaining natural/cipher together."""
    candidate_rows = {
        (row["language"], row["query"], row["mode"]): row
        for row in candidate.get("_rows", [])}
    baseline_rows = {
        (row["language"], row["query"], row["mode"]): row
        for row in baseline.get("_rows", [])}
    if candidate_rows.keys() != baseline_rows.keys() or not candidate_rows:
        raise ValueError("candidate and baseline rows are not paired")
    queries = defaultdict(set)
    modes = defaultdict(set)
    for language, query, mode in candidate_rows:
        queries[language].add(query)
        modes[language].add(mode)

    def macro_delta(sampled=None):
        cells = []
        for language in sorted(queries):
            chosen = (sorted(queries[language]) if sampled is None
                      else sampled[language])
            for mode in sorted(modes[language]):
                diffs = [
                    candidate_rows[(language, query, mode)][metric]
                    - baseline_rows[(language, query, mode)][metric]
                    for query in chosen]
                cells.append(_mean(diffs))
        return _mean(cells)

    rng = random.Random(seed)
    values = []
    for _ in range(int(replicates)):
        sampled = {}
        for language, query_set in queries.items():
            choices = sorted(query_set)
            sampled[language] = [rng.choice(choices)
                                 for _ in range(len(choices))]
        values.append(macro_delta(sampled))
    values.sort()

    def percentile(fraction):
        index = min(len(values) - 1,
                    max(0, round(fraction * (len(values) - 1))))
        return values[index]

    return {
        "metric": metric, "delta": macro_delta(),
        "ci95": [percentile(.025), percentile(.975)],
        "replicates": int(replicates),
        "cluster": "source query; natural/cipher retained together",
    }
