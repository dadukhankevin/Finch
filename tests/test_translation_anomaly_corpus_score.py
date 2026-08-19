import json
import os
import subprocess
import sys
from pathlib import Path

TASK = Path(__file__).resolve().parents[1] / "benchmarks" / "agentic" / "tasks" / "translation_anomaly_corpus"
SCORER = TASK / "score.py"
STARTER = TASK / "train.py"


def run_score(artifact, *flags, env=None):
    merged = os.environ.copy()
    if env:
        merged.update(env)
    out = subprocess.check_output(
        [sys.executable, str(SCORER), str(artifact), *flags],
        text=True,
        env=merged,
    )
    return json.loads(out)


def test_default_is_smoke_and_does_not_need_real_cells():
    result = run_score(STARTER)
    assert result["split"] == "smoke"
    assert result["n_draft"] == 1 and result["n_gold"] == 1
    assert result["draft_flags"] + result["draft_missed"] == result["n_draft"]
    assert result["gold_flags"] + result["gold_clean"] == result["n_gold"]
    assert "aii" not in json.dumps(result)


def test_practice_is_denied_without_allocator_env():
    result = run_score(STARTER, "--practice")
    assert result["split"] == "denied"
    assert result["errors"]


def _split_counts():
    payload = json.loads((TASK / "cells.json").read_text())
    counts = {}
    for cell in payload["cells"]:
        key = (cell["split"], cell["kind"])
        counts[key] = counts.get(key, 0) + 1
    return counts


def test_allocator_practice_score_is_hidden_set():
    result = run_score(STARTER, "--practice", env={"FINCH_SEALED_OK": "1"})
    counts = _split_counts()
    assert result["split"] == "practice"
    assert result["n_draft"] == counts[("practice", "draft")]
    assert result["n_gold"] == counts[("practice", "gold")]
    assert result["errors"] == []


def test_holdout_is_smaller_than_practice_and_denied_without_env():
    denied = run_score(STARTER, "--holdout")
    assert denied["split"] == "denied"
    result = run_score(STARTER, "--holdout", env={"FINCH_SEALED_OK": "1"})
    counts = _split_counts()
    assert result["split"] == "holdout"
    assert result["n_draft"] == counts[("holdout", "draft")]
    assert result["n_draft"] < counts[("practice", "draft")]


def test_public_docs_do_not_name_evaluation_languages_or_verses():
    public = (
        (TASK / "CONTRACT.md").read_text()
        + (TASK / "score.py").read_text()
        + (TASK / "train.py").read_text()
    )
    for leak in ("aii", "sus", "gun", "1JN", "ACT 2:4", "REV 9:14", "Assyrian"):
        assert leak not in public


def test_crashing_flag_errors_do_not_name_cells(tmp_path):
    bad = tmp_path / "crash.py"
    bad.write_text(
        "def flag(source, candidate, examples, corpus):\n    raise RuntimeError('boom')\n"
    )
    result = run_score(bad, "--practice", env={"FINCH_SEALED_OK": "1"})
    blob = json.dumps(result)
    assert "boom" in blob
    assert "aii" not in blob
    assert "REV" not in blob


def test_eval_golds_are_not_in_injected_corpus(tmp_path):
    lookup = tmp_path / "lookup.py"
    lookup.write_text(
        "def flag(source, candidate, examples, corpus):\n"
        "    golds = {row.get('gold') for row in corpus or ()}\n"
        "    return (candidate or '') in golds\n"
    )
    result = run_score(lookup, "--practice", env={"FINCH_SEALED_OK": "1"})
    assert result["errors"] == []
    assert result["should_not_flag"] == 0.0


def test_three_arg_flag_is_an_error_not_a_score(tmp_path):
    old = tmp_path / "old.py"
    old.write_text(
        "def flag(source, candidate, examples):\n    return False\n"
    )
    result = run_score(old, "--practice", env={"FINCH_SEALED_OK": "1"})
    assert result["errors"]
    assert result["score"] == 0.0
