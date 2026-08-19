import json
import os
import subprocess
import sys
from pathlib import Path

TASK = Path(__file__).resolve().parents[1] / "benchmarks" / "agentic" / "tasks" / "gold_corruption"
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
    assert result["task"] == "gold_corruption"
    assert "human" in result["n_by_break_type"]
    assert 0.0 <= result["score"] <= 1.0


def test_practice_is_denied_without_allocator_env():
    result = run_score(STARTER, "--practice")
    assert result["split"] == "denied"
    assert result["errors"]


def test_always_flag_is_chance_on_smoke(tmp_path):
    always = tmp_path / "always.py"
    always.write_text("def flag(source, candidate, examples, corpus):\n    return True\n")
    never = tmp_path / "never.py"
    never.write_text("def flag(source, candidate, examples, corpus):\n    return False\n")
    assert abs(run_score(always)["score"] - 0.5) < 1e-9
    assert abs(run_score(never)["score"] - 0.5) < 1e-9


def test_always_flag_is_chance_on_practice(tmp_path):
    always = tmp_path / "always.py"
    always.write_text("def flag(source, candidate, examples, corpus):\n    return True\n")
    result = run_score(always, "--practice", env={"FINCH_SEALED_OK": "1"})
    assert result["split"] == "practice"
    assert abs(result["score"] - 0.5) < 1e-9
    assert result["right_on_humans"] == 0.0
    assert result["right_on_breaks"] == 1.0


def test_oracle_is_perfect_on_smoke(tmp_path):
    oracle = tmp_path / "oracle.py"
    oracle.write_text(
        "def flag(source, candidate, examples, corpus):\n"
        "    return candidate != 'child-sit.'\n"
    )
    assert run_score(oracle)["score"] == 1.0


def test_public_docs_do_not_name_evaluation_languages_or_verses():
    public = (
        (TASK / "CONTRACT.md").read_text()
        + (TASK / "score.py").read_text()
        + (TASK / "train.py").read_text()
    )
    for leak in ("aii", "sus", "gun", "deu", "Assyrian", "MRK 15:17", "JHN 15:2"):
        assert leak not in public


def test_allocator_practice_uses_hidden_set():
    result = run_score(STARTER, "--practice", env={"FINCH_SEALED_OK": "1"})
    payload = json.loads((TASK / "cells.json").read_text())
    n_human = sum(1 for cell in payload["cells"] if cell["split"] == "practice" and cell["break_type"] == "human")
    assert result["split"] == "practice"
    assert result["n_by_break_type"]["human"] == n_human
    assert result["errors"] == []


def test_holdout_is_smaller_and_denied_without_env():
    denied = run_score(STARTER, "--holdout")
    assert denied["split"] == "denied"
    result = run_score(STARTER, "--holdout", env={"FINCH_SEALED_OK": "1"})
    payload = json.loads((TASK / "cells.json").read_text())
    n_holdout = sum(1 for cell in payload["cells"] if cell["split"] == "holdout")
    n_practice = sum(1 for cell in payload["cells"] if cell["split"] == "practice")
    assert result["split"] == "holdout"
    assert sum(result["n_by_break_type"].values()) == n_holdout
    assert n_holdout < n_practice


def test_eval_golds_are_not_in_injected_corpus(tmp_path):
    lookup = tmp_path / "lookup.py"
    lookup.write_text(
        "def flag(source, candidate, examples, corpus):\n"
        "    golds = {row.get('gold') for row in corpus or ()}\n"
        "    return (candidate or '') in golds\n"
    )
    result = run_score(lookup, "--practice", env={"FINCH_SEALED_OK": "1"})
    assert result["errors"] == []
    assert result["right_on_humans"] >= 0.99


def test_crashing_flag_is_did_not_flag(tmp_path):
    bad = tmp_path / "crash.py"
    bad.write_text(
        "def flag(source, candidate, examples, corpus):\n    raise RuntimeError('boom')\n"
    )
    result = run_score(bad)
    assert result["score"] == 0.5
    assert result["right_on_humans"] == 1.0
    assert result["right_on_breaks"] == 0.0
