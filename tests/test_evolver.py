"""Native evolver: one shared Decoder.md, a population of workers,
and no sealed-evaluation lecture in Finch-owned prompts."""
import json
import os
import subprocess
import sys
from pathlib import Path

from finch4.prompts import EVOLVER_PROMPT, RESEARCHER_PROMPT
from finch4.workspace import prepare_worker_workspace

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS_DIR = os.path.join(REPO, "benchmarks", "agentic", "tasks")
FAKE = os.path.join(REPO, "tests", "fake_solo_agent.py")
LEAKS = ("--practice", "--sealed", "cells.json", "corpus_manifest",
         "FINCH_SEALED", "never pass", "never read", "Do not open",
         "You do not see")


def _repo_text(*parts):
    return Path(REPO, *parts).read_text(encoding="utf-8")


def test_finch_prompts_do_not_name_sealed_evaluation():
    for text in (
        RESEARCHER_PROMPT,
        EVOLVER_PROMPT,
        _repo_text("benchmarks", "agentic", "base_playbook.md"),
        _repo_text(".claude", "skills", "agentic-ga", "SKILL.md"),
    ):
        for leak in LEAKS:
            assert leak not in text


def test_sealed_task_workspace_hides_evaluation_files(tmp_path):
    run = tmp_path / "L0000"
    prepare_worker_workspace(run, TASKS_DIR, "gold_corruption")
    assert not (run / "cells.json").exists()
    assert not (run / "corpus_manifest.json").exists()
    score = (run / "score.py").read_text()
    for leak in LEAKS:
        assert leak not in score
    assert (run / "TASK.md").exists()
    assert "Do not open" not in (run / "TASK.md").read_text()


def test_evolver_runs_a_shared_decoder_population(tmp_path):
    run = tmp_path / "campaign"
    run.mkdir()
    (run / "Decoder.md").write_text(
        "Established: fuller feasible bins first.\n", encoding="utf-8")
    baseline = tmp_path / "baseline.py"
    baseline.write_text("def priority(item, capacities):\n"
                        "    return capacities\n")
    proc = subprocess.run(
        [sys.executable, "-m", "finch4.evolver",
         "--run", str(run), "--task", "binpack",
         "--tasks-dir", TASKS_DIR,
         "--agent-cmd", f"{sys.executable} {FAKE} {{promptfile}}",
         "--population", "2", "--experiments", "1",
         "--agent-timeout", "120",
         "--baseline", str(baseline)],
        capture_output=True, text=True, cwd=REPO)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    state = json.loads((run / "evolver.json").read_text())
    assert len(state["members"]) == 2
    for name, member in state["members"].items():
        prompt = next((run / name).glob("experiment_*.md"))
        text = prompt.read_text(encoding="utf-8")
        assert "fuller feasible bins first" in text
        assert "Your fitness is" in text
        assert (run / name / "Decoder.md").read_text() == (
            run / "Decoder.md").read_text()
        for leak in LEAKS:
            assert leak not in text
        assert (run / name / "driver_results.jsonl").exists()
        assert (run / name / "candidate.py").exists()
