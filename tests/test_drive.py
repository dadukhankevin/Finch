"""The self-driving harness end to end with a scripted fake agent CLI:
founding, breeding waves, mechanical audit, consolidation both in
review-and-stop mode and in auto mode with rewrites."""
import json
import os
import sys

import pytest

from finch4.agentic import AgenticGA
from finch4.drive import Driver

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS_DIR = os.path.join(REPO, "benchmarks", "agentic", "tasks")
FAKE = os.path.join(REPO, "tests", "fake_agentic_agent.py")
CMD = f"{sys.executable} {FAKE}"


def make_driver(tmp_path, **kw):
    args = dict(tasks=["binpack", "tsp"], tasks_dir=TASKS_DIR,
                agent_cmd=CMD, rounds=2, max_parallel=4, agent_timeout=120,
                founders=1, children=2, consolidate_every=2, seed=0)
    args.update(kw)
    return Driver(str(tmp_path / "run"), **args)


def test_auto_mode_full_loop(tmp_path):
    driver = make_driver(tmp_path, auto_consolidate=True)
    driver.run()
    ga = AgenticGA.load(str(tmp_path / "run" / "state.json"))
    s = ga.summary()
    assert s["population"] >= 2
    assert s["tasks_alive"] == ["binpack", "tsp"]
    assert s["base_version"] == 1                       # consolidated once
    assert abs(s["best"]["binpack"] - 0.9417321645639388) < 1e-9
    assert abs(s["best"]["tsp"] - 1.0) < 1e-9
    # mechanical audit ran and passed on both best-evers
    for task, best in ga.consolidation_batch().items():
        audit = json.load(open(os.path.join(
            os.path.dirname(best["artifact"]), "audit.json")))
        assert audit["passed"] and "holdout" in audit
    # base was actually replaced by the applied proposal
    head = open(tmp_path / "run" / "base_playbook.md").read().split("\n")[0]
    assert "consolidated by fake" in head
    # at least one survivor was rewritten via POST /rewrite
    assert any(i["origin"] == "rewrite" for i in ga.individuals.values())
    # prompts were rendered for every job
    assert len(os.listdir(tmp_path / "run" / "jobs")) >= 4


def test_review_mode_stops_cleanly_with_proposal(tmp_path):
    driver = make_driver(tmp_path, auto_consolidate=False)
    with pytest.raises(SystemExit):
        driver.run()
    run = tmp_path / "run"
    assert os.path.exists(run / "proposed_base_v1.md")
    ga = AgenticGA.load(str(run / "state.json"))
    assert ga.summary()["base_version"] == 0            # NOT auto-applied
    head = open(run / "base_playbook.md").read().split("\n")[0]
    assert "version 0" in head                          # base untouched
