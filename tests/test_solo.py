"""solo.py end to end with a scripted fake agent: the keep/revert
protocol, honest logging, and the final canonical + held-out audit —
the same standard of coverage the driver (test_drive) already has."""
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS_DIR = os.path.join(REPO, "benchmarks", "agentic", "tasks")
FAKE = os.path.join(REPO, "tests", "fake_solo_agent.py")


def test_solo_campaign_end_to_end(tmp_path):
    baseline = tmp_path / "baseline.py"
    baseline.write_text("def priority(item, capacities):\n"
                        "    return capacities\n")      # worst-fit: weak
    run = tmp_path / "solo"
    proc = subprocess.run(
        [sys.executable, "-m", "finch4.solo",
         "--run", str(run), "--task", "binpack",
         "--tasks-dir", TASKS_DIR,
         "--agent-cmd", f"{sys.executable} {FAKE} {{promptfile}}",
         "--experiments", "2", "--baseline", str(baseline),
         "--agent-timeout", "120"],
        capture_output=True, text=True, cwd=REPO)
    assert proc.returncode == 0, proc.stderr

    # experiment one improves and is kept; two ties and is reverted
    results = [json.loads(l) for l in open(run / "results.jsonl")]
    assert [r["kept"] for r in results] == [True, False]

    # the improvement genuinely replaced the champion
    assert "return -capacities" in open(run / "champion.py").read()
    worklog = open(run / "WORKLOG.md").read()
    assert "KEPT" in worklog and "REVERTED" in worklog

    # the final audit re-ran the canonical scorer and the held-out seeds,
    # and the reported score reproduces exactly
    audit = json.load(open(run / "final_audit.json"))
    base = json.load(open(run / "baseline_score.json"))
    assert audit["final_canonical"]["score"] > base["score"]
    assert audit["final_canonical"]["score"] == audit["last_reported"]
    assert audit["final_holdout"]["score"] > 0
