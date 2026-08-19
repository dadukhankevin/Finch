"""solo.py end to end with a scripted fake agent: the keep/revert
protocol, honest logging, the final canonical + held-out audit, and —
in evolver mode — the streamed report protocol agentic GAR runs on
(claims never trusted; kept improvements re-scored by the driver)."""
import json
import os
import subprocess
import sys
import threading

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
         "--agent-timeout", "120", "--final-holdout"],
        capture_output=True, text=True, cwd=REPO)
    assert proc.returncode == 0, proc.stderr

    # experiment one improves and is kept; two ties and is reverted
    results = [json.loads(l) for l in open(run / "driver_results.jsonl")]
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


def test_solo_worker_streams_reports_to_a_campaign(tmp_path):
    """As an evolver: Decoder.md enters the prompt, and every
    experiment streams a report — trusted scores only from the driver's
    own scorer reruns, worker claims left pending."""
    from finch4.agentic import Campaign
    from finch4.serve import serve

    srv = serve(str(tmp_path / "campaign"), port=0, tasks=["binpack"])
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    line = srv.service.handle("found", {
        "task": "binpack", "rationale": "worker integration test"})

    decoder = tmp_path / "Decoder.md"
    decoder.write_text("Established: fuller feasible bins first "
                       "(best-fit) beats emptier-first.")
    baseline = tmp_path / "baseline.py"
    baseline.write_text("def priority(item, capacities):\n"
                        "    return capacities\n")
    run = tmp_path / "worker"
    proc = subprocess.run(
        [sys.executable, "-m", "finch4.solo",
         "--run", str(run), "--task", "binpack",
         "--tasks-dir", TASKS_DIR,
         "--agent-cmd", f"{sys.executable} {FAKE} {{promptfile}}",
         "--experiments", "2", "--baseline", str(baseline),
         "--agent-timeout", "120",
         "--decoder", str(decoder), "--idea", "exploit the size bands",
         "--lineage", line["id"],
         "--serve-port", str(srv.server_address[1])],
        capture_output=True, text=True, cwd=REPO)
    assert proc.returncode == 0, proc.stderr
    srv.shutdown()

    # fitness, shared decoder, and the seeded idea reached the worker
    prompt = (run / "experiment_01.md").read_text()
    assert "Your fitness is" in prompt
    assert "fuller feasible bins" in prompt
    assert "exploit the size bands" in prompt
    assert (run / "Decoder.md").read_text() == decoder.read_text()
    for leak in ("--practice", "--sealed", "cells.json",
                 "never pass", "never read", "Do not open"):
        assert leak not in prompt

    campaign = Campaign.load(tmp_path / "campaign" / "state.json")
    reports = campaign.lineages[line["id"]]["reports"]
    # baseline + every experiment; even reverted candidates get trusted
    # driver fitness, while campaign holdout remains closed.
    assert len(reports) == 3
    trusted = [r for r in reports if r["score"] is not None]
    assert len(trusted) == 3
    for report in trusted:
        assert "canonical scorer" in report["source"]
        assert report["artifact_sha256"]
        assert os.path.isfile(report["artifact"])
    # the reverted experiment carries trusted fitness plus the worker claim
    reverted = [r for r in reports if r["kept"] is False]
    assert len(reverted) == 1 and reverted[0]["score"] is not None
    assert reverted[0]["claimed_score"] is not None
    audit = json.load(open(run / "final_audit.json"))
    assert audit["holdout_status"] == "UNTOUCHED"
    assert "final_holdout" not in audit
    assert campaign.best_summary()["binpack"] == max(
        r["score"] for r in trusted)


def test_solo_ingests_leftover_claim_before_starting(tmp_path):
    """A worker result line is not an ingest. If the driver died after
    the agent wrote results.jsonl, resume must score that candidate."""
    baseline = tmp_path / "baseline.py"
    baseline.write_text("def priority(item, capacities):\n"
                        "    return capacities\n")
    run = tmp_path / "solo"
    run.mkdir()
    (run / "champion.py").write_text(baseline.read_text(), encoding="utf-8")
    (run / "candidate.py").write_text(
        "def priority(item, capacities):\n    return -capacities\n",
        encoding="utf-8")
    (run / "results.jsonl").write_text(json.dumps({
        "experiment": 1,
        "hypothesis": "leftover best-fit",
        "claimed_score": 0.99,
    }) + "\n", encoding="utf-8")
    (run / "WORKLOG.md").write_text("# leftover\n", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "finch4.solo",
         "--run", str(run), "--task", "binpack",
         "--tasks-dir", TASKS_DIR,
         "--agent-cmd", f"{sys.executable} {FAKE} {{promptfile}}",
         "--experiments", "1", "--baseline", str(baseline),
         "--agent-timeout", "30"],
        capture_output=True, text=True, cwd=REPO)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    rows = [json.loads(line) for line in open(run / "driver_results.jsonl")]
    assert len(rows) == 1
    assert rows[0]["kept"] is True
    assert rows[0]["hypothesis"] == "leftover best-fit"
    assert "return -capacities" in (run / "champion.py").read_text()


def test_solo_records_holdout_without_using_it_to_keep(tmp_path):
    """Sealed tasks score holdout every experiment. A worse holdout
    cannot block a practice gain, and a better holdout cannot keep a
    practice tie."""
    tasks = tmp_path / "tasks" / "toy"
    tasks.mkdir(parents=True)
    (tasks / "cells.json").write_text("{}", encoding="utf-8")
    (tasks / "TASK.md").write_text("toy\n", encoding="utf-8")
    (tasks / "train.py").write_text("VALUE = 0\n", encoding="utf-8")
    (tasks / "score.py").write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        "text = Path(sys.argv[1]).read_text()\n"
        "practice = 0.9 if 'GAIN' in text else 0.4\n"
        "if '--holdout' in sys.argv:\n"
        "    print(json.dumps({'score': 0.05, 'split': 'holdout'}))\n"
        "else:\n"
        "    print(json.dumps({'score': practice, 'split': 'practice'}))\n",
        encoding="utf-8")
    agent = tmp_path / "agent.py"
    agent.write_text(
        "from pathlib import Path\n"
        "import json, re, sys\n"
        "prompt = Path(sys.argv[1]).read_text()\n"
        "n = int(re.search(r'experiment (\\d+) of', prompt).group(1))\n"
        "Path('candidate.py').write_text('GAIN = 1\\n')\n"
        "Path('results.jsonl').open('a').write(json.dumps({\n"
        "    'experiment': n, 'hypothesis': 'gain', 'claimed_score': 0.9,\n"
        "}) + '\\n')\n",
        encoding="utf-8")
    run = tmp_path / "solo"
    proc = subprocess.run(
        [sys.executable, "-m", "finch4.solo",
         "--run", str(run), "--task", "toy",
         "--tasks-dir", str(tmp_path / "tasks"),
         "--agent-cmd", f"{sys.executable} {agent} {{promptfile}}",
         "--experiments", "1",
         "--agent-timeout", "30"],
        capture_output=True, text=True, cwd=REPO)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    rows = [json.loads(line) for line in open(run / "driver_results.jsonl")]
    assert rows[0]["kept"] is True
    assert rows[0]["score"] == 0.9
    assert rows[0]["holdout_score"] == 0.05
    audit = json.load(open(run / "final_audit.json"))
    assert audit["holdout_status"] == "OBSERVED"
