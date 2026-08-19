"""The task controller keeps benchmark exploits out of inherited defaults."""

from pathlib import Path
import sys

from finch4 import Campaign


TASK_DIR = (Path(__file__).resolve().parents[1] / "benchmarks" / "agentic" /
            "tasks" / "statistical_translation")
sys.path.insert(0, str(TASK_DIR))
import checkpoint_policy as policy  # noqa: E402


DIGEST = "b" * 64
TASK = "statistical_translation"


def _report(campaign, lineage, artifact, score, eligible):
    return campaign.report(
        lineage, "candidate", score=score, source="protected evaluator",
        evidence={"composition_eligibility": {"passed": eligible}},
        kept=True, artifact=str(artifact), artifact_sha256=DIGEST)


def test_composition_failure_cannot_be_kept_despite_higher_fitness():
    assert policy.keep_decision(20.0, 99.0, False) == (
        False, True, "failed compositional feasibility gate")
    assert policy.keep_decision(20.0, 21.0, True)[0] is True
    assert policy.keep_decision(20.0, 19.0, True)[0] is False


def test_seed_ignores_ineligible_checkpoint_already_in_history(tmp_path):
    campaign = Campaign([TASK])
    line = campaign.found(TASK, "test recovery")
    hack = tmp_path / "hack.py"
    viable = tmp_path / "viable.py"
    hack.write_text("# exploit\n", encoding="utf-8")
    viable.write_text("# mechanism\n", encoding="utf-8")
    _report(campaign, line["id"], hack, 100.0, False)
    _report(campaign, line["id"], viable, 10.0, True)

    seed, score = policy.seed_for(
        campaign, campaign.lineages[line["id"]], tmp_path / "starter.py")

    assert seed == viable
    assert score == 10.0


def test_crossover_prefers_eligible_audited_seed_and_keeps_donors(tmp_path):
    campaign = Campaign([TASK])
    refs = []
    specs = [
        ("hack", 100.0, False, "passed"),
        ("eligible_failed_audit", 8.0, True, "failed"),
        ("eligible_passed", 4.0, True, "passed"),
    ]
    artifacts = {}
    for name, score, eligible, audit in specs:
        line = campaign.found(TASK, f"found {name}")
        artifact = tmp_path / f"{name}.py"
        artifact.write_text(f"# {name}\n", encoding="utf-8")
        artifacts[name] = artifact
        report = _report(campaign, line["id"], artifact, score, eligible)
        campaign.audit(line["id"], report["index"], audit, f"audit {name}")
        refs.append([line["id"], report["index"]])

    child = campaign.found(
        TASK, "test disclosed donors", parents=refs)
    seed, score = policy.seed_for(
        campaign, campaign.lineages[child["id"]], tmp_path / "starter.py")

    assert seed == artifacts["eligible_passed"]
    assert score == 4.0
    assert len(child["parents"]) == 3
