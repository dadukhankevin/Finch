"""The Campaign record — agent communication and Elo laws checked without
spawning anything. Finch decides no winner; these tests pin what it must
refuse to record loosely."""
import json

import pytest

from finch4 import Campaign
from finch4.agentic import (ELO_INITIAL, SCHEMA_VERSION,
                            parse_report_citations, report_citation)


DIGEST = "a" * 64


def verified_report(campaign, lineage, score=1.0, label="gain"):
    report = campaign.report(
        lineage, label, score=score, source="evaluator",
        artifact=f"immutable://{lineage}/{label}",
        artifact_sha256=DIGEST)
    campaign.audit(lineage, report["index"], "passed", "reproduced")
    return report


def test_found_kinds_default_by_shape_and_log_decisions():
    c = Campaign(["alpha", "beta"])
    plain = c.found("alpha", "cover the obvious first angle")
    seeded = c.found("alpha", "eureka while reading a paper",
                     idea="try suffix arrays")
    assert plain["kind"] == "found" and seeded["kind"] == "inject"
    a = verified_report(c, plain["id"], 1.0, "plain")
    b = verified_report(c, seeded["id"], 1.1, "seeded")
    cross = c.found("alpha", "their mechanisms are complementary",
                    parents=[[plain["id"], a["index"]],
                             [seeded["id"], b["index"]]])
    assert cross["kind"] == "crossover"
    assert [d["action"] for d in c.decisions] == [
        "found", "inject", "share", "audit", "share", "audit", "crossover"]
    with pytest.raises(ValueError):
        c.found("alpha", "crossover needs two parents",
                parents=[[plain["id"], a["index"]]], kind="crossover")
    with pytest.raises(ValueError):
        c.found("nope", "unknown task")


def test_trusted_reports_append_themselves_to_decoder():
    c = Campaign(["alpha"])
    line = c.found("alpha", "seed")
    report = c.report(
        line["id"], "lookup beat the length band", score=0.7,
        source="evaluator", artifact="immutable://gain",
        artifact_sha256=DIGEST, kept=True)
    token = c.citation(line["id"], report["index"])
    text = c.current_decoder()["artifact"]
    assert token in text
    assert "KEPT 0.70000" in text
    assert "lookup beat the length band" in text
    assert report["decoder_version"] == c.decoder_version


def test_every_decision_requires_a_rationale():
    c = Campaign(["alpha"])
    line = c.found("alpha", "first lineage")
    with pytest.raises(ValueError):
        c.found("alpha", "")
    with pytest.raises(ValueError):
        c.kill(line["id"], "   ")
    with pytest.raises(ValueError):
        c.note(None)


def test_scores_are_trusted_only_with_a_source():
    c = Campaign(["alpha"])
    line = c.found("alpha", "seed")
    with pytest.raises(ValueError):
        c.report(line["id"], "tried a thing", score=1.0)   # no source
    c.report(line["id"], "worker claims a breakthrough", claimed_score=99.0)
    assert c.best["alpha"] is None            # claims never enter selection
    assert c.summary()["untrusted_reports"] == 1
    c.report(line["id"], "evaluator confirmed a real gain", score=1.5,
             source="protected evaluator")
    assert c.best["alpha"]["score"] == 1.5


def test_kill_blocks_reports_and_revive_restores():
    c = Campaign(["alpha"])
    line = c.found("alpha", "seed")
    c.kill(line["id"], "plateaued for two hours", evidence={"reports": 9})
    with pytest.raises(ValueError):
        c.report(line["id"], "posthumous experiment")
    c.revive(line["id"], "new decoder version changes its odds")
    c.report(line["id"], "back to work")
    assert c.lineages[line["id"]]["status"] == "running"
    c.assign(line["id"], "worker-2", "rotate the compute slot")
    assert c.lineages[line["id"]]["worker"] == "worker-2"


def test_incorporate_requires_scored_and_audited_cites():
    c = Campaign(["alpha"])
    line = c.found("alpha", "seed")
    c.report(line["id"], "unscored claim", claimed_score=5.0)
    scored = c.report(
        line["id"], "verified gain", score=2.0, source="evaluator",
        artifact="immutable://gain", artifact_sha256=DIGEST)
    with pytest.raises(ValueError):        # no cites at all
        c.incorporate("Decoder v1", "great vibes")
    with pytest.raises(ValueError):        # cite without a trusted score
        c.incorporate(
            f"Decoder v1 [L0000#0@{DIGEST[:8]}]", "absorb the claim")
    c.compact("# Decoder\n", "strip the auto-shared finding")
    with pytest.raises(ValueError):        # scored but not audited
        c.incorporate(
            f"Decoder v1 {c.citation(line['id'], 1)}", "absorb the gain")
    c.audit(line["id"], scored["index"], "passed",
            "reproduced the evaluator run exactly",
            evidence={"rerun": 2.0})
    token = c.citation(line["id"], scored["index"])
    record = c.incorporate(
        f"# Decoder v1\n\nVerified mechanism {token}.\n",
        "absorb the verified mechanism")
    assert record["version"] == 3 and c.decoder_version == 3
    assert parse_report_citations(record["artifact"])[0]["token"] == token
    assert record["citation_levels"] == {"L0000#1": "audited"}
    assert "cites" not in record
    other = c.found("alpha", "second lineage")
    other_report = c.report(
        other["id"], "gain", score=3.0, source="evaluator",
        artifact="immutable://other", artifact_sha256=DIGEST)
    c.audit(other["id"], other_report["index"], "failed",
            "did not reproduce")
    failed = c.incorporate(
        f"Decoder v2 {c.citation(other['id'], 0)}", "absorb it anyway")
    assert failed["citation_levels"][f"{other['id']}#0"] == "scored"


def test_share_keeps_scored_findings_separate_from_audited_methods():
    c = Campaign(["alpha"])
    negative = c.found("alpha", "test a plausible failure")
    report = c.report(
        negative["id"], "trusted negative result", score=-1.0,
        source="evaluator", artifact="immutable://negative",
        artifact_sha256=DIGEST)
    token = c.citation(negative["id"], report["index"])

    with pytest.raises(ValueError):
        c.share("finding without provenance", "remember it")
    shared = c.share(
        f"# Findings\n\nThis direction failed {token}.\n",
        "prevent rediscovery")
    assert shared["kind"] == "share"
    assert shared["citation_levels"] == {"L0000#0": "scored"}
    assert c.decisions[-1]["action"] == "share"

    # A later audited method can join the same readable file without
    # pretending that the retained negative finding passed audit.
    positive = c.found("alpha", "test a method")
    good = c.report(
        positive["id"], "trusted positive result", score=2.0,
        source="evaluator", artifact="immutable://positive",
        artifact_sha256="b" * 64)
    c.audit(positive["id"], good["index"], "passed", "reproduced")
    good_token = c.citation(positive["id"], good["index"])
    published = c.incorporate(
        shared["artifact"] + f"\n# Method\n\nUse it {good_token}.\n",
        "promote the audited method")
    assert published["citation_levels"] == {
        "L0000#0": "scored", "L0001#0": "audited"}

    compacted = c.compact(
        f"Negative {token}. Positive {good_token}.", "make it concise")
    assert compacted["citation_levels"] == published["citation_levels"]

    # Revising an audited source blocks carrying the method forward, while
    # the explicitly scored finding remains legal scientific memory.
    c.revise_score(positive["id"], good["index"], .5, "correction")
    with pytest.raises(ValueError):
        c.compact(f"Negative {token}. Positive {good_token}.",
                  "cannot retain invalid method")
    final = c.compact(f"Negative {token}.", "remove invalid method")
    assert final["citation_levels"] == {"L0000#0": "scored"}


def test_parallel_workers_can_append_plain_markdown_findings_without_rewrite():
    c = Campaign(["alpha"], decoder="# Shared research\n")
    lines = [c.found("alpha", f"test direction {index}")
             for index in range(2)]
    reports = [
        c.report(
            line["id"], f"result {index}", score=float(index - 1),
            source="evaluator", artifact=f"immutable://{index}",
            artifact_sha256=character * 64)
        for index, (line, character) in enumerate(zip(lines, ("a", "b")))
    ]
    fragments = [
        f"### Result {index}\n\nAgent-authored finding "
        f"{c.citation(line['id'], report['index'])}."
        for index, (line, report) in enumerate(zip(lines, reports))
    ]

    first = c.append_finding(fragments[0], "publish the first result")
    second = c.append_finding(fragments[1], "publish the second result")

    assert first["version"] == 3 and second["version"] == 4
    assert all(fragment in second["artifact"] for fragment in fragments)
    assert second["citation_levels"] == {
        "L0000#0": "scored", "L0001#0": "scored"}
    retried = c.append_finding(fragments[1], "durable retry")
    assert retried["version"] == 4 and retried["idempotent"] is True
    assert len(c.decoder_history) == 5


def test_stale_tracks_decoder_versions_and_compact_bumps():
    c = Campaign(["alpha"])
    a = c.found("alpha", "seed a")
    b = c.found("alpha", "seed b")
    c.report(a["id"], "gain", score=1.0, source="evaluator",
             artifact="immutable://a", artifact_sha256=DIGEST)
    c.audit(a["id"], 0, "passed", "reproduced")
    c.incorporate(
        f"Decoder v1 {c.citation(a['id'], 0)}", "absorb")
    assert c.stale() == [a["id"]]          # scored before v1; b never scored
    c.report(a["id"], "re-scored under v1", score=1.1, source="evaluator")
    assert c.stale() == []
    c.compact("Decoder v1, tighter", "context bloat")
    assert c.decoder_version == 3
    assert c.current_decoder()["kind"] == "compact"
    assert parse_report_citations(c.current_decoder()["artifact"]) == []


def test_revise_score_evicts_a_false_best():
    c = Campaign(["alpha"])
    cheat = c.found("alpha", "seed")
    honest = c.found("alpha", "seed two")
    c.report(cheat["id"], "big number", score=1000.0, source="evaluator")
    c.report(honest["id"], "small honest number", score=5.0,
             source="evaluator")
    assert c.best["alpha"]["lineage"] == cheat["id"]
    c.revise_score(cheat["id"], 0, -99.0, "audit rerun",
                   rationale="audit exposed answer-key embedding")
    assert c.best["alpha"]["lineage"] == honest["id"]
    history = c.lineages[cheat["id"]]["reports"][0]["score_history"]
    assert history[0]["score"] == 1000.0   # the lie stays on the record


def test_fitness_regime_preserves_trajectory_but_resets_selection_state():
    c = Campaign(["alpha"])
    first = c.found("alpha", "try a mechanism")
    old = c.report(
        first["id"], "small-panel result", score=4.0,
        source="12-item screen", artifact="immutable://old",
        artifact_sha256="a" * 64)
    second = c.found("alpha", "independent mechanism")
    c.report(
        second["id"], "other small-panel result", score=2.0,
        source="12-item screen", artifact="immutable://other",
        artifact_sha256="b" * 64)
    c.set_judge_criteria("alpha", {"metric": "chrf"}, "freeze criteria")
    match = c.pair(
        [[first["id"], 0], [second["id"], 0]], "compare old evidence",
        requested_by="allocator")
    c.verdict(match["id"], first["id"], "judge", "first is stronger")

    changed = c.start_fitness_regime(
        "alpha", "large-96-v1", "replace the noisy panel",
        evidence={"items": 96})

    assert changed["previous"] == "initial"
    assert c.best["alpha"] is None
    assert c.lineages[first["id"]]["best_report"] is None
    assert c.lineages[first["id"]]["regime_seed_report"] == old["index"]
    assert c.lineages[first["id"]]["reports"][0]["fitness_regime"] == "initial"
    assert c.lineages[first["id"]]["elo"] == 1500
    with pytest.raises(ValueError, match="active fitness regime"):
        c.pair([[first["id"], 0], [second["id"], 0]], "stale pair",
               requested_by="allocator")

    new = c.report(
        first["id"], "large-panel result", score=0.8,
        source="96-item screen", artifact="immutable://new",
        artifact_sha256="c" * 64)
    assert new["fitness_regime"] == "large-96-v1"
    assert c.best["alpha"] == {
        "score": 0.8, "lineage": first["id"], "report": 1}


def test_audit_is_report_specific_and_revision_invalidates_it():
    c = Campaign(["alpha"])
    line = c.found("alpha", "seed")
    first = verified_report(c, line["id"], 1.0, "first")
    later = c.report(
        line["id"], "later", score=2.0, source="evaluator",
        artifact="immutable://later", artifact_sha256=DIGEST)
    later_pub = c.incorporate(
        f"decoder {c.citation(line['id'], later['index'])}",
        "later report was never audited")
    assert later_pub["citation_levels"][
        f"{line['id']}#{later['index']}"] == "scored"
    c.revise_score(line["id"], first["index"], .5, "correction")
    assert c.lineages[line["id"]]["reports"][0]["audit_status"] == "none"
    with pytest.raises(ValueError):
        c.incorporate(
            f"decoder {c.citation(line['id'], first['index'])}",
            "revision invalidated its audit")
    with pytest.raises(KeyError):
        c.audit(line["id"], -1, "passed", "negative indices are not ids")


def test_crossover_can_breed_scored_failures_but_keeps_audit_context():
    c = Campaign(["alpha"])
    a = c.found("alpha", "a")
    b = c.found("alpha", "b")
    ra = c.report(
        a["id"], "a result", score=1, source="eval",
        artifact="immutable://a", artifact_sha256=DIGEST)
    c.audit(a["id"], ra["index"], "failed", "poor transfer",
            evidence={"weak_cell": "cipher"})
    rb = verified_report(c, b["id"], 2, "b")
    child = c.found("alpha", "combine a's useful part with b's guard",
                    parents=[[a["id"], ra["index"]],
                             [b["id"], rb["index"]]])
    assert child["parents"][0]["audit_status"] == "failed"
    assert child["parents"][0]["audit"]["rationale"] == "poor transfer"
    assert child["parents"][0]["audit"]["reasons"] == []
    assert "evidence" not in child["parents"][0]["audit"]
    assert child["parents"][1]["audit_status"] == "passed"

    unscored = c.found("alpha", "unscored")
    ru = c.report(unscored["id"], "claim only", claimed_score=99,
                  artifact="immutable://u", artifact_sha256=DIGEST)
    with pytest.raises(ValueError):
        c.found("alpha", "untrusted parent", parents=[
            [unscored["id"], ru["index"]], [b["id"], rb["index"]]])


def test_markdown_citations_drive_natural_crossover_tree():
    c = Campaign(["alpha"], decoder="# Shared genome\n")
    source = c.found("alpha", "try a coverage ledger")
    checkpoint = verified_report(c, source["id"], 2.0, "coverage")
    token = report_citation(source["id"], checkpoint["index"], DIGEST)
    assert token == c.citation(source["id"], checkpoint["index"])

    adopter = c.found("alpha", "try a contrastive planner")
    child = c.report(
        adopter["id"], f"Extended the coverage gate {token} with contrast.",
        score=2.2, source="evaluator", artifact="immutable://contrast",
        artifact_sha256="b" * 64)
    c.incorporate(
        f"# Shared genome\n\nUse the verified coverage gate {token}.\n",
        "distill the audited gain")
    inheritor = c.found("alpha", "inherit and make a creative next step")

    tree = c.tree_of_life()
    tuples = {(edge["source"], edge["target"], edge["kind"])
              for edge in tree["edges"]}
    assert (f"{source['id']}#0", f"{adopter['id']}#{child['index']}",
            "inspiration") in tuples
    assert (f"{source['id']}#0", "D1", "distillation") in tuples
    assert ("D3", f"{inheritor['id']}:born", "inheritance") in tuples
    assert tree["counts"]["inspirations"] == 1
    assert tree["unresolved"] == []


def test_citation_hash_and_shape_are_checked_without_extra_schema():
    c = Campaign(["alpha"])
    source = c.found("alpha", "source")
    verified_report(c, source["id"], 1.0, "source")
    target = c.found("alpha", "target")
    with pytest.raises(ValueError, match="does not match"):
        c.report(target["id"], "Borrowed [L0000#0@bbbbbbbb].")
    with pytest.raises(ValueError, match="malformed report citation"):
        c.report(target["id"], "Borrowed [L0000#0].")


def test_rejected_decisions_do_not_mutate_state():
    c = Campaign(["alpha"])
    line = c.found("alpha", "seed")
    with pytest.raises(ValueError):
        c.kill(line["id"], " ")
    assert c.lineages[line["id"]]["status"] == "running"


def test_experiment_ids_are_idempotent_and_void_is_append_only():
    c = Campaign(["alpha"])
    line = c.found("alpha", "seed")
    first = c.report(
        line["id"], "first", score=2, source="eval",
        artifact="immutable://first", artifact_sha256=DIGEST,
        experiment_id="experiment-0001")
    with pytest.raises(ValueError):
        c.report(line["id"], "retry", score=2, source="eval",
                 experiment_id="experiment-0001")
    c.void_report(line["id"], first["index"], "duplicate controller post")
    assert c.lineages[line["id"]]["reports"][0]["voided"] is True
    assert c.best["alpha"] is None
    assert c.summary()["reports"] == 0
    assert c.summary()["recorded_reports"] == 1
    assert c.summary()["voided_reports"] == 1
    with pytest.raises(ValueError):
        c.audit(line["id"], 0, "passed", "cannot restore influence")


def test_rejected_report_is_history_not_lineage_champion():
    c = Campaign(["alpha"])
    line = c.found("alpha", "seed")
    rejected = c.report(
        line["id"], "lost to inherited checkpoint", score=9,
        source="eval", kept=False,
        artifact="immutable://rejected", artifact_sha256=DIGEST)
    assert rejected["score"] == 9
    assert c.lineages[line["id"]]["best_score"] is None
    assert c.best["alpha"] is None
    accepted = c.report(
        line["id"], "first accepted child", score=8,
        source="eval", kept=True,
        artifact="immutable://accepted", artifact_sha256=DIGEST)
    assert c.lineages[line["id"]]["best_report"] == accepted["index"]


def test_save_load_roundtrip_and_schema_gate(tmp_path):
    c = Campaign(["alpha", "beta"])
    line = c.found("alpha", "seed", idea="angle")
    c.report(line["id"], "gain", score=1.0, source="evaluator",
             evidence={"cells": [1, 2]})
    c.kill(line["id"], "done")
    path = tmp_path / "state.json"
    c.save(path)
    loaded = Campaign.load(path)
    assert loaded.summary() == c.summary()
    assert loaded.decisions == c.decisions
    assert loaded.lineages == c.lineages

    state = json.loads(path.read_text())
    state["schema_version"] = SCHEMA_VERSION - 1
    old = tmp_path / "old.json"
    old.write_text(json.dumps(state))
    with pytest.raises(ValueError):
        Campaign.load(old)                 # archived protocols stay archived


def test_engine_protocol_for_dashboards():
    c = Campaign(["alpha", "beta"])
    line = c.found("beta", "seed")
    c.report(line["id"], "gain", score=2.5, source="evaluator")
    assert c.best_summary() == {"beta": 2.5}
    assert c.best_record()["lineage"] == line["id"]
    empty = Campaign(["quiet"])
    assert empty.best_summary() == {} and empty.best_record() is None


def test_campaign_evidence_tiers_and_plateau_are_agent_records():
    c = Campaign(["alpha"])
    c.set_baseline("alpha", 10, "canonical control",
                   "freeze the comparison before replication",
                   evidence={"seeds": [1, 2, 3]})
    line = c.found("alpha", "screen a distinct mechanism")
    lucky = c.report(line["id"], "screening leader", score=20,
                     source="screen", artifact="immutable://lucky",
                     artifact_sha256=DIGEST)
    assert c.summary()["screening_best"]["alpha"] == 20
    assert c.summary()["verified_champion"]["alpha"] is None
    with pytest.raises(ValueError):
        c.set_champion(line["id"], lucky["index"], "matched replicates",
                       "singleton is not verified")
    c.audit(line["id"], lucky["index"], "passed", "artifact reproduced")
    with pytest.raises(ValueError):
        c.set_champion(line["id"], lucky["index"], "matched replicates",
                       "replication must leave evidence")
    champion = c.set_champion(
        line["id"], lucky["index"], "matched replicate evaluator",
        "candidate beat the frozen control distribution",
        evidence={"candidate": [18, 19, 20], "control": [9, 10, 11]})
    assert champion["score"] == 20
    plateau = c.set_plateau(
        "alpha", True, "no verified improvement in the last allocation",
        evidence={"experiments_since_champion": 12})
    summary = c.summary()
    assert summary["baseline"] == {"alpha": 10.0}
    assert summary["verified_champion"] == {"alpha": 20.0}
    assert summary["plateau_alert"] is True
    assert summary["plateau"]["alpha"] == plateau
    c.set_plateau("alpha", False, "new hypothesis reopened the search")
    assert c.summary()["plateau_alert"] is False


def test_judge_agent_verdict_drives_elo_while_metric_stays_evidence():
    c = Campaign(["translation"])
    c.set_judge_criteria(
        "translation",
        {"primary": "paired chrF", "also_consider": ["robustness"]},
        "freeze the comparison contract before tournament play")
    first = c.found("translation", "lexical model")
    second = c.found("translation", "coverage model")
    a = c.report(
        first["id"], "lexical checkpoint", score=41.0, source="chrF",
        evidence={"languages_won": 3}, artifact="immutable://lexical",
        artifact_sha256="a" * 64)
    b = c.report(
        second["id"], "coverage checkpoint", score=42.0, source="chrF",
        evidence={"languages_won": 6}, artifact="immutable://coverage",
        artifact_sha256="b" * 64)

    match = c.pair(
        [[first["id"], a["index"]], [second["id"], b["index"]]],
        "compare the current organisms", requested_by="allocator-1",
        judge="judge-7")
    assignment = c.match_assignment(match["id"])
    assert len(assignment["individuals"]) == 2
    assert assignment["criteria"]["primary"] == "paired chrF"
    assert assignment["individuals"][1]["score"] == 42.0
    assert "elo" not in assignment and "rating" not in assignment

    decided = c.verdict(
        match["id"], second["id"], "judge-7",
        "the second wins on chrF and six-of-eight language robustness",
        evidence={"metric_delta": 1.0}, confidence=.9)
    assert decided["verdict"]["winner"] == second["id"]
    standings = {row["lineage"]: row for row in c.ratings()}
    assert standings[first["id"]]["rating"] == pytest.approx(1484)
    assert standings[second["id"]]["rating"] == pytest.approx(1516)
    assert standings[second["id"]]["wins"] == 1
    assert c.summary()["elo_leader"]["translation"]["lineage"] == second["id"]
    assert c.summary()["best"]["translation"] == 42.0
    assert c.summary()["verdicts"] == 1


def test_only_assigned_judge_can_decide_and_a_match_is_single_use():
    c = Campaign(["alpha"])
    c.set_judge_criteria("alpha", "Prefer the stronger verified result.",
                         "one criterion for every match")
    lines = [c.found("alpha", f"angle {i}") for i in range(2)]
    for i, line in enumerate(lines):
        c.report(line["id"], f"checkpoint {i}",
                 artifact=f"immutable://{i}",
                 artifact_sha256=chr(ord("a") + i) * 64)
    match = c.pair([[lines[0]["id"], 0], [lines[1]["id"], 0]],
                   "schedule a blind comparison", "allocator",
                   judge="judge-a")
    with pytest.raises(ValueError, match="assigned"):
        c.verdict(match["id"], "tie", "judge-b", "looks equal")
    c.verdict(match["id"], "tie", "judge-a", "no material difference")
    with pytest.raises(ValueError, match="already"):
        c.verdict(match["id"], lines[0]["id"], "judge-a", "try again")
    assert all(row["rating"] == ELO_INITIAL for row in c.ratings())
    assert all(row["ties"] == 1 for row in c.ratings())

    rejected_line = c.found("alpha", "promising trajectory with a loss")
    c.report(rejected_line["id"], "did not become the local champion",
             kept=False, artifact="immutable://rejected",
             artifact_sha256="c" * 64)
    trajectory_match = c.pair(
        [[lines[0]["id"], 0], [rejected_line["id"], 0]],
        "judge trajectory promise despite the reverted candidate", "allocator")
    rejected_view = trajectory_match["individuals"][1]
    assert rejected_view["kept"] is False
    assert rejected_view["trajectory"][0]["kept"] is False


def test_void_match_replays_later_elo_without_the_bad_verdict():
    c = Campaign(["alpha"])
    c.set_judge_criteria("alpha", "Choose the better artifact.",
                         "freeze judging")
    lines = [c.found("alpha", f"angle {i}") for i in range(3)]
    for i, line in enumerate(lines):
        c.report(line["id"], f"checkpoint {i}", score=float(i), source="eval",
                 artifact=f"immutable://{i}",
                 artifact_sha256=chr(ord("a") + i) * 64)
    first = c.pair([[lines[0]["id"], 0], [lines[1]["id"], 0]],
                   "first match", "allocator")
    c.verdict(first["id"], lines[1]["id"], "judge-1", "second wins")
    later = c.pair([[lines[1]["id"], 0], [lines[2]["id"], 0]],
                   "later match", "allocator")
    c.verdict(later["id"], lines[1]["id"], "judge-2", "first wins")

    c.void_match(first["id"], "judge disclosed the labels were reversed")
    standings = {row["lineage"]: row for row in c.ratings()}
    assert standings[lines[0]["id"]]["rating"] == ELO_INITIAL
    assert standings[lines[1]["id"]]["rating"] == pytest.approx(1516)
    assert standings[lines[2]["id"]]["rating"] == pytest.approx(1484)
    assert c.matches[1]["elo_update"]["before"][lines[1]["id"]] == ELO_INITIAL


def test_research_payload_resolves_shared_citations_for_evolvers():
    c = Campaign(["alpha"])
    source = c.found("alpha", "discover a mechanism")
    report = verified_report(c, source["id"], 2.0, "coverage gate")
    token = c.citation(source["id"], report["index"])
    c.incorporate(f"# Shared research\n\nUse coverage {token}.\n",
                  "share the verified method")
    target = c.found("alpha", "advance the shared method")
    payload = c.research(target["id"])
    assert payload["decoder"]["version"] == 2
    assert payload["sources"][0]["citation"] == token
    assert payload["sources"][0]["summary"] == "coverage gate"
    assert payload["lineage"]["id"] == target["id"]
