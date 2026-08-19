"""Task-specific checkpoint membrane for the statistical-translation GAR."""

from pathlib import Path


def checkpoint_eligible(report):
    """Whether a report may become a complete inherited organism."""
    evidence = report.get("evidence") or {}
    gate = evidence.get("composition_eligibility") or {}
    return gate.get("passed") is True


def keep_decision(previous, score, eligible):
    improved = previous is None or score > previous
    kept = bool(eligible and improved)
    if not eligible:
        reason = "failed compositional feasibility gate"
    elif not improved:
        reason = "did not improve the inherited eligible checkpoint"
    else:
        reason = "improved and passed compositional feasibility"
    return kept, improved, reason


def seed_for(campaign, line, starter):
    """Choose a complete seed without discarding ineligible gene donors."""
    own = [
        report for report in line["reports"]
        if report.get("score") is not None
        and report.get("artifact")
        and not report.get("voided")
        and report.get("kept") is not False
        and checkpoint_eligible(report)
    ]
    if own:
        report = max(own, key=lambda item: item["score"])
        return Path(report["artifact"]), report["score"]

    if line["parents"]:
        eligible = []
        for parent in line["parents"]:
            parent_line = campaign.lineages[parent["lineage"]]
            report = parent_line["reports"][parent["report"]]
            if checkpoint_eligible(report):
                eligible.append((parent, report))
        if eligible:
            # Audit-passed feasibility is the safest default organism. Failed
            # parents remain available separately for explicit gene use.
            parent, report = max(
                eligible,
                key=lambda item: (
                    item[1].get("audit_status") == "passed",
                    item[1]["score"]))
            return Path(parent["artifact"]), report["score"]

    # No complete viable organism exists. Keep every artifact in the record,
    # but start from the neutral scaffold.
    return Path(starter), None
