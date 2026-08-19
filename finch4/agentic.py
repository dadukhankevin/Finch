"""Bookkeeping and agent communication for GAR (docs/high-agent.md).

Each lineage is a persistent autoresearch trajectory (hypothesize, edit,
run, keep or revert); workers are compute assignments that may rotate without
resetting that research state. The decoder is the LLM plus Decoder.md, the
shared cited research file through which natural crossover occurs.

Finch owns the record and the communication protocol:

- **Lineages and their report streams.** A lineage preserves one local
  champion and its research history; a report is one experiment (what was tried, the evaluator's
  number, kept or reverted). Reports arrive as they happen.
- **Trusted fitness.** A report's score is admitted only with a named
  source; a worker's own claim lives in `claimed_score` and is never
  promoted. This membrane is inherited from the previous protocol and
  is non-negotiable — the compress run's embedded-slice cheater is the
  standing receipt.
- **Pairwise selection.** Finch hands exactly two immutable checkpoints,
  fixed task criteria, and their evaluator evidence to a judge agent. The
  agent returns a winner and rationale; Finch records that verdict and
  performs only the mechanical Elo update. Finch never judges an artifact.
- **The decision log.** Founding, allocation, judging, culling, auditing,
  and shared-memory changes are recorded with their agent and rationale.
- **Decoder.md versions.** A trusted scored report with an artifact
  appends itself. Sharing is scientific memory: it may cite any
  trusted, immutable scored report, including a negative result.
  Incorporation is the training step, and every newly incorporated method
  cites a scored, audit-passed report
  (the arithmetic-fold incident is the receipt for why: plausible-but-
  unvetted absorption once collected two weeks of unearned credit).

Breeding and incorporation deliberately have different membranes. Any
non-void report with trusted fitness and an immutable artifact may be a
crossover parent; its audit outcome travels with the parent reference so
the child is briefed on failures rather than inheriting claims blindly.
Only an audit-passed report may enter Decoder.md as shared methodology.
Scored reports may enter as explicitly non-method scientific findings, so
workers do not repeatedly rediscover known failures.

The orchestrator may allocate work and request comparisons, but evolvers read
and report to Finch directly. That removes a central interpretive feedback
loop without giving Finch semantic authority. The tensor engine is unaffected.
"""
from __future__ import annotations

import json
import hashlib
import math
import os
import re

import numpy as np


SCHEMA_VERSION = 6
LINEAGE_KINDS = ("found", "inject", "crossover")
DECISION_ACTIONS = ("found", "kill", "revive", "crossover", "inject",
                    "assign", "audit", "void", "revise", "incorporate",
                    "share", "compact", "note", "baseline", "champion",
                    "plateau", "criteria", "pair", "verdict",
                    "void-match", "regime")
AUDIT_OUTCOMES = ("none", "passed", "failed", "inconclusive")
ELO_INITIAL = 1500.0
ELO_K = 32.0

# One citation grammar for both experiment reports and Decoder.md.  The
# readable portion locates the checkpoint; the hash prefix proves which
# immutable artifact the author actually meant.  These Markdown tokens are
# the agent-authored provenance record.  Everything else (including the Tree
# of Life dashboard) is derived by parsing them.
REPORT_CITATION_RE = re.compile(
    r"\[(L\d{4,})#(0|[1-9]\d*)@([0-9a-fA-F]{8,64})\]")
REPORT_CITATION_LIKE_RE = re.compile(r"\[L\d{4,}#\d+(?:@[^\]]*)?\]")


def _jsonable(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    raise TypeError(f"Campaign: {type(value).__name__} is not "
                    "JSON-serializable")


def _check_json(payload):
    json.dumps(payload, default=_jsonable, allow_nan=False)


def _require_rationale(rationale):
    if not str(rationale or "").strip():
        raise ValueError("every recorded decision needs a rationale")
    return str(rationale)


def _content_sha256(value):
    payload = value if isinstance(value, str) else json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        default=_jsonable, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def report_citation(lineage_id, report_index, artifact_sha256):
    """The canonical, human-writable identity of one report checkpoint."""
    digest = str(artifact_sha256 or "").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("a report citation requires a full artifact SHA-256")
    lineage_id = str(lineage_id)
    if not re.fullmatch(r"L\d{4,}", lineage_id):
        raise ValueError("a report citation requires a lineage like L0003")
    index = int(report_index)
    if index < 0:
        raise ValueError("a report citation index cannot be negative")
    return f"[{lineage_id}#{index}@{digest[:8]}]"


def parse_report_citations(markdown):
    """Parse exact report citations from Markdown, preserving first use."""
    text = str(markdown or "")
    parsed, seen = [], set()
    for match in REPORT_CITATION_RE.finditer(text):
        key = (match.group(1), int(match.group(2)), match.group(3).lower())
        if key in seen:
            continue
        seen.add(key)
        parsed.append({
            "token": match.group(0), "lineage": key[0],
            "report": key[1], "artifact_sha256_prefix": key[2],
        })
    return parsed


class Campaign:
    """Lineages, communication, judge matches, Elo, and shared research."""

    def __init__(self, tasks, decoder=""):
        if isinstance(tasks, str):
            tasks = [tasks]
        if not tasks:
            raise ValueError("at least one task is required")
        if len(set(tasks)) != len(tasks):
            raise ValueError("task names must be unique")
        self.schema_version = SCHEMA_VERSION
        self.tasks = list(tasks)
        self.lineages = {}
        self.decisions = []
        self.best = {task: None for task in self.tasks}
        # Fitness numbers are only comparable inside the measurement regime
        # that produced them.  A campaign may improve its evaluator without
        # throwing away its research trajectories; switching regimes resets
        # selection state while preserving every historical report.
        self.fitness_regimes = {task: "initial" for task in self.tasks}
        self.regime_history = {task: [{
            "name": "initial", "rationale": "campaign initialization",
            "evidence": None,
        }] for task in self.tasks}
        # Campaign-level evidence tiers are deliberately agent-authored
        # declarations, not conclusions inferred by Finch. ``best`` remains
        # the raw screening maximum; these records say what it was compared
        # with and which checkpoint survived the campaign's replication plan.
        self.baselines = {task: None for task in self.tasks}
        self.champions = {task: None for task in self.tasks}
        self.plateaus = {task: None for task in self.tasks}
        # Judge agents, not Finch, decide pairwise winners. Finch owns only
        # the fixed criteria, exact assignments, verdict record, and the
        # mechanical Elo arithmetic that follows a verdict.
        self.judge_criteria = {task: None for task in self.tasks}
        self.matches = []
        self.elo_k = ELO_K
        self.decoder_version = 0
        self.decoder_history = [{
            "version": 0, "artifact": str(decoder), "artifact_sha256":
            _content_sha256(str(decoder)), "rationale": None,
            "kind": "initial", "citation_levels": {},
        }]
        self._next_lineage = 0
        self._next_report = 0
        self._next_match = 0

    # ------------------------------------------------------------ helpers

    def _decision(self, action, lineage_ids, rationale, **extra):
        record = {
            "index": len(self.decisions), "action": action,
            "lineages": list(lineage_ids),
            "rationale": rationale,
            "decoder_version": self.decoder_version,
        }
        record.update(extra)
        _check_json(record)
        self.decisions.append(record)
        return record

    def _lineage(self, lineage_id):
        try:
            return self.lineages[lineage_id]
        except KeyError:
            raise KeyError(f"unknown lineage {lineage_id!r}")

    def _report(self, lineage_id, report_index):
        line = self._lineage(lineage_id)
        try:
            index = int(report_index)
            if index < 0:
                raise IndexError
            return line, line["reports"][index]
        except (IndexError, TypeError, ValueError):
            raise KeyError(f"unknown report {lineage_id!r}#"
                           f"{report_index!r}")

    def _resolve_report_refs(self, refs, *, require_audited=False,
                             require_artifact=False, require_score=True):
        resolved = []
        for ref in refs:
            if isinstance(ref, dict):
                lineage_id, report_index = ref.get("lineage"), ref.get(
                    "report")
            else:
                try:
                    lineage_id, report_index = ref
                except (TypeError, ValueError):
                    raise ValueError("report references are [lineage, report]"
                                     " pairs")
            line, record = self._report(lineage_id, report_index)
            if require_score and record["score"] is None:
                raise ValueError(f"report {lineage_id}#{report_index} has "
                                 "no trusted score")
            if record.get("voided"):
                raise ValueError(f"report {lineage_id}#{report_index} was "
                                 "voided")
            if require_audited and record["audit_status"] != "passed":
                raise ValueError(f"report {lineage_id}#{report_index} has "
                                 "not passed audit")
            if require_artifact and (not record.get("artifact")
                                     or not record.get("artifact_sha256")):
                raise ValueError(f"report {lineage_id}#{report_index} has "
                                 "no immutable artifact identity")
            audit = None
            if record["audit_history"]:
                latest = record["audit_history"][-1]
                evidence = latest.get("evidence") or {}
                audit = {
                    "outcome": latest.get("outcome"),
                    "rationale": latest.get("rationale"),
                    "reasons": evidence.get("reasons", []),
                    "paired_bootstrap": evidence.get("paired_bootstrap"),
                    "cell_deltas": evidence.get("cell_deltas"),
                }
            resolved.append({
                "lineage": lineage_id, "report": int(report_index),
                "task": line["task"], "score": record["score"],
                "fitness_regime": self._report_regime(
                    record, line["task"]),
                "source": record["source"],
                "artifact": record.get("artifact"),
                "artifact_sha256": record.get("artifact_sha256"),
                "audit_status": record["audit_status"],
                "audit": audit,
            })
        return resolved

    def resolve_citations(self, markdown, *, require_audited=False,
                          require_citations=False):
        """Resolve Markdown citations against immutable campaign reports.

        There is deliberately no parallel ``influences`` field.  Agents cite
        the checkpoint in the prose they already write; Finch checks those
        tokens and derives its graph from the prose later.
        """
        text = str(markdown or "")
        parsed = parse_report_citations(text)
        exact_tokens = {cite["token"] for cite in parsed}
        malformed = [match.group(0)
                     for match in REPORT_CITATION_LIKE_RE.finditer(text)
                     if match.group(0) not in exact_tokens]
        if malformed:
            raise ValueError(
                "malformed report citation; use "
                "[L0003#2@8ddf8b41], including the artifact hash: "
                + ", ".join(dict.fromkeys(malformed)))
        if require_citations and not parsed:
            raise ValueError(
                "Decoder.md publication cites its source reports inline "
                "as [L0003#2@8ddf8b41]")
        resolved = []
        for cite in parsed:
            refs = self._resolve_report_refs(
                [(cite["lineage"], cite["report"])],
                require_audited=require_audited, require_artifact=True)
            ref = refs[0]
            observed = str(ref["artifact_sha256"]).lower()
            prefix = cite["artifact_sha256_prefix"]
            if not observed.startswith(prefix):
                raise ValueError(
                    f"citation {cite['token']} does not match the immutable "
                    f"artifact for {cite['lineage']}#{cite['report']} "
                    f"({observed[:8]})")
            ref["token"] = cite["token"]
            resolved.append(ref)
        return resolved

    def citation(self, lineage_id, report_index):
        """Return the copyable Markdown citation for one checkpoint."""
        _, record = self._report(lineage_id, report_index)
        if not record.get("artifact_sha256"):
            raise ValueError(
                f"report {lineage_id}#{report_index} has no artifact identity")
        return report_citation(
            lineage_id, report_index, record["artifact_sha256"])

    @staticmethod
    def _lineage_audit_status(line):
        if line["best_report"] is None:
            return "none"
        return line["reports"][line["best_report"]]["audit_status"]

    def fitness_regime(self, task):
        """Return the active evaluator identity for ``task``."""
        if task not in self.tasks:
            raise ValueError(f"unknown task {task!r}")
        regimes = getattr(self, "fitness_regimes", {})
        return str(regimes.get(task, "initial"))

    def _report_regime(self, report, task):
        return str(report.get("fitness_regime", "initial"))

    def start_fitness_regime(self, task, name, rationale, evidence=None):
        """Begin a non-comparable fitness regime without erasing history.

        Existing trajectories remain available as starting material and as
        cited scientific history.  Their scores no longer participate in the
        active best/Elo tables.  Agents, rather than Finch, still decide what
        the evaluator change means and which lines remain worth pursuing.
        """
        if task not in self.tasks:
            raise ValueError(f"unknown task {task!r}")
        rationale = _require_rationale(rationale)
        name = str(name or "").strip()
        if not name:
            raise ValueError("a fitness regime needs a name")
        _check_json(evidence)
        if not hasattr(self, "fitness_regimes"):
            self.fitness_regimes = {item: "initial" for item in self.tasks}
        if not hasattr(self, "regime_history"):
            self.regime_history = {item: [{
                "name": self.fitness_regimes.get(item, "initial"),
                "rationale": "loaded historical campaign",
                "evidence": None,
            }] for item in self.tasks}
        previous = self.fitness_regime(task)
        if name == previous:
            raise ValueError(f"fitness regime {name!r} is already active")
        if any(row.get("name") == name
               for row in self.regime_history.setdefault(task, [])):
            raise ValueError(f"fitness regime {name!r} already exists")
        for line in self.lineages.values():
            if line["task"] != task:
                continue
            # Preserve the latest accepted organism as the initial genome for
            # the new measurement regime even though its old score is no
            # longer comparable.
            line["regime_seed_report"] = line.get("best_report")
        self.fitness_regimes[task] = name
        record = {"name": name, "previous": previous,
                  "rationale": rationale, "evidence": evidence}
        self.regime_history[task].append(record)
        self.baselines[task] = None
        self.champions[task] = None
        self.plateaus[task] = None
        for line in self.lineages.values():
            if line["task"] == task:
                self._rebuild_lineage(line)
        self._rebuild_best(task)
        self._rebuild_elo()
        self._decision("regime", [], rationale, task=task, name=name,
                       previous=previous, evidence=evidence)
        return dict(record)

    def living(self):
        return [line for line in self.lineages.values()
                if line["status"] == "running"]

    # ----------------------------------------------------- campaign signals

    def set_baseline(self, task, score, source, rationale, evidence=None):
        """Record the trusted comparison point chosen by the campaign."""
        if task not in self.tasks:
            raise ValueError(f"unknown task {task!r}")
        rationale = _require_rationale(rationale)
        score = float(score)
        if not math.isfinite(score):
            raise ValueError("baseline score must be finite")
        if not str(source or "").strip():
            raise ValueError("a baseline names its fitness source")
        _check_json(evidence)
        record = {"task": task, "score": score, "source": str(source),
                  "fitness_regime": self.fitness_regime(task),
                  "evidence": evidence}
        self.baselines[task] = record
        self._decision("baseline", [], rationale, **record)
        return dict(record)

    def set_champion(self, lineage_id, report_index, source, rationale,
                     evidence=None):
        """Designate a replicated/verified campaign champion.

        Finch checks checkpoint integrity and the existing audit membrane;
        the verifying agent remains responsible for deciding whether supplied
        replication evidence is adequate for this campaign.
        """
        rationale = _require_rationale(rationale)
        line, report = self._report(lineage_id, report_index)
        if self._report_regime(report, line["task"]) != self.fitness_regime(
                line["task"]):
            raise ValueError("a champion must belong to the active fitness regime")
        if report.get("voided") or report.get("kept") is False:
            raise ValueError("a champion must be an accepted report")
        if report.get("score") is None or report["audit_status"] != "passed":
            raise ValueError("a champion needs trusted fitness and a passed audit")
        if not report.get("artifact_sha256"):
            raise ValueError("a champion needs an immutable artifact identity")
        if not str(source or "").strip():
            raise ValueError("champion verification names its source")
        if evidence is None:
            raise ValueError("champion verification records replication evidence")
        _check_json(evidence)
        record = {"task": line["task"], "lineage": lineage_id,
                  "report": int(report_index), "score": report["score"],
                  "fitness_regime": self.fitness_regime(line["task"]),
                  "artifact_sha256": report["artifact_sha256"],
                  "source": str(source), "evidence": evidence}
        self.champions[line["task"]] = record
        self._decision("champion", [lineage_id], rationale, **record)
        return dict(record)

    def set_plateau(self, task, plateaued, rationale, evidence=None):
        """Publish an agent-authored plateau judgment for dashboard alerting."""
        if task not in self.tasks:
            raise ValueError(f"unknown task {task!r}")
        rationale = _require_rationale(rationale)
        if not isinstance(plateaued, bool):
            raise ValueError("plateaued must be boolean")
        _check_json(evidence)
        record = {"task": task, "plateaued": plateaued,
                  "evidence": evidence, "rationale": rationale}
        self.plateaus[task] = record
        self._decision("plateau", [], rationale, task=task,
                       plateaued=plateaued, evidence=evidence)
        return dict(record)

    # ------------------------------------------------ pairwise selection

    def set_judge_criteria(self, task, criteria, rationale):
        """Freeze what judge agents should value for one task.

        Criteria may name an automatic metric such as chrF and any broader
        robustness or research-trajectory considerations. Finch stores and
        forwards the criteria; it never applies or interprets them.
        """
        if task not in self.tasks:
            raise ValueError(f"unknown task {task!r}")
        rationale = _require_rationale(rationale)
        if criteria is None or (isinstance(criteria, str)
                                and not criteria.strip()):
            raise ValueError("judge criteria cannot be empty")
        _check_json(criteria)
        record = {"task": task, "criteria": criteria}
        self.judge_criteria[task] = record
        self._decision("criteria", [], rationale, **record)
        return dict(record)

    def _individual(self, ref):
        resolved = self._resolve_report_refs(
            [ref], require_artifact=True, require_score=False)[0]
        line, report = self._report(
            resolved["lineage"], resolved["report"])
        trajectory = []
        for prior in line["reports"][:report["index"] + 1]:
            citation = (None if not prior.get("artifact_sha256") else
                        report_citation(line["id"], prior["index"],
                                        prior["artifact_sha256"]))
            trajectory.append({
                "report": prior["index"], "citation": citation,
                "summary": prior["summary"], "score": prior["score"],
                "source": prior["source"], "evidence": prior.get("evidence"),
                "kept": prior.get("kept"), "voided": prior.get("voided"),
                "audit_status": prior["audit_status"],
                "artifact": prior.get("artifact"),
                "artifact_sha256": prior.get("artifact_sha256"),
            })
        resolved.update({
            "citation": self.citation(line["id"], report["index"]),
            "summary": report["summary"],
            "evidence": report.get("evidence"),
            "kept": report.get("kept"),
            "decoder_version": report.get("decoder_version", 0),
            "lineage_status": line["status"],
            "lineage_idea": line.get("idea"),
            "trajectory": trajectory,
        })
        return resolved

    def _match(self, match_id):
        if isinstance(match_id, str) and re.fullmatch(r"M\d{4,}", match_id):
            index = int(match_id[1:])
        else:
            try:
                index = int(match_id)
            except (TypeError, ValueError):
                raise KeyError(f"unknown match {match_id!r}")
        if index < 0 or index >= len(self.matches):
            raise KeyError(f"unknown match {match_id!r}")
        match = self.matches[index]
        if match["id"] != f"M{index:04d}":
            raise RuntimeError("match record order is corrupt")
        return match

    def pair(self, individuals, rationale, requested_by, judge=None):
        """Create one two-individual assignment for an external judge.

        The individuals are exact report checkpoints. A task's criteria and
        both checkpoints' evaluator evidence travel with the assignment, but
        population ratings do not: the judge sees the pair, not the table.
        """
        rationale = _require_rationale(rationale)
        requested_by = str(requested_by or "").strip()
        if not requested_by:
            raise ValueError("a pairing names the agent or process requesting it")
        refs = list(individuals or ())
        if len(refs) != 2:
            raise ValueError("a judge pairing contains exactly two individuals")
        pair = [self._individual(ref) for ref in refs]
        if pair[0]["lineage"] == pair[1]["lineage"]:
            raise ValueError("an Elo match compares two different lineages")
        if pair[0]["task"] != pair[1]["task"]:
            raise ValueError("an Elo match cannot cross tasks")
        task = pair[0]["task"]
        active_regime = self.fitness_regime(task)
        if any(item.get("fitness_regime") != active_regime for item in pair):
            raise ValueError(
                "an Elo match uses checkpoints from the active fitness regime")
        criteria = self.judge_criteria.get(task)
        if criteria is None:
            raise ValueError(f"set judge criteria for {task!r} before pairing")
        if judge is not None:
            judge = str(judge).strip()
            if not judge:
                raise ValueError("assigned judge cannot be blank")
        match_id = f"M{self._next_match:04d}"
        record = {
            "id": match_id, "index": self._next_match, "task": task,
            "fitness_regime": self.fitness_regime(task),
            "status": "open", "individuals": pair,
            "criteria": criteria["criteria"],
            "requested_by": requested_by, "judge": judge,
            "rationale": rationale, "verdict": None,
            "elo_update": None, "voided": False,
        }
        _check_json(record)
        self._next_match += 1
        self.matches.append(record)
        self._decision(
            "pair", [item["lineage"] for item in pair], rationale,
            match=match_id, task=task, requested_by=requested_by, judge=judge,
            individuals=[item["citation"] for item in pair])
        return dict(record)

    def match_assignment(self, match_id):
        """Return the deliberately narrow payload handed to one judge."""
        match = self._match(match_id)
        return {
            "id": match["id"], "task": match["task"],
            "fitness_regime": str(match.get("fitness_regime", "initial")),
            "status": match["status"], "criteria": match["criteria"],
            "individuals": match["individuals"], "judge": match["judge"],
            "verdict": match["verdict"], "voided": match["voided"],
        }

    @staticmethod
    def _expected_score(first_rating, second_rating):
        return 1.0 / (1.0 + 10.0 ** (
            (second_rating - first_rating) / 400.0))

    def _apply_match_elo(self, match):
        first_id = match["individuals"][0]["lineage"]
        second_id = match["individuals"][1]["lineage"]
        first, second = self._lineage(first_id), self._lineage(second_id)
        before_first, before_second = first["elo"], second["elo"]
        expected_first = self._expected_score(before_first, before_second)
        outcome = match["verdict"]["outcome"]
        actual_first = {"first": 1.0, "second": 0.0, "tie": 0.5}[outcome]
        delta = self.elo_k * (actual_first - expected_first)
        first["elo"] = before_first + delta
        second["elo"] = before_second - delta
        first["elo_games"] += 1
        second["elo_games"] += 1
        if outcome == "first":
            first["elo_wins"] += 1
            second["elo_losses"] += 1
        elif outcome == "second":
            second["elo_wins"] += 1
            first["elo_losses"] += 1
        else:
            first["elo_ties"] += 1
            second["elo_ties"] += 1
        update = {
            "k": self.elo_k,
            "before": {first_id: before_first, second_id: before_second},
            "after": {first_id: first["elo"], second_id: second["elo"]},
            "delta": {first_id: delta, second_id: -delta},
            "expected": {first_id: expected_first,
                         second_id: 1.0 - expected_first},
        }
        second_result = {"first": "loss", "second": "win",
                         "tie": "tie"}[outcome]
        first_result = {"first": "win", "second": "loss",
                        "tie": "tie"}[outcome]
        first["elo_history"].append({
            "match": match["id"], "opponent": second_id,
            "result": first_result, "before": before_first,
            "after": first["elo"], "delta": delta})
        second["elo_history"].append({
            "match": match["id"], "opponent": first_id,
            "result": second_result, "before": before_second,
            "after": second["elo"], "delta": -delta})
        match["elo_update"] = update
        return update

    def verdict(self, match_id, winner, judge, rationale, evidence=None,
                confidence=None):
        """Record a judge agent's decision, then update Elo mechanically."""
        match = self._match(match_id)
        if match["voided"]:
            raise ValueError("cannot judge a voided match")
        if str(match.get("fitness_regime", "initial")) != self.fitness_regime(
                match["task"]):
            raise ValueError("cannot judge a match from an inactive fitness regime")
        if match["status"] != "open":
            raise ValueError(f"match {match['id']} already has a verdict")
        judge = str(judge or "").strip()
        if not judge:
            raise ValueError("a verdict names the judge agent")
        if match["judge"] is not None and judge != match["judge"]:
            raise ValueError(f"match {match['id']} is assigned to {match['judge']!r}")
        rationale = _require_rationale(rationale)
        first_id = match["individuals"][0]["lineage"]
        second_id = match["individuals"][1]["lineage"]
        if winner == "tie":
            outcome, winner_id = "tie", None
        elif winner == first_id:
            outcome, winner_id = "first", first_id
        elif winner == second_id:
            outcome, winner_id = "second", second_id
        else:
            raise ValueError(
                f"winner must be {first_id!r}, {second_id!r}, or 'tie'")
        if confidence is not None:
            confidence = float(confidence)
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("confidence must be between 0 and 1")
        _check_json(evidence)
        verdict = {
            "judge": judge, "winner": winner_id, "outcome": outcome,
            "rationale": rationale, "evidence": evidence,
            "confidence": confidence,
        }
        _check_json(verdict)
        match["verdict"] = verdict
        match["status"] = "decided"
        update = self._apply_match_elo(match)
        self._decision(
            "verdict", [first_id, second_id], rationale,
            match=match["id"], judge=judge, winner=winner_id,
            outcome=outcome, elo_update=update)
        return dict(match)

    def _rebuild_elo(self):
        for line in self.lineages.values():
            line.update({
                "elo": ELO_INITIAL, "elo_games": 0, "elo_wins": 0,
                "elo_losses": 0, "elo_ties": 0, "elo_history": [],
            })
        for match in self.matches:
            if (match["status"] == "decided" and not match["voided"]
                    and str(match.get("fitness_regime", "initial"))
                    == self.fitness_regime(match["task"])):
                self._apply_match_elo(match)

    def void_match(self, match_id, rationale):
        """Invalidate a bad assignment/verdict and replay later Elo updates."""
        match = self._match(match_id)
        if match["voided"]:
            raise ValueError("match is already voided")
        rationale = _require_rationale(rationale)
        match["voided"] = True
        match["status"] = "voided"
        self._rebuild_elo()
        lineages = [item["lineage"] for item in match["individuals"]]
        self._decision("void-match", lineages, rationale, match=match["id"])
        return dict(match)

    def ratings(self, task=None):
        """Return Elo standings; this is selection state, not a judgment."""
        if task is not None and task not in self.tasks:
            raise ValueError(f"unknown task {task!r}")
        lines = [line for line in self.lineages.values()
                 if task is None or line["task"] == task]
        lines.sort(key=lambda line: (-line["elo"], -line["elo_games"],
                                    line["id"]))
        task_rank = {}
        out = []
        for line in lines:
            task_rank[line["task"]] = task_rank.get(line["task"], 0) + 1
            out.append({
                "rank": task_rank[line["task"]], "task": line["task"],
                "lineage": line["id"], "rating": line["elo"],
                "games": line["elo_games"], "wins": line["elo_wins"],
                "losses": line["elo_losses"], "ties": line["elo_ties"],
                "best_report": line["best_report"],
                "best_score": line["best_score"], "status": line["status"],
            })
        return out

    def research(self, lineage_id=None):
        """The shared cited memory an evolver reads without an orchestrator."""
        decoder = self.current_decoder()
        sources = []
        for cite in parse_report_citations(decoder["artifact"]):
            line, report = self._report(cite["lineage"], cite["report"])
            sources.append({
                "citation": cite["token"], "lineage": line["id"],
                "report": report["index"], "summary": report["summary"],
                "score": report["score"], "source": report["source"],
                "evidence": report.get("evidence"),
                "audit_status": report["audit_status"],
                "artifact": report.get("artifact"),
                "artifact_sha256": report.get("artifact_sha256"),
            })
        payload = {"decoder": decoder, "sources": sources}
        if lineage_id is not None:
            line = self._lineage(lineage_id)
            payload["lineage"] = dict(line)
        return payload

    # ------------------------------------------------------------ founding

    def found(self, task, rationale, idea=None, parents=(), worker=None,
              kind=None):
        """Record a new lineage an allocator or evolver decided to start.

        kind defaults by shape: plain founding without parents is
        "found"; founding with a seeded idea and no parents is the
        eureka valve ("inject"); founding with parent lineages is a
        crossover an agent judged worth briefing ("crossover")."""
        if task not in self.tasks:
            raise ValueError(f"unknown task {task!r}")
        rationale = _require_rationale(rationale)
        parents = list(parents)
        if kind is None:
            kind = ("crossover" if parents
                    else "inject" if idea is not None else "found")
        if kind not in LINEAGE_KINDS:
            raise ValueError(f"kind must be one of {list(LINEAGE_KINDS)}")
        if kind == "crossover" and len(parents) < 2:
            raise ValueError("a crossover lineage records >= 2 parents")
        if kind != "crossover" and parents:
            raise ValueError("only crossover lineages have parents")
        # Crossover is a sandboxed hypothesis, not shared truth. Trusted,
        # immutable parents may breed even after a failed/inconclusive audit;
        # the audit outcome is carried in parent_refs. The child still earns
        # its own fitness. Incorporation keeps the stricter audit membrane.
        parent_refs = self._resolve_report_refs(
            parents, require_audited=False, require_artifact=True)
        _check_json(idea)
        lineage_id = f"L{self._next_lineage:04d}"
        self._next_lineage += 1
        self.lineages[lineage_id] = {
            "id": lineage_id, "task": task, "kind": kind, "idea": idea,
            "parents": parent_refs,
            "worker": None if worker is None else str(worker),
            "status": "running", "reports": [],
            "best_score": None, "best_report": None,
            "elo": ELO_INITIAL, "elo_games": 0,
            "elo_wins": 0, "elo_losses": 0, "elo_ties": 0,
            "elo_history": [],
            "decoder_version_at_found": self.decoder_version,
            "founding_decision": len(self.decisions),
        }
        self._decision(kind, [lineage_id], rationale, task=task,
                       parents=parent_refs, idea=idea)
        return dict(self.lineages[lineage_id])

    # ------------------------------------------------------------- reports

    def report(self, lineage_id, summary, score=None, source=None,
               evidence=None, kept=None, claimed_score=None, artifact=None,
               artifact_sha256=None, experiment_id=None, sealed_score=None,
               holdout_score=None):
        """One experiment from a worker's autoresearch loop, streamed as
        it completes. `score` is search fitness (makers may see those
        cells). `sealed_score` is held-in fitness the dashboard may
        treat as trusted for keep/best. `holdout_score` is observation
        only: graphed, never used for keep, best, or Elo. A worker's
        own number goes in claimed_score and is never promoted."""
        line = self._lineage(lineage_id)
        if line["status"] != "running":
            raise ValueError(f"{lineage_id} is {line['status']}; revive it "
                             "before reporting")
        if not str(summary or "").strip():
            raise ValueError("a report needs a summary of what was tried")
        # A citation is optional, but if the worker declares intellectual
        # inheritance it must name a real, immutable earlier checkpoint.
        self.resolve_citations(summary)
        if experiment_id is not None:
            experiment_id = str(experiment_id).strip()
            if not experiment_id:
                raise ValueError("experiment_id cannot be blank")
            if any(report.get("experiment_id") == experiment_id
                   for report in line["reports"]):
                raise ValueError(f"duplicate experiment_id {experiment_id!r} "
                                 f"for {lineage_id}")
        if score is not None:
            score = float(score)
            if not math.isfinite(score):
                raise ValueError("score must be finite")
            if not str(source or "").strip():
                raise ValueError("a scored report names its fitness source")
        if claimed_score is not None:
            claimed_score = float(claimed_score)
            if not math.isfinite(claimed_score):
                raise ValueError("claimed_score must be finite")
        if sealed_score is not None:
            sealed_score = float(sealed_score)
            if not math.isfinite(sealed_score):
                raise ValueError("sealed_score must be finite")
            if not str(source or "").strip():
                raise ValueError("a sealed score names its fitness source")
        if holdout_score is not None:
            holdout_score = float(holdout_score)
            if not math.isfinite(holdout_score):
                raise ValueError("holdout_score must be finite")
            if not str(source or "").strip():
                raise ValueError("a holdout score names its fitness source")
        if artifact is None and artifact_sha256 is not None:
            raise ValueError("artifact_sha256 requires an artifact")
        if artifact is not None:
            if not isinstance(artifact_sha256, str) or len(
                    artifact_sha256) != 64:
                raise ValueError("an artifact requires its SHA-256 identity")
            try:
                int(artifact_sha256, 16)
            except ValueError:
                raise ValueError("artifact_sha256 must be hexadecimal")
        record = {
            "index": len(line["reports"]),
            "arrival": self._next_report,
            "lineage": lineage_id,
            "summary": str(summary),
            "score": score,
            "sealed_score": sealed_score,
            "holdout_score": holdout_score,
            "source": None if source is None else str(source),
            "evidence": evidence, "kept": kept,
            "claimed_score": claimed_score, "artifact": artifact,
            "artifact_sha256": artifact_sha256,
            "experiment_id": experiment_id, "voided": False,
            "fitness_regime": self.fitness_regime(line["task"]),
            "decoder_version": self.decoder_version,
            "audit_status": "none", "audit_history": [],
        }
        _check_json(record)
        self._next_report += 1
        line["reports"].append(record)
        if score is not None and artifact_sha256:
            token = report_citation(
                lineage_id, record["index"], artifact_sha256)
            if kept is True:
                verdict = "KEPT"
            elif kept is False:
                verdict = "REVERTED"
            else:
                verdict = "SCORED"
            self.append_finding(
                f"{token} {verdict} {score:.5f}. {str(summary).strip()}",
                "scored report")
            record["decoder_version"] = self.decoder_version
        if score is not None and kept is not False and (
                line["best_score"] is None or score > line["best_score"]):
            line["best_score"] = score
            line["best_report"] = record["index"]
        self._rebuild_best(line["task"])
        return dict(record)

    def _rebuild_best(self, task):
        candidates = [
            (line["best_score"], line["id"], line["best_report"])
            for line in self.lineages.values()
            if line["task"] == task and line["best_score"] is not None]
        if not candidates:
            self.best[task] = None
            return
        score, lineage_id, report_index = max(candidates)
        self.best[task] = {"score": score, "lineage": lineage_id,
                           "report": report_index}

    def _rebuild_lineage(self, line):
        active = self.fitness_regime(line["task"])
        best = max((record for record in line["reports"]
                    if record["score"] is not None
                    and not record.get("voided")
                    and record.get("kept") is not False
                    and self._report_regime(record, line["task"]) == active),
                   key=lambda record: record["score"], default=None)
        line["best_score"] = None if best is None else best["score"]
        line["best_report"] = None if best is None else best["index"]

    def revise_score(self, lineage_id, report_index, score, source,
                     evidence=None, rationale="correction"):
        """Correct a report's trusted score. An audit exposing a false
        number must be able to evict a best-ever — the previous value
        stays in the report's score_history."""
        rationale = _require_rationale(rationale)
        line, record = self._report(lineage_id, report_index)
        if record.get("voided"):
            raise ValueError("cannot revise a voided report")
        score = float(score)
        if not math.isfinite(score):
            raise ValueError("score must be finite")
        if not str(source or "").strip():
            raise ValueError("a revised score names its fitness source")
        _check_json(evidence)
        previous_audit = record["audit_status"]
        history = list(record.get("score_history", []))
        history.append({"score": record["score"],
                        "source": record["source"],
                        "evidence": record["evidence"]})
        _check_json(history)
        record["score_history"] = history
        record["score"], record["source"] = score, str(source)
        record["evidence"] = evidence
        record["audit_status"] = "none"
        self._rebuild_lineage(line)
        self._rebuild_best(line["task"])
        self._decision("revise", [lineage_id],
                       rationale,
                       report=int(report_index), score=score,
                       source=str(source),
                       invalidated_audit=previous_audit)

    def set_sealed_score(self, lineage_id, report_index, sealed_score,
                         source, rationale="sealed rescore"):
        """Attach allocator-only sealed fitness. Does not change search
        score or keep/revert."""
        rationale = _require_rationale(rationale)
        line, record = self._report(lineage_id, report_index)
        if record.get("voided"):
            raise ValueError("cannot seal a voided report")
        sealed_score = float(sealed_score)
        if not math.isfinite(sealed_score):
            raise ValueError("sealed_score must be finite")
        if not str(source or "").strip():
            raise ValueError("a sealed score names its fitness source")
        record["sealed_score"] = sealed_score
        evidence = dict(record.get("evidence") or {})
        evidence["sealed_score"] = sealed_score
        record["evidence"] = evidence
        self._decision("seal", [lineage_id], rationale,
                       report=int(report_index), sealed_score=sealed_score,
                       source=str(source))
        return dict(record)

    def set_holdout_score(self, lineage_id, report_index, holdout_score,
                          source, rationale="holdout observation"):
        """Attach allocator-only holdout fitness. Observation only:
        does not change search score, sealed score, keep/revert, best,
        or Elo."""
        rationale = _require_rationale(rationale)
        line, record = self._report(lineage_id, report_index)
        if record.get("voided"):
            raise ValueError("cannot attach holdout to a voided report")
        holdout_score = float(holdout_score)
        if not math.isfinite(holdout_score):
            raise ValueError("holdout_score must be finite")
        if not str(source or "").strip():
            raise ValueError("a holdout score names its fitness source")
        record["holdout_score"] = holdout_score
        self._decision("holdout", [lineage_id], rationale,
                       report=int(report_index), holdout_score=holdout_score,
                       source=str(source))
        return dict(record)

    # ----------------------------------------------------------- decisions

    def kill(self, lineage_id, rationale, evidence=None):
        rationale = _require_rationale(rationale)
        _check_json(evidence)
        line = self._lineage(lineage_id)
        line["status"] = "killed"
        return self._decision("kill", [lineage_id], rationale,
                              evidence=evidence)

    def revive(self, lineage_id, rationale):
        rationale = _require_rationale(rationale)
        line = self._lineage(lineage_id)
        line["status"] = "running"
        return self._decision("revive", [lineage_id], rationale)

    def assign(self, lineage_id, worker, rationale):
        """Record which orchestrator-managed worker currently owns a line."""
        rationale = _require_rationale(rationale)
        worker = str(worker or "").strip()
        if not worker:
            raise ValueError("worker cannot be blank")
        line = self._lineage(lineage_id)
        previous = line.get("worker")
        line["worker"] = worker
        return self._decision(
            "assign", [lineage_id], rationale, worker=worker,
            previous_worker=previous)

    def audit(self, lineage_id, report_index, outcome, rationale,
              evidence=None):
        """Audit one exact scored report/artifact checkpoint."""
        rationale = _require_rationale(rationale)
        if outcome not in AUDIT_OUTCOMES[1:]:
            raise ValueError(f"outcome must be one of "
                             f"{list(AUDIT_OUTCOMES[1:])}")
        _check_json(evidence)
        _, record = self._report(lineage_id, report_index)
        if record.get("voided"):
            raise ValueError("cannot audit a voided report")
        if record["score"] is None:
            raise ValueError("cannot audit a report without trusted fitness")
        if not record.get("artifact") or not record.get("artifact_sha256"):
            raise ValueError("cannot audit a report without an immutable "
                             "artifact identity")
        history = list(record["audit_history"])
        history.append({
            "outcome": outcome, "rationale": rationale,
            "evidence": evidence,
            "decoder_version": self.decoder_version,
            "artifact_sha256": record.get("artifact_sha256"),
        })
        _check_json(history)
        record["audit_status"] = outcome
        record["audit_history"] = history
        return self._decision("audit", [lineage_id],
                              rationale, report=int(report_index),
                              outcome=outcome, evidence=evidence,
                              artifact_sha256=record.get("artifact_sha256"))

    def void_report(self, lineage_id, report_index, rationale):
        """Append-only correction for a duplicate or invalid experiment.

        The report remains in history but stops contributing to fitness,
        audits, curves, or influence.
        """
        rationale = _require_rationale(rationale)
        line, record = self._report(lineage_id, report_index)
        if record.get("voided"):
            raise ValueError("report is already voided")
        record["voided"] = True
        record["audit_status"] = "none"
        self._rebuild_lineage(line)
        self._rebuild_best(line["task"])
        return self._decision(
            "void", [lineage_id], rationale, report=int(report_index),
            previous_score=record.get("score"),
            artifact_sha256=record.get("artifact_sha256"))

    def note(self, rationale, lineage_ids=()):
        """A recorded observation that is not any other action."""
        for lineage_id in lineage_ids:
            self._lineage(lineage_id)
        return self._decision("note", list(lineage_ids),
                              _require_rationale(rationale))

    # -------------------------------------------------------- the decoder

    @staticmethod
    def _citation_key(lineage, report):
        return f"{lineage}#{int(report)}"

    def _decoder_levels(self, record=None):
        """Return evidence levels for citations in one Decoder version.

        Records written before scored findings were supported contain no
        map.  Those citations necessarily entered through the historical
        audit-only incorporate path, so ``audited`` is the safe migration.
        """
        record = self.current_decoder() if record is None else record
        levels = dict(record.get("citation_levels") or {})
        for cite in parse_report_citations(record.get("artifact", "")):
            levels.setdefault(
                self._citation_key(cite["lineage"], cite["report"]),
                "audited")
        return levels

    def _publish_decoder(self, artifact, rationale, kind):
        """Publish one mixed-evidence Decoder.md without a parallel schema.

        Existing citations retain the membrane through which they entered.
        ``share`` may add trusted scored findings; ``incorporate`` may add
        only audit-passed methodology; ``compact`` may add nothing.  This
        keeps one readable shared file while preventing a negative finding
        from silently becoming an inherited method.
        """
        rationale = _require_rationale(rationale)
        _check_json(artifact)
        require_citations = kind in ("share", "incorporate")
        resolved = self.resolve_citations(
            artifact, require_citations=require_citations)
        before_record = self.current_decoder()
        before = {
            (cite["lineage"], cite["report"])
            for cite in parse_report_citations(before_record["artifact"])}
        after = {(cite["lineage"], cite["report"]): cite
                 for cite in resolved}
        introduced = set(after) - before
        if kind == "compact" and introduced:
            refs = ", ".join(f"{lineage}#{report}"
                             for lineage, report in sorted(introduced))
            raise ValueError(
                f"compact cannot introduce new sources ({refs}); use "
                "share for a scored finding or incorporate for an audited "
                "method")

        previous_levels = self._decoder_levels(before_record)
        levels = {}
        for lineage, report in after:
            key = self._citation_key(lineage, report)
            if (lineage, report) in introduced:
                level = "scored" if kind == "share" else "audited"
            else:
                level = previous_levels.get(key, "audited")
                if kind == "incorporate" and level != "audited":
                    try:
                        self._resolve_report_refs(
                            [(lineage, report)], require_audited=True,
                            require_artifact=True)
                    except ValueError:
                        pass
                    else:
                        level = "audited"
            self._resolve_report_refs(
                [(lineage, report)], require_audited=(level == "audited"),
                require_artifact=True)
            levels[key] = level

        artifact_sha256 = _content_sha256(artifact)
        self.decoder_version += 1
        record = {"version": self.decoder_version, "artifact": artifact,
                  "artifact_sha256": artifact_sha256,
                  "rationale": rationale, "kind": kind,
                  "citation_levels": levels}
        self.decoder_history.append(record)
        lineages = ([] if kind == "compact" else
                    sorted({lineage for lineage, _ in introduced}))
        self._decision(kind, lineages, rationale,
                       version=self.decoder_version)
        return dict(record)

    def share(self, artifact, rationale):
        """Publish scored scientific memory, including negative results.

        A shared finding must cite trusted fitness and an immutable artifact,
        but need not claim generalization.  It is evidence for future agents,
        not inherited methodology.
        """
        return self._publish_decoder(artifact, rationale, "share")

    def append_finding(self, fragment, rationale):
        """Atomically append one agent-authored finding to shared research.

        Parallel evolvers should not race by reading, rewriting, and replacing
        the whole ``Decoder.md`` file.  They instead submit an ordinary
        Markdown fragment containing the same inline report citations used
        everywhere else.  Finch validates provenance and serializes the
        append; it does not summarize, classify, or otherwise interpret the
        finding.

        Re-submitting the exact fragment is idempotent.  This matters when a
        durable worker finishes publication but crashes before recording its
        own success.
        """
        text = str(fragment or "").strip()
        if not text:
            raise ValueError("a shared finding cannot be empty")
        self.resolve_citations(text, require_citations=True)
        current = self.current_decoder()
        before = str(current["artifact"])
        # Match the complete submitted fragment, including multi-paragraph
        # findings, without treating an incidental sentence prefix as a
        # completed publication.
        if f"\n\n{text}\n\n" in f"\n\n{before.strip()}\n\n":
            record = dict(current)
            record["idempotent"] = True
            return record
        artifact = ((before.rstrip() + "\n\n") if before.strip() else "")
        artifact += text + "\n"
        return self._publish_decoder(artifact, rationale, "share")

    def incorporate(self, artifact, rationale):
        """Publish audit-passed shared methodology — the training step.

        Existing scored findings may remain in the same Markdown file.  Every
        newly introduced citation must have trusted fitness, an immutable
        artifact, and a passed audit.
        """
        return self._publish_decoder(artifact, rationale, "incorporate")

    def compact(self, artifact, rationale):
        """Rewrite Decoder.md for density without introducing sources.

        Existing inline citations may be retained or removed with their
        claims, but a compact operation cannot introduce a new report. Use
        share for a scored finding or incorporate for audited methodology.

        Each retained citation is re-checked against the evidence membrane
        through which it entered. If a score revision invalidated an audited
        method, compacting is blocked until that claim is removed or
        re-audited; scored findings retain their explicitly weaker status.
        """
        return self._publish_decoder(artifact, rationale, "compact")

    def current_decoder(self):
        return dict(self.decoder_history[-1])

    # ---------------------------------------------------------- reporting

    def tree_of_life(self):
        """Derive the campaign's inspectable evolutionary graph.

        No lineage edges are authored in a second schema.  Ordinary descent
        comes from the report stream, directed recombination from the
        founding checkpoints, and emergent crossover from citations parsed
        out of report prose and Decoder.md versions.
        """
        nodes, edges, unresolved = [], [], []
        node_ids = set()

        def add_node(node):
            if node["id"] not in node_ids:
                node_ids.add(node["id"])
                nodes.append(node)

        def add_edge(source, target, kind, **extra):
            if source not in node_ids or target not in node_ids:
                return
            edge = {"id": f"{kind}:{source}>{target}:{len(edges)}",
                    "source": source, "target": target, "kind": kind}
            edge.update(extra)
            edges.append(edge)

        def report_node_id(lineage_id, report_index):
            return f"{lineage_id}#{int(report_index)}"

        def checked_source(cite, where):
            try:
                _, report = self._report(cite["lineage"], cite["report"])
            except KeyError as error:
                unresolved.append({"where": where, "token": cite["token"],
                                   "reason": str(error)})
                return None
            observed = str(report.get("artifact_sha256") or "").lower()
            prefix = cite["artifact_sha256_prefix"]
            if not observed.startswith(prefix):
                unresolved.append({
                    "where": where, "token": cite["token"],
                    "reason": ("artifact hash mismatch; recorded "
                               f"{observed[:8] or 'none'}"),
                })
                return None
            return report_node_id(cite["lineage"], cite["report"])

        all_reports = [report for line in self.lineages.values()
                       for report in line["reports"]]
        last_moment = max((report["arrival"] + 1 for report in all_reports),
                          default=1)

        # Lineage births and experiment checkpoints.  A report is the useful
        # biological unit here: it is the exact organism that can be cited.
        for line in sorted(self.lineages.values(), key=lambda item: item["id"]):
            reports = line["reports"]
            first_moment = (reports[0]["arrival"] + .55
                            if reports else last_moment + 1)
            decision = self.decisions[line["founding_decision"]]
            born_id = f"{line['id']}:born"
            add_node({
                "id": born_id, "kind": "lineage", "label": line["id"],
                "lineage": line["id"], "task": line["task"],
                "moment": first_moment, "status": line["status"],
                "lineage_kind": line["kind"], "worker": line.get("worker"),
                "elo": line.get("elo", ELO_INITIAL),
                "elo_games": line.get("elo_games", 0),
                "decoder_version": line["decoder_version_at_found"],
                "idea": line.get("idea"),
                "summary": decision.get("rationale", ""),
                "score": line.get("best_score"),
            })
            previous = born_id
            for report in reports:
                node_id = report_node_id(line["id"], report["index"])
                citation = None
                if report.get("artifact_sha256"):
                    citation = report_citation(
                        line["id"], report["index"],
                        report["artifact_sha256"])
                parsed = parse_report_citations(report["summary"])
                add_node({
                    "id": node_id, "kind": "report", "label": node_id,
                    "lineage": line["id"], "task": line["task"],
                    "report": report["index"],
                    "moment": report["arrival"] + 1,
                    "arrival": report["arrival"], "score": report["score"],
                    "source": report.get("source"),
                    "kept": report.get("kept"),
                    "voided": bool(report.get("voided")),
                    "audit": report.get("audit_status", "none"),
                    "decoder_version": report.get("decoder_version", 0),
                    "summary": report["summary"], "citation": citation,
                    "artifact_sha256": report.get("artifact_sha256"),
                    "citations": [cite["token"] for cite in parsed],
                })
                add_edge(previous, node_id, "descent")
                previous = node_id

        # Explicit agent-directed recombination remains available, but
        # is only one edge type rather than the definition of crossover.
        for line in self.lineages.values():
            target = f"{line['id']}:born"
            for parent in line.get("parents", []):
                source = report_node_id(parent["lineage"], parent["report"])
                add_edge(source, target, "directed", token=report_citation(
                    parent["lineage"], parent["report"],
                    parent["artifact_sha256"]),
                    audit=parent.get("audit_status"), score=parent.get("score"))

        # Citations in experiment prose are declared intellectual descent:
        # the natural, emergent crossover that happens when workers combine
        # discoveries they encountered in the shared scientific record.
        for line in self.lineages.values():
            for report in line["reports"]:
                target = report_node_id(line["id"], report["index"])
                for cite in parse_report_citations(report["summary"]):
                    source = checked_source(cite, target)
                    if source:
                        add_edge(source, target, "inspiration",
                                 token=cite["token"])

        # Decoder versions are shared-genome nodes.  Newly appearing inline
        # citations flow into the decoder; the version then flows into the
        # first report each lineage produced while reading that version.
        decoder_moments = {}
        prior_refs = set()
        for decoder in self.decoder_history:
            version = decoder["version"]
            decoder_id = f"D{version}"
            parsed = parse_report_citations(decoder["artifact"])
            current_refs = {(cite["lineage"], cite["report"]): cite
                            for cite in parsed}
            introduced = [cite for key, cite in current_refs.items()
                          if key not in prior_refs]
            source_moments = []
            for cite in introduced:
                try:
                    _, source_report = self._report(
                        cite["lineage"], cite["report"])
                    source_moments.append(source_report["arrival"] + 1)
                except KeyError:
                    pass
            usages = [report["arrival"] + 1
                      for line in self.lineages.values()
                      for report in line["reports"]
                      if report.get("decoder_version", 0) == version]
            if version == 0:
                moment = .1
            elif source_moments:
                moment = max(source_moments) + .3
            elif usages:
                moment = min(usages) - .3
            else:
                moment = decoder_moments.get(version - 1, last_moment) + .3
            decoder_moments[version] = moment
            add_node({
                "id": decoder_id, "kind": "decoder",
                "label": f"Decoder v{version}", "version": version,
                "moment": moment, "decoder_kind": decoder["kind"],
                "summary": decoder.get("rationale") or "initial shared genome",
                "artifact_sha256": decoder.get("artifact_sha256"),
                "citations": [cite["token"] for cite in parsed],
                "preview": str(decoder["artifact"])[:1200],
            })
            if version:
                add_edge(f"D{version - 1}", decoder_id, "decoder")
            for cite in introduced:
                source = checked_source(cite, decoder_id)
                if source:
                    add_edge(source, decoder_id, "distillation",
                             token=cite["token"])
            prior_refs = set(current_refs)

        # A lineage inherits the decoder at birth, then reloads new versions
        # as it continues.  Only the first checkpoint under each version needs
        # an edge; ordinary descent carries it onward from there.
        for line in self.lineages.values():
            born = f"{line['id']}:born"
            version = line.get("decoder_version_at_found", 0)
            add_edge(f"D{version}", born, "inheritance")
            seen_versions = {version}
            for report in line["reports"]:
                report_version = report.get("decoder_version", 0)
                if report_version in seen_versions:
                    continue
                seen_versions.add(report_version)
                add_edge(f"D{report_version}", report_node_id(
                    line["id"], report["index"]), "inheritance")

        return {
            "nodes": nodes, "edges": edges, "unresolved": unresolved,
            "max_moment": max((node["moment"] for node in nodes), default=1),
            "counts": {
                "lineages": len(self.lineages),
                "reports": len(all_reports),
                "decoder_versions": len(self.decoder_history),
                "inspirations": sum(edge["kind"] == "inspiration"
                                    for edge in edges),
                "distillations": sum(edge["kind"] == "distillation"
                                     for edge in edges),
            },
        }

    def stale(self):
        """Living lineages whose best score predates the current
        Decoder.md — candidates for an allocator to refresh."""
        out = []
        for line in self.living():
            if line["best_report"] is None:
                continue
            record = line["reports"][line["best_report"]]
            if record["decoder_version"] < self.decoder_version:
                out.append(line["id"])
        return out

    def summary(self):
        living = self.living()
        reports = [report for line in self.lineages.values()
                   for report in line["reports"]
                   if not report.get("voided")]
        passed = [report for report in reports
                  if report["audit_status"] == "passed"
                  and report["score"] is not None]
        audited_best = {}
        for task in self.tasks:
            active = self.fitness_regime(task)
            candidates = [
                (report["score"], line["id"], report["index"])
                for line in self.lineages.values()
                if line["task"] == task
                for report in line["reports"]
                if report["audit_status"] == "passed"
                and report["score"] is not None
                and self._report_regime(report, task) == active]
            audited_best[task] = (None if not candidates
                                  else max(candidates)[0])
        elo_leaders = {}
        for task in self.tasks:
            played = [row for row in self.ratings(task)
                      if row["games"] > 0]
            leader = None if not played else played[0]
            elo_leaders[task] = (None if leader is None else {
                key: leader[key] for key in (
                    "lineage", "rating", "games", "wins", "losses", "ties")
            })
        valid_matches = [match for match in self.matches
                         if not match["voided"]
                         and str(match.get("fitness_regime", "initial"))
                         == self.fitness_regime(match["task"])]
        return {
            "tasks": list(self.tasks),
            "lineages": len(self.lineages),
            "running": len(living),
            "killed": sum(1 for line in self.lineages.values()
                          if line["status"] == "killed"),
            "reports": len(reports),
            "recorded_reports": self._next_report,
            "voided_reports": self._next_report - len(reports),
            "untrusted_reports": sum(report["score"] is None
                                     for report in reports),
            "audit_passed_reports": len(passed),
            "best": {task: (None if best is None else best["score"])
                     for task, best in self.best.items()},
            "screening_best": {task: (None if best is None else best["score"])
                               for task, best in self.best.items()},
            "baseline": {task: (None if record is None else record["score"])
                         for task, record in self.baselines.items()},
            "verified_champion": {
                task: (None if record is None else record["score"])
                for task, record in self.champions.items()},
            "plateau": self.plateaus,
            "plateau_alert": any(record is not None and record["plateaued"]
                                 for record in self.plateaus.values()),
            "audit_passed_best": audited_best,
            "fitness_regime": dict(getattr(
                self, "fitness_regimes",
                {task: "initial" for task in self.tasks})),
            "judge_criteria": self.judge_criteria,
            "matches": len(valid_matches),
            "open_matches": sum(match["status"] == "open"
                                for match in valid_matches),
            "verdicts": sum(match["status"] == "decided"
                            for match in valid_matches),
            "elo_leader": elo_leaders,
            "decoder_version": self.decoder_version,
            "decisions": len(self.decisions),
            "stale": len(self.stale()),
        }

    # ----------------------------------------------------- engine protocol
    # The duck-typed surface Environment and the dashboards read.

    def best_summary(self):
        return {task: best["score"] for task, best in self.best.items()
                if best is not None}

    def best_record(self):
        records = [best for best in self.best.values() if best is not None]
        return None if not records else max(
            records, key=lambda best: best["score"])

    # -------------------------------------------------------- persistence

    def save(self, path):
        state = dict(self.__dict__)
        path = os.fspath(path)
        temporary = path + ".tmp"
        try:
            with open(temporary, "w") as file:
                json.dump(state, file, indent=1, default=_jsonable,
                          allow_nan=False)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, path)
        except Exception:
            if os.path.exists(temporary):
                os.remove(temporary)
            raise

    @classmethod
    def load(cls, path):
        with open(path) as file:
            state = json.load(file)
        version = state.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"state schema {version!r} is not agentic GAR schema "
                f"{SCHEMA_VERSION}; that run belongs to an archived "
                "protocol — preserve it, don't migrate it")
        campaign = cls(tasks=state["tasks"])
        for key, value in state.items():
            setattr(campaign, key, value)
        # Schema-6 campaigns created before evaluator regimes existed remain
        # valid.  Their reports and matches belong to the initial regime.
        if not hasattr(campaign, "fitness_regimes"):
            campaign.fitness_regimes = {
                task: "initial" for task in campaign.tasks}
        if not hasattr(campaign, "regime_history"):
            campaign.regime_history = {task: [{
                "name": campaign.fitness_regimes[task],
                "rationale": "loaded historical campaign",
                "evidence": None,
            }] for task in campaign.tasks}
        return campaign
