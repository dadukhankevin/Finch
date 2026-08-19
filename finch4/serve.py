"""Persistent HTTP communication plane for GAR (docs/high-agent.md).

One process holds the ONE Campaign record; a lock serializes every
request, so parallel autoresearch workers can stream reports the moment
each experiment finishes without racing each other. Evolvers read shared
research and report directly. Judge agents receive isolated two-individual
assignments and return verdicts. State is saved after every mutation. Finch
records the communication and updates Elo; it never chooses a winner.

    python3 -m finch4.serve --run benchmarks/agentic/runs/r2 \
        --tasks binpack            # creates state.json if absent

The server binds localhost only and writes `server.json` (port, pid)
into the run directory so workers can discover it. JSON in, JSON out:

    GET  /summary        campaign summary
    GET  /lineages       every lineage with its report stream
    GET  /decisions      append-only evolutionary decision log
    GET  /decoder        current Decoder.md version record
    GET  /tree           derived Tree of Life (parsed from Markdown citations)
    GET  /stale          running lineages whose best predates Decoder.md
    GET  /research       shared Decoder plus resolved cited sources
                         (?lineage=L0003 also returns that trajectory)
    GET  /ratings        Elo standings (?task=binpack optional)
    GET  /matches        pairwise judge assignments and verdicts
    GET  /match?id=M0003 one deliberately narrow judge assignment
    GET  /jobs           durable native-worker queue and process state
    POST /found          {task, rationale, idea?, parents?, worker?,
                          kind?} -> lineage (parents are trusted immutable
                          [lineage, report] checkpoints; audit outcomes are
                          carried into the child briefing;
                          idea without parents = eureka injection)
    POST /report         {lineage, summary, score?, source?, evidence?,
                          kept?, claimed_score?, artifact?, sealed_score?,
                          holdout_score?}
                         <- streamed by workers as experiments finish;
                            a trusted score plus artifact appends itself
                            to Decoder.md; score is search fitness;
                            sealed_score is held-in (keep/best);
                            holdout_score is observation only and never
                            used to pick; claimed_score is never promoted
    POST /seal-score     {lineage, report, sealed_score, source, rationale?}
    POST /holdout-score  {lineage, report, holdout_score, source, rationale?}
                         <- observation only; does not change keep/best/Elo
    POST /revise-score   {lineage, report, score, source, evidence?,
                          rationale?}
    POST /void-report    {lineage, report, rationale}
    POST /kill           {lineage, rationale, evidence?}
    POST /revive         {lineage, rationale}
    POST /assign         {lineage, worker, rationale}
    POST /audit          {lineage, report, outcome:
                          passed|failed|inconclusive, rationale, evidence?}
    POST /baseline       {task, score, source, rationale, evidence?}
    POST /champion       {lineage, report, source, rationale, evidence?}
                          externally verified champion designation
    POST /plateau        {task, plateaued, rationale, evidence?}
                          agent-authored dashboard signal (never inferred)
    POST /criteria       {task, criteria, rationale}
    POST /pair           {individuals: [[lineage, report],
                          [lineage, report]], rationale, requested_by,
                          judge?} -> exact two-checkpoint judge assignment
    POST /verdict        {match, winner: L####|tie, judge, rationale,
                          evidence?, confidence?} -> mechanical Elo update
    POST /void-match     {match, rationale} -> replay later Elo without it
    POST /note           {rationale, lineages?}
    POST /share          {artifact, rationale} -> new Decoder.md version;
                          new citations need trusted scored artifacts but
                          not a passed audit (scientific memory, not method)
    POST /finding        {fragment, rationale} -> atomically append one
                          agent-authored cited Markdown finding; concurrent
                          agents cannot overwrite one another
    POST /incorporate    {artifact, rationale} -> new Decoder.md version;
                          artifact is the full Markdown and cites scored,
                          audit-passed reports inline as
                          [L0003#2@8ddf8b41]
    POST /compact        {artifact, rationale}
    POST /media          {name, kind: image|svg|text, data, epoch?,
                          evaluations?} — see below

MEDIA is the dashboard's second channel: alongside the fitness curves, a
run may post what it is evolving — latest-wins per name, images as PNG
data URIs (png_data_uri encodes any numpy image array with stdlib only),
SVG and text inline. The latest item per name is mirrored to
run_dir/media/, so finished runs keep their media and the hub shows it
on the run's card. Producers: live_progress(images="auto") posts each
function's best-ever phenotype automatically when it is image-shaped;
progress.report_media / media_client(run_dir) post anything else.
Media is telemetry for the eyes, NEVER evidence: a picture is not a
score, and the evaluator and the audit record remain the only sources
of truth.
"""
from __future__ import annotations

import argparse
import base64
import copy
import fcntl
import hashlib
import html
import json
import os
import re
import shutil
import struct
import threading
import time
import zlib
from collections import deque
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import numpy as np

from .agentic import Campaign

from .ui import CHART_PALETTE as PALETTE


# ----------------------------------------------------------------- media
#
# The dashboard's second channel: what a run is evolving, not just how
# well. Latest-wins per name, bounded, mirrored to run_dir/media/ so
# finished runs keep it. Telemetry for the eyes, never evidence.

MEDIA_EXT = {"image": ".png", "svg": ".svg", "text": ".txt"}
MEDIA_MAX_BYTES = 256 * 1024      # per item, encoded
MEDIA_MAX_NAMES = 16              # per run
FILMSTRIP_FRAMES = 10             # in-memory trajectory, images only


def _as_rgb8(array) -> np.ndarray:
    """Any image-ish numpy array -> (H, W, C) uint8. Accepts grayscale
    (H, W), channels-last (H, W, C<=4), channels-first (C<=4, H, W);
    floats are read in [0, 1], uint8 passes through."""
    a = np.asarray(array)
    if a.ndim == 3 and a.shape[0] <= 4 and a.shape[-1] > 4:
        a = np.transpose(a, (1, 2, 0))
    if a.dtype != np.uint8:
        a = np.rint(np.clip(a.astype(np.float64), 0.0, 1.0)
                    * 255).astype(np.uint8)
    if a.ndim == 2:
        a = a[..., None]
    if a.ndim != 3 or a.shape[-1] not in (1, 3, 4):
        raise ValueError(f"cannot render shape {np.asarray(array).shape} "
                         "as an image")
    return np.ascontiguousarray(a)


def _png_bytes(rgb8: np.ndarray) -> bytes:
    """Minimal PNG encoder, stdlib only (zlib + struct): 8-bit
    grayscale/RGB/RGBA, no filtering — keeps the no-dependency rule
    (PIL never enters the library)."""
    h, w, c = rgb8.shape
    color_type = {1: 0, 3: 2, 4: 6}[c]

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data)))

    header = struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)
    raw = b"".join(b"\x00" + rgb8[y].tobytes() for y in range(h))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def png_data_uri(array) -> str:
    """Encode a numpy image array as a PNG data URI (the wire format of
    media kind "image"). Layouts and value ranges as _as_rgb8."""
    return ("data:image/png;base64,"
            + base64.b64encode(_png_bytes(_as_rgb8(array))).decode())


def looks_like_image(shape) -> bool:
    """The images="auto" test: could a phenotype of this shape be
    rendered? 2-D grids and 3-D arrays with a small channel dim."""
    if len(shape) == 2:
        return min(shape) >= 2
    if len(shape) == 3:
        return shape[0] <= 4 or shape[-1] <= 4
    return False


def _media_body(name, image=None, svg=None, text=None, epoch=None,
                evaluations=None):
    """One media item as a /media request body; exactly one of image=
    (numpy array), svg= (inline markup) or text= must be given."""
    if image is not None:
        kind, data = "image", png_data_uri(image)
    elif svg is not None:
        kind, data = "svg", str(svg)
    elif text is not None:
        kind, data = "text", str(text)
    else:
        raise ValueError("pass image=, svg= or text=")
    return {"name": str(name), "kind": kind, "data": data,
            "epoch": epoch, "evaluations": evaluations}


def registry_path():
    """Global registry of every run this machine has served — one JSON
    line per server start. The hub (hub.py) reads it to show ALL
    evolution jobs, live and finished, on one page."""
    return (os.environ.get("FINCH4_REGISTRY")
            or os.path.expanduser("~/.finch4/registry.jsonl"))


def register_run(run_dir, port):
    path = registry_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps({"run_dir": os.path.abspath(run_dir),
                            "port": port, "pid": os.getpid(),
                            "started": time.time()}) + "\n")


def curve_svg(series, points=None, xlabel="evaluations", w=720, h=260,
              labels=True):
    """Fitness-over-time chart, dependency-free. series maps a name to
    its best-so-far [(x, y), ...] step curve — one line per task or
    fitness function, same rendering for every kind of run. points are
    optional (x, y, label) markers (scored reports)."""
    series = {k: v for k, v in series.items() if len(v) >= 2}
    if not series:
        return "<p>curve appears after two scored points</p>"
    allx = [x for c in series.values() for x, _ in c]
    ally = [y for c in series.values() for _, y in c]
    if points:
        ally += [p[1] for p in points]
    lo, hi = min(ally), max(ally)
    pad = max((hi - lo) * 0.15, 1e-9)
    lo, hi = lo - pad, hi + pad
    ML, MB = (62, 28) if labels else (8, 8)
    xmax = max(allx) * 1.05 + 1e-9

    def X(x):
        return ML + (w - ML - 14) * x / xmax

    def Y(y):
        return 12 + (h - 12 - MB) * (hi - y) / (hi - lo)

    s = [f'<svg width="{w}" height="{h}" style="background:#161616;'
         'border:1px solid #333">']
    if labels:
        s.append(f'<text x="{ML}" y="{h-8}" font-size="10" fill="#888">'
                 f'x &#8212; {xlabel} &#183; y &#8212; best score so far '
                 '(higher is better)</text>')
    for i, (name, curve) in enumerate(sorted(series.items())):
        color = PALETTE[i % len(PALETTE)]
        path = ""
        for j, (x, y) in enumerate(curve):
            path += (f"M {X(x):.1f} {Y(y):.1f} " if j == 0
                     else f"L {X(x):.1f} {Y(curve[j-1][1]):.1f} "
                          f"L {X(x):.1f} {Y(y):.1f} ")
        s.append(f'<path d="{path}" fill="none" stroke="{color}" '
                 'stroke-width="2"/>')
        if labels:
            lx, ly = curve[-1]
            s.append(f'<text x="{X(lx)-4:.1f}" y="{Y(ly)-6:.1f}" '
                     f'font-size="10" fill="{color}" text-anchor="end">'
                     f'{html.escape(str(name))} {ly:.5g}</text>')
    if labels:
        for x, y, lab in (points or []):
            s.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="3.5" '
                     'fill="#666"/>')
            s.append(f'<text x="{X(x)+5:.1f}" y="{Y(y)+11:.1f}" '
                     f'font-size="8.5" fill="#777">{lab}</text>')
        for gy in (lo + pad, hi - pad):
            s.append(f'<text x="{ML-6}" y="{Y(gy)+3:.1f}" font-size="9" '
                     f'fill="#888" text-anchor="end">{gy:.5g}</text>')
    s.append("</svg>")
    return "".join(s)


def _sealed_score(report):
    if report.get("sealed_score") is not None:
        return report["sealed_score"]
    evidence = report.get("evidence")
    if isinstance(evidence, dict) and evidence.get("sealed_score") is not None:
        return evidence["sealed_score"]
    return None


def _holdout_score(report):
    if report.get("holdout_score") is not None:
        return report["holdout_score"]
    return None


def lineage_curves(lineages, fitness_regimes=None):
    """Per-task curves over report arrival order.

    Search scores are always gray point markers. If any report has a
    sealed score, the step line that counts is sealed/held-in
    best-so-far — search gets its own labeled series so a 1.0 in-sample
    climb cannot pose as confirmation. Holdout uses the same
    best-so-far step, as observation only: it never decides keep.
    If nothing is sealed, the legacy task series is search-only and
    the page must title it as in-sample.
    """
    tasks = {line["id"]: line["task"] for line in lineages}
    active = fitness_regimes or {}
    scored = sorted(
        (report for line in lineages for report in line["reports"]
         if report.get("score") is not None
         and not report.get("voided")
         and (not active or report.get("fitness_regime", "initial")
              == active.get(tasks[report["lineage"]], "initial"))),
        key=lambda report: report["arrival"])
    has_sealed = any(_sealed_score(report) is not None for report in scored)
    series, points, search_best, sealed_best = {}, [], {}, {}
    holdout_best, audited_search, audited_sealed = {}, {}, {}
    for report in scored:
        task = tasks[report["lineage"]]
        x = report["arrival"] + 1
        points.append((x, report["score"], report["lineage"]))
        sealed = _sealed_score(report)
        if sealed is None:
            if task not in search_best or report["score"] > search_best[task]:
                search_best[task] = report["score"]
            if report.get("audit_status") == "passed":
                if (task not in audited_search
                        or report["score"] > audited_search[task]):
                    audited_search[task] = report["score"]
        if task in search_best:
            search_name = f"{task} · search" if has_sealed else task
            series.setdefault(search_name, []).append((x, search_best[task]))
            if task in audited_search:
                suffix = (" · search · audit-passed" if has_sealed
                          else " · audit-passed")
                series.setdefault(task + suffix, []).append(
                    (x, audited_search[task]))
        holdout = _holdout_score(report)
        if holdout is not None:
            if task not in holdout_best or holdout > holdout_best[task]:
                holdout_best[task] = holdout
            series.setdefault(f"{task} · holdout", []).append(
                (x, holdout_best[task]))
        if sealed is None:
            continue
        if task not in sealed_best or sealed > sealed_best[task]:
            sealed_best[task] = sealed
        series.setdefault(f"{task} · sealed", []).append((x, sealed_best[task]))
        if report.get("audit_status") == "passed":
            if (task not in audited_sealed
                    or sealed > audited_sealed[task]):
                audited_sealed[task] = sealed
            series.setdefault(f"{task} · sealed · audit-passed", []).append(
                (x, audited_sealed[task]))
    return series, points


def telemetry_curves(telemetry):
    series = {}
    for point in telemetry:
        x = point.get("evaluations") or point.get("epoch") or 0
        for name, score in (point.get("best") or {}).items():
            if score is None:
                continue
            cur = series.setdefault(name, [])
            y = max(score, cur[-1][1]) if cur else score
            cur.append((x, y))
    return series


class GAService:
    """The campaign record plus the lock and the save-after-every-
    mutation rule.

    Also the ONE telemetry sink for every kind of run: agentic
    campaigns feed it through found/report/decide routes, and the tensor
    solver feeds it through POST /telemetry (see live_progress below),
    so the /progress dashboard is the same page for every evolutionary
    problem this library runs. campaign may be None (telemetry-only
    mode)."""

    def __init__(self, campaign, state_path, run_dir=None):
        self.campaign = campaign
        self.state_path = state_path
        # Callers commonly use pathlib.Path for run directories.
        self.run_dir = None if run_dir is None else os.fspath(run_dir)
        # Reentrant: handle() holds the transaction lock across mutation
        # and persistence, while call() also protects direct callers.
        self.lock = threading.RLock()
        self.events = deque(maxlen=300)
        self.telemetry = []
        self.media = {}          # name -> latest item (kind, data, epoch)
        self.filmstrips = {}     # name -> deque of recent image frames
        self.started = time.time()

    def _worker_jobs(self):
        """Read the process queue without giving the dashboard authority.

        ``finch4.workers`` owns the locked writes and atomically replaces the
        file.  The HTTP plane only mirrors that durable process state so an
        allocator can see what Finch has queued, launched, or failed.
        """
        empty = {"schema_version": 1, "next_id": 0, "jobs": []}
        if self.run_dir is None:
            return empty
        path = os.path.join(self.run_dir, "jobs.json")
        if not os.path.isfile(path):
            return empty
        try:
            with open(path, encoding="utf-8") as f:
                value = json.load(f)
        except (OSError, json.JSONDecodeError):
            return empty
        if not isinstance(value, dict) or not isinstance(
                value.get("jobs"), list):
            return empty
        return value

    @contextmanager
    def _state_file_lock(self):
        """Serialize state mutations across independently started servers."""
        if not self.state_path:
            yield
            return
        path = self.state_path + ".lock"
        with open(path, "a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def event(self, text):
        self.events.append((time.strftime("%H:%M:%S"), text))

    def _snapshot_artifact(self, lineage_id, artifact):
        """Copy a scored file into the immutable campaign artifact store."""
        if artifact is None:
            return None, None
        source = os.path.abspath(os.path.expanduser(os.fspath(artifact)))
        if not os.path.isfile(source):
            raise ValueError(f"artifact is not a file: {source}")
        with open(source, "rb") as f:
            payload = f.read()
        digest = hashlib.sha256(payload).hexdigest()
        report_index = len(self.campaign.lineages[lineage_id]["reports"])
        suffix = os.path.splitext(source)[1]
        directory = os.path.join(self.run_dir, "artifacts")
        os.makedirs(directory, exist_ok=True)
        target = os.path.join(
            directory,
            f"{lineage_id}-r{report_index:04d}-{digest[:16]}{suffix}")
        if not os.path.exists(target):
            shutil.copy2(source, target)
        return target, digest

    def _write_decoder(self, content):
        if not isinstance(content, str):
            raise ValueError("Decoder.md artifact must be its full text")
        path = os.path.join(self.run_dir, "Decoder.md")
        temporary = path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)

    def _verify_artifact_refs(self, refs):
        """Refuse influence if a stored snapshot no longer matches its ID."""
        for ref in refs:
            if isinstance(ref, dict):
                lineage_id = ref.get("lineage")
                report_index = ref.get("report")
            else:
                lineage_id, report_index = ref
            _, record = self.campaign._report(lineage_id, report_index)
            path = record.get("artifact")
            expected = record.get("artifact_sha256")
            if not path or not expected or not os.path.isfile(path):
                raise ValueError(
                    f"artifact snapshot missing for {lineage_id}#"
                    f"{report_index}")
            with open(path, "rb") as f:
                observed = hashlib.sha256(f.read()).hexdigest()
            if observed != expected:
                raise ValueError(
                    f"artifact snapshot hash mismatch for {lineage_id}#"
                    f"{report_index}")

    def call(self, name, body):
        campaign = self.campaign
        with self.lock:
            if name == "telemetry":
                point = {"epoch": body.get("epoch"),
                         "evaluations": body.get("evaluations"),
                         "best": body.get("best", {})}
                self.telemetry.append(point)
                if self.run_dir:
                    with open(os.path.join(self.run_dir,
                                           "telemetry.jsonl"), "a") as f:
                        f.write(json.dumps(point) + "\n")
                return {"ok": True}, False
            if name == "media":
                return self._store_media(body), False
            if name == "page.json":
                return self._page_data(), False
            if campaign is None:
                if name == "summary":
                    last = self.telemetry[-1] if self.telemetry else {}
                    return {"telemetry_points": len(self.telemetry),
                            "best": last.get("best", {})}, False
                raise KeyError(f"{name} needs a campaign (telemetry-only "
                               "server)")
            if name == "summary":
                return campaign.summary(), False
            if name == "jobs":
                return self._worker_jobs(), False
            if name == "lineages":
                return [dict(line) for line in
                        campaign.lineages.values()], False
            if name == "decisions":
                return list(campaign.decisions), False
            if name == "decoder":
                return campaign.current_decoder(), False
            if name == "tree":
                return campaign.tree_of_life(), False
            if name == "stale":
                return campaign.stale(), False
            if name == "ratings":
                return campaign.ratings(task=body.get("task")), False
            if name == "matches":
                return [dict(match) for match in campaign.matches], False
            if name == "match":
                assignment = campaign.match_assignment(body["id"])
                self._verify_artifact_refs(assignment["individuals"])
                return assignment, False
            if name == "research":
                return campaign.research(body.get("lineage")), False
            if name == "found":
                self._verify_artifact_refs(body.get("parents", ()))
                return campaign.found(
                    body["task"], body["rationale"],
                    idea=body.get("idea"),
                    parents=body.get("parents", ()),
                    worker=body.get("worker"),
                    kind=body.get("kind")), True
            if name == "report":
                cited = campaign.resolve_citations(body["summary"])
                self._verify_artifact_refs(cited)
                artifact, artifact_sha256 = self._snapshot_artifact(
                    body["lineage"], body.get("artifact"))
                return campaign.report(
                    body["lineage"], body["summary"],
                    score=body.get("score"), source=body.get("source"),
                    evidence=body.get("evidence"), kept=body.get("kept"),
                    claimed_score=body.get("claimed_score"),
                    artifact=artifact,
                    artifact_sha256=artifact_sha256,
                    experiment_id=body.get("experiment_id"),
                    sealed_score=body.get("sealed_score"),
                    holdout_score=body.get("holdout_score")), True
            if name == "seal-score":
                return campaign.set_sealed_score(
                    body["lineage"], body["report"], body["sealed_score"],
                    body["source"],
                    rationale=body.get("rationale", "sealed rescore")), True
            if name == "holdout-score":
                return campaign.set_holdout_score(
                    body["lineage"], body["report"], body["holdout_score"],
                    body["source"],
                    rationale=body.get("rationale", "holdout observation")), True
            if name == "revise-score":
                campaign.revise_score(
                    body["lineage"], body["report"], body["score"],
                    body["source"], evidence=body.get("evidence"),
                    rationale=body.get("rationale", "correction"))
                return {"ok": True}, True
            if name == "void-report":
                return campaign.void_report(
                    body["lineage"], body["report"],
                    body["rationale"]), True
            if name == "kill":
                return campaign.kill(
                    body["lineage"], body["rationale"],
                    evidence=body.get("evidence")), True
            if name == "revive":
                return campaign.revive(
                    body["lineage"], body["rationale"]), True
            if name == "assign":
                return campaign.assign(
                    body["lineage"], body["worker"],
                    body["rationale"]), True
            if name == "audit":
                self._verify_artifact_refs([
                    (body["lineage"], body["report"])])
                return campaign.audit(
                    body["lineage"], body["report"], body["outcome"],
                    body["rationale"],
                    evidence=body.get("evidence")), True
            if name == "baseline":
                return campaign.set_baseline(
                    body["task"], body["score"], body["source"],
                    body["rationale"], evidence=body.get("evidence")), True
            if name == "regime":
                return campaign.start_fitness_regime(
                    body["task"], body["name"], body["rationale"],
                    evidence=body.get("evidence")), True
            if name == "champion":
                self._verify_artifact_refs([
                    (body["lineage"], body["report"])])
                return campaign.set_champion(
                    body["lineage"], body["report"], body["source"],
                    body["rationale"], evidence=body.get("evidence")), True
            if name == "plateau":
                return campaign.set_plateau(
                    body["task"], body["plateaued"], body["rationale"],
                    evidence=body.get("evidence")), True
            if name == "criteria":
                return campaign.set_judge_criteria(
                    body["task"], body["criteria"],
                    body["rationale"]), True
            if name == "pair":
                self._verify_artifact_refs(body["individuals"])
                return campaign.pair(
                    body["individuals"], body["rationale"],
                    body["requested_by"], judge=body.get("judge")), True
            if name == "verdict":
                return campaign.verdict(
                    body["match"], body["winner"], body["judge"],
                    body["rationale"], evidence=body.get("evidence"),
                    confidence=body.get("confidence")), True
            if name == "void-match":
                return campaign.void_match(
                    body["match"], body["rationale"]), True
            if name == "note":
                return campaign.note(
                    body["rationale"],
                    lineage_ids=body.get("lineages", ())), True
            if name == "share":
                cited = campaign.resolve_citations(
                    body["artifact"], require_citations=True)
                self._verify_artifact_refs(cited)
                return campaign.share(
                    body["artifact"], body["rationale"]), True
            if name == "finding":
                cited = campaign.resolve_citations(
                    body["fragment"], require_citations=True)
                self._verify_artifact_refs(cited)
                result = campaign.append_finding(
                    body["fragment"], body["rationale"])
                return result, not result.get("idempotent", False)
            if name == "incorporate":
                cited = campaign.resolve_citations(body["artifact"])
                self._verify_artifact_refs(cited)
                return campaign.incorporate(
                    body["artifact"], body["rationale"]), True
            if name == "compact":
                # Campaign.compact re-checks each retained citation through
                # the evidence level by which it entered (scored finding or
                # audited method). Do not silently upgrade all retained
                # findings to the stricter method membrane here.
                cited = campaign.resolve_citations(body["artifact"])
                self._verify_artifact_refs(cited)
                return campaign.compact(
                    body["artifact"], body["rationale"]), True
            raise KeyError(name)

    # Routes that may mutate the campaign record; only these pay for the
    # rollback snapshot below. Reads (including the dashboard's page.json
    # poll) must stay O(1) in campaign size.
    MUTATING = frozenset((
        "found", "report", "seal-score", "revise-score", "void-report", "kill",
        "revive", "assign", "audit", "note", "share", "finding", "incorporate",
        "compact", "baseline", "regime", "champion", "plateau", "criteria",
        "pair", "verdict", "void-match"))

    def handle(self, name, body):
        if name not in self.MUTATING:
            return self._handle(name, body)
        # A process-local RLock was enough for one HTTP server, but two
        # controllers can legitimately discover the same run directory.  Hold
        # a filesystem lock through reload -> mutation -> durable save so a
        # stale Campaign cannot hand out an already reserved lineage ID.
        with self._state_file_lock():
            with self.lock:
                if self.campaign is not None and os.path.exists(self.state_path):
                    self.campaign = Campaign.load(self.state_path)
                return self._handle(name, body)

    def _handle(self, name, body):
        # One mutation and its durable snapshot are one transaction.
        with self.lock:
            if name not in self.MUTATING:
                result, _ = self.call(name, body)
                return result
            state_before = (None if self.campaign is None else
                            copy.deepcopy(self.campaign.__dict__))
            decoder_path = (None if self.run_dir is None else
                            os.path.join(self.run_dir, "Decoder.md"))
            decoder_before = None
            decoder_existed = bool(
                decoder_path and name in (
                    "share", "finding", "incorporate", "compact", "report")
                and os.path.exists(decoder_path))
            if decoder_existed:
                with open(decoder_path, "rb") as f:
                    decoder_before = f.read()
            try:
                result, mutated = self.call(name, body)
                if mutated and name in (
                        "share", "finding", "incorporate", "compact"):
                    self._write_decoder(result["artifact"])
                elif (mutated and name == "report" and self.campaign
                      and self.campaign.decoder_version):
                    self._write_decoder(
                        self.campaign.current_decoder()["artifact"])
                if mutated:
                    self.campaign.save(self.state_path)
            except Exception:
                if self.campaign is not None and state_before is not None:
                    self.campaign.__dict__.clear()
                    self.campaign.__dict__.update(state_before)
                if decoder_path and name in (
                        "share", "finding", "incorporate", "compact",
                        "report"):
                    if decoder_existed:
                        with open(decoder_path, "wb") as f:
                            f.write(decoder_before)
                    elif os.path.exists(decoder_path):
                        os.remove(decoder_path)
                raise
            if mutated:
                if name == "found":
                    self.event(f"{result['kind']} {result['id']} "
                               f"task={result['task']}"
                               + (f" parents={result['parents']}"
                                  if result["parents"] else ""))
                elif name == "report":
                    score = result.get("score")
                    holdout = result.get("holdout_score")
                    self.event(
                        f"report {result['lineage']}#{result['index']} "
                        + (f"score={score:.5f} [{result['source']}]"
                           if score is not None else "unscored")
                        + (f" holdout={holdout:.5f}"
                           if holdout is not None else "")
                        + f" — {result['summary'][:60]}")
                elif name == "holdout-score":
                    self.event(
                        f"holdout {body['lineage']}#{body['report']} "
                        f"-> {body['holdout_score']:.5f} (observation)")
                elif name == "revise-score":
                    self.event(f"revise {body['lineage']}#{body['report']}"
                               f" -> {body['score']:.5f}")
                elif name in ("kill", "revive", "assign", "audit", "note",
                              "void-report", "void-match"):
                    detail = ""
                    if name == "audit":
                        detail = " " + str(body.get("outcome", ""))
                    self.event(f"{name}{detail} "
                               f"{body.get('lineage', '')} — "
                               f"{str(body.get('rationale', ''))[:70]}")
                elif name == "criteria":
                    self.event(f"criteria task={body['task']} — "
                               f"{str(body.get('rationale', ''))[:70]}")
                elif name == "pair":
                    ids = [item["lineage"] for item in result["individuals"]]
                    self.event(f"pair {result['id']} {' vs '.join(ids)}")
                elif name == "verdict":
                    winner = result["verdict"]["winner"] or "tie"
                    self.event(f"verdict {result['id']} winner={winner} "
                               f"judge={result['verdict']['judge']}")
                elif name in ("share", "finding", "incorporate"):
                    citation_count = len(
                        self.campaign.resolve_citations(result["artifact"]))
                    self.event(f"{name.upper()} -> Decoder.md "
                               f"v{result['version']} "
                               f"({citation_count} inline cites)")
                elif name == "compact":
                    self.event(f"compact -> Decoder.md "
                               f"v{result['version']}")
            return result

    def _store_media(self, body):
        """Latest-wins media telemetry: display only, never evidence — a
        picture is not a score. Bounded: MEDIA_MAX_NAMES names,
        MEDIA_MAX_BYTES per item; images also feed a short in-memory
        filmstrip (the trajectory), and the latest item per name is
        mirrored to run_dir/media/ (what finished runs and the hub
        read)."""
        name = str(body.get("name") or "media")
        kind = body.get("kind")
        data = body.get("data")
        if kind not in MEDIA_EXT:
            raise ValueError(f"unknown media kind {kind!r}; "
                             f"kinds: {sorted(MEDIA_EXT)}")
        if not isinstance(data, str):
            raise ValueError("media data must be a string (image: a PNG "
                             "data URI — see png_data_uri; svg/text: the "
                             "content itself)")
        if len(data) > MEDIA_MAX_BYTES:
            raise ValueError(f"media item {name!r} over "
                             f"{MEDIA_MAX_BYTES} bytes")
        if name not in self.media and len(self.media) >= MEDIA_MAX_NAMES:
            raise ValueError(f"over {MEDIA_MAX_NAMES} media names; "
                             "reuse a name (latest wins)")
        self.media[name] = {"kind": kind, "data": data,
                            "epoch": body.get("epoch"),
                            "evaluations": body.get("evaluations")}
        if kind == "image":
            strip = self.filmstrips.setdefault(
                name, deque(maxlen=FILMSTRIP_FRAMES))
            strip.append(data)
        if self.run_dir:
            media_dir = os.path.join(self.run_dir, "media")
            os.makedirs(media_dir, exist_ok=True)
            safe = re.sub(r"[^A-Za-z0-9._-]", "_", name)
            content = (base64.b64decode(data.split(",", 1)[1])
                       if kind == "image" else data.encode())
            with open(os.path.join(media_dir, safe + MEDIA_EXT[kind]),
                      "wb") as f:
                f.write(content)
        return {"ok": True}

    # ------------------------------------------------------ progress page

    def _page_data(self):
        """Everything the dashboard needs, one JSON payload — the page
        polls this instead of reloading itself."""
        campaign = self.campaign
        data = {"name": os.path.basename((self.run_dir or "run")
                                         .rstrip("/")),
                "up_min": round((time.time() - self.started) / 60, 1),
                "events": [list(e) for e in self.events][-120:],
                "media": {k: dict(v) for k, v in self.media.items()},
                "filmstrips": {k: list(v)
                               for k, v in self.filmstrips.items()}}
        if campaign is None:
            last = self.telemetry[-1] if self.telemetry else {}
            if not data["events"]:
                data["events"] = [
                    ["", f"epoch {p.get('epoch')} · "
                         f"evals {p.get('evaluations')} · "
                         f"best {p.get('best')}"]
                    for p in self.telemetry[-40:]]
            data.update(
                mode="solver",
                series=telemetry_curves(self.telemetry), points=[],
                summary={"reports": len(self.telemetry),
                         "evaluations": last.get("evaluations") or 0},
                best={k: {"score": v} for k, v in
                      (last.get("best") or {}).items()}, lineages=[])
            return data
        lineages = list(campaign.lineages.values())
        jobs = self._worker_jobs()["jobs"]
        job_counts = {}
        for job in jobs:
            status = str(job.get("status") or "unknown")
            job_counts[status] = job_counts.get(status, 0) + 1
        series, points = lineage_curves(
            lineages, getattr(campaign, "fitness_regimes", None))
        series.update(telemetry_curves(self.telemetry))
        rows = []
        has_verdicts = any(line.get("elo_games", 0) for line in lineages)
        for line in sorted(
                lineages,
                key=lambda l: (l["status"] != "running",
                               -l.get("elo", 1500.0) if has_verdicts else
                               -(l["best_score"] if l["best_score"] is not None
                                 else float("-inf")))):
            last = line["reports"][-1] if line["reports"] else None
            rows.append({
                "id": line["id"], "task": line["task"],
                "kind": line["kind"], "status": line["status"],
                "worker": line.get("worker") or "unassigned",
                "best": line["best_score"],
                "elo": line.get("elo", 1500.0),
                "games": line.get("elo_games", 0),
                "wins": line.get("elo_wins", 0),
                "losses": line.get("elo_losses", 0),
                "ties": line.get("elo_ties", 0),
                "reports": sum(not report.get("voided")
                               for report in line["reports"]),
                "pending": sum(report["score"] is None
                               and not report.get("voided")
                               for report in line["reports"]),
                "audit": campaign._lineage_audit_status(line),
                "stale": line["id"] in campaign.stale(),
                "last": "" if last is None else last["summary"][:140],
            })
        summary = campaign.summary()
        summary.update(
            jobs=len(jobs), queued_jobs=job_counts.get("queued", 0),
            running_jobs=job_counts.get("running", 0),
            failed_jobs=sum(job_counts.get(status, 0) for status in
                            ("failed", "timed_out", "crashed")))
        sealed_best = {}
        holdout_observe = {}
        for name, curve in series.items():
            if " · sealed" in name and "audit-passed" not in name and curve:
                sealed_best[name.split(" · ")[0]] = {"score": curve[-1][1]}
            if name.endswith(" · holdout") and curve:
                holdout_observe[name.split(" · ")[0]] = {"score": curve[-1][1]}
        data.update(
            mode="agentic", summary=summary, series=series,
            points=points,
            decisions=list(campaign.decisions)[-80:],
            best={task: dict(best)
                  for task, best in campaign.best.items()
                  if best is not None},
            sealed_best=sealed_best,
            holdout_observe=holdout_observe,
            decoder=campaign.current_decoder(),
            lineages=rows, tree=campaign.tree_of_life(),
            ratings=campaign.ratings(), matches=list(campaign.matches)[-80:],
            jobs=jobs[-120:])
        return data

    def progress_html(self):
        from .ui import page
        return page("Finch 4 run", PROGRESS_BODY, PROGRESS_JS)


PROGRESS_BODY = """
<div class="hdr"><h1 id="title">run</h1>
<span class="pill live"><span class="dot"></span><span id="mode">live</span></span>
<span id="taskpills"></span>
<span class="sub" id="up"></span></div>
<div class="tiles" id="tiles"></div>
<div class="panel" id="treepanel" style="display:none">
<div class="tol-controls">
<h2 style="margin:0 6px 0 0">Tree of Life</h2>
<input id="treesearch" type="search"
  placeholder="search mechanism, lineage, citation…">
<select id="treetask" style="display:none"><option value="">all tasks</option></select>
<span class="chips" id="treechips"></span>
<span class="tol-zoom" id="treezoom"></span>
<span class="tol-time">time
<input id="treetime" type="range" min="1" value="1" step="1">
<span id="treetimevalue">latest</span></span>
<span class="tol-stats" id="treestats"></span>
</div>
<div class="tol-grid">
<div class="tol-rail" id="treerail"></div>
<div class="tol-canvas" id="treecanvas"><svg id="tree"></svg></div>
<aside class="tol-inspect" id="treeinspect"></aside>
</div>
<div class="tol-tip" id="treetip"></div>
</div>
<div class="panel"><h2 id="charttitle">fitness over time</h2>
<p class="chart-note" id="chartnote"></p>
<div class="chartwrap"><div id="chart"></div><div class="tip"></div></div></div>
<details class="panel decoderpanel" id="decoderpanel" style="display:none">
<summary>Decoder.md — shared knowledge<span class="count" id="decodermeta"></span></summary>
<pre id="decoderbody"></pre>
</details>
<div class="panel" id="mediapanel" style="display:none"><h2>evolved media</h2>
<div class="mediagrid" id="media"></div></div>
<div class="panel" id="linepanel" style="display:none"><h2>lineages</h2>
<div style="overflow-x:auto"><table><thead><tr><th>lineage</th><th>task</th>
<th>kind</th><th>status</th><th>worker</th><th>Elo</th><th>best evidence</th><th>experiments</th>
<th>audit</th><th>last experiment</th></tr></thead>
<tbody id="lineages"></tbody></table></div></div>
<div class="panel" id="matchpanel" style="display:none"><h2>pairwise judge tournament</h2>
<div style="overflow-x:auto"><table><thead><tr><th>match</th><th>individuals</th>
<th>judge</th><th>verdict</th><th>Elo update</th></tr></thead>
<tbody id="matches"></tbody></table></div></div>
<div class="panel" id="jobpanel" style="display:none"><h2>native worker pool</h2>
<div style="overflow-x:auto"><table><thead><tr><th>job</th><th>lane</th>
<th>kind</th><th>name</th><th>status</th><th>attempt</th></tr></thead>
<tbody id="jobs"></tbody></table></div></div>
<div class="panel"><h2 id="streamtitle">decision log</h2>
<div class="events" id="events"></div></div>
"""

PROGRESS_JS = """
/* ---------------- Tree of Life ---------------- */
const TREE={data:null,sel:null,laneSel:null,xstep:46,timeDirty:false,
  kinds:{inspiration:true,directed:true,distillation:true,decoder:true,
         inheritance:false}};
const EDGE_STYLE={
  inspiration:{color:'#b0480f',label:'cited inspiration',w:1.7},
  directed:{color:'#8d4a9e',label:'directed recombination',w:2.1},
  distillation:{color:'#41803a',label:'distillation',w:1.7},
  inheritance:{color:'#0089a1',label:'shared inheritance',w:1.3,dash:'4 4'},
  decoder:{color:'#ab7a0c',label:'decoder history',w:2}};
const KIND_MARK={found:'&#9642;',inject:'&#10022;',crossover:'&#9282;'};
const TRUNK_H=54,LANE_H=56,AXIS_H=26,PAD_L=30;

function nodesById(){const m=new Map();
  (TREE.data.nodes||[]).forEach(n=>m.set(n.id,n));return m;}

function treeIndex(){
  const t=TREE.data,byId=nodesById();
  const until=+document.getElementById('treetime').value;
  const maxM=Math.max(1,Math.ceil(t.max_moment||1));
  const taskSel=document.getElementById('treetask').value;
  const visible=t.nodes.filter(n=>n.moment<=until&&
    (!taskSel||n.kind==='decoder'||n.task===taskSel));
  const vis=new Set(visible.map(n=>n.id));
  const laneIds=[...new Set(t.nodes.filter(n=>n.lineage&&
    (!taskSel||n.task===taskSel)).map(n=>n.lineage))].sort();
  const laneNodes=new Map(laneIds.map(id=>[id,
    visible.filter(n=>n.lineage===id).sort((a,b)=>a.moment-b.moment)]));
  const decoders=visible.filter(n=>n.kind==='decoder')
    .sort((a,b)=>a.version-b.version);
  let champ=null;
  visible.forEach(n=>{if(n.kind==='report'&&n.score!=null&&!n.voided&&
    n.kept!==false&&(!champ||n.score>champ.score))champ=n;});
  const q=document.getElementById('treesearch').value.trim().toLowerCase();
  const matched=q?new Set(visible.filter(n=>
    JSON.stringify(n).toLowerCase().includes(q)).map(n=>n.id)):null;
  return {byId,until,maxM,visible,vis,laneIds,laneNodes,decoders,
          champ,q,matched};
}

function ancestrySet(id,ix){
  const inMap=new Map();
  TREE.data.edges.forEach(e=>{
    if(!inMap.has(e.target))inMap.set(e.target,[]);
    inMap.get(e.target).push(e);});
  const keep=new Set([id]),queue=[id];
  while(queue.length){
    const cur=queue.pop();
    (inMap.get(cur)||[]).forEach(e=>{
      if(!ix.vis.has(e.source))return;
      if(!keep.has(e.source)){keep.add(e.source);queue.push(e.source);}});}
  TREE.data.edges.forEach(e=>{
    if(e.source===id&&ix.vis.has(e.target))keep.add(e.target);});
  return keep;
}

function laneY(i){return TRUNK_H+i*LANE_H+LANE_H/2;}
function nodeX(n){return PAD_L+n.moment*TREE.xstep;}

function drawTreeOfLife(tree){
  if(!tree||!(tree.nodes||[]).length)return;
  TREE.data=tree;
  document.getElementById('treetip').style.display='none';
  document.getElementById('treepanel').style.display='';
  const slider=document.getElementById('treetime');
  const maxM=Math.max(1,Math.ceil(tree.max_moment||1));
  slider.max=maxM;
  if(!TREE.timeDirty||+slider.value>maxM)slider.value=maxM;
  document.getElementById('treetimevalue').textContent=
    +slider.value>=maxM?'latest':'exp '+slider.value+' / '+maxM;
  const tasks=[...new Set(tree.nodes.filter(n=>n.task).map(n=>n.task))]
    .sort();
  const sel=document.getElementById('treetask');
  sel.style.display=tasks.length>1?'':'none';
  if(tasks.length>1){const cur=sel.value;
    sel.innerHTML='<option value="">all tasks</option>'+tasks.map(t=>
      '<option value="'+esc(t)+'"'+(t===cur?' selected':'')+'>'+
      esc(t)+'</option>').join('');}
  const ix=treeIndex();
  renderChips(); renderRail(ix); renderCanvas(ix); renderInspect(ix);
  const c=tree.counts||{},unres=(tree.unresolved||[]).length;
  document.getElementById('treestats').innerHTML=
    (c.lineages||0)+' lineages · '+(c.reports||0)+' experiments · '+
    (c.inspirations||0)+' cited crossovers · '+
    (c.distillations||0)+' distilled'+
    (unres?' · <span class="bad">'+unres+' unresolved cites</span>':'');
}

function renderChips(){
  const box=document.getElementById('treechips');
  box.innerHTML=Object.entries(EDGE_STYLE).map(([k,st])=>
    '<span class="chip'+(TREE.kinds[k]?'':' off')+'" data-kind="'+k+'">'+
    '<span class="swatch'+(st.dash?' dashed':'')+'" style="border-color:'+
    st.color+'"></span>'+esc(st.label)+'</span>').join('');
  box.querySelectorAll('.chip').forEach(el=>el.onclick=()=>{
    TREE.kinds[el.dataset.kind]=!TREE.kinds[el.dataset.kind];
    drawTreeOfLife(TREE.data);});
}

function renderRail(ix){
  const rail=document.getElementById('treerail');
  const latest=ix.decoders[ix.decoders.length-1];
  let html='<div class="trunkcard" data-jump="'+
    (latest?esc(latest.id):'')+'" style="height:'+(TRUNK_H-8)+
    'px;margin-bottom:8px"><div class="r1">Decoder.md'+
    '<span class="v">v'+(latest?latest.version:0)+'</span></div>'+
    '<div class="r2">shared knowledge · '+ix.decoders.length+
    ' versions</div></div>';
  html+=ix.laneIds.map((id,i)=>{
    const nodes=ix.laneNodes.get(id)||[];
    const born=nodes.find(n=>n.kind==='lineage');
    const reports=nodes.filter(n=>n.kind==='report');
    const audits=reports.filter(n=>n.audit==='passed').length;
    const scored=reports.filter(n=>n.score!=null&&!n.voided&&
      n.kept!==false).map(n=>n.score);
    const best=scored.length?Math.max(...scored):null;
    const isChamp=ix.champ&&ix.champ.lineage===id;
    const status=born?born.status:'running';
    const faded=(TREE.laneSel&&TREE.laneSel!==id)||
      (ix.matched&&!nodes.some(n=>ix.matched.has(n.id)));
    return '<div class="lanecard'+(status==='killed'?' killed':'')+
      (TREE.laneSel===id?' selected':'')+(faded?' faded':'')+
      '" data-lane="'+esc(id)+'" style="height:'+(LANE_H-8)+
      'px;margin-bottom:8px"><div class="r1">'+
      '<span class="statusdot '+esc(status)+'"></span>'+
      '<span class="id">'+esc(id)+'</span>'+
      '<span class="kindmark" title="'+esc(born?born.lineage_kind:'')+'">'+
      (KIND_MARK[born?born.lineage_kind:'found']||'')+'</span>'+
      '<span class="best'+(isChamp?' champ':'')+'">'+
      (isChamp?'&#9733; ':'')+(best==null?'—':fmt(best))+'</span></div>'+
      '<div class="r2">'+esc(born&&born.worker?born.worker:'unassigned')+
      ' · '+reports.length+' exp'+(audits?' · '+audits+' &#10003;audit':'')+
      '</div></div>';
  }).join('');
  rail.innerHTML=html;
  rail.querySelectorAll('.lanecard').forEach(el=>el.onclick=()=>{
    TREE.laneSel=TREE.laneSel===el.dataset.lane?null:el.dataset.lane;
    TREE.sel=null; drawTreeOfLife(TREE.data);});
  const trunk=rail.querySelector('.trunkcard');
  if(trunk)trunk.onclick=()=>{if(trunk.dataset.jump)treeJump(trunk.dataset.jump);};
}

function edgeVisible(e,ix){
  if(e.kind==='descent')return false;   /* drawn as the lane track */
  if(!TREE.kinds[e.kind])return false;
  return ix.vis.has(e.source)&&ix.vis.has(e.target);
}

function renderCanvas(ix){
  const laneIdx=new Map(ix.laneIds.map((id,i)=>[id,i]));
  const xs=TREE.xstep;
  const width=Math.max(760,PAD_L+(ix.until+1.2)*xs);
  const height=TRUNK_H+ix.laneIds.length*LANE_H+AXIS_H;
  const pos=new Map();
  ix.visible.forEach(n=>{
    const x=nodeX(n);
    const y=n.kind==='decoder'?TRUNK_H/2:laneY(laneIdx.get(n.lineage));
    pos.set(n.id,[x,y]);});
  let keep=null;
  if(TREE.sel&&ix.vis.has(TREE.sel))keep=ancestrySet(TREE.sel,ix);
  const laneKeep=id=>!TREE.laneSel||TREE.laneSel===id;
  const nodeFaded=n=>{
    if(keep)return !keep.has(n.id);
    if(TREE.laneSel&&n.kind!=='decoder'&&n.lineage!==TREE.laneSel)
      return true;
    if(ix.matched&&!ix.matched.has(n.id))return true;
    return false;};
  let s='<rect width="'+width+'" height="'+height+
    '" fill="transparent" data-bg="1"/>';
  s+='<rect x="0" y="0" width="'+width+'" height="'+TRUNK_H+
    '" fill="rgba(171,122,12,.05)"/>';
  ix.laneIds.forEach((id,i)=>{if(i%2)
    s+='<rect x="0" y="'+(TRUNK_H+i*LANE_H)+'" width="'+width+
    '" height="'+LANE_H+'" fill="rgba(94,80,63,.025)"/>';});
  for(let m=5;m<=ix.until;m+=5)
    s+='<line x1="'+(PAD_L+m*xs)+'" y1="'+TRUNK_H+'" x2="'+(PAD_L+m*xs)+
      '" y2="'+(height-AXIS_H)+'" stroke="rgba(94,80,63,.06)"/>'+
      '<text x="'+(PAD_L+m*xs)+'" y="'+(height-9)+'" font-size="9.5" '+
      'fill="#a49a82" text-anchor="middle" font-family="ui-monospace,Menlo,monospace">'+m+'</text>';
  s+='<text x="'+(width-10)+'" y="'+(height-9)+'" font-size="9.5" '+
    'fill="#a49a82" text-anchor="end">experiment arrival &#8594;</text>';
  /* lane tracks */
  ix.laneIds.forEach((id,i)=>{
    const nodes=ix.laneNodes.get(id)||[];
    if(!nodes.length)return;
    const y=laneY(i),x0=nodeX(nodes[0]),x1=nodeX(nodes[nodes.length-1]);
    const born=nodes.find(n=>n.kind==='lineage');
    const fade=(keep&&!nodes.some(n=>keep.has(n.id)))||
      (TREE.laneSel&&TREE.laneSel!==id)||
      (ix.matched&&!nodes.some(n=>ix.matched.has(n.id)));
    s+='<line class="tol-lanetrack'+(fade?' faded':'')+'" x1="'+x0+
      '" y1="'+y+'" x2="'+x1+'" y2="'+y+
      '" stroke="#cfc4ac" stroke-width="1.5"'+
      (born&&born.status==='killed'?' stroke-dasharray="1.5 3"':'')+'/>';});
  /* trunk line */
  if(ix.decoders.length>1){
    const xd0=pos.get(ix.decoders[0].id)[0],
          xd1=pos.get(ix.decoders[ix.decoders.length-1].id)[0];
    s+='<line x1="'+xd0+'" y1="'+(TRUNK_H/2)+'" x2="'+xd1+'" y2="'+
      (TRUNK_H/2)+'" stroke="#ab7a0c" stroke-width="2" opacity="'+
      (TREE.kinds.decoder?'0.55':'0.15')+'"/>';}
  /* cross edges */
  const arrows=new Set();
  TREE.data.edges.filter(e=>edgeVisible(e,ix)&&e.kind!=='decoder')
    .forEach(e=>{
    const a=pos.get(e.source),b=pos.get(e.target);
    if(!a||!b)return;
    const st=EDGE_STYLE[e.kind];arrows.add(e.kind);
    const fade=keep?!(keep.has(e.source)&&keep.has(e.target)):
      (TREE.laneSel&&![e.source,e.target].some(id=>{
        const n=ix.byId.get(id);
        return n&&(n.lineage===TREE.laneSel||n.kind==='decoder');}))||
      (ix.matched&&!(ix.matched.has(e.source)&&ix.matched.has(e.target)));
    const dx=Math.max(30,Math.abs(b[0]-a[0])/2);
    const path='M '+a[0]+' '+a[1]+' C '+(a[0]+dx)+' '+a[1]+', '+
      (b[0]-dx)+' '+b[1]+', '+b[0]+' '+b[1];
    s+='<path class="tol-edge'+(fade?' faded':'')+'" d="'+path+
      '" fill="none" stroke="'+st.color+'" stroke-width="'+st.w+
      '" opacity="0.55"'+(st.dash?' stroke-dasharray="'+st.dash+'"':'')+
      ' marker-end="url(#arr-'+e.kind+')"><title>'+
      esc(st.label+(e.token?' '+e.token:''))+'</title></path>';});
  /* nodes */
  ix.visible.forEach(n=>{
    const p=pos.get(n.id),x=p[0],y=p[1];
    const fade=nodeFaded(n),selc=n.id===TREE.sel;
    let body='';
    if(n.kind==='decoder'){
      body='<rect x="'+(x-16)+'" y="'+(y-11)+'" width="32" height="22" '+
        'rx="6" fill="'+(n.decoder_kind==='compact'?'#fffdf8':'#f7edd6')+
        '" stroke="#ab7a0c" stroke-width="'+(selc?2.4:1.4)+'"/>'+
        '<text x="'+x+'" y="'+(y+4)+'" text-anchor="middle" '+
        'font-size="10.5" font-weight="700" fill="#8a6510" '+
        'font-family="ui-monospace,Menlo,monospace">v'+n.version+'</text>';
    }else if(n.kind==='lineage'){
      const col=n.lineage_kind==='crossover'?'#8d4a9e':
        n.lineage_kind==='inject'?'#ab7a0c':'#6c6252';
      body='<circle cx="'+x+'" cy="'+y+'" r="5.5" fill="#fffdf8" '+
        'stroke="'+col+'" stroke-width="'+(selc?2.6:1.7)+'"/>'+
        '<circle cx="'+x+'" cy="'+y+'" r="1.8" fill="'+col+'"/>';
    }else{
      const isChamp=ix.champ&&ix.champ.id===n.id;
      const unscored=n.score==null;
      let r=unscored?3.2:6, fill='#fffdf8', stroke='#b3a88f', sw=1.6;
      if(n.voided){fill='#e7e1d2';stroke='#a49a82';}
      else if(unscored){fill='#cfc4ac';stroke='#cfc4ac';}
      else if(n.kept){fill='#5d9150';stroke='#41803a';}
      if(isChamp){r=8;fill='#b0480f';stroke='#8a3406';}
      if(selc)sw=3;
      if(n.audit==='passed')
        body+='<circle cx="'+x+'" cy="'+y+'" r="'+(r+3.4)+
          '" fill="none" stroke="#ab7a0c" stroke-width="2"/>';
      if(n.audit==='failed')
        body+='<circle cx="'+x+'" cy="'+y+'" r="'+(r+3.4)+
          '" fill="none" stroke="#a84434" stroke-width="2" '+
          'stroke-dasharray="2.5 2.5"/>';
      body+='<circle cx="'+x+'" cy="'+y+'" r="'+r+'" fill="'+fill+
        '" stroke="'+stroke+'" stroke-width="'+sw+'"/>';
      if(n.voided)
        body+='<path d="M '+(x-3.4)+' '+(y-3.4)+' L '+(x+3.4)+' '+
          (y+3.4)+' M '+(x+3.4)+' '+(y-3.4)+' L '+(x-3.4)+' '+(y+3.4)+
          '" stroke="#a84434" stroke-width="1.6"/>';
      if(isChamp)
        body+='<text x="'+x+'" y="'+(y+3.6)+'" text-anchor="middle" '+
          'font-size="9" fill="#fff">&#9733;</text>';
      if(n.score!=null&&(n.kept||isChamp))
        body+='<text x="'+x+'" y="'+(y+(n.audit==='passed'?22:17))+
          '" text-anchor="middle" font-size="9" '+
          'font-family="ui-monospace,Menlo,monospace" fill="'+
          (isChamp?'#b0480f':'#6c6252')+'"'+
          (isChamp?' font-weight="700"':'')+'>'+fmt(n.score)+'</text>';
    }
    s+='<g class="tol-node'+(fade?' faded':'')+'" data-node="'+
      esc(n.id)+'">'+
      '<rect x="'+(x-14)+'" y="'+(y-16)+'" width="28" height="32" '+
      'fill="transparent"/>'+body+'</g>';});
  let defs='<defs>';
  arrows.forEach(k=>{defs+='<marker id="arr-'+k+
    '" markerWidth="7" markerHeight="7" refX="6" refY="3.5" '+
    'orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="'+
    EDGE_STYLE[k].color+'"/></marker>';});
  defs+='</defs>';
  const svg=document.getElementById('tree');
  svg.setAttribute('width',width);svg.setAttribute('height',height);
  svg.innerHTML=defs+s;
  bindCanvas(ix);
  /* follow the newest experiments (like a git graph opening at HEAD)
     until the user scrolls back into history */
  const canvas=document.getElementById('treecanvas');
  if(!canvas.dataset.bound){
    canvas.dataset.bound='1';
    canvas.addEventListener('scroll',()=>{
      if(TREE.autoScrolling)return;
      TREE.userScrolled=canvas.scrollLeft+canvas.clientWidth
        <canvas.scrollWidth-48;});}
  if(!TREE.userScrolled){
    TREE.autoScrolling=true;
    canvas.scrollLeft=canvas.scrollWidth;
    setTimeout(()=>{TREE.autoScrolling=false;},50);}
}

function bindCanvas(ix){
  const svg=document.getElementById('tree'),
        tip=document.getElementById('treetip');
  svg.querySelectorAll('.tol-node').forEach(el=>{
    const n=ix.byId.get(el.dataset.node);
    el.onclick=ev=>{ev.stopPropagation();
      TREE.sel=TREE.sel===n.id?null:n.id;
      drawTreeOfLife(TREE.data);};
    el.onmouseenter=ev=>{tip.innerHTML=tipHTML(n,ix);
      tip.style.display='block';};
    el.onmousemove=ev=>{
      tip.style.left=Math.min(ev.clientX+16,innerWidth-360)+'px';
      tip.style.top=(ev.clientY+12)+'px';};
    el.onmouseleave=()=>{tip.style.display='none';};});
  svg.onclick=ev=>{
    if(ev.target.dataset&&ev.target.dataset.bg){
      TREE.sel=null;TREE.laneSel=null;drawTreeOfLife(TREE.data);}};
}

function tipHTML(n,ix){
  if(n.kind==='decoder')
    return '<span class="tt-id">Decoder.md v'+n.version+'</span> · '+
      esc(n.decoder_kind)+'<div class="tt-sum">'+
      esc((n.summary||'').slice(0,160))+'</div>';
  if(n.kind==='lineage')
    return '<span class="tt-id">'+esc(n.lineage)+'</span> · '+
      esc(n.lineage_kind)+' · '+esc(n.status)+
      (n.idea?'<div class="tt-sum">'+esc(String(n.idea).slice(0,160))+
      '</div>':'');
  const verdict=n.voided?'voided':n.score==null?'unscored':
    n.kept?'kept':'reverted';
  return '<span class="tt-id">'+esc(n.lineage)+' #'+n.report+'</span>'+
    (n.score!=null?' · <span class="tt-score">'+fmt(n.score)+'</span>':'')+
    ' · '+verdict+(n.audit!=='none'?' · audit '+esc(n.audit):'')+
    (ix.champ&&ix.champ.id===n.id?' · &#9733; champion':'')+
    '<div class="tt-sum">'+esc((n.summary||'').slice(0,180))+'</div>';
}

function badge(text,cls){
  return '<span class="badge '+(cls||'')+'">'+esc(text)+'</span>';}

function jumpButtons(list){
  return list.map(item=>'<button class="jump" data-jump="'+
    esc(item.id)+'">'+esc(item.text)+'</button>').join('');}

function renderInspect(ix){
  const box=document.getElementById('treeinspect');
  const n=TREE.sel?ix.byId.get(TREE.sel):null;
  if(!n){
    box.innerHTML='<div class="sec">inspector</div>'+
      '<p class="dim" style="margin-top:6px">Click any experiment, birth, '+
      'or decoder version.<br><br>&#9679; filled = kept &nbsp; '+
      '&#9675; hollow = reverted<br>gold ring = audit passed &nbsp; '+
      '&#10005; = voided<br>&#9733; = campaign champion</p>'+
      (TREE.laneSel?'<div class="sec">selected lineage</div><p class="prose">'+
        esc(TREE.laneSel)+' — click its card again to release.</p>':'');
    return;}
  const edges=TREE.data.edges;
  const into=edges.filter(e=>e.target===n.id&&e.kind!=='descent'),
        outof=edges.filter(e=>e.source===n.id&&e.kind!=='descent');
  let html='';
  if(n.kind==='decoder'){
    html='<h3>Decoder.md v'+n.version+'</h3>'+
      '<div class="sub">'+esc(n.decoder_kind)+' · shared knowledge</div>'+
      (n.summary?'<p class="prose">'+esc(n.summary)+'</p>':'')+
      (n.preview?'<div class="sec">content</div><pre class="decoderpre">'+
        esc(n.preview)+'</pre>':'');
  }else if(n.kind==='lineage'){
    html='<h3>'+esc(n.lineage)+'</h3>'+
      '<div class="sub">'+esc(n.lineage_kind)+' lineage · '+esc(n.task)+
      '</div><div class="badges">'+
      badge(n.status,n.status==='running'?'good':'bad')+
      (n.worker?badge(n.worker):'')+'</div>'+
      (n.score!=null?'<div class="bigscore">'+fmt(n.score)+
        '<span class="delta dim"> lineage best</span></div>':'')+
      (n.idea?'<div class="sec">assigned direction</div><p class="prose">'+
        esc(typeof n.idea==='string'?n.idea:JSON.stringify(n.idea))+
        '</p>':'')+
      (n.summary?'<div class="sec">founding rationale</div>'+
        '<p class="prose">'+esc(n.summary)+'</p>':'');
  }else{
    const lane=(ix.laneNodes.get(n.lineage)||[]).filter(r=>
      r.kind==='report'&&r.report<n.report&&r.score!=null&&!r.voided&&
      r.kept!==false);
    const prevBest=lane.length?Math.max(...lane.map(r=>r.score)):null;
    const delta=(n.score!=null&&prevBest!=null)?n.score-prevBest:null;
    const isChamp=ix.champ&&ix.champ.id===n.id;
    html='<h3>'+esc(n.lineage)+' · experiment #'+n.report+'</h3>'+
      '<div class="sub">arrival '+(n.arrival+1)+' · Decoder v'+
      n.decoder_version+'</div><div class="badges">'+
      (isChamp?badge('\u2605 champion','champ'):'')+
      (n.voided?badge('voided','bad'):
        n.score==null?badge('unscored','warn'):
        n.kept?badge('kept','good'):badge('reverted',''))+
      (n.audit==='passed'?badge('audit \u2713','good'):
       n.audit==='failed'?badge('audit \u2717','bad'):
       n.audit==='inconclusive'?badge('audit ?','warn'):'')+
      '</div>'+
      (n.score!=null?'<div class="bigscore">'+fmt(n.score)+
        (delta!=null?'<span class="delta '+(delta>=0?'up':'down')+'">'+
        (delta>=0?'+':'')+fmt(delta)+'</span>':'')+'</div>'+
        '<div class="src">trusted · '+esc(n.source||'unknown source')+
        '</div>':'')+
      '<p class="prose">'+esc(n.summary||'')+'</p>'+
      (n.citation?'<div class="sec">cite this checkpoint</div>'+
        '<code class="token" id="copycite" title="click to copy">'+
        esc(n.citation)+'</code>':'');
  }
  if(into.length)html+='<div class="sec">influenced by</div>'+
    jumpButtons(into.map(e=>({id:e.source,
      text:'\u2190 '+e.kind+' · '+e.source+(e.token?' '+e.token:'')})));
  if(outof.length)html+='<div class="sec">influences</div>'+
    jumpButtons(outof.map(e=>({id:e.target,
      text:'\u2192 '+e.kind+' · '+e.target})));
  const unres=(TREE.data.unresolved||[]).filter(u=>u.where===n.id);
  html+=unres.map(u=>'<div class="warnrow">unresolved '+esc(u.token)+
    ': '+esc(u.reason)+'</div>').join('');
  box.innerHTML=html;
  box.querySelectorAll('[data-jump]').forEach(el=>
    el.onclick=()=>treeJump(el.dataset.jump));
  const copy=box.querySelector('#copycite');
  if(copy)copy.onclick=()=>{
    navigator.clipboard&&navigator.clipboard.writeText(copy.textContent);
    copy.textContent=copy.textContent.replace(/ · copied$/,'')+' · copied';};
}

function treeJump(id){
  if(!TREE.data)return;
  TREE.sel=id;TREE.laneSel=null;
  drawTreeOfLife(TREE.data);
  const node=(TREE.data.nodes||[]).find(n=>n.id===id);
  if(!node)return;
  const canvas=document.getElementById('treecanvas');
  canvas.scrollLeft=Math.max(0,nodeX(node)-canvas.clientWidth/2);
  document.getElementById('treepanel')
    .scrollIntoView({behavior:'smooth',block:'nearest'});
}

function initTreeControls(){
  const redraw=()=>TREE.data&&drawTreeOfLife(TREE.data);
  document.getElementById('treesearch').addEventListener('input',redraw);
  document.getElementById('treetask').addEventListener('change',redraw);
  document.getElementById('treetime').addEventListener('input',
    ()=>{TREE.timeDirty=true;redraw();});
  const zoom=document.getElementById('treezoom');
  [['S',30],['M',46],['L',68]].forEach(([label,step])=>{
    const b=document.createElement('button');
    b.textContent=label;
    if(step===TREE.xstep)b.className='on';
    b.onclick=()=>{TREE.xstep=step;
      zoom.querySelectorAll('button').forEach(x=>x.className='');
      b.className='on';redraw();};
    zoom.appendChild(b);});
  document.addEventListener('keydown',e=>{
    if(e.key==='Escape'&&(TREE.sel||TREE.laneSel)){
      TREE.sel=null;TREE.laneSel=null;redraw();}});
}
initTreeControls();

/* ---------------- page tick ---------------- */
function tile(v,k,sub,hot){
  return '<div class="tile'+(hot?' hot':'')+'"><div class="v">'+v+
    '</div><div class="k">'+esc(k)+'</div>'+
    (sub?'<div class="sub">'+sub+'</div>':'')+'</div>';}

function eventChip(text){
  const word=(text.match(/^([A-Za-z-]+)/)||['','note'])[1].toLowerCase();
  const cls=word.replace('revise','audit').replace('void-report','void');
  return '<span class="echip '+esc(cls)+'">'+esc(word)+'</span>';}

async function tick(){
  let d; try{d=await (await fetch('page.json')).json();}catch(e){return;}
  document.title=d.name;
  document.getElementById('title').textContent=d.name;
  document.getElementById('mode').textContent=d.mode;
  document.getElementById('up').textContent=d.up_min+' min up';
  const s=d.summary||{}, tiles=[];
  if(d.mode==='agentic'){
    document.getElementById('taskpills').innerHTML=(s.tasks||[]).map(t=>
      '<span class="pill">'+esc(t)+
      (s.fitness_regime&&s.fitness_regime[t]
        ?' · '+esc(s.fitness_regime[t]):'')+'</span>').join(' ');
    for(const task of (s.tasks||[])){
      const b=(d.best||{})[task],
            sb=(d.sealed_best||{})[task],
            ho=(d.holdout_observe||{})[task],
            audited=(s.audit_passed_best||{})[task],
            baseline=(s.baseline||{})[task],
            champion=(s.verified_champion||{})[task],
            elo=(s.elo_leader||{})[task],
            lead=elo==null?'—':fmt(elo.rating),
            label='Elo selection · '+task;
      tiles.push([lead,label,
        (elo==null?'awaiting judge verdicts':
          '<b>'+esc(elo.lineage)+'</b> · '+elo.games+' matches')+' · '+
        'held-in <b>'+(sb==null?'—':fmt(sb.score))+'</b> · '+
        'holdout <b>'+(ho==null?'—':fmt(ho.score))+'</b> · '+
        'search best <b>'+(b==null?'—':fmt(b.score))+'</b> · '+
        'baseline <b>'+(baseline==null?'—':fmt(baseline))+'</b> · '+
        'verified champion <b>'+(champion==null?'—':fmt(champion))+'</b> · '+
        'audit-passed <b>'+(audited==null?'—':fmt(audited))+'</b>',
        elo!=null]);}
    if(s.plateau_alert)tiles.push(['⚠','campaign plateau',
      Object.entries(s.plateau||{}).filter(x=>x[1]&&x[1].plateaued)
        .map(x=>esc(x[0])).join(' · '),true]);
    tiles.push([s.running+'<span style="font-size:14px;color:var(--ink3)"> / '+
        s.lineages+'</span>','living lineages',
        (s.killed?s.killed+' killed':'all alive')],
      [s.reports,'experiments',
        s.untrusted_reports+' unscored · '+s.voided_reports+' voided'],
      [s.audit_passed_reports,'audit-passed',
        'breeding needs trust; Decoder.md needs audit'],
      [s.running_jobs+'<span style="font-size:14px;color:var(--ink3)"> / '+
        s.queued_jobs+'</span>','native worker jobs',
        'running / queued · '+s.failed_jobs+' failed'],
      [s.verdicts+'<span style="font-size:14px;color:var(--ink3)"> / '+
        s.matches+'</span>','judge verdicts',s.open_matches+' open pairings'],
      ['v'+s.decoder_version,'Decoder.md',
        s.decisions+' decisions · '+s.stale+' stale']);
  } else {
    tiles.push([s.reports,'progress reports',''],
      [s.evaluations,'evaluations','']);
    for(const [task,b] of Object.entries(d.best||{}))
      tiles.push([fmt(b.score),'best · '+task,'',true]);
  }
  document.getElementById('tiles').innerHTML=
    tiles.map(t=>tile(t[0],t[1],t[2],t[3])).join('');
  if(d.mode==='agentic'&&d.tree)drawTreeOfLife(d.tree);
  const series=d.series||{};
  const names=Object.keys(series);
  const hasSealed=names.some(n=>n.includes('· sealed'));
  const hasHoldout=names.some(n=>n.includes('· holdout'));
  const title=document.getElementById('charttitle');
  const note=document.getElementById('chartnote');
  if(title&&note){
    if(hasHoldout){
      title.textContent='held-in vs holdout';
      note.className='chart-note';
      note.textContent='Held-in is what we keep on. Holdout is scored every time and never used to pick a winner.';
    }else if(hasSealed){
      title.textContent='sealed fitness over time';
      note.className='chart-note';
      note.textContent='Sealed is the line that counts. Search can hit 1.0 by fitting cells the makers can see.';
    }else if(d.mode==='agentic'){
      title.textContent='in-sample fitness — not confirmation';
      note.className='chart-note warn';
      note.textContent='No sealed scores. This climb is the same data the detectors can see. Do not treat it as working.';
    }else{
      title.textContent='fitness over time';
      note.className='chart-note';
      note.textContent='';
    }
  }
  drawChart(document.getElementById('chart'), series, d.points||[],
    {xlabel: d.mode==='agentic'?'experiment arrival':'evaluations',
     footer: hasHoldout?'held-in best ↑ · holdout best is observation, not selection':
       (hasSealed?'sealed best ↑ · search is in-sample':'in-sample only — not evidence')});
  if(d.decoder&&d.decoder.artifact!==undefined){
    document.getElementById('decoderpanel').style.display='';
    document.getElementById('decodermeta').textContent=
      ' · v'+d.decoder.version+' · '+d.decoder.kind+
      (d.decoder.rationale?' · '+d.decoder.rationale.slice(0,90):'');
    document.getElementById('decoderbody').textContent=
      d.decoder.artifact||'(empty — nothing incorporated yet)';}
  const media=d.media||{}, strips=d.filmstrips||{},
        mkeys=Object.keys(media).sort();
  if(mkeys.length){
    document.getElementById('mediapanel').style.display='';
    document.getElementById('media').innerHTML=mkeys.map(k=>{
      const m=media[k]; let body;
      if(m.kind==='image') body='<img class="pix big" src="'+m.data+'">';
      else if(m.kind==='svg') body='<div class="svgwrap">'+m.data+'</div>';
      else body='<pre class="mediatext">'+esc(m.data.slice(0,2000))+'</pre>';
      const strip=(strips[k]||[]).slice(0,-1).map(s=>
        '<img class="pix" src="'+s+'">').join('');
      return '<figure class="mediacard">'+body+
        (strip?'<div class="filmstrip">'+strip+'</div>':'')+
        '<figcaption>'+esc(k)+
        (m.epoch!=null?' · epoch '+m.epoch:'')+'</figcaption></figure>';
    }).join('');
  }
  if(d.mode==='agentic'&&(d.lineages||[]).length){
    document.getElementById('linepanel').style.display='';
    const scored=d.lineages.filter(l=>l.best!=null).map(l=>l.best);
    const max=Math.max(...scored), min=Math.min(...scored);
    document.getElementById('lineages').innerHTML=d.lineages.map(l=>{
      const w=(l.best!=null&&max>min)?4+80*(l.best-min)/(max-min):42;
      return '<tr><td><button data-tree-lineage="'+esc(l.id)+
      '" style="border:0;background:none;color:var(--accent);'+
      'font:600 12.5px var(--mono);cursor:pointer;padding:0">'+esc(l.id)+
      '</button></td><td>'+esc(l.task)+
      '</td><td><span class="badge '+esc(l.kind)+'">'+esc(l.kind)+
      '</span></td><td><span class="statusdot '+esc(l.status)+
      '"></span>'+esc(l.status)+'</td><td class="dim">'+esc(l.worker)+
      '</td><td><span class="num">'+fmt(l.elo)+'</span> '+
      '<span class="dim">'+l.wins+'–'+l.losses+'–'+l.ties+'</span>'+
      '</td><td>'+(l.best!=null?'<span class="scorebar"><i style="width:'+
      w+'px"></i></span><span class="num">'+fmt(l.best)+'</span>':
      '<span class="dim">—</span>')+'</td><td class="num">'+l.reports+
      (l.pending?' <span class="dim">('+l.pending+' unscored)</span>':'')+
      (l.stale?' <span class="badge warn">stale</span>':'')+
      '</td><td>'+(l.audit==='passed'?'<span class="badge good">\u2713 passed</span>':
        l.audit==='failed'?'<span class="badge bad">failed</span>':
        l.audit==='inconclusive'?'<span class="badge warn">?</span>':
        '<span class="dim">—</span>')+
      '</td><td class="dim" title="'+esc(l.last)+'">'+esc(l.last)+
      '</td></tr>';
    }).join('');
    document.querySelectorAll('[data-tree-lineage]').forEach(el=>
      el.onclick=()=>treeJump(el.dataset.treeLineage+':born'));
  }
  if(d.mode==='agentic'&&(d.matches||[]).length){
    document.getElementById('matchpanel').style.display='';
    document.getElementById('matches').innerHTML=d.matches.slice().reverse()
      .map(m=>{
        const ids=(m.individuals||[]).map(x=>x.citation||x.lineage).join(' vs '),
              v=m.verdict,
              winner=!v?'open':(v.winner||'tie'),
              delta=m.elo_update&&m.elo_update.delta,
              deltaText=delta?Object.entries(delta).map(([id,n])=>
                esc(id)+' '+(n>=0?'+':'')+fmt(n)).join(' · '):'—';
        return '<tr><td class="num">'+esc(m.id)+'</td><td>'+esc(ids)+
          '</td><td class="dim">'+esc((v&&v.judge)||m.judge||'unassigned')+
          '</td><td>'+esc(m.voided?'voided':winner)+'</td><td class="dim">'+
          deltaText+'</td></tr>';}).join('');
  }
  if(d.mode==='agentic'&&(d.jobs||[]).length){
    document.getElementById('jobpanel').style.display='';
    document.getElementById('jobs').innerHTML=d.jobs.slice().reverse()
      .map(j=>{
        const attempts=j.attempts||[], last=attempts[attempts.length-1],
              attempt=last?last.number+'/'+j.max_attempts:'0/'+j.max_attempts;
        return '<tr><td class="num">'+esc(j.id)+'</td><td>'+esc(j.lane)+
          '</td><td>'+esc(j.kind)+'</td><td class="dim">'+esc(j.name)+
          '</td><td><span class="statusdot '+esc(j.status)+'"></span>'+
          esc(j.status)+'</td><td class="num">'+esc(attempt)+'</td></tr>';})
      .join('');
  }
  if(d.mode==='agentic'&&(d.decisions||[]).length){
    /* the persistent record, newest first — every evolutionary decision
       with its rationale, straight from the campaign state */
    document.getElementById('streamtitle').textContent='decision log';
    document.getElementById('events').innerHTML=d.decisions.slice()
      .reverse().map(dec=>{
        const extra=dec.action==='audit'?' → '+dec.outcome:
          dec.action==='revise'?' → '+fmt(dec.score):
          (dec.action==='share'||dec.action==='incorporate'||dec.action==='compact')?
            ' → Decoder v'+dec.version:'';
        const text=(dec.lineages||[]).join(' ')+extra+' — '+
          (dec.rationale||'');
        return '<div class="row"><span class="t">#'+dec.index+'</span>'+
          eventChip(dec.action)+'<span class="x" title="'+esc(text)+
          '">'+esc(text)+'</span></div>';}).join('');
  } else {
    document.getElementById('streamtitle').textContent='event stream';
    document.getElementById('events').innerHTML=(d.events||[]).slice()
      .reverse().map(e=>'<div class="row"><span class="t">'+esc(e[0])+
      '</span>'+eventChip(e[1])+'<span class="x" title="'+esc(e[1])+'">'+
      esc(e[1])+'</span></div>').join('');
  }
}
tick(); setInterval(tick, 2000);
"""


GETS = {"summary", "lineages", "decisions", "decoder", "tree", "stale",
        "ratings", "matches", "match", "research", "jobs", "page.json"}
POSTS = {"found", "report", "seal-score", "holdout-score", "revise-score", "void-report", "kill", "revive", "assign", "audit",
         "note", "share", "finding", "incorporate", "compact", "baseline",
         "regime", "champion", "plateau", "criteria", "pair", "verdict",
         "void-match", "telemetry", "media"}


def make_handler(service):
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, code, payload):
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _route(self, allowed):
            parsed = urlsplit(self.path)
            name = parsed.path.lstrip("/")
            if name in ("", "progress") and allowed is GETS:
                page = service.progress_html().encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)
                return
            if name not in allowed:
                self._reply(404, {"error": f"unknown route {name!r}"})
                return
            try:
                if allowed is GETS:
                    body = {key: values[-1] for key, values in
                            parse_qs(parsed.query).items()}
                else:
                    n = int(self.headers.get("Content-Length") or 0)
                    body = json.loads(self.rfile.read(n) or b"{}")
                self._reply(200, service.handle(name, body))
            except KeyError as e:
                self._reply(404, {"error": f"unknown id {e}"})
            except Exception as e:
                self._reply(400, {"error": repr(e)})

        def do_GET(self):
            self._route(GETS)

        def do_POST(self):
            self._route(POSTS)

        def log_message(self, fmt, *args):
            # The dashboard polls this endpoint continuously.  Logging every
            # poll can fill a detached PTY buffer and eventually block the
            # otherwise healthy server, so retain only actionable traffic.
            message = str(args[0]) if args else ""
            if "GET /page.json " in message:
                return
            print(f"[serve] {message}", flush=True)
    return Handler


def serve(run_dir, port=0, tasks=None, telemetry_only=False):
    """Build (or load) the run's campaign record and return a ready
    server. port=0 picks a free port. Caller runs .serve_forever().
    telemetry_only=True starts the same server with no campaign — the
    dashboard for tensor solve() runs (see live_progress)."""
    os.makedirs(run_dir, exist_ok=True)
    if telemetry_only:
        campaign, state_path = None, None
    else:
        state_path = os.path.join(run_dir, "state.json")
        decoder_path = os.path.join(run_dir, "Decoder.md")
        if os.path.exists(state_path):
            campaign = Campaign.load(state_path)
        else:
            if not tasks:
                raise SystemExit("no state.json — pass --tasks to "
                                 "create a run")
            initial_decoder = ""
            if os.path.exists(decoder_path):
                with open(decoder_path, encoding="utf-8") as f:
                    initial_decoder = f.read()
            campaign = Campaign(tasks=tasks, decoder=initial_decoder)
            campaign.save(state_path)
        if os.path.exists(decoder_path):
            with open(decoder_path, encoding="utf-8") as f:
                on_disk_decoder = f.read()
            if on_disk_decoder != campaign.current_decoder()["artifact"]:
                raise ValueError(
                    "Decoder.md differs from Finch's versioned campaign "
                    "record; publish the edit through /share, /finding, "
                    "/incorporate, or /compact instead of editing it directly")
        else:
            with open(decoder_path, "w", encoding="utf-8") as f:
                f.write(campaign.current_decoder()["artifact"])
    service = GAService(campaign, state_path, run_dir=run_dir)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(service))
    server.service = service
    with open(os.path.join(run_dir, "server.json"), "w") as f:
        json.dump({"port": server.server_address[1], "pid": os.getpid()}, f)
    register_run(run_dir, server.server_address[1])
    return server


def live_progress(run_dir=None, port=0, names=None, images="auto"):
    """One dashboard for every run: pass the result as solve()'s
    progress= callback and open the printed URL.

        from finch4 import solve, live_progress
        solve(fitness_fns, output_shape=(64, 64), epochs=10_000,
              progress=live_progress())

    Starts a telemetry-only reporting server (same /progress page the
    agentic substrate uses) in a daemon thread and returns a callback
    with solve()'s progress signature. names labels the fitness
    functions on the chart (default fn0, fn1, ...). The default run
    directory lives under ~/.finch4/runs/ — deliberately persistent, so
    the hub keeps showing the run after it finishes.

    images="auto" additionally posts each function's best-ever phenotype
    to the dashboard as a PNG whenever it is image-shaped ((H, W),
    (H, W, C<=4) or (C<=4, H, W)); "off" disables. Anything else goes
    through progress.report_media(name, image=|svg=|text=) — or, from a
    separate process, media_client(run_dir). Media is telemetry for the
    eyes, never evidence."""
    run_dir = run_dir or os.path.expanduser(
        f"~/.finch4/runs/live-{os.getpid()}-{int(time.time())}")
    server = serve(run_dir, port=port, telemetry_only=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/progress"
    print(f"[finch4] live progress: {url}", flush=True)

    def progress(epoch, epochs, evaluations, best_pheno, best_score):
        best = {}
        for i, score in enumerate(best_score):
            label = names[i] if names and i < len(names) else f"fn{i}"
            value = float(score)
            if value == value and abs(value) != float("inf"):
                best[label] = value
        server.service.handle("telemetry", {
            "epoch": int(epoch), "evaluations": int(evaluations),
            "best": best})
        server.service.event(f"epoch {epoch}/{epochs} "
                             f"evals={evaluations} best={best}")
        if images == "auto":
            for i, pheno in enumerate(best_pheno):
                if pheno is None or not looks_like_image(pheno.shape):
                    continue
                label = names[i] if names and i < len(names) else f"fn{i}"
                server.service.handle("media", {
                    "name": f"best {label}", "kind": "image",
                    "data": png_data_uri(pheno),
                    "epoch": int(epoch),
                    "evaluations": int(evaluations)})

    def report_media(name, image=None, svg=None, text=None, epoch=None,
                     evaluations=None):
        return server.service.handle("media", _media_body(
            name, image=image, svg=svg, text=text, epoch=epoch,
            evaluations=evaluations))

    progress.url = url
    progress.server = server
    progress.run_dir = run_dir
    progress.report_media = report_media
    return progress


def media_client(run_dir):
    """Media POSTer for a separate process (a worker, a wrapper around
    an evaluator) — discovers the run's live server via server.json, the
    same discovery workers use to stream reports. Returns
    send(name, image=|svg=|text=, epoch=None, evaluations=None)."""
    import urllib.request
    port = json.load(open(os.path.join(run_dir, "server.json")))["port"]

    def send(name, image=None, svg=None, text=None, epoch=None,
             evaluations=None):
        body = json.dumps(_media_body(
            name, image=image, svg=svg, text=text, epoch=epoch,
            evaluations=evaluations)).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/media", data=body, method="POST")
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())

    return send


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", required=True)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--tasks", nargs="*", default=None)
    p.add_argument("--telemetry", action="store_true",
                   help="no campaign: dashboard-only server for solver "
                        "runs")
    a = p.parse_args()
    server = serve(a.run, a.port, a.tasks, telemetry_only=a.telemetry)
    print(f"[serve] run={a.run} port={server.server_address[1]}",
          flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
