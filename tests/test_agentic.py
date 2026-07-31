"""The ask/tell engine driven by scripted fake agents — every law the
agentic substrate relies on, checked without spawning anything."""
import numpy as np

from finch4 import AgenticGA


def fake_score(task, variation):
    """Deterministic 'fitness': rewards longer variations, task-scaled so
    raw scores are NOT comparable across tasks (shares must handle it)."""
    scale = 1.0 if task == "alpha" else 1000.0
    return scale * (len(variation) % 17)


def drive(ga, rounds):
    for _ in range(rounds):
        for job in ga.ask():
            parents = job["parents"]
            if job["kind"] == "found":
                var = f"found-{job['job_id']}"
            elif job["kind"] == "mutate":
                var = parents[0]["variation"] + "-m"
            else:
                var = parents[0]["variation"] + "+" + parents[1]["variation"]
            ga.tell(job["job_id"], var, fake_score(job["task"], var),
                    artifact=f"/tmp/{job['job_id']}.py")
        if ga.consolidation_due():
            batch = ga.consolidation_batch()
            assert set(batch) <= {"alpha", "beta"}
            for ind in ga.record_consolidation():
                ga.tell_rewrite(ind["id"], ind["variation"] + "!")


def test_founding_covers_every_task():
    ga = AgenticGA(tasks=["alpha", "beta"], founders=3, seed=0)
    jobs = ga.ask()
    assert len(jobs) == 6
    assert {j["task"] for j in jobs} == {"alpha", "beta"}
    assert all(j["kind"] == "found" for j in jobs)


def test_cap_and_shares_keep_both_tasks_alive():
    ga = AgenticGA(tasks=["alpha", "beta"], founders=2, population_cap=6,
                   children=4, consolidate_every=1000, seed=1)
    drive(ga, rounds=8)
    living_tasks = {i["task"] for i in ga.individuals.values() if i["alive"]}
    # beta scores 1000x higher raw, yet shares must protect alpha members
    assert living_tasks == {"alpha", "beta"}
    assert sum(i["alive"] for i in ga.individuals.values()) <= 6


def test_extinct_task_is_refounded():
    ga = AgenticGA(tasks=["alpha", "beta"], founders=1, seed=2)
    for job in ga.ask():
        ga.tell(job["job_id"], "x", fake_score(job["task"], "x"))
    for ind in ga.individuals.values():        # kill beta by hand
        if ind["task"] == "beta":
            ind["alive"] = False
    jobs = ga.ask()
    assert any(j["kind"] == "found" and j["task"] == "beta" for j in jobs)


def test_consolidation_cadence_staleness_and_rewrite():
    ga = AgenticGA(tasks=["alpha"], founders=2, children=2,
                   consolidate_every=2, seed=3)

    def round_no_consolidate():
        for job in ga.ask():
            var = (job["parents"][0]["variation"] + "-m"
                   if job["parents"] else f"found-{job['job_id']}")
            ga.tell(job["job_id"], var, fake_score("alpha", var))

    round_no_consolidate()
    assert not ga.consolidation_due()
    round_no_consolidate()
    assert ga.consolidation_due()
    survivors = ga.record_consolidation()
    assert ga.base_version == 1
    assert len(ga.stale()) == len(survivors)   # no automatic re-score
    ga.tell_rewrite(survivors[0]["id"], "rewritten")
    assert ga.individuals[survivors[0]["id"]]["origin"] == "rewrite"
    assert len(ga.stale()) == len(survivors)   # rewrite alone stays stale
    ga.retell_score(survivors[0]["id"], 5.0)
    assert len(ga.stale()) == len(survivors) - 1


def test_best_ever_survives_culling():
    ga = AgenticGA(tasks=["alpha"], founders=1, population_cap=2,
                   children=3, consolidate_every=1000, seed=4)
    drive(ga, rounds=6)
    best = ga.best["alpha"]
    scores = [i["score"] for i in ga.individuals.values()]
    assert best is not None and best["score"] == max(scores)


def test_contradiction_report_and_flags():
    ga = AgenticGA(tasks=["alpha"], founders=2, seed=5)
    jobs = ga.ask()
    ga.tell(jobs[0]["job_id"], "obedient", 1.0)
    ga.tell(jobs[1]["job_id"], "heretic", 9.0, contradicts_base=True)
    rep = ga.contradiction_report()["alpha"]
    assert rep["contradicting"] == 9.0 and rep["compliant"] == 1.0


def test_save_load_roundtrip_is_deterministic(tmp_path):
    ga = AgenticGA(tasks=["alpha", "beta"], founders=2, children=4, seed=6)
    drive(ga, rounds=3)
    path = tmp_path / "state.json"
    ga.save(str(path))
    ga2 = AgenticGA.load(str(path))
    assert ga2.summary() == ga.summary()
    # identical RNG state -> identical future jobs
    jobs_a = [(j["kind"], j["task"], [p["id"] for p in j["parents"]])
              for j in ga.ask()]
    jobs_b = [(j["kind"], j["task"], [p["id"] for p in j["parents"]])
              for j in ga2.ask()]
    assert jobs_a == jobs_b


def test_abandoned_job_leaves_no_trace():
    ga = AgenticGA(tasks=["alpha"], founders=2, seed=7)
    jobs = ga.ask()
    ga.tell(jobs[0]["job_id"], "ok", 1.0)
    ga.abandon(jobs[1]["job_id"])
    assert ga.summary()["open_jobs"] == 0
    assert len(ga._living()) == 1


def test_audit_correction_evicts_false_best_ever():
    ga = AgenticGA(tasks=["alpha"], founders=2, seed=8)
    jobs = ga.ask()
    cheat = ga.tell(jobs[0]["job_id"], "cheat", 1000.0)
    honest = ga.tell(jobs[1]["job_id"], "honest", 5.0)
    assert ga.best["alpha"]["id"] == cheat
    ga.retell_score(cheat, -99.0)          # audit exposes the fraud
    assert ga.best["alpha"]["id"] == honest
    assert ga.best["alpha"]["score"] == 5.0


def test_lineage_exhaustion_refound_and_no_more_breeding():
    ga = AgenticGA(tasks=["alpha"], founders=1, children=3, seed=9)
    [job] = ga.ask()
    parent = ga.tell(job["job_id"], "the idea", 5.0)
    mut = [j for j in ga.ask() if j["kind"] != "found"][0]
    child = ga.tell(mut["job_id"], "a totally new idea", 3.0,
                    fresh_start=True)
    assert ga.individuals[child]["origin"] == "refound"
    assert ga.individuals[parent]["exhausted"] is True
    # the exhausted parent never breeds again (archive still holds it)
    for j in ga.ask():
        assert all(p["id"] != parent for p in j["parents"])
    assert ga.best["alpha"]["id"] == parent      # best-ever survives


def test_fully_exhausted_task_is_refounded():
    ga = AgenticGA(tasks=["alpha"], founders=1, children=2, seed=10)
    [job] = ga.ask()
    only = ga.tell(job["job_id"], "x", 1.0)
    ga.individuals[only]["exhausted"] = True
    jobs = ga.ask()
    assert any(j["kind"] == "found" for j in jobs)
