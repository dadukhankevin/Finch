"""A scripted stand-in for the agent CLI, used by the driver tests.
Dispatches on the env pointers the driver sets: a decoder job if
AGENTIC_JOB is present, a consolidation if AGENTIC_PROPOSAL, a rewrite
if AGENTIC_SURVIVORS."""
import json
import os
import subprocess
import sys
import urllib.request


def post(route, body):
    req = urllib.request.Request(
        f"http://127.0.0.1:{os.environ['AGENTIC_PORT']}/{route}",
        data=json.dumps(body).encode(), method="POST")
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


ARTIFACTS = {
    "binpack": "def priority(item, capacities):\n    return -capacities\n",
    "tsp": ("import numpy as np\n"
            "def next_city(current, unvisited, coords):\n"
            "    d = np.linalg.norm(coords[unvisited] - coords[current],"
            " axis=1)\n"
            "    return unvisited[int(np.argmin(d))]\n"),
}


def decoder():
    job = json.loads(os.environ["AGENTIC_JOB"])
    out = os.environ["AGENTIC_OUT"]
    artifact = os.path.join(out, "artifact.py")
    with open(artifact, "w") as f:
        f.write(f"# {job['job_id']}\n" + ARTIFACTS[job["task"]])
    scorer = os.path.join(os.environ["AGENTIC_TASKS_DIR"], job["task"],
                          "score.py")
    result = json.loads(subprocess.run(
        [sys.executable, scorer, artifact],
        capture_output=True, text=True).stdout)
    with open(os.path.join(out, "score.json"), "w") as f:
        json.dump(result, f)
    post("tell", {"job_id": job["job_id"],
                  "variation": f"fake variation from {job['job_id']} "
                               f"({job['kind']})",
                  "score": result["score"], "artifact": artifact})


def consolidator():
    with open(os.path.join(os.environ["AGENTIC_RUN"], "base_playbook.md")) as f:
        lines = f.read().split("\n")
    lines[0] = "# Base playbook — consolidated by fake"
    with open(os.environ["AGENTIC_PROPOSAL"], "w") as f:
        f.write("\n".join(lines))


def rewriter():
    with open(os.environ["AGENTIC_SURVIVORS"]) as f:
        survivors = json.load(f)
    post("rewrite", {"id": survivors[0]["id"],
                     "variation": survivors[0]["variation"] + " — deeper"})


if "AGENTIC_JOB" in os.environ:
    decoder()
elif "AGENTIC_PROPOSAL" in os.environ:
    consolidator()
elif "AGENTIC_SURVIVORS" in os.environ:
    rewriter()
