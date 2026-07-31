"""A scripted stand-in for the solo agent CLI (tests/test_solo.py).

Receives the experiment prompt file as argv[1] and performs exactly the
autoresearch protocol the prompt describes — copy the champion to the
candidate, apply ONE change (best-fit instead of worst-fit bin priority),
score with the canonical scorer, keep only if strictly better, log
honestly — without any model. On a champion that already has the change
the candidate ties, so experiment one is KEPT and experiment two
REVERTED, exercising both verdicts."""
import json
import re
import shutil
import subprocess
import sys

prompt = open(sys.argv[1]).read()

worklog = re.search(r"result\): (\S+)", prompt).group(1)
champion, champ_score = re.search(
    r"champion script: (\S+) \(canonical score (\S+)\)", prompt).groups()
scorer = re.search(r"NEVER edit it\): (\S+)", prompt).group(1)
candidate = re.search(r"Copy the champion to (\S+) and", prompt).group(1)
results = re.search(r"JSON line to (\S+):", prompt).group(1)
number = int(re.search(r"experiment (\d+) of", prompt).group(1))
champ_score = float(champ_score)

hypothesis = "best-fit (fullest feasible bin) beats worst-fit priority"
source = open(champion).read()
with open(candidate, "w") as f:
    f.write(source.replace("return capacities", "return -capacities"))

score = json.loads(subprocess.run(
    [sys.executable, scorer, candidate],
    capture_output=True, text=True).stdout)["score"]

kept = score > champ_score
if kept:
    shutil.copy(candidate, champion)

with open(worklog, "a") as f:
    f.write(f"## Experiment {number}\nHypothesis: {hypothesis}\n"
            f"Canonical score: {score} vs champion {champ_score} — "
            f"{'KEPT' if kept else 'REVERTED'}\n\n")
with open(results, "a") as f:
    f.write(json.dumps({"experiment": number, "hypothesis": hypothesis,
                        "score": score, "kept": kept}) + "\n")
print("KEPT" if kept else "REVERTED", score)
