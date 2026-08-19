"""Scripted stand-in for the solo agent CLI (tests/test_solo.py).

Runs in the lineage working directory. Copies champion.py to candidate.py,
applies best-fit instead of worst-fit, and leaves keep/revert to the driver.
"""
import json
import re
import sys
from pathlib import Path

prompt = Path(sys.argv[1]).read_text(encoding="utf-8")
number = int(re.search(r"experiment (\d+) of", prompt).group(1))
champ_score = float(re.search(r"Your fitness is ([0-9eE.+-]+)", prompt).group(1))

hypothesis = "best-fit (fullest feasible bin) beats worst-fit priority"
source = Path("champion.py").read_text(encoding="utf-8")
Path("candidate.py").write_text(
    source.replace("return capacities", "return -capacities"),
    encoding="utf-8")

claimed = champ_score
try:
    import subprocess
    claimed = json.loads(subprocess.run(
        [sys.executable, "score.py", "candidate.py"],
        capture_output=True, text=True, check=True).stdout)["score"]
except Exception:
    pass

with open("WORKLOG.md", "a", encoding="utf-8") as handle:
    handle.write(f"## Experiment {number}\nHypothesis: {hypothesis}\n"
                 f"Proposed: {claimed} vs champion {champ_score}\n\n")
with open("results.jsonl", "a", encoding="utf-8") as handle:
    handle.write(json.dumps({
        "experiment": number,
        "hypothesis": hypothesis,
        "claimed_score": claimed,
    }) + "\n")
print("PROPOSED", claimed)
