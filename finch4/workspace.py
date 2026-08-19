"""Worker workspaces contain only what a researcher should see."""
from __future__ import annotations

import shutil
from pathlib import Path


SEALED_NAMES = ("cells.json", "corpus_manifest.json")

SMOKE_SCORER = '''"""Smoke check. Not campaign fitness."""
import json
import sys
from pathlib import Path

def main() -> None:
    path = Path(sys.argv[1])
    namespace: dict = {}
    try:
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"),
             namespace)
    except Exception as exc:
        print(json.dumps({"score": 0.0, "split": "smoke",
                          "errors": [type(exc).__name__]}))
        return
    fn = namespace.get("flag") or namespace.get("priority")
    print(json.dumps({"score": 1.0 if callable(fn) else 0.0,
                      "split": "smoke"}))

if __name__ == "__main__":
    main()
'''


def task_dir(tasks_dir, task) -> Path:
    return Path(tasks_dir).resolve() / task


def has_sealed(directory) -> bool:
    root = Path(directory)
    return any((root / name).exists() for name in SEALED_NAMES)


def prepare_worker_workspace(run, tasks_dir, task) -> Path:
    """Fill a lineage directory with public files only. Never copy sealed data."""
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    source = task_dir(tasks_dir, task)
    brief = source / "TASK.md"
    if brief.exists():
        shutil.copy2(brief, run / "TASK.md")
    if has_sealed(source):
        (run / "score.py").write_text(SMOKE_SCORER, encoding="utf-8")
    else:
        shutil.copy2(source / "score.py", run / "score.py")
    return run


def refresh_decoder(run, decoder) -> str:
    if not decoder:
        return ""
    text = Path(decoder).read_text(encoding="utf-8")
    Path(run, "Decoder.md").write_text(text, encoding="utf-8")
    return text
