"""Append-only audit trail: who called which tool, with what (hashed) arguments, under which model and prompt version."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

AUDIT_PATH = Path(os.environ.get("HIS_AUDIT_LOG", "audit.jsonl"))


def record(tool: str, args: dict, actor: str, model: str = "n/a", prompt_version: str = "n/a", outcome: str = "ok") -> dict:
    entry = {
        "ts": round(time.time(), 3),
        "actor": actor,
        "tool": tool,
        "args_sha256": hashlib.sha256(json.dumps(args, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16],
        "model": model,
        "prompt_version": prompt_version,
        "outcome": outcome,
    }
    with AUDIT_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry
