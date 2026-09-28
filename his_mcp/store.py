"""Read-only view over a FHIR-shaped store. In production this would call the HIS FHIR API with a service account."""
from __future__ import annotations

import json
from pathlib import Path

from .deid import deidentify_patient, pseudonym

_SEED = Path(__file__).resolve().parent.parent / "data" / "fhir_seed.json"


class FhirStore:
    def __init__(self, path: Path = _SEED):
        self.db = json.loads(path.read_text(encoding="utf-8"))
        self._by_pid = {pseudonym(p["id"]): p["id"] for p in self.db["Patient"]}

    def resolve(self, pid: str) -> str:
        if pid not in self._by_pid:
            raise KeyError(f"unknown patient token {pid}")
        return self._by_pid[pid]

    def patients(self) -> list[dict]:
        return [deidentify_patient(p) for p in self.db["Patient"]]

    def _rows(self, kind: str, pid: str) -> list[dict]:
        real = self.resolve(pid)
        return [{k: v for k, v in r.items() if k not in ("patient", "id")} for r in self.db[kind] if r["patient"] == real]

    def encounters(self, pid: str) -> list[dict]:
        return self._rows("Encounter", pid)

    def conditions(self, pid: str) -> list[dict]:
        return self._rows("Condition", pid)

    def medications(self, pid: str) -> list[dict]:
        return [m for m in self._rows("MedicationRequest", pid) if m["status"] == "active"]

    def observations(self, pid: str, abnormal_only: bool = False) -> list[dict]:
        rows = self._rows("Observation", pid)
        for r in rows:
            r["flag"] = "H" if "ref_high" in r and r["value"] > r["ref_high"] else "L" if "ref_low" in r and r["value"] < r["ref_low"] else ""
        return [r for r in rows if r["flag"]] if abnormal_only else rows
