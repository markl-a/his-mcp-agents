"""MCP server exposing HIS tools.

Design rules:
- read tools return de-identified data only (pseudonymous pid, age band)
- every call is written to the audit log
- write tools are two-phase: the agent can only *draft*; a clinician must confirm with the returned token
"""
from __future__ import annotations

import secrets
import time

from mcp.server.fastmcp import FastMCP

from . import audit, drug_rules
from .store import FhirStore

mcp = FastMCP("his-tools")
STORE = FhirStore()
_PENDING: dict[str, dict] = {}
DRAFT_TTL_S = 900


def _log(tool: str, args: dict, outcome: str = "ok") -> None:
    audit.record(tool, args, actor="agent", outcome=outcome)


@mcp.tool()
def list_patients() -> list[dict]:
    """List patients as pseudonymous tokens with age band and gender."""
    _log("list_patients", {})
    return STORE.patients()


@mcp.tool()
def get_encounters(pid: str) -> list[dict]:
    """Encounters (date, type, department, reason) for a pseudonymous patient id."""
    _log("get_encounters", {"pid": pid})
    return STORE.encounters(pid)


@mcp.tool()
def get_conditions(pid: str) -> list[dict]:
    """Active diagnoses (ICD-10 code and text)."""
    _log("get_conditions", {"pid": pid})
    return STORE.conditions(pid)


@mcp.tool()
def get_observations(pid: str, abnormal_only: bool = False) -> list[dict]:
    """Lab / vital observations with H/L flag against reference range."""
    _log("get_observations", {"pid": pid, "abnormal_only": abnormal_only})
    return STORE.observations(pid, abnormal_only)


@mcp.tool()
def get_medications(pid: str) -> list[dict]:
    """Active medication requests."""
    _log("get_medications", {"pid": pid})
    return STORE.medications(pid)


@mcp.tool()
def check_interactions(pid: str, extra_drugs: list[str] | None = None) -> list[dict]:
    """Check pairwise interactions among active meds plus optional proposed drugs."""
    _log("check_interactions", {"pid": pid, "extra": extra_drugs or []})
    drugs = [m["drug"] for m in STORE.medications(pid)] + list(extra_drugs or [])
    return drug_rules.check(drugs)


@mcp.tool()
def draft_medication_request(pid: str, drug: str, dose: str, rationale: str) -> dict:
    """Create a DRAFT order. It is not written to the HIS until a clinician confirms it."""
    STORE.resolve(pid)
    token = secrets.token_urlsafe(12)
    _PENDING[token] = {"pid": pid, "drug": drug, "dose": dose, "rationale": rationale, "created": time.time()}
    _log("draft_medication_request", {"pid": pid, "drug": drug, "dose": dose})
    warnings = check_interactions(pid, [drug])
    return {"status": "draft", "confirm_token": token, "interaction_warnings": warnings}


def confirm_medication_request(token: str, clinician_id: str) -> dict:
    """Called by the clinician UI, NOT exposed as an MCP tool: the model cannot confirm its own drafts."""
    draft = _PENDING.pop(token, None)
    if draft is None or time.time() - draft["created"] > DRAFT_TTL_S:
        audit.record("confirm_medication_request", {"token": token}, actor=clinician_id, outcome="rejected")
        raise PermissionError("unknown or expired draft token")
    audit.record("confirm_medication_request", {"pid": draft["pid"], "drug": draft["drug"]}, actor=clinician_id)
    return {"status": "submitted", **{k: draft[k] for k in ("pid", "drug", "dose")}}


if __name__ == "__main__":
    mcp.run()
