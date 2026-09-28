import json

import pytest

from his_mcp import audit, deid, server
from his_mcp.store import FhirStore
from agents.graph import build


@pytest.fixture(autouse=True)
def tmp_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "AUDIT_PATH", tmp_path / "audit.jsonl")
    yield tmp_path / "audit.jsonl"


def pid_of(age_band="70-79"):
    return next(p["pid"] for p in server.list_patients() if p["age_band"] == age_band)


def test_patient_list_is_deidentified():
    rows = server.list_patients()
    blob = json.dumps(rows, ensure_ascii=False)
    for raw in ("王小明", "A123456789", "1948-03-02", "p001"):
        assert raw not in blob
    assert all(r["pid"].startswith("pt_") for r in rows)


def test_pseudonym_is_stable_and_unguessable():
    assert deid.pseudonym("p001") == deid.pseudonym("p001")
    assert deid.pseudonym("p001") != deid.pseudonym("p002")


def test_abnormal_flags():
    store = FhirStore()
    pid = next(p["pid"] for p in store.patients() if p["gender"] == "female")
    flags = {o["code"]: o["flag"] for o in store.observations(pid, abnormal_only=True)}
    assert flags == {"eGFR": "L", "K": "H"}


def test_every_tool_call_is_audited(tmp_audit):
    pid = server.list_patients()[0]["pid"]
    server.get_medications(pid)
    lines = [json.loads(l) for l in tmp_audit.read_text().splitlines()]
    assert [l["tool"] for l in lines] == ["list_patients", "get_medications"]
    assert all("args_sha256" in l for l in lines)


def test_draft_needs_clinician_confirmation():
    pid = server.list_patients()[0]["pid"]
    draft = server.draft_medication_request(pid, "simvastatin", "20 mg qn", "LDL above target")
    assert draft["status"] == "draft"
    assert any(w["severity"] == "major" for w in draft["interaction_warnings"])  # simvastatin + clarithromycin
    done = server.confirm_medication_request(draft["confirm_token"], clinician_id="dr_lee")
    assert done["status"] == "submitted"
    with pytest.raises(PermissionError):  # token is single-use
        server.confirm_medication_request(draft["confirm_token"], clinician_id="dr_lee")


def test_confirm_is_not_an_mcp_tool():
    import asyncio
    names = {t.name for t in asyncio.run(server.mcp.list_tools())}
    assert "draft_medication_request" in names and "confirm_medication_request" not in names


def test_graph_brief_is_grounded_and_traced():
    out = build().invoke({"pid": server.list_patients()[1]["pid"], "trace": []})
    assert "spironolactone + lisinopril" in out["brief"] or "lisinopril + spironolactone" in out["brief"]
    assert "[tool:" in out["brief"] and ".md:L" in out["brief"]
    nodes = [t["node"] for t in out["trace"]]
    assert nodes[0] == "gather" and nodes[-1] == "writer" and "interactions" in nodes
    assert out["trace"][-1]["prompt_version"] == "brief-v3"


def test_supervisor_skips_interactions_for_single_med():
    pid = next(p["pid"] for p in server.list_patients() if p["age_band"].startswith("5"))
    out = build().invoke({"pid": pid, "trace": []})
    assert "interactions" not in [t["node"] for t in out["trace"]]
