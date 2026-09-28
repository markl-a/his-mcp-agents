"""De-identification applied at the server boundary: the model never sees names, national IDs or exact birth dates."""
from __future__ import annotations

import hashlib
import hmac
import os
from datetime import date

_SECRET = os.environ.get("HIS_PSEUDONYM_KEY", "dev-only-key").encode()


def pseudonym(patient_id: str) -> str:
    """Stable, non-reversible token for a patient id (HMAC-SHA256, truncated)."""
    digest = hmac.new(_SECRET, patient_id.encode(), hashlib.sha256).hexdigest()
    return f"pt_{digest[:12]}"


def age_band(birth_date: str, today: date | None = None) -> str:
    today = today or date.today()
    y, m, d = (int(x) for x in birth_date.split("-"))
    age = today.year - y - ((today.month, today.day) < (m, d))
    lo = (age // 10) * 10
    return f"{lo}-{lo + 9}"


def deidentify_patient(p: dict) -> dict:
    return {"pid": pseudonym(p["id"]), "age_band": age_band(p["birthDate"]), "gender": p["gender"]}
