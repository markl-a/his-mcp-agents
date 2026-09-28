"""Tiny illustrative interaction table. Real systems use a licensed knowledge base; this only demonstrates the tool contract."""
INTERACTIONS = {
    frozenset({"amlodipine", "clarithromycin"}): ("moderate", "CYP3A4 inhibition may raise amlodipine levels; monitor BP and edema."),
    frozenset({"spironolactone", "lisinopril"}): ("major", "Additive hyperkalemia risk, especially with reduced renal function."),
    frozenset({"simvastatin", "clarithromycin"}): ("major", "Strong CYP3A4 inhibition; risk of myopathy/rhabdomyolysis."),
}


def check(drugs: list[str]) -> list[dict]:
    drugs = sorted({d.lower() for d in drugs})
    out = []
    for i, a in enumerate(drugs):
        for b in drugs[i + 1:]:
            hit = INTERACTIONS.get(frozenset({a, b}))
            if hit:
                out.append({"pair": [a, b], "severity": hit[0], "note": hit[1]})
    return out
