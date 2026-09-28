"""Clinical pre-visit brief as a supervisor + workers graph (LangGraph).

supervisor -> gather (MCP tools) -> [guidelines?] -> [interactions?] -> writer -> END
Every node appends to `trace` (latency, token estimate) so cost and TTFT can be measured per step.
The output is a *draft for a clinician*, with every claim tied to a tool result or guideline source.
"""
from __future__ import annotations

from pathlib import Path
from typing import Annotated, TypedDict
import operator

from langgraph.graph import END, START, StateGraph

from his_mcp import server as his
from .llm import get_llm, timed
from .retriever import BM25

_GUIDES = BM25(Path(__file__).parent / "guidelines")
SYSTEM = ("You write a pre-visit brief for a clinician. Use ONLY the facts given. "
          "Cite every statement with its [source]. If something is unknown, say so. Do not recommend a dose.")
PROMPT_VERSION = "brief-v3"


class State(TypedDict, total=False):
    pid: str
    facts: dict
    plan: list[str]
    guidelines: list[dict]
    interactions: list[dict]
    brief: str
    trace: Annotated[list[dict], operator.add]


def gather(state: State) -> dict:
    pid = state["pid"]
    facts, ms = timed(lambda: {
        "conditions": his.get_conditions(pid),
        "abnormal_labs": his.get_observations(pid, abnormal_only=True),
        "medications": his.get_medications(pid),
        "encounters": his.get_encounters(pid),
    })
    return {"facts": facts, "trace": [{"node": "gather", "ms": ms, "tools": 4}]}


def supervisor(state: State) -> dict:
    f = state["facts"]
    plan = []
    if f["conditions"] or f["abnormal_labs"]:
        plan.append("guidelines")
    if len(f["medications"]) >= 2:
        plan.append("interactions")
    return {"plan": plan, "trace": [{"node": "supervisor", "plan": plan}]}


def guidelines(state: State) -> dict:
    f = state["facts"]
    query = " ".join([c["text"] for c in f["conditions"]] + [o["code"] for o in f["abnormal_labs"]] + [m["drug"] for m in f["medications"]])
    hits, ms = timed(_GUIDES.search, query, 4)
    return {"guidelines": hits, "trace": [{"node": "guidelines", "ms": ms, "hits": len(hits)}]}


def interactions(state: State) -> dict:
    hits, ms = timed(his.check_interactions, state["pid"])
    return {"interactions": hits, "trace": [{"node": "interactions", "ms": ms, "hits": len(hits)}]}


def writer(state: State) -> dict:
    f = state["facts"]
    lines = ["# facts"]
    lines += [f"Diagnosis {c['code']} {c['text']} [tool:get_conditions]" for c in f["conditions"]]
    lines += [f"{o['code']} {o['value']} {o['unit']} flag {o['flag']} on {o['date']} [tool:get_observations]" for o in f["abnormal_labs"]]
    lines += [f"Active med {m['drug']} {m['dose']} [tool:get_medications]" for m in f["medications"]]
    lines += [f"Interaction {' + '.join(i['pair'])} ({i['severity']}): {i['note']} [tool:check_interactions]" for i in state.get("interactions", [])]
    lines += [f"{g['text']} [{g['source']}]" for g in state.get("guidelines", [])]
    prompt = "\n".join(lines)
    llm = get_llm()
    brief, ms = timed(llm.complete, SYSTEM, prompt)
    est_tokens = (len(SYSTEM) + len(prompt) + len(brief)) // 4
    return {"brief": brief, "trace": [{"node": "writer", "ms": ms, "model": llm.name, "prompt_version": PROMPT_VERSION, "est_tokens": est_tokens}]}


def route(state: State) -> str:
    plan = state.get("plan", [])
    done = {t["node"] for t in state.get("trace", [])}
    for step in plan:
        if step not in done:
            return step
    return "writer"


def build():
    g = StateGraph(State)
    for name, fn in [("gather", gather), ("supervisor", supervisor), ("guidelines", guidelines), ("interactions", interactions), ("writer", writer)]:
        g.add_node(name, fn)
    g.add_edge(START, "gather")
    g.add_edge("gather", "supervisor")
    for n in ("supervisor", "guidelines", "interactions"):
        g.add_conditional_edges(n, route, {"guidelines": "guidelines", "interactions": "interactions", "writer": "writer"})
    g.add_edge("writer", END)
    return g.compile()


if __name__ == "__main__":
    import json, sys
    app = build()
    pid = sys.argv[1] if len(sys.argv) > 1 else his.list_patients()[0]["pid"]
    out = app.invoke({"pid": pid, "trace": []})
    print(out["brief"]); print(json.dumps(out["trace"], ensure_ascii=False, indent=1))
