# his-mcp-agents

**An MCP server for hospital information system (HIS) data, plus a supervisor/worker agent graph that writes a grounded pre-visit brief for a clinician.**

Portfolio demo (Sept 2026). Mock FHIR-shaped data, runs fully offline, 8 tests. It shows the design decisions I would make when putting LLM agents next to an HIS — not a production system.

```
               MCP (stdio)                         read-only, de-identified
 LLM / agent  ───────────────►  his-tools server  ───────────────────────►  FHIR store
                                 │  every call → audit.jsonl
                                 │  writes = draft only; clinician confirms out-of-band
```

## What it demonstrates

| Concern | How it is handled |
|---|---|
| **PHI never reaches the model** | Tools return a pseudonymous `pid` (HMAC-SHA256), an age band and gender — no names, national IDs or birth dates (`his_mcp/deid.py`) |
| **Auditability** | Every tool call is appended to `audit.jsonl` with actor, tool, hashed args, model and prompt version (`his_mcp/audit.py`) |
| **Human-in-the-loop writes** | The agent can only call `draft_medication_request`; it returns a single-use token. `confirm_medication_request` is **not** exposed over MCP — only the clinician UI can call it (`test_confirm_is_not_an_mcp_tool`) |
| **Grounding** | The brief is built only from tool results and guideline chunks, each line carries `[tool:…]` or `[file.md:Lx]` |
| **Multi-agent routing** | LangGraph: `gather → supervisor → guidelines? → interactions? → writer`. The supervisor skips workers that have nothing to do (single-med patient ⇒ no interaction check) |
| **Cost / latency visibility** | Every node appends `ms`, token estimate, model and `prompt_version` to `trace` — the basis for TTFT and per-call cost tracking |
| **Model-agnostic** | `LLM_BACKEND=template` (default, deterministic), `ollama` (local — keeps data on-prem), or `anthropic` |

## Run

```bash
pip install -r requirements.txt
pytest -q                          # 8 tests
python -m agents.graph             # prints the brief + per-node trace
python -m his_mcp.server           # MCP server over stdio (e.g. register in Claude Desktop)
LLM_BACKEND=ollama OLLAMA_MODEL=qwen2.5:7b python -m agents.graph
```

## Tools exposed

`list_patients` · `get_encounters` · `get_conditions` · `get_observations(abnormal_only)` · `get_medications` · `check_interactions(extra_drugs)` · `draft_medication_request`

## Limits (on purpose)

- Mock data; the interaction table has three illustrative pairs and is not clinical knowledge.
- BM25 over markdown stands in for a vector store; the retrieval interface is the same.
- Guideline texts are short illustrative summaries I wrote, not clinical guidance.
