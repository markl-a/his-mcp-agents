"""LLM backends. Default is a deterministic template writer so the pipeline runs and tests offline.
Set LLM_BACKEND=anthropic (ANTHROPIC_API_KEY) or LLM_BACKEND=ollama (OLLAMA_MODEL) to use a real model."""
from __future__ import annotations

import json
import os
import time
import urllib.request


class TemplateLLM:
    name = "template-v1"

    def complete(self, system: str, prompt: str) -> str:
        # Echo the structured facts as bullet points: grounded by construction, used for tests and demos.
        return "\n".join(f"- {line}" for line in prompt.splitlines() if line.strip() and not line.startswith("#"))


class OllamaLLM:
    def __init__(self):
        self.model = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
        self.name = f"ollama/{self.model}"

    def complete(self, system: str, prompt: str) -> str:
        body = json.dumps({"model": self.model, "system": system, "prompt": prompt, "stream": False, "options": {"temperature": 0}}).encode()
        req = urllib.request.Request("http://localhost:11434/api/generate", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())["response"]


class AnthropicLLM:
    def __init__(self):
        import anthropic  # optional dependency

        self.client = anthropic.Anthropic()
        self.model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
        self.name = f"anthropic/{self.model}"

    def complete(self, system: str, prompt: str) -> str:
        msg = self.client.messages.create(model=self.model, max_tokens=800, temperature=0, system=system, messages=[{"role": "user", "content": prompt}])
        return msg.content[0].text


def get_llm():
    backend = os.environ.get("LLM_BACKEND", "template")
    return {"template": TemplateLLM, "ollama": OllamaLLM, "anthropic": AnthropicLLM}[backend]()


def timed(fn, *args):
    t0 = time.perf_counter()
    out = fn(*args)
    return out, round((time.perf_counter() - t0) * 1000, 1)
