"""Provider-agnostic, disk-cached chat client.

Backends
--------
* ``openai_compat``  any OpenAI-compatible endpoint: Groq, OpenRouter, Ollama,
  vLLM, LM Studio, Gemini's OpenAI-compatible endpoint. Configure with
  ``base_url`` and the *name* of the env var holding the key (never the key).
* ``hf``             local transformers model (Colab GPU), optional 4-bit.

Every (model, messages, params) request is hashed and cached as one JSONL
line in ``llm_experiments/cache/<model>.jsonl`` so reruns and analysis cost no
quota. Free-tier rate limits are handled with exponential backoff.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent / "cache"

PRESETS = {
    # name: (backend, base_url, api_key_env)
    "groq": ("openai_compat", "https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "openrouter": ("openai_compat", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "gemini": ("openai_compat", "https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    "ollama": ("openai_compat", "http://localhost:11434/v1", None),
}


def reasoning_params(provider: str, model: str) -> dict:
    """Keep chain-of-thought out of the answer text so the JSON can be parsed."""
    if provider != "groq":
        return {}
    if model.startswith("openai/gpt-oss"):
        return {"reasoning_effort": "low", "include_reasoning": False}
    if "qwen3" in model:
        return {"reasoning_format": "hidden"}
    return {}


@dataclass
class LLM:
    model: str
    provider: str = "groq"           # key of PRESETS or "hf"
    temperature: float = 0.0
    max_tokens: int = 1024           # reasoning models spend tokens before the JSON answer
    min_interval_s: float = 2.0      # free-tier pacing
    hf_4bit: bool = True
    _last: float = field(default=0.0, repr=False)
    _hf: object = field(default=None, repr=False)
    _cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        CACHE_DIR.mkdir(exist_ok=True)
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in self.model)[:80]
        digest = hashlib.sha256(self.model.encode()).hexdigest()[:8]
        self._cache_path = CACHE_DIR / f"{safe}-{digest}.jsonl"   # never a raw user-supplied path
        if self._cache_path.exists():
            for line in self._cache_path.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                self._cache[rec["key"]] = rec["response"]

    def _key(self, messages, sample_idx: int) -> str:
        blob = json.dumps([self.model, messages, self.temperature, self.max_tokens, sample_idx],
                          sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()

    def chat(self, messages: list[dict], sample_idx: int = 0) -> str:
        key = self._key(messages, sample_idx)
        if key in self._cache:
            return self._cache[key]
        text = self._hf_chat(messages) if self.provider == "hf" else self._api_chat(messages)
        self._cache[key] = text
        with self._cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "messages": messages, "response": text},
                               ensure_ascii=False) + "\n")
        return text

    # ------------------------------------------------------------------ API
    def _api_chat(self, messages) -> str:
        _, base, key_env = PRESETS[self.provider]
        # Explicit User-Agent: Cloudflare-fronted providers (e.g. Groq) reject
        # Python-urllib's default agent with 403 "error code 1010".
        headers = {"Content-Type": "application/json", "User-Agent": "fairdp-research/1.0"}
        if key_env:
            key = os.environ.get(key_env)
            if not key:
                raise RuntimeError(f"set {key_env} in your environment (.env is gitignored)")
            headers["Authorization"] = f"Bearer {key}"
        payload = {"model": self.model, "messages": messages, "temperature": self.temperature, "max_tokens": self.max_tokens}
        payload.update(reasoning_params(self.provider, self.model))
        body = json.dumps(payload).encode()
        for attempt in range(8):
            wait = self.min_interval_s - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            req = urllib.request.Request(f"{base}/chat/completions", data=body, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = json.loads(r.read())
                return data["choices"][0]["message"]["content"] or ""
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504):
                    time.sleep(min(60, 2 ** attempt * 2))
                    continue
                raise
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                # transient network failure (DNS, TLS handshake timeout, reset): back off and retry
                time.sleep(min(60, 2 ** attempt * 2))
                last_err = e
                continue
        raise RuntimeError(f"request failed after retries ({locals().get('last_err', 'rate limited')})")

    # ------------------------------------------------------------- local HF
    def _hf_chat(self, messages) -> str:
        if self._hf is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            kw = {"device_map": "auto", "torch_dtype": torch.bfloat16}
            if self.hf_4bit:
                from transformers import BitsAndBytesConfig
                kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True,
                                                               bnb_4bit_compute_dtype=torch.bfloat16)
            tok = AutoTokenizer.from_pretrained(self.model)
            self._hf = (tok, AutoModelForCausalLM.from_pretrained(self.model, **kw))
        tok, mdl = self._hf
        ids = tok.apply_chat_template(messages, add_generation_prompt=True,
                                      return_tensors="pt").to(mdl.device)
        gen = mdl.generate(ids, max_new_tokens=self.max_tokens,
                           do_sample=self.temperature > 0,
                           temperature=self.temperature or None)
        return tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
