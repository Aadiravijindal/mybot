"""Optional LLM research extractor (Claude).

Role: turn unstructured text into structured, *cited* facts. It does NOT predict
prices and it cannot trade. Inputs are only public text (headline/body); no
account data, keys, or positions are ever included in prompts.

Runs off the hot path (thread pool); results arrive later as RESEARCH events. If
the API is unavailable the system continues with the lexicon classifier.
"""
from __future__ import annotations

import json
import logging
import os
from concurrent.futures import Future, ThreadPoolExecutor

log = logging.getLogger(__name__)

SYSTEM = (
    "You are a financial news analyst. Extract facts from the provided article only. "
    "Do not speculate beyond the text. If the text does not support a field, use null. "
    "Return JSON only."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "event_type": {"type": "string"},
        "direction": {"type": "integer", "enum": [-1, 0, 1]},
        "materiality": {"type": "number"},
        "is_new_information": {"type": "boolean"},
        "key_facts": {"type": "array", "items": {"type": "string"}},
        "evidence_quotes": {"type": "array", "items": {"type": "string"}},
        "risks_to_interpretation": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["event_type", "direction", "materiality", "is_new_information", "key_facts",
                 "evidence_quotes", "risks_to_interpretation"],
    "additionalProperties": False,
}


class LLMResearcher:
    def __init__(self, model: str = "claude-opus-5-5", max_workers: int = 2):
        self.enabled = bool(os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model
        self._pool = ThreadPoolExecutor(max_workers=max_workers) if self.enabled else None
        self._client = None
        self.failures = 0
        if self.enabled:
            try:
                import anthropic  # optional dependency
                self._client = anthropic.Anthropic(timeout=30.0, max_retries=1)
            except Exception:  # package missing
                log.warning("anthropic package not installed; LLM research disabled")
                self.enabled = False

    def submit(self, headline: str, body: str, symbols: tuple[str, ...]) -> Future | None:
        if not self.enabled or self._pool is None:
            return None
        return self._pool.submit(self._extract, headline, body, symbols)

    def _extract(self, headline: str, body: str, symbols: tuple[str, ...]) -> dict | None:
        try:
            # Server-side fallback: if a safety classifier declines, the API retries on a
            # suitable model instead of failing the extraction.
            resp = self._client.beta.messages.create(
                model=self.model,
                max_tokens=2000,
                system=SYSTEM,
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
                messages=[{"role": "user", "content": f"Tickers: {', '.join(symbols)}\nHeadline: {headline}\n\n{body[:20000]}"}],
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
            if resp.stop_reason in ("refusal", "max_tokens"):
                return None
            text = "".join(b.text for b in resp.content if b.type == "text")
            return json.loads(text)
        except Exception as exc:  # network, rate limit, parse: never fatal
            self.failures += 1
            log.warning("LLM extraction failed: %s", exc)
            return None
