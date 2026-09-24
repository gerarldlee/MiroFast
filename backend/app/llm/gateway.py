"""Generation gateway.

MiroFast's LLM spend is generation-only: persona bios, simulated post and
comment text, report prose, interviews and graph NER. Each purpose can point
at a different OpenAI-compatible backend via environment variables:

    {PREFIX}_API_KEY / {PREFIX}_BASE_URL / {PREFIX}_MODEL

with PREFIX in (POSTS, PROFILES, ONTOLOGY, NER, REPORT, INTERVIEW), falling
back to the global LLM_API_KEY / LLM_BASE_URL / LLM_MODEL_NAME. This keeps a
single cloud model for prose while simulation content can run on a cheaper
endpoint (or Ollama) without code changes.
"""

import asyncio
import os
from typing import Dict, List, Optional

from ..utils.llm_client import LLMClient

_PURPOSE_PREFIXES = {
    "posts": "POSTS",
    "profiles": "PROFILES",
    "ontology": "ONTOLOGY",
    "ner": "NER",
    "report": "REPORT",
    "interview": "INTERVIEW",
}


def get_client(purpose: str = "default", timeout: float = 300.0) -> LLMClient:
    """LLMClient configured for a purpose, with LLM_* fallback."""
    prefix = _PURPOSE_PREFIXES.get(purpose.lower())
    if prefix is None:
        return LLMClient(timeout=timeout)
    return LLMClient(
        api_key=os.environ.get(f"{prefix}_API_KEY") or None,
        base_url=os.environ.get(f"{prefix}_BASE_URL") or None,
        model=os.environ.get(f"{prefix}_MODEL") or None,
        timeout=timeout,
    )


class GenerationGateway:
    """Async facade for generating simulated content in the hot loop."""

    def __init__(self, purpose: str = "posts", client: Optional[LLMClient] = None):
        self.client = client or get_client(purpose, timeout=120.0)

    def generate(
        self,
        system: str,
        user: str,
        temperature: float = 0.9,
        max_tokens: int = 200,
    ) -> str:
        return self.client.chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def agenerate(
        self,
        system: str,
        user: str,
        temperature: float = 0.9,
        max_tokens: int = 200,
    ) -> str:
        return await asyncio.to_thread(
            self.generate, system, user, temperature, max_tokens
        )
