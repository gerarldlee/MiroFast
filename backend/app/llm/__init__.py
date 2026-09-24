"""Generation gateway: per-purpose LLM clients (the only generative calls)."""

from .gateway import GenerationGateway, get_client

__all__ = ["GenerationGateway", "get_client"]
