"""Laya engine wrapper: one Router instance, batched round decisions.

Deliberately framework-free (no Flask dependency) so both the Flask app and
the simulation subprocess can use it. All settings come from environment
variables with sane defaults:

    LAYA_MODEL                  checkpoint: multilingual (default) | english | typed-decisions
    LAYA_DEVICE                 auto | cuda | cpu | mps
    LAYA_BATCH_SIZE             states per forward pass (default 16)
    LAYA_CONFIDENCE_THRESHOLD   decision confidence floor (default 0.55)
    LAYA_MODERATION_THRESHOLD   toxicity probability that blocks content (default 0.8)
"""

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class AgentDecision:
    """One agent's System-1 decision for a round."""

    agent_id: int
    action: str
    target_label: str
    confidence: float
    stance: Optional[float] = None
    model: str = ""
    latency_ms: float = 0.0
    raw: Dict[str, Any] = field(default_factory=dict)


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


class LayaEngine:
    """Thin, lazily-initialized wrapper around laya.Router."""

    def __init__(
        self,
        model: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        preload: Optional[bool] = None,
    ):
        self.model = model or _env("LAYA_MODEL", "multilingual")
        device = device or _env("LAYA_DEVICE", "auto")
        self.device = None if device == "auto" else device
        self.batch_size = int(
            batch_size if batch_size is not None
            else _env("LAYA_BATCH_SIZE", "16")
        )
        self.confidence_threshold = float(
            _env("LAYA_CONFIDENCE_THRESHOLD", "0.55")
        )
        self.moderation_threshold = float(
            _env("LAYA_MODERATION_THRESHOLD", "0.8")
        )
        if preload is None:
            preload = _env("LAYA_PRELOAD", "1").lower() in ("1", "true", "yes")
        self.preload = preload
        self._router: Any = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ router

    def _get_router(self):
        if self._router is None:
            with self._lock:
                if self._router is None:
                    self._tune_torch_threads()
                    from laya import Router
                    self._router = Router(
                        device=self.device,
                        preload=self.preload,
                        default=self.model,
                    )
        return self._router

    @staticmethod
    def _tune_torch_threads():
        """Pin torch CPU threads to avoid oversubscription.

        torch defaults to every logical core, which makes CPU inference
        dramatically slower (measured 24s vs 1.2s per decision on a
        20-core machine). LAYA_THREADS overrides the pinned count.
        """
        try:
            import torch
            default = min(8, os.cpu_count() or 8)
            n = int(_env("LAYA_THREADS", str(default)))
            torch.set_num_threads(max(1, n))
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass  # already started; fine
        except Exception:
            pass

    def warmup(self) -> None:
        """Force checkpoint load (called at simulation start)."""
        self._get_router()

    # ------------------------------------------------------------- primitives

    def decide_batch(
        self,
        agent_ids: Sequence[int],
        states: Sequence[Any],
        questions: Dict[str, Any],
    ) -> List[AgentDecision]:
        """Score every agent's state against the same questions.

        All states share one schema so the whole round is answered in shared
        forward passes (one batched pass per round on a single checkpoint).
        """
        if not states:
            return []
        router = self._get_router()
        requests = [
            {"state": state, "questions": questions, "model": self.model}
            for state in states
        ]
        started = time.perf_counter()
        results = router.predict_batch(requests, batch_size=self.batch_size)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        decisions: List[AgentDecision] = []
        for agent_id, result in zip(agent_ids, results):
            answers = result.get("answers", {})
            action_ans = answers.get("action", {})
            target_ans = answers.get("target_post", {})
            stance_ans = answers.get("stance", {})
            decisions.append(
                AgentDecision(
                    agent_id=agent_id,
                    action=action_ans.get("choice", "do_nothing"),
                    target_label=target_ans.get("choice", "none"),
                    confidence=float(action_ans.get("confidence", 0.0)),
                    stance=(
                        float(stance_ans["score"])
                        if "score" in stance_ans else None
                    ),
                    model=result.get("routing", {}).get("model", self.model),
                    latency_ms=elapsed_ms / max(len(results), 1),
                    raw=answers,
                )
            )
        return decisions

    def decide_one(self, state: Any, questions: Dict[str, Any]) -> Dict[str, Any]:
        """Single-state typed decision (dynamic schemas go through here)."""
        router = self._get_router()
        return router.predict(state, questions, model=self.model)

    def choice(self, state: Any, instructions: str, criteria: Dict[str, str]) -> Tuple[str, float]:
        """One dynamic choice: returns (label, confidence)."""
        result = self.decide_one(state, {
            "pick": {"type": "choice", "instructions": instructions, "criteria": dict(criteria)},
        })
        ans = result.get("answers", {}).get("pick", {})
        return ans.get("choice", ""), float(ans.get("confidence", 0.0))

    def score(self, state: Any, instructions: str, criteria: List[str]) -> Tuple[float, float]:
        """One ordinal score: returns (score, confidence)."""
        result = self.decide_one(state, {
            "level": {"type": "score", "instructions": instructions, "criteria": list(criteria)},
        })
        ans = result.get("answers", {}).get("level", {})
        return float(ans.get("score", 0.0)), float(ans.get("confidence", 0.0))

    def noul(self, state: Any, instructions: str) -> float:
        """One calibrated boolean probability."""
        result = self.decide_one(state, {
            "prob": {"type": "noul", "instructions": instructions},
        })
        ans = result.get("answers", {}).get("prob", {})
        return float(ans.get("noul", 0.0))

    def noul_batch(self, states: Sequence[Any], instructions: str) -> List[float]:
        """Calibrated boolean probabilities for many states, one pass."""
        if not states:
            return []
        router = self._get_router()
        questions = {"prob": {"type": "noul", "instructions": instructions}}
        requests = [
            {"state": s, "questions": questions, "model": self.model}
            for s in states
        ]
        results = router.predict_batch(requests, batch_size=self.batch_size)
        return [
            float(r.get("answers", {}).get("prob", {}).get("noul", 0.0))
            for r in results
        ]

    # ----------------------------------------------------------------- uses

    def moderate(self, content: str, persona: str = "") -> Dict[str, float]:
        """Toxicity / persona-fit gate for LLM-generated content."""
        from .questions import moderation_questions
        state = {"post": content, "author_profile": persona}
        result = self.decide_one(state, moderation_questions())
        answers = result.get("answers", {})
        return {
            "toxic": float(answers.get("toxic", {}).get("noul", 0.0)),
            "on_persona": float(answers.get("on_persona", {}).get("noul", 0.0)),
        }

    def stance_batch(self, texts: Sequence[str], topic: str) -> List[float]:
        """Stance scores for many texts (report analytics, setup)."""
        from .questions import stance_question
        if not texts:
            return []
        router = self._get_router()
        questions = stance_question(topic)
        requests = [
            {"state": {"text": t}, "questions": questions, "model": self.model}
            for t in texts
        ]
        results = router.predict_batch(requests, batch_size=self.batch_size)
        return [
            float(r.get("answers", {}).get("stance", {}).get("score", 0.0))
            for r in results
        ]

    def stance_sentiment_batch(
        self, states: Sequence[Any], topic: str
    ) -> List[Dict[str, float]]:
        """Combined stance + sentiment scores (setup pipeline, analytics)."""
        from .questions import sentiment_question, stance_question
        if not states:
            return []
        router = self._get_router()
        questions = dict(stance_question(topic))
        questions.update(sentiment_question())
        requests = [
            {"state": s, "questions": questions, "model": self.model}
            for s in states
        ]
        results = router.predict_batch(requests, batch_size=self.batch_size)
        out: List[Dict[str, float]] = []
        for r in results:
            answers = r.get("answers", {})
            out.append({
                "stance": float(answers.get("stance", {}).get("score", 2.0)),
                "sentiment": float(answers.get("sentiment", {}).get("score", 2.0)),
            })
        return out

    def sentiment_batch(self, texts: Sequence[str]) -> List[float]:
        """Sentiment scores for many texts (report analytics)."""
        from .questions import sentiment_question
        if not texts:
            return []
        router = self._get_router()
        questions = sentiment_question()
        requests = [
            {"state": {"text": t}, "questions": questions, "model": self.model}
            for t in texts
        ]
        results = router.predict_batch(requests, batch_size=self.batch_size)
        return [
            float(r.get("answers", {}).get("sentiment", {}).get("score", 0.0))
            for r in results
        ]
