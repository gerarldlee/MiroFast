"""Hybrid round policy: Laya decides, the LLM only writes.

This replaces OASIS's per-agent `LLMAction` tool-calling step. Per round:

1. Refresh every active agent's feed (no LLM).
2. ONE batched Laya forward pass for the whole crowd: action choice,
   target post and stance per agent.
3. Non-generative actions (do_nothing / like / dislike / repost) execute
   directly as OASIS ManualActions — zero LLM calls.
4. Generative actions (post / comment / quote) get a short, focused LLM
   prompt (persona + target only), then a Laya moderation gate, then execute
   as ManualActions.
5. Optionally, low-confidence decisions fall back to the original
   `perform_action_by_llm` for that agent only.

Everything runs inside the existing `env.step` execution machinery, so the
platform database, trace log and downstream tooling behave identically.
"""

import asyncio
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from oasis import ActionType, ManualAction

from .laya_engine import AgentDecision, LayaEngine
from .questions import DEFAULT_MAX_FEED_POSTS, GENERATIVE_ACTIONS, action_target_questions
from .state_builder import build_agent_state, extract_persona

_MAX_PERSONA_IN_PROMPT = 800
_MAX_TARGET_IN_PROMPT = 300

_ACTION_TO_ACTION_TYPE = {
    ("twitter", "like_post"): ActionType.LIKE_POST,
    ("twitter", "repost"): ActionType.REPOST,
    ("twitter", "quote_post"): ActionType.QUOTE_POST,
    ("twitter", "create_post"): ActionType.CREATE_POST,
    ("reddit", "like_post"): ActionType.LIKE_POST,
    ("reddit", "dislike_post"): ActionType.DISLIKE_POST,
    ("reddit", "create_comment"): ActionType.CREATE_COMMENT,
    ("reddit", "create_post"): ActionType.CREATE_POST,
}

_PLATFORM_TEXT_LIMIT = {"twitter": 280, "reddit": 600}


@dataclass
class GenerationSpec:
    agent: Any
    agent_id: int
    action: str
    platform: str
    target_post: Optional[Dict[str, Any]]
    decision: AgentDecision


@dataclass
class RoundStats:
    manual_count: int = 0
    generated_count: int = 0
    rejected_count: int = 0
    failed_count: int = 0
    fallback_count: int = 0
    decision_ms: float = 0.0
    generation_ms: float = 0.0
    actions: Dict[str, int] = field(default_factory=dict)
    stance_avg: Optional[float] = None


def _resolve_target_post(
    decision: AgentDecision, post_ids: List[int], posts: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Map a target label like 'post_2' back to its platform post.

    When a target-needing action chose 'none', fall back to the argmax over
    the post_* probabilities (the choice head still carries per-post signal).
    """
    label = decision.target_label
    if label in (None, "", "none"):
        label = _argmax_post_label(decision)
        if label is None:
            return None
    try:
        idx = int(label.split("_")[1]) - 1
    except (ValueError, IndexError):
        return None
    if 0 <= idx < len(post_ids) and idx < len(posts):
        return posts[idx]
    return None


def _argmax_post_label(decision: AgentDecision) -> Optional[str]:
    """Highest-probability post_k label from the raw target answer."""
    target_answer = decision.raw.get("target_post", {})
    probs = target_answer.get("probabilities", {})
    best_label, best_p = None, 0.0
    for label, p in probs.items():
        if isinstance(p, (int, float)) and label.startswith("post_") and p > best_p:
            best_label, best_p = label, float(p)
    return best_label


def _manual_for(
    agent: Any, action: str, platform: str, post: Optional[Dict[str, Any]]
) -> Optional[ManualAction]:
    key = (platform, action)
    action_type = _ACTION_TO_ACTION_TYPE.get(key)
    if action_type is None:
        return None
    args: Dict[str, Any] = {}
    if action in ("like_post", "dislike_post", "repost"):
        if post is None:
            return None
        args["post_id"] = post["post_id"]
    return ManualAction(action_type=action_type, action_args=args)


async def _refresh_feed(agent: Any) -> List[Dict[str, Any]]:
    try:
        response = await agent.env.action.refresh()
        if isinstance(response, dict) and response.get("success"):
            posts = response.get("posts", [])
            return posts if isinstance(posts, list) else []
    except Exception:
        pass
    return []


def _generation_prompt(spec: GenerationSpec, name: str, persona: str) -> Tuple[str, str]:
    platform = spec.platform
    limit = _PLATFORM_TEXT_LIMIT.get(platform, 280)
    system = (
        f"You are {name}, a simulated user on {platform}. Stay strictly in "
        f"character.\nYour profile: {persona[:_MAX_PERSONA_IN_PROMPT]}"
    )
    if spec.action == "create_post":
        user = (
            f"Write your next original {platform} post. Express your current "
            f"opinion or reaction, in character. Output only the post text — "
            f"no quotes, no explanation, under {limit} characters."
        )
    elif spec.action == "quote_post":
        target = (spec.target_post or {}).get("content", "")
        user = (
            f"Quote this post, adding your own commentary:\n"
            f"\"{target[:_MAX_TARGET_IN_PROMPT]}\"\n"
            f"Output only your commentary text — no quotes, under {limit} "
            f"characters."
        )
    else:  # create_comment
        target = (spec.target_post or {}).get("content", "")
        user = (
            f"Write a short reply comment to this post:\n"
            f"\"{target[:_MAX_TARGET_IN_PROMPT]}\"\n"
            f"Output only the comment text — no quotes, under {limit} "
            f"characters."
        )
    return system, user


async def _generate_one(
    spec: GenerationSpec,
    gateway: Any,
    engine: LayaEngine,
    moderate: bool,
    stats: RoundStats,
    log: Callable[[str], None],
) -> Optional[ManualAction]:
    name, persona, _ = extract_persona(spec.agent)
    system, user = _generation_prompt(spec, name, persona)
    try:
        content = await gateway.agenerate(system=system, user=user, max_tokens=200)
    except Exception as exc:
        stats.failed_count += 1
        log(f"Agent {spec.agent_id} generation failed: {exc}")
        return None
    content = (content or "").strip().strip('"')
    if not content:
        stats.failed_count += 1
        return None

    if moderate:
        try:
            verdict = await asyncio.to_thread(engine.moderate, content, persona)
            if verdict.get("toxic", 0.0) >= engine.moderation_threshold:
                stats.rejected_count += 1
                log(f"Agent {spec.agent_id} content blocked by moderation gate")
                return None
        except Exception:
            pass  # moderation is best-effort; never block the simulation

    key = (spec.platform, spec.action)
    action_type = _ACTION_TO_ACTION_TYPE[key]
    args: Dict[str, Any] = {}
    if spec.action == "quote_post":
        args = {
            "post_id": spec.target_post["post_id"],
            "quote_content": content,
        }
    elif spec.action == "create_comment":
        args = {"post_id": spec.target_post["post_id"], "content": content}
    else:  # create_post
        args = {"content": content}
    return ManualAction(action_type=action_type, action_args=args)


async def run_hybrid_round(
    env: Any,
    active_agents: List[Tuple[int, Any]],
    engine: LayaEngine,
    gateway: Any,
    platform: str,
    round_num: int,
    simulated_hour: int,
    decision_log_path: Optional[str] = None,
    moderate: bool = True,
    llm_fallback: bool = False,
    log: Callable[[str], None] = lambda msg: None,
) -> RoundStats:
    """One hybrid simulation round for a platform environment."""
    stats = RoundStats()

    # 1. Refresh feeds and build compact decision states (no LLM).
    feeds = await asyncio.gather(
        *[_refresh_feed(agent) for _, agent in active_agents],
        return_exceptions=True,
    )
    states: List[str] = []
    post_ids_per_agent: List[List[int]] = []
    meta: List[Tuple[int, Any, List[Dict[str, Any]]]] = []
    for (agent_id, agent), feed in zip(active_agents, feeds):
        posts = feed if isinstance(feed, list) else []
        posts = posts[:DEFAULT_MAX_FEED_POSTS]
        name, persona, bio = extract_persona(agent)
        state, post_ids = build_agent_state(name, persona, bio, posts, simulated_hour)
        states.append(state)
        post_ids_per_agent.append(post_ids)
        meta.append((agent_id, agent, posts))

    # 2. One batched Laya pass decides the whole crowd.
    import time as _time
    decision_started = _time.perf_counter()
    max_feed = max((len(posts) for _, _, posts in meta), default=0)
    questions = action_target_questions(platform, max_feed_posts=max_feed)
    agent_ids = [agent_id for agent_id, _, _ in meta]
    decisions = engine.decide_batch(agent_ids, states, questions)
    stats.decision_ms = (_time.perf_counter() - decision_started) * 1000.0

    stance_values = [d.stance for d in decisions if d.stance is not None]
    if stance_values:
        stats.stance_avg = sum(stance_values) / len(stance_values)

    # 3. Split into direct manual actions / generation / LLM fallback.
    manual_actions: Dict[Any, ManualAction] = {}
    fallback_agents: List[Any] = []
    gen_specs: List[GenerationSpec] = []

    for (agent_id, agent, posts), decision, post_ids in zip(
        meta, decisions, post_ids_per_agent
    ):
        action = decision.action
        target = _resolve_target_post(decision, post_ids, posts)
        stats.actions[action] = stats.actions.get(action, 0) + 1

        needs_llm_fallback = (
            llm_fallback and decision.confidence < engine.confidence_threshold
        )
        if needs_llm_fallback:
            fallback_agents.append(agent)
            stats.fallback_count += 1
            _log_decision(decision_log_path, round_num, agent_id, decision,
                          action="llm_fallback", executed=True)
            continue

        if action == "do_nothing":
            manual_actions[agent] = ManualAction(
                action_type=ActionType.DO_NOTHING, action_args={}
            )
            _log_decision(decision_log_path, round_num, agent_id, decision,
                          action="do_nothing", executed=True)
            continue

        if action in GENERATIVE_ACTIONS:
            if action in ("create_comment", "quote_post") and target is None:
                # Generative action lost its target: fall back to a plain post.
                action = "create_post"
                decision.action = action
            spec = GenerationSpec(
                agent=agent,
                agent_id=agent_id,
                action=action,
                platform=platform,
                target_post=target,
                decision=decision,
            )
            gen_specs.append(spec)
            _log_decision(decision_log_path, round_num, agent_id, decision,
                          action=action,
                          target_post_id=(target or {}).get("post_id"))
            continue

        manual = _manual_for(agent, action, platform, target)
        if manual is None:
            manual_actions[agent] = ManualAction(
                action_type=ActionType.DO_NOTHING, action_args={}
            )
            _log_decision(decision_log_path, round_num, agent_id, decision,
                          action="do_nothing", executed=True)
        else:
            manual_actions[agent] = manual
            _log_decision(decision_log_path, round_num, agent_id, decision,
                          action=action,
                          target_post_id=(target or {}).get("post_id"))

    # 4. Execute direct actions while content generation runs concurrently.
    generation_started = _time.perf_counter()
    generation_task = asyncio.gather(
        *[
            _generate_one(spec, gateway, engine, moderate, stats, log)
            for spec in gen_specs
        ]
    ) if gen_specs else None

    if manual_actions:
        await env.step(manual_actions)
        stats.manual_count = len(manual_actions)
    if fallback_agents:
        from oasis import LLMAction
        await env.step({agent: LLMAction() for agent in fallback_agents})

    generated: List[Optional[ManualAction]] = (
        await generation_task if generation_task else []
    )
    stats.generation_ms = (_time.perf_counter() - generation_started) * 1000.0

    # 5. Execute generated content.
    generated_actions: Dict[Any, ManualAction] = {}
    for spec, manual in zip(gen_specs, generated):
        if manual is not None:
            generated_actions[spec.agent] = manual
    if generated_actions:
        await env.step(generated_actions)
        stats.generated_count = len(generated_actions)

    return stats


def _log_decision(
    path: Optional[str],
    round_num: int,
    agent_id: int,
    decision: AgentDecision,
    action: Optional[str] = None,
    target_post_id: Optional[int] = None,
    executed: bool = True,
) -> None:
    if not path:
        return
    record = {
        "timestamp": datetime.now().isoformat(),
        "round": round_num,
        "agent_id": agent_id,
        "action": action or decision.action,
        "target_post_id": target_post_id,
        "confidence": round(decision.confidence, 4),
        "stance": decision.stance,
        "model": decision.model,
        "decision_ms": round(decision.latency_ms, 2),
        "executed": executed,
        "engine": "laya",
    }
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass
