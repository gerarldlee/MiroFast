"""MiroFast hybrid round benchmark.

Measures the Laya decision path at scale with a mock platform + mock or
real generation, and compares against the MiroFish baseline cost model
(one LLM tool-calling step per active agent per round).

Usage:
    python bench_round.py --agents 50 --rounds 10
    python bench_round.py --agents 50 --rounds 10 --real-generation   # needs LLM_*
    python bench_round.py --agents 200 --rounds 5 --no-moderate
"""

import argparse
import asyncio
import os
import random
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))

from app.decisions import LayaEngine, run_hybrid_round
from app.llm.gateway import GenerationGateway


class MockChannelAction:
    def __init__(self, posts, agent_id):
        self.posts = posts
        self.agent_id = agent_id
        self.calls = 0

    async def refresh(self):
        self.calls += 1
        return {"success": True, "posts": self.posts}


class MockAgent:
    def __init__(self, agent_id, persona, posts):
        self.social_agent_id = agent_id
        self.agent_id = agent_id
        self.executed = []
        self.user_info = type("U", (), {
            "name": f"agent_{agent_id}",
            "description": "bench agent",
            "profile": {"other_info": {"user_profile": persona}},
        })()
        self.env = type("E", (), {
            "action": MockChannelAction(posts, agent_id),
        })()


class MockEnv:
    def __init__(self):
        self.steps = 0

    async def step(self, actions):
        self.steps += 1
        for agent, action in actions.items():
            agent.executed.append(action)


PERSONAS = [
    "A university student, outspoken about campus conditions.",
    "An official institutional account, cautious and formal.",
    "A local journalist covering the story.",
    "A worried parent who rarely posts.",
    "An alumnus following campus news.",
    "A professor concerned with safety standards.",
    "A dormitory administrator managing renovation work.",
    "A contractor employee working on the renovation.",
]

POST_TEXTS = [
    "Breaking: dorm renovation reports published, concerns raised!",
    "The university says all dorms passed safety checks.",
    "The smell is real. We need independent testing.",
    "Official statement: third-party inspection scheduled next week.",
    "Students deserve transparency about air quality data.",
    "Renovation was completed months ago per the contractor.",
    "My roommate has been sick since moving in. Coincidence?",
    "As a chemistry professor, the reported levels are concerning.",
]


def make_posts(n_posts):
    return [
        {
            "post_id": 1000 + i,
            "user_id": random.randint(1, 99),
            "content": POST_TEXTS[i % len(POST_TEXTS)],
            "num_likes": random.randint(0, 200),
        }
        for i in range(n_posts)
    ]


class MockGateway:
    def __init__(self, delay=1.5):
        self.delay = delay
        self.calls = 0

    async def agenerate(self, system, user, temperature=0.9, max_tokens=200):
        self.calls += 1
        await asyncio.sleep(self.delay)  # simulate one LLM round-trip
        return "bench generated content"


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", type=int, default=50, help="agents in the pool")
    parser.add_argument("--active", type=int, default=0, help="active per round (0 = all)")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--feed", type=int, default=5, help="posts per feed")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--real-generation", action="store_true",
                        help="use the configured LLM instead of a mock")
    parser.add_argument("--no-moderate", action="store_true")
    args = parser.parse_args()

    active_per_round = args.active or args.agents
    posts = make_posts(args.feed)
    agents = [
        MockAgent(i, random.choice(PERSONAS), posts)
        for i in range(args.agents)
    ]
    env = MockEnv()

    engine = LayaEngine(device=os.environ.get("LAYA_DEVICE", "cpu"),
                        preload=False, batch_size=args.batch_size)
    gateway = MockGateway() if not args.real_generation else GenerationGateway("posts")

    print(f"warming Laya (model={engine.model})...")
    t0 = time.perf_counter()
    engine.warmup()
    print(f"warmup: {time.perf_counter() - t0:.1f}s")

    total_decisions_ms = 0.0
    total_generation_ms = 0.0
    total_generated = 0
    total_manual = 0
    total_gen_calls = 0
    rounds_with_llm = 0
    action_counts: Dict[str, int] = {}

    for r in range(args.rounds):
        active = [
            (a.agent_id, a)
            for a in random.sample(agents, min(active_per_round, len(agents)))
        ]
        t = time.perf_counter()
        stats = await run_hybrid_round(
            env=env,
            active_agents=active,
            engine=engine,
            gateway=gateway,
            platform="twitter",
            round_num=r + 1,
            simulated_hour=(r * 2) % 24,
            decision_log_path=None,
            moderate=not args.no_moderate,
            llm_fallback=False,
            log=lambda m: None,
        )
        wall = (time.perf_counter() - t) * 1000
        total_decisions_ms += stats.decision_ms
        total_generation_ms += stats.generation_ms
        total_manual += stats.manual_count
        total_generated += stats.generated_count
        gen_calls = getattr(gateway, "calls", 0)
        llm_calls_this_round = gen_calls - total_gen_calls
        total_gen_calls = gen_calls
        if llm_calls_this_round:
            rounds_with_llm += 1
        for k, v in stats.actions.items():
            action_counts[k] = action_counts.get(k, 0) + v
        print(
            f"round {r + 1:>3}: wall={wall:7.0f}ms decisions={stats.decision_ms:6.0f}ms "
            f"generation={stats.generation_ms:6.0f}ms manual={stats.manual_count} "
            f"generated={stats.generated_count} llm_calls={llm_calls_this_round}"
        )

    n = args.rounds * active_per_round
    print("\n===== SUMMARY =====")
    print(f"rounds: {args.rounds}, active/round: {active_per_round}, total agent-decisions: {n}")
    print(f"decision time: {total_decisions_ms / 1000:.1f}s total "
          f"({total_decisions_ms / n:.0f}ms/agent)")
    print(f"generation time: {total_generation_ms / 1000:.1f}s total "
          f"({total_generation_ms / n:.0f}ms/agent amortized)")
    print(f"actions: {dict(sorted(action_counts.items(), key=lambda x: -x[1]))}")
    print(f"LLM calls (hybrid): {total_gen_calls} in {rounds_with_llm}/{args.rounds} rounds")
    print(f"LLM calls (MiroFish baseline): {n} (one tool-call per active agent per round)")
    print(f"LLM call reduction: {100 * (1 - total_gen_calls / max(n, 1)):.1f}%")


asyncio.run(main())