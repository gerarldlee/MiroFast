<div align="center">

# MiroFast

**Laya-decided swarm simulation. The LLM only writes.**

*A faster MiroFish: upload a document, simulate hundreds of agents reacting on
Twitter and Reddit, get a prediction report — at a fraction of the LLM cost.*

> **This is a fork.** MiroFast is a derivative work of
> [MiroFish-Offline](https://github.com/nikmcfly/MiroFish-Offline), which is
> itself a fork of [MiroFish](https://github.com/666ghj/MiroFish) by 666ghj
> (supported by Shanda Group). The simulation engine is
> [OASIS](https://github.com/camel-ai/oasis) by CAMEL-AI; the decision layer
> is powered by [Laya](https://github.com/NandhaKishorM/laya). Most of the
> pipeline code (API, graph storage, frontend) is inherited from those
> projects; MiroFast's original contribution is the hybrid decision engine
> (`backend/app/decisions/`, the rewritten round loop, and the Laya-powered
> setup/report stages). Licensed AGPL-3.0, inherited from MiroFish.

</div>

## Why it's fast

MiroFish spends one LLM tool-calling round-trip per active agent per round —
choosing an action *and* writing its content. Most of that spend is decisions,
not prose. MiroFast splits the job:

| | MiroFish | MiroFast |
|---|---|---|
| Action choice (like / repost / comment / post / lurk) | LLM call, every agent, every round | **Laya: one batched forward pass per round** |
| Stance / sentiment (setup + analytics) | LLM calls | **Laya, batched** |
| Report tool routing (ReACT) | ~6 LLM calls / section | **Laya plans, ~2 LLM calls / section** |
| Interviewee selection | LLM call | **Laya, one batched pass** |
| Post / comment / quote text | LLM call | LLM call (the only thing an LLM must do) |
| Content safety | none | **Laya moderation gate** |

Direct actions (likes, reposts, dislikes, do-nothing) execute as OASIS
`ManualAction`s with **zero LLM calls** — so many rounds complete with no LLM
traffic at all. Per-round decision cost is flat regardless of crowd size
(one forward pass for the whole round), which unlocks agent counts the
LLM-per-decision architecture can't economically reach.

Measured on the dev machine (CPU-only, no GPU): ~1.2s per agent decision
amortized in batches, ~24s per 24-agent round, 0 LLM calls in 5 rounds
(mocks). With a GPU, Laya runs at ~33ms per decision (~7ms batched) — per the
Laya benchmarks. LLM calls drop by 60–90% depending on how talkative the
crowd is.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│ Vue 3 frontend (upload → graph → setup → run → report → chat) │
├──────────────────────────────────────────────────────────────┤
│ Flask API (60 routes, 1:1 MiroFish contract)                 │
├───────────────────────┬──────────────────────────────────────┤
│ Laya decision engine  │ Generation gateway                     │
│ · round action+target │ OpenAI-compatible (Ollama, cloud, …)  │
│   +stance, batched    │ per-purpose model overrides            │
│ · stance / sentiment  │ used ONLY for: personas, posts,       │
│ · moderation gate     │ comments, reports, interviews, NER    │
│ · tool planning       │                                       │
│ · interviewee ranking │                                       │
├───────────────────────┴──────────────────────────────────────┤
│ OASIS 0.2.5 simulation: hybrid round loop — ManualActions for │
│ direct actions, LLM only writes content (app/decisions/…)    │
├──────────────────────────────────────────────────────────────┤
│ Neo4j CE 5.x knowledge graph + Ollama embeddings             │
│ (hybrid 0.7 vector + 0.3 BM25 search)                        │
└──────────────────────────────────────────────────────────────┘
```

Key modules (all under `backend/app/`):

- `decisions/laya_engine.py` — Laya wrapper: batched round decisions, stance,
  sentiment, moderation, noul ranking, torch thread pinning
- `decisions/questions.py` — runtime question schemas (no fine-tuning)
- `decisions/round_policy.py` — the hybrid round: refresh → one batched Laya
  pass → ManualActions / generation / optional LLM fallback
- `llm/gateway.py` — per-purpose generation clients
  (`POSTS_*`, `REPORT_*`, … fall back to `LLM_*`)
- `scripts/run_parallel_simulation.py` — the simulation subprocess (all
  platforms; `--twitter-only` / `--reddit-only` supported)
- `decisions.jsonl` per platform — every decision with confidence + stance

## Quick start

### 1. Services

```bash
docker compose up -d neo4j ollama
docker exec mirofast-ollama ollama pull qwen2.5:7b
docker exec mirofast-ollama ollama pull nomic-embed-text
```

(or run your own Neo4j 5.15+ and Ollama; any OpenAI-compatible endpoint works
for generation — cloud or local)

### 2. Configure

```bash
cp .env.example .env
```

Fill `LLM_API_KEY` (any non-empty value for Ollama) — Laya downloads its
checkpoint from Hugging Face on first run (~1.3 GB, cached afterwards).

### 3. Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
python run.py                      # http://localhost:5001
```

### 4. Frontend

```bash
cd frontend
npm install
npm run dev                        # http://localhost:3000
```

## Tuning the decision engine

| Env var | Default | Purpose |
|---|---|---|
| `LAYA_MODEL` | `multilingual` | `english` is faster for pure-English corpora |
| `LAYA_DEVICE` | `auto` | `cuda` makes decisions ~33ms |
| `LAYA_THREADS` | `min(8, cores)` | torch CPU pinning (oversubscription is 20x slower) |
| `LAYA_BATCH_SIZE` | `16` | states per forward pass |
| `LAYA_CONFIDENCE_THRESHOLD` | `0.55` | floor for `LAYA_LLM_FALLBACK` |
| `LAYA_LLM_FALLBACK` | `false` | low-confidence agents fall back to the original LLM decision step |
| `LAYA_MODERATE_CONTENT` | `true` | Laya gate on generated text |
| `USE_LLM_FOR_AGENT_CONFIGS` | `false` | restore LLM-based agent config generation |
| `MIROFAST_PLAN_TOOLS` | `true` | Laya plans report tools; `false` restores full ReACT |

Decision quality levers: the action/target label wording in
`app/decisions/questions.py` shapes crowd behavior (e.g. how passive vs.
active the crowd skews). Decisions log to
`<sim>/twitter/decisions.jsonl` with per-agent confidence and stance for
auditing and calibration.

## Benchmark

```bash
python bench/bench_round.py --agents 50 --rounds 10
```

Prints per-round decision/generation latency, action distribution, LLM call
counts vs the MiroFish baseline (one call per active agent per round), and the
resulting call reduction. `--real-generation` switches from the mock to the
configured LLM.

## License

AGPL-3.0 — inherited from MiroFish. Laya is Apache-2.0; OASIS is Apache-2.0.