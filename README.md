# Intentest — Intent-Graph-Guided Automated Penetration Testing

> Reproduction of **“Staying on the Attack Path: Structured State for Long-Horizon Automated
> Penetration Testing.”**
>
> Intentest externalizes long-horizon penetration-testing state out of the LLM context window
> into a persistent, immutable **Fact–Intent DAG** `G = (F, I, E_f→i, E_i→f)`. Every exploration
> step is an intent edge grounded in at least one verified predecessor fact. A lease/FSM
> dispatcher schedules work with 180 s / 90 s two-phase degradation, a six-stage adaptive load
> balancer, and a five-stage intent retrieval/prediction pipeline whose recommendation is
> injected into the next Explore through an asynchronous **intent bridge** (`θ = 0.3`).

---

## 1. What is in this repository

| Plane | Location | Responsibility |
|---|---|---|
| API | `backend/app/api/**`, `backend/app/services/**`, `backend/app/schemas/**` | REST + WebSocket surface, thin routes, business logic in services |
| State | `backend/app/db/models/**`, `backend/app/db/repositories/**`, `backend/alembic/**` | Six core tables + support tables; immutable facts, write-once hyperedges |
| Graph core | `backend/app/core/dag/**` | `FactIntentDAG`, `IntentFSM`, invariants I1–I6, belief projection, events |
| Scheduler | `backend/app/core/scheduler/**` | Algorithm 1 tick, task factory, priority/Reason gate, 6-stage load balancer, leases, degradation policy, stall detector, harvester, queue cleanup |
| Knowledge plane | `backend/app/core/intent/**` | Offline MITRE-mapped intent graph, runtime attack graph, merge, Algorithm 2 (S1–S5), Algorithm 3 bridge, dedupe, prompts |
| LLM | `backend/app/core/llm/**` | OpenAI-compatible async client, capability probe, strategy resolution, parser/repair, usage accounting |
| Security | `backend/app/core/security/**` | Scope guard, sandbox isolation policy, prompt-injection scan, redaction |
| Execution | `backend/app/worker/**` | Worker daemon, agent loop, Docker sandbox, tools, fact extractor, verifier, failure classifier, degradation client |
| Frontend | `frontend/src/**` | React/TS/Vite/Tailwind/React Flow UI (Dashboard, Projects, Graph, Tasks, Workers, Logs, Findings, Retrieval Trace, Experiments, Settings) |
| Evaluation | `eval/**` | Harness (runner, metrics, ablation, stats, Fisher, report), challenge manifest, scripts |
| Labs | `labs/challenges/**` | Six intentionally vulnerable local targets (benchmark substitute) |
| Docs | `*.md` | `ARCHITECTURE.md`, `ALGORITHMS.md`, `API.md`, `THREAT_MODEL.md`, `LIMITATIONS.md`, `REPRODUCTION_REPORT.md` |

Processes: `backend` (FastAPI), `scheduler` (tick loop), `worker` (sandboxed executor). All three
share one image and differ only by entrypoint (`app/main.py`, `app/scheduler_main.py`,
`app/worker_main.py`).

---

## 2. Prerequisites

* **Python 3.12+** (3.13 supported) — `uv` recommended, `pip` works.
* **Node 20+** and npm for the frontend.
* **PostgreSQL 16 with the `pgvector` extension** (required by
  `intent_pattern.embedding VECTOR(2560)`).
* **Docker daemon** reachable from the worker (DinD or DooD socket mount) for the sandbox.
  A `local subprocess` sandbox mode exists **for tests only** and is loudly flagged.
* An **OpenAI-compatible LLM endpoint** (paper: DeepSeek V4 Pro via API) and an
  **embeddings endpoint** (paper: Qwen3-Embedding-4B, 2560-d). Both are configurable.
  A startup capability probe must report `chat` support or the process refuses to start with a
  clear message.

Minimum practical hardware: 8 vCPU / 16 GB RAM with a hosted LLM. The paper used an
Intel Xeon Silver 4114 @ 2.20 GHz / 128 GB RAM, Ubuntu 20.04.1; no GPU is required.

---

## 3. Quickstart (Docker Compose)

```bash
cp .env.example .env            # set LLM_BASE_URL / LLM_API_KEY / models
docker compose up --build       # postgres+pgvector, backend, scheduler, worker x2, frontend
```

Then:

* UI — <http://localhost:5173>
* API — <http://localhost:8000/api/v1> (OpenAPI at `/docs`, `/openapi.json`)
* Liveness — `GET /healthz`; readiness (db/docker/llm/embedding) — `GET /readyz`

Run migrations inside the stack (also executed automatically by the backend entrypoint):

```bash
docker compose exec backend alembic upgrade head
```

### Lab benchmark stack

```bash
docker compose -f docker-compose.yml -f docker-compose.labs.yml up --build
docker compose exec backend python eval/scripts/seed_labs.py
```

The six lab targets (`sqli-easy-01`, `sqli-medium-waf`, `sqli-hard-sourceleak`, `ssrf-medium`,
`upload-easy`, `deserialize-hard`) are declared in `eval/challenges/manifest.yaml` with
`writeup_excluded: true` so that no write-up ever enters the offline graph `G_o`.

---

## 4. Local development (without Compose)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                 # or: uv sync --extra dev

cp .env.example .env
export DATABASE_URL=postgresql+asyncpg://intentest:intentest@localhost:5432/intentest
alembic upgrade head

uvicorn app.main:app --reload --app-dir backend            # API
python -m app.scheduler_main --app-dir backend             # scheduler
python -m app.worker_main --app-dir backend                # worker
```

Frontend:

```bash
cd frontend
npm install
npm run dev        # Vite dev server, proxies /api and /ws to :8000
npm run build      # production bundle
```

---

## 5. Configuration

All settings are environment variables (see `.env.example`); secrets are redacted in
`GET /api/v1/settings`.

| Variable | Default | Meaning |
|---|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://intentest:intentest@localhost:5432/intentest` | Postgres 16 + pgvector |
| `LLM_BASE_URL` / `LLM_API_KEY` | — | OpenAI-compatible endpoint |
| `LLM_MODEL_PLANNING` / `_EXECUTION` / `_VERIFICATION` / `_REPORTING` | — | per-role model ids |
| `EMBEDDING_MODEL` | `Qwen3-Embedding-4B` | embedding model name |
| `EMBEDDING_BASE_URL` / `EMBEDDING_API_KEY` | falls back to `LLM_*` | embeddings endpoint |
| `EMBEDDING_DIM` | `2560` | vector dimension (`VECTOR(2560)`) |
| `MAIN_TIMEOUT_S` | `180` | phase-1 task timeout (Bootstrap/Explore) |
| `DEGRADATION_TIMEOUT_S` | `90` | phase-2 emergency timeout |
| `INTENT_LEASE_S` | `360` | intent lease TTL (heartbeat 20 s) |
| `PROJECT_LEASE_S` | `600` | exclusive project lease TTL (renewed each tick) |
| `TICK_SECONDS` | `2` | scheduler Δt |
| `MAX_ROUNDS` | `40` | per-task round budget (all agent turns, incl. Reason) |
| `BRIDGE_THETA` | `0.3` | bridge activation cosine threshold |
| `SUBGRAPH_TOP_K` | `10` | S1 candidate count `k` |
| `JACCARD_THRESHOLD` | `0.5` | S3 fuzzy-inference threshold |
| `GOAL_ALIGN_THRESHOLD` | `0.55` | S4 alignment threshold |
| `CONFIDENCE_THRESHOLD` | `0.6` | offline graph validator confidence floor |
| `ALPHA` / `BETA` / `GAMMA` | `0.5` / `0.3` / `0.2` | S4 score weights |
| `STALL_TICKS` | `15` | ticks without a new fact before `STALLED` |
| `INTENT_GUIDANCE_ENABLED` | `true` | ablation flag: keeps DAG+scheduler, disables retrieval+bridge |
| `SANDBOX_IMAGE` | `intentest/tools:latest` | tool sandbox image |
| `LOG_LEVEL` | `INFO` | structlog JSON level |

`INTENT_GUIDANCE_ENABLED=false` is the **Intentest–noIntent** ablation arm: the graph, FSM,
scheduler and degradation path stay intact; only `core/intent` retrieval and the bridge are
disabled (assert no `intent_pattern_id` and no `injected_context` are produced).

---

## 6. Using the system

1. **Create a project** (`POST /api/v1/projects`) with `goal_text`, a `goal_anchor`
   (`{"flag_regex": "flag\\{[^}]+\\}"}`), an isolated `scope` (allowed hosts/CIDRs/ports and
   denied actions), `max_rounds`, and `config`.
2. **Start it** (`POST /projects/{id}/start`). `INITIALIZING → RUNNING` seeds the BOOTSTRAP
   intent grounded in `START` + goal-anchor facts.
3. **Watch it live.** The scheduler ticks every 2 s: harvest → reclaim leases → queue cleanup →
   project lease → `TaskFactory.evaluate` → priority sort → Reason gate → 6-stage load balance →
   CAS claim → dispatch. Facts, intents, tasks, logs and retrieval traces stream over
   `WS /ws?channel=project:{id}` (fed by Postgres `LISTEN intentest_events`).
4. **Termination** is one of `GOAL | UNREACHABLE | STALLED | BUDGET | CANCELLED`. A goal hit
   freezes the graph: unfinished intents are bulk-migrated to `COMPLETED` and the project lease is
   cleared.
5. **Export** a report: `GET /api/v1/projects/{id}/report?format=md|json`.

Equivalent CLI walkthrough:

```bash
curl -s localhost:8000/api/v1/projects -H 'content-type: application/json' -d '{
  "name": "sqli-easy-01",
  "goal_text": "Exploit the target and retrieve the flag",
  "goal_anchor": {"flag_regex": "flag\\{[^}]+\\}"},
  "scope": {"name": "lab", "allowed_hosts": ["sqli-easy-01"], "allowed_cidrs": ["172.28.0.0/16"],
            "allowed_ports": [80, 443], "denied_actions": ["rm -rf", "DROP TABLE"]},
  "max_rounds": 40,
  "config": {"intent_guidance_enabled": true, "challenge_id": "sqli-easy-01", "arm": "intentest", "repeat": 1}
}' | jq -r .id

curl -s -X POST localhost:8000/api/v1/projects/<ID>/start
```

---

## 7. Intent knowledge plane

* **Offline graph `G_o`** — build it once from a corpus of technique documents:
  ```bash
  python eval/scripts/build_offline_graph.py --corpus eval/corpus --validate
  ```
  Pipeline: corpus → blacklist filter → LLM extractor (temperature 0, strict JSON, ≤3 retries;
  every record must carry a target network entity, a vulnerability type and a MITRE ATT&CK id;
  error paths go to a **separate** `error_path` collection) → graph assembly → validator
  (rejects the whole graph on disconnected topology, degree-0 nodes, `confidence < 0.6`, or a
  missing MITRE id) → embeddings → persistence.
* **Runtime attack graph `G_a`** — reconstructed from verified facts plus persisted `E_f→i` /
  `E_i→f`.
* **Merge** — cross-links by exact `(kind, entity, vuln)` → cosine ≥ 0.75 → attribute Jaccard;
  every link is tagged with the matching method and rendered as a dotted `CORRESPONDS` edge.
* **Algorithm 2 (S1–S5)** — semantic probing (Eq. 2 cosine, top-k) → bounded subgraph
  isomorphism over the top-k only (≤12 query nodes, ≤60 pattern nodes, 5 s) → Jaccard fallback
  (only when S2 is empty) → strategic-intent alignment (scope/authorization/destructive pruning,
  `α·cos(obj,goal) + β·cos(obj,primary_chain) + γ·confidence ≥ 0.55`) → LLM heuristic reasoning on
  a reduced log. **Every** candidate, including S5 output, must pass
  `submit_for_validation()`, which re-runs DAG precondition validation (I3 + I5); rejection
  returns `None`. Full per-stage traces are persisted for the Retrieval Trace UI.
* **Algorithm 3 (bridge)** — fires only when an EXPLORE concluded successfully with ≥1 new fact
  and a recommendation exists; drops when `cos(state_vector, φ(i*.objective)) < 0.3`; dedupes on
  `sha256(i*.id | hash_blocks)`; builds B1 (next tactical action), B2 (evidence chain: offline
  historical chain, top-3 Jaccard-adjacent patterns, cross-links), B3 (goal, phase, allowed
  targets, recent fact clusters n=10, pending intents); applies `strip_meta` as an **allow-list
  serializer** (no scores, report names, matching methods or stage indices survive); stores in
  `intent.injected_context`, consumed at the next execution start.

---

## 8. Security model (summary)

* **Scope guard** — every tool invocation is checked against the project's isolated `scope`
  (hosts/CIDRs/ports, denied actions); violations raise `ScopeViolation` and emit a metric.
* **Sandbox isolation** — per-project bridge network, read-only rootfs, `tmpfs /tmp 256m` and
  `/work 512m`, memory/CPU/pids limits, `cap_drop=ALL`, `no-new-privileges`, and labels
  (`intentest.project=<id>`) for orphan reaping. Teardown is `SIGTERM → 5 s → SIGKILL` in a
  `finally` block, plus a lifecycle supervisor that reaps orphans by label.
* **Prompt-injection defence** — tool output is always wrapped in `<untrusted_tool_output>` with a
  treat-as-data directive; instruction-like patterns are logged. Recommendations (including S5
  output) can never create graph edges directly: the IPIA defence is
  `submit_for_validation()`.
* **Redaction** — secrets are redacted in logs, events, reports and API responses.

See `THREAT_MODEL.md` for the full analysis.

---

## 9. Testing

```bash
make test            # pytest -q  (unit + integration)
pytest -q backend/tests/unit
pytest -q backend/tests/integration
pytest -q -m e2e     # full episode on sqli-easy-01, ablation flag, crash-resume
```

Coverage highlights (see `backend/tests/**`):

* `test_fsm` — exactly four legal transitions, all others raise `IllegalTransition`, 50-way CAS
  race has exactly one winner.
* `test_dag_acyclicity` — 10 k random bipartite graphs, injected back-edge detected.
* `test_fact_immutability` — duplicate canonical key returns the existing row, `UPDATE` raises,
  `supersedes_fact_id` creates a new row.
* `test_load_balancer` — one test per filter stage plus a hand-computed three-worker table.
* `test_bridge` — θ boundary at 0.299 (drop) / 0.300 (inject), metadata fully stripped, dedupe,
  activation guard.
* `test_retrieval_pipeline` — synthetic chain yields S2, then S3, then S5 as edges/patterns are
  removed; full trace at every stage.
* `test_degradation_recovery` — 200 s stall flips to `DEGRADATION`, container id unchanged, ≥1
  fact recovered with `verification_method="degradation_log"`, container destroyed exactly once;
  a REASON task fails without degradation.
* `test_concurrency` — 20 parallel workers, no duplicate facts, no orphan containers, no intent
  executed twice.

---

## 10. Experiments

```bash
make experiment       # or: python eval/scripts/run_experiment.py --config eval/config/experiment.yaml
python eval/scripts/analyze.py --results eval/results/*.json
```

The harness creates **one project per (challenge, arm, repeat)**, enforces the 40-round budget and
records `success`, `rounds`, `termination_reason`, per-difficulty aggregates, mean rounds for
successful and failed tasks, and one-sided Fisher exact p-values
(`scipy.stats.fisher_exact([[a,b],[c,d]], alternative="greater")`). `report.py` emits
`REPRODUCTION_REPORT.md` shaped like Tables 1/2/3 of the paper, containing **only measured
numbers** from this repository's runs.

Arms: `intentest`, `no_intent` (ablation via `INTENT_GUIDANCE_ENABLED=false`), and
`plaintext_memory` (in-repo approximate plain-text-memory baseline, labelled approximate).
PentestGPT, PentestAgent, Pentest-R1, VulnBot and Claude Code are **not reproducible** here; their
adapters raise an explicit, documented `BASELINE_UNAVAILABLE` error rather than fabricating a
number.

**The local benchmark is a substitute**, not the paper's private 34-challenge CTF corpus. Results
from it must never be equated with the published Table 1/2/3 values. See
`REPRODUCTION_REPORT.md` and `LIMITATIONS.md`.

---

## 11. Repository map

```
intentest/
├── backend/app/            # FastAPI app, core, worker, services, schemas, prompts
├── backend/alembic/        # 0001_initial migration (tables, triggers, partial indexes)
├── backend/tests/          # unit / integration / e2e
├── backend/Dockerfile{,.worker,.tools}
├── frontend/src/           # React/TS UI (no mock data)
├── eval/                   # harness, challenge manifest, scripts
├── labs/challenges/        # six vulnerable targets
├── docker-compose.yml, docker-compose.labs.yml, .env.example, Makefile
├── pyproject.toml, alembic.ini
└── ARCHITECTURE.md, ALGORITHMS.md, API.md, THREAT_MODEL.md,
    LIMITATIONS.md, REPRODUCTION_REPORT.md, README.md
```

---

## 12. Make targets

| Target | Action |
|---|---|
| `make up` | `docker compose up --build` |
| `make down` | `docker compose down -v` |
| `make labs` | compose + lab overlay, then seed labs |
| `make migrate` | `alembic upgrade head` |
| `make test` | `pytest -q` |
| `make lint` | `ruff check backend eval` |
| `make graph` | build/validate the offline intent graph |
| `make experiment` | run the evaluation harness end-to-end |
| `make report` | regenerate `REPRODUCTION_REPORT.md` from results |

---

## 13. Documentation index

| Document | Contents |
|---|---|
| `ARCHITECTURE.md` | deployment topology, layers, invariants I1–I6, tick order, event bus, directory map |
| `ALGORITHMS.md` | normative pseudocode for Definitions 1–2, Eq. 1–2, Algorithms 1–3, constants table |
| `API.md` | every REST endpoint, WS channels/event types, error envelope and codes |
| `THREAT_MODEL.md` | assets, trust boundaries, attacker classes, IPIA defence, residual risks |
| `LIMITATIONS.md` | what is not reproducible (private corpus, exact prompts, baselines), chosen defaults, deviations |
| `REPRODUCTION_REPORT.md` | measured results on the substitute benchmark, tables, Fisher tests, threats to validity |

---

## 14. Citation

```bibtex
@article{intentest2025staying,
  title   = {Staying on the Attack Path: Structured State for Long-Horizon Automated Penetration Testing},
  author  = {Anonymous},
  journal = {Under review},
  year    = {2025}
}
```

## 15. License / responsible use

Released for research and education only. The lab targets are intentionally vulnerable and must be
run on an isolated network. Never point Intentest at infrastructure you do not own or have explicit
written authorization to test; the scope guard is a safety rail, not a legal authorization.
