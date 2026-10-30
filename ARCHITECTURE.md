# Intentest — Architecture

> Structured State for Long-Horizon Automated Penetration Testing.
> Intentest is an *intent-graph-guided* pentest agent: the long-horizon state of an
> engagement lives in Postgres as an immutable **Fact–Intent DAG**, never in the LLM
> context window. Every LLM call receives a *projection* of that state (a token-budgeted
> belief snapshot) plus, optionally, an asynchronously injected retrieval recommendation
> (the *intent bridge*).

---

## 1. Deployment topology

```
                       ┌──────────────────────────────┐
                       │        frontend (nginx)      │  React 18 / TS / Vite / Tailwind
                       │  React Flow graph explorer   │  WS client + REST client
                       └──────────────┬───────────────┘
                                      │ /api  +  /ws
                       ┌──────────────▼───────────────┐
                       │      backend (FastAPI)       │  REST + WebSocket hub
                       │  api/routes/*  api/ws/*      │  services/*  (business logic)
                       └──────────────┬───────────────┘
                                      │ SQLAlchemy async (asyncpg)
      ┌───────────────────────────────┼────────────────────────────────┐
      │                               │                                │
┌─────▼──────┐              ┌─────────▼─────────┐            ┌─────────▼─────────┐
│ PostgreSQL │              │ scheduler_main.py │            │  worker_main.py   │
│ 16+pgvector│◄────────────►│  TickScheduler    │            │  WorkerDaemon ×N  │
│  LISTEN    │   leases     │  Algorithm 1      │  dispatch  │  AgentLoop        │
│  NOTIFY    │   CAS claims │  LoadBalancer     │───────────►│  DockerSandbox    │
└─────┬──────┘              │  LeaseReaper      │            │  tools/registry   │
      │                     │  StallDetector    │            └─────────┬─────────┘
      │                     └───────────────────┘                      │ docker SDK
      │                                                                │
      │                     ┌───────────────────┐            ┌─────────▼─────────┐
      └── event outbox ────►│  intentest_events │            │ intentest/tools   │
           (pg_notify)       │  LISTEN → WS hub  │            │ (kali sandbox)    │
                            └───────────────────┘            └───────────────────┘
```

* **Postgres is the only state store.** Redis is not required; the event bus is
  `LISTEN/NOTIFY` on channel `intentest_events`, fed by a trigger on the `event` table.
* **Three processes share one image**: `backend` (API), `scheduler` (control plane),
  `worker` (execution plane). The scheduler never talks to Docker; the worker never
  mutates the DAG structure (it only appends facts and concludes intents).
* **Scaling**: N workers register their capability descriptor
  `(tools_available D_n, max_parallelism C_n, priority_weight w(n))`; the scheduler is
  safe to run as multiple replicas because every project is guarded by an exclusive
  `project_lease` and every intent claim is a CAS.

---

## 2. Layers

| Layer | Path | Responsibility |
|---|---|---|
| Config / logging | `backend/app/config.py`, `logging.py` | every env var, structlog JSON with `project_id`/`intent_id`/`task_run_id` correlation |
| Data | `backend/app/db/models/*`, `repositories/*` | six core tables + support tables; all invariants enforced at the repo boundary |
| Graph core | `backend/app/core/dag/*` | `FactIntentDAG`, `IntentFSM`, `validator`, `belief`, `events` |
| Control plane | `backend/app/core/scheduler/*` | tick loop, task factory, priority, 6-stage load balancer, leases, degradation policy, stall detection, harvester, queue cleanup |
| Intent knowledge | `backend/app/core/intent/*` | offline graph build, runtime attack graph, merge, 5-stage retrieval, embedding provider, bridge, dedupe, prompts |
| LLM | `backend/app/core/llm/*` | OpenAI-compatible async client, capability probe, strategy resolution, parser/repair, retry, usage accounting |
| Security | `backend/app/core/security/*` | scope guard, sandbox isolation policy, prompt-injection detection, redaction |
| Execution plane | `backend/app/worker/*` | daemon, executor, agent loop, fact extractor, verifier, failure classifier, degradation client, sandbox |
| API | `backend/app/api/*`, `services/*`, `schemas/*` | thin routes, Pydantic v2 schemas, service objects |
| Evaluation | `eval/harness/*`, `eval/challenges`, `eval/scripts` | runner, metrics, ablation, Fisher statistics, report writer |
| Labs | `labs/challenges/*` | six intentionally vulnerable targets with `flag{...}` |

---

## 3. The Fact–Intent DAG (Definition 1)

```
      f_start ──E_f→i──► i_bootstrap ──E_i→f──► f_host:10.0.0.5
                              │                       │
                              │                       ├─E_f→i─► i_explore(sqli-probe)
                              │                       │             │
                              │                       │             └─E_i→f─► f_param:id
                              └─E_f→i─► i_reason ◄───┘
```

* `F` — immutable facts. `kind ∈ {START, GOAL, HOST, PORT, SERVICE, ENDPOINT, PARAM,
  CREDENTIAL, VULNERABILITY, SOURCE_LEAK, FILE, SESSION, FLAG, NOTE}`.
* `I` — intents (`BOOTSTRAP | EXPLORE | REASON | VERIFY`), the *edges of exploration*.
* `E_f→i` — `intent_source` hyperedges, roles `PRECONDITION | SUPPORTING | GOAL_ANCHOR |
  CONTRADICTION`, write-once at intent creation.
* `E_i→f` — exactly the `fact.derived_from_intent_id` column. A fact produced by an
  intent points back at it; this makes the graph bipartite and *alternating*.

Invariants (I1–I6) are enforced in code, not by convention:

| id | invariant | enforcement |
|---|---|---|
| I1 | facts are never UPDATEd | repo has no update path; a SQLAlchemy event listener raises on `UPDATE fact`; corrections insert a new row with `supersedes_fact_id` |
| I2 | `intent_source` is write-once | insert-only repo; `UNIQUE(intent_id, fact_id)` |
| I3 | every predecessor is verified, same project & scope | `validator.validate_preconditions` |
| I4 | `E_i→f` == `derived_from_intent_id` | single writer (`fact_repo.create`) |
| I5 | acyclicity is structural | all `E_f→i` frozen before an intent can be claimed; `verify_acyclic()` DFS colour sweep for audit |
| I6 | terminal irreversibility | `IntentFSM` allows no transition out of `COMPLETED`; goal hit bulk-completes the project |

**Why this defeats IPIA (injected-precondition attacks):** a recommendation — including
one produced by the S5 LLM heuristic — can only enter the graph through
`submit_for_validation()`, which re-runs I3 (verified + same scope) and I5. A fabricated
precondition cannot be attached to an intent, so an attacker-controlled tool output can
never create an intent edge.

---

## 4. Control plane: one tick (Algorithm 1)

```
while True:                                  # Δt = TICK_SECONDS (2s)
  1  harvest()          task_runs finished? write back facts + conclude intents
  2  reclaim()          intent leases past expires_at → CLAIMED→CREATED
                        project leases past expires_at → released
  3  queue_cleanup()    drop task queues of cancelled/finished projects
  4  for each active project:
       if not project_lease.acquire_or_renew(TTL=600s): continue
  5     pending = TaskFactory.evaluate(project)     # Bootstrap/Explore/Verify/Reason
  6     pending.sort(PRIORITY_RANK, priority, created_at)
  7     τ = pending[0]
        if τ.kind == REASON and not reason_gate(project): continue
  8     worker = LoadBalancer.select(τ)             # 6 stages
        if worker is None: emit scheduler.backpressure; continue
  9     if not IntentFSM.claim(τ, worker, lease=360s): continue   # CAS, 1 winner
  10    dispatch(τ, worker)
```

**Reason gate** — REASON runs only when
`(facts_since_reason > 0 OR unconsumed prompt exists) AND no unclaimed EXPLORE exists
globally AND the per-project reason mutex is free`. This prevents an idle-spinning loop
that burns the 40-round budget on reflection.

**Six-stage adaptive load balancing** — each stage narrows the candidate pool, in order:
`protocol coherence (D_τ ⊆ D_n)` → `capacity (L(n) < C_n)` → `health` → `min w(n)` →
`min L(n)` → `seeded jitter tie-break`. An empty pool emits `scheduler.backpressure`
instead of starving the intent.

---

## 5. Two-phase degradation (§4.3)

```
task_run(phase=MAIN, timeout=180s)
   ├── OK ──────────────────────────────────────────► harvest facts
   ├── TimeoutError / LLMMalformedOutput
   │       │  container session is PRESERVED (never destroyed)
   │       ▼
   │   phase = DEGRADATION, timeout = 90s
   │   prompt  = EMERGENCY_TRUNCATION
   │   parse residual stdout/stderr → facts with
   │           verification_method="degradation_log"
   │   recovered_fact_ids = [...]
   └── REASON tasks are NOT degradable → fail, release lease
finally: SIGTERM → 5s → SIGKILL, container destroyed exactly once
```

A lifecycle supervisor reaps orphan containers by label `intentest.project`.

---

## 6. Intent knowledge plane

```
offline (build time)                       runtime (per project)
────────────────────                       ─────────────────────
corpus → blacklist → LLM extractor         verified facts + E_f→i / E_i→f
  (temp 0, ≤3 retries, MITRE id mandatory)          │
  → assemble graph → validator              G_a = attack graph
  (reject: disconnected / isolated /                │
   confidence<thr / no MITRE id)                    │
  → embed (2560-d) → persist                merge  (exact → cosine≥0.75 → Jaccard)
        │                                           │
        └──────────────► G_o  ◄─────────────────────┘
                          │
        Algorithm 2 (5 stages): S1 cosine → S2 bounded isomorphism →
        S3 Jaccard fallback → S4 goal alignment → S5 LLM heuristic
                          │
                  submit_for_validation()  ← every candidate, no exception
                          │
        Algorithm 3 bridge: activation guard → θ=0.3 → dedupe →
        3 blocks → strip_meta (allow-list) → intent.injected_context
```

`INTENT_GUIDANCE_ENABLED=false` keeps the DAG, scheduler, and worker untouched but
disables retrieval + bridge (the *noIntent* ablation arm).

---

## 7. Security model (two layers)

1. **Pre-execution**: `scope_guard` rejects any action whose host/CIDR/port is outside
   the project `scope`, and any action in `denied_actions`. `validate_action` enforces
   the per-intent tool allow-list and JSON schema.
2. **Execution isolation**: per-project bridge network, read-only rootfs, `tmpfs` for
   `/tmp` (256m) and `/work` (512m), memory/cpu/pids limits, `cap_drop=ALL`,
   `no-new-privileges`, label-based reaping.
3. **Post-execution**: every tool output is wrapped in `<untrusted_tool_output>` with a
   treat-as-data directive; instruction-like patterns are logged
   (`prompt_injection.scan`) and redaction removes secrets from logs/LLM context.

See `THREAT_MODEL.md`.

---

## 8. Persistence & replay

* `event` is a BIGSERIAL audit log; a trigger `pg_notify('intentest_events', payload)`.
* WS clients can replay from `?since=<event_id>`; the hub subscribes once per process.
* Every LLM call is one `llm_call` row (role, model, strategy, tokens, latency, hashes),
  so cost and prompt drift are auditable after the fact.
* Because the DAG is the state, a crashed scheduler/worker restarts and resumes: no fact
  is lost and no intent is executed twice (CAS + lease).

---

## 9. Directory map

```
backend/app/{config,logging,main,scheduler_main,worker_main}.py
backend/app/db/{models,repositories}/…
backend/app/core/{dag,scheduler,intent,llm,security,metrics}/
backend/app/worker/{sandbox,tools}/
backend/app/{api,services,schemas,prompts}/
backend/alembic/versions/0001_initial.py
backend/tests/{unit,integration,e2e}/
frontend/src/{pages,components,api,hooks,types}/
eval/{harness,challenges,scripts}/
labs/challenges/{sqli-easy-01,sqli-medium-waf,sqli-hard-sourceleak,ssrf-medium,upload-easy,deserialize-hard}/
```

See `ALGORITHMS.md` for the pseudocode-to-code mapping, `API.md` for the REST/WS
surface, and `LIMITATIONS.md` for every parameter that the paper does not publish.
