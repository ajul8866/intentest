# Intentest — HTTP & WebSocket API

All endpoints are served by the FastAPI application in `backend/app/main.py`
under the prefix `/api/v1` unless noted otherwise. Every response body is
Pydantic-validated; list endpoints support `limit` / `offset` pagination and
return `{ "items": [...], "total": <int> }`.

Errors use a single envelope:

```json
{
  "error": {
    "code": "SCOPE_VIOLATION",
    "message": "target 10.0.0.9:22 is outside scope 'lab'",
    "details": {"target": "10.0.0.9:22", "scope_id": "..."},
    "request_id": "..."
  }
}
```

Codes: `NOT_FOUND`, `VALIDATION_ERROR`, `ILLEGAL_TRANSITION`,
`PRECONDITION_FAILED`, `SCOPE_VIOLATION`, `LEASE_HELD`, `BUDGET_EXHAUSTED`,
`LLM_UNAVAILABLE`, `SANDBOX_UNAVAILABLE`, `BASELINE_UNAVAILABLE`,
`INTEGRITY_VIOLATION`, `CONFLICT`, `INTERNAL`.

---

## 1. Health & settings

| Method | Path | Purpose |
|---|---|---|
| GET | `/healthz` | liveness — returns `{"status":"ok"}` without touching Postgres |
| GET | `/readyz` | readiness — Postgres, pgvector, Docker socket, LLM capability cache |
| GET | `/api/v1/settings` | effective settings (secrets redacted) + capability matrix |
| POST | `/api/v1/settings/llm/probe` | re-run the capability probe (GET /models, chat, tool_calling, json_schema, json_object, embeddings, streaming, max_context) |
| GET | `/api/v1/settings/llm` | resolved per-role model ids and strategy resolution (`TOOL_CALLING > JSON_SCHEMA > JSON_OBJECT > PROMPT_JSON`) |

`/readyz` body:

```json
{
  "status": "ready",
  "checks": {
    "database": {"ok": true, "detail": "postgresql 16.3 / vector 0.7.0"},
    "docker":   {"ok": true, "detail": "server 27.1"},
    "llm":      {"ok": true, "detail": "strategy=TOOL_CALLING model=deepseek-v4-pro"},
    "embedding":{"ok": true, "detail": "api:Qwen3-Embedding-4B dim=2560"}
  }
}
```

---

## 2. Projects

| Method | Path | Body / Query | Notes |
|---|---|---|---|
| POST | `/api/v1/projects` | `ProjectCreate` | creates project + default scope + START fact |
| GET | `/api/v1/projects` | `?status=&limit=&offset=` | list with counters |
| GET | `/api/v1/projects/{id}` | — | project + counters (facts, intents, rounds) |
| PATCH | `/api/v1/projects/{id}` | `ProjectUpdate` | goal text, max_rounds, config |
| POST | `/api/v1/projects/{id}/start` | — | `INITIALIZING -> RUNNING`; seeds BOOTSTRAP intent |
| POST | `/api/v1/projects/{id}/pause` | — | `RUNNING -> STALLED` (operator pause), releases project lease |
| POST | `/api/v1/projects/{id}/resume` | — | `STALLED -> RUNNING` |
| POST | `/api/v1/projects/{id}/cancel` | — | `-> CANCELLED`, bulk-completes unfinished intents, reason `CANCELLED` |
| GET | `/api/v1/projects/{id}/status` | — | compact status for polling clients |
| DELETE | `/api/v1/projects/{id}` | — | only when not `RUNNING` |
| GET | `/api/v1/projects/{id}/scopes` | — | list scopes |
| POST | `/api/v1/projects/{id}/scopes` | `ScopeCreate` | `UNIQUE(project_id,name)` |
| DELETE | `/api/v1/projects/{id}/scopes/{scope_id}` | — | 409 if facts reference the scope |

`ProjectCreate`:

```json
{
  "name": "sqli-easy-01-intentest-r1",
  "goal_text": "Retrieve the flag from http://10.20.0.11:8080",
  "goal_anchor": {"flag_regex": "flag\\{[^}]+\\}"},
  "scope": {"name": "lab", "allowed_hosts": ["10.20.0.11"], "allowed_cidrs": ["10.20.0.0/24"],
            "allowed_ports": [80, 8080], "denied_actions": ["rm -rf /"]},
  "max_rounds": 40,
  "config": {"intent_guidance_enabled": true, "arm": "intentest", "repeat": 1,
             "challenge_id": "sqli-easy-01"}
}
```

Project status enum: `INITIALIZING | RUNNING | STALLED | COMPLETED | FAILED | CANCELLED`.
Termination reasons: `GOAL | UNREACHABLE | STALLED | BUDGET | CANCELLED`.

---

## 3. Graph

| Method | Path | Query | Notes |
|---|---|---|---|
| GET | `/api/v1/projects/{id}/graph` | `?view=attack\|merged\|full&limit=` | nodes + edges of G |
| GET | `/api/v1/projects/{id}/graph/neighbours` | `?node_id=&depth=1` | lazy expansion for the Graph Explorer |
| GET | `/api/v1/projects/{id}/graph/path-to-goal` | — | shortest alternating fact→intent→fact path to the goal anchor |
| POST | `/api/v1/projects/{id}/graph/integrity-check` | — | runs `verify_acyclic()` + invariant sweep |
| GET | `/api/v1/projects/{id}/graph/stats` | — | nodes/edges/depth/frontier/branching metrics |

Graph payload (`view=full`):

```json
{
  "project_id": "...",
  "nodes": [
    {"id": "f:...", "type": "fact", "kind": "ENDPOINT", "label": "GET /login.php",
     "verified": true, "attributes": {...}, "mitre_technique": null, "created_at": "..."},
    {"id": "i:...", "type": "intent", "action_type": "EXPLORE", "state": "COMPLETED",
     "tactic": "Initial Access", "objective": "SQLi on login.php", "round_index": 3}
  ],
  "edges": [
    {"id": "efi:...", "source": "f:...", "target": "i:...", "type": "TRIGGERS",
     "role": "PRECONDITION", "weight": 1.0},
    {"id": "eif:...", "source": "i:...", "target": "f:...", "type": "PRODUCES"},
    {"id": "corr:...", "source": "i:...", "target": "p:...", "type": "CORRESPONDS",
     "method": "cosine"}
  ]
}
```

Edge legend: solid `TRIGGERS` (E_f→i), dashed `PRODUCES` (E_i→f),
dotted `CORRESPONDS` (cross-link to an offline `intent_pattern`).

`integrity-check` response:

```json
{"acyclic": true, "violations": [],
 "counts": {"facts": 41, "intents": 17, "efi": 44, "eif": 15, "cycles": 0}}
```

---

## 4. Facts

| Method | Path | Notes |
|---|---|---|
| GET | `/api/v1/projects/{id}/facts` | `?kind=&verified=&q=&limit=&offset=` |
| GET | `/api/v1/facts/{fact_id}` | fact + producing intent + superseded-by chain |
| GET | `/api/v1/facts/{fact_id}/evidence` | raw evidence text/blob ref |
| POST | `/api/v1/projects/{id}/facts` | manual fact injection (operator), still immutable |
| GET | `/api/v1/projects/{id}/findings` | verified `VULNERABILITY`/`FLAG`/`CREDENTIAL` facts grouped for the Findings tab |

Facts are append-only. `POST` with an existing canonical key returns `200` and
the **existing** row (nothing is mutated). Corrections are expressed with
`supersedes_fact_id`, which creates a new row.

---

## 5. Intents

| Method | Path | Notes |
|---|---|---|
| GET | `/api/v1/projects/{id}/intents` | `?state=&action_type=&limit=&offset=` |
| GET | `/api/v1/intents/{intent_id}` | intent + predecessors (`intent_source`) + result + injected context |
| POST | `/api/v1/projects/{id}/intents` | manual intent creation (validated: ≥1 verified predecessor, in scope) |
| POST | `/api/v1/intents/{intent_id}/cancel` | `CREATED -> COMPLETED` with `success=false` (operator abort) |
| GET | `/api/v1/projects/{id}/intents/frontier` | CREATED intents whose preconditions still hold |

---

## 6. Tasks, task runs, logs

| Method | Path | Notes |
|---|---|---|
| GET | `/api/v1/projects/{id}/tasks` | current task queue snapshot (pending + claimed) with priority ranks |
| GET | `/api/v1/projects/{id}/task-runs` | `?phase=&exit_status=` |
| GET | `/api/v1/task-runs/{task_run_id}` | timings, container id, tokens, agent turns, recovered facts |
| GET | `/api/v1/task-runs/{task_run_id}/logs` | `?stream=stdout\|stderr&tail=200` |
| POST | `/api/v1/task-runs/{task_run_id}/cancel` | sends abort to the worker, releases lease |
| GET | `/api/v1/projects/{id}/rounds` | per-turn round ledger used by the RQ2 metrics |

The task queue entry exposes the two-phase timers used by the UI:

```json
{"intent_id": "...", "action_type": "EXPLORE", "priority_rank": 0, "priority": 5,
 "worker_id": "...", "state": "CLAIMED",
 "main_timeout_s": 180, "degradation_timeout_s": 90, "phase": "MAIN",
 "started_at": "...", "lease_expires_at": "..."}
```

---

## 7. Workers

| Method | Path | Notes |
|---|---|---|
| POST | `/api/v1/workers/register` | `{name, tools_available, max_parallelism, priority_weight}` |
| POST | `/api/v1/workers/{id}/heartbeat` | updates `last_heartbeat`, `active_tasks` |
| GET | `/api/v1/workers` | load gauges: `active_tasks/max_parallelism`, status, cooldown |
| POST | `/api/v1/workers/{id}/drain` | `DRAINING`: stops new claims, keeps running tasks |
| POST | `/api/v1/workers/{id}/reset-circuit` | clears `CIRCUIT_OPEN` / `consecutive_failures` |
| POST | `/api/v1/workers/{id}/reap-orphans` | reaps containers labelled `intentest.project` for the worker |

Worker status: `IDLE | ACTIVE | COOLDOWN | CIRCUIT_OPEN | UNHEALTHY | DRAINING`.

---

## 8. Expert prompts (Cairn "Hint" primitive)

| Method | Path | Notes |
|---|---|---|
| POST | `/api/v1/projects/{id}/prompts` | `{content, priority}` — async expert hint |
| GET | `/api/v1/projects/{id}/prompts` | `?consumed=` |
| POST | `/api/v1/prompts/{prompt_id}/consume` | marks consumed by an intent (normally done by the worker) |

An unconsumed prompt wakes a `REASON` task even when
`facts_since_reason == 0` (Algorithm 1, line 7).

---

## 9. Intent retrieval & prediction (Algorithm 2) and the bridge (Algorithm 3)

| Method | Path | Notes |
|---|---|---|
| POST | `/api/v1/projects/{id}/retrieve` | run the 5-stage pipeline for a given intent/predecessor set |
| GET | `/api/v1/projects/{id}/retrievals` | persisted retrieval traces (funnel UI) |
| GET | `/api/v1/retrievals/{trace_id}` | full per-stage trace incl. rejections |
| GET | `/api/v1/projects/{id}/bridges` | injected bridge payloads (`bridge.injected`) |
| POST | `/api/v1/projects/{id}/patterns/ingest` | ingest corpus docs into the offline graph G_o |
| GET | `/api/v1/patterns` | `?vulnerability_type=&tactic=&min_confidence=` |
| POST | `/api/v1/patterns/validate` | re-runs the offline graph validator (isolated node, low confidence, missing MITRE id, disconnected topology) |
| GET | `/api/v1/patterns/error-paths` | negative samples, strictly separate from success edges |

`retrieve` request:

```json
{"intent_id": "...", "predecessor_fact_ids": ["..."], "top_k": 10, "depth": 2}
```

`retrieve` / trace response:

```json
{
  "trace_id": "...",
  "final_stage": "S2",
  "recommendation": {
    "pattern_id": "...", "name": "Union-based SQLi", "tactic": "Initial Access",
    "mitre_attack_id": "T1190", "vulnerability_type": "SQLI",
    "objective": "Exploit UNION injection on the id parameter",
    "confidence": 0.81, "match_method": "isomorphism"
  },
  "stages": [
    {"stage": "S1", "name": "semantic_probing", "backend": "api:Qwen3-Embedding-4B",
     "candidates": [{"pattern_id": "...", "score": 0.83}], "duration_ms": 41},
    {"stage": "S2", "name": "bounded_isomorphism",
     "candidates": [{"pattern_id": "...", "isomorphic": true, "query_nodes": 4,
                     "pattern_nodes": 9, "examined": 10}], "duration_ms": 63},
    {"stage": "S3", "name": "jaccard_fallback", "candidates": [], "duration_ms": 0},
    {"stage": "S4", "name": "strategic_alignment",
     "candidates": [{"pattern_id": "...", "score": 0.74, "rejections": []}], "duration_ms": 2},
    {"stage": "S5", "name": "llm_heuristic", "candidates": [], "duration_ms": 0}
  ],
  "rejections": [{"stage": "S4", "pattern_id": "...", "reason": "out_of_scope:10.0.0.9"}]
}
```

Note: the recommendation returned by `/retrieve` is *advisory only* — it is
never executed directly. Any candidate (including S5 output) must pass
`submit_for_validation()` (DAG precondition validation) before it can become an
intent, which is the IPIA defence described in `THREAT_MODEL.md`.

---

## 10. Experiments & reports

| Method | Path | Notes |
|---|---|---|
| POST | `/api/v1/experiments` | `{name, arms[], challenge_ids[], repeats, max_rounds}` |
| POST | `/api/v1/experiments/{id}/start` | materialises one project per (challenge, arm, repeat) |
| GET | `/api/v1/experiments` | list with progress counters |
| GET | `/api/v1/experiments/{id}/results` | per-challenge success/rounds/termination + aggregate tables |
| GET | `/api/v1/experiments/{id}/stats` | one-sided Fisher exact per arm pair |
| GET | `/api/v1/projects/{id}/report` | `?format=md\|json` — episode report |

`/experiments/{id}/results` mirrors the paper layout:

```json
{
  "table1": [{"arm": "intentest", "easy": 1.0, "medium": 0.909, "hard": 0.75, "overall": 0.882}],
  "table2": {"success": [{"arm": "intentest", "easy": 5.0, "medium": 7.2, "hard": 9.4, "overall": 7.0}],
             "failure": [{"arm": "intentest", "easy": null, "medium": 40.0, "hard": 31.7, "overall": 34.0}]},
  "table3": [{"arm": "intentest", "easy": 5.0, "medium": 7.2, "hard": 9.4, "overall": 7.0},
             {"arm": "no_intent", "easy": 5.0, "medium": 10.8, "hard": 18.3, "overall": 11.0}],
  "fisher": [{"arm_a": "intentest", "arm_b": "no_intent", "difficulty": "hard",
              "p_value": 0.0123, "alternative": "greater"}],
  "note": "Substitute benchmark: numbers are NOT the paper's Table 1/2/3 values."
}
```

---

## 11. WebSocket

```
GET /ws?channel=project:{project_id}
GET /ws?channel=global
```

Transport: Postgres `LISTEN intentest_events` → in-memory hub → connected
clients. Message envelope:

```json
{"channel": "project:...", "type": "fact.created", "ts": "2024-05-01T10:00:00Z",
 "project_id": "...", "payload": {...}}
```

Event types: `fact.created`, `fact.superseded`, `intent.created`,
`intent.state_changed`, `task.started`, `task.log`, `task.finished`,
`worker.status`, `project.status`, `retrieval.result`, `bridge.injected`,
`degradation.triggered`, `scheduler.backpressure`, `prompt.created`.

Clients send `{"type":"ping"}` and receive `{"type":"pong"}`; the server closes
idle sockets after 60 s. Late subscribers can replay from `event` using
`GET /api/v1/projects/{id}/events?since_id=` (audit/replay endpoint).

---

## 12. OpenAPI

`GET /openapi.json` and the Swagger UI at `/docs` are enabled outside
production. The frontend typed client (`frontend/src/api/client.ts`) is derived
from this schema; no mock data is used anywhere in the UI.
