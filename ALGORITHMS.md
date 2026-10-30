# ALGORITHMS.md — Intentest Algorithms & Invariants

This document specifies, precisely and in executable-level detail, every algorithm
implemented by Intentest. It is the reference for the code in
`backend/app/core/` and `backend/app/worker/`. Each section names the source
module that implements it.

Paper: *Staying on the Attack Path: Structured State for Long-Horizon Automated
Penetration Testing*.

---

## 0. Notation

| Symbol | Meaning |
| --- | --- |
| `F` | set of facts (immutable, verified observations) |
| `I` | set of intents (exploration steps) |
| `E_f→i` | hyperedges intent ← predecessor facts (`intent_source`) |
| `E_i→f` | edges fact ← producing intent (`fact.derived_from_intent_id`) |
| `G = (F, I, E_f→i, E_i→f)` | the Fact–Intent DAG |
| `G_o` | offline (MITRE-mapped) intent pattern graph |
| `G_a` | runtime attack graph projected from `G` |
| `φ(·)` | embedding function (Qwen3-Embedding-4B, 2560-d) |
| `b_t` | belief state projection at round `t` |
| `D_n` | tool set available on worker `n` |
| `C_n` | max parallelism of worker `n` |
| `w(n)` | priority weight of worker `n` |
| `L(n)` | active tasks on worker `n` |

---

## 1. Definition 1 — Fact–Intent DAG

`backend/app/core/dag/graph.py :: FactIntentDAG`

`G = (F, I, E_f→i, E_i→f)` is a **bipartite**, **alternating**, **acyclic** graph:

* every edge has exactly one fact endpoint and one intent endpoint;
* every path alternates `f → i → f → i → …`;
* edges are **frozen before the producing intent executes**, so acyclicity is
  structural rather than checked-then-broken.

Reconstruction from Postgres:

```
F  = SELECT * FROM fact            WHERE project_id = :p
I  = SELECT * FROM intent          WHERE project_id = :p
Ef = SELECT * FROM intent_source   WHERE intent_id IN I
Ei = { (f.derived_from_intent_id, f.id) | f ∈ F, f.derived_from_intent_id ≠ NULL }
```

`FactIntentDAG.to_networkx()` returns a `networkx.DiGraph` with node attributes
`node_type ∈ {fact, intent}`, `kind`, `state`, and edge attributes
`relation ∈ {TRIGGERS, PRODUCES, CORRESPONDS}`.

### Invariants

| ID | Invariant | Enforcement site |
| --- | --- | --- |
| **I1** | Facts are never `UPDATE`d. Corrections insert a *new* fact whose `supersedes_fact_id` points at the old row. | `db/repositories/fact_repo.py`, DB trigger `fact_immutable` |
| **I2** | `intent_source` is write-once, at intent creation. No `UPDATE`/`DELETE`. | `intent_repo.create_intent()`, DB trigger `intent_source_write_once` |
| **I3** | Every predecessor fact of an intent must be `verified = true` and belong to the same project & scope. | `core/dag/validator.py :: validate_preconditions` |
| **I4** | `E_i→f` is exactly `fact.derived_from_intent_id` — a fact has at most one producing intent. | schema `UNIQUE`-free single column + repo write path |
| **I5** | Acyclicity is structural; `verify_acyclic()` is a DFS colour sweep exposed as `POST /graph/integrity-check`. | `core/dag/validator.py :: verify_acyclic` |
| **I6** | Terminal irreversibility: once `project.status = COMPLETED`, no intent may leave `CREATED`. | `core/dag/fsm.py`, `services/project_service.py` |

### Fact canonical key

`backend/app/db/repositories/fact_repo.py`

```
VOLATILE_ATTRS = {timestamp, ts, scanned_at, request_id, duration_ms, nonce}
fact_key     = sha256(kind ‖ "|" ‖ json.dumps(non-volatile attrs, sort_keys=True, separators=(',',':')))
content_hash = sha256(json.dumps({kind, attributes, value, evidence_hash}, sort_keys=True))
```

Insert is idempotent: `INSERT … ON CONFLICT (project_id, kind, fact_key) DO NOTHING
RETURNING *`; if no row is returned the existing row is fetched and returned
unchanged.

---

## 2. Definition 2 / Eq. 1 — Intent FSM

`backend/app/core/dag/fsm.py`

State space `S = {CREATED, CLAIMED, CONCLUDED, COMPLETED}`. Exactly **four**
transitions are legal:

| # | From | Action | To | Guard |
| --- | --- | --- | --- | --- |
| T1 | `CREATED` | `claim` | `CLAIMED` | CAS `UPDATE … SET state='CLAIMED', worker_id=:w, lease_expires_at=now()+360s WHERE id=:i AND state='CREATED'` → `rowcount == 1` |
| T2 | `CLAIMED` | `conclude` | `CONCLUDED` | `worker_id` of the caller must match |
| T3 | `CONCLUDED` | `goal` | `COMPLETED` | goal anchor matched, or project terminal bulk-migrate |
| T4 | `CLAIMED` | `timeout` | `CREATED` | reaper only, `now() > lease_expires_at`; clears `worker_id`, `lease_expires_at` |

Everything else raises `IllegalTransition(from_state, action)`.

```
δ(s, σ) defined only for the four rows above
```

**Goal freeze.** On `(CONCLUDED, goal)` the FSM additionally:

1. bulk-migrates every unfinished intent (`CREATED`/`CLAIMED`/`CONCLUDED`) to
   `COMPLETED` with `result.reason = "goal_reached"`;
2. sets `project.status = COMPLETED`, `project.termination_reason = GOAL`;
3. clears the project lease (`lease_owner = NULL`).

**CAS exactly-one-winner.** Under 50 concurrent claimers, the
`UPDATE … WHERE state='CREATED'` returns `rowcount == 1` for exactly one
transaction; all others see `0` and retry the next intent.

---

## 3. Algorithm 1 — Scheduler tick

`backend/app/core/scheduler/tick.py :: TickScheduler.tick()`

Runs every `TICK_SECONDS` (default `2`). The order is normative.

```
Algorithm 1: Tick(projects, workers)
 1  for each finished task_run in HARVEST_QUEUE:  harvest(task_run)        # facts/intents written back
 2  reclaim_expired_intent_leases()                                       # CLAIMED → CREATED
 3  reclaim_expired_project_leases()                                      # owner crash recovery
 4  queue_cleanup()                                                       # drop stale pending entries
 5  for p in active_projects:                                             # status = RUNNING
 6      if not acquire_or_renew_project_lease(p, ttl=600s): continue
 7      pending ← task_factory.evaluate(p)                                # {BOOTSTRAP, EXPLORE, VERIFY, REASON}
 8      sort pending by (PRIORITY_RANK[action_type], priority, created_at)
 9      τ ← first(pending);  if τ = ∅: continue
10      if τ.action_type = REASON:
11          require (p.facts_since_reason > 0 OR unconsumed_prompt(p))
12          and   no unclaimed EXPLORE globally
13          and   reason_mutex(p).acquire()
14          else: release lease; continue                                   # no idle spinning
15      n ← load_balancer.select(τ)                                        # Algorithm 1 lines 9–14
16      if n = ∅: emit scheduler.backpressure; continue
17      claimed ← intent_fsm.claim(τ, n, lease=360s)                        # CAS
18      if not claimed: continue
19      dispatch(n, claimed)                                                # worker channel
```

`PRIORITY_RANK = {EXPLORE: 0, BOOTSTRAP: 1, VERIFY: 2, REASON: 3}`.

### TaskFactory emission rules

`backend/app/core/scheduler/task_factory.py`

| Task | Emitted when |
| --- | --- |
| `BOOTSTRAP` | project in initial state: `START` + goal-anchor facts exist, **no** intents yet. The agent may solve directly and emit `Complete`. |
| `EXPLORE` | one per `CREATED` intent whose predecessors are all still `verified` |
| `VERIFY` | an unverified candidate fact exists (extension — see LIMITATIONS §b) |
| `REASON` | wake condition in line 11 above |

---

## 4. Algorithm 1 lines 9–14 — Six-stage adaptive load balancing

`backend/app/core/scheduler/load_balancer.py :: LoadBalancer.select()`

Stages run **in this order**, each *narrowing* the candidate pool. The first
stage that empties the pool yields `None` and a `scheduler.backpressure` event.

| Stage | Filter | Predicate |
| --- | --- | --- |
| 1 | protocol coherence | `task.required_tools ⊆ worker.tools_available` |
| 2 | capacity safety | `worker.active_tasks < worker.max_parallelism` |
| 3 | health probe | `worker.status ∉ {COOLDOWN, CIRCUIT_OPEN, UNHEALTHY, DRAINING}` and `cooldown_until ≤ now` |
| 4 | priority weight | keep only workers with `min(w(n))` over the pool |
| 5 | active load | keep only workers with `min(L(n))` over the pool |
| 6 | jitter tie-break | deterministic sort by `id`, then `random.choice` under the tick's seeded RNG |

Stage 4 keeps the *minimum* weight because lower `w(n)` means higher scheduling
priority (documented default, see LIMITATIONS). The stage that produced the
final pool is recorded on the dispatch event for the UI.

---

## 5. Two-phase degradation recovery

`backend/app/worker/degradation_client.py`, `backend/app/core/scheduler/degradation.py`

```
MAIN_TIMEOUT_S        = 180     # BOOTSTRAP / EXPLORE
DEGRADATION_TIMEOUT_S = 90
```

Trigger: `TimeoutError` or `LLMMalformedOutput` during a `BOOTSTRAP`/`EXPLORE`
task.

Recovery, exactly:

1. **Do not destroy the container.** Keep the same session — the residual
   stdout/stderr and the still-running processes are the recovery substrate.
2. `task_run.phase ← DEGRADATION`; emit `degradation.triggered`.
3. Re-enter the agent loop with the frozen `EMERGENCY_TRUNCATION` prompt:

   > 1. Stop all further probing.  2. Read residual stdout and stderr.
   > 3. Summarise every verified security fact you have observed.
   > 4. Emit ONLY the FACTS_SCHEMA JSON. Anything unreported is lost.

4. Budget `DEGRADATION_TIMEOUT_S = 90`.
5. Parse residual logs into partial facts; tag each
   `verification_method = "degradation_log"` (i.e. *not* independently verified).
6. Record the recovered ids in `task_run.recovered_fact_ids`.

`REASON` tasks are **not** degradable: on timeout they fail and release the
lease. Container teardown is guaranteed in `finally`
(`SIGTERM` → 5 s → `SIGKILL`) plus a lifecycle supervisor that reaps orphans by
the label `intentest.project=<project_id>`.

---

## 6. Offline intent-graph construction

`backend/app/core/intent/offline_graph.py`

```
corpus
  → blacklist filter                       (writeups excluded: manifest.writeup_excluded)
  → LLM extractor  (temperature 0, strict JSON, ≤3 retries)
        record MUST carry: target_entity, vulnerability_type, mitre_attack_id
        error paths → separate collection (error_path table)
  → graph assembly
  → internal validator
        REJECT WHOLE GRAPH if:
          - disconnected topology
          - any isolated node (degree 0)
          - confidence < CONFIDENCE_THRESHOLD (0.6)
          - missing MITRE ATT&CK id
  → embeddings φ (2560-d)
  → persistence (intent_pattern, intent_pattern_edge)
```

Error paths are stored in `error_path`, **strictly separate** from success edges,
so they can never be mistaken for enabling edges.

---

## 7. Runtime attack graph & merge

`backend/app/core/intent/attack_graph.py`, `merge.py`

`G_a` is projected from verified facts plus persisted `E_f→i` / `E_i→f`.

Cross-links `G_o ↔ G_a` are produced in this precedence order, and each link
records the method that produced it:

1. exact match on `(kind, entity, vuln)` → `method = "exact"`
2. cosine similarity `≥ 0.75` → `method = "cosine"`
3. attribute Jaccard `≥ JACCARD_THRESHOLD` → `method = "jaccard"`

---

## 8. Algorithm 2 — Five-stage retrieval & prediction

`backend/app/core/intent/retrieval.py` orchestrates
`backend/app/core/intent/stages/s1_embedding.py … s5_llm.py`.

### Line 1 — state vector

```
v = φ(predecessor attribute vector)
```

`φ` = Qwen3-Embedding-4B, 2560-d, with a logged fallback chain
(API → sentence-transformers → deterministic hashing).

### S1 — semantic probing

Top-k of `G_o` by Eq. 2:

```
sim_cos(v, φ(i)) = (v · φ(i)) / (‖v‖ ‖φ(i)‖)                              (Eq. 2)
```

`k = SUBGRAPH_TOP_K` (default 10). Always runs; always emits candidates.

### S2 — bounded subgraph isomorphism

Query = depth-2 predecessor pattern of the current frontier. Checked **only**
against the S1 Top-k candidates:

```
cost = O(k)          (never O(|G_o|))
bounds: ≤ 12 query nodes, ≤ 60 pattern nodes, 5 s timeout
networkx.algorithms.isomorphism.DiGraphMatcher(
    pattern, candidate,
    node_match=..., edge_match=...).subgraph_is_isomorphic()
```

### S3 — degraded fuzzy inference (only if S2 is empty)

```
J(A, B) = |A ∩ B| / |A ∪ B|
```

over attributed-neighbourhood token sets: node labels + `key=value` attributes +
edge relations. Threshold `JACCARD_THRESHOLD` (default `0.5`).

### S4 — strategic-intent alignment

Prunes candidates that are

* out of scope (`scope_guard`), or
* unauthorised lateral movement, or
* destructive conflicts with the goal (e.g. exfil goal vs `DROP TABLE`).

Surviving candidates are scored:

```
score = α · cos(obj, goal)
      + β · cos(obj, primary_chain(G_a))
      + γ · confidence
      ≥ GOAL_ALIGN_THRESHOLD                                             (0.55)
α = 0.5, β = 0.3, γ = 0.2
```

### S5 — LLM heuristic reasoning

Only reachable when S1–S4 yield nothing. Runs on a *reduced* log (token-budgeted
belief projection), never the full transcript.

### Mandatory validation gate

Every candidate — **including S5 output** — is returned only through

```python
retrieval.submit_for_validation(candidate) -> Intent | None
```

which re-runs `core/dag/validator.validate_preconditions`. Rejection returns
`None`. This is the IPIA (indirect prompt-injection-to-intent) defence: a
retrieved or model-generated recommendation can never create an intent with no
verified predecessor, an unverified predecessor, or a cross-scope predecessor.

### Trace

Each stage records `{stage, candidates, scores, rejections[]}` and is persisted
to `retrieval_trace` for the Retrieval Trace UI funnel.

---

## 9. Algorithm 3 — Asynchronous intent bridge

`backend/app/core/intent/bridge.py`

**Activation guard** — all must hold:

1. an `EXPLORE` intent concluded **successfully**;
2. it produced **≥ 1 new fact**;
3. a recommendation `i*` exists from Algorithm 2.

```
 1  s ← cos(state_vector, φ(i*.objective))
 2  if s < θ (θ = 0.3): drop, record bridge.dropped
 3  key ← sha256(i*.id ‖ hash_blocks(B1,B2,B3))
 4  if key seen for this project: drop (dedupe)
 5  B1 ← recommended next tactical action
 6  B2 ← supporting evidence chain
 7        (offline historical chain, Jaccard-adjacent patterns top-3, cross-links)
 8  B3 ← current project context
 9        (goal, phase, allowed targets, recent fact clusters n=10, pending intents)
10  payload ← strip_meta(B1, B2, B3)
11  intent.injected_context ← payload ; emit bridge.injected
12  consumed at the next execution start (injected_context → belief block)
```

`strip_meta` is an **allow-list serializer**, not a regex scrub: only the keys
in the schema survive. Similarity scores, source-report names, matching-method
details and stage indices are structurally absent from the output.

---

## 10. LLM integration

`backend/app/core/llm/`

* OpenAI-compatible async client; per-role `base_url`/`api_key`/`model`
  (`planning`, `execution`, `verification`, `reporting`).
* **Mandatory startup capability probe**: `GET /models`, `chat`,
  `tool_calling`, `json_schema`, `json_object`, `embeddings`, `streaming`,
  `max_context`. Persisted and surfaced in Settings. Startup fails with a clear
  message if `chat` is unsupported.
* Strategy resolution:
  `TOOL_CALLING > JSON_SCHEMA > JSON_OBJECT > PROMPT_JSON`.
  `PROMPT_JSON` = fenced block + repair round-trip.
* Parser: strip fences → first balanced `{…}` → Pydantic validate → **one**
  repair round-trip → `LLMMalformedOutput` (a degradation trigger).
* One `llm_call` row per call (role, model, strategy, tokens, latency, hashes).

---

## 11. Rounds, termination, metrics

`backend/app/core/rounds.py`, `core/goals.py`, `core/metrics/`

A **round** is one context-assembly + action-generation + write-back iteration
(one agent turn inside a `task_run`), recorded per turn. All agent turns —
including `REASON` — count toward the 40-round budget (documented choice, see
LIMITATIONS §c).

| Termination | Condition |
| --- | --- |
| `GOAL` | `FLAG` fact matches `goal_anchor_fact.attributes["flag_regex"]` via `re.fullmatch(..., re.IGNORECASE)`, or a structured matcher predicate returns true |
| `UNREACHABLE` | goal structurally unreachable (no path to goal anchor) |
| `STALLED` | no new fact for `STALL_TICKS` (15) **and** no `CREATED` intents; one final Reason wake if `facts_since_reason > 0` |
| `BUDGET` | `rounds_used > MAX_ROUNDS` (40) |
| `CANCELLED` | operator cancel |

Drift metrics: `repeat_ratio`, `goal_alignment_mean`, `goal_drift_events`,
`pruned_stage4`, `scope_violations`, `backtrack_count`, `stall_ticks`.

Statistics: `scipy.stats.fisher_exact([[a,b],[c,d]], alternative="greater")` per
arm pair.

---

## 12. Security layers

`backend/app/core/security/`

1. **Scope guard** — every tool invocation's target is checked against
   `scope.allowed_hosts` / `allowed_cidrs` / `allowed_ports` and
   `scope.denied_actions`. Violations raise `ScopeViolation` and are counted in
   drift metrics.
2. **Sandbox isolation** — per-project bridge network, read-only rootfs,
   `tmpfs /tmp 256m` + `/work 512m`, memory/CPU/PID limits, `cap_drop=ALL`,
   `no-new-privileges`, labels for reaping.
3. **Prompt-injection defence** — all tool output is wrapped in
   `<untrusted_tool_output>` with a treat-as-data directive; instruction-like
   patterns are logged and the offending text is never placed in an
   instruction position.
4. **Redaction** — secrets are redacted before logging / WS emission.

---

## 13. Parameter reference

| Name | Default | Paper-stated? |
| --- | --- | --- |
| `MAX_ROUNDS` | 40 | ✅ |
| `MAIN_TIMEOUT_S` | 180 | ✅ |
| `DEGRADATION_TIMEOUT_S` | 90 | ✅ |
| `BRIDGE_THETA` | 0.3 | ✅ |
| `PROJECT_LEASE_S` | 600 | default |
| `INTENT_LEASE_S` | 360 | default |
| heartbeat interval | 20 | default |
| `TICK_SECONDS` | 2 | default |
| `SUBGRAPH_TOP_K` | 10 | default |
| `JACCARD_THRESHOLD` | 0.5 | default |
| cross-link cosine τ | 0.75 | default |
| `CONFIDENCE_THRESHOLD` | 0.6 | default |
| α, β, γ (S4) | 0.5 / 0.3 / 0.2 | default |
| `GOAL_ALIGN_THRESHOLD` | 0.55 | default |
| S2 bounds | ≤12 query nodes / ≤60 pattern nodes / 5 s | default |
| `STALL_TICKS` | 15 | default |
| extractor temperature | 0 | ✅ |
| extractor retries | ≤3 | ✅ |
| embedding dim | 2560 | ✅ |
