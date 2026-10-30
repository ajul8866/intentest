# LIMITATIONS.md

This document records **everything that this reproduction does not, cannot, or only approximately
reproduces** from the paper:

> *Staying on the Attack Path: Structured State for Long-Horizon Automated Penetration Testing*
> (Intentest — intent-graph-guided pentest agent built on Cairn).

It is deliberately blunt. Every limitation below is either (a) a resource the authors did not
publish, (b) a parameter the authors did not report, or (c) a deliberate design decision made by
this reproduction. Where a substitute was used, the substitute is named and labelled as such.
**No number produced by this repository is ever presented as a reproduction of the paper's
Table 1 / Table 2 / Table 3 values.**

---

## 1. Non-reproducible artifacts (declared, not faked)

| Artifact | Paper reference | Status here | Consequence |
|---|---|---|---|
| 34 real CTF web challenges (11 easy / 11 medium / 12 hard) | §5.1, Table 1 | **Unavailable (private corpus)** | Replaced by a 6-target local lab benchmark. Absolute success rates are *not comparable* to the paper. |
| Challenge write-ups / solution text used to seed the offline graph | §4.4 (corpus) | **Unavailable** | The offline graph is built from MITRE ATT&CK-style seed documents shipped in-repo. |
| DeepSeek V4 Pro (all agents & baselines) | §5.1 | **API model not available** | Any OpenAI-compatible endpoint may be substituted; the actual model id is recorded on every `experiment_task` row. Cross-model results are not claimed to equal the paper's. |
| Qwen3-Embedding-4B (2560-d) | §4.4, Eq. 2 | **Optional** | Three-tier fallback chain (API → sentence-transformers → deterministic hashing), always logged and surfaced in the retrieval trace. |
| PentestGPT baseline | Table 1/2 | **Not reproducible** | Adapter raises `BaselineUnavailable` — never fabricates a number. |
| PentestAgent baseline | Table 1/2 | **Not reproducible** | Same. |
| Pentest-R1 baseline | Table 1/2/3 | **Not reproducible** | Same. |
| VulnBot baseline | Table 1/2/3 | **Not reproducible** | Same; approximated (clearly labelled) by `PlainTextMemoryBaseline`. |
| Claude Code baseline | Table 1/2 | **Not reproducible** | Same. |
| Exact prompt text for executor / planner / extractor / stage-5 / bridge blocks | §4.2–4.4 (described by purpose only) | **Not published** | Prompts are authored from the role descriptions, kept in `backend/app/prompts/`, and versioned so they are auditable. |
| Exact embedding index / offline graph contents | §4.4 | **Not published** | Built locally by `eval/scripts/build_offline_graph.py`. |
| Hardware (Xeon Silver 4114, 128 GB) | §5.1 | **Not required** | LLM is API-hosted; no GPU needed. Practical minimum 8 vCPU / 16 GB. |

### 1.1 The substitute benchmark is *not* the paper's benchmark

`eval/challenges/manifest.yaml` declares six locally built, intentionally vulnerable targets:

| id | difficulty | vuln types |
|---|---|---|
| `sqli-easy-01` | easy | SQL injection (error-based, no filtering) |
| `upload-easy` | easy | unrestricted file upload |
| `sqli-medium-waf` | medium | SQLi behind a WAF (keyword blacklist, union blocked) |
| `ssrf-medium` | medium | SSRF to an internal metadata service |
| `sqli-hard-sourceleak` | hard | SQLi + source leak + blind exfiltration |
| `deserialize-hard` | hard | PHP object injection / deserialization |

The resulting difficulty split (2 easy / 2 medium / 2 hard) is **not** the paper's 11/11/12 split.
Therefore:

* Only **directional** claims are made on this benchmark (Intentest > noIntent on success rate;
  ≥25 % fewer rounds on medium/hard successes; Intentest > plain-text-memory baseline overall).
* Per-difficulty percentages from this repository **must not** be compared to 100.0 / 90.9 / 75.0 /
  88.2, nor to any baseline row of Table 1.
* With n = 2 per difficulty tier, per-tier Fisher tests have essentially no power. Only the pooled
  (all-difficulty) comparison is statistically meaningful, and even then the sample is small.

---

## 2. Parameters the paper does not publish

The paper states only: the **40-round** per-task budget, the **180 s / 90 s** two-phase timeouts,
the bridge threshold **θ = 0.3**, the extractor temperature **0** with **≤ 3** retries, and the
**2560-d** embedding dimension. Every other numeric constant in this repository is a *chosen
default*, configurable via environment variables / project config, and is listed here so that no
reader mistakes it for a paper value.

| Parameter | Env var | Default | Basis |
|---|---|---|---|
| Subgraph top-k (S1) | `SUBGRAPH_TOP_K` | 10 | **default** |
| Jaccard fallback threshold (S3) | `JACCARD_THRESHOLD` | 0.5 | **default** |
| Cross-link cosine threshold (merge) | `CROSSLINK_COSINE` | 0.75 | **default** |
| Offline-graph confidence threshold | `CONFIDENCE_THRESHOLD` | 0.6 | **default** |
| S4 weights α, β, γ | `ALIGN_ALPHA/BETA/GAMMA` | 0.5 / 0.3 / 0.2 | **default** |
| S4 goal-alignment threshold | `GOAL_ALIGN_THRESHOLD` | 0.55 | **default** |
| S2 query-node bound | `SUBGRAPH_MAX_QUERY_NODES` | 12 | **default** |
| S2 pattern-node bound | `SUBGRAPH_MAX_PATTERN_NODES` | 60 | **default** |
| S2 per-candidate timeout | `SUBGRAPH_TIMEOUT_S` | 5 | **default** |
| Scheduler tick period Δt | `TICK_SECONDS` | 2 | **default** |
| Stall detector horizon | `STALL_TICKS` | 15 | **default** |
| Project lease TTL | `PROJECT_LEASE_S` | 600 | **default** |
| Intent lease TTL | `INTENT_LEASE_S` | 360 | **default** |
| Intent heartbeat period | `INTENT_HEARTBEAT_S` | 20 | **default** |
| Extractor retry count | `EXTRACTOR_MAX_RETRIES` | 3 | paper-stated (≤ 3) |
| Exploration sampling temperature | `EXPLORATION_TEMPERATURE` | 0.2 | **default** (paper says "low") |

Because these values were not published, the ablation deltas reported by this repository are
**sensitive to them**. In particular, `SUBGRAPH_TOP_K`, `JACCARD_THRESHOLD` and
`GOAL_ALIGN_THRESHOLD` jointly control how often the retrieval pipeline falls through to S5, which
in turn changes how much guidance reaches the next Explore.

---

## 3. Semantic / definitional ambiguities and the choices made

### (a) `VERIFY` is not a paper task type
The paper names exactly three task types: **Bootstrap**, **Explore**, **Reason**. This
reproduction adds a fourth, **VERIFY**, as an *optional escalation path* used when a fact is
candidate-but-unverified after inline verification inside the worker. Its priority rank is 2
(below Explore and Bootstrap, above Reason). With `VERIFY_ESCALATION_ENABLED=false` the system
behaves exactly like the paper's three-type scheduler. See `ALGORITHMS.md` §3.

### (b) Round accounting for REASON
The paper does not say whether a Reason turn consumes budget. This reproduction **counts every
agent turn, including REASON turns**, toward the 40-round budget. Rationale: the budget protects
wall-clock/API spend, and Reason calls a full LLM. This is documented and can be changed with
`COUNT_REASON_TOWARD_BUDGET=false`.

### (c) "No DAG" ablation is infeasible
The paper states a fully DAG-free ablation is impossible (the scheduler, leases and fact
provenance all presuppose the graph). This reproduction therefore implements the paper's
**three-line triangulation** instead:

1. `INTENT_GUIDANCE_ENABLED=false` (**noIntent**): the DAG, FSM, leases, degradation and
   verification all remain intact; only *retrieval + bridge* are disabled. Asserted by
   `e2e/test_ablation_flag.py` (no `intent_pattern_id`, no `injected_context`).
2. **Plain-text memory baseline** (`PlainTextMemoryBaseline`): a single shared append-only text
   buffer in the LLM context, approximating VulnBot's shared plain-text memory. **Labelled
   approximate**; it is not VulnBot and must not be reported as VulnBot.
3. **F1–F4 failure taxonomy** from manual inspection of failed episodes: F1 scope/authorization
   dead-end, F2 unrecoverable tool error, F3 hallucinated fact (verifier-rejected), F4 budget
   exhaustion without progress.

### (d) `intent_source` "write-once"
The paper says predecessor edges are fixed when the intent is created. The reproduction enforces
this at the database level with a `BEFORE UPDATE OR DELETE` trigger on `intent_source`. It does
**not** model the case where an intent must be re-created because a predecessor was superseded — in
that case a new intent (new `dedupe_key`) is created instead of mutating the old one.

### (e) Scope semantics
`scope` is a namespace of allowed hosts / CIDRs / ports plus `denied_actions`. The paper describes
scope enforcement but not its data model. A fact that references an out-of-scope entity is not
rejected on *ingest* (the agent may legitimately have observed a redirect target), but any intent
whose `target_entity` resolves out of scope is rejected by `validator.validate_preconditions`
(S4 also prunes it). Deleting a scope that facts reference returns HTTP 409 rather than cascading.

### (f) Goal anchoring
Goal matching uses `re.fullmatch(flag_regex, value, re.IGNORECASE)` against a `FLAG` fact's
`value`, or a structured matcher predicate stored on the goal anchor fact. The paper describes a
"goal anchor fact" but not the matcher. Regex is a reproduction choice; it can produce false
negatives if a target emits a flag with a different prefix.

---

## 4. Architectural deviations from the paper's description

1. **Postgres-only state store.** No Redis. The event bus is Postgres `LISTEN/NOTIFY` on channel
   `intentest_events`, driven by a trigger on the `event` table. This makes replay (`?since=<id>`)
   trivial but caps event throughput at what one Postgres can notify — fine for tens of projects,
   not for thousands.
2. **Scheduler is stateless.** All scheduler state lives in `project_lease` and the intent rows;
   the process holds no in-memory queue. Consequence: horizontal scaling is safe, but each tick
   costs a handful of queries per project, so the practical project count per scheduler instance is
   in the low hundreds at Δt = 2 s.
3. **`retrieval_trace` persistence.** The paper shows a "retrieval trace" UI but does not specify
   storage. Here the full per-stage trace (candidate sets, scores, rejection reasons) is stored as
   JSONB on the intent / retrieval row. Traces are *not* garbage-collected; long runs accumulate.
4. **Embedding fallback.** When no embeddings endpoint is configured, the deterministic hashing
   provider produces vectors that are *lexical*, not semantic. Cosine similarities from this
   provider are therefore only weakly meaningful; the trace records `provider="hash"` so that no
   reader is misled. S1 quality — and hence the whole retrieval funnel — degrades under this
   fallback.
5. **Local-subprocess sandbox mode.** `SANDBOX_MODE=local` exists **for tests only**. It is loudly
   flagged in logs and in `GET /readyz`; it provides **no isolation whatsoever** and must never be
   used against a real target.
6. **Container session preservation during degradation.** The paper's "keep the same container
   session" is implemented by *not* tearing down the container and re-attaching the agent loop to
   the same `container_id` with the frozen `EMERGENCY_TRUNCATION` prompt. Residual stdout/stderr
   is read from the sandbox log files rather than from a live PTY; output produced by a process
   that is still running at the 180 s boundary is only partially visible.
7. **REASON non-degradability.** REASON tasks are explicitly excluded from the degradation path:
   on timeout they fail and release the lease. This follows the paper but means a REASON task that
   times out at 180 s wastes the whole budget slot.

---

## 5. Known correctness limits

* **Acyclicity is structural, not proven at runtime.** `verify_acyclic()` is an *audit*
  (`POST /graph/integrity-check`), not a write barrier. If a bug ever allowed an intent to be
  claimed before its predecessor edges were frozen, a cycle could be written and only detected
  afterwards.
* **CAS claim is exactly-one-winner per intent, not per project.** Two intents may run
  concurrently in one project by design (parallelism `C_n`). Ordering guarantees are therefore
  per-task, not global; "Explore before Bootstrap before Reason" is a *priority* guarantee within a
  tick, not a serialization guarantee across ticks.
* **Reason gate is global-best-effort.** "No unclaimed EXPLORE globally" is evaluated across all
  active projects in the scheduler's view; with multiple scheduler instances there is a race window
  of one tick in which two Reason tasks can be dispatched.
* **Fact dedupe uses non-volatile attributes only.** `VOLATILE_ATTRS` (timestamps, request ids,
  durations, nonces) are stripped before hashing. Two genuinely different observations that differ
  *only* in a volatile attribute collapse into one fact. This is intentional (it suppresses noise)
  but it can hide repeated-behaviour evidence needed by `repeat_ratio`.
* **Verifier LLM cross-check is a second model call, not a proof.** `verification_method ∈
  {deterministic, llm_crosscheck, degradation_log}`; only `deterministic` checks are reproducible
  offline.
* **Stage-2 isomorphism is bounded by design.** Query graphs above 12 nodes, patterns above 60
  nodes, or candidates that take > 5 s are silently skipped (recorded as `timeout` in the trace).
  Deep multi-step chains therefore fall through to S3/S5 more often than an unbounded matcher
  would.
* **`strip_meta` is an allow-list, so new payload fields are dropped by default.** This is the
  safe direction for prompt-injection defence, but adding a new useful field to the bridge requires
  editing the allow-list in `core/intent/bridge.py`.

---

## 6. Statistical limitations

* Baseline rows of Tables 1/2/3 cannot be produced (see §1). `eval/harness/runner.py` will run any
  arm whose adapter is installed and will otherwise raise `BaselineUnavailable` with an explicit
  message — it never writes a placeholder number into the report.
* The paper reports one sample per (challenge, arm). This reproduction fixes a seed per repeat and
  supports `--repeats N`, but the default is 1 for cost reasons; with N = 1 the Fisher test is
  descriptive only.
* `scipy.stats.fisher_exact(..., alternative="greater")` is one-sided, matching the paper. Multiple
  comparisons are **not** corrected (no Holm/BH); treat p-values as indicative.
* Efficiency claims (rounds per task) are means over a handful of episodes; a single 40-round
  failure moves the mean by several rounds.

---

## 7. Operational limitations

* **Docker required.** Without a Docker socket (or DinD) the worker cannot start a sandbox; the
  system reports `SANDBOX_UNAVAILABLE` and refuses to dispatch rather than silently running tools
  on the host.
* **One container per project at a time.** Concurrency is across projects, not within a project's
  sandbox; `max_parallelism C_n` bounds *tasks*, and a task's container is created/destroyed per
  task_run except during degradation.
* **No multi-tenant isolation.** `scope` is a logical namespace, not a security boundary; two
  projects on the same host can reach the same targets if both scopes allow them.
* **Event bus has no backpressure.** A slow WebSocket consumer falls behind; replay via
  `?since=<event_id>` is the recovery mechanism.
* **Long-run storage growth.** `event`, `llm_call` and `retrieval_trace` are append-only and not
  partitioned. A multi-day experiment run will need pruning.
* **No authN/authZ.** The API is unauthenticated by design for local evaluation. See
  `THREAT_MODEL.md` §7 — do not expose it.

---

## 8. What *is* faithfully reproduced (for contrast)

To avoid the impression that everything is approximate, the following mechanisms are implemented
exactly as specified and are covered by tests:

* The Fact–Intent DAG `G = (F, I, E_f→i, E_i→f)` with bipartite, alternating, acyclic structure and
  invariants **I1–I6** (fact immutability, `intent_source` write-once, verified same-scope
  predecessors, `E_i→f ≡ derived_from_intent_id`, structural acyclicity + DFS audit, terminal
  irreversibility).
* The four-transition intent FSM with CAS claim (`rowcount == 1`) and timeout reclamation.
* Algorithm 1's tick order, priority rank `{EXPLORE:0, BOOTSTRAP:1, VERIFY:2, REASON:3}`, the Reason
  wake gate, and the six-stage adaptive load balancer in the specified order.
* Two-phase degradation: **180 s → 90 s**, container preserved, `EMERGENCY_TRUNCATION` injected,
  facts recovered from residual logs with `verification_method="degradation_log"`, REASON excluded,
  teardown guaranteed.
* Algorithm 2's five stages in order (S1 cosine → S2 bounded isomorphism → S3 Jaccard fallback →
  S4 strategic alignment → S5 LLM), with `submit_for_validation()` re-running DAG precondition
  validation on *every* candidate including S5 output.
* Algorithm 3's bridge: activation guard, θ = 0.3, dedupe by `sha256(id‖hash_blocks)`, three
  content blocks, allow-list `strip_meta`, injection into the next Explore.
* The 40-round budget, the five termination reasons, and the drift metrics.
* One-sided Fisher exact tests via `scipy.stats.fisher_exact`.

---

## 9. How to report results from this repository

When quoting numbers produced here, always state:

1. the model id actually used (from `experiment_task.model_id`),
2. the embedding provider actually used (`api` / `sentence-transformers` / `hash`),
3. the lab benchmark (`eval/challenges/manifest.yaml`, 6 targets, 2/2/2 split),
4. the repeat count and seed,
5. that the paper's corpus and baselines were unavailable.

Suggested phrasing: *"On our six-target substitute benchmark with <model> and <embedding provider>,
Intentest solved X/Y tasks; the noIntent ablation solved X'/Y' with Z vs Z' mean rounds. These
numbers are not comparable to the paper's Table 1–3."*
