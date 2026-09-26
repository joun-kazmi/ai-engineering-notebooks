# AI Engineering Notebooks

A hands-on lab covering the retrieval, agent, and observability patterns behind production LLM systems — built to understand each piece end to end rather than treat it as a black box.

Every notebook runs from the repo checkout, and outputs are committed so you can see what each one actually produced without running it yourself. Most are standalone. The eval and reliability notebooks keep their logic in `src/ai_engineering/` (so it can be unit tested) and their labeled sets in `data/`.

```
notebooks/
  retrieval/       chunking, embeddings, hybrid search, reranking, full RAG
  agents/          tool calling, LangGraph agents, tracing with Langfuse
  llm-behaviour/   temperature, logprobs, tokenisation, structured output, streaming
  fine-tuning/     dataset construction and validation
  safety/          prompt injection and guardrails
src/               FastAPI service wrapping a LangGraph agent; ai_engineering/ (provider config, eval harness, hardened tool runtime)
scripts/           plain .py exports of every notebook (easier to diff than JSON)
data/              small sample corpus, training data, and labeled eval sets
```

---

## Retrieval & RAG

A retrieval pipeline is four independent decisions. These work through each one separately, so the tradeoffs are visible.

| Notebook | What it covers |
|---|---|
| [`chunking_strategies`](notebooks/retrieval/chunking_strategies.ipynb) | Fixed-size, recursive, and semantic chunking, and what each does to retrieval quality |
| [`cosine_similarity`](notebooks/retrieval/cosine_similarity.ipynb) | Similarity from first principles — what the distance metric actually measures |
| [`embedding_vs_chroma`](notebooks/retrieval/embedding_vs_chroma.ipynb) | Raw embeddings versus a vector store, and when the store earns its complexity |
| [`chroma_db_example`](notebooks/retrieval/chroma_db_example.ipynb) | Chroma collections, metadata filtering, persistence |
| [`retrieval_strategies`](notebooks/retrieval/retrieval_strategies.ipynb) | Dense, sparse, and hybrid retrieval compared on one corpus |
| [`rag_pipeline_with_hybrid_search`](notebooks/retrieval/rag_pipeline_with_hybrid_search.ipynb) | BM25 and dense retrieval fused into a single ranked result set |
| [`llm_based_reranking`](notebooks/retrieval/llm_based_reranking.ipynb) | Reranking the shortlist — trading latency for precision at the top |
| [`full_rag_pipeline`](notebooks/retrieval/full_rag_pipeline.ipynb) | The whole path assembled: ingest, chunk, embed, retrieve, rerank, generate |
| [`rag_evaluation_benchmark`](notebooks/retrieval/rag_evaluation_benchmark.ipynb) | The question the notebooks above don't answer: which pipeline is actually better, and at what cost? BM25, dense, hybrid/RRF, and hybrid+LLM-rerank on a labeled set with distractor passages and paraphrase queries — Recall@5, MRR, nDCG@5, p50/p95 latency, query-time embedding and LLM tokens, and answer faithfulness. See [Results](#results). |

## Agents & Orchestration

Multi-step agents on LangGraph, using incident response and support triage as the problem domain.

| Notebook | What it covers |
|---|---|
| [`tool_calling`](notebooks/agents/tool_calling.ipynb) | Tool schemas, the call loop, handling malformed tool arguments |
| [`incident_triage_langgraph`](notebooks/agents/incident_triage_langgraph.ipynb) | Routing an incident by classified severity and type |
| [`incident_response_agent`](notebooks/agents/incident_response_agent.ipynb) | A stateful agent that investigates, forms hypotheses, proposes remediation |
| [`customer_support_router_langgraph`](notebooks/agents/customer_support_router_langgraph.ipynb) | Conditional graph routing across specialist branches |
| [`escalation_agent_langgraph_with_langfuse_observability`](notebooks/agents/escalation_agent_langgraph_with_langfuse_observability.ipynb) | An escalation agent instrumented with Langfuse — traces, spans, per-step cost |
| [`agent_observability_eval`](notebooks/agents/agent_observability_eval.ipynb) | Turns tracing into numbers you can regress on: model-node latency/tokens/validation retries and every tool call's arguments, and an eval suite over incidents whose correct action (rollback, page, restart, monitor, close) must be derived from per-incident tool evidence. Scored separately for severity, action, tool targeting, and RCA groundedness (figure check + claim-by-claim judge). Includes one **intentionally broken run**, where evidence tools silently queried the wrong service and the verifier still said `ok`, diagnosed from the trace and scored in Langfuse. See [Results](#results). |
| [`agent_reliability_hardening`](notebooks/agents/agent_reliability_hardening.ipynb) | The same agent, hardened for when things around it go wrong. Tools have Pydantic input and output contracts, and their schemas are generated from the input model. Every tool call gets a per-tool timeout and bounded retries with exponential backoff and jitter, using one retryable-vs-not classification. Investigation is read-only and the remediation actions are real write tools. Approval is bound to the exact arguments and idempotency key of one write. Run budgets cover LLM calls, tool calls, tokens, cost, time and graph steps, and exhausting one escalates instead of crashing. Every call, including refused ones, gets an audit record. Faults are injected deterministically: 503s, hangs, malformed output, a prompt-injected log line, a runaway loop, and a write that commits but loses its response. See [Results](#results). |

**On observability:** the escalation agent is the piece worth reading first. It traces every node in the graph, so you can see which step burned the latency and which one produced the wrong answer. Debugging a multi-step agent without that is guesswork. [`agent_observability_eval`](notebooks/agents/agent_observability_eval.ipynb) is what that tracing is *for*: a regression you can catch automatically instead of a trace you have to remember to go look at.

## LLM Behaviour

| Notebook | What it covers |
|---|---|
| [`temperature_llm`](notebooks/llm-behaviour/temperature_llm.ipynb) | Temperature and its effect on the output distribution |
| [`probability_math_llm`](notebooks/llm-behaviour/probability_math_llm.ipynb) | Token probabilities, logprobs, what sampling is actually doing |
| [`token_count`](notebooks/llm-behaviour/token_count.ipynb) | Tokenisation with tiktoken — the basis of cost and context budgeting |
| [`get_structured_output_from_llm`](notebooks/llm-behaviour/get_structured_output_from_llm.ipynb) | Schema-constrained output with Pydantic, and validation failure handling |
| [`streaming_response`](notebooks/llm-behaviour/streaming_response.ipynb) | Token streaming |
| [`streaming_response_over_llm_with_tool_call`](notebooks/llm-behaviour/streaming_response_over_llm_with_tool_call.ipynb) | Streaming while a tool call is in flight — the awkward case |

## Fine-tuning

| Notebook | What it covers |
|---|---|
| [`create_jsonl_data_for_fine_tuning`](notebooks/fine-tuning/create_jsonl_data_for_fine_tuning.ipynb) | Building and validating a JSONL training set — `data/train.jsonl`, `data/val.jsonl` |

## Safety

| Notebook | What it covers |
|---|---|
| [`ai_security_pipeline`](notebooks/safety/ai_security_pipeline.ipynb) | Prompt injection and input/output guardrail checks |

## Serving

| File | What it covers |
|---|---|
| [`src/fastapi_serve.py`](src/fastapi_serve.py) | The hardened escalation agent served over FastAPI. `/alert` returns the proposal waiting at the approval gate. `/approve` must name that proposal's `args_hash`, and a repeated submit replays the stored result instead of acting twice. `/runs/{id}` returns the budget and the audit log. Runs are stored in SQLite: a pending approval survives a restart, an approval is claimed atomically across processes, and a run whose process crashed can be recovered without repeating a write it already made. |

---

## Results

Live runs on NVIDIA NIM (`nemotron-3-super-120b-a12b`, `nemotron-3-embed-1b`); full outputs are committed in the notebooks.

**Retrieval** — [`rag_evaluation_benchmark`](notebooks/retrieval/rag_evaluation_benchmark.ipynb), 30 passages (8 of them distractors), 35 queries (16 paraphrases):

| Pipeline | Recall@5 | MRR | nDCG@5 | p50 latency | Tokens / query (embed + LLM) | Recall@5 on paraphrases |
|---|---|---|---|---|---|---|
| BM25 | 0.829 | 0.762 | 0.763 | 1 ms | 0 | 0.719 |
| Dense | **1.000** | 0.971 | 0.973 | 311 ms | 16 | **1.000** |
| Hybrid (RRF) | 0.914 | 0.887 | 0.877 | 234 ms | 16 | 0.906 |
| Hybrid + LLM rerank | **1.000** | **1.000** | **1.000** | 3.2 s | 16 + 599 | **1.000** |

- Fusing BM25 into a strong embedder *hurt*: −0.086 Recall@5 against dense alone, because keyword-matching distractors took the top slots.
- The LLM reranker ranked perfectly, but for +0.029 MRR over dense alone it cost about 38× the query-time tokens and about 10× the p50 latency. On this corpus, dense-only is the better tradeoff. All 35 reranker replies passed strict validation (every candidate scored once, scores 0–10).
- 32/34 judged answers were faithful (88/92 claims supported). Faithfulness is computed from the judge's claim list, not its summary flag. Both failures added outside knowledge to correctly retrieved context.

**Agent** — [`agent_observability_eval`](notebooks/agents/agent_observability_eval.ipynb), 8 incidents whose correct action must be derived from per-incident tool evidence against a runbook:

| Metric | Result |
|---|---|
| Action accuracy (rollback / page / restart / monitor / close) | 8/8 |
| Severity accuracy | 7/8: the one SEV2 → SEV1 miss repeats across every live run |
| Tool targeting (every evidence call queried the alert's service) | 6/6 investigated |
| RCA groundedness (figure check + claim-by-claim judge) | 6/6, 44/44 claims |
| Crashed runs / transient provider errors retried | 0 / 50 |

- In the broken run (evidence tools silently pointed at the wrong service, weakened verifier), the model *wrote in its RCA* that the tools had returned the wrong service's data, and still recommended a rollback, which reached the approval gate. `tool_target_ok = 0` flagged it from the trace. Noticing a problem in prose isn't a control; a trace-level check like this belongs in front of the approval gate.
- With Langfuse configured, each incident is one trace: the node tree, every Nemotron call as a generation with token counts, the RCA judge's calls under an `rca-judge` span, and the eval results as scores.

**Agent reliability** — [`agent_reliability_hardening`](notebooks/agents/agent_reliability_hardening.ipynb), the same 8 incidents with faults injected into every run. `get_metrics` returns a 503 once, `get_recent_deploys` hangs past its timeout once, and every write commits and then loses its response once:

| Metric | Result |
|---|---|
| Action accuracy | 7/8; the miss did nothing wrong (see below) |
| Runs where every invariant held (each executed write matches the proposal and approval given at the gate, checked independently of the executor; no duplicate side effects; no write outside `execute`; within every hard budget cap; budget hits escalated) | 8/8 |
| Side effects / approved writes | 5 / 5: every write timed out once and was `deduplicated` on retry |
| Tool calls that failed on the first attempt and recovered | 20 / 46 |

- **Prompt injection.** A log line told the agent to roll back checkout-api, whose last deploy was 3 days ago. Read-only scoping refused the direct write. The live model still *recommended* the rollback, in both cells that test it, and each time a deterministic runbook precondition, checked before the approval gate, stopped it. In a development run before that check existed, the injected rollback was approved and executed. Scoping stops the model writing; it doesn't stop it proposing a bad write.
- **The miss.** inc07 was triaged SEV2 instead of SEV3 (the same in several runs), investigated rather than closed, and ended as `monitor`. That's correct for a service inside its SLO, with no write.
- **Every final action is checked against the evidence before it takes effect.**
  - A rollback needs errors above 10% within 30 minutes of that deploy.
  - A page has to go to the owner of a failing dependency.
  - A restart has to target a replica the logs show as stuck.
  - `monitor` has to show that no earlier runbook rule applies.
  - A SEV3 close at triage has to be confirmed by a metrics read; otherwise the incident is investigated.
- **Budgets.** LLM calls, tool calls, graph steps and time are hard caps. Every retried LLM request counts, with the SDK's retries off, and each attempt has a deadline no longer than the time left. Tokens and cost are soft limits: the call that crosses one completes, further work stops, and the overshoot is reported.
- **What surfaced only in live runs:**
  - A stalled in-flight request.
  - A tool that never recovered burning the whole LLM budget, which is why there's a circuit breaker.
  - Heavy congestion: at 124×429, a short retry policy gave up.
  - Free-text action targets.
  - A restart aimed at the service's own name.
  - A keyword-filtered log search hiding evidence.
  - A triage returning `service="service"`.

  Each now has a fix and a test.

## Running these

All models are served from [NVIDIA NIM](https://build.nvidia.com)'s OpenAI-compatible endpoint (`nvidia/nemotron-3-super-120b-a12b` for chat, `nvidia/nemotron-3-embed-1b` for embeddings), so the only key you need for most of the repo is `NVIDIA_API_KEY`. The Langfuse keys are needed only for the escalation agent.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # then fill in NVIDIA_API_KEY
jupyter lab
```

Nemotron reasons out loud by default, so every chat call to NIM passes `chat_template_kwargs.enable_thinking=False` (`NO_THINK`) to get direct answers. Without it, short-answer calls such as YES/NO judges or single-token logprobs come back as the start of the model's reasoning.

Notebooks, scripts, and the service load `.env` from the repo root automatically (via `python-dotenv`); variables already exported in your shell take precedence. No keys are committed, and `.env` is gitignored.

Scripts can be run from any directory, e.g. `python scripts/full_rag_pipeline.py`.

## Configuration

`src/ai_engineering/config.py` is the single place that resolves provider, model, and
credentials — `LLM_PROVIDER`, `LLM_BASE_URL`, `LLM_MODEL`, `EMBEDDING_MODEL`, and the
per-provider API keys, all read from `.env` (see `.env.example`) via `pydantic-settings`.
The eval notebooks (`rag_evaluation_benchmark`, `agent_observability_eval`) and
`src/fastapi_serve.py` import it instead of hardcoding a base URL and reading
`os.environ` directly:

```python
from ai_engineering.config import get_settings, make_chat_client

settings = get_settings()
client = make_chat_client(settings)   # OpenAI-compatible client for the configured provider
client.chat.completions.create(model=settings.resolved_model,
                               extra_body=settings.adapter.chat_extra_body(), ...)
client.embeddings.create(model=settings.resolved_embedding_model, input=texts,
                         extra_body=settings.adapter.embedding_extra_body("query"))
```

Provider differences beyond URL and model live in `ProviderAdapter`, not at call sites.
NIM's `chat_template_kwargs` (the thinking switch) and its embeddings' `input_type`
(query vs passage) are sent only when `LLM_PROVIDER=nvidia`: OpenAI has neither
parameter and rejects unknown request arguments. Only OpenAI-compatible providers
(`nvidia`, `openai`) are accepted; anything else fails at config load, not halfway
through a run. Switching to OpenAI is a `.env` edit (`LLM_PROVIDER=openai`,
`OPENAI_API_KEY=...`). The request shape per provider is unit tested, but the committed
results are NIM runs only. The older
notebooks and their `scripts/*.py` exports still read `NVIDIA_API_KEY` directly and are
unaffected — this hasn't been backported across all of them, since that would mean
re-running and re-committing outputs for every notebook, not just editing imports.

## Running the eval notebooks

`rag_evaluation_benchmark` and `agent_observability_eval` keep their logic in
`src/ai_engineering/rag_eval.py` and `agent_eval.py` (unit tested offline) and pick a mode
from whether an API key is configured for `LLM_PROVIDER`:

* **Key configured → LIVE**, using `LLM_PROVIDER`/`LLM_MODEL`/`EMBEDDING_MODEL`. The committed
  outputs and the [Results](#results) below are live runs on NVIDIA NIM.
* **No key → OFFLINE.** A hashed bag-of-words embedder, a token-overlap reranker, and
  rule-based agent nodes stand in for the models. This is what `pytest -q tests` exercises;
  its numbers say nothing about model quality (the agent's offline rules were written against
  the eval set, so its offline accuracy is perfect by construction).

Hosted endpoints return transient 503s under load, so both harnesses raise the OpenAI SDK's
retry budget (`make_chat_client(max_retries=...)`) rather than dying on one bad call.

`agent_reliability_hardening` works the same way (logic in `tool_runtime.py`, which is
generic, and `agent_reliability.py`, the hardened agent). Its faults are injected
deterministically, so every failure mode shows up in both modes. Set
`LLM_USD_PER_MTOK_IN`/`_OUT` to enforce a per-run cost budget; without them, cost is
reported as unpriced.

```bash
python scripts/rag_evaluation_benchmark.py
python scripts/agent_observability_eval.py
python scripts/agent_reliability_hardening.py
```

To run the FastAPI service (from the repo root):

```bash
uvicorn src.fastapi_serve:app_fastapi --reload
```

`app_fastapi` is the FastAPI instance. `app` in the same module is the graph's structure, for introspection; each alert gets its own compiled copy, bound to that run's budget, executor and audit log. Without an API key the agent runs offline, with rule-based stand-ins. Its tools read a demo catalog built from `data/incident_eval_set.json` and write to an in-memory simulator.

```bash
curl -s -X POST localhost:8000/alert -H 'content-type: application/json' \
  -d '{"alert_text": "checkout-api p95 latency 6s and timeouts on checkout, error rate climbing"}'
# -> {"status": "awaiting_approval", "thread_id": "...", "proposal": {"tool": "page_oncall",
#     "args": {"team": "db-team", ...}, "args_hash": "fd788be9149bbff0", ...}, ...}

curl -s -X POST localhost:8000/approve -H 'content-type: application/json' \
  -d '{"thread_id": "...", "approved": true, "approver": "alice", "args_hash": "fd788be9149bbff0"}'
```

Runs are durable. LangGraph's `SqliteSaver` stores the graph checkpoints. A run store in the same file (`SERVE_DB_PATH`, default `.data/serve.sqlite3`) keeps each run's status, decision, budget, idempotency ledger, circuit-breaker counts, simulated backend and audit log, and writes each one as it changes. As a result:

- **A pending approval survives a restart.** It resumes from its checkpoint with its budget and audit log intact.
- **An approval executes exactly once,** across threads and processes. It's claimed with a compare-and-set on the run's status. While it runs, `GET /runs/{id}` reads `executing` instead of waiting.
- **A proposal left undecided past `APPROVAL_TTL_S` expires** (default 24 h), and approving it returns `410 Gone`, because its evidence is stale. Settled runs are purged after `RUN_RETENTION_S` (default 7 days).
- **A run whose process died mid-execution becomes `interrupted`** after `RUN_LEASE_S` of silence (default 300 s; a running run writes on every budget charge). `POST /runs/{id}/recover` continues it from its last checkpoint. If the crash came after a write committed, the write is re-sent with the same idempotency key and the backend answers it as a duplicate, so nothing happens twice.

To run the offline smoke tests (no API keys or LLM calls; the same suite runs in CI):

```bash
pip install pytest
pytest -q tests
```

## Scope

These are reference implementations and experiments, not a production system. They exist to work through each pattern in isolation, because the tradeoffs are easier to see when one thing changes at a time.

## License

MIT — see [LICENSE](LICENSE).
