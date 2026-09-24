# AI Engineering Notebooks

A hands-on lab covering the retrieval, agent, and observability patterns behind production LLM systems — built to understand each piece end to end rather than treat it as a black box.

Every notebook is self-contained and runnable. Outputs are committed so you can see what each one actually produced without running it yourself.

```
notebooks/
  retrieval/       chunking, embeddings, hybrid search, reranking, full RAG
  agents/          tool calling, LangGraph agents, tracing with Langfuse
  llm-behaviour/   temperature, logprobs, tokenisation, structured output, streaming
  fine-tuning/     dataset construction and validation
  safety/          prompt injection and guardrails
src/               FastAPI service wrapping a LangGraph agent; ai_engineering/config.py (provider/model config)
scripts/           plain .py exports of every notebook (easier to diff than JSON), plus two
                   standalone eval harnesses (rag_evaluation_benchmark, agent_observability_eval)
data/              small sample corpus and training data, plus labeled eval sets for the above
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
| [`rag_evaluation_benchmark`](scripts/rag_evaluation_benchmark.py) | The question the notebooks above don't answer: which retrieval architecture is actually better? BM25, dense, hybrid/RRF, and hybrid+rerank measured on a labeled query set (`data/rag_eval_queries.json`) with Recall@5, MRR, nDCG@5, p50/p95 latency, and estimated token cost — plus a lexical-vs-paraphrase breakdown, since a query set skewed toward exact keyword matches flatters BM25. See [Running the eval scripts](#running-the-eval-scripts) below. |

## Agents & Orchestration

Multi-step agents on LangGraph, using incident response and support triage as the problem domain.

| Notebook | What it covers |
|---|---|
| [`tool_calling`](notebooks/agents/tool_calling.ipynb) | Tool schemas, the call loop, handling malformed tool arguments |
| [`incident_triage_langgraph`](notebooks/agents/incident_triage_langgraph.ipynb) | Routing an incident by classified severity and type |
| [`incident_response_agent`](notebooks/agents/incident_response_agent.ipynb) | A stateful agent that investigates, forms hypotheses, proposes remediation |
| [`customer_support_router_langgraph`](notebooks/agents/customer_support_router_langgraph.ipynb) | Conditional graph routing across specialist branches |
| [`escalation_agent_langgraph_with_langfuse_observability`](notebooks/agents/escalation_agent_langgraph_with_langfuse_observability.ipynb) | An escalation agent instrumented with Langfuse — traces, spans, per-step cost |
| [`agent_observability_eval`](scripts/agent_observability_eval.py) | Turns that tracing into a number you can track: per-node latency/tokens/retries/tool success, run across a labeled incident dataset (`data/incident_eval_set.json`) for severity/action accuracy and RCA groundedness, plus one **intentionally broken run** — the investigator is fed the wrong service's evidence and the verifier's check is too weak to catch it — showing how an independent groundedness check catches what the graph's own `verdict: ok` didn't. See [Running the eval scripts](#running-the-eval-scripts) below. |

**On observability:** the escalation agent is the piece worth reading first. It traces every node in the graph, so you can see which step burned the latency and which one produced the wrong answer. Debugging a multi-step agent without that is guesswork. `agent_observability_eval.py` is what that tracing is *for*: a regression you can catch automatically instead of a trace you have to remember to go look at.

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
| [`src/fastapi_serve.py`](src/fastapi_serve.py) | A LangGraph agent served over FastAPI, with checkpointing and human-in-the-loop interrupts |

---

## Running these

All models are served from [NVIDIA NIM](https://build.nvidia.com)'s OpenAI-compatible endpoint (`nvidia/nemotron-3-super-120b-a12b` for chat, `nvidia/nemotron-3-embed-1b` for embeddings), so the only key you need for most of the repo is `NVIDIA_API_KEY`. The Langfuse keys are needed only for the escalation agent.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # then fill in NVIDIA_API_KEY
jupyter lab
```

Nemotron reasons out loud by default, so every chat call passes `extra_body=NO_THINK` (`chat_template_kwargs.enable_thinking=False`) to get direct answers. Without it, short-answer calls such as YES/NO judges or single-token logprobs come back as the start of the model's reasoning.

Notebooks, scripts, and the service load `.env` from the repo root automatically (via `python-dotenv`); variables already exported in your shell take precedence. No keys are committed, and `.env` is gitignored.

Scripts can be run from any directory, e.g. `python scripts/full_rag_pipeline.py`.

## Configuration

`src/ai_engineering/config.py` is the single place that resolves provider, model, and
credentials — `LLM_PROVIDER`, `LLM_BASE_URL`, `LLM_MODEL`, `EMBEDDING_MODEL`, and the
per-provider API keys, all read from `.env` (see `.env.example`) via `pydantic-settings`.
Newer scripts (`rag_evaluation_benchmark.py`, `agent_observability_eval.py`,
`src/fastapi_serve.py`) import it instead of hardcoding a base URL and reading
`os.environ` directly:

```python
from ai_engineering.config import NO_THINK, get_settings, make_chat_client

settings = get_settings()
client = make_chat_client(settings)   # OpenAI-compatible client for the configured provider
client.chat.completions.create(model=settings.resolved_model, extra_body=NO_THINK, ...)
```

Switching from NVIDIA NIM to OpenAI, or pointing at a different model, is then a `.env`
edit (`LLM_PROVIDER=openai`, `OPENAI_API_KEY=...`) rather than a code change. The older
notebooks and their `scripts/*.py` exports still read `NVIDIA_API_KEY` directly and are
unaffected — this hasn't been backported across all of them, since that would mean
re-running and re-committing outputs for every notebook, not just editing imports.

## Running the eval scripts

`rag_evaluation_benchmark.py` and `agent_observability_eval.py` are evaluation harnesses,
not thin demos — they compare retrieval architectures and score agent runs against
labeled data. Both run in two modes, detected automatically from whether an API key is
configured for `LLM_PROVIDER`:

* **No key configured → OFFLINE mode.** Dense retrieval uses a deterministic hashed
  bag-of-words embedder instead of a real embedding model, and reranking / triage /
  verification use rule-based stand-ins instead of an LLM. This is what produced the
  numbers currently committed for these two scripts — genuine output from a genuine run,
  clearly not evidence about a real embedding model's or Nemotron's quality. It exists so
  the eval logic itself (Recall@K/MRR/nDCG, groundedness checking, the broken-run
  diagnosis) is checkable with `pytest -q tests` and runnable with zero setup.
* **Key configured → LIVE mode**, using whatever `LLM_PROVIDER`/`LLM_MODEL`/
  `EMBEDDING_MODEL` you set. Same code path, real numbers — and different numbers again if
  you switch provider or model, which is the point of the config layer above.

```bash
python scripts/rag_evaluation_benchmark.py
python scripts/agent_observability_eval.py
```

To run the FastAPI service (from the repo root):

```bash
uvicorn src.fastapi_serve:app_fastapi --reload
```

`app_fastapi` is the FastAPI instance; `app` in the same module is the compiled LangGraph.

To run the offline smoke tests (no API keys or LLM calls; the same suite runs in CI):

```bash
pip install pytest
pytest -q tests
```

## Scope

These are reference implementations and experiments, not a production system. They exist to work through each pattern in isolation, because the tradeoffs are easier to see when one thing changes at a time.

## License

MIT — see [LICENSE](LICENSE).
