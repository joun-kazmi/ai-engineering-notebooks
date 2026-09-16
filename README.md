# AI Engineering Notebooks

A hands-on lab covering the retrieval, agent, and observability patterns behind production LLM systems — built to understand each piece end to end rather than treat it as a black box.

Each notebook is self-contained and runnable: it sets up the problem, implements the pattern, and keeps the outputs in place so you can see what it actually produced.

---

## Retrieval & RAG

Building a retrieval pipeline properly means making deliberate choices at four separate stages. These work through each one.

| Notebook | What it covers |
|---|---|
| [`chunking_strategies.ipynb`](chunking_strategies.ipynb) | Fixed-size, recursive, and semantic chunking, and what each does to retrieval quality |
| [`embedding_vs_chroma.ipynb`](embedding_vs_chroma.ipynb) | Raw embeddings versus a vector store, and when the store earns its complexity |
| [`cosine_similarity.ipynb`](cosine_similarity.ipynb) | Similarity from first principles — what the distance metric is actually measuring |
| [`chroma_db_example.ipynb`](chroma_db_example.ipynb) | Chroma collections, metadata filtering, and persistence |
| [`retrieval_strategies.ipynb`](retrieval_strategies.ipynb) | Dense, sparse, and hybrid retrieval compared on the same corpus |
| [`rag_pipeline_with_hybrid_search.ipynb`](rag_pipeline_with_hybrid_search.ipynb) | BM25 + dense retrieval fused into one ranked result set |
| [`llm_based_reranking.ipynb`](llm_based_reranking.ipynb) | Cross-encoder style reranking — trading latency for precision at the top of the list |
| [`full_rag_pipeline.ipynb`](full_rag_pipeline.ipynb) | The whole path assembled: ingest, chunk, embed, retrieve, rerank, generate |

## Agents & Orchestration

Multi-step agents built on LangGraph, using incident response and support triage as the problem domain.

| Notebook | What it covers |
|---|---|
| [`tool_calling.ipynb`](tool_calling.ipynb) | Tool schemas, the call loop, and handling malformed tool arguments |
| [`incident_triage_langgraph.ipynb`](incident_triage_langgraph.ipynb) | Routing an incident to the right handler based on classified severity and type |
| [`incident_response_agent.ipynb`](incident_response_agent.ipynb) | A stateful agent that investigates, forms hypotheses, and proposes remediation |
| [`customer_support_router_langgraph.ipynb`](customer_support_router_langgraph.ipynb) | Conditional graph routing across multiple specialist branches |
| [`escalation_agent_langgraph_with_langfuse_observability.ipynb`](escalation_agent_langgraph_with_langfuse_observability.ipynb) | An escalation agent instrumented with Langfuse — traces, spans, and per-step cost |

## Observability & Evaluation

The escalation agent above is the main piece here: it traces every node in the graph, so you can see which step burned the latency and which one produced the wrong answer. Debugging a multi-step agent without this is guesswork.

## LLM Behaviour

| Notebook | What it covers |
|---|---|
| [`temperature_llm.ipynb`](temperature_llm.ipynb) | Temperature and its effect on output distribution |
| [`probability_math_llm.ipynb`](probability_math_llm.ipynb) | Token probabilities, logprobs, and what sampling is actually doing |
| [`token_count.ipynb`](token_count.ipynb) | Tokenisation and counting with tiktoken — the basis of cost and context budgeting |
| [`get_structured_output_from_llm.ipynb`](get_structured_output_from_llm.ipynb) | Schema-constrained output with Pydantic, and validation failure handling |
| [`streaming_response.ipynb`](streaming_response.ipynb) | Token streaming |
| [`streaming_response_over_llm_with_tool_call.ipynb`](streaming_response_over_llm_with_tool_call.ipynb) | Streaming while a tool call is in flight — the awkward case |

## Fine-tuning

| Notebook | What it covers |
|---|---|
| [`create_jsonl_data_for_fine_tuning.ipynb`](create_jsonl_data_for_fine_tuning.ipynb) | Building and validating a JSONL training set (`train.jsonl` / `val.jsonl` included) |

## Serving & Safety

| File | What it covers |
|---|---|
| [`fastapi_serve.py`](fastapi_serve.py) | A LangGraph agent served over FastAPI, with checkpointing and human-in-the-loop interrupts |
| [`ai_security_pipeline.ipynb`](ai_security_pipeline.ipynb) | Prompt injection and input/output guardrail checks |

---

## Running these

```bash
pip install -r requirements.txt
```

Set whichever provider keys the notebook needs:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
# Langfuse notebook only
export LANGFUSE_PUBLIC_KEY=...
export LANGFUSE_SECRET_KEY=...
```

No keys are committed — every notebook reads them from the environment.

`python_scripts/` holds plain `.py` exports of the same notebooks, which are easier to diff and review than notebook JSON.

## Scope

These are reference implementations and experiments, not a production system. They exist to work through each pattern in isolation — the tradeoffs are easier to see when one thing changes at a time.
