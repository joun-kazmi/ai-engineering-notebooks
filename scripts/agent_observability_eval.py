#!/usr/bin/env python
# coding: utf-8

# # Agent observability → evaluation
# 
# The [escalation agent notebook](escalation_agent_langgraph_with_langfuse_observability.ipynb) attaches Langfuse to a LangGraph incident-response agent. That gives you a trace viewer: useful when you already know a run went wrong and want to see why.
# 
# This notebook turns the same kind of trace into something you can **measure and regress on**:
# 
# 1. **Model-node spans**: latency, tokens, schema-validation retries, and every tool call with its arguments and result.
# 2. **An eval suite** over labeled incidents whose correct action has to be *derived from the evidence*, scored on four separate questions: right severity? right action? did the tools query the right service? is the RCA supported by what the tools returned?
# 3. **One intentionally broken run**, diagnosed from the trace. A tool-layer bug sends every evidence call to the wrong service, and the verifier's check is too weak to notice. The graph finishes, `verdict` says `ok`, and the RCA goes to the approval gate looking like any other.
# 
# Library code: [`src/ai_engineering/agent_eval.py`](../../src/ai_engineering/agent_eval.py).

# In[1]:


import sys
from pathlib import Path

# Repo root, whether this runs as a notebook (cwd = notebooks/<topic>/) or as the scripts/ export
try:
    ROOT = Path(__file__).resolve().parent.parent
except NameError:
    ROOT = Path.cwd().resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import json
from ai_engineering import agent_eval as ae

print("Mode:", "OFFLINE (rule-based stand-ins apply the runbook; accuracy is perfect by construction)" if ae.OFFLINE
      else f"LIVE ({ae.settings.llm_provider}: {ae.settings.resolved_model})")
print("Langfuse:", "configured: traces, generations and scores are sent" if ae.settings.langfuse_configured
      else "not configured, local trace only")


# ## The graph and the incident set
# 
# `triage → investigate ⇄ verify → write_rca → human_gate → execute`. SEV3 alerts close at triage, and an RCA whose action is `monitor` closes without the approval gate. `human_gate` is a LangGraph `interrupt`; the harness auto-approves it so runs are reproducible.
# 
# `investigate` is a tool-calling loop over five tools (`search_logs`, `get_recent_deploys`, `get_metrics`, `get_dependencies`, `check_runbook`). Each incident has its own fixtures behind those tools, and one shared runbook maps evidence to an action. A recent deploy with a spike means roll back. A failing dependency owned by another team means page. One stuck replica means restart. Degradation still inside SLO means monitor. So the right answer differs per incident and has to be read out of the tool results:

# In[2]:


incidents = ae.load_incidents()
print(ae.RUNBOOK, "\n")
print(f"{'incident':9s} {'service':22s} {'severity':9s} {'action':16s} why")
for inc in incidents:
    print(f"{inc['id']:9s} {inc['service']:22s} {inc['expected_severity']:9s} {inc['expected_action']:16s} {inc['rationale']}")


# ## One healthy run, fully traced

# In[3]:


state, trace = ae.run_once(incidents[0], thread_id="narrated-healthy")
healthy_eval = ae.evaluate_run(incidents[0], state, trace)
ae.score_in_langfuse(healthy_eval, trace)
print("alert:  ", incidents[0]["alert"])
print("triage: ", state["parsed"])
print("verdict:", state["verdict"], "| outcome:", state["outcome"])
print("RCA:", json.dumps(state["report"], indent=2))
print()
trace.print_table()


# ## Eval suite
# 
# Four scores per incident, each answering a different question:
# 
# | Score | Question | How |
# |---|---|---|
# | `sev` | Right severity? | Triage output vs label |
# | `action` | Right action? | RCA `next_action` (or "close" at triage) vs label |
# | `on_target` | Did every evidence tool call query the alert's service? | Tool-call arguments in the trace |
# | `grounded` | Is the RCA supported by what the tools returned? | Every number, version and time in the RCA must appear in the tool results, **and** an LLM judge must mark every claim in the RCA as supported |
# 
# `on_target` and `grounded` are complementary. An agent can query the right service and still invent a root cause, or faithfully summarize evidence that came from the wrong service. Incidents closed at triage have no tool calls and no RCA, so those two scores are `-` (not applicable) rather than a free pass.

# In[4]:


def fmt(x):
    return "-" if x is None else str(x)

results = ae.run_eval_suite(incidents)
print(f"{'incident':9s} {'sev':5s} {'expected':16s} {'predicted':16s} {'action':7s} {'on_target':10s} {'grounded':9s} {'claims':7s} {'tools':6s} {'tokens':7s} {'latency_ms':>10s}")
for r in results:
    claims = f"{r.rca_claims_supported}/{r.rca_claims}" if r.rca_claims else "-"
    print(f"{r.incident_id:9s} {fmt(r.severity_correct):5s} {r.expected_action:16s} {r.predicted_action:16s} "
          f"{fmt(r.action_correct):7s} {fmt(r.tool_target_ok):10s} {fmt(r.rca_grounded):9s} {claims:7s} "
          f"{r.tool_call_count:<6d} {r.tokens:<7d} {r.latency_ms:10.0f}")
print()
print(json.dumps(ae.summarize(results), indent=2))
print("transient provider errors retried so far:", ae.TRANSIENT_ERRORS or "none")
for r in results:
    if r.error:
        print(f"\nCRASHED {r.incident_id}: {r.error}")
        continue
    if not r.severity_correct:
        print(f"\nseverity miss {r.incident_id}: expected {r.expected_severity}, got {r.predicted_severity}")
    if not r.action_correct:
        print(f"action miss {r.incident_id}: expected {r.expected_action}, got {r.predicted_action}")
    if r.unsupported:
        print(f"{r.incident_id}: figures in the RCA not found in any tool result: {r.unsupported}")


# ## The broken run
# 
# Same incident as the healthy run above, with `broken=True`:
# 
# - **Tool layer:** every evidence call is rewritten to `service="unrelated-service"`, as if a stale service name leaked in from a previous incident. The model asked for the right thing and got plausible, error-laden evidence about something else.
# - **Verifier:** the check degrades to "was *some* error found?", not "was it found for *this* service?"
# 
# Both are code bugs, so they behave the same in live and offline mode.

# In[5]:


broken_state, broken_trace = ae.run_once(incidents[0], thread_id="narrated-broken", broken=True)
broken_eval = ae.evaluate_run(incidents[0], broken_state, broken_trace)
ae.score_in_langfuse(broken_eval, broken_trace)
print("verdict inside the graph:", broken_state["verdict"])
print("outcome:", broken_state["outcome"])
print("RCA:", json.dumps(broken_state["report"], indent=2))
print()
print("on_target:", broken_eval.tool_target_ok,
      "| grounded:", broken_eval.rca_grounded,
      f"({broken_eval.rca_claims_supported}/{broken_eval.rca_claims} claims supported)" if broken_eval.rca_claims else "",
      "| RCA names payments-api:", broken_eval.mentions_service)
print()
broken_trace.print_table()


# ## Diagnosis from the trace
# 
# Comparing the two traces call by call, rather than eyeballing them:

# In[6]:


def evidence_targets(t):
    return sorted({tc.args.get("service") for tc in t.tool_calls if tc.success})

def mean_relevance(t):
    calls = [tc for tc in t.tool_calls if tc.success]
    return sum(tc.retrieval_score for tc in calls) / len(calls) if calls else 0.0

expected = ae._extract_service(incidents[0]["alert"])
print(f"alert is about: {expected}")
print(f"healthy run queried: {evidence_targets(trace)}   mean retrieval_score={mean_relevance(trace):.2f}")
print(f"broken run queried:  {evidence_targets(broken_trace)}   mean retrieval_score={mean_relevance(broken_trace):.2f}")
print(f"verify said: healthy={state['verdict']}, broken={broken_state['verdict']}")


# ## In Langfuse
# 
# With `LANGFUSE_*` set, every run above is also one Langfuse trace. Each has a root `incident:<thread_id>` span holding the LangGraph node tree, and every agent LLM call as a generation (model, tokens, latency) inside the node that made it. The `RunTrace` summary with tool-call arguments goes in as metadata. The eval results go in as scores: `severity_correct`, `action_correct`, `tool_target_ok`, `rca_grounded` (skipped where not applicable) and `eval_score`. The broken run is the trace with `tool_target_ok = 0` even though its `verify` node said `ok`. Filtering on that score is how you'd find it in production traffic. The RCA judge's calls are nested in the same trace under an `rca-judge` span, so each trace holds both the agent run and its grading.

# In[7]:


print("transient provider errors retried over the whole notebook:", ae.TRANSIENT_ERRORS or "none")

lf = ae.langfuse_client()
if lf is None:
    print("Langfuse not configured; set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY to send traces and scores.")
else:
    ae.flush_langfuse()
    print(f"sent {len(results) + 2} traces with scores to Langfuse")
    for label, t in (("healthy", trace), ("broken", broken_trace)):
        print(f"  {label:8s} {lf.get_trace_url(trace_id=t.langfuse_trace_id)}")


# ## Findings
# 
# From the live run above (`nemotron-3-super-120b-a12b` doing triage, tool calling, verification, RCA writing and the claim judging):
# 
# - **Actions were derived, not guessed.** 8/8 actions correct across five different answers: two rollbacks, two pages, one restart, one monitor, two triage closes. Each had to be read from that incident's evidence against the runbook. Two cases turn on a single detail: inc02 and inc04 both involve pool saturation, but only inc04 has a deploy in the last 30 minutes. The model got both right, rolling back one and paging on-call for the other.
# - **Severity is the weak spot, and consistently so.** 7/8. inc04 ("connection pool exhausted, error rate climbing after last deploy") was triaged SEV1 instead of SEV2, the same miss in every live run of this notebook. It is a stable disagreement with the severity policy, not noise, and the first place to tighten the triage prompt.
# - **Groundedness: 6/6 RCAs, 44/44 claims supported.** Every figure in every RCA traced back to a tool result, allowing unit conversions (6200 ms written as 6.2 s, 1500 minutes as 25 hours). A figure that doesn't convert, like 620 for 6200, is flagged.
# - **The broken run: the model noticed, and the graph went ahead anyway.** Its RCA says outright that the tooling "returned data for 'unrelated-service' instead of 'payments-api'". It still recommended `rollback_deploy`, from the wrong service's deploy and error rate. The weakened `verify` passed it, and the rollback went to the approval gate. `tool_target_ok = False` caught it from the trace, and the judge flagged 1 of 7 claims. Noticing a problem in prose doesn't stop an agent. A trace-level check like `tool_target_ok` belongs *in front of* the approval gate, not just in an offline eval.
# - **Tool-call robustness showed up in the healthy run.** One call came back with the chat template leaking into the argument (`"<parameter=service>\npayments-api"`). The harness recorded it as a failed call and handed the error back; the model retried correctly, and the run completed.
# - **Cost:** about 5.5K tokens and 4 tool calls per investigated incident, and about 51 s latency (provider time, with pacing and retry backoff excluded). The shared endpoint returned 49×429 and 1×500 over this run, all absorbed by retries; none crashed an incident.
# 
# **Limits.** One run over 8 incidents (6 investigated) is a regression check, not a benchmark. Repeated trials per incident would turn single outcomes into pass rates. The same model acts and judges. The fixtures are synthetic.
