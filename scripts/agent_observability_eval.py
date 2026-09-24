#!/usr/bin/env python
# coding: utf-8

# # Agent observability → evaluation
# 
# The [escalation agent notebook](escalation_agent_langgraph_with_langfuse_observability.ipynb) attaches Langfuse to a LangGraph incident-response agent. That gives you a trace viewer: useful when you already know a run went wrong and want to see why.
# 
# This notebook turns the same kind of trace into something you can **measure and regress on**:
# 
# 1. **Per-node spans**: latency, tokens, schema-validation retries, and every tool call with its arguments, success, and a relevance score.
# 2. **An eval suite** over a labeled incident set: severity accuracy, action accuracy, RCA groundedness.
# 3. **One intentionally broken run**, diagnosed from the trace. A tool-layer bug sends every evidence call to the wrong service, and the verifier's check is too weak to notice. The graph finishes, `verdict` says `ok`, and the RCA goes to the approval gate looking like any other. Only a check that reads the trace instead of trusting the verdict catches it.
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

print("Mode:", "OFFLINE (rule-based stand-ins; accuracy is perfect by construction)" if ae.OFFLINE
      else f"LIVE ({ae.settings.llm_provider}: {ae.settings.resolved_model})")
print("Langfuse:", "configured, node spans forwarded" if ae.settings.langfuse_configured
      else "not configured, local trace only")


# ## The graph
# 
# `triage → investigate ⇄ verify → write_rca → human_gate → execute`, with SEV3 alerts auto-closed at triage. `investigate` is a tool-calling loop over four mock tools (`search_logs`, `get_recent_deploys`, `get_metrics`, `check_runbook`) whose results always name the service they were asked about, as real log lines and metric labels do. `human_gate` is a LangGraph `interrupt`; the eval harness auto-approves it so runs are reproducible.

# ## One healthy run, fully traced

# In[2]:


incidents = ae.load_incidents()
state, trace = ae.run_once(incidents[0]["alert"], thread_id="narrated-healthy")
ae.score_in_langfuse(ae.evaluate_run(incidents[0], state, trace), trace)
print("alert:  ", incidents[0]["alert"])
print("triage: ", state["parsed"])
print("verdict:", state["verdict"], "| outcome:", state["outcome"])
print("root cause:", state["report"]["root_cause"])
print()
trace.print_table()


# ## Eval suite
# 
# Groundedness here is independent of the graph's own `verify` node. It reads the trace: an incident is **grounded** when every successful evidence tool call targeted the alert's service (`on_target`). An incident auto-closed at triage has no evidence, and counts as grounded only if it really was low severity.
# 
# `names_svc` records whether the final draft names the service. It is deliberately **not** scored; the broken run below shows why.

# In[3]:


results = ae.run_eval_suite(incidents)
print(f"{'incident':9s} {'pred':5s} {'sev_ok':7s} {'act_ok':7s} {'grounded':9s} {'names_svc':10s} {'retries':8s} {'tools':6s} {'tokens':7s} {'latency_ms':>10s}")
for r in results:
    print(f"{r.incident_id:9s} {r.predicted_severity:5s} {str(r.severity_correct):7s} {str(r.action_correct):7s} "
          f"{str(r.grounded):9s} {str(r.mentions_service):10s} {r.retries:<8d} {r.tool_call_count:<6d} {r.tokens:<7d} {r.latency_ms:10.0f}")
print()
healthy_summary = ae.summarize(results)
print(json.dumps(healthy_summary, indent=2))
for r, inc in zip(results, incidents):
    if not r.severity_correct:
        print(f"\nseverity miss {r.incident_id}: expected {inc['expected_severity']}, got {r.predicted_severity}: {inc['alert']}")


# ## The broken run
# 
# Same incident as the healthy run above, with `broken=True`:
# 
# - **Tool layer:** every evidence call is rewritten to `service="unrelated-service"`, as if a stale service name leaked in from a previous incident. The model asked for the right thing and got evidence about something else.
# - **Verifier:** the check degrades to "was *some* error found?", not "was it found for *this* service?"
# 
# Both are code bugs, so they behave the same in live and offline mode.

# In[4]:


broken_state, broken_trace = ae.run_once(incidents[0]["alert"], thread_id="narrated-broken", broken=True)
broken_eval = ae.evaluate_run(incidents[0], broken_state, broken_trace)
ae.score_in_langfuse(broken_eval, broken_trace)
print("verdict inside the graph:", broken_state["verdict"])
print("outcome:", broken_state["outcome"])
print("root cause written:", broken_state["report"]["root_cause"])
print()
print("independent check -> grounded:", broken_eval.grounded,
      "| on_target:", broken_eval.tools_on_target,
      "| draft names the service:", broken_eval.mentions_service)
print()
broken_trace.print_table()


# ## Diagnosis from the trace
# 
# Comparing the two traces call by call, rather than eyeballing them:

# In[5]:


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
# With `LANGFUSE_*` set, every run above is also one Langfuse trace: a root `incident:<thread_id>` span holding the LangGraph node tree, each Nemotron call as a generation (model, tokens, latency) inside the node that made it, the `RunTrace` summary with tool-call arguments as metadata, and the eval results as scores (`severity_correct`, `action_correct`, `grounded`, `tools_on_target`, `eval_score`). The broken run is the trace with `grounded = 0` even though its `verify` node said `ok`. Filtering on that score is how you'd find it in production traffic.

# In[6]:


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
# From the live run above (`nemotron-3-super-120b-a12b` doing triage, tool calling, verification and RCA writing):
# 
# - **The graph's own verdict can't be trusted as a quality signal.** In the broken run `verify` said `ok`, the RCA was written, and the approval gate would have been asked to approve a rollback, all on evidence about the wrong service. The run finished without an error. It showed up only in the trace: every tool call's `service` argument was `unrelated-service`, and mean retrieval relevance fell from 0.31 to 0.12.
# - **Check what the agent *did*, not what it *wrote*.** The first version of this harness scored groundedness by whether the RCA draft named the alert's service. It was wrong both ways. In 2 of the 4 investigated incidents above, the draft never names the service (`names_svc = False`) even though every tool call was on target. And the broken run's RCA opens with "The payments-api service has an error rate of 31%…" although every tool it called returned data about `unrelated-service`. The model took the name from its task prompt and the numbers from the wrong service's metrics. Scoring from the trace's tool arguments has neither problem.
# - **Healthy-suite scores:** severity 1.00, action 1.00, grounded 1.00 over 7 incidents, about 4.8K tokens and about 14 s per incident, and no schema-validation retries. A single run on 7 incidents is a smoke test, not a benchmark. In the earlier live run, inc04 (connection pool exhausted, error rate climbing) was triaged SEV1 instead of SEV2, and inc01's action came out `page_oncall` in one narrated run and `rollback_deploy` in another. Severity at the SEV1/SEV2 boundary is where this agent is unstable, so it's the first place to spend repeated trials.
# - **Investigation dominates cost.** In both traced runs the tool-calling loop is the largest node: about 70–80% of tokens and 45–90% of latency. The SEV1/SEV2 incidents it runs on cost 6–11K tokens against about 300 for an incident auto-closed at triage. It is the node to budget and cap (tool-call limits, per-tool timeouts) first.
# 
# **Next:** run each incident k times and report pass rates rather than single outcomes, and grow the incident set past the SEV1/SEV2 boundary.
