#!/usr/bin/env python
# coding: utf-8
"""Observability + evaluation for the escalation/incident-response agent.

The existing escalation-agent notebook wires Langfuse's LangGraph callback
into the graph, which is the right first step but only gives you a trace
viewer — it doesn't turn into a number you can track over time or a way to
catch a regression automatically. This script adds that layer on top:

  * Per-node spans: latency, model/provider, prompt/completion tokens,
    retries, and per-tool-call success/failure + a retrieval-relevance
    score — collected independently of Langfuse (see RunTrace below), and
    additionally forwarded to Langfuse via the same CallbackHandler pattern
    as the existing notebook whenever LANGFUSE_* is configured.
  * An evaluation harness that runs the agent across a small labeled
    incident dataset (data/incident_eval_set.json) and reports
    severity/action accuracy, RCA groundedness, retry rate, and
    human-escalation rate — not just "it ran without crashing".
  * One intentionally broken run: the investigator is fed the wrong
    service's evidence (a stand-in for a tool/context mix-up) and the
    verifier's check is weakened so it doesn't catch it. The independent
    groundedness check in the eval layer does catch it — exactly the kind
    of bug a trace viewer alone won't flag for you, because the verdict
    field says "ok".

Like scripts/rag_evaluation_benchmark.py, this runs two ways:

  * No LLM_PROVIDER API key configured -> OFFLINE mode: triage/verify/RCA
    use deterministic rule-based stand-ins instead of an LLM, so the whole
    graph, and every metric below, runs end to end with no network access.
    This produced the numbers committed in this repo.
  * A real key configured -> the LLM does triage/verify/RCA generation for
    real, same graph, same instrumentation. Expect different numbers.

Langfuse is optional in both modes: without LANGFUSE_PUBLIC_KEY/SECRET_KEY
this falls back to printing the locally-collected RunTrace instead of
sending spans anywhere.
"""
import json
import operator
import re
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field, ValidationError

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ai_engineering.config import NO_THINK, get_settings, make_chat_client

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
settings = get_settings()
OFFLINE = not settings.has_llm_credentials


def _estimate_tokens(text: str) -> int:
    """chars/4 heuristic — used for the offline mode's token figures, and as
    a fallback if a live response doesn't carry usage data."""
    return max(1, len(text) // 4)


# ========== OBSERVABILITY: per-node/per-tool trace, independent of Langfuse ==========

@dataclass
class ToolCallRecord:
    name: str
    success: bool
    latency_ms: float
    retrieval_score: float = 0.0  # heuristic relevance of the tool result to the alert, 0-1


@dataclass
class NodeSpan:
    node: str
    latency_ms: float = 0.0
    model: str = ""
    provider: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    retries: int = 0
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    error: str | None = None


@dataclass
class RunTrace:
    thread_id: str
    spans: list[NodeSpan] = field(default_factory=list)

    @property
    def total_latency_ms(self) -> float:
        return sum(s.latency_ms for s in self.spans)

    @property
    def total_tokens(self) -> int:
        return sum(s.prompt_tokens + s.completion_tokens for s in self.spans)

    @property
    def total_retries(self) -> int:
        return sum(s.retries for s in self.spans)

    @property
    def tool_calls(self) -> list[ToolCallRecord]:
        return [tc for s in self.spans for tc in s.tool_calls]

    def print_table(self) -> None:
        print(f"  trace {self.thread_id}: {self.total_latency_ms:.1f}ms total, "
              f"{self.total_tokens} tokens (est.), {self.total_retries} retries")
        for s in self.spans:
            tool_summary = ""
            if s.tool_calls:
                ok = sum(tc.success for tc in s.tool_calls)
                avg_score = sum(tc.retrieval_score for tc in s.tool_calls) / len(s.tool_calls)
                tool_summary = f", {ok}/{len(s.tool_calls)} tools ok, avg retrieval_score={avg_score:.2f}"
            err = f", ERROR={s.error}" if s.error else ""
            print(f"    [{s.node:14s}] {s.latency_ms:7.2f}ms  "
                  f"tokens={s.prompt_tokens + s.completion_tokens:4d}  retries={s.retries}{tool_summary}{err}")


@contextmanager
def traced(trace: RunTrace, node: str):
    span = NodeSpan(node=node, model=settings.resolved_model, provider=settings.llm_provider)
    start = time.perf_counter()
    try:
        yield span
    except Exception as e:
        span.error = str(e)
        raise
    finally:
        span.latency_ms = (time.perf_counter() - start) * 1000
        trace.spans.append(span)


def _langfuse_handler():
    """Real Langfuse CallbackHandler if configured, else None — same pattern
    as notebooks/agents/escalation_agent_langgraph_with_langfuse_observability."""
    if not settings.langfuse_configured:
        return None
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler

    Langfuse(
        public_key=settings.langfuse_public_key.get_secret_value(),
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        host=settings.langfuse_base_url,
    )
    return CallbackHandler()


def _relevance(alert: str, text: str) -> float:
    """Heuristic token-overlap relevance score, 0-1 — a stand-in for an LLM
    judge, same tradeoff as rag_evaluation_benchmark.py's offline reranker."""
    a = set(re.findall(r"\w+", alert.lower()))
    t = set(re.findall(r"\w+", text.lower()))
    if not a:
        return 0.0
    return len(a & t) / len(a)


# ========== TOOLS (same mock data regardless of mode — no real infra) ==========

def _search_logs(service, keyword="ERROR"):
    return {"matches": [f"10:42:03 ERROR timeout connecting to postgres-primary ({service})"]}


def _get_recent_deploys(service):
    return {"deploys": [{"version": "v2.14.3", "time": "10:35", "author": "dana@corp"}]}


def _get_metrics(service):
    return {"p95_latency_ms": 4200, "error_rate": 0.31, "db_conn_active": 100}


def _check_runbook(service):
    return "1. Check DB pool. 2. Rollback if error_rate > 10%. 3. Page dana."


TOOL_REGISTRY = {
    "search_logs": _search_logs,
    "get_recent_deploys": _get_recent_deploys,
    "get_metrics": _get_metrics,
    "check_runbook": _check_runbook,
}
RAW_TOOLS = [
    {"type": "function", "function": {"name": "search_logs", "description": "Search logs",
     "parameters": {"type": "object", "properties": {"service": {"type": "string"}, "keyword": {"type": "string"}}, "required": ["service"]}}},
    {"type": "function", "function": {"name": "get_recent_deploys", "description": "Get deploys",
     "parameters": {"type": "object", "properties": {"service": {"type": "string"}}, "required": ["service"]}}},
    {"type": "function", "function": {"name": "get_metrics", "description": "Get metrics",
     "parameters": {"type": "object", "properties": {"service": {"type": "string"}}, "required": ["service"]}}},
    {"type": "function", "function": {"name": "check_runbook", "description": "Read runbook",
     "parameters": {"type": "object", "properties": {"service": {"type": "string"}}, "required": ["service"]}}},
]


def _call_tool(name, args, alert, span: NodeSpan) -> dict:
    start = time.perf_counter()
    try:
        result = TOOL_REGISTRY[name](**args)
        success = True
    except Exception as e:
        result, success = {"error": str(e)}, False
    latency_ms = (time.perf_counter() - start) * 1000
    score = _relevance(alert, json.dumps(result))
    span.tool_calls.append(ToolCallRecord(name, success, latency_ms, score))
    return result


# ========== SCHEMAS & STATE ==========

class EscalationState(TypedDict):
    alert: str
    parsed: dict
    evidence: Annotated[list, operator.add]
    attempts: int
    verdict: str
    feedback: str
    report: dict
    approved_by: str
    outcome: str


class ParsedAlert(BaseModel):
    service: str
    severity: Literal["SEV1", "SEV2", "SEV3"]
    symptom: str = Field(max_length=120)


class Verdict(BaseModel):
    verdict: Literal["ok", "revise"]
    feedback: str = ""


class RCAReport(BaseModel):
    root_cause: str = Field(min_length=20)
    evidence: list[str] = Field(min_length=1)
    severity: Literal["SEV1", "SEV2", "SEV3"]
    next_action: Literal["rollback_deploy", "page_oncall", "monitor", "restart_service"]


WRITE_ACTIONS = {"rollback_deploy", "page_oncall", "restart_service"}


def _classify_severity(text: str) -> str:
    low = text.lower()
    if any(k in low for k in ("critical", "can't checkout", "cant checkout", "revenue impact")):
        return "SEV1"
    if any(k in low for k in ("error rate", "exhausted", "climbing", "spiking")):
        return "SEV2"
    return "SEV3"


def _extract_service(text: str) -> str:
    match = re.search(r"([a-z0-9]+-(?:api|service))", text.lower())
    return match.group(1) if match else "unknown-service"


# ========== LIVE (real LLM) helpers — used when OFFLINE is False ==========

def llm_structured_live(prompt: str, schema, max_retries: int = 3):
    client = make_chat_client(settings)
    schema_json = json.dumps(schema.model_json_schema(), indent=2)
    response_format = {"type": "json_schema", "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()}}
    feedback = ""
    prompt_tokens = completion_tokens = 0
    for attempt in range(1, max_retries + 1):
        full_prompt = f"{prompt}\n\nRespond with ONLY a raw JSON object matching this exact schema:\n{schema_json}\nNo markdown, no commentary."
        if feedback:
            full_prompt += f"\n\nYour previous output FAILED validation:\n{feedback}\nFix these issues."
        resp = client.chat.completions.create(
            model=settings.resolved_model, extra_body=NO_THINK,
            messages=[{"role": "user", "content": full_prompt}], response_format=response_format,
        )
        if resp.usage:
            prompt_tokens += resp.usage.prompt_tokens
            completion_tokens += resp.usage.completion_tokens
        raw = (resp.choices[0].message.content or "").strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        if not raw.startswith("{"):
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            raw = m.group(0) if m else raw
        try:
            return schema.model_validate_json(raw), attempt, prompt_tokens, completion_tokens
        except ValidationError as e:
            feedback = "\n".join(f"- {'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors())
    raise RuntimeError(f"{schema.__name__} validation failed after {max_retries} attempts")


def run_investigation_live(task: str, alert: str, span: NodeSpan, max_iterations: int = 8) -> str:
    client = make_chat_client(settings)
    system = ("You investigate incidents. Use your tools to gather evidence. "
              "When done, end with a 3-line RCA draft: root cause / evidence / suggested action.")
    messages = [{"role": "system", "content": system}, {"role": "user", "content": task}]
    for _ in range(max_iterations):
        resp = client.chat.completions.create(model=settings.resolved_model, extra_body=NO_THINK,
                                               messages=messages, tools=RAW_TOOLS, tool_choice="auto")
        if resp.usage:
            span.prompt_tokens += resp.usage.prompt_tokens
            span.completion_tokens += resp.usage.completion_tokens
        msg = resp.choices[0].message
        assistant_msg = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [{"id": tc.id, "type": "function",
                                            "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                                           for tc in msg.tool_calls]
        messages.append(assistant_msg)
        if not msg.tool_calls:
            return msg.content or ""
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            result = _call_tool(tc.function.name, args, alert, span)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
    return "Max iterations reached."


# ========== GRAPH FACTORY ==========

def build_graph(trace: RunTrace, broken: bool = False):
    """`broken=True` reproduces one specific class of bug: the investigator's
    evidence doesn't actually match the alert's service (as if a tool had
    been called with the wrong argument, or context from a prior incident
    leaked in), and the verifier's check is too weak to catch it. The graph
    still completes and `verdict` still says "ok" — only an independent
    groundedness check (see evaluate_run below) catches the mismatch.
    """

    def triage(state):
        with traced(trace, "triage") as s:
            if OFFLINE:
                service, severity = _extract_service(state["alert"]), _classify_severity(state["alert"])
                parsed = ParsedAlert(service=service, severity=severity, symptom=state["alert"][:120])
                s.prompt_tokens, s.completion_tokens = _estimate_tokens(state["alert"]), 15
            else:
                parsed, s.retries, s.prompt_tokens, s.completion_tokens = llm_structured_live(
                    f"Parse this alert:\n{state['alert']}", ParsedAlert)
        return {"parsed": parsed.model_dump()}

    def investigate(state):
        service = state["parsed"]["service"]
        task = f"Investigate {service}: {state['alert']}"
        if state.get("verdict") == "revise":
            task += f"\nYour previous RCA was rejected. Fix: {state['feedback']}"
        with traced(trace, "investigate") as s:
            if OFFLINE:
                # A "broken" run investigates the wrong service — the bug
                # under test. A healthy run investigates the right one.
                probe_service = "unrelated-service" if broken else service
                logs = _call_tool("search_logs", {"service": probe_service}, state["alert"], s)
                metrics = _call_tool("get_metrics", {"service": probe_service}, state["alert"], s)
                runbook = _call_tool("check_runbook", {"service": probe_service}, state["alert"], s)
                draft = (f"root cause: {probe_service} — {logs['matches'][0]}\n"
                        f"evidence: error_rate={metrics['error_rate']}, p95={metrics['p95_latency_ms']}ms\n"
                        f"runbook: {runbook}")
                s.prompt_tokens, s.completion_tokens = _estimate_tokens(task), _estimate_tokens(draft)
            else:
                draft = run_investigation_live(task, state["alert"], s)
        return {"evidence": [draft], "attempts": state.get("attempts", 0) + 1}

    def verify(state):
        draft = state["evidence"][-1]
        with traced(trace, "verify") as s:
            if OFFLINE:
                service = state["parsed"]["service"]
                if broken:
                    # The bug: only checks that *an* error was found, not
                    # that it was found for *this* service.
                    ok = "error" in draft.lower()
                else:
                    ok = service in draft.lower() and ("error" in draft.lower() or "metric" in draft.lower())
                verdict = Verdict(verdict="ok" if ok else "revise",
                                  feedback="" if ok else "Evidence does not clearly reference the affected service.")
                s.prompt_tokens, s.completion_tokens = _estimate_tokens(draft), 10
            else:
                prompt = (f"EVIDENCE:\n{state['evidence']}\n\nRCA DRAFT:\n{draft}\n\n"
                         f"'ok' only if every claim is backed by evidence. Else 'revise' with feedback.")
                verdict, s.retries, s.prompt_tokens, s.completion_tokens = llm_structured_live(prompt, Verdict)
        return {"verdict": verdict.verdict, "feedback": verdict.feedback}

    def write_rca(state):
        draft = state["evidence"][-1]
        with traced(trace, "write_rca") as s:
            if OFFLINE:
                severity = state["parsed"]["severity"]
                action = "rollback_deploy" if severity in ("SEV1", "SEV2") else "monitor"
                report = RCAReport(root_cause=draft.split("\n")[0][:200] or "Investigation evidence gathered.",
                                   evidence=[draft], severity=severity, next_action=action)
                s.prompt_tokens, s.completion_tokens = _estimate_tokens(draft), 20
            else:
                report, s.retries, s.prompt_tokens, s.completion_tokens = llm_structured_live(
                    f"Convert this into a final RCA report:\n{draft}", RCAReport)
        return {"report": report.model_dump()}

    def human_gate(state):
        decision = interrupt({"action": state["report"]["next_action"], "root_cause": state["report"]["root_cause"]})
        ok = decision.get("approved")
        return {"approved_by": decision.get("approver") if ok else None, "outcome": "approved" if ok else "rejected"}

    def execute(state):
        return {"outcome": f"Executed {state['report']['next_action']}"}

    def escalate_human(state):
        return {"outcome": f"Human takeover. Evidence: {state['evidence']}"}

    def auto_close(state):
        return {"outcome": "Low severity — closed."}

    def route_triage(s): return "investigate" if s["parsed"]["severity"] in ("SEV1", "SEV2") else "auto_close"
    def route_verify(s): return "investigate" if s["verdict"] == "revise" and s["attempts"] < 3 else "write_rca"
    def route_rca(s): return "human_gate" if s["report"]["next_action"] in WRITE_ACTIONS else "auto_close"
    def route_gate(s): return "execute" if s["outcome"] == "approved" else "escalate_human"

    b = StateGraph(EscalationState)
    for name, fn in [("triage", triage), ("investigate", investigate), ("verify", verify),
                     ("write_rca", write_rca), ("human_gate", human_gate), ("execute", execute),
                     ("escalate_human", escalate_human), ("auto_close", auto_close)]:
        b.add_node(name, fn)
    b.add_edge(START, "triage")
    b.add_conditional_edges("triage", route_triage, {"investigate": "investigate", "auto_close": "auto_close"})
    b.add_edge("investigate", "verify")
    b.add_conditional_edges("verify", route_verify, {"investigate": "investigate", "write_rca": "write_rca"})
    b.add_conditional_edges("write_rca", route_rca, {"human_gate": "human_gate", "auto_close": "auto_close"})
    b.add_conditional_edges("human_gate", route_gate, {"execute": "execute", "escalate_human": "escalate_human"})
    b.add_edge("execute", END)
    b.add_edge("escalate_human", END)
    b.add_edge("auto_close", END)
    return b.compile(checkpointer=InMemorySaver())


def run_once(alert: str, thread_id: str, broken: bool = False):
    """Runs one incident to completion, auto-approving the human gate.
    Returns (final_state, RunTrace)."""
    trace = RunTrace(thread_id=thread_id)
    app = build_graph(trace, broken=broken)
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 30}
    handler = _langfuse_handler()
    if handler:
        config["callbacks"] = [handler]

    state = app.invoke({"alert": alert, "attempts": 0, "evidence": []}, config)
    if state.get("__interrupt__") or "report" in state and state.get("outcome") not in ("approved", "rejected") and "Low severity" not in state.get("outcome", ""):
        # Paused at human_gate — auto-approve for reproducible evaluation runs.
        state = app.invoke(Command(resume={"approved": True, "approver": "eval-harness"}), config)
    return state, trace


# ========== EVALUATION ==========

@dataclass
class EvalResult:
    incident_id: str
    severity_correct: bool
    action_correct: bool
    grounded: bool
    retries: int
    tool_call_count: int
    escalated: bool
    latency_ms: float

    @property
    def score(self) -> float:
        return sum([self.severity_correct, self.action_correct, self.grounded]) / 3


def evaluate_run(incident: dict, state: dict, trace: RunTrace) -> EvalResult:
    parsed = state.get("parsed", {})
    report = state.get("report", {})
    evidence = state["evidence"][-1] if state.get("evidence") else ""

    severity_correct = parsed.get("severity") == incident["expected_severity"]
    if incident["expected_action"] is None:
        action_correct = "closed" in state.get("outcome", "").lower()
    else:
        action_correct = report.get("next_action") == incident["expected_action"]

    # Independent groundedness check — deliberately NOT the same logic the
    # (possibly broken) verify node used, so it can catch what verify missed.
    expected_service = _extract_service(incident["alert"])
    grounded = expected_service in evidence.lower() if evidence else severity_correct

    return EvalResult(
        incident_id=incident["id"],
        severity_correct=severity_correct,
        action_correct=action_correct,
        grounded=grounded,
        retries=trace.total_retries,
        tool_call_count=len(trace.tool_calls),
        escalated="Human takeover" in state.get("outcome", ""),
        latency_ms=trace.total_latency_ms,
    )


def run_eval_suite(dataset: list[dict], broken: bool = False) -> list[EvalResult]:
    results = []
    for incident in dataset:
        state, trace = run_once(incident["alert"], thread_id=f"eval-{incident['id']}", broken=broken)
        results.append(evaluate_run(incident, state, trace))
    return results


def print_eval_summary(results: list[EvalResult]) -> None:
    n = len(results)
    avg = lambda xs: sum(xs) / n if n else 0.0
    print(f"{'incident':10s} {'severity':9s} {'action':7s} {'grounded':9s} {'retries':8s} {'tools':6s} {'latency_ms':11s} {'score':6s}")
    for r in results:
        print(f"{r.incident_id:10s} {str(r.severity_correct):9s} {str(r.action_correct):7s} "
              f"{str(r.grounded):9s} {r.retries:<8d} {r.tool_call_count:<6d} {r.latency_ms:<11.1f} {r.score:.2f}")
    print("-" * 80)
    print(f"severity accuracy: {avg([r.severity_correct for r in results]):.2f}   "
          f"action accuracy: {avg([r.action_correct for r in results]):.2f}   "
          f"grounded rate: {avg([r.grounded for r in results]):.2f}   "
          f"escalation rate: {avg([r.escalated for r in results]):.2f}   "
          f"avg retries: {avg([r.retries for r in results]):.2f}   "
          f"avg latency_ms: {avg([r.latency_ms for r in results]):.1f}")


if __name__ == "__main__":
    print(f"Mode: {'OFFLINE (no LLM_PROVIDER API key configured)' if OFFLINE else f'LIVE ({settings.llm_provider})'}")
    print(f"Langfuse: {'configured — spans forwarded' if settings.langfuse_configured else 'not configured — local trace only'}\n")

    dataset = json.loads((DATA_DIR / "incident_eval_set.json").read_text())

    print("=== Eval suite: healthy graph across the incident dataset ===")
    healthy_results = run_eval_suite(dataset, broken=False)
    print_eval_summary(healthy_results)

    print("\n=== Single narrated run (inc01), with its full trace ===")
    state, trace = run_once(dataset[0]["alert"], thread_id="narrated-1")
    print(f"alert: {dataset[0]['alert']}")
    print(f"outcome: {state.get('outcome')}")
    trace.print_table()

    print("\n=== Intentionally broken run: investigator targets the wrong service, "
          "verify's check is too weak to notice ===")
    broken_incident = dataset[0]
    broken_state, broken_trace = run_once(broken_incident["alert"], thread_id="broken-1", broken=True)
    broken_eval = evaluate_run(broken_incident, broken_state, broken_trace)
    print(f"alert: {broken_incident['alert']}")
    print(f"verify's own verdict inside the graph: ok (it always is here — that's the bug)")
    print(f"independent groundedness check in the eval layer: grounded={broken_eval.grounded}")
    broken_trace.print_table()
    print(
        "\nDiagnosis: the graph completed, verify said 'ok', and the RCA got written and "
        "routed to human_gate like any other SEV1 — nothing inside the run looks wrong. "
        "The trace is what makes it visible: the investigate node's tool calls all carry "
        f"a retrieval_score near 0 (evidence about 'unrelated-service' has ~no lexical "
        f"overlap with an alert about '{_extract_service(broken_incident['alert'])}'), and the "
        "eval layer's independent groundedness check — which doesn't trust the graph's own "
        "verdict — catches what verify missed."
    )
