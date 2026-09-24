"""Observability + evaluation for the escalation/incident-response agent.

Library code for notebooks/agents/agent_observability_eval.ipynb. Kept here
(not in the notebook) so the graph, the trace, and the eval logic are
importable and unit tested.

The escalation-agent notebook wires Langfuse's LangGraph callback into the
graph, which gives you a trace viewer — not a number you can track over time
or a way to catch a regression automatically. This module adds that layer:

  * Per-node spans (RunTrace): latency, model/provider, prompt/completion
    tokens, schema-validation retries, and per-tool-call arguments, success,
    and a retrieval-relevance score.
  * Langfuse, when LANGFUSE_* is configured: each incident is one trace — a
    root span holding the LangGraph node tree (LangChain callback), every
    LLM call as a generation inside its node (Langfuse's OpenAI wrapper;
    the raw client is invisible to the callback), the RunTrace summary as
    metadata, and the eval results attached as scores, so a regression shows
    up as a failing score on a specific trace in the Langfuse UI.
  * An eval harness over a labeled incident set (data/incident_eval_set.json):
    severity/action accuracy, RCA groundedness, retries, tool calls, latency.
  * One intentionally broken run: a tool-layer bug sends every evidence tool
    call to the wrong service (as if a stale service name leaked in from a
    previous incident), and the verifier's check is weakened so it only asks
    "was any error found?". The graph completes and says `verdict: ok`; the
    eval layer's groundedness check, which reads the tool-call arguments off
    the trace instead of trusting the verdict, catches it.

Two modes, picked from whether an API key is configured for LLM_PROVIDER:

  * OFFLINE: triage/investigate/verify/RCA use deterministic rule-based
    stand-ins. The rules were written against this dataset, so OFFLINE
    accuracy is always perfect by construction — it checks the harness,
    not an agent.
  * LIVE: the configured LLM does triage, tool-calling investigation,
    verification, and RCA writing. Same graph, same instrumentation. The
    broken run is the same code-level bug in both modes.
"""
import json
import operator
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field, ValidationError

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from ai_engineering.config import NO_THINK, get_settings, make_chat_client

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
settings = get_settings()
OFFLINE = not settings.has_llm_credentials

# Hosted NIM returns transient 503 "overloaded"; the SDK's default of 2
# retries isn't enough for an eval that makes dozens of calls in a row.
EVAL_MAX_RETRIES = 6
WRONG_SERVICE = "unrelated-service"


def load_incidents() -> list[dict]:
    return json.loads((DATA_DIR / "incident_eval_set.json").read_text())


def _estimate_tokens(text: str) -> int:
    """chars/4 heuristic — OFFLINE mode's token figures only."""
    return max(1, len(text) // 4)


def _norm(text: str) -> str:
    """'payments-api', 'Payments API' and 'payments_api' all compare equal."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


# ========== OBSERVABILITY: per-node/per-tool trace, independent of Langfuse ==========

@dataclass
class ToolCallRecord:
    name: str
    args: dict
    success: bool
    latency_ms: float
    retrieval_score: float = 0.0  # token-overlap relevance of the tool result to the alert, 0-1


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
    langfuse_trace_id: str | None = None

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
        print(f"  trace {self.thread_id}: {self.total_latency_ms:.0f}ms total, "
              f"{self.total_tokens} tokens{' (est.)' if OFFLINE else ''}, {self.total_retries} retries")
        for s in self.spans:
            err = f"  ERROR={s.error}" if s.error else ""
            print(f"    [{s.node:12s}] {s.latency_ms:8.0f}ms  "
                  f"tokens={s.prompt_tokens + s.completion_tokens:5d}  retries={s.retries}{err}")
            for tc in s.tool_calls:
                status = "ok" if tc.success else "FAIL"
                print(f"        -> {tc.name}({json.dumps(tc.args)}) {status}  retrieval_score={tc.retrieval_score:.2f}")


@contextmanager
def traced(trace: RunTrace, node: str):
    span = NodeSpan(node=node, model="rules" if OFFLINE else settings.resolved_model,
                    provider="offline" if OFFLINE else settings.llm_provider)
    start = time.perf_counter()
    try:
        yield span
    except Exception as e:
        span.error = str(e)
        raise
    finally:
        span.latency_ms = (time.perf_counter() - start) * 1000
        trace.spans.append(span)


_langfuse = None


def langfuse_client():
    """The Langfuse client if LANGFUSE_* is configured, else None. Initialized
    once with the keys from Settings (which reads .env itself, so they may not
    be in os.environ for the SDK to find on its own)."""
    global _langfuse
    if _langfuse is None and settings.langfuse_configured:
        from langfuse import Langfuse

        _langfuse = Langfuse(
            public_key=settings.langfuse_public_key.get_secret_value(),
            secret_key=settings.langfuse_secret_key.get_secret_value(),
            host=settings.langfuse_base_url,
        )
    return _langfuse


def trace_summary(trace: "RunTrace") -> dict:
    """RunTrace as plain JSON for Langfuse metadata."""
    return {
        "total_latency_ms": round(trace.total_latency_ms),
        "total_tokens": trace.total_tokens,
        "total_retries": trace.total_retries,
        "nodes": [{"node": s.node, "latency_ms": round(s.latency_ms), "retries": s.retries,
                   "tokens": s.prompt_tokens + s.completion_tokens,
                   "tool_calls": [{"name": tc.name, "args": tc.args, "success": tc.success,
                                   "retrieval_score": round(tc.retrieval_score, 2)} for tc in s.tool_calls]}
                  for s in trace.spans],
    }


def _relevance(alert: str, text: str) -> float:
    """Token-overlap relevance of `text` to the alert, 0-1."""
    a = set(re.findall(r"\w+", alert.lower()))
    t = set(re.findall(r"\w+", text.lower()))
    if not a:
        return 0.0
    return len(a & t) / len(a)


# ========== TOOLS (mock data in both modes — no real infra) ==========
# Every result names the service it was asked about, the way real log lines
# and metric labels do, so evidence from the wrong service is visibly wrong.

def _search_logs(service, keyword="ERROR"):
    return {"service": service, "matches": [f"10:42:03 {keyword} {service}: timeout connecting to postgres-primary"]}


def _get_recent_deploys(service):
    return {"service": service, "deploys": [{"version": "v2.14.3", "time": "10:35", "author": "dana@corp"}]}


def _get_metrics(service):
    return {"service": service, "p95_latency_ms": 4200, "error_rate": 0.31, "db_conn_active": 100}


def _check_runbook(service):
    return {"service": service, "runbook": "1. Check DB pool. 2. Rollback if error_rate > 10%. 3. Page dana."}


TOOL_REGISTRY = {
    "search_logs": _search_logs,
    "get_recent_deploys": _get_recent_deploys,
    "get_metrics": _get_metrics,
    "check_runbook": _check_runbook,
}
RAW_TOOLS = [
    {"type": "function", "function": {"name": "search_logs", "description": "Search a service's logs for a keyword",
     "parameters": {"type": "object", "properties": {"service": {"type": "string"}, "keyword": {"type": "string"}}, "required": ["service"]}}},
    {"type": "function", "function": {"name": "get_recent_deploys", "description": "Recent deploys of a service",
     "parameters": {"type": "object", "properties": {"service": {"type": "string"}}, "required": ["service"]}}},
    {"type": "function", "function": {"name": "get_metrics", "description": "Current metrics for a service",
     "parameters": {"type": "object", "properties": {"service": {"type": "string"}}, "required": ["service"]}}},
    {"type": "function", "function": {"name": "check_runbook", "description": "Read a service's runbook",
     "parameters": {"type": "object", "properties": {"service": {"type": "string"}}, "required": ["service"]}}},
]


def _call_tool(name: str, args: dict, alert: str, span: NodeSpan, broken: bool = False) -> dict:
    """Runs one tool call and records it on the span. `broken=True` is the
    bug under test: the tool layer swaps in a stale service name, so the
    agent asked for the right thing but got evidence about something else."""
    if broken and "service" in args:
        args = {**args, "service": WRONG_SERVICE}
    start = time.perf_counter()
    try:
        if name not in TOOL_REGISTRY:
            raise KeyError(f"unknown tool {name!r}")
        result = TOOL_REGISTRY[name](**args)
        success = True
    except Exception as e:
        result, success = {"error": str(e)}, False
    latency_ms = (time.perf_counter() - start) * 1000
    span.tool_calls.append(ToolCallRecord(name, args, success, latency_ms, _relevance(alert, json.dumps(result))))
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

SEVERITY_POLICY = (
    "Severity policy: SEV1 = customer-facing outage or revenue impact (users cannot complete core actions). "
    "SEV2 = degraded or rising errors on a customer-facing service without a full outage. "
    "SEV3 = no customer impact yet."
)


def _classify_severity(text: str) -> str:
    low = text.lower()
    if any(k in low for k in ("critical", "can't checkout", "cant checkout", "revenue impact", "completely down")):
        return "SEV1"
    if any(k in low for k in ("error rate", "exhausted", "climbing", "spiking")):
        return "SEV2"
    return "SEV3"


def _extract_service(text: str) -> str:
    match = re.search(r"([a-z0-9]+-(?:api|service))", text.lower())
    return match.group(1) if match else "unknown-service"


# ========== LIVE (real LLM) helpers ==========

_client = None


def _live_client():
    global _client
    if _client is None:
        if langfuse_client():
            # Drop-in wrapper: same OpenAI client, but each call is recorded as
            # a Langfuse generation (model, tokens, latency) under the current span.
            from langfuse.openai import OpenAI

            _client = OpenAI(base_url=settings.resolved_base_url, api_key=settings.require_api_key(),
                             max_retries=EVAL_MAX_RETRIES)
        else:
            _client = make_chat_client(settings, max_retries=EVAL_MAX_RETRIES)
    return _client


def llm_structured_live(prompt: str, schema, max_attempts: int = 3):
    """Returns (parsed, retries, prompt_tokens, completion_tokens). `retries`
    counts schema-validation re-asks only; transport retries (503s etc.)
    are the SDK's and don't show up here."""
    schema_json = json.dumps(schema.model_json_schema(), indent=2)
    response_format = {"type": "json_schema", "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()}}
    feedback = ""
    prompt_tokens = completion_tokens = 0
    for attempt in range(1, max_attempts + 1):
        full_prompt = f"{prompt}\n\nRespond with ONLY a raw JSON object matching this exact schema:\n{schema_json}\nNo markdown, no commentary."
        if feedback:
            full_prompt += f"\n\nYour previous output FAILED validation:\n{feedback}\nFix these issues."
        resp = _live_client().chat.completions.create(
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
            return schema.model_validate_json(raw), attempt - 1, prompt_tokens, completion_tokens
        except ValidationError as e:
            feedback = "\n".join(f"- {'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors())
    raise RuntimeError(f"{schema.__name__} validation failed after {max_attempts} attempts")


def run_investigation_live(task: str, alert: str, span: NodeSpan, broken: bool, max_iterations: int = 8) -> str:
    system = ("You investigate production incidents. Use your tools to gather evidence about the affected service. "
              "When done, reply with a 3-line RCA draft: root cause / evidence / suggested action. "
              "Only state what the tool results show.")
    messages = [{"role": "system", "content": system}, {"role": "user", "content": task}]
    for _ in range(max_iterations):
        resp = _live_client().chat.completions.create(model=settings.resolved_model, extra_body=NO_THINK,
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
            try:
                args = json.loads(tc.function.arguments or "{}")
                if not isinstance(args, dict):
                    raise ValueError("tool arguments must be a JSON object")
            except (json.JSONDecodeError, ValueError) as e:
                # Record the malformed call and hand the error back to the
                # model instead of crashing the whole run.
                span.tool_calls.append(ToolCallRecord(tc.function.name, {"_raw": tc.function.arguments}, False, 0.0))
                result = {"error": f"invalid arguments: {e}"}
            else:
                result = _call_tool(tc.function.name, args, alert, span, broken=broken)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
    return "Max iterations reached."


# ========== GRAPH FACTORY ==========

def build_graph(trace: RunTrace, broken: bool = False):
    """`broken=True` reproduces one class of bug in both modes: every evidence
    tool call is redirected to WRONG_SERVICE, and verify only checks that
    *some* error was found, not that it was found for *this* service."""

    def triage(state):
        with traced(trace, "triage") as s:
            if OFFLINE:
                service, severity = _extract_service(state["alert"]), _classify_severity(state["alert"])
                parsed = ParsedAlert(service=service, severity=severity, symptom=state["alert"][:120])
                s.prompt_tokens, s.completion_tokens = _estimate_tokens(state["alert"]), 15
            else:
                parsed, s.retries, s.prompt_tokens, s.completion_tokens = llm_structured_live(
                    f"Parse this production alert. `service` is the exact service name as written in the alert.\n"
                    f"{SEVERITY_POLICY}\n\nAlert: {state['alert']}", ParsedAlert)
        return {"parsed": parsed.model_dump()}

    def investigate(state):
        service = state["parsed"]["service"]
        task = f"Investigate {service}: {state['alert']}"
        if state.get("verdict") == "revise":
            task += f"\nYour previous RCA was rejected. Fix: {state['feedback']}"
        with traced(trace, "investigate") as s:
            if OFFLINE:
                logs = _call_tool("search_logs", {"service": service}, state["alert"], s, broken)
                metrics = _call_tool("get_metrics", {"service": service}, state["alert"], s, broken)
                runbook = _call_tool("check_runbook", {"service": service}, state["alert"], s, broken)
                draft = (f"root cause: {logs['matches'][0]}\n"
                         f"evidence: {metrics['service']} error_rate={metrics['error_rate']}, p95={metrics['p95_latency_ms']}ms\n"
                         f"runbook: {runbook['runbook']}")
                s.prompt_tokens, s.completion_tokens = _estimate_tokens(task), _estimate_tokens(draft)
            else:
                draft = run_investigation_live(task, state["alert"], s, broken)
        return {"evidence": [draft], "attempts": state.get("attempts", 0) + 1}

    def verify(state):
        draft = state["evidence"][-1]
        with traced(trace, "verify") as s:
            if broken:
                # The bug: only checks that *an* error was found, not that it
                # was found for *this* service. Same code in both modes.
                ok = "error" in draft.lower() or "timeout" in draft.lower()
                verdict = Verdict(verdict="ok" if ok else "revise", feedback="" if ok else "No error evidence found.")
            elif OFFLINE:
                ok = _norm(state["parsed"]["service"]) in _norm(draft) and "error" in draft.lower()
                verdict = Verdict(verdict="ok" if ok else "revise",
                                  feedback="" if ok else "Evidence does not clearly reference the affected service.")
                s.prompt_tokens, s.completion_tokens = _estimate_tokens(draft), 10
            else:
                prompt = (f"ALERT: {state['alert']}\nAFFECTED SERVICE: {state['parsed']['service']}\n\n"
                          f"RCA DRAFT:\n{draft}\n\n"
                          "'ok' only if every claim is backed by evidence about the affected service. "
                          "Else 'revise' with feedback.")
                verdict, s.retries, s.prompt_tokens, s.completion_tokens = llm_structured_live(prompt, Verdict)
        return {"verdict": verdict.verdict, "feedback": verdict.feedback}

    def write_rca(state):
        draft = state["evidence"][-1]
        with traced(trace, "write_rca") as s:
            if OFFLINE:
                severity = state["parsed"]["severity"]
                action = "rollback_deploy" if severity in ("SEV1", "SEV2") else "monitor"
                report = RCAReport(root_cause=draft.split("\n")[0][:200], evidence=[draft],
                                   severity=severity, next_action=action)
                s.prompt_tokens, s.completion_tokens = _estimate_tokens(draft), 20
            else:
                report, s.retries, s.prompt_tokens, s.completion_tokens = llm_structured_live(
                    f"Convert this into a final RCA report. Severity is {state['parsed']['severity']}.\n{draft}", RCAReport)
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

    def invoke():
        state = app.invoke({"alert": alert, "attempts": 0, "evidence": []}, config)
        if state.get("__interrupt__"):
            # Paused at human_gate — auto-approve for reproducible evaluation runs.
            state = app.invoke(Command(resume={"approved": True, "approver": "eval-harness"}), config)
        return state

    lf = langfuse_client()
    if lf is None:
        return invoke(), trace

    from langfuse.langchain import CallbackHandler

    config["callbacks"] = [CallbackHandler()]
    # One root span per incident, so both invokes (before and after the
    # human_gate interrupt) and every LLM generation land in a single trace.
    with lf.start_as_current_observation(name=f"incident:{thread_id}", as_type="span") as root:
        state = invoke()
        trace.langfuse_trace_id = lf.get_current_trace_id()
        root.update(
            input={"alert": alert},
            output={"outcome": state.get("outcome"), "report": state.get("report")},
            metadata={"broken": broken, "mode": "offline" if OFFLINE else "live",
                      "model": settings.resolved_model, "run_trace": trace_summary(trace)},
        )
    return state, trace


# ========== EVALUATION ==========

@dataclass
class EvalResult:
    incident_id: str
    predicted_severity: str
    severity_correct: bool
    action_correct: bool
    grounded: bool
    tools_on_target: bool   # every evidence tool call targeted the alert's service
    mentions_service: bool  # the final draft names the alert's service (informational, not scored)
    retries: int
    tool_call_count: int
    tokens: int
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

    # Independent groundedness check — deliberately NOT the verify node's
    # logic. It reads what the tools were actually asked (from the trace),
    # which the draft's wording can't paper over. Whether the draft *names*
    # the service is recorded but not scored: in live runs the model copies
    # the name from its task prompt even when every tool result was about a
    # different service, and healthy drafts often omit it — a false signal
    # in both directions.
    expected = _norm(_extract_service(incident["alert"]))
    evidence_calls = [tc for tc in trace.tool_calls if tc.name in TOOL_REGISTRY and tc.success]
    tools_on_target = all(expected == _norm(str(tc.args.get("service", ""))) for tc in evidence_calls)
    mentions_service = expected in _norm(evidence)
    if evidence_calls:
        grounded = tools_on_target
    else:
        # No investigation (auto-closed at triage) or no successful tool
        # call: there's no evidence to be grounded in. Only an auto-close
        # of a genuinely low-severity incident counts as fine.
        grounded = not evidence and incident["expected_action"] is None

    return EvalResult(
        incident_id=incident["id"],
        predicted_severity=parsed.get("severity", "?"),
        severity_correct=severity_correct,
        action_correct=action_correct,
        grounded=grounded,
        tools_on_target=tools_on_target,
        mentions_service=mentions_service,
        retries=trace.total_retries,
        tool_call_count=len(trace.tool_calls),
        tokens=trace.total_tokens,
        latency_ms=trace.total_latency_ms,
    )


def score_in_langfuse(result: EvalResult, trace: RunTrace) -> None:
    """Attach eval results to the run's Langfuse trace as scores. No-op
    without Langfuse. Call flush_langfuse() before the process exits."""
    lf = langfuse_client()
    if lf is None or trace.langfuse_trace_id is None:
        return
    for name, value in [("severity_correct", result.severity_correct), ("action_correct", result.action_correct),
                        ("grounded", result.grounded), ("tools_on_target", result.tools_on_target)]:
        lf.create_score(trace_id=trace.langfuse_trace_id, name=name, value=float(value), data_type="BOOLEAN")
    lf.create_score(trace_id=trace.langfuse_trace_id, name="eval_score", value=result.score, data_type="NUMERIC")


def flush_langfuse() -> None:
    if langfuse_client():
        langfuse_client().flush()


def run_eval_suite(dataset: list[dict], broken: bool = False) -> list[EvalResult]:
    results = []
    for incident in dataset:
        state, trace = run_once(incident["alert"], thread_id=f"eval-{incident['id']}", broken=broken)
        result = evaluate_run(incident, state, trace)
        score_in_langfuse(result, trace)
        results.append(result)
    return results


def summarize(results: list[EvalResult]) -> dict:
    n = len(results) or 1
    avg = lambda xs: round(sum(xs) / n, 2)
    return {
        "severity_accuracy": avg([r.severity_correct for r in results]),
        "action_accuracy": avg([r.action_correct for r in results]),
        "grounded_rate": avg([r.grounded for r in results]),
        "avg_retries": avg([r.retries for r in results]),
        "avg_tool_calls": avg([r.tool_call_count for r in results]),
        "avg_tokens": round(sum(r.tokens for r in results) / n),
        "avg_latency_ms": round(sum(r.latency_ms for r in results) / n),
    }
