"""Observability + evaluation for the escalation/incident-response agent.

Library code for notebooks/agents/agent_observability_eval.ipynb. Kept here
(not in the notebook) so the graph, the trace, and the eval logic are
importable and unit tested.

The escalation-agent notebook wires Langfuse's LangGraph callback into the
graph, which gives you a trace viewer — not a number you can track over time
or a way to catch a regression automatically. This module adds that layer:

  * Model-node spans (RunTrace) for the four LLM-backed nodes (triage,
    investigate, verify, write_rca): latency, model/provider,
    prompt/completion tokens, schema-validation retries, and every tool call
    with its arguments, result, success, and a relevance score. The
    deterministic routing/terminal nodes (human_gate, execute, auto_close,
    escalate_human) aren't spanned; human_gate re-executes on resume, so a
    span there would double-count.
  * Langfuse, when LANGFUSE_* is configured: each incident is one trace — a
    root span holding the LangGraph node tree (LangChain callback), every
    agent LLM call as a generation inside its node (Langfuse's OpenAI
    wrapper; the raw client is invisible to the callback), the RunTrace
    summary as metadata, and the eval results attached as scores.
  * An eval harness over a labeled incident set (data/incident_eval_set.json).
    Each incident carries its own tool fixtures (logs, deploys, metrics,
    dependencies), and one shared runbook maps evidence to an action, so the
    correct action — rollback, page, restart, monitor, or close at triage —
    has to be derived from what the tools return. Scored separately:
      - severity and action accuracy against the labels;
      - tool targeting: did every evidence tool call query the alert's service?
      - RCA groundedness: are the RCA's claims supported by the tool results?
        (a deterministic check that every number/version/time in the RCA
        appears in the evidence, plus a claim-by-claim LLM judge when live).
    Tool targeting and RCA groundedness answer different questions: an agent
    can query the right service and still invent a root cause, or faithfully
    summarize evidence that came from the wrong service.
  * One intentionally broken run: a tool-layer bug sends every evidence tool
    call to the wrong service (as if a stale service name leaked in from a
    previous incident), and the verifier's check is weakened so it only asks
    "was any error found?". The graph completes and says `verdict: ok`; the
    tool-targeting check, which reads the tool-call arguments off the trace
    instead of trusting the verdict, catches it.

Two modes, picked from whether an API key is configured for LLM_PROVIDER:

  * OFFLINE: rule-based stand-ins apply the runbook to the fixtures. They're
    written against this runbook, so OFFLINE accuracy is perfect by
    construction — it checks the harness, not an agent.
  * LIVE: the configured LLM does triage, tool-calling investigation,
    verification, RCA writing, and the RCA groundedness judging. The broken
    run is the same code-level bug in both modes.
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

from ai_engineering.config import get_settings, make_chat_client, pacing_wait_seconds

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
settings = get_settings()
OFFLINE = not settings.has_llm_credentials
CHAT_EXTRA = settings.adapter.chat_extra_body()

# Hosted NIM returns transient 503 "overloaded" and 429s past ~40 req/min.
# Requests are paced client-side; the SDK retries short blips (its backoff
# caps at ~8s), and _chat() below rides out longer congestion on the shared
# endpoint — minutes of 429s at peak hours — with backoff up to 2 minutes.
EVAL_MAX_RETRIES = 4
EVAL_MAX_RPM = settings.llm_max_rpm or 30
CONGESTION_GIVE_UP_S = 20 * 60
_congestion_wait_s = 0.0  # total backoff slept in _chat(), excluded from span latency
# Transient provider errors retried by _chat(), by HTTP status ("429", "503", ...,
# "connection"). Reported by the notebook: it's a reliability number for the
# provider, and the reason a run took as long as it did.
TRANSIENT_ERRORS: dict[str, int] = {}


class _DropRetriedProviderErrors:
    """logging.Filter for Langfuse's OpenAI wrapper, which logs a warning for
    every failed call. Transient errors are retried and counted in
    TRANSIENT_ERRORS instead; anything else still gets logged."""

    def filter(self, record) -> bool:
        import openai

        transient = (openai.RateLimitError, openai.InternalServerError, openai.APIConnectionError)
        return not isinstance(record.msg, transient)


def _waits() -> float:
    """Self-imposed waiting so far: client-side pacing plus congestion backoff."""
    return pacing_wait_seconds() + _congestion_wait_s
WRONG_SERVICE = "unrelated-service"
ACTIONS = ("rollback_deploy", "page_oncall", "restart_service", "monitor")


def load_incidents() -> list[dict]:
    return json.loads((DATA_DIR / "incident_eval_set.json").read_text())


def _estimate_tokens(text: str) -> int:
    """chars/4 heuristic — OFFLINE mode's token figures only."""
    return max(1, len(text) // 4)


def _norm(text: str) -> str:
    """'payments-api', 'Payments API' and 'payments_api' all compare equal."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


# ========== OBSERVABILITY: model-node spans, independent of Langfuse ==========

@dataclass
class ToolCallRecord:
    name: str
    args: dict
    success: bool
    latency_ms: float
    retrieval_score: float = 0.0  # token-overlap relevance of the tool result to the alert, 0-1
    result: dict = field(default_factory=dict)


@dataclass
class NodeSpan:
    node: str
    latency_ms: float = 0.0
    model: str = ""
    provider: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    validation_retries: int = 0  # structured-output re-asks; SDK transport retries aren't counted
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
    def total_validation_retries(self) -> int:
        return sum(s.validation_retries for s in self.spans)

    @property
    def tool_calls(self) -> list[ToolCallRecord]:
        return [tc for s in self.spans for tc in s.tool_calls]

    @property
    def evidence_calls(self) -> list[ToolCallRecord]:
        """Successful calls to the evidence tools — what the RCA can rest on."""
        return [tc for tc in self.tool_calls if tc.success and tc.name in TOOL_NAMES]

    def print_table(self) -> None:
        print(f"  trace {self.thread_id}: {self.total_latency_ms:.0f}ms total, "
              f"{self.total_tokens} tokens{' (est.)' if OFFLINE else ''}, "
              f"{self.total_validation_retries} validation retries")
        for s in self.spans:
            err = f"  ERROR={s.error}" if s.error else ""
            print(f"    [{s.node:12s}] {s.latency_ms:8.0f}ms  "
                  f"tokens={s.prompt_tokens + s.completion_tokens:5d}  validation_retries={s.validation_retries}{err}")
            for tc in s.tool_calls:
                status = "ok" if tc.success else "FAIL"
                print(f"        -> {tc.name}({json.dumps(tc.args)}) {status}  retrieval_score={tc.retrieval_score:.2f}")


@contextmanager
def traced(trace: RunTrace, node: str):
    span = NodeSpan(node=node, model="rules" if OFFLINE else settings.resolved_model,
                    provider="offline" if OFFLINE else settings.llm_provider)
    start, waited0 = time.perf_counter(), _waits()
    try:
        yield span
    except Exception as e:
        span.error = str(e)
        raise
    finally:
        # Exclude pacing and congestion-backoff waits: latency should be the provider's.
        span.latency_ms = ((time.perf_counter() - start) - (_waits() - waited0)) * 1000
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
        "total_validation_retries": trace.total_validation_retries,
        "nodes": [{"node": s.node, "latency_ms": round(s.latency_ms), "validation_retries": s.validation_retries,
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


# ========== TOOLS (per-incident fixtures in both modes — no real infra) ==========
# Every result names the service it was asked about, the way real log lines
# and metric labels do. A service with no fixture (e.g. the broken run's
# WRONG_SERVICE) gets DECOY_FIXTURES: plausible, error-laden, and about the
# wrong thing.

RUNBOOK = """Incident runbook (applies to every service). Apply the first rule that matches:
1. Error rate above 10% that started within 30 minutes after a deploy of this service -> rollback_deploy.
2. A dependency owned by another team is failing or saturated (see get_dependencies) -> page_oncall for that team. Do not roll back this service.
3. A single worker/replica is stuck or crash-looping while the other replicas are healthy -> restart_service.
4. Otherwise, if the error rate is under 5% and p95 latency is inside its SLO -> monitor.
5. Anything else -> page_oncall."""

DECOY_FIXTURES = {
    "logs": ["10:41:12 ERROR {service}: timeout connecting to postgres-replica-3",
             "10:41:15 ERROR {service}: 503 from upstream"],
    "deploys": [{"version": "v9.1.0", "minutes_ago": 12, "author": "sam"}],
    "metrics": {"error_rate": 0.27, "p95_latency_ms": 3900, "slo_p95_ms": 1000, "replicas_healthy": "4/4"},
    "dependencies": [{"name": "postgres-replica-3", "owner": "db-team", "status": "healthy"}],
}


def _fixtures_for(service: str, incident: dict) -> dict:
    if _norm(service) == _norm(incident["service"]):
        return incident["fixtures"]
    return json.loads(json.dumps(DECOY_FIXTURES).replace("{service}", service))


def _tool_search_logs(fx, service, keyword=""):
    lines = [line.replace("{service}", service) for line in fx["logs"]]
    if keyword:
        lines = [line for line in lines if keyword.lower() in line.lower()]
    return {"service": service, "keyword": keyword, "matches": lines}


def _tool_get_recent_deploys(fx, service):
    return {"service": service, "deploys": fx["deploys"]}


def _tool_get_metrics(fx, service):
    return {"service": service, **fx["metrics"]}


def _tool_get_dependencies(fx, service):
    return {"service": service, "dependencies": fx["dependencies"]}


def _tool_check_runbook(fx, service):
    return {"service": service, "runbook": RUNBOOK}


TOOLS = {
    "search_logs": _tool_search_logs,
    "get_recent_deploys": _tool_get_recent_deploys,
    "get_metrics": _tool_get_metrics,
    "get_dependencies": _tool_get_dependencies,
    "check_runbook": _tool_check_runbook,
}
TOOL_NAMES = set(TOOLS)
_SERVICE_PARAM = {"service": {"type": "string", "description": "Exact service name, e.g. payments-api"}}
RAW_TOOLS = [
    {"type": "function", "function": {"name": "search_logs", "description": "Search a service's recent logs; empty keyword returns all lines",
     "parameters": {"type": "object", "properties": {**_SERVICE_PARAM, "keyword": {"type": "string"}}, "required": ["service"]}}},
    {"type": "function", "function": {"name": "get_recent_deploys", "description": "Recent deploys of a service, with minutes_ago",
     "parameters": {"type": "object", "properties": _SERVICE_PARAM, "required": ["service"]}}},
    {"type": "function", "function": {"name": "get_metrics", "description": "Current metrics for a service: error rate, latency vs SLO, replica health, saturation",
     "parameters": {"type": "object", "properties": _SERVICE_PARAM, "required": ["service"]}}},
    {"type": "function", "function": {"name": "get_dependencies", "description": "Status and owning team of a service's dependencies",
     "parameters": {"type": "object", "properties": _SERVICE_PARAM, "required": ["service"]}}},
    {"type": "function", "function": {"name": "check_runbook", "description": "The incident runbook mapping evidence to an action",
     "parameters": {"type": "object", "properties": _SERVICE_PARAM, "required": ["service"]}}},
]


def _call_tool(name: str, args: dict, incident: dict, span: NodeSpan, broken: bool = False) -> dict:
    """Runs one tool call and records it on the span. `broken=True` is the
    bug under test: the tool layer swaps in a stale service name, so the
    agent asked for the right thing but got evidence about something else."""
    if broken and "service" in args:
        args = {**args, "service": WRONG_SERVICE}
    start = time.perf_counter()
    try:
        if name not in TOOLS:
            raise KeyError(f"unknown tool {name!r}")
        service = args.get("service")
        if not isinstance(service, str) or not service:
            raise ValueError("`service` is required")
        result = TOOLS[name](_fixtures_for(service, incident), **args)
        success = True
    except Exception as e:
        result, success = {"error": str(e)}, False
    latency_ms = (time.perf_counter() - start) * 1000
    span.tool_calls.append(ToolCallRecord(name, args, success, latency_ms,
                                          _relevance(incident["alert"], json.dumps(result)), result))
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
    next_action: Literal["rollback_deploy", "page_oncall", "restart_service", "monitor"]


class ClaimCheck(BaseModel):
    claim: str
    supported: bool


class RCAClaims(BaseModel):
    claims: list[ClaimCheck]


WRITE_ACTIONS = {"rollback_deploy", "page_oncall", "restart_service"}

SEVERITY_POLICY = (
    "Severity policy: SEV1 = customer-facing outage or revenue impact (users cannot complete core actions). "
    "SEV2 = degraded or rising errors on a customer-facing service without a full outage. "
    "SEV3 = no customer impact yet."
)


# ========== OFFLINE rule-based stand-ins ==========

def _classify_severity(text: str) -> str:
    low = text.lower()
    if any(k in low for k in ("critical", "can't checkout", "cant checkout", "revenue impact", "completely down")):
        return "SEV1"
    if any(k in low for k in ("error rate", "exhausted", "climbing", "spiking", "rising", "delayed", "stuck")):
        return "SEV2"
    return "SEV3"


def _extract_service(text: str) -> str:
    match = re.search(r"([a-z0-9]+-(?:api|service))", text.lower())
    return match.group(1) if match else "unknown-service"


def apply_runbook(deploys: list[dict], metrics: dict, dependencies: list[dict]) -> tuple[str, str]:
    """RUNBOOK as code: (action, which rule fired). The OFFLINE agent uses
    this; the LIVE agent has to reach the same answer by reading the runbook."""
    error_rate = metrics.get("error_rate", 0)
    if error_rate > 0.10 and any(d.get("minutes_ago", 1e9) <= 30 for d in deploys):
        return "rollback_deploy", "rule 1: errors >10% within 30 min of a deploy"
    unhealthy = [d for d in dependencies if d.get("status", "healthy") != "healthy"]
    if unhealthy:
        return "page_oncall", f"rule 2: {unhealthy[0]['name']} ({unhealthy[0]['owner']}) is {unhealthy[0]['status']}"
    healthy, _, total = str(metrics.get("replicas_healthy", "1/1")).partition("/")
    if total and int(healthy) < int(total) and int(healthy) > 0:
        return "restart_service", f"rule 3: {healthy}/{total} replicas healthy"
    if error_rate < 0.05 and metrics.get("p95_latency_ms", 0) <= metrics.get("slo_p95_ms", float("inf")):
        return "monitor", "rule 4: inside SLO"
    return "page_oncall", "rule 5: no rule matched"


def _offline_rca(trace: RunTrace, severity: str) -> RCAReport:
    latest = {tc.name: tc.result for tc in trace.evidence_calls}
    logs, deploys = latest["search_logs"], latest["get_recent_deploys"]
    metrics, deps = latest["get_metrics"], latest["get_dependencies"]
    action, rule = apply_runbook(deploys["deploys"], metrics, deps["dependencies"])
    deploy = deploys["deploys"][0] if deploys["deploys"] else None
    evidence = [f"log: {logs['matches'][0]}" if logs["matches"] else "log: no matching lines",
                f"{metrics['service']} error_rate={metrics['error_rate']}, p95={metrics['p95_latency_ms']}ms",
                f"last deploy {deploy['version']} {deploy['minutes_ago']} minutes ago" if deploy else "no recent deploys",
                f"runbook {rule}"]
    return RCAReport(root_cause=f"{metrics['service']}: {evidence[0][5:]}"[:200], evidence=evidence,
                     severity=severity, next_action=action)


# ========== LIVE (real LLM) helpers ==========

_client = None


def _live_client():
    global _client
    if _client is None:
        if langfuse_client():
            # Drop-in wrapper: same OpenAI client, but each call is recorded as
            # a Langfuse generation (model, tokens, latency) under the current span.
            import logging

            from langfuse.openai import OpenAI

            logging.getLogger("langfuse").addFilter(_DropRetriedProviderErrors())

            _client = make_chat_client(settings, EVAL_MAX_RETRIES, EVAL_MAX_RPM, client_cls=OpenAI)
        else:
            _client = make_chat_client(settings, EVAL_MAX_RETRIES, EVAL_MAX_RPM)
    return _client


def _chat(client, **kwargs):
    """chat.completions.create, retried with exponential backoff (5s -> 120s,
    for up to CONGESTION_GIVE_UP_S) on rate limits, 5xx and connection errors.
    Other errors (400s, auth) raise immediately."""
    import openai
    from tenacity import retry, retry_if_exception_type, stop_after_delay, wait_exponential_jitter

    transient = (openai.RateLimitError, openai.InternalServerError, openai.APIConnectionError, openai.APITimeoutError)

    def record_wait(retry_state):
        global _congestion_wait_s
        _congestion_wait_s += retry_state.next_action.sleep
        exc = retry_state.outcome.exception()
        key = str(getattr(exc, "status_code", None) or "connection")
        TRANSIENT_ERRORS[key] = TRANSIENT_ERRORS.get(key, 0) + 1

    @retry(retry=retry_if_exception_type(transient), wait=wait_exponential_jitter(initial=5, max=120),
           stop=stop_after_delay(CONGESTION_GIVE_UP_S), before_sleep=record_wait, reraise=True)
    def call():
        return client.chat.completions.create(**kwargs)

    return call()


def llm_structured_live(prompt: str, schema, max_attempts: int = 3, client=None):
    """Returns (parsed, validation_retries, prompt_tokens, completion_tokens).
    Validation retries are schema re-asks only; transport retries (503s etc.)
    are the SDK's and aren't counted here."""
    client = client or _live_client()
    schema_json = json.dumps(schema.model_json_schema(), indent=2)
    response_format = {"type": "json_schema", "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()}}
    feedback = ""
    prompt_tokens = completion_tokens = 0
    for attempt in range(1, max_attempts + 1):
        full_prompt = f"{prompt}\n\nRespond with ONLY a raw JSON object matching this exact schema:\n{schema_json}\nNo markdown, no commentary."
        if feedback:
            full_prompt += f"\n\nYour previous output FAILED validation:\n{feedback}\nFix these issues."
        resp = _chat(
            client, model=settings.resolved_model, extra_body=CHAT_EXTRA,
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


def run_investigation_live(task: str, incident: dict, span: NodeSpan, broken: bool, max_iterations: int = 10) -> str:
    system = ("You investigate production incidents. Use your tools to gather evidence about the affected service, "
              "including its dependencies, and read the runbook. When done, reply with a short RCA draft: "
              f"root cause / evidence / recommended action (exactly one of: {', '.join(ACTIONS)}) and which runbook rule applies. "
              "Only state what the tool results show.")
    messages = [{"role": "system", "content": system}, {"role": "user", "content": task}]
    for _ in range(max_iterations):
        resp = _chat(_live_client(), model=settings.resolved_model, extra_body=CHAT_EXTRA,
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
                result = _call_tool(tc.function.name, args, incident, span, broken=broken)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
    return "Max iterations reached."


# ========== GRAPH FACTORY ==========

def build_graph(trace: RunTrace, incident: dict, broken: bool = False):
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
                parsed, s.validation_retries, s.prompt_tokens, s.completion_tokens = llm_structured_live(
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
                for tool in ("search_logs", "get_recent_deploys", "get_metrics", "get_dependencies", "check_runbook"):
                    _call_tool(tool, {"service": service}, incident, s, broken)
                draft = "\n".join(json.dumps(tc.result) for tc in s.tool_calls)
                s.prompt_tokens, s.completion_tokens = _estimate_tokens(task), _estimate_tokens(draft)
            else:
                draft = run_investigation_live(task, incident, s, broken)
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
                ok = _norm(state["parsed"]["service"]) in _norm(draft)
                verdict = Verdict(verdict="ok" if ok else "revise",
                                  feedback="" if ok else "Evidence does not reference the affected service.")
                s.prompt_tokens, s.completion_tokens = _estimate_tokens(draft), 10
            else:
                prompt = (f"ALERT: {state['alert']}\nAFFECTED SERVICE: {state['parsed']['service']}\n\n"
                          f"RCA DRAFT:\n{draft}\n\n"
                          "'ok' only if every claim is backed by evidence about the affected service. "
                          "Else 'revise' with feedback.")
                verdict, s.validation_retries, s.prompt_tokens, s.completion_tokens = llm_structured_live(prompt, Verdict)
        return {"verdict": verdict.verdict, "feedback": verdict.feedback}

    def write_rca(state):
        draft = state["evidence"][-1]
        with traced(trace, "write_rca") as s:
            if OFFLINE:
                report = _offline_rca(trace, state["parsed"]["severity"])
                s.prompt_tokens, s.completion_tokens = _estimate_tokens(draft), 60
            else:
                report, s.validation_retries, s.prompt_tokens, s.completion_tokens = llm_structured_live(
                    f"Convert this investigation into a final RCA report. Severity is {state['parsed']['severity']}. "
                    f"`next_action` must follow the runbook rule the investigation identified.\n\n{draft}", RCAReport)
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
        if state.get("report"):
            return {"outcome": f"No write action ({state['report']['next_action']}) — closed."}
        return {"outcome": "Low severity — closed at triage."}

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


def run_once(incident: dict, thread_id: str, broken: bool = False):
    """Runs one incident to completion, auto-approving the human gate.
    Returns (final_state, RunTrace)."""
    trace = RunTrace(thread_id=thread_id)
    app = build_graph(trace, incident, broken=broken)
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 30}
    alert = incident["alert"]

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
            metadata={"incident_id": incident["id"], "broken": broken, "mode": "offline" if OFFLINE else "live",
                      "model": settings.resolved_model, "run_trace": trace_summary(trace)},
        )
    return state, trace


# ========== EVALUATION ==========

_VERSION = re.compile(r"\bv\d+(?:\.\d+)+\b")
_TIME = re.compile(r"\b\d{1,2}:\d{2}\b")
# Digits not preceded by a letter (so p95 / SEV1 / v2 are identifiers, not figures),
# with optional thousands separators and decimals; a trailing unit (6200ms) is fine.
_NUMBER = re.compile(r"(?<![\w.,])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?!\d|,\d)|(?<![\w.,])\d+(?:\.\d+)?(?!\d|\.\d|,\d)")


def evidence_text(incident: dict, calls: list[ToolCallRecord]) -> str:
    """The alert plus every tool result, flattened to plain `key: value` lines.
    Not json.dumps: its escaping ("\\n1.") would hide figures from _facts."""
    lines = [incident["alert"]]

    def walk(obj, key=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, k)
        elif isinstance(obj, list):
            for v in obj:
                walk(v, key)
        else:
            lines.append(f"{key}: {obj}")

    for tc in calls:
        lines.append(f"[{tc.name}]")
        walk(tc.result)
    return "\n".join(lines)


def _facts(text: str) -> tuple[set[str], set[str], set[float]]:
    """Versions, clock times, and numbers mentioned in `text`. "6,200ms" is
    6200; identifiers with a letter before the digits (p95, SEV1) are skipped."""
    versions = set(_VERSION.findall(text))
    times = set(_TIME.findall(text))
    rest = _TIME.sub(" ", _VERSION.sub(" ", text))
    numbers = {float(n.replace(",", "")) for n in _NUMBER.findall(rest)}
    return versions, times, numbers


# Conversions an RCA may legitimately apply to a figure from the evidence:
# fraction <-> percent, ms -> s, minutes -> hours / days, and "48k"-style thousands.
_UNIT_FACTORS = (1, 100, 1 / 100, 1 / 1000, 1000, 1 / 60, 1 / 1440)


def _decimals(n_text: str) -> int:
    return len(n_text.split(".")[1]) if "." in n_text else 0


def _supported_number(n: float, n_text: str, evidence: set[float]) -> bool:
    """True if n equals some evidence value under a unit conversion, at the
    precision it was written: 6.2 matches 6200 (ms -> s), 25 matches 1500
    (minutes -> hours), 48 matches 48210 ("48k"), 31 matches 0.31."""
    places = _decimals(n_text)
    return any(round(x * f, places) == n for x in evidence for f in _UNIT_FACTORS)


def unsupported_facts(rca_text: str, evidence_text: str) -> list[str]:
    """Deterministic grounding check: every version, clock time, and number
    in the RCA must appear in the evidence (the tool results plus the alert),
    allowing standard unit conversions. Catches invented or mistyped figures
    (620 for 6200); says nothing about invented prose — that's the judge's job."""
    ev_versions, ev_times, ev_numbers = _facts(evidence_text)
    versions, times, _ = _facts(rca_text)
    missing = [v for v in sorted(versions - ev_versions)] + [t for t in sorted(times - ev_times)]
    rest = _TIME.sub(" ", _VERSION.sub(" ", rca_text))
    for n_text in dict.fromkeys(m.replace(",", "") for m in _NUMBER.findall(rest)):
        if not _supported_number(float(n_text), n_text, ev_numbers):
            missing.append(n_text)
    return missing


def judge_rca_claims(report: dict, evidence_text: str, langfuse_trace_id: str | None = None) -> tuple[int, int]:
    """LLM judge (live only): split the RCA into factual claims and mark each
    as supported by the tool results or not. Returns (claims, supported).

    Once langfuse.openai is imported it traces every OpenAI client in the
    process, so judge calls are traced too. Given the incident's trace id,
    they're nested under an `rca-judge` span in that trace, next to the agent
    run they grade, instead of landing as orphan traces."""
    prompt = ("You audit incident RCAs. Below are the raw TOOL RESULTS the investigating agent received, and the "
              "RCA it wrote. List every factual claim in the RCA (root cause, each evidence item, and why the "
              "action was chosen) and mark `supported` true only if the TOOL RESULTS state it. Claims about "
              "a different service than the tool results describe are unsupported.\n\n"
              f"TOOL RESULTS:\n{evidence_text}\n\nRCA:\n{json.dumps(report, indent=2)}")
    lf = langfuse_client()
    if lf is not None and langfuse_trace_id:
        with lf.start_as_current_observation(trace_context={"trace_id": langfuse_trace_id},
                                             name="rca-judge", as_type="span") as span:
            parsed, *_ = llm_structured_live(prompt, RCAClaims, client=_judge_client())
            span.update(output={"claims": [c.model_dump() for c in parsed.claims]})
    else:
        parsed, *_ = llm_structured_live(prompt, RCAClaims, client=_judge_client())
    return len(parsed.claims), sum(c.supported for c in parsed.claims)


_judge = None


def _judge_client():
    global _judge
    if _judge is None:
        _judge = make_chat_client(settings, EVAL_MAX_RETRIES, EVAL_MAX_RPM)
    return _judge


@dataclass
class EvalResult:
    incident_id: str
    expected_severity: str
    predicted_severity: str
    expected_action: str
    predicted_action: str
    severity_correct: bool
    action_correct: bool
    # None = not applicable (auto-closed at triage: no tool calls, no RCA)
    tool_target_ok: bool | None      # every evidence tool call queried the alert's service
    rca_grounded: bool | None        # RCA's facts/claims supported by the tool results
    unsupported: list[str]           # deterministic check: figures in the RCA not in the evidence
    rca_claims: int
    rca_claims_supported: int
    mentions_service: bool | None    # RCA names the alert's service (informational, not scored)
    validation_retries: int
    tool_call_count: int
    tokens: int
    latency_ms: float
    error: str | None = None         # the run crashed (e.g. retries exhausted); scored as a failure

    @property
    def score(self) -> float:
        parts = [self.severity_correct, self.action_correct, self.tool_target_ok, self.rca_grounded]
        parts = [p for p in parts if p is not None]
        return sum(parts) / len(parts)


def evaluate_run(incident: dict, state: dict, trace: RunTrace, judge: bool | None = None) -> EvalResult:
    """`judge` defaults to LIVE mode; the LLM claim judge needs a real model."""
    judge = (not OFFLINE) if judge is None else judge
    parsed = state.get("parsed", {})
    report = state.get("report") or {}
    predicted_action = report.get("next_action", "close")

    # Tool targeting reads what the tools were actually asked (from the trace),
    # which the RCA's wording can't paper over.
    expected_service = _norm(incident["service"])
    calls = trace.evidence_calls
    tool_target_ok = (all(expected_service == _norm(str(tc.args.get("service", ""))) for tc in calls)
                      if calls else (None if not report else False))

    rca_grounded, unsupported, n_claims, n_supported, mentions = None, [], 0, 0, None
    if report:
        evidence = evidence_text(incident, calls)
        rca_text = report["root_cause"] + "\n" + "\n".join(report["evidence"])
        unsupported = unsupported_facts(rca_text, evidence)
        mentions = expected_service in _norm(rca_text)
        rca_grounded = not unsupported and bool(calls)
        if judge and calls:
            n_claims, n_supported = judge_rca_claims(report, evidence, trace.langfuse_trace_id)
            rca_grounded = rca_grounded and n_claims > 0 and n_claims == n_supported

    return EvalResult(
        incident_id=incident["id"],
        expected_severity=incident["expected_severity"],
        predicted_severity=parsed.get("severity", "?"),
        expected_action=incident["expected_action"],
        predicted_action=predicted_action,
        severity_correct=parsed.get("severity") == incident["expected_severity"],
        action_correct=predicted_action == incident["expected_action"],
        tool_target_ok=tool_target_ok,
        rca_grounded=rca_grounded,
        unsupported=unsupported,
        rca_claims=n_claims,
        rca_claims_supported=n_supported,
        mentions_service=mentions,
        validation_retries=trace.total_validation_retries,
        tool_call_count=len(trace.tool_calls),
        tokens=trace.total_tokens,
        latency_ms=trace.total_latency_ms,
    )


def score_in_langfuse(result: EvalResult, trace: RunTrace) -> None:
    """Attach eval results to the run's Langfuse trace as scores. Metrics that
    don't apply to a run (None) are skipped rather than scored. No-op without
    Langfuse; call flush_langfuse() before the process exits."""
    lf = langfuse_client()
    if lf is None or trace.langfuse_trace_id is None:
        return
    for name, value in [("severity_correct", result.severity_correct), ("action_correct", result.action_correct),
                        ("tool_target_ok", result.tool_target_ok), ("rca_grounded", result.rca_grounded)]:
        if value is not None:
            lf.create_score(trace_id=trace.langfuse_trace_id, name=name, value=float(value), data_type="BOOLEAN")
    lf.create_score(trace_id=trace.langfuse_trace_id, name="eval_score", value=result.score, data_type="NUMERIC")


def flush_langfuse() -> None:
    if langfuse_client():
        langfuse_client().flush()


def failed_result(incident: dict, error: Exception) -> EvalResult:
    """A run that crashed counts as wrong on every applicable metric rather
    than silently dropping out of the denominator."""
    return EvalResult(
        incident_id=incident["id"], expected_severity=incident["expected_severity"], predicted_severity="?",
        expected_action=incident["expected_action"], predicted_action="error",
        severity_correct=False, action_correct=False,
        tool_target_ok=None if incident["expected_action"] == "close" else False,
        rca_grounded=None if incident["expected_action"] == "close" else False,
        unsupported=[], rca_claims=0, rca_claims_supported=0, mentions_service=None,
        validation_retries=0, tool_call_count=0, tokens=0, latency_ms=0.0,
        error=f"{type(error).__name__}: {str(error)[:200]}",
    )


def run_eval_suite(dataset: list[dict], broken: bool = False) -> list[EvalResult]:
    """One crashed incident (provider outage outlasting the retry budget,
    schema validation failing 3 times) is recorded and scored as a failure;
    it doesn't abort the rest of the suite."""
    results = []
    for incident in dataset:
        try:
            state, trace = run_once(incident, thread_id=f"eval-{incident['id']}", broken=broken)
            result = evaluate_run(incident, state, trace)
            score_in_langfuse(result, trace)
        except Exception as e:
            result = failed_result(incident, e)
        results.append(result)
    return results


def summarize(results: list[EvalResult]) -> dict:
    def rate(xs):
        xs = [x for x in xs if x is not None]
        return f"{sum(xs)}/{len(xs)}" if xs else "n/a"

    n = len(results) or 1
    return {
        "crashed_runs": sum(r.error is not None for r in results),
        "severity_accuracy": rate([r.severity_correct for r in results]),
        "action_accuracy": rate([r.action_correct for r in results]),
        "tool_target_accuracy": rate([r.tool_target_ok for r in results]),
        "rca_grounded_rate": rate([r.rca_grounded for r in results]),
        "rca_claims_supported": f"{sum(r.rca_claims_supported for r in results)}/{sum(r.rca_claims for r in results)}",
        "avg_validation_retries": round(sum(r.validation_retries for r in results) / n, 2),
        "avg_tool_calls": round(sum(r.tool_call_count for r in results) / n, 2),
        "avg_tokens": round(sum(r.tokens for r in results) / n),
        "avg_latency_ms": round(sum(r.latency_ms for r in results) / n),
    }
