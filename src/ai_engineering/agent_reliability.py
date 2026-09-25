"""The escalation agent, hardened: the same graph and incidents as
agent_eval.py, with every tool call going through tool_runtime.

Library code for notebooks/agents/agent_reliability_hardening.ipynb.
agent_eval.py measures the agent; this module changes how its tools run.
It reuses agent_eval's incident set, fixtures, runbook, schemas, offline
stand-ins and LLM helpers, so differences in behaviour come from the
reliability layer, not from a different agent.

What changes relative to agent_eval's graph:

  * Tools are `ToolSpec`s with Pydantic input and output models. The
    function-calling schema is generated from the input model.
  * The remediation actions are real WRITE tools (`rollback_deploy`,
    `page_oncall`, `restart_service`) against `InfraSimulator`, an in-memory
    deploy system / pager / orchestrator that honors idempotency keys. In
    agent_eval, `execute` only returns a string.
  * `investigate` gets a READ-scoped registry, `execute` a WRITE-scoped one.
    `write_rca` names the action *and its target* (the bad version, the team
    to page, the stuck replica); `propose_action` validates that against the
    write tool's input model, checks the evidence was gathered for the
    alert's service, and checks the runbook's precondition for that action
    (`action_precondition`) before anything reaches the approval gate. The approval
    is bound to the exact arguments and idempotency key.
  * Every run has a `RunBudget`. Every work node charges a graph step; LLM
    calls go through `BudgetedChatClient`; tool calls through the executor.
    A limit hit (or any other node failure) routes to `escalate_human` with
    the reason in state, instead of a GraphRecursionError or a crash.
  * `Scenario` injects faults deterministically: transient 503s, slow
    calls, malformed tool output, a prompt-injected log line, a runaway
    investigator, and writes that commit but lose their response.
"""
import json
import re
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from typing import Annotated, Callable, Literal, TypedDict

import operator
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from ai_engineering import agent_eval as ae
from ai_engineering.tool_runtime import (
    Approval, AuditLog, BudgetedChatClient, BudgetExceeded, BudgetLimits, InvalidToolInput, Permission,
    ProposedCall, RetryPolicy, RunBudget, ToolExecutor, ToolRegistry, ToolSpec, decide, propose,
)

OFFLINE = ae.OFFLINE
settings = ae.settings
WRITE_ACTIONS = ("rollback_deploy", "page_oncall", "restart_service")

# Sized for one incident on a paced (30 rpm) hosted endpoint: a healthy live
# run makes ~10 LLM calls, ~5 tool calls and ~8 graph steps.
DEFAULT_LIMITS = BudgetLimits(max_llm_calls=30, max_tool_calls=30, max_tokens=60_000,
                              max_cost_usd=None, max_seconds=600.0, max_graph_steps=20)


# ========== TOOL CONTRACTS ==========

Service = Annotated[str, StringConstraints(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$", max_length=63)]
Version = Annotated[str, StringConstraints(pattern=r"^v\d+(?:\.\d+)+$")]
Slug = Annotated[str, StringConstraints(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$", max_length=63)]


class _Args(BaseModel):
    # Invented arguments are an error the model gets back, not a silent drop.
    model_config = ConfigDict(extra="forbid")


class ServiceQuery(_Args):
    service: Service = Field(description="Exact service name as written in the alert, e.g. payments-api")


class LogQuery(ServiceQuery):
    keyword: str = Field("", max_length=64, description="Optional substring filter; empty returns all lines")


class RollbackArgs(_Args):
    service: Service
    from_version: Version = Field(description="The bad version currently deployed, to roll back from")


class PageArgs(_Args):
    team: Slug = Field(description="Owning team to page, e.g. db-team")
    service: Service
    summary: str = Field(min_length=10, max_length=300)


class RestartArgs(_Args):
    service: Service
    replica: Slug = Field(description="The stuck replica or worker, e.g. worker-3")


class Deploy(BaseModel):
    version: Version
    minutes_ago: int = Field(ge=0)
    author: str


class Dependency(BaseModel):
    name: str
    owner: str
    status: str


class LogsOut(BaseModel):
    service: Service
    keyword: str
    matches: list[str]


class DeploysOut(BaseModel):
    service: Service
    deploys: list[Deploy]


class MetricsOut(BaseModel):
    model_config = ConfigDict(extra="allow")  # queue_depth etc. vary by service
    service: Service
    error_rate: float = Field(ge=0, le=1, description="fraction, not percent")
    p95_latency_ms: float = Field(ge=0)
    slo_p95_ms: float = Field(gt=0)
    replicas_healthy: str = Field(pattern=r"^\d+/\d+$")


class DependenciesOut(BaseModel):
    service: Service
    dependencies: list[Dependency]


class RunbookOut(BaseModel):
    service: Service
    runbook: str


class WriteReceipt(BaseModel):
    action: str
    service: Service
    detail: str
    deduplicated: bool = False


# ========== SIMULATED INFRASTRUCTURE ==========

class BackendError(Exception):
    """What an HTTP backend raises. Classified by `status_code`, like the
    OpenAI SDK's and httpx's errors: 5xx/429 are retried, 4xx are not."""

    def __init__(self, status_code: int, message: str):
        super().__init__(f"{status_code}: {message}")
        self.status_code = status_code


def _previous_version(version: str) -> str:
    nums = [int(n) for n in version[1:].split(".")]
    for i in range(len(nums) - 1, -1, -1):
        if nums[i] > 0:
            nums[i] -= 1
            break
    return "v" + ".".join(map(str, nums))


class InfraSimulator:
    """In-memory deploy system, pager and orchestrator for one incident.

    Like a payments API, it honors idempotency keys: a request with a key it
    has already completed returns the stored result, flagged
    `deduplicated=True`, without acting again. `honor_keys=False` models a
    backend that doesn't, so a retried write acts twice. Side effects are
    listed in `effects`, so double-actions are visible as data."""

    def __init__(self, incident: dict, honor_keys: bool = True):
        self.honor_keys = honor_keys
        self.deployed: dict[str, str] = {}
        deploys = incident["fixtures"]["deploys"]
        if deploys:
            self.deployed[incident["service"]] = deploys[0]["version"]
        self.effects: list[dict] = []
        self._keys: dict[str, dict] = {}
        self._lock = threading.Lock()

    def _once(self, key: str | None, apply: Callable[[], dict]) -> dict:
        with self._lock:
            if self.honor_keys and key and key in self._keys:
                return {**self._keys[key], "deduplicated": True}
            result = apply()
            self.effects.append({**result, "idempotency_key": key})
            if key:
                self._keys[key] = result
            return result

    def rollback(self, service: str, from_version: str, key: str | None) -> dict:
        def apply():
            current = self.deployed.get(service)
            # Precondition, not a retryable conflict: rolling back "from" a
            # version that isn't live would roll back the wrong thing.
            if current != from_version:
                raise BackendError(412, f"{service} is at {current or 'no known version'}, not {from_version}")
            self.deployed[service] = _previous_version(from_version)
            return {"action": "rollback_deploy", "service": service,
                    "detail": f"{from_version} -> {self.deployed[service]}"}
        return self._once(key, apply)

    def page(self, team: str, service: str, summary: str, key: str | None) -> dict:
        return self._once(key, lambda: {"action": "page_oncall", "service": service,
                                        "detail": f"paged {team}: {summary[:80]}"})

    def restart(self, service: str, replica: str, key: str | None) -> dict:
        return self._once(key, lambda: {"action": "restart_service", "service": service,
                                        "detail": f"restarted {replica}"})


# ========== FAULT INJECTION ==========

@dataclass(frozen=True)
class Fault:
    """Deterministic misbehaviour for one tool, by call number:
    the first `transient_failures` calls raise a 503; the next `slow_calls`
    sleep `delay_s` before returning (or, with `after_commit`, apply the
    change *then* sleep — the "write succeeded, response lost" case);
    `malformed` rewrites every result (a tool-side bug)."""
    transient_failures: int = 0
    slow_calls: int = 0
    delay_s: float = 0.0
    after_commit: bool = False
    malformed: Callable[[dict], dict] | None = None


@dataclass
class Scenario:
    faults: dict[str, Fault] = field(default_factory=dict)
    # Appended to search_logs results for the alert's service: text an
    # attacker controls (a log line, a ticket body) that reaches the model.
    injected_log: str | None = None
    # "agent": the real (LIVE) or rule-based (OFFLINE) investigator.
    # "runaway": a code bug that keeps calling tools and never finishes.
    # "obeys_injection": an investigator that first does what the injected
    #   line says — calls a write tool — then investigates normally. The worst
    #   case for a prompt injection: the model fully complies.
    investigator: Literal["agent", "runaway", "obeys_injection"] = "agent"
    limits: BudgetLimits = DEFAULT_LIMITS
    backend_honors_keys: bool = True
    retry_writes: bool = True  # mark write tools idempotent (safe to retry) in the registry
    read_timeout_s: float = 5.0
    write_timeout_s: float = 5.0
    retry: RetryPolicy = RetryPolicy(max_attempts=3, initial_s=0.2, max_s=2.0, jitter_s=0.2)


class _Faults:
    def __init__(self, faults: dict[str, Fault]):
        self.faults = faults
        self.calls: Counter = Counter()
        self._lock = threading.Lock()

    def wrap(self, name: str, fn):
        fault = self.faults.get(name)
        if fault is None:
            return fn

        def run(args, ctx):
            with self._lock:
                self.calls[name] += 1
                n = self.calls[name]
            if n <= fault.transient_failures:
                raise BackendError(503, f"{name}: service unavailable (injected)")
            slow = n <= fault.transient_failures + fault.slow_calls
            if slow and not fault.after_commit:
                time.sleep(fault.delay_s)
            out = fn(args, ctx)
            if slow and fault.after_commit:
                time.sleep(fault.delay_s)
            return fault.malformed(dict(out)) if fault.malformed else out
        return run


def build_registry(incident: dict, infra: InfraSimulator, scenario: Scenario) -> ToolRegistry:
    """The agent's tools, bound to one incident's fixtures and one simulator."""
    faults = _Faults(scenario.faults)

    def fixture_tool(impl):
        def fn(args, ctx):
            params = args.model_dump()
            out = impl(ae._fixtures_for(params["service"], incident), **params)
            if impl is ae._tool_search_logs and scenario.injected_log and \
                    ae._norm(params["service"]) == ae._norm(incident["service"]):
                out = {**out, "matches": out["matches"] + [scenario.injected_log]}
            return out
        return fn

    read = dict(permission=Permission.READ, timeout_s=scenario.read_timeout_s, retry=scenario.retry)
    write = dict(permission=Permission.WRITE, timeout_s=scenario.write_timeout_s, retry=scenario.retry,
                 idempotent=scenario.retry_writes)
    specs = [
        ToolSpec("search_logs", "Search a service's recent logs; empty keyword returns all lines",
                 fixture_tool(ae._tool_search_logs), LogQuery, LogsOut, **read),
        ToolSpec("get_recent_deploys", "Recent deploys of a service, with minutes_ago",
                 fixture_tool(ae._tool_get_recent_deploys), ServiceQuery, DeploysOut, **read),
        ToolSpec("get_metrics", "Current metrics for a service: error rate, latency vs SLO, replica health, saturation",
                 fixture_tool(ae._tool_get_metrics), ServiceQuery, MetricsOut, **read),
        ToolSpec("get_dependencies", "Status and owning team of a service's dependencies",
                 fixture_tool(ae._tool_get_dependencies), ServiceQuery, DependenciesOut, **read),
        ToolSpec("check_runbook", "The incident runbook mapping evidence to an action",
                 fixture_tool(ae._tool_check_runbook), ServiceQuery, RunbookOut, **read),
        ToolSpec("rollback_deploy", "Roll a service back from its current (bad) version to the previous one",
                 lambda a, ctx: infra.rollback(a.service, a.from_version, ctx.idempotency_key),
                 RollbackArgs, WriteReceipt, **write),
        ToolSpec("page_oncall", "Page a team's on-call engineer",
                 lambda a, ctx: infra.page(a.team, a.service, a.summary, ctx.idempotency_key),
                 PageArgs, WriteReceipt, **write),
        ToolSpec("restart_service", "Restart one replica or worker of a service",
                 lambda a, ctx: infra.restart(a.service, a.replica, ctx.idempotency_key),
                 RestartArgs, WriteReceipt, **write),
    ]
    return ToolRegistry([replace(s, fn=faults.wrap(s.name, s.fn)) for s in specs])


# ========== STATE & SCHEMAS ==========

class HardenedRCAReport(ae.RCAReport):
    action_target: str | None = Field(
        None, description="What the action applies to. rollback_deploy: the bad version to roll back from "
                          "(e.g. v2.14.3); page_oncall: the owning team to page (e.g. db-team); "
                          "restart_service: the stuck replica (e.g. worker-3); monitor: null")


class HardenedState(TypedDict, total=False):
    alert: str
    parsed: dict
    evidence: Annotated[list, operator.add]
    attempts: int
    verdict: str
    feedback: str
    report: dict
    proposal: dict
    approval: dict
    halted: str      # why the run stopped early (budget, contract, failed write); routes to escalate_human
    outcome: str


def proposed_args(report: dict, service: str) -> dict:
    """The write tool's arguments, from the RCA. Not validated here: that's
    `propose()`, against the tool's own input model."""
    action, target = report["next_action"], report.get("action_target")
    if action == "rollback_deploy":
        return {"service": service, "from_version": target}
    if action == "page_oncall":
        return {"team": target, "service": service, "summary": report["root_cause"][:300]}
    if action == "restart_service":
        return {"service": service, "replica": target}
    raise ValueError(f"{action} is not a write action")


def action_precondition(tool: str, args: dict, observations: dict) -> str | None:
    """Why the evidence doesn't support this write, or None. A deterministic
    check of the runbook's condition for the proposed action only — not a
    re-derivation of the whole decision. It's there because text the model
    reads (a log line, a ticket) can steer what it *recommends*: read-only
    scoping stops it writing directly, not proposing a bad write."""
    if tool == "rollback_deploy":
        deploy = next((d for d in observations["get_recent_deploys"]["deploys"]
                       if d["version"] == args["from_version"]), None)
        if deploy is None:
            return f"{args['from_version']} is not among the service's recent deploys"
        if deploy["minutes_ago"] > 30:
            return (f"{args['from_version']} was deployed {deploy['minutes_ago']} minutes ago; the runbook only "
                    f"rolls back a deploy from the last 30 minutes")
    if tool == "restart_service":
        healthy, _, total = observations["get_metrics"]["replicas_healthy"].partition("/")
        if int(healthy) >= int(total):
            return f"all replicas are healthy ({healthy}/{total}); nothing to restart"
        # A valid slug isn't necessarily a replica: the target has to be one the logs name.
        if not re.search(rf"\b{re.escape(args['replica'])}\b", "\n".join(observations["search_logs"]["matches"])):
            return f"{args['replica']!r} doesn't appear in the logs as a replica or worker"
    return None


_STUCK = re.compile(r"\b([a-z]+-\d+)\b[^\n]*(?:no heartbeat|stuck|crash)", re.I)


def _offline_target(action: str, observations: dict, service: str) -> str | None:
    if action == "rollback_deploy":
        deploys = observations["get_recent_deploys"]["deploys"]
        return deploys[0]["version"] if deploys else None
    if action == "page_oncall":
        unhealthy = [d for d in observations["get_dependencies"]["dependencies"] if d["status"] != "healthy"]
        return unhealthy[0]["owner"] if unhealthy else f"{service}-oncall"
    if action == "restart_service":
        m = next(filter(None, (_STUCK.search(line) for line in observations["search_logs"]["matches"])), None)
        return m.group(1) if m else None
    return None


def _offline_rca(observations: dict, severity: str) -> ae.RCAReport:
    """agent_eval's rule-based RCA, fed from the executor's validated outputs."""
    span = ae.NodeSpan(node="investigate", tool_calls=[
        ae.ToolCallRecord(name, {"service": out["service"]}, True, 0.0, result=out) for name, out in observations.items()])
    return ae._offline_rca(ae.RunTrace("offline", [span]), severity)


# ========== THE RUN ==========

@dataclass
class HardenedRun:
    incident_id: str
    state: dict
    budget: RunBudget
    audit: AuditLog
    infra: InfraSimulator
    langfuse_trace_id: str | None = None

    @property
    def outcome(self) -> str:
        return self.state.get("outcome", "")

    @property
    def action(self) -> str:
        report = self.state.get("report") or {}
        return report.get("next_action", "close") if not self.state.get("halted") else "escalated"


LIVE_SYSTEM = ("You investigate production incidents. Use your tools to gather evidence about the affected service, "
               "including its dependencies, and read the runbook. When done, reply with a short RCA draft: "
               f"root cause / evidence / recommended action (exactly one of: {', '.join(ae.ACTIONS)}), what it "
               "applies to (the bad version, the team to page, or the stuck replica) and which runbook rule applies. "
               "Only state what the tool results show.")


def build_graph(incident: dict, scenario: Scenario, budget: RunBudget, executor: ToolExecutor,
                observations: dict):
    """`observations` collects the latest successful output of each read
    tool, for the OFFLINE RCA and the pre-approval targeting check."""
    reads = executor.with_registry(executor.registry.scoped(Permission.READ))
    writes = executor.with_registry(executor.registry.scoped(Permission.WRITE))
    # max_retries=0 on the underlying client would be cleaner, but it's
    # agent_eval's shared client; the per-request timeout bounds each attempt.
    llm = None if OFFLINE else BudgetedChatClient(ae._live_client(), budget)
    targets: list[str] = []  # the `service` of every successful read, in order
    required = ("search_logs", "get_recent_deploys", "get_metrics", "get_dependencies")

    def missing_evidence() -> list[str]:
        return [t for t in required if t not in observations]

    def read(name: str, args, node: str = "investigate"):
        res = reads.call(name, args, node=node)
        if res.ok:
            observations[name] = res.output
            targets.append(res.output.get("service", ""))
        return res

    def work(name, fn):
        """Charge a graph step; turn a budget hit or any node failure into a
        `halted` reason (-> escalate_human) instead of a crashed run."""
        def node(state):
            try:
                budget.charge_graph_step()
                return fn(state)
            except BudgetExceeded as e:
                return {"halted": f"{name}: {e}"}
            except Exception as e:
                return {"halted": f"{name}: {type(e).__name__}: {str(e)[:200]}"}
        return node

    def triage(state):
        if OFFLINE:
            parsed = ae.ParsedAlert(service=ae._extract_service(state["alert"]),
                                    severity=ae._classify_severity(state["alert"]), symptom=state["alert"][:120])
        else:
            prompt = (f"Parse this production alert. `service` is the exact service name as written in the alert.\n"
                      f"{ae.SEVERITY_POLICY}\n\nAlert: {state['alert']}")
            parsed, *_ = ae.llm_structured_live(prompt, ae.ParsedAlert, client=llm)
            # Schema-valid isn't grounded: a live run returned service="service".
            # Re-ask once with the problem stated, then give up.
            if ae._norm(parsed.service) not in ae._norm(state["alert"]):
                parsed, *_ = ae.llm_structured_live(
                    f"{prompt}\n\nYour previous answer gave service={parsed.service!r}, which does not appear in "
                    f"the alert. Copy the service name exactly as the alert writes it.", ae.ParsedAlert, client=llm)
        if ae._norm(parsed.service) not in ae._norm(state["alert"]):
            return {"halted": f"triage: service {parsed.service!r} does not appear in the alert"}
        return {"parsed": parsed.model_dump()}

    def investigate(state):
        service = state["parsed"]["service"]
        if scenario.investigator == "runaway":
            while True:  # the bug: no stopping condition. The tool budget is the one it hits.
                read("get_metrics", {"service": service})
        if scenario.investigator == "obeys_injection" and not state.get("attempts"):
            read("search_logs", {"service": service})
            deploys = read("get_recent_deploys", {"service": service}).output or {"deploys": []}
            version = deploys["deploys"][0]["version"] if deploys["deploys"] else "v0.0.0"
            # Does what the injected line says. The READ-scoped registry refuses
            # it; the refusal is what a real model would get back.
            reads.call("rollback_deploy", {"service": service, "from_version": version}, node="investigate")
        task = f"Investigate {service}: {state['alert']}"
        if state.get("verdict") == "revise":
            task += f"\nYour previous RCA was rejected. Fix: {state['feedback']}"
        if OFFLINE:
            for tool in ("search_logs", "get_recent_deploys", "get_metrics", "get_dependencies", "check_runbook"):
                read(tool, {"service": service})
            draft = "\n".join(json.dumps(observations[t]) for t in observations)
        else:
            draft = _investigate_live(task, reads.registry.schemas(), read, llm)
        return {"evidence": [draft], "attempts": state.get("attempts", 0) + 1}

    def verify(state):
        draft = state["evidence"][-1]
        missing = missing_evidence()
        if unavailable := [t for t in missing if reads.circuit_open(t)]:
            return {"halted": f"verify: no usable evidence from {', '.join(unavailable)}, which keeps failing "
                              f"(circuit open)"}
        if missing:
            # Checked in code, before any judge: the model skipped a tool (or it
            # failed after retries). Send it back once more rather than escalate.
            return {"verdict": "revise", "feedback": f"No evidence yet from {', '.join(missing)}. "
                                                     f"Call {', '.join(missing)} for {state['parsed']['service']}."}
        if OFFLINE:
            ok = ae._norm(state["parsed"]["service"]) in ae._norm(draft)
            v = ae.Verdict(verdict="ok" if ok else "revise",
                           feedback="" if ok else "Evidence does not cover the affected service.")
        else:
            v, *_ = ae.llm_structured_live(
                f"ALERT: {state['alert']}\nAFFECTED SERVICE: {state['parsed']['service']}\n\nRCA DRAFT:\n{draft}\n\n"
                "'ok' only if every claim is backed by evidence about the affected service. Else 'revise' with feedback.",
                ae.Verdict, client=llm)
        return {"verdict": v.verdict, "feedback": v.feedback}

    def write_rca(state):
        service = state["parsed"]["service"]
        missing = missing_evidence()
        # An RCA choosing between rollback and page without, say, metrics is a
        # guess — in live mode too, where the model will happily write one.
        if missing:
            return {"halted": f"write_rca: no usable evidence from {', '.join(missing)}"}
        if OFFLINE:
            base = _offline_rca(observations, state["parsed"]["severity"])
            report = HardenedRCAReport(**base.model_dump(),
                                       action_target=_offline_target(base.next_action, observations, service))
        else:
            report, *_ = ae.llm_structured_live(
                f"Convert this investigation into a final RCA report. Severity is {state['parsed']['severity']}. "
                f"`next_action` must follow the runbook rule the investigation identified, and `action_target` "
                f"must name what it applies to.\n\n{state['evidence'][-1]}", HardenedRCAReport, client=llm)
        return {"report": report.model_dump()}

    def propose_action(state):
        report, service = state["report"], state["parsed"]["service"]
        # agent_eval's broken run showed an RCA built on the wrong service's
        # evidence reaching the approval gate. Checked here, before a human sees it.
        off = sorted({t for t in targets if ae._norm(t) != ae._norm(service)})
        if off or not targets:
            return {"halted": f"propose_action: evidence was gathered for {off or 'nothing'}, not {service}"}
        try:
            p = propose(writes.registry, executor.run_id, report["next_action"], proposed_args(report, service))
        except InvalidToolInput as e:
            return {"halted": f"propose_action: {report['next_action']} arguments fail the tool contract: {e}"}
        if (why := action_precondition(p.tool, p.args, observations)) is not None:
            return {"halted": f"propose_action: {p.tool} not supported by the evidence: {why}"}
        return {"proposal": asdict(p)}

    def human_gate(state):
        budget.suspend()  # the approver's time isn't the agent's budget
        decision = interrupt({"proposal": state["proposal"], "root_cause": state["report"]["root_cause"]})
        budget.resume()
        approval = decide(ProposedCall(**state["proposal"]), decision.get("approver", "unknown"),
                          bool(decision.get("approved")))
        return {"approval": asdict(approval)}

    def execute(state):
        p, a = state["proposal"], Approval(**state["approval"])
        res = writes.call(p["tool"], p["args"], node="execute", approval=a)
        if not res.ok:
            return {"halted": f"execute: {p['tool']} {res.record.outcome}: {res.error}"}
        return {"outcome": f"{res.record.outcome}: {res.output['detail']}"}

    def escalate_human(state):
        if state.get("halted"):
            return {"outcome": f"Escalated to a human: {state['halted']}"}
        return {"outcome": f"Escalated to a human: rejected by {state['approval']['approver']}"}

    def auto_close(state):
        if state.get("report"):
            return {"outcome": f"No write action ({state['report']['next_action']}) — closed."}
        return {"outcome": "Low severity — closed at triage."}

    def halted_or(route):
        return lambda s: "escalate_human" if s.get("halted") else route(s)

    b = StateGraph(HardenedState)
    for name, fn in [("triage", triage), ("investigate", investigate), ("verify", verify),
                     ("write_rca", write_rca), ("propose_action", propose_action), ("execute", execute)]:
        b.add_node(name, work(name, fn))
    for name, fn in [("human_gate", human_gate), ("escalate_human", escalate_human), ("auto_close", auto_close)]:
        b.add_node(name, fn)
    b.add_edge(START, "triage")
    b.add_conditional_edges("triage", halted_or(
        lambda s: "investigate" if s["parsed"]["severity"] in ("SEV1", "SEV2") else "auto_close"))
    b.add_conditional_edges("investigate", halted_or(lambda s: "verify"))
    b.add_conditional_edges("verify", halted_or(
        lambda s: "investigate" if s["verdict"] == "revise" and s["attempts"] < 3 else "write_rca"))
    b.add_conditional_edges("write_rca", halted_or(
        lambda s: "propose_action" if s["report"]["next_action"] in WRITE_ACTIONS else "auto_close"))
    b.add_conditional_edges("propose_action", halted_or(lambda s: "human_gate"))
    b.add_conditional_edges("human_gate", lambda s: "execute" if s["approval"]["approved"] else "escalate_human")
    b.add_conditional_edges("execute", halted_or(lambda s: END))
    b.add_edge("escalate_human", END)
    b.add_edge("auto_close", END)
    return b.compile(checkpointer=InMemorySaver())


def _investigate_live(task: str, schemas: list[dict], read, llm, max_iterations: int = 10) -> str:
    """agent_eval's tool-calling loop, with the executor doing what
    `_call_tool` and the inline JSON parsing did. Arguments go to the
    executor raw; its validation error goes back to the model."""
    messages = [{"role": "system", "content": LIVE_SYSTEM}, {"role": "user", "content": task}]
    for _ in range(max_iterations):
        resp = ae._chat(llm, model=settings.resolved_model, extra_body=ae.CHAT_EXTRA,
                        messages=messages, tools=schemas, tool_choice="auto")
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
            res = read(tc.function.name, tc.function.arguments or "{}")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(res.for_model())})
    return "Max iterations reached."


def _auto_approve(proposal: dict) -> tuple[bool, str]:
    return True, "eval-harness"


def run_once(incident: dict, thread_id: str, scenario: Scenario | None = None,
             approve: Callable[[dict], tuple[bool, str]] = _auto_approve,
             audit: AuditLog | None = None, sleep: Callable[[float], None] = time.sleep) -> HardenedRun:
    """Runs one incident to completion. `approve(proposal) -> (approved,
    approver)` stands in for the human at the gate; the default approves."""
    scenario = scenario or Scenario()
    budget = RunBudget(scenario.limits, settings.llm_usd_per_mtok_in, settings.llm_usd_per_mtok_out)
    infra = InfraSimulator(incident, honor_keys=scenario.backend_honors_keys)
    executor = ToolExecutor(build_registry(incident, infra, scenario), thread_id, budget,
                            audit if audit is not None else AuditLog(), sleep=sleep)
    app = build_graph(incident, scenario, budget, executor, observations={})
    # Graph steps are budgeted, so recursion_limit is only a backstop.
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 100}

    def invoke():
        state = app.invoke({"alert": incident["alert"], "attempts": 0, "evidence": []}, config)
        if state.get("__interrupt__"):
            approved, approver = approve(state["__interrupt__"][0].value["proposal"])
            state = app.invoke(Command(resume={"approved": approved, "approver": approver}), config)
        budget.suspend()  # the run is over: freeze elapsed_s for reporting
        return state

    lf = ae.langfuse_client()
    if lf is None:
        return HardenedRun(incident["id"], invoke(), budget, executor.audit, infra)

    from langfuse.langchain import CallbackHandler

    config["callbacks"] = [CallbackHandler()]
    with lf.start_as_current_observation(name=f"hardened:{thread_id}", as_type="span") as root:
        state = invoke()
        trace_id = lf.get_current_trace_id()
        root.update(input={"alert": incident["alert"]},
                    output={"outcome": state.get("outcome"), "halted": state.get("halted")},
                    metadata={"incident_id": incident["id"], "mode": "offline" if OFFLINE else "live",
                              "budget": budget.snapshot(),
                              "audit": [r.to_dict() for r in executor.audit.records]})
    return HardenedRun(incident["id"], state, budget, executor.audit, infra, trace_id)


# ========== INVARIANTS ==========

def check_invariants(run: HardenedRun) -> dict[str, bool]:
    """Properties that must hold for every run, whatever the model does:

    * every write that took effect had an approval covering its exact args;
    * no idempotency key produced more than one side effect;
    * at most one side effect per run (this agent takes one action);
    * no write was attempted from a READ-scoped step;
    * the run ended with an outcome (no crash) and within its budget, or
      stopped *because of* the budget and was escalated.
    """
    writes = [r for r in run.audit.records if r.permission == "write"]
    keys = Counter(e["idempotency_key"] for e in run.infra.effects)
    halted = run.state.get("halted") or ""
    return {
        "writes_approved": all(r.approval and r.approval["approved"] for r in writes if r.outcome in ("ok", "deduplicated")),
        "no_duplicate_effects": all(n == 1 for n in keys.values()) and len(run.infra.effects) <= 1,
        "no_write_from_read_scope": not [r for r in writes if r.node != "execute" and r.outcome != "denied"],
        "finished": bool(run.outcome),
        "within_budget_or_escalated": run.budget.exceeded is None or "budget exhausted" in halted,
    }
