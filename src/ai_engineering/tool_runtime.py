"""A hardened runtime for agent tool calls.

Generic, no incident-specific code: the hardened escalation agent in
agent_reliability.py is built on it, and src/fastapi_serve.py serves that
agent. Every tool call an agent makes
goes through `ToolExecutor.call()`, which applies, in order:

  1. Scope: the node's registry view decides which tools exist for it. A
     tool outside the view is `denied`, not `unknown` — the audit log should
     say "the investigator tried to roll back", not "typo".
  2. Input contract: arguments are validated against the tool's Pydantic
     input model (`extra="forbid"`, so invented arguments are rejected, not
     passed through to `**kwargs`). The validation message goes back to the
     model so it can fix the call.
  3. Approval (WRITE tools only): the call must carry an `Approval` for this
     exact tool, argument hash and idempotency key. Approving "rollback
     payments-api from v2.14.3" doesn't approve anything else.
  4. Budget: every attempt is charged to the run's `RunBudget` first.
  5. Timeout + bounded retry: each attempt runs with a timeout; retryable
     failures (timeouts, connection errors, 429, 5xx) are retried with
     exponential backoff and jitter. Non-retryable ones (validation,
     permission, 4xx, budget) fail immediately.
  6. Output contract: the result is validated against the tool's output
     model; a malformed result is a tool bug and never reaches the model.
  Circuit breaker: after `breaker_threshold` calls to one tool fail in a
     run (after their retries — timeouts, backend errors, malformed output),
     later calls to it fail fast as `circuit_open`. Invalid *input* doesn't
     count: that's the model's mistake to fix, not the tool being down.
  7. Audit: one `AuditRecord` per call — including denied and invalid ones —
     with args, idempotency key, approval, outcome, latency and retry count.

Timeouts and writes: a Python thread can't be killed, so a timed-out attempt
may still finish in the background. For a read that's wasted work; for a
write it means "may have happened". So a write tool is only retried when it's
marked `idempotent` — its backend honors the idempotency key and replays the
stored result for a repeated key instead of acting twice. A non-idempotent
write that times out is reported as `timeout` and left for a human.
"""
import contextvars
import hashlib
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from pydantic import BaseModel, ValidationError
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential_jitter


class Permission(str, Enum):
    READ = "read"
    WRITE = "write"


# ========== ERRORS & CLASSIFICATION ==========

class ToolError(Exception):
    """Base for failures the executor knows how to classify. `outcome` is the
    audit label; `retryable` decides whether another attempt could help."""
    outcome = "failed"
    retryable = False


class TransientToolError(ToolError):
    """The backend said "try again": 429, 5xx, connection reset."""
    retryable = True


class ToolTimeout(TransientToolError):
    outcome = "timeout"


class InvalidToolInput(ToolError):
    outcome = "invalid_input"


class InvalidToolOutput(ToolError):
    outcome = "invalid_output"


class PermissionDenied(ToolError):
    outcome = "denied"


class CircuitOpen(ToolError):
    """The tool already failed repeatedly in this run; fail fast instead of
    spending more budget (and LLM calls deciding to try it again) on it."""
    outcome = "circuit_open"


class BudgetExceeded(ToolError):
    """A run-level limit was hit. Raised out of the executor (not handed back
    to the model): the run has to stop, not try something else."""
    outcome = "budget_exceeded"

    def __init__(self, limit: str, used: float, cap: float):
        super().__init__(f"{limit} budget exhausted ({used:g}/{cap:g})")
        self.limit = limit


_RETRYABLE_STATUS = {408, 409, 425, 429}


def is_retryable(exc: BaseException) -> bool:
    """One classification for tool calls and LLM calls alike.

    Retryable: timeouts, connection errors, and HTTP 408/409/425/429/5xx
    (by `status_code`, which the OpenAI SDK's and httpx's errors carry).
    Everything else — validation, permission, other 4xx, budget, plain bugs
    — fails the same way on every attempt, so retrying only burns budget.
    """
    if isinstance(exc, ToolError):
        return exc.retryable
    if isinstance(exc, (TimeoutError, ConnectionError, FutureTimeout)):
        return True
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status in _RETRYABLE_STATUS or status >= 500
    try:
        import openai
    except ImportError:  # pragma: no cover - openai is a hard dependency of the repo
        return False
    return isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError))


def is_retryable_write(exc: BaseException) -> bool:
    """is_retryable, except a 409 Conflict. From a write API a 409 usually
    means the request conflicts with the current state (a concurrent change,
    a duplicate), which another attempt won't fix — unlike an LLM endpoint's
    409, which is congestion. The default for WRITE tools."""
    return getattr(exc, "status_code", None) != 409 and is_retryable(exc)


# ========== CONTRACTS: specs, registry, schemas ==========

@dataclass(frozen=True)
class RetryPolicy:
    """Bounded exponential backoff with jitter: waits of roughly
    initial_s, 2*initial_s, ... capped at max_s, plus up to jitter_s."""
    max_attempts: int = 3
    initial_s: float = 0.2
    max_s: float = 2.0
    jitter_s: float = 0.2


NO_RETRY = RetryPolicy(max_attempts=1)


@dataclass(frozen=True)
class ToolContext:
    """What a tool function gets besides its validated arguments."""
    run_id: str
    idempotency_key: str | None = None


@dataclass(frozen=True)
class ToolSpec:
    """`fn(args: input_model, ctx: ToolContext) -> dict` (or a BaseModel).

    `idempotent` asserts the backend honors `ctx.idempotency_key`: a repeated
    key returns the stored result without acting again, flagged as
    `deduplicated=True` in the output. Only then is a write safe to retry.

    `retryable` overrides the error classification for this tool; the
    default is is_retryable for READ tools and is_retryable_write for WRITE.
    """
    name: str
    description: str
    fn: Callable[[BaseModel, ToolContext], Any]
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    permission: Permission = Permission.READ
    timeout_s: float = 5.0
    retry: RetryPolicy = RetryPolicy()
    idempotent: bool = False
    retryable: Callable[[BaseException], bool] | None = None

    def classify(self, exc: BaseException) -> bool:
        if self.retryable is not None:
            return self.retryable(exc)
        return (is_retryable_write if self.permission is Permission.WRITE else is_retryable)(exc)

    def openai_schema(self) -> dict:
        """The function-calling schema, generated from the input model — the
        same model that validates the call, so the two can't drift."""
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": _strip_titles(self.input_model.model_json_schema())}}


def _strip_titles(schema):
    """Pydantic adds a `title` to every model and field; it's noise in the
    prompt. Drops the `title` keyword only, not a property named "title"."""
    if isinstance(schema, list):
        return [_strip_titles(s) for s in schema]
    if not isinstance(schema, dict):
        return schema
    out = {}
    for k, v in schema.items():
        if k == "title" and isinstance(v, str):
            continue
        if k == "properties" and isinstance(v, dict):
            out[k] = {name: _strip_titles(sub) for name, sub in v.items()}
        else:
            out[k] = _strip_titles(v)
    return out


class ToolRegistry:
    """All tools of an agent, plus scoped views of them. A view's
    `schemas()` only advertises the tools in scope, and `resolve()` tells
    "not in scope" (denied) apart from "doesn't exist" (invalid input)."""

    def __init__(self, specs, allowed: frozenset[Permission] = frozenset(Permission)):
        self._specs = {s.name: s for s in specs}
        if len(self._specs) != len(specs):
            raise ValueError("duplicate tool names")
        self.allowed = frozenset(allowed)

    def scoped(self, *permissions: Permission) -> "ToolRegistry":
        return ToolRegistry(list(self._specs.values()), frozenset(permissions))

    @property
    def names(self) -> set[str]:
        return {n for n, s in self._specs.items() if s.permission in self.allowed}

    def schemas(self) -> list[dict]:
        return [s.openai_schema() for s in self._specs.values() if s.permission in self.allowed]

    def lookup(self, name: str) -> ToolSpec | None:
        """The spec regardless of scope — for labeling a denied call."""
        return self._specs.get(name)

    def resolve(self, name: str) -> ToolSpec:
        spec = self._specs.get(name)
        if spec is None:
            raise InvalidToolInput(f"unknown tool {name!r}; available: {sorted(self.names)}")
        if spec.permission not in self.allowed:
            raise PermissionDenied(f"{name!r} is a {spec.permission.value} tool and this step may only use "
                                   f"{'/'.join(sorted(p.value for p in self.allowed))} tools")
        return spec


# ========== IDEMPOTENCY & APPROVAL ==========

def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def args_hash(tool: str, args: dict) -> str:
    return hashlib.sha256(canonical_json([tool, args]).encode()).hexdigest()[:16]


def idempotency_key(run_id: str, tool: str, args: dict) -> str:
    """Derived from the inputs, never random: a LangGraph resume or replay
    that re-executes the node recomputes the same key, so the backend sees a
    repeat, not a new request."""
    return hashlib.sha256(canonical_json([run_id, tool, args]).encode()).hexdigest()[:32]


@dataclass(frozen=True)
class ProposedCall:
    """A write the agent wants to make, with arguments already validated
    against the tool's input model — what the human actually approves."""
    tool: str
    args: dict
    args_hash: str
    idempotency_key: str


def propose(registry: ToolRegistry, run_id: str, tool: str, raw_args: dict) -> ProposedCall:
    """Validate a proposed write *before* it goes to the approval gate, so a
    human never approves arguments the tool would reject."""
    spec = registry.resolve(tool)
    args = _validate_input(spec, raw_args)
    return ProposedCall(tool, args, args_hash(tool, args), idempotency_key(run_id, tool, args))


@dataclass(frozen=True)
class Approval:
    approval_id: str
    approver: str
    approved: bool
    tool: str
    args_hash: str
    idempotency_key: str
    decided_at: str

    def covers(self, tool: str, args: dict, key: str) -> bool:
        return self.tool == tool and self.args_hash == args_hash(tool, args) and self.idempotency_key == key


def decide(proposal: ProposedCall, approver: str, approved: bool) -> Approval:
    """Record a human decision on one proposal. The approval is bound to the
    proposal's tool, argument hash and idempotency key."""
    return Approval(approval_id=uuid.uuid4().hex[:12], approver=approver, approved=approved,
                    tool=proposal.tool, args_hash=proposal.args_hash, idempotency_key=proposal.idempotency_key,
                    decided_at=_now())


# ========== BUDGETS ==========

@dataclass(frozen=True)
class BudgetLimits:
    """Per-run limits. `None` disables one. All are checked *before* the next
    unit of work, against what's already been spent, but they bound usage
    differently:

    * hard caps — LLM calls, tool calls, graph steps, elapsed time: final
      usage never exceeds them (time within a small scheduling grace, since
      each LLM attempt's deadline is the time left);
    * soft limits — tokens, cost: only known after a call returns, so the
      call that crosses the threshold completes and all further work is
      stopped. Final usage can exceed them by one call's usage; see
      RunBudget.soft_overshoot()."""
    max_llm_calls: int | None = 40
    max_tool_calls: int | None = 40
    max_tokens: int | None = 80_000
    max_cost_usd: float | None = None
    max_seconds: float | None = 900.0
    max_graph_steps: int | None = 25


class RunBudget:
    """Counts what a run has spent and raises BudgetExceeded when the next
    unit of work would go past a limit (see BudgetLimits for which limits
    are hard caps and which are soft). Cost is only computed (and only
    limited) when per-million-token prices are given."""

    def __init__(self, limits: BudgetLimits = BudgetLimits(), usd_per_mtok_in: float | None = None,
                 usd_per_mtok_out: float | None = None, clock: Callable[[], float] = time.monotonic):
        self.limits = limits
        self.usd_per_mtok_in = usd_per_mtok_in
        self.usd_per_mtok_out = usd_per_mtok_out
        self._clock = clock
        self.started = clock()
        self.llm_calls = self.tool_calls = self.graph_steps = 0
        self.prompt_tokens = self.completion_tokens = 0
        self.exceeded: str | None = None  # first limit hit, if any
        self._suspended_at: float | None = None
        # Called after every change to the counters or the clock, e.g. to
        # persist the budget so a crash doesn't hide what the run had spent.
        self.on_change: Callable[["RunBudget"], None] | None = None

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def priced(self) -> bool:
        return self.usd_per_mtok_in is not None and self.usd_per_mtok_out is not None

    @property
    def cost_usd(self) -> float | None:
        if not self.priced:
            return None
        return (self.prompt_tokens * self.usd_per_mtok_in + self.completion_tokens * self.usd_per_mtok_out) / 1e6

    @property
    def elapsed_s(self) -> float:
        now = self._suspended_at if self._suspended_at is not None else self._clock()
        return now - self.started

    def suspend(self) -> None:
        """Stop the clock, e.g. while the run waits at a human approval gate:
        the time budget is for the agent's work, not the approver's lunch.
        Idempotent, because LangGraph re-executes the interrupted node on resume."""
        if self._suspended_at is None:
            self._suspended_at = self._clock()
            self._changed()

    def resume(self) -> None:
        if self._suspended_at is not None:
            self.started += self._clock() - self._suspended_at
            self._suspended_at = None
            self._changed()

    def remaining_s(self) -> float | None:
        if self.limits.max_seconds is None:
            return None
        return max(0.0, self.limits.max_seconds - self.elapsed_s)

    def _check(self, limit: str, used: float, cap: float | None) -> None:
        if cap is not None and used > cap:
            self.exceeded = self.exceeded or limit
            raise BudgetExceeded(limit, used, cap)

    def _check_spent(self) -> None:
        self._check("time_s", round(self.elapsed_s, 1), self.limits.max_seconds)
        # Tokens/cost can only be counted after a call returns, so the check is
        # "already at or past the cap" -> +1 to reuse _check's strict `>`.
        if self.limits.max_tokens is not None and self.tokens >= self.limits.max_tokens:
            self._check("tokens", self.tokens + 1, self.limits.max_tokens)
        cost = self.cost_usd
        if cost is not None and self.limits.max_cost_usd is not None and cost >= self.limits.max_cost_usd:
            self._check("cost_usd", cost + 1e-9, self.limits.max_cost_usd)

    def charge_llm_call(self) -> None:
        self._check_spent()
        self._check("llm_calls", self.llm_calls + 1, self.limits.max_llm_calls)
        self.llm_calls += 1
        self._changed()

    def record_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens += prompt_tokens or 0
        self.completion_tokens += completion_tokens or 0
        self._changed()

    def charge_tool_call(self) -> None:
        self._check_spent()
        self._check("tool_calls", self.tool_calls + 1, self.limits.max_tool_calls)
        self.tool_calls += 1
        self._changed()

    def charge_graph_step(self) -> None:
        self._check_spent()
        self._check("graph_steps", self.graph_steps + 1, self.limits.max_graph_steps)
        self.graph_steps += 1
        self._changed()

    def _changed(self) -> None:
        if self.on_change is not None:
            self.on_change(self)

    def to_dict(self) -> dict:
        """What's been spent, for persisting. Elapsed time is stored as a
        duration (the clock is per process), plus whether the clock was
        stopped — e.g. at an approval gate."""
        return {"llm_calls": self.llm_calls, "tool_calls": self.tool_calls, "graph_steps": self.graph_steps,
                "prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
                "elapsed_s": self.elapsed_s, "suspended": self._suspended_at is not None, "exceeded": self.exceeded}

    @classmethod
    def from_dict(cls, data: dict, limits: "BudgetLimits" = None, usd_per_mtok_in: float | None = None,
                  usd_per_mtok_out: float | None = None, clock: Callable[[], float] = time.monotonic) -> "RunBudget":
        """Continue a persisted budget in this process. Time that passed while
        no process was running the run (a restart, a crash) isn't counted:
        the clock resumes from the stored elapsed time."""
        budget = cls(limits or BudgetLimits(), usd_per_mtok_in, usd_per_mtok_out, clock)
        for name in ("llm_calls", "tool_calls", "graph_steps", "prompt_tokens", "completion_tokens", "exceeded"):
            setattr(budget, name, data[name])
        now = clock()
        budget.started = now - data["elapsed_s"]
        if data["suspended"]:
            budget._suspended_at = now
        return budget

    def overruns(self, time_grace_s: float = 1.0) -> list[str]:
        """Limits the run's *final* usage is over, checked directly rather than
        trusting `exceeded` (which is only set when a later charge notices).
        Tokens and cost are excluded: they're only known after a call returns,
        so overshooting them by one call's usage is by design. `time_grace_s`
        allows for scheduling slack around an attempt's deadline."""
        lim = self.limits
        checks = [("llm_calls", self.llm_calls, lim.max_llm_calls), ("tool_calls", self.tool_calls, lim.max_tool_calls),
                  ("graph_steps", self.graph_steps, lim.max_graph_steps)]
        over = [name for name, used, cap in checks if cap is not None and used > cap]
        if lim.max_seconds is not None and self.elapsed_s > lim.max_seconds + time_grace_s:
            over.append("time_s")
        return over

    def soft_overshoot(self) -> dict[str, float]:
        """How far final usage went past the soft limits (tokens, cost), by
        design at most one call's usage. Empty when within them. Reported,
        not treated as a violation."""
        over = {}
        if self.limits.max_tokens is not None and self.tokens > self.limits.max_tokens:
            over["tokens"] = self.tokens - self.limits.max_tokens
        cost = self.cost_usd
        if cost is not None and self.limits.max_cost_usd is not None and cost > self.limits.max_cost_usd:
            over["cost_usd"] = round(cost - self.limits.max_cost_usd, 6)
        return over

    def snapshot(self) -> dict:
        lim = self.limits
        cost = self.cost_usd
        return {
            "llm_calls": f"{self.llm_calls}/{lim.max_llm_calls}",
            "tool_calls": f"{self.tool_calls}/{lim.max_tool_calls}",
            "tokens": f"{self.tokens}/{lim.max_tokens}",
            "cost_usd": (f"{cost:.4f}/{lim.max_cost_usd}" if cost is not None else "unpriced"),
            "elapsed_s": f"{self.elapsed_s:.1f}/{lim.max_seconds}",
            "graph_steps": f"{self.graph_steps}/{lim.max_graph_steps}",
            "exceeded": self.exceeded,
            "soft_overshoot": self.soft_overshoot() or None,
        }


class LLMUnavailable(ToolError):
    """Retryable LLM errors outlasted the retry policy. Deliberately *not* an
    OpenAI error type, so an outer retry wrapper keyed on those types doesn't
    start the whole cycle again."""
    outcome = "failed"


class BudgetedChatClient:
    """An OpenAI-compatible chat client whose every HTTP attempt is charged to
    the run budget and bounded by it. Duck-types `client.chat.completions.create`.

    Wrap a client built with `max_retries=0`: this class owns the retries, so
    each attempt is a charged LLM call and none happen out of sight. Per
    attempt:

      * the budget is charged first (LLM calls, and the time/tokens/cost
        already spent), raising BudgetExceeded when a limit is reached;
      * the timeout is min(call_timeout_s, time left in the run) and is set
        after the caller's kwargs, so a caller can't widen it;
      * the attempt runs on a worker thread and is abandoned at that
        deadline. httpx timeouts are per network operation, not per request,
        so they alone don't bound wall time.

    Retryable failures (is_retryable) back off with jitter, but never sleep
    past the run's deadline: that raises BudgetExceeded instead. The default
    policy (10 attempts, backoff capped at 60 s) can wait out several minutes
    of a congested endpoint, so the run's time budget, not the attempt count,
    is what usually ends it. Retries that
    run out raise LLMUnavailable. `retried` counts retried errors by HTTP
    status ("429", "503", "timeout", "connection")."""

    def __init__(self, client, budget: RunBudget, call_timeout_s: float = 60.0,
                 retry: RetryPolicy = RetryPolicy(max_attempts=10, initial_s=2.0, max_s=60.0, jitter_s=2.0),
                 sleep: Callable[[float], None] = time.sleep):
        self._client = client
        self.budget = budget
        self.call_timeout_s = call_timeout_s
        self.retry = retry
        self.retried: dict[str, int] = {}
        self._sleep = sleep
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _deadline_exceeded(self, extra_s: float = 0.0) -> BudgetExceeded:
        self.budget.exceeded = self.budget.exceeded or "time_s"
        return BudgetExceeded("time_s", round(self.budget.elapsed_s + extra_s, 1), self.budget.limits.max_seconds)

    def _attempt(self, kwargs: dict):
        self.budget.charge_llm_call()
        remaining = self.budget.remaining_s()
        timeout = self.call_timeout_s if remaining is None else min(self.call_timeout_s, remaining)
        if timeout <= 0:
            raise self._deadline_exceeded()
        # copy_context: tracing (Langfuse/OpenTelemetry) keeps its parent span on the worker thread.
        future = _LLM_POOL.submit(contextvars.copy_context().run, self._client.chat.completions.create,
                                  **{**kwargs, "timeout": timeout})
        try:
            return future.result(timeout=timeout)
        except FutureTimeout:
            raise ToolTimeout(f"LLM call did not return within {timeout:.2g}s") from None

    def _create(self, **kwargs):
        wait = wait_exponential_jitter(initial=self.retry.initial_s, max=self.retry.max_s, jitter=self.retry.jitter_s)
        for attempt in range(1, self.retry.max_attempts + 1):
            try:
                resp = self._attempt(kwargs)
            except BudgetExceeded:
                raise
            except Exception as e:
                if not is_retryable(e):
                    raise
                timed_out = isinstance(e, (TimeoutError, ToolTimeout)) or "Timeout" in type(e).__name__
                key = str(getattr(e, "status_code", None) or ("timeout" if timed_out else "connection"))
                self.retried[key] = self.retried.get(key, 0) + 1
                if attempt == self.retry.max_attempts:
                    raise LLMUnavailable(f"{type(e).__name__} after {attempt} attempts: {str(e)[:200]}") from e
                delay = wait(SimpleNamespace(attempt_number=attempt))
                remaining = self.budget.remaining_s()
                if remaining is not None and delay >= remaining:
                    raise self._deadline_exceeded(delay) from e
                self._sleep(delay)
                continue
            if getattr(resp, "usage", None):
                self.budget.record_usage(resp.usage.prompt_tokens, resp.usage.completion_tokens)
            return resp


# ========== AUDIT ==========

@dataclass
class AuditRecord:
    run_id: str
    ts: str
    node: str
    tool: str
    permission: str | None
    args: dict
    idempotency_key: str | None
    approval: dict | None
    # ok | deduplicated | failed | timeout | invalid_input | invalid_output | denied | circuit_open
    # | budget_exceeded
    outcome: str
    error_class: str | None
    error: str | None
    latency_ms: float
    attempts: int

    @property
    def retry_count(self) -> int:
        return max(0, self.attempts - 1)

    def to_dict(self) -> dict:
        return {**asdict(self), "retry_count": self.retry_count}


class AuditLog:
    """Append-only. In memory always; also one JSON line per record to
    `path` when given, written as each call finishes (not at the end of the
    run), so a crash doesn't lose the calls that led up to it."""

    def __init__(self, path: str | Path | None = None):
        self.records: list[AuditRecord] = []
        self.path = Path(path) if path else None

    def append(self, record: AuditRecord) -> None:
        self.records.append(record)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record.to_dict(), default=str) + "\n")

    def where(self, **match) -> list[AuditRecord]:
        return [r for r in self.records if all(getattr(r, k) == v for k, v in match.items())]

    def print_table(self) -> None:
        print(f"  {'node':12s} {'tool':18s} {'perm':5s} {'outcome':15s} {'tries':>5s} {'ms':>7s}  "
              f"{'key':10s} {'approval':22s} args / error")
        for r in self.records:
            appr = f"{'yes' if r.approval['approved'] else 'NO'} by {r.approval['approver']}" if r.approval else "-"
            detail = json.dumps(r.args)
            if r.error:
                detail += f"  ! {r.error_class}: {r.error[:90]}"
            print(f"  {r.node:12s} {r.tool:18s} {(r.permission or '-'):5s} {r.outcome:15s} {r.attempts:5d} "
                  f"{r.latency_ms:7.0f}  {(r.idempotency_key or '-')[:10]:10s} {appr[:22]:22s} {detail}")


# ========== EXECUTOR ==========

@dataclass
class ToolResult:
    ok: bool
    output: dict | None
    error: str | None
    record: AuditRecord

    def for_model(self) -> dict:
        """What goes back to the LLM as the tool message."""
        return self.output if self.ok else {"error": self.error}


# Worker pools for timeouts. A timed-out attempt can't be killed and keeps
# its worker until it returns. That's fine for a notebook or a batch eval,
# where a stuck call eventually hits its own network timeout. In a
# long-lived service, a tool that hangs *forever* leaks one worker per call
# until the pool is exhausted and every tool call queues behind it — use
# per-tool concurrency limits, or async tools with real cancellation, there.
_POOL = ThreadPoolExecutor(max_workers=16, thread_name_prefix="tool")
_LLM_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="llm")


def _validate_input(spec: ToolSpec, raw_args) -> dict:
    if isinstance(raw_args, str):
        try:
            raw_args = json.loads(raw_args or "{}")
        except json.JSONDecodeError as e:
            raise InvalidToolInput(f"arguments are not valid JSON: {e}") from None
    if not isinstance(raw_args, dict):
        raise InvalidToolInput("arguments must be a JSON object")
    try:
        return spec.input_model.model_validate(raw_args).model_dump(mode="json")
    except ValidationError as e:
        raise InvalidToolInput(_format_errors(e)) from None


def _format_errors(e: ValidationError) -> str:
    return "; ".join(f"{'.'.join(map(str, err['loc'])) or '(root)'}: {err['msg']}" for err in e.errors())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class ToolExecutor:
    """Runs tool calls for one agent run. Tool-level failures come back as a
    `ToolResult` with `ok=False` (hand the error to the model, keep going);
    `BudgetExceeded` is raised, because the run itself has to stop.

    `completed` is a client-side idempotency ledger: a write whose key has
    already succeeded in this run is answered from it without calling the
    backend. It's a cache that saves a round trip on replay — the guarantee
    has to come from the backend honoring the key, since this ledger dies
    with the process."""

    def __init__(self, registry: ToolRegistry, run_id: str, budget: RunBudget | None = None,
                 audit: AuditLog | None = None, sleep: Callable[[float], None] = time.sleep,
                 breaker_threshold: int | None = 2):
        self.registry = registry
        self.run_id = run_id
        self.budget = budget or RunBudget(BudgetLimits(None, None, None, None, None, None))
        self.audit = audit if audit is not None else AuditLog()
        self.completed: dict[str, dict] = {}
        self.failures: dict[str, int] = {}  # consecutive failed calls per tool, for the breaker
        self.breaker_threshold = breaker_threshold
        self._sleep = sleep
        # Called after every call's audit record, ledger and breaker count are
        # updated, e.g. to persist them.
        self.on_finish: Callable[["ToolExecutor"], None] | None = None

    def with_registry(self, registry: ToolRegistry) -> "ToolExecutor":
        """Same run, budget, audit log, ledger and breaker; different tool scope."""
        ex = ToolExecutor(registry, self.run_id, self.budget, self.audit, self._sleep, self.breaker_threshold)
        ex.completed = self.completed
        ex.failures = self.failures
        ex.on_finish = self.on_finish
        return ex

    def state(self) -> dict:
        """The idempotency ledger and breaker counts, for persisting."""
        return {"completed": self.completed, "failures": self.failures}

    def restore(self, state: dict) -> None:
        """Load persisted ledger and breaker counts (in place: scoped views
        made by with_registry share these dicts)."""
        self.completed.update(state.get("completed", {}))
        self.failures.update(state.get("failures", {}))

    def circuit_open(self, name: str) -> bool:
        return self.breaker_threshold is not None and self.failures.get(name, 0) >= self.breaker_threshold

    def call(self, name: str, raw_args, *, node: str, approval: Approval | None = None) -> ToolResult:
        start = time.perf_counter()
        spec: ToolSpec | None = None
        args: dict = raw_args if isinstance(raw_args, dict) else {"_raw": raw_args}
        key: str | None = None
        attempts = 0

        def finish(outcome: str, output: dict | None = None, exc: BaseException | None = None) -> ToolResult:
            rec = AuditRecord(
                run_id=self.run_id, ts=_now(), node=node, tool=name,
                permission=spec.permission.value if spec else None, args=args, idempotency_key=key,
                approval=asdict(approval) if approval else None, outcome=outcome,
                error_class=type(exc).__name__ if exc else None, error=str(exc) if exc else None,
                latency_ms=(time.perf_counter() - start) * 1000, attempts=attempts)
            self.audit.append(rec)
            ok = outcome in ("ok", "deduplicated")
            if ok:
                self.failures[name] = 0
            elif outcome in ("failed", "timeout", "invalid_output"):
                self.failures[name] = self.failures.get(name, 0) + 1
            if self.on_finish is not None:
                self.on_finish(self)
            return ToolResult(ok, output if ok else None, None if ok else rec.error, rec)

        try:
            spec = self.registry.lookup(name)
            self.registry.resolve(name)
            args = _validate_input(spec, raw_args)
            if spec.permission is Permission.WRITE:
                key = idempotency_key(self.run_id, name, args)
                self._check_approval(spec, args, key, approval)
                if key in self.completed:
                    return finish("deduplicated", {**self.completed[key], "deduplicated": True})
            if self.circuit_open(name):
                raise CircuitOpen(f"{name} failed {self.failures[name]} times in this run; not calling it again")
        except ToolError as e:
            return finish(e.outcome, exc=e)

        ctx = ToolContext(self.run_id, key)
        # A write is only retried when its backend dedupes on the key.
        can_retry = spec.permission is Permission.READ or spec.idempotent
        policy = spec.retry if can_retry else NO_RETRY

        def attempt():
            nonlocal attempts
            self.budget.charge_tool_call()  # BudgetExceeded is non-retryable: stops the loop
            attempts += 1
            return self._run_with_timeout(spec, args, ctx)

        retrying = Retrying(
            stop=stop_after_attempt(policy.max_attempts),
            wait=wait_exponential_jitter(initial=policy.initial_s, max=policy.max_s, jitter=policy.jitter_s),
            retry=retry_if_exception(spec.classify), sleep=self._sleep, reraise=True)
        try:
            raw_out = retrying(attempt)
        except BudgetExceeded as e:
            finish(e.outcome, exc=e)
            raise
        except ToolError as e:
            return finish(e.outcome, exc=e)
        except Exception as e:
            return finish("timeout" if isinstance(e, TimeoutError) else "failed", exc=e)

        try:
            output = spec.output_model.model_validate(
                raw_out.model_dump() if isinstance(raw_out, BaseModel) else raw_out).model_dump(mode="json")
        except ValidationError as e:
            return finish("invalid_output", exc=InvalidToolOutput(f"{name} returned malformed data: {_format_errors(e)}"))

        if key is not None:
            self.completed[key] = output
        return finish("deduplicated" if output.get("deduplicated") else "ok", output)

    def _check_approval(self, spec: ToolSpec, args: dict, key: str, approval: Approval | None) -> None:
        if approval is None:
            raise PermissionDenied(f"{spec.name!r} is a write tool and needs an explicit approval")
        if not approval.covers(spec.name, args, key):
            raise PermissionDenied(f"approval {approval.approval_id} does not cover {spec.name}({canonical_json(args)})")
        if not approval.approved:
            raise PermissionDenied(f"rejected by {approval.approver} (approval {approval.approval_id})")

    def _run_with_timeout(self, spec: ToolSpec, args: dict, ctx: ToolContext):
        timeout = spec.timeout_s
        remaining = self.budget.remaining_s()
        if remaining is not None:
            timeout = min(timeout, remaining)
        future = _POOL.submit(spec.fn, spec.input_model.model_validate(args), ctx)
        try:
            return future.result(timeout=timeout)
        except FutureTimeout:
            raise ToolTimeout(f"{spec.name} did not return within {timeout:.2g}s") from None
