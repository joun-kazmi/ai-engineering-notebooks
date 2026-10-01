"""The escalation agent served over FastAPI, on the hardened tool runtime.

Every alert runs the hardened graph from ai_engineering.agent_reliability:
Pydantic tool contracts, timeouts and bounded retries, read-only
investigation, evidence-checked actions, run budgets and an audit record per
tool call. The service adds the parts that only exist when a human answers
over HTTP, and keeps runs durable across restarts. Every agent API endpoint
needs a Bearer token with its scope (FastAPI's own /docs, /redoc and
/openapi.json are not behind it):

  POST /alert                alerts:create. Start a run; returns the proposal
                             waiting at the approval gate (tool, args,
                             args_hash, idempotency key, and the evidence
                             code checked — no model prose), or the outcome
                             if none is needed.
  POST /approve              runs:approve. Decide on that proposal. `args_hash`
                             must name the proposal the approver reviewed, or
                             it's a 409. The same approver resubmitting the
                             same decision gets the stored result without
                             anything acting again; a different decision, or
                             another approver, is a 409. A proposal left
                             undecided past its TTL is 410 Gone: its evidence
                             is stale.
  GET  /runs/{id}            runs:read. Status, outcome, budget, audit log,
                             side effects. Never waits: a run being carried
                             out reads as `executing`.
  POST /runs/{id}/recover    runs:recover. Continue a run whose process died
                             mid-execution.

Authentication. The approver is whoever the token says, never the request
body: the verified principal (`iss|sub`, ai_engineering.auth) is the
decision's approver, so it's what replays and conflicts compare, what
`approved_by` and the audit record name. A missing or invalid token is 401
and a token without the endpoint's scope 403, before any run is touched or
traced. Then the approval policy (by default: approving needs group `sre`,
rejecting only the scope) is asked after the proposal checks and before the
claim, so a refusal is a 403 that leaves the run waiting. AUTH_MODE=disabled
serves every request as one local principal with every scope, for demos.

Durability. Graph checkpoints are written by LangGraph's SqliteSaver, and the
run store (ai_engineering.run_store) keeps each run's status, decision,
budget, idempotency ledger, breaker counts, simulated backend and audit log
in the same SQLite file, written as they change. So:

  * a run paused at the approval gate survives a restart and resumes from
    its checkpoint, with its budget and audit log intact;
  * an approval is claimed with a compare-and-set on the run's status, so
    across threads and processes exactly one request executes it;
  * a run whose process dies mid-execution is marked `interrupted` once it
    has been silent for RUN_LEASE_S (a running run writes on every budget
    charge), and /recover continues it from where it got to (Service.recover
    lists the cases). A write it had already made is sent again with the
    same idempotency key, and the backend answers it as a duplicate;
  * every claim, recovery and stale-marking bumps the run's epoch, and every
    write — run state, audit, graph checkpoints — requires the current one,
    so a worker that was only stalled can't overwrite a recovered run;
  * checkpoints are written synchronously and deserialized strictly.

Tracing. With LANGFUSE_* set, each run is one Langfuse trace, and every
request on it (alert, approve, recover) is a span inside it: executions hold
their LangGraph node tree and LLM generations; replays and rejections
(conflict, stale hash, expiry) are recorded too, rejections as WARNING,
crashes as ERROR (LangfuseTracer). Requests for unknown runs aren't traced,
nor are requests refused for their token. A span names its caller only by
principal id and the scope used — no email, no groups.

Tools read a demo service catalog (each service's fixtures from the first
incident about it in data/incident_eval_set.json) and write to a simulated
backend, one per run, persisted with the run.
"""
import logging
import sqlite3
import sys
import threading
import time
import uuid
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

from ai_engineering import agent_eval as ae
from ai_engineering import agent_reliability as ar
from ai_engineering.auth import (ApprovalPolicy, AuthUnavailable, Forbidden, Identity, InvalidToken,
                                  JwksKeyProvider, SreApprovalPolicy, TokenVerifier)
from ai_engineering.config import get_settings
from ai_engineering.run_store import FencedOut, RunStore
from ai_engineering.tool_runtime import AuditLog, RunBudget, ToolExecutor

# ══════════════════════════════════════════════════════════
# 1. THE DEMO BACKEND
# ══════════════════════════════════════════════════════════
CATALOG = ar.catalog_from_incidents(ae.load_incidents())
LIMITS = ar.DEFAULT_LIMITS

# The graph's structure, for introspection (README, tests). Each run gets its
# own compiled graph, bound to its budget, executor and audit log.
_template_infra = ar.InfraSimulator.for_catalog(CATALOG)
app = ar.build_graph(ar.Scenario(), RunBudget(),
                     ToolExecutor(ar.catalog_registry(CATALOG, _template_infra, ar.Scenario()), "template"),
                     observations={})


class ServiceError(Exception):
    def __init__(self, status_code: int, detail):
        super().__init__(detail)
        self.status_code, self.detail = status_code, detail


class StoredAuditLog(AuditLog):
    """Audit records go to the run store as each call finishes — only while
    this process still holds the run (its epoch)."""

    def __init__(self, store: RunStore, thread_id: str, epoch: int):
        super().__init__()
        self.store, self.thread_id, self.epoch = store, thread_id, epoch

    def append(self, record) -> None:
        super().append(record)
        self.store.append_audit(self.thread_id, self.epoch, record.to_dict())


def strict_serde() -> JsonPlusSerializer:
    """Checkpoint (de)serialization limited to LangGraph's safe built-in
    types: a tampered checkpoint database can't make the service construct
    arbitrary Python objects. Run state here is plain JSON-like data."""
    return JsonPlusSerializer(allowed_msgpack_modules=None)


class FencedCheckpointer(SqliteSaver):
    """One run's checkpoint writes, allowed only while its epoch is current.
    Fencing the run store alone isn't enough: a stalled worker that wakes up
    after its run was recovered elsewhere would otherwise still write graph
    checkpoints over the recovered run's.

    The epoch check and the write are one SQLite transaction. Every write
    SqliteSaver makes (put, put_writes, delete_thread) goes through
    `cursor(transaction=True)`; here that starts with BEGIN IMMEDIATE, which
    takes the database's write lock, then reads the epoch, then lets the
    write run and commits. SQLite admits one writer at a time across
    processes, so the epoch bump that revokes this run (itself a write)
    can't land between the check and the write — it waits for the commit,
    and every write after it fails the check."""

    def __init__(self, db_path: str, store: RunStore, thread_id: str, epoch: int):
        # Autocommit mode: this class issues BEGIN/COMMIT itself.
        conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None, timeout=30)
        super().__init__(conn, serde=strict_serde())
        self.run_thread_id, self.epoch = thread_id, epoch

    def _check_epoch(self, cur: sqlite3.Cursor) -> None:
        row = cur.execute("SELECT epoch FROM runs WHERE thread_id = ?", (self.run_thread_id,)).fetchone()
        if row is None or row[0] != self.epoch:
            raise FencedOut(f"run {self.run_thread_id}: epoch {self.epoch} is no longer current")

    @contextmanager
    def cursor(self, transaction: bool = True):
        if not transaction:  # reads aren't fenced
            with super().cursor(transaction=False) as cur:
                yield cur
            return
        with self.lock:
            self.setup()
            cur = self.conn.cursor()
            cur.execute("BEGIN IMMEDIATE")
            try:
                self._check_epoch(cur)
                yield cur
                cur.execute("COMMIT")
            except BaseException:
                cur.execute("ROLLBACK")
                raise
            finally:
                cur.close()


class NullTracer:
    """No tracing (Langfuse not configured)."""

    def span(self, thread_id: str, name: str, input=None, metadata=None):
        return nullcontext(None)

    def callbacks(self) -> list:
        return []

    def url(self, thread_id: str) -> str | None:
        return None


class LangfuseTracer:
    """One Langfuse trace per run, however many requests and processes it
    takes. The trace id is derived from the run's thread_id, so /alert,
    /approve and /recover — possibly in different processes, days apart —
    each add a span to the same trace without anything stored to link them.
    Inside a span, the LangGraph callback nests the node tree and the
    Langfuse-wrapped LLM client nests each generation."""

    def __init__(self, lf):
        self.lf = lf

    def trace_id(self, thread_id: str) -> str:
        return self.lf.create_trace_id(seed=thread_id)

    def span(self, thread_id: str, name: str, input=None, metadata=None):
        return self.lf.start_as_current_observation(trace_context={"trace_id": self.trace_id(thread_id)},
                                                    name=name, as_type="span", input=input, metadata=metadata)

    def callbacks(self) -> list:
        from langfuse.langchain import CallbackHandler

        return [CallbackHandler()]

    def url(self, thread_id: str) -> str | None:
        return self.lf.get_trace_url(trace_id=self.trace_id(thread_id))


def default_tracer():
    lf = ae.langfuse_client()
    return LangfuseTracer(lf) if lf is not None else NullTracer()


# ══════════════════════════════════════════════════════════
# 2. THE SERVICE
# ══════════════════════════════════════════════════════════
class Service:
    """Runs, approvals and recovery over a durable store. Several instances
    (processes) can share one database; `clock` is wall-clock time."""

    def __init__(self, db_path: str | Path, approval_ttl_s: float, retention_s: float, lease_s: float,
                 clock=time.time, tracer=None):
        self.db_path = str(db_path)
        self.store = RunStore(db_path)
        # For reading checkpoints and deleting purged ones; runs write through
        # their own FencedCheckpointer.
        self.checkpointer = SqliteSaver(sqlite3.connect(self.db_path, check_same_thread=False), serde=strict_serde())
        self.checkpointer.setup()
        self.approval_ttl_s, self.retention_s, self.lease_s = approval_ttl_s, retention_s, lease_s
        self.clock = clock
        self.tracer = tracer or default_tracer()

    # ---- building a run, new or from the store

    def _open(self, thread_id: str, alert: str, epoch: int, row: dict | None = None) -> ar.RunHandle:
        """A RunHandle whose state is written to the store as it changes,
        every write fenced by `epoch`. With `row`, it continues a stored run:
        budget, ledger, breaker, backend and graph checkpoint as they were."""
        settings = get_settings()
        infra = (ar.InfraSimulator.from_dict(row["infra"]) if row and row["infra"]
                 else ar.InfraSimulator.for_catalog(CATALOG))
        budget = (RunBudget.from_dict(row["budget"], LIMITS, settings.llm_usd_per_mtok_in,
                                      settings.llm_usd_per_mtok_out) if row and row["budget"] else None)
        run = ar.RunHandle(alert, thread_id, ar.catalog_registry(CATALOG, infra, ar.Scenario()), infra,
                           audit=StoredAuditLog(self.store, thread_id, epoch),
                           checkpointer=FencedCheckpointer(self.db_path, self.store, thread_id, epoch),
                           budget=budget, executor_state=row["executor"] if row else None)
        infra.on_change = lambda i: self.store.update(thread_id, self.clock(), epoch, infra=i.to_dict())
        run.budget.on_change = lambda b: self.store.update(thread_id, self.clock(), epoch, budget=b.to_dict())
        run.executor.on_finish = lambda ex: self.store.update(thread_id, self.clock(), epoch, executor=ex.state())
        self.store.update(thread_id, self.clock(), epoch, budget=run.budget.to_dict(), infra=infra.to_dict())
        if row is not None:
            run.load()
        return run

    def _settle(self, thread_id: str, run: ar.RunHandle, epoch: int, from_status: str,
                decision: dict | None = None) -> dict:
        """Record where the run stopped: at the gate, or finished."""
        now = self.clock()
        proposal = run.pending_proposal
        if proposal is not None:
            # The proposal carries the exact call and the facts code checked
            # (compile_action); the model's RCA prose is not part of it.
            self.store.settle(thread_id, epoch, from_status, "awaiting_approval", now, proposal=proposal,
                              expires_at=now + self.approval_ttl_s)
        else:
            state = run.state
            rejected = state.get("approval", {}).get("approved") is False
            status = "escalated" if state.get("halted") or rejected else "completed"
            response = {"status": status, "thread_id": thread_id, "outcome": state.get("outcome"),
                        "halted": state.get("halted")}
            if decision is not None:
                response["approved_by"] = decision["approver"] if decision["approved"] else None
            self.store.settle(thread_id, epoch, from_status, status, now, response=response)
        return self.summary(self.store.get(thread_id))

    # ---- housekeeping

    def housekeeping(self) -> None:
        now = self.clock()
        self.store.expire_pending(now)
        self.store.mark_stale(now, self.lease_s)
        for thread_id in self.store.purge(now - self.retention_s):
            self.checkpointer.delete_thread(thread_id)

    def _row(self, thread_id: str) -> dict:
        self.housekeeping()
        row = self.store.get(thread_id)
        if row is None:
            raise ServiceError(404, "Thread not found")
        return row

    @staticmethod
    def summary(row: dict) -> dict:
        status, thread_id = row["status"], row["thread_id"]
        if status == "awaiting_approval":
            return {"status": status, "thread_id": thread_id, "proposal": row["proposal"],
                    "expires_at": row["expires_at"]}
        if status in ("completed", "escalated"):
            return row["response"]
        if status == "expired":
            return {"status": status, "thread_id": thread_id, "proposal": row["proposal"]}
        if status == "interrupted":
            return {"status": status, "thread_id": thread_id,
                    "detail": f"the process running this run stopped; POST /runs/{thread_id}/recover to continue"}
        return {"status": status, "thread_id": thread_id}  # starting, executing

    # ---- the API

    @contextmanager
    def _traced(self, thread_id: str, name: str, request=None, trace_input=None, actor: dict | None = None):
        """A span for this request in the run's trace, with `actor` (who made
        the request: actor_id and the scope used) as its metadata. Yields `attach(run)`,
        which puts the run's LangGraph node tree under the span, and
        `record(result, run=None)`, which records what the request returned.
        A rejected request (ServiceError) is recorded as a WARNING with its
        status code; anything else that escapes, as an ERROR."""
        with self.tracer.span(thread_id, name, input=request, metadata=actor) as span:
            def attach(run: ar.RunHandle) -> ar.RunHandle:
                run.config["callbacks"] = self.tracer.callbacks()
                return run

            def record(result: dict, run: ar.RunHandle | None = None) -> dict:
                if span is not None:
                    span.update(output=result, metadata={"budget": run.budget.snapshot()} if run else None)
                    if run is not None:  # this request moved the run: its result is the trace's latest
                        span.set_trace_io(output=result)
                return result

            if span is not None and trace_input is not None:
                span.set_trace_io(input=trace_input)
            try:
                yield attach, record
            except ServiceError as e:
                if span is not None:
                    span.update(output={"status_code": e.status_code, "detail": e.detail}, level="WARNING",
                                status_message=f"rejected: {e.status_code}")
                raise
            except BaseException as e:
                if span is not None:
                    span.update(level="ERROR", status_message=f"{type(e).__name__}: {str(e)[:200]}")
                raise

    @staticmethod
    def _taken_over(thread_id: str) -> ServiceError:
        return ServiceError(409, {"error": "This process's claim on the run was revoked (it was presumed dead and "
                                           "the run marked interrupted or recovered elsewhere)",
                                  "thread_id": thread_id})

    def alert(self, alert_text: str, actor: dict | None = None) -> dict:
        self.housekeeping()
        thread_id = str(uuid.uuid4())
        self.store.create(thread_id, alert_text, self.clock())  # epoch 1
        with self._traced(thread_id, "served:alert", {"alert": alert_text}, trace_input={"alert": alert_text},
                          actor=actor) as (attach, record):
            try:
                run = attach(self._open(thread_id, alert_text, epoch=1))
                run.start()
                return record(self._settle(thread_id, run, 1, "starting"), run)
            except FencedOut:
                raise self._taken_over(thread_id) from None

    def approve(self, thread_id: str, approved: bool, args_hash: str, principal_id: str, authorize,
                actor: dict | None = None) -> dict:
        """Decide on the run's pending proposal as `principal_id`, a verified
        identity. `authorize(proposal, approved)` is the approval policy for
        that identity: it raises auth.Forbidden to refuse, and is asked only
        once the proposal checks pass, before the run is claimed."""
        row = self._row(thread_id)  # unknown runs: 404, no span (no trace for made-up ids)
        decision = {"approved": approved, "args_hash": args_hash, "approver": principal_id}
        # Every approval request on a run is a span in its trace — replays,
        # conflicts, stale hashes, expiries and policy refusals included: at
        # an approval boundary, five retries or another approver's conflict
        # are evidence.
        with self._traced(thread_id, "served:approve", request={"approved": approved, "args_hash": args_hash},
                          actor=actor or {"actor_id": principal_id}) as (attach, record):
            for _ in range(2):  # a lost claim re-reads once: the winner's decision is stored by then
                if row["decision"] is not None:
                    # A replay is the same principal resubmitting the same decision on
                    # the same proposal (double click, client retry): same answer, no
                    # second action.
                    if row["decision"] != decision:
                        raise ServiceError(409, {"error": "This run was already decided",
                                                 "decided_by": row["decision"]["approver"],
                                                 "approved": row["decision"]["approved"]})
                    return record({**self.summary(row), "replayed": True})
                if row["status"] == "expired":
                    raise ServiceError(410, {"error": "This proposal expired before it was decided; "
                                                      "its evidence is stale", "proposal": row["proposal"]})
                if row["status"] != "awaiting_approval":
                    raise ServiceError(409, "This run has no proposal waiting for approval")
                if args_hash != row["proposal"]["args_hash"]:
                    raise ServiceError(409, {"error": "args_hash does not match the pending proposal; "
                                                      "review it and approve that one", "proposal": row["proposal"]})
                try:
                    authorize(row["proposal"], approved)
                except Forbidden as e:
                    raise ServiceError(403, {"error": str(e)}) from None
                # Claim it. Across threads and processes, exactly one request wins.
                epoch = self.store.claim(thread_id, "awaiting_approval", "executing", self.clock(),
                                         decision=decision)
                if epoch is not None:
                    break
                row = self._row(thread_id)
            else:
                raise ServiceError(409, "Another request is deciding this run")
            try:
                run = attach(self._open(thread_id, row["alert"], epoch, self.store.get(thread_id)))
                run.resume(approved, principal_id)
                return record({**self._settle(thread_id, run, epoch, "executing", decision), "replayed": False}, run)
            except FencedOut:
                raise self._taken_over(thread_id) from None

    def recover(self, thread_id: str, actor: dict | None = None) -> dict:
        """Continue an interrupted run. Where to continue from depends on how
        far it got before its process died, which the stored decision and
        the graph checkpoint tell together:

          no checkpoint at all        crashed before LangGraph's first write:
                                      start again from the stored alert
          paused at the gate, decided crashed after the approval was claimed
                                      but before it was delivered: deliver
                                      the stored decision
          paused at the gate, no decision   it had reached the gate: back to
                                      awaiting_approval
          anywhere else               continue from the last checkpoint; a
                                      write it had made is re-sent with the
                                      same idempotency key and deduplicated
        """
        row = self._row(thread_id)
        with self._traced(thread_id, "served:recover", actor=actor) as (attach, record):
            if row["status"] != "interrupted":
                raise ServiceError(409, f"Only an interrupted run can be recovered; this one is {row['status']}")
            resuming = "executing" if row["decision"] else "starting"
            epoch = self.store.claim(thread_id, "interrupted", resuming, self.clock())
            if epoch is None:
                raise ServiceError(409, "Another request is already recovering this run")
            decision = row["decision"]
            try:
                run = attach(self._open(thread_id, row["alert"], epoch, self.store.get(thread_id)))
                if self.checkpointer.get_tuple(run.config) is None:
                    run.start()
                elif run.pending_proposal is not None:
                    if decision is not None:
                        run.resume(decision["approved"], decision["approver"])
                else:
                    run.recover()
                return record(self._settle(thread_id, run, epoch, resuming, decision), run)
            except FencedOut:
                raise self._taken_over(thread_id) from None

    def get(self, thread_id: str) -> dict:
        row = self._row(thread_id)
        budget = RunBudget.from_dict(row["budget"], LIMITS) if row["budget"] else RunBudget(LIMITS)
        return {**self.summary(row), "budget": budget.snapshot(), "audit": self.store.audit(thread_id),
                "side_effects": (row["infra"] or {}).get("effects", []), "trace_url": self.tracer.url(thread_id)}


_service: Service | None = None


def configure(db_path: str | Path | None = None, **overrides) -> Service:
    """(Re)create the service, e.g. on a given database in tests. Calling it
    again on the same path is what a restart looks like."""
    global _service
    s = get_settings()
    _service = Service(db_path or s.resolved_serve_db_path,
                       approval_ttl_s=overrides.get("approval_ttl_s", s.approval_ttl_s),
                       retention_s=overrides.get("retention_s", s.run_retention_s),
                       lease_s=overrides.get("lease_s", s.run_lease_s),
                       clock=overrides.get("clock", time.time), tracer=overrides.get("tracer"))
    return _service


def service() -> Service:
    return _service or configure()


# ══════════════════════════════════════════════════════════
# 3. AUTHENTICATION
# ══════════════════════════════════════════════════════════
log = logging.getLogger(__name__)

SCOPES = frozenset({"alerts:create", "runs:read", "runs:approve", "runs:recover"})
LOCAL_PRINCIPAL = "local|demo"


class AuthNotConfigured(RuntimeError):
    pass


@dataclass(frozen=True)
class AuthConfig:
    mode: str                           # "oidc" or "disabled"
    verifier: TokenVerifier | None      # None when disabled
    policy: ApprovalPolicy
    local_identity: Identity | None = None


_auth: AuthConfig | None = None
_auth_lock = threading.Lock()


def configure_auth(verifier: TokenVerifier | None = None, policy: ApprovalPolicy | None = None,
                   mode: str | None = None) -> AuthConfig:
    """(Re)create the auth config: a given verifier and policy (tests), or
    one built from AUTH_* settings. With AUTH_MODE=oidc (the default) a
    missing issuer, audience or JWKS URL raises, and every agent API request is refused
    until it's fixed."""
    global _auth
    s = get_settings()
    mode = mode or ("oidc" if verifier is not None else s.auth_mode)
    policy = policy or SreApprovalPolicy(s.auth_approver_group)
    if mode == "disabled":
        log.warning("AUTH_MODE=disabled: authentication is OFF. Every agent API request is %r with every scope "
                    "and group %r. Use only for local demos.", LOCAL_PRINCIPAL, s.auth_approver_group)
        _auth = AuthConfig(mode, None, policy, Identity(LOCAL_PRINCIPAL, SCOPES,
                                                        frozenset({s.auth_approver_group})))
        return _auth
    if mode != "oidc":
        raise AuthNotConfigured(f"unknown AUTH_MODE {mode!r}")
    if verifier is None:
        missing = [name for name, value in (("AUTH_ISSUER", s.auth_issuer), ("AUTH_AUDIENCE", s.auth_audience),
                                            ("AUTH_JWKS_URL", s.auth_jwks_url)) if not value]
        if missing:
            raise AuthNotConfigured(f"AUTH_MODE=oidc needs {', '.join(missing)} "
                                    "(or AUTH_MODE=disabled for a local demo)")
        verifier = TokenVerifier(s.auth_issuer, s.auth_audience, JwksKeyProvider(s.auth_jwks_url),
                                 algorithms=s.resolved_auth_algorithms, scope_claim=s.auth_scope_claim,
                                 groups_claim=s.auth_groups_claim)
    _auth = AuthConfig(mode, verifier, policy)
    return _auth


def reset_auth() -> None:
    """Forget the auth config; the next request builds it from settings."""
    global _auth
    _auth = None


def auth_config() -> AuthConfig:
    with _auth_lock:
        return _auth or configure_auth()


@dataclass(frozen=True)
class Caller:
    identity: Identity
    scope: str

    @property
    def actor(self) -> dict:
        """What a span records about the caller: no email, no groups."""
        return {"actor_id": self.identity.principal_id, "scope": self.scope}


_bearer = HTTPBearer(auto_error=False)


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(status_code=401, detail=detail, headers={"WWW-Authenticate": "Bearer"})


def require(scope: str):
    """A dependency: the verified caller, who must hold `scope`. Refusals
    here happen before any run is read or traced."""
    def dependency(credentials: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> Caller:
        try:
            auth = auth_config()
        except AuthNotConfigured as e:
            log.error("refusing request: %s", e)
            raise HTTPException(status_code=503, detail="Authentication is not configured") from None
        if auth.verifier is None:
            identity = auth.local_identity
        elif credentials is None:
            raise _unauthorized("Missing bearer token")
        else:
            try:
                identity = auth.verifier.verify(credentials.credentials)
            except InvalidToken as e:
                log.info("rejected token: %s", e)
                raise _unauthorized("Invalid token") from None
            except AuthUnavailable as e:
                log.error("cannot verify tokens: %s", e)
                raise HTTPException(status_code=503, detail="Token verification unavailable") from None
        if scope not in identity.scopes:
            raise HTTPException(status_code=403, detail=f"Token lacks scope {scope}")
        return Caller(identity, scope)

    return dependency


# ══════════════════════════════════════════════════════════
# 4. FASTAPI
# ══════════════════════════════════════════════════════════
app_fastapi = FastAPI(title="Escalation Agent API")


class AlertPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alert_text: str = Field(min_length=1, max_length=2000)


class ApprovalPayload(BaseModel):
    # No `approver`: it's the token's principal. A client still sending one
    # gets a 422 rather than believing it was used.
    model_config = ConfigDict(extra="forbid")

    thread_id: str
    approved: bool
    # The proposal the approver reviewed. Binds the decision to exact
    # arguments: approving an earlier or different proposal is refused.
    args_hash: str = Field(min_length=1)


def _call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except ServiceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)


@app_fastapi.post("/alert")
def receive_alert(payload: AlertPayload, caller: Caller = Depends(require("alerts:create"))):
    return _call(service().alert, payload.alert_text, actor=caller.actor)


@app_fastapi.post("/approve")
def approve_action(payload: ApprovalPayload, caller: Caller = Depends(require("runs:approve"))):
    authorize = partial(auth_config().policy.authorize, caller.identity)
    return _call(service().approve, payload.thread_id, payload.approved, payload.args_hash,
                 caller.identity.principal_id, authorize, actor=caller.actor)


@app_fastapi.get("/runs/{thread_id}")
def get_run(thread_id: str, caller: Caller = Depends(require("runs:read"))):
    return _call(service().get, thread_id)


@app_fastapi.post("/runs/{thread_id}/recover")
def recover_run(thread_id: str, caller: Caller = Depends(require("runs:recover"))):
    return _call(service().recover, thread_id, actor=caller.actor)
