"""The escalation agent served over FastAPI, on the hardened tool runtime.

Every alert runs the hardened graph from ai_engineering.agent_reliability:
Pydantic tool contracts, timeouts and bounded retries, read-only
investigation, evidence-checked actions, run budgets and an audit record per
tool call. The service adds the parts that only exist when a human answers
over HTTP, and keeps runs durable across restarts:

  POST /alert                start a run; returns the proposal waiting at the
                             approval gate (tool, validated args, args_hash,
                             idempotency key), or the outcome if none is needed.
  POST /approve              decide on that proposal. `args_hash` must name the
                             proposal the approver reviewed, or it's a 409. The
                             same approver resubmitting the same decision gets
                             the stored result without anything acting again;
                             a different decision, or another approver, is a
                             409. A proposal left undecided past its TTL is
                             410 Gone: its evidence is stale.
  GET  /runs/{id}            status, outcome, budget, audit log, side effects.
                             Never waits: a run being carried out reads as
                             `executing`.
  POST /runs/{id}/recover    continue a run whose process died mid-execution.
  POST /generate             a plain chat completion (not the agent).

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

Tools read a demo service catalog (each service's fixtures from the first
incident about it in data/incident_eval_set.json) and write to a simulated
backend, one per run, persisted with the run.
"""
import sqlite3
import sys
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

from ai_engineering import agent_eval as ae
from ai_engineering import agent_reliability as ar
from ai_engineering.config import get_settings, make_chat_client
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
    checkpoints over the recovered run's. (Check-then-write, so a write that
    races the revocation itself can land; everything after it can't.)"""

    def __init__(self, db_path: str, store: RunStore, thread_id: str, epoch: int):
        super().__init__(sqlite3.connect(db_path, check_same_thread=False), serde=strict_serde())
        self._fence = lambda: store.check_epoch(thread_id, epoch)

    def put(self, *args, **kwargs):
        self._fence()
        return super().put(*args, **kwargs)

    def put_writes(self, *args, **kwargs):
        self._fence()
        return super().put_writes(*args, **kwargs)


# ══════════════════════════════════════════════════════════
# 2. THE SERVICE
# ══════════════════════════════════════════════════════════
class Service:
    """Runs, approvals and recovery over a durable store. Several instances
    (processes) can share one database; `clock` is wall-clock time."""

    def __init__(self, db_path: str | Path, approval_ttl_s: float, retention_s: float, lease_s: float,
                 clock=time.time):
        self.db_path = str(db_path)
        self.store = RunStore(db_path)
        # For reading checkpoints and deleting purged ones; runs write through
        # their own FencedCheckpointer.
        self.checkpointer = SqliteSaver(sqlite3.connect(self.db_path, check_same_thread=False), serde=strict_serde())
        self.checkpointer.setup()
        self.approval_ttl_s, self.retention_s, self.lease_s = approval_ttl_s, retention_s, lease_s
        self.clock = clock

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
            self.store.settle(thread_id, epoch, from_status, "awaiting_approval", now, proposal=proposal,
                              root_cause=(run.state.get("report") or {}).get("root_cause"),
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
                    "root_cause": row["root_cause"], "expires_at": row["expires_at"]}
        if status in ("completed", "escalated"):
            return row["response"]
        if status == "expired":
            return {"status": status, "thread_id": thread_id, "proposal": row["proposal"]}
        if status == "interrupted":
            return {"status": status, "thread_id": thread_id,
                    "detail": f"the process running this run stopped; POST /runs/{thread_id}/recover to continue"}
        return {"status": status, "thread_id": thread_id}  # starting, executing

    # ---- the API

    @staticmethod
    def _taken_over(thread_id: str) -> ServiceError:
        return ServiceError(409, {"error": "This process's claim on the run was revoked (it was presumed dead and "
                                           "the run marked interrupted or recovered elsewhere)",
                                  "thread_id": thread_id})

    def alert(self, alert_text: str) -> dict:
        self.housekeeping()
        thread_id = str(uuid.uuid4())
        self.store.create(thread_id, alert_text, self.clock())  # epoch 1
        try:
            run = self._open(thread_id, alert_text, epoch=1)
            run.start()
            return self._settle(thread_id, run, 1, "starting")
        except FencedOut:
            raise self._taken_over(thread_id) from None

    def approve(self, thread_id: str, approved: bool, approver: str, args_hash: str) -> dict:
        row = self._row(thread_id)
        decision = {"approved": approved, "args_hash": args_hash, "approver": approver}

        if row["decision"] is not None:
            # A replay is the same approver resubmitting the same decision on
            # the same proposal (double click, client retry): same answer, no
            # second action. `approver` is caller-supplied for now; once there's
            # auth it should come from the authenticated identity.
            if row["decision"] != decision:
                raise ServiceError(409, {"error": "This run was already decided",
                                         "decided_by": row["decision"]["approver"],
                                         "approved": row["decision"]["approved"]})
            return {**self.summary(row), "replayed": True}
        if row["status"] == "expired":
            raise ServiceError(410, {"error": "This proposal expired before it was decided; its evidence is stale",
                                     "proposal": row["proposal"]})
        if row["status"] != "awaiting_approval":
            raise ServiceError(409, "This run has no proposal waiting for approval")
        if args_hash != row["proposal"]["args_hash"]:
            raise ServiceError(409, {"error": "args_hash does not match the pending proposal; "
                                              "review it and approve that one", "proposal": row["proposal"]})

        # Claim it. Across threads and processes, exactly one request wins; a
        # loser re-reads the row and gets the replay/409 answer above.
        epoch = self.store.claim(thread_id, "awaiting_approval", "executing", self.clock(), decision=decision)
        if epoch is None:
            return self.approve(thread_id, approved, approver, args_hash)
        try:
            run = self._open(thread_id, row["alert"], epoch, self.store.get(thread_id))
            run.resume(approved, approver)
            return {**self._settle(thread_id, run, epoch, "executing", decision), "replayed": False}
        except FencedOut:
            raise self._taken_over(thread_id) from None

    def recover(self, thread_id: str) -> dict:
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
        if row["status"] != "interrupted":
            raise ServiceError(409, f"Only an interrupted run can be recovered; this one is {row['status']}")
        resuming = "executing" if row["decision"] else "starting"
        epoch = self.store.claim(thread_id, "interrupted", resuming, self.clock())
        if epoch is None:
            raise ServiceError(409, "Another request is already recovering this run")
        decision = row["decision"]
        try:
            run = self._open(thread_id, row["alert"], epoch, self.store.get(thread_id))
            if self.checkpointer.get_tuple(run.config) is None:
                run.start()
            elif run.pending_proposal is not None:
                if decision is not None:
                    run.resume(decision["approved"], decision["approver"])
            else:
                run.recover()
            return self._settle(thread_id, run, epoch, resuming, decision)
        except FencedOut:
            raise self._taken_over(thread_id) from None

    def get(self, thread_id: str) -> dict:
        row = self._row(thread_id)
        budget = RunBudget.from_dict(row["budget"], LIMITS) if row["budget"] else RunBudget(LIMITS)
        return {**self.summary(row), "budget": budget.snapshot(), "audit": self.store.audit(thread_id),
                "side_effects": (row["infra"] or {}).get("effects", [])}


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
                       clock=overrides.get("clock", time.time))
    return _service


def service() -> Service:
    return _service or configure()


# ══════════════════════════════════════════════════════════
# 3. FASTAPI
# ══════════════════════════════════════════════════════════
app_fastapi = FastAPI(title="Escalation Agent API")


class AlertPayload(BaseModel):
    alert_text: str = Field(min_length=1, max_length=2000)


class ApprovalPayload(BaseModel):
    thread_id: str
    approved: bool
    approver: str = Field(min_length=1, max_length=100)
    # The proposal the approver reviewed. Binds the decision to exact
    # arguments: approving an earlier or different proposal is refused.
    args_hash: str = Field(min_length=1)


class GenerateRequest(BaseModel):
    prompt: str


def _call(fn, *args):
    try:
        return fn(*args)
    except ServiceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)


@app_fastapi.post("/alert")
def receive_alert(payload: AlertPayload):
    return _call(service().alert, payload.alert_text)


@app_fastapi.post("/approve")
def approve_action(payload: ApprovalPayload):
    return _call(service().approve, payload.thread_id, payload.approved, payload.approver, payload.args_hash)


@app_fastapi.get("/runs/{thread_id}")
def get_run(thread_id: str):
    return _call(service().get, thread_id)


@app_fastapi.post("/runs/{thread_id}/recover")
def recover_run(thread_id: str):
    return _call(service().recover, thread_id)


_llm_client = None


@app_fastapi.post("/generate")
def generate(req: GenerateRequest):
    # Created on first use, so the service starts (and serves the agent in
    # offline mode) without an API key.
    global _llm_client
    settings = get_settings()
    if not settings.has_llm_credentials:
        raise HTTPException(status_code=503, detail="No API key configured for LLM_PROVIDER")
    if _llm_client is None:
        _llm_client = make_chat_client(settings)
    response = _llm_client.chat.completions.create(
        model=settings.resolved_model,
        extra_body=settings.adapter.chat_extra_body(),
        messages=[{"role": "user", "content": req.prompt}],
        max_tokens=200,
        temperature=0.3,
    )
    return {"response": response.choices[0].message.content}
