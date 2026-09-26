"""The escalation agent served over FastAPI, on the hardened tool runtime.

Every alert runs the hardened graph from ai_engineering.agent_reliability:
Pydantic tool contracts, timeouts and bounded retries, read-only
investigation, evidence-checked actions, run budgets and an audit record per
tool call. The service adds the parts that only exist when a human answers
over HTTP:

  POST /alert            start a run; returns the proposal waiting at the
                         approval gate (tool, validated args, args_hash,
                         idempotency key), or the outcome if none is needed.
  POST /approve          decide on that proposal. `args_hash` must name the
                         proposal the approver reviewed, or it's a 409. The
                         same decision submitted twice returns the stored
                         result without acting again; a different second
                         decision is a 409.
  GET  /runs/{thread_id} status, outcome, budget and the run's audit log.
  POST /generate         a plain chat completion (not the agent).

Tools read a demo service catalog (each service's fixtures from the first
incident about it in data/incident_eval_set.json) and write to an in-memory
simulator. Runs live in process memory: a restart loses them, and nothing
expires them yet. Persisting runs with the LangGraph checkpoint is the next
step.
"""
import sys
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ai_engineering import agent_eval as ae
from ai_engineering import agent_reliability as ar
from ai_engineering.config import get_settings, make_chat_client
from ai_engineering.tool_runtime import AuditLog, RunBudget, ToolExecutor

# ══════════════════════════════════════════════════════════
# 1. THE DEMO BACKEND
# ══════════════════════════════════════════════════════════
CATALOG = ar.catalog_from_incidents(ae.load_incidents())

# The graph's structure, for introspection (README, tests). Each alert gets
# its own compiled graph, bound to that run's budget, executor and audit log.
_template_infra = ar.InfraSimulator.for_catalog(CATALOG)
app = ar.build_graph(ar.Scenario(), RunBudget(),
                     ToolExecutor(ar.catalog_registry(CATALOG, _template_infra, ar.Scenario()), "template"),
                     observations={})


class ServedRun:
    """A run plus what the service remembers about the human's decision."""

    def __init__(self, alert: str):
        self.thread_id = str(uuid.uuid4())
        self.infra = ar.InfraSimulator.for_catalog(CATALOG)
        self.run = ar.RunHandle(alert, self.thread_id, ar.catalog_registry(CATALOG, self.infra, ar.Scenario()),
                                self.infra, audit=AuditLog())
        self.lock = threading.Lock()  # one decision at a time per run
        self.decision: dict | None = None  # {"approved", "args_hash", "response"} once decided


RUNS: dict[str, ServedRun] = {}


def _summary(served: ServedRun) -> dict:
    state = served.run.state
    proposal = served.run.pending_proposal
    if proposal is not None:
        return {"status": "awaiting_approval", "thread_id": served.thread_id, "proposal": proposal,
                "root_cause": (state.get("report") or {}).get("root_cause")}
    return {"status": "escalated" if state.get("halted") or state.get("approval", {}).get("approved") is False
            else "completed",
            "thread_id": served.thread_id, "outcome": state.get("outcome"), "halted": state.get("halted")}


# ══════════════════════════════════════════════════════════
# 2. FASTAPI
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


@app_fastapi.post("/alert")
def receive_alert(payload: AlertPayload):
    served = ServedRun(payload.alert_text)
    RUNS[served.thread_id] = served
    try:
        served.run.start()
    except Exception as e:  # graph nodes already turn failures into escalations; this is a bug
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
    return _summary(served)


@app_fastapi.post("/approve")
def approve_action(payload: ApprovalPayload):
    served = RUNS.get(payload.thread_id)
    if served is None:
        raise HTTPException(status_code=404, detail="Thread not found")

    with served.lock:
        if served.decision is not None:
            same = (served.decision["approved"] == payload.approved
                    and served.decision["args_hash"] == payload.args_hash)
            if not same:
                raise HTTPException(status_code=409, detail="This run was already decided differently")
            # A retried submit (double click, client retry): same answer, no second action.
            return {**served.decision["response"], "replayed": True}

        proposal = served.run.pending_proposal
        if proposal is None:
            raise HTTPException(status_code=409, detail="This run has no proposal waiting for approval")
        if payload.args_hash != proposal["args_hash"]:
            raise HTTPException(status_code=409, detail={
                "error": "args_hash does not match the pending proposal; review it and approve that one",
                "proposal": proposal})

        served.run.resume(payload.approved, payload.approver)
        response = {**_summary(served), "approved_by": payload.approver if payload.approved else None}
        served.decision = {"approved": payload.approved, "args_hash": payload.args_hash, "response": response}
        return {**response, "replayed": False}


@app_fastapi.get("/runs/{thread_id}")
def get_run(thread_id: str):
    served = RUNS.get(thread_id)
    if served is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    return {**_summary(served), "budget": served.run.budget.snapshot(),
            "audit": [r.to_dict() for r in served.run.executor.audit.records],
            "side_effects": served.infra.effects}


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
