"""Offline tests for the served escalation agent (src/fastapi_serve.py).

conftest.py blanks every API key, so the hardened graph runs with the
rule-based stand-ins; the HTTP contract around the approval gate and the
durability of runs are what's under test here. A "restart" is a new Service
on the same SQLite file; "two processes" are two Services sharing one.
"""
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

import ai_engineering.agent_eval as ae
import ai_engineering.agent_reliability as ar
import src.fastapi_serve as serve
from ai_engineering.tool_runtime import RunBudget, ToolExecutor

INC = {i["id"]: i for i in ae.load_incidents()}


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def db(tmp_path):
    return tmp_path / "serve.sqlite3"


@pytest.fixture
def clock():
    return Clock()


def restart(db, clock):
    """A new process on the same database."""
    return serve.configure(db, clock=clock, approval_ttl_s=3600, retention_s=86400, lease_s=300)


@pytest.fixture
def client(db, clock):
    restart(db, clock)
    return TestClient(serve.app_fastapi)


def alert(client, inc_id="inc01"):
    resp = client.post("/alert", json={"alert_text": INC[inc_id]["alert"]})
    assert resp.status_code == 200, resp.text
    return resp.json()


def approve(client, run, approved=True, approver="alice", args_hash=None):
    return client.post("/approve", json={
        "thread_id": run["thread_id"], "approved": approved, "approver": approver,
        "args_hash": args_hash or run["proposal"]["args_hash"]})


def get(client, run):
    return client.get(f"/runs/{run['thread_id']}").json()


def side_effects(client, run):
    return get(client, run)["side_effects"]


# ---- the approval contract

def test_service_imports_without_an_api_key():
    assert ae.OFFLINE, "the served agent must run offline when no key is configured"


def test_alert_returns_the_validated_proposal(client):
    run = alert(client)
    assert run["status"] == "awaiting_approval"
    p = run["proposal"]
    assert p["tool"] == "rollback_deploy"
    assert p["args"] == {"service": "payments-api", "from_version": "v2.14.3"}
    assert len(p["args_hash"]) == 16 and len(p["idempotency_key"]) == 32
    assert not side_effects(client, run)  # nothing happens before approval


def test_approval_executes_exactly_the_proposal(client):
    run = alert(client)
    resp = approve(client, run)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed" and body["replayed"] is False and body["approved_by"] == "alice"
    assert "v2.14.3 -> v2.14.2" in body["outcome"]
    [write] = [r for r in get(client, run)["audit"] if r["permission"] == "write"]
    assert write["args"] == run["proposal"]["args"]
    assert write["approval"]["approver"] == "alice"
    assert write["approval"]["args_hash"] == run["proposal"]["args_hash"]


def test_approval_for_other_args_is_refused_and_the_run_stays_pending(client):
    run = alert(client)
    resp = approve(client, run, args_hash="0" * 16)
    assert resp.status_code == 409
    assert resp.json()["detail"]["proposal"] == run["proposal"]
    assert not side_effects(client, run)
    assert get(client, run)["status"] == "awaiting_approval"
    assert approve(client, run).status_code == 200  # the right hash still works


def test_same_decision_twice_replays_without_acting_again(client):
    run = alert(client)
    first, second = approve(client, run).json(), approve(client, run).json()
    assert second["replayed"] is True
    assert {k: v for k, v in second.items() if k != "replayed"} == {k: v for k, v in first.items() if k != "replayed"}
    assert len(side_effects(client, run)) == 1


def test_a_different_second_decision_is_refused(client):
    run = alert(client)
    approve(client, run, approved=False)
    resp = approve(client, run, approved=True)
    assert resp.status_code == 409 and resp.json()["detail"]["decided_by"] == "alice"
    assert not side_effects(client, run)


def test_another_approver_is_not_a_replay(client):
    run = alert(client)
    assert approve(client, run, approver="alice").json()["replayed"] is False
    bob = approve(client, run, approver="bob")
    assert bob.status_code == 409 and bob.json()["detail"] == {
        "error": "This run was already decided", "decided_by": "alice", "approved": True}
    assert approve(client, run, approver="alice").json()["replayed"] is True  # alice retrying is
    assert len(side_effects(client, run)) == 1


def test_concurrent_approvals_act_once(client):
    run = alert(client)
    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: approve(client, run), range(8)))
    assert all(r.status_code == 200 for r in responses)
    assert sum(not r.json()["replayed"] for r in responses) == 1
    assert len(side_effects(client, run)) == 1


def test_rejection_escalates_with_no_write(client):
    run = alert(client)
    body = approve(client, run, approved=False, approver="sre-lead").json()
    assert body["status"] == "escalated" and body["outcome"] == "Escalated to a human: rejected by sre-lead"
    assert body["approved_by"] is None and not side_effects(client, run)


def test_sev3_alert_closes_after_a_metrics_check_and_has_nothing_to_approve(client):
    run = alert(client, "inc03")
    assert run["status"] == "completed" and "metrics confirm no impact" in run["outcome"]
    resp = client.post("/approve", json={"thread_id": run["thread_id"], "approved": True,
                                         "approver": "alice", "args_hash": "x"})
    assert resp.status_code == 409 and "no proposal waiting" in resp.json()["detail"]


def test_unknown_service_escalates(client):
    run = client.post("/alert", json={"alert_text": "billing-api error rate spiking after deploy, customers affected"}).json()
    assert run["status"] == "escalated" and "billing-api" not in serve.CATALOG
    audit = get(client, run)["audit"]
    assert audit and all(r["outcome"] in ("failed", "circuit_open") for r in audit)
    assert "404" in audit[0]["error"] and audit[0]["attempts"] == 1  # a 404 isn't retried


def test_run_endpoint_reports_budget_and_audit(client):
    run = alert(client, "inc02")
    body = get(client, run)
    assert body["status"] == "awaiting_approval" and body["proposal"]["args"]["team"] == "db-team"
    assert body["budget"]["tool_calls"].startswith("5/")
    assert {r["tool"] for r in body["audit"]} == set(ae.TOOLS)
    assert client.get("/runs/nope").status_code == 404


def test_approval_payload_requires_the_args_hash(client):
    run = alert(client)
    resp = client.post("/approve", json={"thread_id": run["thread_id"], "approved": True, "approver": "alice"})
    assert resp.status_code == 422


def test_generate_without_a_key_is_503(client):
    assert client.post("/generate", json={"prompt": "hi"}).status_code == 503


# ---- the run lifecycle

def test_a_read_during_execution_says_executing_without_waiting(client, monkeypatch):
    """While an approval is being carried out, GET doesn't block and doesn't
    mix states: it reports `executing`, then the settled run."""
    run = alert(client)
    applied, release = threading.Event(), threading.Event()
    real_rollback = ar.InfraSimulator.rollback

    def slow_rollback(self, *args):
        result = real_rollback(self, *args)  # the side effect exists from here on
        applied.set()
        release.wait(timeout=3)
        return result

    monkeypatch.setattr(ar.InfraSimulator, "rollback", slow_rollback)
    approval = threading.Thread(target=approve, args=(client, run))
    approval.start()
    try:
        assert applied.wait(timeout=3)
        during = get(client, run)  # returns immediately
        assert during["status"] == "executing" and "proposal" not in during
        # The same approver retrying meanwhile: told it's executing; nothing re-runs.
        assert approve(client, run).json() == {"status": "executing", "thread_id": run["thread_id"], "replayed": True}
    finally:
        release.set()
        approval.join(timeout=3)
    after = get(client, run)
    assert after["status"] == "completed" and len(after["side_effects"]) == 1


def test_undecided_proposal_expires(client, clock):
    run = alert(client)
    clock.t += 3601
    assert get(client, run)["status"] == "expired"
    resp = approve(client, run)
    assert resp.status_code == 410 and resp.json()["detail"]["proposal"] == run["proposal"]
    assert not side_effects(client, run)


def test_settled_runs_are_purged_with_their_checkpoints(client, clock):
    done = alert(client, "inc03")
    pending = alert(client)
    clock.t += 86400 + 1
    client.post("/alert", json={"alert_text": INC["inc03"]["alert"]})  # housekeeping runs on each request
    assert client.get(f"/runs/{done['thread_id']}").status_code == 404
    checkpointer = serve.service().checkpointer
    assert checkpointer.get_tuple({"configurable": {"thread_id": done["thread_id"]}}) is None
    # An expired proposal stays visible for the retention period after it expired, then goes too.
    assert get(client, pending)["status"] == "expired"
    clock.t += 86400 + 1
    assert client.get(f"/runs/{pending['thread_id']}").status_code == 404


# ---- durability

def test_pending_approval_survives_a_restart(client, db, clock):
    run = alert(client)
    restart(db, clock)
    body = approve(client, run).json()
    assert body["status"] == "completed" and "v2.14.3 -> v2.14.2" in body["outcome"]
    after = get(client, run)
    assert after["budget"]["tool_calls"].startswith("6/")  # 5 before the restart, 1 after
    assert [r["node"] for r in after["audit"]] == ["investigate"] * 5 + ["execute"]
    assert len(after["side_effects"]) == 1


def test_two_processes_approving_at_once_act_once(client, db, clock):
    run = alert(client)
    a, b = restart(db, clock), restart(db, clock)
    barrier = threading.Barrier(2)

    def approve_via(svc):
        barrier.wait()
        return svc.approve(run["thread_id"], True, "alice", run["proposal"]["args_hash"])

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(approve_via, [a, b]))
    assert sorted(r["replayed"] for r in results) == [False, True]
    assert len(side_effects(client, run)) == 1


class SimulatedCrash(BaseException):
    """Escapes every `except Exception`, like the process dying."""


def test_crash_mid_write_recovers_without_a_second_rollback(client, db, clock, monkeypatch):
    """The rollback commits at the backend, then the process dies before the
    executor records it. After a restart the run is found silent, marked
    interrupted, and recovered from its checkpoint: execute re-runs, the write
    is re-sent with the same idempotency key, and the backend answers it as a
    duplicate."""
    run = alert(client)
    real_rollback = ar.InfraSimulator.rollback

    def rollback_then_crash(self, *args):
        real_rollback(self, *args)
        raise SimulatedCrash()

    monkeypatch.setattr(ar.InfraSimulator, "rollback", rollback_then_crash)
    with pytest.raises(SimulatedCrash):
        serve.service().approve(run["thread_id"], True, "alice", run["proposal"]["args_hash"])
    monkeypatch.setattr(ar.InfraSimulator, "rollback", real_rollback)

    restart(db, clock)
    assert get(client, run)["status"] == "executing"  # within the lease: presumed still running
    clock.t += 301
    stuck = get(client, run)
    assert stuck["status"] == "interrupted" and "/recover" in stuck["detail"]
    assert len(stuck["side_effects"]) == 1  # the rollback did happen

    recovered = client.post(f"/runs/{run['thread_id']}/recover").json()
    assert recovered["status"] == "completed" and recovered["approved_by"] == "alice"
    assert recovered["outcome"].startswith("deduplicated: v2.14.3 -> v2.14.2")
    after = get(client, run)
    assert len(after["side_effects"]) == 1  # not rolled back twice
    # Retrying the original approval now replays the recovered result.
    assert approve(client, run).json()["replayed"] is True
    assert client.post(f"/runs/{run['thread_id']}/recover").status_code == 409


def test_crash_mid_investigation_recovers_to_the_gate(client, db, clock, monkeypatch):
    """A crash before any write: recovery continues from the last checkpoint
    and stops at the approval gate as usual."""
    real_call, calls = ToolExecutor.call, {"n": 0}

    def crash_on_third_call(self, name, raw_args, **kw):
        calls["n"] += 1
        if calls["n"] == 3:
            raise SimulatedCrash()
        return real_call(self, name, raw_args, **kw)

    monkeypatch.setattr(ToolExecutor, "call", crash_on_third_call)
    with pytest.raises(SimulatedCrash):
        serve.service().alert(INC["inc01"]["alert"])
    monkeypatch.setattr(ToolExecutor, "call", real_call)
    [row] = serve.service().store._exec("SELECT thread_id FROM runs").fetchall()
    run = {"thread_id": row["thread_id"]}

    restart(db, clock)
    clock.t += 301
    assert get(client, run)["status"] == "interrupted"
    recovered = client.post(f"/runs/{run['thread_id']}/recover").json()
    assert recovered["status"] == "awaiting_approval"
    assert recovered["proposal"]["args"] == {"service": "payments-api", "from_version": "v2.14.3"}
    assert approve(client, {**run, "proposal": recovered["proposal"]}).json()["status"] == "completed"


def test_budget_round_trips_and_keeps_counting():
    budget = RunBudget(ar.DEFAULT_LIMITS)
    budget.charge_tool_call()
    budget.charge_graph_step()
    budget.suspend()
    restored = RunBudget.from_dict(budget.to_dict(), ar.DEFAULT_LIMITS)
    assert (restored.tool_calls, restored.graph_steps) == (1, 1)
    assert restored.to_dict()["suspended"] is True
    restored.resume()
    restored.charge_tool_call()
    assert restored.tool_calls == 2


def test_simulated_backend_round_trips_its_key_store():
    infra = ar.InfraSimulator(INC["inc01"])
    infra.rollback("payments-api", "v2.14.3", key="k1")
    restored = ar.InfraSimulator.from_dict(infra.to_dict())
    again = restored.rollback("payments-api", "v2.14.3", key="k1")
    assert again["deduplicated"] is True and len(restored.effects) == 1


# ---- recovery from every crash window, fencing, safe checkpoints

def test_crash_after_the_claim_before_resume_delivers_the_stored_decision(client, db, clock, monkeypatch):
    """The approval is claimed (decision stored, status executing) and the
    process dies before handing it to the graph: the checkpoint is still at
    the gate. Recovery must deliver the stored decision — not put the run
    back to awaiting_approval with a phantom decision that every later
    /approve would 'replay'."""
    run = alert(client)

    def crash(self, *args):
        raise SimulatedCrash()

    monkeypatch.setattr(ar.RunHandle, "resume", crash)
    with pytest.raises(SimulatedCrash):
        serve.service().approve(run["thread_id"], True, "alice", run["proposal"]["args_hash"])
    monkeypatch.undo()

    restart(db, clock)
    clock.t += 301
    assert get(client, run)["status"] == "interrupted"
    recovered = client.post(f"/runs/{run['thread_id']}/recover").json()
    assert recovered["status"] == "completed" and recovered["approved_by"] == "alice"
    assert "v2.14.3 -> v2.14.2" in recovered["outcome"]
    assert len(side_effects(client, run)) == 1
    assert approve(client, run).json() == {**recovered, "replayed": True}


def test_crash_after_a_rejection_claim_delivers_the_rejection(client, db, clock, monkeypatch):
    run = alert(client)
    monkeypatch.setattr(ar.RunHandle, "resume", lambda self, *a: (_ for _ in ()).throw(SimulatedCrash()))
    with pytest.raises(SimulatedCrash):
        serve.service().approve(run["thread_id"], False, "sre-lead", run["proposal"]["args_hash"])
    monkeypatch.undo()
    restart(db, clock)
    clock.t += 301
    recovered = client.post(f"/runs/{run['thread_id']}/recover").json()
    assert recovered["status"] == "escalated" and recovered["outcome"] == "Escalated to a human: rejected by sre-lead"
    assert not side_effects(client, run)


def test_crash_before_the_first_checkpoint_starts_again(client, db, clock, monkeypatch):
    """The run row exists but LangGraph never wrote a checkpoint: nothing to
    resume (invoke(None) would raise EmptyInputError), so recovery starts
    the run again from the stored alert."""
    monkeypatch.setattr(ar.RunHandle, "start", lambda self: (_ for _ in ()).throw(SimulatedCrash()))
    with pytest.raises(SimulatedCrash):
        serve.service().alert(INC["inc01"]["alert"])
    monkeypatch.undo()
    [row] = serve.service().store._rows("SELECT thread_id FROM runs")
    run = {"thread_id": row["thread_id"]}

    restart(db, clock)
    clock.t += 301
    assert serve.service().checkpointer.get_tuple({"configurable": {"thread_id": run["thread_id"]}}) is None
    recovered = client.post(f"/runs/{run['thread_id']}/recover").json()
    assert recovered["status"] == "awaiting_approval"
    assert recovered["proposal"]["args"] == {"service": "payments-api", "from_version": "v2.14.3"}


def test_a_stalled_worker_that_wakes_up_after_recovery_is_fenced_out(client, db, clock, monkeypatch):
    """Worker A stalls mid-write for longer than the lease. B finds the run
    silent, marks it interrupted and recovers it to completion. Then A wakes
    up and finishes its write: its claim was revoked, so nothing it does is
    persisted and it stops — B's result stands."""
    run = alert(client)
    a_blocked, a_release = threading.Event(), threading.Event()
    real_rollback, first = ar.InfraSimulator.rollback, {"call": True}

    def rollback(self, *args):
        if first.pop("call", False):  # worker A's attempt stalls before it commits
            a_blocked.set()
            a_release.wait(timeout=5)
        return real_rollback(self, *args)

    monkeypatch.setattr(ar.InfraSimulator, "rollback", rollback)
    a_result = {}

    def worker_a():
        try:
            serve.service().approve(run["thread_id"], True, "alice", run["proposal"]["args_hash"])
        except serve.ServiceError as e:
            a_result["error"] = (e.status_code, e.detail)

    worker = threading.Thread(target=worker_a)
    worker.start()
    assert a_blocked.wait(timeout=3)

    b = restart(db, clock)  # worker B, on the same database
    clock.t += 301
    recovered = b.recover(run["thread_id"])
    assert recovered["status"] == "completed" and "v2.14.3 -> v2.14.2" in recovered["outcome"]

    a_release.set()  # A wakes up and completes its write
    worker.join(timeout=5)
    assert a_result["error"][0] == 409 and "revoked" in a_result["error"][1]["error"]

    final = b.get(run["thread_id"])
    assert final["status"] == "completed"
    assert len(final["side_effects"]) == 1 and final["side_effects"][0]["detail"] == "v2.14.3 -> v2.14.2"
    assert [r["node"] for r in final["audit"]].count("execute") == 1  # A's write was never recorded
    snapshot = b.checkpointer.get_tuple({"configurable": {"thread_id": run["thread_id"]}})
    assert snapshot.checkpoint["channel_values"]["outcome"].startswith("ok: v2.14.3")  # B's, not overwritten


def test_a_revoked_epoch_can_not_write_checkpoints(db, clock):
    svc = restart(db, clock)
    svc.store.create("t1", "alert", clock())
    cp = serve.FencedCheckpointer(svc.db_path, svc.store, "t1", epoch=1)
    config = {"configurable": {"thread_id": "t1", "checkpoint_ns": ""}}
    from langgraph.checkpoint.base import empty_checkpoint
    cp.put(config, empty_checkpoint(), {}, {})  # epoch 1 is current
    assert svc.store.claim("t1", "starting", "executing", clock()) == 2  # someone else claims it
    with pytest.raises(serve.FencedOut):
        cp.put(config, empty_checkpoint(), {}, {})
    with pytest.raises(serve.FencedOut):
        cp.put_writes(config, [("x", 1)], "task")


def test_run_state_writes_require_the_current_epoch(db, clock):
    svc = restart(db, clock)
    svc.store.create("t1", "alert", clock())
    svc.store.update("t1", clock(), 1, budget={"n": 1})
    svc.store.mark_stale(clock() + 1000, lease_s=300)  # presumed dead: epoch 2
    for write in (lambda: svc.store.update("t1", clock(), 1, budget={"n": 2}),
                  lambda: svc.store.append_audit("t1", 1, {"tool": "x"}),
                  lambda: svc.store.settle("t1", 1, "interrupted", "completed", clock())):
        with pytest.raises(serve.FencedOut):
            write()
    assert svc.store.get("t1")["budget"] == {"n": 1} and svc.store.audit("t1") == []


def test_checkpoints_are_deserialized_strictly():
    import dataclasses
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    @dataclasses.dataclass
    class Payload:
        cmd: str

    blob = JsonPlusSerializer().dumps_typed(Payload("rm -rf /"))
    loaded = serve.strict_serde().loads_typed(blob)
    assert not isinstance(loaded, Payload) and loaded == {"cmd": "rm -rf /"}


def test_runs_checkpoint_synchronously(monkeypatch):
    seen = []
    real_invoke = type(serve.app).invoke

    def spy(self, *args, **kwargs):
        seen.append(kwargs.get("durability"))
        return real_invoke(self, *args, **kwargs)

    monkeypatch.setattr(type(serve.app), "invoke", spy)
    ar.run_once(INC["inc01"], "durability-check")
    assert seen and set(seen) == {"sync"}
