"""Offline tests for the served escalation agent (src/fastapi_serve.py).

conftest.py blanks every API key, so the hardened graph runs with the
rule-based stand-ins; the HTTP contract around the approval gate is what's
under test here.
"""
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

import ai_engineering.agent_eval as ae
import src.fastapi_serve as serve

INC = {i["id"]: i for i in ae.load_incidents()}


@pytest.fixture
def client():
    serve.RUNS.clear()
    return TestClient(serve.app_fastapi)


def alert(client, inc_id="inc01"):
    resp = client.post("/alert", json={"alert_text": INC[inc_id]["alert"]})
    assert resp.status_code == 200, resp.text
    return resp.json()


def approve(client, run, approved=True, approver="alice", args_hash=None):
    return client.post("/approve", json={
        "thread_id": run["thread_id"], "approved": approved, "approver": approver,
        "args_hash": args_hash or run["proposal"]["args_hash"]})


def side_effects(run):
    return serve.RUNS[run["thread_id"]].infra.effects


def test_service_imports_without_an_api_key():
    assert ae.OFFLINE, "the served agent must run offline when no key is configured"


def test_alert_returns_the_validated_proposal(client):
    run = alert(client)
    assert run["status"] == "awaiting_approval"
    p = run["proposal"]
    assert p["tool"] == "rollback_deploy"
    assert p["args"] == {"service": "payments-api", "from_version": "v2.14.3"}
    assert len(p["args_hash"]) == 16 and len(p["idempotency_key"]) == 32
    assert not side_effects(run)  # nothing happens before approval


def test_approval_executes_exactly_the_proposal(client):
    run = alert(client)
    resp = approve(client, run)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed" and body["replayed"] is False and body["approved_by"] == "alice"
    assert "v2.14.3 -> v2.14.2" in body["outcome"]
    [write] = [r for r in client.get(f"/runs/{run['thread_id']}").json()["audit"] if r["permission"] == "write"]
    assert write["args"] == run["proposal"]["args"]
    assert write["approval"]["approver"] == "alice"
    assert write["approval"]["args_hash"] == run["proposal"]["args_hash"]


def test_approval_for_other_args_is_refused_and_the_run_stays_pending(client):
    run = alert(client)
    resp = approve(client, run, args_hash="0" * 16)
    assert resp.status_code == 409
    assert resp.json()["detail"]["proposal"] == run["proposal"]
    assert not side_effects(run)
    assert client.get(f"/runs/{run['thread_id']}").json()["status"] == "awaiting_approval"
    assert approve(client, run).status_code == 200  # the right hash still works


def test_same_decision_twice_replays_without_acting_again(client):
    run = alert(client)
    first, second = approve(client, run).json(), approve(client, run).json()
    assert second["replayed"] is True
    assert {k: v for k, v in second.items() if k != "replayed"} == {k: v for k, v in first.items() if k != "replayed"}
    assert len(side_effects(run)) == 1


def test_a_different_second_decision_is_refused(client):
    run = alert(client)
    approve(client, run, approved=False)
    resp = approve(client, run, approved=True)
    assert resp.status_code == 409 and resp.json()["detail"]["decided_by"] == "alice"
    assert not side_effects(run)


def test_concurrent_approvals_act_once(client):
    run = alert(client)
    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: approve(client, run), range(8)))
    assert all(r.status_code == 200 for r in responses)
    assert sum(not r.json()["replayed"] for r in responses) == 1
    assert len(side_effects(run)) == 1


def test_rejection_escalates_with_no_write(client):
    run = alert(client)
    body = approve(client, run, approved=False, approver="sre-lead").json()
    assert body["status"] == "escalated" and body["outcome"] == "Escalated to a human: rejected by sre-lead"
    assert body["approved_by"] is None and not side_effects(run)


def test_sev3_alert_closes_after_a_metrics_check_and_has_nothing_to_approve(client):
    run = alert(client, "inc03")
    assert run["status"] == "completed" and "metrics confirm no impact" in run["outcome"]
    resp = client.post("/approve", json={"thread_id": run["thread_id"], "approved": True,
                                         "approver": "alice", "args_hash": "x"})
    assert resp.status_code == 409 and "no proposal waiting" in resp.json()["detail"]


def test_unknown_service_escalates(client):
    resp = client.post("/alert", json={"alert_text": "billing-api error rate spiking after deploy, customers affected"})
    run = resp.json()
    assert run["status"] == "escalated" and "billing-api" not in serve.CATALOG
    audit = client.get(f"/runs/{run['thread_id']}").json()["audit"]
    assert audit and all(r["outcome"] in ("failed", "circuit_open") for r in audit)
    assert "404" in audit[0]["error"] and audit[0]["attempts"] == 1  # a 404 isn't retried


def test_run_endpoint_reports_budget_and_audit(client):
    run = alert(client, "inc02")
    body = client.get(f"/runs/{run['thread_id']}").json()
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


def test_another_approver_is_not_a_replay(client):
    run = alert(client)
    assert approve(client, run, approver="alice").json()["replayed"] is False
    bob = approve(client, run, approver="bob")
    assert bob.status_code == 409 and bob.json()["detail"] == {
        "error": "This run was already decided", "decided_by": "alice", "approved": True}
    assert approve(client, run, approver="alice").json()["replayed"] is True  # alice retrying is
    assert len(side_effects(run)) == 1


def test_run_snapshot_is_never_taken_mid_execution(client):
    """The rollback takes effect, then the write blocks. A GET issued then
    must not report the run as still awaiting approval next to a side effect
    that already happened: it waits, and sees the finished run."""
    run = alert(client)
    infra = serve.RUNS[run["thread_id"]].infra
    applied, release = threading.Event(), threading.Event()
    real_rollback = infra.rollback

    def slow_rollback(*args):
        result = real_rollback(*args)  # the side effect exists from here on
        applied.set()
        release.wait(timeout=3)
        return result

    infra.rollback = slow_rollback
    approval = threading.Thread(target=approve, args=(client, run))
    approval.start()
    assert applied.wait(timeout=3)

    snapshots = []
    reader = threading.Thread(target=lambda: snapshots.append(client.get(f"/runs/{run['thread_id']}").json()))
    reader.start()
    reader.join(timeout=0.3)
    assert reader.is_alive(), "GET returned while the approval was still executing"

    release.set()
    approval.join(timeout=3)
    reader.join(timeout=3)
    [snap] = snapshots
    assert snap["status"] == "completed" and len(snap["side_effects"]) == 1
    assert any(r["permission"] == "write" for r in snap["audit"])
