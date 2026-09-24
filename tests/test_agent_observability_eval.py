"""Offline tests for the agent observability/eval harness.

OFFLINE mode requires no API key and no network, so the full graph can run
here — these exercise the real triage -> investigate -> verify -> write_rca
-> human_gate -> execute path, not just the pure helper functions.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import agent_observability_eval as aoe  # noqa: E402


def test_classify_severity():
    assert aoe._classify_severity("critical, revenue impact") == "SEV1"
    assert aoe._classify_severity("error rate climbing after deploy") == "SEV2"
    assert aoe._classify_severity("slightly up, no user impact") == "SEV3"


def test_extract_service():
    assert aoe._extract_service("payments-api error rate up") == "payments-api"
    assert aoe._extract_service("auth-service latency up") == "auth-service"
    assert aoe._extract_service("no service mentioned here") == "unknown-service"


def test_relevance_scores_higher_overlap_higher():
    alert = "payments-api error rate high"
    close = aoe._relevance(alert, "payments-api error rate is 31%")
    far = aoe._relevance(alert, "completely unrelated text about weather")
    assert close > far


def test_incident_eval_set_is_well_formed():
    dataset = json.loads((ROOT / "data" / "incident_eval_set.json").read_text())
    assert dataset
    for incident in dataset:
        assert incident["expected_severity"] in ("SEV1", "SEV2", "SEV3")
        assert incident["id"]
        assert incident["alert"]


def test_healthy_run_is_grounded_and_matches_expectations():
    dataset = json.loads((ROOT / "data" / "incident_eval_set.json").read_text())
    incident = dataset[0]
    state, trace = aoe.run_once(incident["alert"], thread_id="test-healthy")
    result = aoe.evaluate_run(incident, state, trace)
    assert result.severity_correct
    assert result.action_correct
    assert result.grounded
    assert trace.spans, "expected at least one recorded span"
    assert trace.total_latency_ms >= 0


def test_broken_run_is_not_grounded_even_though_verify_says_ok():
    dataset = json.loads((ROOT / "data" / "incident_eval_set.json").read_text())
    incident = dataset[0]  # payments-api alert
    state, trace = aoe.run_once(incident["alert"], thread_id="test-broken", broken=True)
    assert state["verdict"] == "ok", "the bug: the weakened verify check should still say ok"
    result = aoe.evaluate_run(incident, state, trace)
    assert not result.grounded, "the independent groundedness check should catch the mismatch"


def test_sev3_incident_auto_closes_without_reaching_write_rca():
    dataset = json.loads((ROOT / "data" / "incident_eval_set.json").read_text())
    sev3 = next(i for i in dataset if i["expected_severity"] == "SEV3")
    state, trace = aoe.run_once(sev3["alert"], thread_id="test-sev3")
    assert "closed" in state["outcome"].lower()
    assert not any(s.node == "write_rca" for s in trace.spans)
