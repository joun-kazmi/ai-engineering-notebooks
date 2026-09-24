"""Offline tests for the agent observability/eval harness
(src/ai_engineering/agent_eval.py).

conftest.py blanks every API key, so OFFLINE mode runs the real
triage -> investigate -> verify -> write_rca -> human_gate -> execute graph
with rule-based stand-ins — no network.
"""
import ai_engineering.agent_eval as aoe


def test_suite_is_offline():
    assert aoe.OFFLINE, "conftest.py should force OFFLINE mode regardless of the developer's env"


def test_classify_severity():
    assert aoe._classify_severity("critical, revenue impact") == "SEV1"
    assert aoe._classify_severity("error rate climbing after deploy") == "SEV2"
    assert aoe._classify_severity("slightly up, no user impact") == "SEV3"


def test_extract_service():
    assert aoe._extract_service("payments-api error rate up") == "payments-api"
    assert aoe._extract_service("auth-service latency up") == "auth-service"
    assert aoe._extract_service("no service mentioned here") == "unknown-service"


def test_norm_ignores_case_and_separators():
    assert aoe._norm("Payments API") == aoe._norm("payments-api") == aoe._norm("payments_api")


def test_relevance_scores_higher_overlap_higher():
    alert = "payments-api error rate high"
    assert aoe._relevance(alert, "payments-api error rate is 31%") > aoe._relevance(alert, "unrelated weather text")


def test_call_tool_records_unknown_tool_as_failure_instead_of_raising():
    span = aoe.NodeSpan(node="investigate")
    result = aoe._call_tool("delete_database", {"service": "x"}, "alert", span)
    assert "error" in result
    assert span.tool_calls[0].success is False


def test_incident_eval_set_is_well_formed():
    for incident in aoe.load_incidents():
        assert incident["id"] and incident["alert"]
        assert incident["expected_severity"] in ("SEV1", "SEV2", "SEV3")


def test_healthy_run_is_grounded_and_matches_expectations():
    incident = aoe.load_incidents()[0]
    state, trace = aoe.run_once(incident["alert"], thread_id="test-healthy")
    result = aoe.evaluate_run(incident, state, trace)
    assert result.severity_correct and result.action_correct and result.grounded
    assert result.tools_on_target and result.mentions_service
    assert trace.spans and trace.total_latency_ms >= 0


def test_broken_run_is_not_grounded_even_though_verify_says_ok():
    incident = aoe.load_incidents()[0]  # payments-api alert
    state, trace = aoe.run_once(incident["alert"], thread_id="test-broken", broken=True)
    assert state["verdict"] == "ok", "the bug: the weakened verify check should still say ok"
    result = aoe.evaluate_run(incident, state, trace)
    assert not result.grounded, "the independent groundedness check should catch the mismatch"
    assert not result.tools_on_target
    assert {tc.args["service"] for tc in trace.tool_calls} == {aoe.WRONG_SERVICE}


def test_sev3_incident_auto_closes_without_reaching_write_rca():
    sev3 = next(i for i in aoe.load_incidents() if i["expected_severity"] == "SEV3")
    state, trace = aoe.run_once(sev3["alert"], thread_id="test-sev3")
    assert "closed" in state["outcome"].lower()
    assert not any(s.node == "write_rca" for s in trace.spans)
    assert aoe.evaluate_run(sev3, state, trace).grounded
