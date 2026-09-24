"""Offline tests for the agent observability/eval harness
(src/ai_engineering/agent_eval.py).

conftest.py blanks every API key, so OFFLINE mode runs the real graph with
rule-based stand-ins that apply the runbook to each incident's fixtures —
no network.
"""
import json

import pytest

import ai_engineering.agent_eval as ae


def test_suite_is_offline():
    assert ae.OFFLINE, "conftest.py should force OFFLINE mode regardless of the developer's env"


def test_classify_severity():
    assert ae._classify_severity("critical, revenue impact") == "SEV1"
    assert ae._classify_severity("error rate climbing after deploy") == "SEV2"
    assert ae._classify_severity("slightly up, no user impact") == "SEV3"


def test_extract_service():
    assert ae._extract_service("payments-api error rate up") == "payments-api"
    assert ae._extract_service("auth-service latency up") == "auth-service"
    assert ae._extract_service("no service mentioned here") == "unknown-service"


def test_norm_ignores_case_and_separators():
    assert ae._norm("Payments API") == ae._norm("payments-api") == ae._norm("payments_api")


def test_call_tool_records_unknown_tool_and_missing_service_as_failures():
    incident = ae.load_incidents()[0]
    span = ae.NodeSpan(node="investigate")
    assert "error" in ae._call_tool("delete_database", {"service": "x"}, incident, span)
    assert "error" in ae._call_tool("get_metrics", {}, incident, span)
    assert [tc.success for tc in span.tool_calls] == [False, False]


def test_wrong_service_gets_decoy_evidence_that_names_it():
    incident = ae.load_incidents()[0]
    result = ae._call_tool("search_logs", {"service": "other-api"}, incident, ae.NodeSpan(node="x"))
    assert all("other-api" in line for line in result["matches"])
    assert "payments-api" not in json.dumps(result)


# ---- dataset: labels must be derivable, and must not all be the same answer

def test_incident_set_is_well_formed_and_heterogeneous():
    incidents = ae.load_incidents()
    actions = {i["expected_action"] for i in incidents}
    assert actions >= {"rollback_deploy", "page_oncall", "restart_service", "monitor", "close"}
    for i in incidents:
        assert i["expected_severity"] in ("SEV1", "SEV2", "SEV3")
        assert i["service"] in i["alert"]
        assert set(i["fixtures"]) == {"logs", "deploys", "metrics", "dependencies"}


@pytest.mark.parametrize("incident", [i for i in ae.load_incidents() if i["expected_action"] != "close"],
                         ids=lambda i: i["id"])
def test_runbook_as_code_reproduces_each_label(incident):
    fx = incident["fixtures"]
    assert ae.apply_runbook(fx["deploys"], fx["metrics"], fx["dependencies"])[0] == incident["expected_action"]


# ---- deterministic RCA grounding

def test_unsupported_facts_matches_percentages_versions_and_times():
    evidence = 'alert at 10:42 {"error_rate": 0.31, "version": "v2.14.3", "minutes_ago": 9}'
    assert ae.unsupported_facts("31% errors since 10:42, 9 minutes after v2.14.3", evidence) == []
    assert ae.unsupported_facts("45% errors after v2.14.4 at 11:00", evidence) == ["v2.14.4", "11:00", "45"]


def test_unsupported_facts_skips_identifiers_but_checks_units_and_separators():
    assert ae.unsupported_facts("p95 above SLO on SEV1", "nothing") == []
    evidence = '"p95_latency_ms": 6200'
    assert ae.unsupported_facts("p95 hit 6,200ms", evidence) == []
    assert ae.unsupported_facts("p95 hit 6200 ms", evidence) == []
    assert ae.unsupported_facts("p95 hit 620ms", evidence) == ["620"]


def test_unsupported_facts_allows_unit_conversions_at_written_precision():
    evidence = '"p95_latency_ms": 6200, "minutes_ago": 1500, "queue_depth": 48210, "error_rate": 0.021'
    # ms -> s, minutes -> hours, "48k", fraction -> percent
    assert ae.unsupported_facts("p95 6.2 s; deployed 25 hours ago; 48k queued; 2.1% errors", evidence) == []
    # but not arbitrary nearby numbers
    assert ae.unsupported_facts("p95 6.5 s; deployed 26 hours ago; 2.4% errors", evidence) == ["6.5", "26", "2.4"]


def test_crashed_run_is_scored_as_failure_not_dropped(monkeypatch):
    incidents = ae.load_incidents()[:2]

    def boom(*a, **k):
        raise RuntimeError("503 after all retries")

    monkeypatch.setattr(ae, "run_once", boom)
    results = ae.run_eval_suite(incidents)
    assert [r.error is not None for r in results] == [True, True]
    assert all(r.score == 0.0 for r in results)
    assert ae.summarize(results)["crashed_runs"] == 2


# ---- full graph runs

def test_healthy_run_scores_on_every_metric():
    incident = ae.load_incidents()[0]
    state, trace = ae.run_once(incident, thread_id="test-healthy")
    result = ae.evaluate_run(incident, state, trace)
    assert result.severity_correct and result.action_correct
    assert result.tool_target_ok is True
    assert result.rca_grounded is True and result.unsupported == []
    assert trace.spans and trace.total_latency_ms >= 0


def test_offline_suite_gets_every_action_right():
    results = ae.run_eval_suite(ae.load_incidents())
    assert all(r.action_correct and r.severity_correct for r in results), [
        (r.incident_id, r.predicted_action) for r in results if not r.action_correct]


def test_broken_run_fails_tool_targeting_even_though_verify_says_ok():
    incident = ae.load_incidents()[0]  # payments-api alert
    state, trace = ae.run_once(incident, thread_id="test-broken", broken=True)
    assert state["verdict"] == "ok", "the bug: the weakened verify check should still say ok"
    result = ae.evaluate_run(incident, state, trace)
    assert result.tool_target_ok is False, "tool targeting should catch the wrong-service evidence"
    assert {tc.args["service"] for tc in trace.tool_calls} == {ae.WRONG_SERVICE}
    # The offline RCA faithfully summarizes the (wrong) evidence, so it's grounded
    # in what the tools returned — the two checks answer different questions.
    assert result.rca_grounded is True


def test_triage_close_marks_evidence_metrics_not_applicable():
    sev3 = next(i for i in ae.load_incidents() if i["expected_action"] == "close")
    state, trace = ae.run_once(sev3, thread_id="test-sev3")
    result = ae.evaluate_run(sev3, state, trace)
    assert "closed at triage" in state["outcome"]
    assert not any(s.node == "write_rca" for s in trace.spans)
    assert result.action_correct
    assert result.tool_target_ok is None and result.rca_grounded is None
    assert result.score == 1.0


def test_summary_rates_skip_not_applicable_runs():
    results = ae.run_eval_suite(ae.load_incidents())
    summary = ae.summarize(results)
    investigated = sum(r.tool_target_ok is not None for r in results)
    assert summary["tool_target_accuracy"] == f"{investigated}/{investigated}"


def test_chat_retries_transient_errors_but_not_client_errors(monkeypatch):
    import httpx
    import openai

    req = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    calls = {"n": 0}

    class FlakyClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    calls["n"] += 1
                    if calls["n"] < 3:
                        raise openai.RateLimitError("429", response=httpx.Response(429, request=req), body=None)
                    return "ok"

    monkeypatch.setattr("tenacity.nap.time.sleep", lambda s: None)
    monkeypatch.setattr(ae, "TRANSIENT_ERRORS", {})
    assert ae._chat(FlakyClient, model="m") == "ok" and calls["n"] == 3
    assert ae.TRANSIENT_ERRORS == {"429": 2}

    class BadRequestClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    calls["n"] += 1
                    raise openai.BadRequestError("400", response=httpx.Response(400, request=req), body=None)

    calls["n"] = 0
    with pytest.raises(openai.BadRequestError):
        ae._chat(BadRequestClient, model="m")
    assert calls["n"] == 1
