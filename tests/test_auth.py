"""Offline tests for authentication on the served agent (ai_engineering.auth
and its use in src/fastapi_serve.py).

Tokens are signed here with an RSA key made for the test run (jwt_helpers)
and verified through StaticKeyProvider: no identity provider, no network.
"""
import base64
import hashlib
import hmac
import json
import logging
import time

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from fastapi.testclient import TestClient

import ai_engineering.agent_eval as ae
import ai_engineering.auth as auth
import src.fastapi_serve as serve
from ai_engineering.auth import (Forbidden, Identity, InvalidToken, JwksKeyProvider, SreApprovalPolicy,
                                 StaticKeyProvider, TokenVerifier, normalized_claim_set)
from jwt_helpers import (ALL_SCOPES, AUDIENCE, ISSUER, PUBLIC_KEY, as_user, bearer, claims,
                         configure_test_auth, principal, token, verifier)

INC = {i["id"]: i for i in ae.load_incidents()}


class Spans:
    """Records the spans the service opens: name, level, actor."""

    def __init__(self):
        self.spans = []

    def span(self, thread_id, name, input=None, metadata=None):
        rec = {"name": name, "level": None, "actor": metadata, "output": None}
        self.spans.append(rec)

        class Span:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def update(self, output=None, metadata=None, level=None, status_message=None):
                rec.update({k: v for k, v in (("output", output), ("level", level)) if v is not None})

            def set_trace_io(self, **io):
                pass

        return Span()

    def callbacks(self):
        return []

    def url(self, thread_id):
        return None


@pytest.fixture
def tracer():
    return Spans()


@pytest.fixture
def svc(tmp_path, tracer):
    configure_test_auth(serve)
    s = serve.configure(tmp_path / "serve.sqlite3", approval_ttl_s=3600, retention_s=86400, lease_s=300,
                        tracer=tracer)
    yield s
    serve.reset_auth()


@pytest.fixture
def client(svc):
    return TestClient(serve.app_fastapi)


def start_run(client) -> dict:
    resp = client.post("/alert", headers=as_user("oncall"), json={"alert_text": INC["inc01"]["alert"]})
    assert resp.status_code == 200, resp.text
    return resp.json()


def approve(client, run, headers, approved=True, args_hash=None):
    return client.post("/approve", headers=headers, json={
        "thread_id": run["thread_id"], "approved": approved, "args_hash": args_hash or run["proposal"]["args_hash"]})


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def unsigned_token(header: dict, body: dict, signature: bytes = b"") -> str:
    head = b64url(json.dumps(header).encode()) + "." + b64url(json.dumps(body).encode())
    return head + "." + b64url(signature)


def hs256_with_public_key() -> str:
    """The key-confusion attack: an HS256 token whose HMAC secret is the
    verifier's public key, which an attacker can download."""
    secret = PUBLIC_KEY.public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    header, body = {"alg": "HS256", "typ": "JWT"}, claims("mallory")
    signing_input = b64url(json.dumps(header).encode()) + "." + b64url(json.dumps(body).encode())
    return signing_input + "." + b64url(hmac.new(secret, signing_input.encode(), hashlib.sha256).digest())


ENDPOINTS = [
    ("post", "/alert", "alerts:create", lambda run: {"json": {"alert_text": "payments-api errors"}}),
    ("post", "/approve", "runs:approve", lambda run: {"json": {"thread_id": run["thread_id"], "approved": True,
                                                              "args_hash": run["proposal"]["args_hash"]}}),
    ("get", "/runs/{id}", "runs:read", lambda run: {}),
    ("post", "/runs/{id}/recover", "runs:recover", lambda run: {}),
]


def call(client, endpoint, run, headers=None):
    method, path, _, kwargs = endpoint
    return getattr(client, method)(path.replace("{id}", run["thread_id"]), headers=headers or {}, **kwargs(run))


# ---- 401: no identity

class SpyKeys(StaticKeyProvider):
    def __init__(self):
        super().__init__(PUBLIC_KEY)
        self.calls = 0

    def get_key(self, token):
        self.calls += 1
        return super().get_key(token)


BAD_TOKENS = {
    "malformed": lambda: "not-a-jwt",
    "garbage segments": lambda: "aaa.bbb.ccc",
    "wrong issuer": lambda: token(iss="https://evil.test"),
    "wrong audience": lambda: token(aud="another-api"),
    "expired": lambda: token(exp=int(time.time()) - 3600),
    "not yet valid": lambda: token(nbf=int(time.time()) + 3600),
    "signed by another key": lambda: token(key=rsa.generate_private_key(public_exponent=65537, key_size=2048)),
    "alg none": lambda: unsigned_token({"alg": "none", "typ": "JWT"}, claims("mallory")),
    "HS256 with the public key": hs256_with_public_key,
    "no exp": lambda: token(exp=None),
    "no sub": lambda: token(sub=None),
    "no iss": lambda: token(iss=None),
    "no aud": lambda: token(aud=None),
    "empty sub": lambda: token(sub=""),
    "scope claim not a string or list": lambda: token(scope={"runs:approve": True}),
}


@pytest.mark.parametrize("make", BAD_TOKENS.values(), ids=BAD_TOKENS.keys())
def test_bad_tokens_are_401(client, make):
    run = start_run(client)
    for endpoint in ENDPOINTS:
        resp = call(client, endpoint, run, bearer(make()))
        assert resp.status_code == 401, (endpoint[1], resp.text)
        assert resp.headers["www-authenticate"] == "Bearer"


def test_missing_token_is_401_on_every_endpoint(client):
    run = start_run(client)
    for endpoint in ENDPOINTS:
        assert call(client, endpoint, run).status_code == 401, endpoint[1]
        assert call(client, endpoint, run, {"Authorization": "Basic YWxpY2U6cHc="}).status_code == 401


@pytest.mark.parametrize("make", [BAD_TOKENS["alg none"], hs256_with_public_key], ids=["none", "HS256"])
def test_the_algorithm_is_never_taken_from_the_token(make):
    """Refused on the header alone: no key is looked up (so no JWKS fetch is
    triggered by junk), and the key is never handed to an HMAC check."""
    keys = SpyKeys()
    v = TokenVerifier(ISSUER, AUDIENCE, keys)
    with pytest.raises(InvalidToken, match="not accepted"):
        v.verify(make())
    assert keys.calls == 0
    assert v.verify(token()).principal_id == principal("alice") and keys.calls == 1


@pytest.mark.parametrize("algorithms", [("HS256",), ("RS256", "HS512"), ("none",), ("RS256", "None"), ()])
def test_symmetric_and_none_algorithms_can_not_be_configured(algorithms):
    with pytest.raises(ValueError):
        TokenVerifier(ISSUER, AUDIENCE, StaticKeyProvider(PUBLIC_KEY), algorithms=algorithms)


def test_es256_tokens_verify():
    key = ec.generate_private_key(ec.SECP256R1())
    v = TokenVerifier(ISSUER, AUDIENCE, StaticKeyProvider(key.public_key()))
    assert v.verify(token(key=key, algorithm="ES256")).principal_id == principal("alice")


def test_leeway_tolerates_small_clock_skew():
    assert verifier().verify(token(exp=int(time.time()) - 10)).principal_id == principal("alice")


# ---- 403: an identity without the scope

@pytest.mark.parametrize("endpoint", ENDPOINTS, ids=[e[1] for e in ENDPOINTS])
def test_missing_scope_is_403(client, endpoint):
    run = start_run(client)
    scope = endpoint[2]
    resp = call(client, endpoint, run, as_user("alice", scopes=ALL_SCOPES - {scope}))
    assert resp.status_code == 403 and scope in resp.json()["detail"]
    assert call(client, endpoint, run, as_user("alice", scopes={scope})).status_code != 403


def test_refused_tokens_get_no_span_and_touch_no_run(client, svc, tracer):
    """A bad token or a missing scope is refused before the run is read: no
    span, so unauthenticated callers can't fill a run's trace (or probe
    which run ids exist)."""
    run = start_run(client)
    before = svc.store.get(run["thread_id"])
    assert [s["name"] for s in tracer.spans] == ["served:alert"]
    for endpoint in ENDPOINTS:
        call(client, endpoint, run)
        call(client, endpoint, run, bearer(token(iss="https://evil.test")))
        call(client, endpoint, run, as_user("alice", scopes=ALL_SCOPES - {endpoint[2]}))
    assert approve(client, {"thread_id": "made-up", "proposal": {"args_hash": "x"}}, {}).status_code == 401
    assert [s["name"] for s in tracer.spans] == ["served:alert"]  # still only the first, valid /alert
    assert svc.store.get(run["thread_id"]) == before


# ---- the payload

def test_approver_in_the_body_is_422(client):
    run = start_run(client)
    resp = client.post("/approve", headers=as_user("alice"), json={
        "thread_id": run["thread_id"], "approved": True, "approver": "alice",
        "args_hash": run["proposal"]["args_hash"]})
    assert resp.status_code == 422
    assert client.get(f"/runs/{run['thread_id']}", headers=as_user("alice")).json()["status"] == "awaiting_approval"


def test_alert_payload_rejects_unknown_fields(client):
    resp = client.post("/alert", headers=as_user("alice"), json={"alert_text": "x", "severity": "SEV1"})
    assert resp.status_code == 422


# ---- the approval policy

def test_non_sre_approval_is_403_before_the_claim(client, svc, tracer):
    run = start_run(client)
    before, audit_before = svc.store.get(run["thread_id"]), svc.store.audit(run["thread_id"])
    resp = approve(client, run, as_user("dev", groups=("developers",)))
    assert resp.status_code == 403 and "sre" in resp.json()["detail"]["error"]
    # Never claimed: nothing about the run changed, nothing executed.
    after = svc.store.get(run["thread_id"])
    assert after == before and after["status"] == "awaiting_approval" and after["decision"] is None
    assert svc.store.audit(run["thread_id"]) == audit_before
    assert not (after["infra"] or {}).get("effects")
    denied = tracer.spans[-1]
    assert denied["name"] == "served:approve" and denied["level"] == "WARNING"
    assert denied["output"]["status_code"] == 403
    assert denied["actor"] == {"actor_id": principal("dev"), "scope": "runs:approve"}
    # An SRE can still approve it.
    assert approve(client, run, as_user("alice")).json()["status"] == "completed"


def test_non_sre_may_reject(client):
    """Deliberate: a rejection takes no action (the run escalates to a
    human), so it needs only the runs:approve scope, not the group."""
    run = start_run(client)
    resp = approve(client, run, as_user("dev", groups=()), approved=False)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "escalated" and body["approved_by"] is None
    assert body["outcome"] == f"Escalated to a human: rejected by {principal('dev')}"


def test_the_policy_runs_after_the_proposal_checks(client, svc):
    """A stale args_hash is a 409 whoever sends it: the policy is asked only
    about the proposal actually waiting."""
    run = start_run(client)
    dev = as_user("dev", groups=())
    assert approve(client, run, dev, args_hash="0" * 16).status_code == 409
    svc.clock = lambda: time.time() + 3601
    assert approve(client, run, dev).status_code == 410


def test_the_token_subject_is_the_approver_everywhere(client, svc):
    run = start_run(client)
    alice = principal("alice")
    resp = approve(client, run, bearer(token("alice", email="alice@example.com", name="Alice")))
    assert resp.json()["approved_by"] == alice
    row = svc.store.get(run["thread_id"])
    assert row["decision"] == {"approved": True, "args_hash": run["proposal"]["args_hash"], "approver": alice}
    [write] = [r for r in svc.store.audit(run["thread_id"]) if r["permission"] == "write"]
    assert write["approval"]["approver"] == alice
    assert "alice@example.com" not in json.dumps(row) + json.dumps(svc.store.audit(run["thread_id"]))
    # Same subject from another issuer is someone else.
    other_issuer = TokenVerifier("https://other-idp.test", AUDIENCE, StaticKeyProvider(PUBLIC_KEY))
    assert other_issuer.verify(token("alice", iss="https://other-idp.test")).principal_id != alice


# ---- claim shapes

@pytest.mark.parametrize("value, expected", [
    (None, set()), ("", set()), ("runs:read  runs:approve", {"runs:read", "runs:approve"}),
    (["sre", "dba"], {"sre", "dba"}), ([], set()),
])
def test_normalized_claim_set(value, expected):
    assert normalized_claim_set(value) == frozenset(expected)


@pytest.mark.parametrize("value", [42, {"sre": True}, ["sre", 1], True])
def test_malformed_claims_are_invalid_tokens(value):
    with pytest.raises(InvalidToken):
        normalized_claim_set(value)


def test_scopes_and_groups_as_string_or_list():
    v = verifier()
    from_strings = v.verify(token(scope="runs:read runs:approve", groups="sre dba"))
    from_lists = v.verify(token(scope=["runs:read", "runs:approve"], groups=["sre", "dba"]))
    assert from_strings == from_lists == Identity(principal("alice"), frozenset({"runs:read", "runs:approve"}),
                                                  frozenset({"sre", "dba"}))


def test_custom_claim_names(client):
    """E.g. Entra ID: scopes in `scp`, app roles in `roles`."""
    configure_test_auth(serve, scope_claim="scp", groups_claim="roles")
    run = client.post("/alert", headers=bearer(token(scope=None, groups=None, scp="alerts:create")),
                      json={"alert_text": INC["inc01"]["alert"]}).json()
    custom = bearer(token("alice", scope=None, groups=None, scp=["runs:approve"], roles=["sre"]))
    assert approve(client, run, custom).json()["approved_by"] == principal("alice")
    # The default claim names now mean nothing.
    ignored = bearer(token("bob", scope="runs:read", groups=["sre"]))
    assert client.get(f"/runs/{run['thread_id']}", headers=ignored).status_code == 403


# ---- policy, keys, configuration

def test_sre_policy():
    policy = SreApprovalPolicy("sre")
    sre = Identity("i|a", frozenset({"runs:approve"}), frozenset({"sre"}))
    dev = Identity("i|b", frozenset({"runs:approve"}), frozenset({"developers"}))
    policy.authorize(sre, {"tool": "rollback_deploy"}, True)
    policy.authorize(dev, {"tool": "rollback_deploy"}, False)
    with pytest.raises(Forbidden):
        policy.authorize(dev, {"tool": "rollback_deploy"}, True)
    with pytest.raises(Forbidden):
        policy.authorize(Identity("i|c", frozenset(), frozenset({"sre"})), {}, False)


def test_jwks_provider_reuses_one_client(monkeypatch):
    made = []

    class FakeClient:
        def __init__(self, uri, **kw):
            self.uri, self.lookups = uri, 0
            made.append(self)

        def get_signing_key_from_jwt(self, tok):
            self.lookups += 1
            return type("Key", (), {"key": PUBLIC_KEY})()

    monkeypatch.setattr(auth.jwt, "PyJWKClient", FakeClient)
    keys = JwksKeyProvider("https://idp.test/jwks")
    v = TokenVerifier(ISSUER, AUDIENCE, keys)
    for _ in range(3):
        v.verify(token())
    assert len(made) == 1 and made[0].lookups == 3 and made[0].uri == "https://idp.test/jwks"


def test_jwks_outage_is_503_not_401(client, monkeypatch):
    class Down:
        def get_signing_key_from_jwt(self, tok):
            raise jwt.PyJWKClientConnectionError("connection refused")

    keys = JwksKeyProvider.__new__(JwksKeyProvider)
    keys.client = Down()
    serve.configure_auth(verifier=TokenVerifier(ISSUER, AUDIENCE, keys))
    assert client.get("/runs/x", headers=as_user("alice")).status_code == 503


def blank_auth_env(monkeypatch, **values):
    for name in ("AUTH_MODE", "AUTH_ISSUER", "AUTH_AUDIENCE", "AUTH_JWKS_URL", "AUTH_ALGORITHMS",
                 "AUTH_SCOPE_CLAIM", "AUTH_GROUPS_CLAIM", "AUTH_APPROVER_GROUP"):
        monkeypatch.setenv(name, values.get(name, ""))


def test_oidc_without_its_settings_fails_closed(client, monkeypatch):
    blank_auth_env(monkeypatch)  # AUTH_MODE blank: the default, oidc
    serve.reset_auth()
    for _ in range(2):  # stays refused; nothing half-configured is cached
        resp = client.get("/runs/x", headers=as_user("alice"))
        assert resp.status_code == 503 and resp.json()["detail"] == "Authentication is not configured"
    assert client.post("/alert", json={"alert_text": "x"}).status_code == 503


def test_oidc_settings_build_a_jwks_verifier_lazily(monkeypatch):
    blank_auth_env(monkeypatch, AUTH_ISSUER=ISSUER, AUTH_AUDIENCE=AUDIENCE, AUTH_JWKS_URL="https://idp.test/jwks",
                   AUTH_SCOPE_CLAIM="scp", AUTH_ALGORITHMS="ES256")
    serve.reset_auth()
    assert serve._auth is None  # nothing until a request needs it
    try:
        cfg = serve.auth_config()
    finally:
        serve.reset_auth()
    assert cfg.mode == "oidc" and isinstance(cfg.verifier.key_provider, JwksKeyProvider)
    assert cfg.verifier.algorithms == ("ES256",) and cfg.verifier.scope_claim == "scp"
    assert cfg.verifier.groups_claim == "groups" and cfg.policy.group == "sre"


def test_disabled_mode_is_one_local_principal_and_warns_once(svc, monkeypatch, caplog):
    blank_auth_env(monkeypatch, AUTH_MODE="disabled")
    serve.reset_auth()
    client = TestClient(serve.app_fastapi)
    with caplog.at_level(logging.WARNING, logger=serve.__name__):
        run = client.post("/alert", json={"alert_text": INC["inc01"]["alert"]}).json()
        body = approve(client, run, {}).json()
        client.get(f"/runs/{run['thread_id']}")
    assert body["status"] == "completed" and body["approved_by"] == serve.LOCAL_PRINCIPAL
    warnings = [r for r in caplog.records if "AUTH_MODE=disabled" in r.getMessage()]
    assert len(warnings) == 1 and warnings[0].levelno == logging.WARNING
