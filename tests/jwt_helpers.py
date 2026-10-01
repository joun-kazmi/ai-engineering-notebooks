"""Locally signed tokens for the HTTP tests: an RSA key pair made here, a
verifier that trusts only its public key. No identity provider, no network."""
import time

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa

from ai_engineering.auth import SreApprovalPolicy, StaticKeyProvider, TokenVerifier

ISSUER = "https://idp.test"
AUDIENCE = "escalation-agent"
ALL_SCOPES = {"alerts:create", "runs:read", "runs:approve", "runs:recover"}

PRIVATE_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
PUBLIC_KEY = PRIVATE_KEY.public_key()


def principal(sub: str) -> str:
    return f"{ISSUER}|{sub}"


def claims(sub="alice", scopes=ALL_SCOPES, groups=("sre",), **overrides) -> dict:
    now = int(time.time())
    body = {"iss": ISSUER, "aud": AUDIENCE, "sub": sub, "iat": now, "exp": now + 300,
            "scope": " ".join(sorted(scopes)) if scopes is not None else None,
            "groups": groups if groups is None or isinstance(groups, str) else sorted(groups)}
    body.update(overrides)
    return {k: v for k, v in body.items() if v is not None}  # None drops a claim


def token(sub="alice", scopes=ALL_SCOPES, groups=("sre",), key=PRIVATE_KEY, algorithm="RS256", **overrides) -> str:
    return jwt.encode(claims(sub, scopes, groups, **overrides), key, algorithm=algorithm)


def bearer(tok: str) -> dict:
    return {"Authorization": f"Bearer {tok}"}


def as_user(sub="alice", **kw) -> dict:
    return bearer(token(sub, **kw))


def verifier(**kw) -> TokenVerifier:
    return TokenVerifier(ISSUER, AUDIENCE, StaticKeyProvider(PUBLIC_KEY), **kw)


def configure_test_auth(serve, **kw):
    """The service trusting this module's key, with the default SRE policy."""
    return serve.configure_auth(verifier=verifier(**kw), policy=SreApprovalPolicy("sre"))
