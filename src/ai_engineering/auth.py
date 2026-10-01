"""Verified caller identity for the served agent (src/fastapi_serve.py).

A request carries a Bearer JWT from an OIDC identity provider. TokenVerifier
checks it against one issuer and audience and turns it into an Identity:

    principal_id   f"{iss}|{sub}" — the subject is stable and unique only
                   within its issuer, so the pair is the principal. Never the
                   email: it can change, and be reassigned to someone else.
    scopes         what the token lets its holder do (`scope` claim)
    groups         who the holder is, for policies (`groups` claim)

The algorithm comes from the verifier's configuration, never from the token:
a token whose header names anything else (HS256 signed with the public key,
`none`) is refused before any key is looked up. HMAC and `none` can't be
configured at all — a shared secret would let anyone holding the verifier's
key mint tokens.

Keys come from a KeyProvider: StaticKeyProvider (tests) or JwksKeyProvider,
PyJWT's PyJWKClient over the issuer's JWKS endpoint, which caches the key set.

ApprovalPolicy decides whether an identity may decide a given proposal; it's
called by the service after the proposal checks and before the approval is
claimed. This module issues no tokens and knows no provider.
"""
from dataclasses import dataclass, field
from typing import Any, Protocol

import jwt

REQUIRED_CLAIMS = ("exp", "iss", "aud", "sub")


class AuthError(Exception):
    pass


class InvalidToken(AuthError):
    """Missing, malformed, unverifiable or unacceptable token: 401."""


class Forbidden(AuthError):
    """A verified identity that may not do this: 403."""


class AuthUnavailable(AuthError):
    """The keys to verify with couldn't be fetched: 503, not the caller's fault."""


@dataclass(frozen=True)
class Identity:
    principal_id: str
    scopes: frozenset[str]
    groups: frozenset[str] = field(default_factory=frozenset)


def normalized_claim_set(value: Any) -> frozenset[str]:
    """A scope/group claim as a set: absent is empty, a string is
    space-delimited (OAuth's `scope`), a list is taken as is (`scp`, `groups`,
    `roles` in most providers). Anything else is a malformed token."""
    if value is None:
        return frozenset()
    if isinstance(value, str):
        return frozenset(value.split())
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return frozenset(value)
    raise InvalidToken(f"claim must be a string or a list of strings, not {type(value).__name__}")


# ---- keys

class KeyProvider(Protocol):
    def get_key(self, token: str) -> Any:
        """The public key to verify `token` with (chosen by its `kid`)."""


class StaticKeyProvider:
    """A fixed public key, or {kid: key}. For tests and fixed deployments."""

    def __init__(self, keys):
        self.keys = keys

    def get_key(self, token: str) -> Any:
        if not isinstance(self.keys, dict):
            return self.keys
        kid = jwt.get_unverified_header(token).get("kid")
        if kid not in self.keys:
            raise InvalidToken("unknown signing key")
        return self.keys[kid]


class JwksKeyProvider:
    """The issuer's published keys, through one PyJWKClient: it caches the
    key set and refetches on an unknown `kid` (key rotation)."""

    def __init__(self, jwks_url: str, **client_kwargs):
        self.client = jwt.PyJWKClient(jwks_url, **client_kwargs)

    def get_key(self, token: str) -> Any:
        try:
            return self.client.get_signing_key_from_jwt(token).key
        except jwt.PyJWKClientConnectionError as e:
            raise AuthUnavailable(f"could not fetch signing keys: {e}") from None
        except jwt.PyJWTError as e:
            raise InvalidToken(str(e)) from None


# ---- tokens

class TokenVerifier:
    def __init__(self, issuer: str, audience: str, key_provider: KeyProvider,
                 algorithms=("RS256", "ES256"), leeway_s: float = 30,
                 scope_claim: str = "scope", groups_claim: str = "groups"):
        algorithms = tuple(algorithms)
        unsafe = [a for a in algorithms if a.upper().startswith("HS") or a.lower() == "none"]
        if not algorithms or unsafe:
            raise ValueError(f"only asymmetric signature algorithms are accepted, got {list(algorithms)}")
        if not issuer or not audience:
            raise ValueError("issuer and audience are required")
        self.issuer, self.audience, self.key_provider = issuer, audience, key_provider
        self.algorithms, self.leeway_s = algorithms, leeway_s
        self.scope_claim, self.groups_claim = scope_claim, groups_claim

    def verify(self, token: str) -> Identity:
        try:
            alg = jwt.get_unverified_header(token).get("alg")
        except jwt.PyJWTError as e:
            raise InvalidToken(f"malformed token: {e}") from None
        if alg not in self.algorithms:  # before the key lookup: no JWKS fetch for junk
            raise InvalidToken(f"algorithm {alg!r} is not accepted")
        key = self.key_provider.get_key(token)
        try:
            claims = jwt.decode(token, key, algorithms=list(self.algorithms), audience=self.audience,
                                issuer=self.issuer, leeway=self.leeway_s,
                                options={"require": list(REQUIRED_CLAIMS)})
        except jwt.PyJWTError as e:
            raise InvalidToken(str(e)) from None
        if not isinstance(claims["sub"], str) or not claims["sub"]:
            raise InvalidToken("sub must be a non-empty string")
        return Identity(principal_id=f"{claims['iss']}|{claims['sub']}",
                        scopes=normalized_claim_set(claims.get(self.scope_claim)),
                        groups=normalized_claim_set(claims.get(self.groups_claim)))


# ---- who may decide a proposal

class ApprovalPolicy(Protocol):
    def authorize(self, identity: Identity, proposal: dict, approved: bool) -> None:
        """Return to allow; raise Forbidden to refuse."""


class SreApprovalPolicy:
    """Approving a write needs membership of `group`. Rejecting one needs only
    the `runs:approve` scope: a rejection takes no action — the run escalates
    to a human — so anyone allowed to look at the gate may stop it."""

    def __init__(self, group: str = "sre"):
        self.group = group

    def authorize(self, identity: Identity, proposal: dict, approved: bool) -> None:
        if "runs:approve" not in identity.scopes:
            raise Forbidden("missing scope runs:approve")
        if approved and self.group not in identity.groups:
            raise Forbidden(f"approving {proposal.get('tool', 'this action')} requires group {self.group!r}")
