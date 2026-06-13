"""DRF authentication backend that validates AWS Cognito ID/access tokens.

The mobile client signs in via the Cognito hosted UI using OAuth 2.0
Authorization Code with PKCE (no client secret). Cognito returns an RS256-signed
JWT whose `kid` references one of the public keys exposed at the user pool's
JWKS endpoint. We download the JWKS once, cache it, and use the matching key
to verify every request.

Configuration (read from environment / Secrets Manager):
    COGNITO_USER_POOL_ID    e.g. ap-south-1_AbCdEfGhI
    COGNITO_APP_CLIENT_ID   audience claim (the mobile client id)
    COGNITO_ISSUER          https://cognito-idp.<region>.amazonaws.com/<pool>
    COGNITO_JWKS_URL        <issuer>/.well-known/jwks.json
    AWS_REGION              fallback for deriving the issuer if not set

If `COGNITO_USER_POOL_ID` is unset we treat the auth class as a no-op (returns
None), so the legacy `JWTAuthentication` (SimpleJWT) keeps working during the
migration window.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.request
from typing import Any, Optional

from django.contrib.auth import get_user_model
from rest_framework import authentication, exceptions

logger = logging.getLogger(__name__)


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name, default) or "").strip()


def _settings() -> dict[str, str]:
    pool_id = _env("COGNITO_USER_POOL_ID")
    region = _env("AWS_REGION", "us-east-1")
    issuer = _env("COGNITO_ISSUER") or (
        f"https://cognito-idp.{region}.amazonaws.com/{pool_id}" if pool_id else ""
    )
    jwks_url = _env("COGNITO_JWKS_URL") or (
        f"{issuer}/.well-known/jwks.json" if issuer else ""
    )
    return {
        "pool_id": pool_id,
        "region": region,
        "issuer": issuer,
        "jwks_url": jwks_url,
        "audience": _env("COGNITO_AUDIENCE") or _env("COGNITO_APP_CLIENT_ID"),
        "client_id": _env("COGNITO_APP_CLIENT_ID"),
    }


class _JwksCache:
    """Tiny TTL cache for the Cognito JWKS document."""

    def __init__(self, ttl_seconds: int = 3600):
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        self._jwks: dict[str, Any] | None = None
        self._fetched_at: float = 0.0

    def get(self, jwks_url: str) -> dict[str, Any]:
        with self._lock:
            now = time.monotonic()
            if self._jwks is not None and (now - self._fetched_at) < self._ttl:
                return self._jwks
            try:
                req = urllib.request.Request(
                    jwks_url, headers={"User-Agent": "NeuraVia/1.0"}
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    raw = resp.read()
                self._jwks = json.loads(raw)
                self._fetched_at = now
                return self._jwks
            except Exception as exc:  # noqa: BLE001
                logger.warning("Cognito JWKS fetch failed: %s", exc)
                if self._jwks is not None:
                    return self._jwks
                raise


_jwks_cache = _JwksCache()


class CognitoAuthentication(authentication.BaseAuthentication):
    """Validate Cognito-issued JWT bearer tokens.

    On success we either return the existing Django user matched by the
    Cognito `sub` (stored in `User.email` for now, to keep models.py untouched)
    or transparently provision one. PHI is *not* in the token, so this is
    safe to log at INFO level.
    """

    keyword = "Bearer"

    def authenticate(self, request) -> Optional[tuple[Any, str]]:
        cfg = _settings()
        if not cfg["pool_id"] or not cfg["jwks_url"]:
            return None  # Cognito not configured — defer to next auth class.

        auth_header = authentication.get_authorization_header(request).decode("utf-8")
        if not auth_header:
            return None
        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != self.keyword.lower():
            return None
        token = parts[1]

        try:
            import jwt  # PyJWT
            from jwt import PyJWKClient
        except ImportError as exc:  # pragma: no cover
            logger.error("PyJWT not installed; cannot validate Cognito tokens: %s", exc)
            return None

        try:
            unverified_header = jwt.get_unverified_header(token)
        except jwt.PyJWTError as exc:
            raise exceptions.AuthenticationFailed(f"Malformed JWT: {exc}") from exc

        kid = unverified_header.get("kid")
        if not kid:
            raise exceptions.AuthenticationFailed("JWT missing kid")

        # Use PyJWKClient when available; fall back to manual JWKS lookup.
        signing_key = None
        try:
            jwks_client = PyJWKClient(cfg["jwks_url"], cache_keys=True, lifespan=3600)
            signing_key = jwks_client.get_signing_key_from_jwt(token).key
        except Exception:  # noqa: BLE001
            jwks = _jwks_cache.get(cfg["jwks_url"])
            keys = {k["kid"]: k for k in jwks.get("keys", [])}
            jwk_dict = keys.get(kid)
            if not jwk_dict:
                raise exceptions.AuthenticationFailed("Unknown JWT kid")
            signing_key = jwt.algorithms.RSAAlgorithm.from_jwk(jwk_dict)

        try:
            options = {"verify_aud": bool(cfg["audience"])}
            decoded = jwt.decode(
                token,
                signing_key,
                algorithms=["RS256"],
                audience=cfg["audience"] or None,
                issuer=cfg["issuer"],
                options=options,
            )
        except jwt.ExpiredSignatureError as exc:
            raise exceptions.AuthenticationFailed("Token expired") from exc
        except jwt.InvalidTokenError as exc:
            raise exceptions.AuthenticationFailed(f"Invalid token: {exc}") from exc

        token_use = decoded.get("token_use")
        if token_use not in {"id", "access"}:
            raise exceptions.AuthenticationFailed(
                f"Unsupported token_use: {token_use}"
            )

        # Access tokens carry `client_id`; id tokens carry `aud` (already verified).
        if token_use == "access" and cfg["client_id"]:
            if decoded.get("client_id") != cfg["client_id"]:
                raise exceptions.AuthenticationFailed("client_id mismatch")

        sub = decoded.get("sub")
        email = (decoded.get("email") or "").lower()
        name = decoded.get("name") or decoded.get("preferred_username") or ""
        if not sub:
            raise exceptions.AuthenticationFailed("JWT missing sub")

        user = self._get_or_create_user(sub=sub, email=email, name=name)
        return (user, token)

    def authenticate_header(self, request) -> str:
        return self.keyword

    @staticmethod
    def _get_or_create_user(*, sub: str, email: str, name: str):
        UserModel = get_user_model()
        # Prefer matching by email (stable identifier in our schema). Fall back
        # to a synthesized email built from the Cognito sub so we never collide.
        synthetic_email = f"{sub}@cognito.local"
        lookup_email = email or synthetic_email
        try:
            user = UserModel.objects.get(email=lookup_email)
        except UserModel.DoesNotExist:
            user = UserModel.objects.create_user(
                email=lookup_email,
                password=None,
                name=name or "",
                role="user",
            )
            # Mark the password unusable so the legacy email/password flow
            # cannot be hijacked via this account.
            try:
                user.set_unusable_password()
                user.save(update_fields=["password"])
            except Exception:  # noqa: BLE001
                logger.warning("Could not mark Cognito user password unusable")
        return user
