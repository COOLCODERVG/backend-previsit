import os
from pathlib import Path
from dotenv import load_dotenv
from datetime import timedelta
from urllib.parse import urlparse

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / '.env')

# Prefer an explicit Django secret; fall back to legacy JWT_SECRET for backwards compatibility.
SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY') or os.environ.get('JWT_SECRET') or 'django-insecure-fallback-key-change-me'

DEBUG = True

ALLOWED_HOSTS = ['*']

INSTALLED_APPS = [
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'corsheaders',
    'rest_framework',
    'rest_framework_simplejwt.token_blacklist',
    'api',
    'vectors',
    'medications',
    'documents',
    'notifications',
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.common.CommonMiddleware',
    'previsit.middleware.StructuredAccessLogMiddleware',
]

ROOT_URLCONF = 'previsit.urls'

def _database_from_url(url: str):
    parsed = urlparse(url)
    if parsed.scheme not in {"postgres", "postgresql"}:
        raise ValueError("Only postgres URLs are supported for DATABASE_URL in this project")

    sslmode = os.environ.get("DB_SSLMODE", "").strip() or "prefer"
    sslrootcert = (os.environ.get("DB_SSLROOTCERT", "") or "").strip()
    options: dict[str, str] = {"sslmode": sslmode}

    # If a CA bundle is configured (e.g. AWS RDS global-bundle.pem in the Docker image)
    # and the file actually exists, pass it to psycopg as `sslrootcert`. This is required
    # for sslmode=verify-ca / verify-full.
    if sslrootcert and Path(sslrootcert).is_file():
        options["sslrootcert"] = sslrootcert
    elif sslmode in {"verify-ca", "verify-full"} and not sslrootcert:
        # Don't crash startup; downgrade to `require` and log a warning so the app
        # still boots if the cert file isn't mounted in this environment.
        import logging
        logging.getLogger(__name__).warning(
            "DB_SSLMODE=%s requested but DB_SSLROOTCERT not set; falling back to sslmode=require",
            sslmode,
        )
        options["sslmode"] = "require"

    return {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": (parsed.path or "").lstrip("/"),
        "USER": parsed.username or "",
        "PASSWORD": parsed.password or "",
        "HOST": parsed.hostname or "",
        "PORT": str(parsed.port or ""),
        "CONN_MAX_AGE": int(os.environ.get("DB_CONN_MAX_AGE", "60")),
        "OPTIONS": options,
    }


CORE_DATABASE_URL = os.environ.get("CORE_DATABASE_URL", "").strip()

# Single unified database — user/application data, personalization vectors
# (pgvector-backed, see the `vectors` app), and document/export metadata (see
# the `documents` app) all live as separate tables in this one connection.
if CORE_DATABASE_URL:
    DATABASES = {"default": _database_from_url(CORE_DATABASE_URL)}
else:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": os.environ.get("DB_NAME", "syniviadb"),
            "USER": os.environ.get("DB_USER", "syniviaadmin"),
            "PASSWORD": os.environ.get("DB_PASSWORD", ""),
            "HOST": os.environ.get("DB_HOST", "syniviadb.crk4uo2kehfn.us-west-2.rds.amazonaws.com"),
            "PORT": os.environ.get("DB_PORT", "5432"),
            "OPTIONS": {
                "sslmode": os.environ.get("DB_SSLMODE", "require"),
            },
        }
    }

AUTH_USER_MODEL = 'api.User'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# CORS
# Security: allow all origins only in DEBUG/dev. In production, set CORS_ALLOWED_ORIGINS.
CORS_ALLOW_CREDENTIALS = True
if DEBUG:
    CORS_ALLOW_ALL_ORIGINS = True
else:
    CORS_ALLOW_ALL_ORIGINS = False
    allowed = os.environ.get('CORS_ALLOWED_ORIGINS', '')
    CORS_ALLOWED_ORIGINS = [s.strip() for s in allowed.split(',') if s.strip()]

# REST Framework — JSON-only (no django.template; avoids Browsable API HTML errors).
REST_FRAMEWORK = {
    # Cognito (OIDC + OAuth2 + JWT) is the production auth path. The legacy
    # SimpleJWT class is kept second so existing email/password sessions keep
    # working during the migration window.
    'DEFAULT_AUTHENTICATION_CLASSES': (
        'api.auth_oidc.CognitoAuthentication',
        'rest_framework_simplejwt.authentication.JWTAuthentication',
    ),
    'DEFAULT_PERMISSION_CLASSES': (
        'rest_framework.permissions.IsAuthenticated',
    ),
    'DEFAULT_RENDERER_CLASSES': (
        'rest_framework.renderers.JSONRenderer',
    ),
    'EXCEPTION_HANDLER': 'api.utils.custom_exception_handler',
}

# Simple JWT (legacy email/password path)
SIMPLE_JWT = {
    'ACCESS_TOKEN_LIFETIME': timedelta(hours=24),
    'REFRESH_TOKEN_LIFETIME': timedelta(days=7),
    'ROTATE_REFRESH_TOKENS': True,
    'BLACKLIST_AFTER_ROTATION': True,
    'AUTH_HEADER_TYPES': ('Bearer',),
    'USER_ID_FIELD': 'id',
    'USER_ID_CLAIM': 'sub',
}

# If an explicit JWT secret is provided, use it as SimpleJWT signing key (keeps symmetric JWTs separate)
if os.environ.get('JWT_SECRET'):
    SIMPLE_JWT['SIGNING_KEY'] = os.environ.get('JWT_SECRET')

# Disable unused middleware/templates for API-only project
TEMPLATES = []
WSGI_APPLICATION = 'previsit.wsgi.application'
ASGI_APPLICATION = 'previsit.asgi.application'

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_TZ = True

# Admin credentials from .env
ADMIN_EMAIL = os.environ.get('ADMIN_EMAIL', 'hello@gmail.com')
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'hello')

# Max upload size for audio recordings (50MB)
DATA_UPLOAD_MAX_MEMORY_SIZE = 52428800

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
