import os
from pathlib import Path
from dotenv import load_dotenv
from datetime import timedelta
from urllib.parse import urlparse

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / '.env')


def _env_bool(name: str, default: str = 'False') -> bool:
    return os.environ.get(name, default).strip().lower() in {'1', 'true', 'yes'}


def _required_env(name: str) -> str:
    value = os.environ.get(name, '').strip()
    if not value:
        raise RuntimeError(
            f'Missing required environment variable: {name}. '
            'See backend/.env.example for the required values.'
        )
    return value


def _optional_list(name: str) -> list[str]:
    return [item.strip() for item in os.environ.get(name, '').split(',') if item.strip()]


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / '.env')

DEBUG = _env_bool('DEBUG', 'False')

SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY', '').strip()
if not SECRET_KEY:
    raise RuntimeError(
        'Missing required environment variable: DJANGO_SECRET_KEY. '
        'Copy backend/.env.example to backend/.env and set it before starting the app.'
    )

if DEBUG:
    ALLOWED_HOSTS = ['*']
else:
    ALLOWED_HOSTS = _optional_list('ALLOWED_HOSTS')
    if not ALLOWED_HOSTS:
        raise RuntimeError(
            'Missing required environment variable: ALLOWED_HOSTS. '
            'Provide at least one hostname for production.'
        )

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
    if parsed.scheme not in {'postgres', 'postgresql'}:
        raise ValueError('Only postgres URLs are supported for DATABASE_URL in this project')

    sslmode = os.environ.get('DB_SSLMODE', '').strip() or 'prefer'
    sslrootcert = os.environ.get('DB_SSLROOTCERT', '').strip()
    options: dict[str, str] = {'sslmode': sslmode}

    if sslrootcert and Path(sslrootcert).is_file():
        options['sslrootcert'] = sslrootcert
    elif sslmode in {'verify-ca', 'verify-full'} and not sslrootcert:
        import logging
        logging.getLogger(__name__).warning(
            'DB_SSLMODE=%s requested but DB_SSLROOTCERT not set; falling back to sslmode=require',
            sslmode,
        )
        options['sslmode'] = 'require'

    return {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': (parsed.path or '').lstrip('/'),
        'USER': parsed.username or '',
        'PASSWORD': parsed.password or '',
        'HOST': parsed.hostname or '',
        'PORT': str(parsed.port or ''),
        'CONN_MAX_AGE': int(os.environ.get('DB_CONN_MAX_AGE', '60')),
        'OPTIONS': options,
    }


CORE_DATABASE_URL = os.environ.get('CORE_DATABASE_URL', '').strip()

if CORE_DATABASE_URL:
    DATABASES = {'default': _database_from_url(CORE_DATABASE_URL)}
else:
    db_sslmode = os.environ.get('DB_SSLMODE', 'require').strip()
    db_sslrootcert = os.environ.get('DB_SSLROOTCERT', '').strip()
    db_options: dict[str, str] = {'sslmode': db_sslmode}
    if db_sslrootcert and Path(db_sslrootcert).is_file():
        db_options['sslrootcert'] = db_sslrootcert
    elif db_sslmode in {'verify-ca', 'verify-full'} and not db_sslrootcert:
        import logging
        logging.getLogger(__name__).warning(
            'DB_SSLMODE=%s requested but DB_SSLROOTCERT not set; falling back to sslmode=require',
            db_sslmode,
        )
        db_options['sslmode'] = 'require'

    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.postgresql',
            'NAME': _required_env('DB_NAME'),
            'USER': _required_env('DB_USER'),
            'PASSWORD': _required_env('DB_PASSWORD'),
            'HOST': _required_env('DB_HOST'),
            'PORT': _required_env('DB_PORT'),
            'OPTIONS': db_options,
        }
    }

AUTH_USER_MODEL = 'api.User'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

CORS_ALLOW_CREDENTIALS = True
if DEBUG:
    CORS_ALLOW_ALL_ORIGINS = True
else:
    CORS_ALLOW_ALL_ORIGINS = False
    CORS_ALLOWED_ORIGINS = _optional_list('CORS_ALLOWED_ORIGINS')

REST_FRAMEWORK = {
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

SIMPLE_JWT = {
    'ACCESS_TOKEN_LIFETIME': timedelta(hours=24),
    'REFRESH_TOKEN_LIFETIME': timedelta(days=7),
    'ROTATE_REFRESH_TOKENS': True,
    'BLACKLIST_AFTER_ROTATION': True,
    'AUTH_HEADER_TYPES': ('Bearer',),
    'USER_ID_FIELD': 'id',
    'USER_ID_CLAIM': 'sub',
}

if os.environ.get('JWT_SECRET'):
    SIMPLE_JWT['SIGNING_KEY'] = os.environ.get('JWT_SECRET')

TEMPLATES = []
WSGI_APPLICATION = 'previsit.wsgi.application'
ASGI_APPLICATION = 'previsit.asgi.application'

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_TZ = True

ADMIN_EMAIL = os.environ.get('ADMIN_EMAIL', 'hello@gmail.com')
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'hello')

DATA_UPLOAD_MAX_MEMORY_SIZE = 52428800

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

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
