"""
For one-off export tasks against the legacy SQLite DB.

This module intentionally forces SQLite regardless of CORE_DATABASE_URL so that
`dumpdata` can reliably read from `backend/db.sqlite3` during migrations.
"""

from .settings import *  # noqa

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

