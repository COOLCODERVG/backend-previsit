from __future__ import annotations

from django.conf import settings


class NeuraViaDbRouter:
    """
    Multi-database routing to support the \"3 RDS\" split:
    - default: core clinical/user data
    - vectors: pgvector tables (semantic + steering vectors)
    - documents: PDF metadata

    Routing is based on Django app_label. If the secondary databases are not configured,
    Django will fall back to the default DB for all operations.
    """

    APP_TO_DB = {
        "vectors": "vectors",
        "documents": "documents",
    }

    def _target_db(self, app_label: str) -> str:
        preferred = self.APP_TO_DB.get(app_label)
        if preferred and preferred in settings.DATABASES:
            return preferred
        return "default"

    def db_for_read(self, model, **hints):
        return self._target_db(model._meta.app_label)

    def db_for_write(self, model, **hints):
        return self._target_db(model._meta.app_label)

    def allow_relation(self, obj1, obj2, **hints):
        # Allow relations within the same DB group; disallow cross-DB FKs.
        db1 = self._target_db(obj1._meta.app_label)
        db2 = self._target_db(obj2._meta.app_label)
        return db1 == db2

    def allow_migrate(self, db, app_label, model_name=None, **hints):
        target = self._target_db(app_label)
        return db == target

