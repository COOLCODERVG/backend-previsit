from __future__ import annotations


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

    def db_for_read(self, model, **hints):
        return self.APP_TO_DB.get(model._meta.app_label)

    def db_for_write(self, model, **hints):
        return self.APP_TO_DB.get(model._meta.app_label)

    def allow_relation(self, obj1, obj2, **hints):
        # Allow relations within the same DB group; disallow cross-DB FKs.
        db1 = self.APP_TO_DB.get(obj1._meta.app_label, "default")
        db2 = self.APP_TO_DB.get(obj2._meta.app_label, "default")
        return db1 == db2

    def allow_migrate(self, db, app_label, model_name=None, **hints):
        target = self.APP_TO_DB.get(app_label, "default")
        return db == target

