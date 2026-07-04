"""Cluster a user's preference-context embeddings into semantic groups/topics.

Usage:
    python manage.py cluster_user_preferences --user-id 42
    python manage.py cluster_user_preferences --all
    python manage.py cluster_user_preferences --user-id 42 --method dbscan --eps 0.4
    python manage.py cluster_user_preferences --user-id 42 --method kmeans --n-clusters 4

Runs K-Means (default) or DBSCAN over that user's `PreferenceContext.semantic_vec`
rows via the embedding service's `/v1/cluster` endpoint
(`machinelearning/utils/clustering.py` -- no clustering logic is reimplemented
here), and writes the resulting cluster label back onto each row's
`representation_meta["cluster"]` so:

  * the app can group/label a user's preferences by theme ("semantic grouping" /
    "topic discovery" / "user preference clustering" from the ML integration spec),
  * and so retrieval/analytics code can filter or explain results by cluster
    without recomputing clustering on every read.

This is an offline/batch operation (like `derive_user_preferences`), not run
inline during a request.
"""

from __future__ import annotations

import json
import logging
from typing import List

from django.core.management.base import BaseCommand, CommandError

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Cluster a user's PreferenceContext embeddings (K-Means/DBSCAN) and label them by cluster."

    def add_arguments(self, parser):
        parser.add_argument("--user-id", type=int, default=None, help="User to process. Required unless --all.")
        parser.add_argument("--all", action="store_true", help="Process every user with preference contexts.")
        parser.add_argument("--method", type=str, default="kmeans", choices=["kmeans", "dbscan"])
        parser.add_argument("--n-clusters", type=int, default=3, help="K-Means cluster count.")
        parser.add_argument("--eps", type=float, default=0.5, help="DBSCAN neighborhood radius.")
        parser.add_argument("--min-samples", type=int, default=2, help="DBSCAN minimum samples per cluster.")

    def handle(self, *args, **options):
        from api.models import User
        from vectors.ml_bridge import cluster_vectors
        from vectors.models import PreferenceContext

        user_id = options.get("user_id")
        if not user_id and not options.get("all"):
            raise CommandError("Pass --user-id <id> or --all")

        user_ids: List[int]
        if options.get("all"):
            user_ids = list(
                PreferenceContext.objects.values_list("user_id", flat=True).distinct()
            )
        else:
            user_ids = [int(user_id)]
            if not User.objects.filter(id=user_ids[0]).exists():
                raise CommandError(f"No such user: {user_ids[0]}")

        method = options["method"]
        results = []

        for uid in user_ids:
            contexts = list(
                PreferenceContext.objects.filter(user_id=uid).exclude(semantic_vec=None)
            )
            vectors = [c.semantic_vec for c in contexts if c.semantic_vec]
            if len(vectors) < 2:
                self.stdout.write(self.style.WARNING(f"user {uid}: fewer than 2 embedded contexts, skipping"))
                continue

            cluster_result = cluster_vectors(
                vectors,
                method=method,
                n_clusters=options["n_clusters"],
                eps=options["eps"],
                min_samples=options["min_samples"],
            )
            if not cluster_result:
                self.stderr.write(
                    self.style.ERROR(f"user {uid}: embedding service unavailable for clustering")
                )
                continue

            labels = cluster_result.get("labels") or []
            if len(labels) != len(contexts):
                self.stderr.write(
                    self.style.ERROR(f"user {uid}: label count {len(labels)} != context count {len(contexts)}")
                )
                continue

            updated = 0
            for ctx, label in zip(contexts, labels):
                meta = dict(ctx.representation_meta or {})
                meta["cluster"] = {
                    "label": int(label),
                    "method": cluster_result.get("method"),
                    "silhouette": cluster_result.get("silhouette"),
                    "meaningful": cluster_result.get("meaningful"),
                }
                ctx.representation_meta = meta
                ctx.save(update_fields=["representation_meta"])
                updated += 1

            n_clusters = cluster_result.get("n_clusters")
            self.stdout.write(
                self.style.SUCCESS(
                    f"user {uid}: labeled {updated} contexts into {n_clusters} {method} cluster(s) "
                    f"(silhouette={cluster_result.get('silhouette'):.3f})"
                )
            )
            results.append({"user_id": uid, "updated": updated, "n_clusters": n_clusters})

        self.stdout.write(json.dumps({"results": results}, default=str))
