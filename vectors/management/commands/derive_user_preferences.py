"""Derive per-user preference contexts and steering vectors.

Usage:
    python manage.py derive_user_preferences --user-id 42
    python manage.py derive_user_preferences --all                  # backfill
    python manage.py derive_user_preferences --user-id 42 --source clinical_update --content "..."

The command splits the user's onboarding personalization profile and any
existing in-DB context content into discrete preference contexts, runs each
through `vectors.steering_workflow.derive_user_steering`, and persists one
`PreferenceContext` (with `semantic_vec`) plus one `SteeringVector` per
input context into the `vectors` DB.

This is the "Phase 2 worker" referenced from the construction overview.
"""

from __future__ import annotations

import json
import logging
from typing import List

from django.core.management.base import BaseCommand, CommandError

logger = logging.getLogger(__name__)


def _profile_contexts_for_user(user_id: int) -> List[tuple[str, str]]:
    """Return a list of (source, content) tuples drawn from the user's profile."""
    from api.models import PersonalizationProfile, Symptom, Question

    out: List[tuple[str, str]] = []
    try:
        profile = PersonalizationProfile.objects.get(user_id=user_id)
    except PersonalizationProfile.DoesNotExist:
        profile = None

    if profile:
        if profile.main_reason:
            out.append(("signup", f"Reason for upcoming care: {profile.main_reason}"))
        if profile.biggest_concern:
            out.append(("signup", f"Biggest concern: {profile.biggest_concern}"))
        if profile.condition_stage:
            out.append(("signup", f"Condition stage: {profile.condition_stage}"))
        if profile.appointment_outcome:
            out.append(("signup", f"Desired outcome: {profile.appointment_outcome}"))
        if profile.prepared_items:
            out.append(("signup", "Prepared items: " + ", ".join(map(str, profile.prepared_items))))

    # Folding in lightweight historical signals helps PCA find a meaningful axis.
    for sym in Symptom.objects.filter(user_id=user_id).order_by("-created_at")[:8]:
        flags = []
        if sym.is_new:
            flags.append("new")
        if sym.is_worsening:
            flags.append("worsening")
        flag_text = f" ({', '.join(flags)})" if flags else ""
        out.append(("history", f"Symptom {sym.name} severity {sym.severity}{flag_text}"))

    for q in Question.objects.filter(user_id=user_id).order_by("-created_at")[:8]:
        out.append(("history", f"Question to ask: {q.text}"))

    return out


def _persist(user_id: int, source: str, content: str, derivation) -> tuple[str, str]:
    from vectors.models import PreferenceContext, SteeringVector

    ctx = PreferenceContext.objects.using("vectors").create(
        user_id=user_id,
        source=source,
        content=content,
        semantic_vec=derivation.semantic_vec,
        representation_meta={
            "pca_explained_variance": derivation.pca_explained_variance,
            "umap_silhouette": derivation.umap_silhouette,
            "umap_meaningful": derivation.umap_meaningful,
            "extract_layer": derivation.layer,
            "diagnostics": derivation.diagnostics,
        },
    )
    sv = SteeringVector.objects.using("vectors").create(
        user_id=user_id,
        context_id=ctx.id,
        layer=derivation.layer,
        steering_vec=derivation.steering_vec,
        norm=derivation.norm,
    )
    return str(ctx.id), str(sv.id)


class Command(BaseCommand):
    help = "Derive PreferenceContext + SteeringVector rows from a user's onboarding profile + history."

    def add_arguments(self, parser):
        parser.add_argument("--user-id", type=int, default=None, help="User to process. Required unless --all.")
        parser.add_argument("--all", action="store_true", help="Backfill every user with a personalization profile.")
        parser.add_argument(
            "--source",
            type=str,
            default=None,
            help="Override source label (e.g. clinical_update). Used together with --content.",
        )
        parser.add_argument(
            "--content",
            type=str,
            default=None,
            help="Single explicit content string to process (skips profile/history extraction).",
        )
        parser.add_argument(
            "--layer",
            type=int,
            default=None,
            help="Override the LLM activation layer (defaults to LLM_DEFAULT_EXTRACT_LAYER).",
        )

    def handle(self, *args, **options):
        from api.models import User
        from vectors.steering_workflow import derive_user_steering

        user_id = options.get("user_id")
        if not user_id and not options.get("all"):
            raise CommandError("Pass --user-id <id> or --all")

        user_ids: List[int]
        if options.get("all"):
            user_ids = list(User.objects.values_list("id", flat=True))
        else:
            user_ids = [int(user_id)]

        layer = options.get("layer")
        explicit_source = options.get("source")
        explicit_content = options.get("content")

        results = []
        for uid in user_ids:
            if explicit_content:
                contexts = [(explicit_source or "manual", explicit_content)]
            else:
                contexts = _profile_contexts_for_user(uid)
            if not contexts:
                self.stdout.write(self.style.WARNING(f"user {uid}: no contexts to derive"))
                continue

            try:
                derived = derive_user_steering([c[1] for c in contexts], layer=layer)
            except Exception as exc:
                logger.exception("derivation failed for user %s", uid)
                self.stderr.write(self.style.ERROR(f"user {uid}: derivation failed: {exc}"))
                continue

            persisted = []
            for (source, content), d in zip(contexts, derived):
                try:
                    pair = _persist(uid, source, content, d)
                    persisted.append(pair)
                except Exception as exc:
                    logger.exception("persist failed for user %s", uid)
                    self.stderr.write(self.style.ERROR(f"user {uid}: persist failed: {exc}"))

            self.stdout.write(
                self.style.SUCCESS(
                    f"user {uid}: derived {len(persisted)} contexts (layer={derived[0].layer if derived else 'n/a'})"
                )
            )
            results.append({"user_id": uid, "persisted": persisted})

        self.stdout.write(json.dumps({"results": results}, default=str))
