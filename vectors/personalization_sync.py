from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from django.conf import settings
from django.db import transaction

from api.models import PersonalizationProfile, Question, Recording, Symptom
from medications.models import Medication

from .models import PreferenceContext, SteeringVector
from .steering_workflow import derive_user_steering

logger = logging.getLogger(__name__)


def _vectors_alias() -> str:
    return "vectors" if "vectors" in settings.DATABASES else "default"


def _dedupe_contexts(items: Iterable[Tuple[str, str]]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    seen: set[str] = set()
    for source, content in items:
        text = (content or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(((source or "profile").strip() or "profile", text))
    return out


def build_user_contexts(user_id: int, *, extra_contexts: Sequence[str] | None = None) -> List[Tuple[str, str]]:
    contexts: List[Tuple[str, str]] = []

    profile = PersonalizationProfile.objects.filter(user_id=user_id).first()
    if profile:
        if profile.main_reason:
            contexts.append(("profile", f"Main reason for care: {profile.main_reason}"))
        if profile.condition_stage:
            contexts.append(("profile", f"Condition stage: {profile.condition_stage}"))
        if profile.biggest_concern:
            contexts.append(("profile", f"Biggest concern: {profile.biggest_concern}"))
        if profile.appointment_outcome:
            contexts.append(("profile", f"Desired appointment outcome: {profile.appointment_outcome}"))
        if profile.prepared_items:
            contexts.append(("profile", "Prepared items: " + ", ".join(map(str, profile.prepared_items))))
        if profile.family_history:
            contexts.append(("profile", f"Family history: {profile.family_history}"))
        for item in profile.ml_preferences or []:
            if not isinstance(item, dict):
                continue
            q = str(item.get("question") or "").strip()
            a = str(item.get("answer") or "").strip()
            if q:
                contexts.append(("preference", f"{q} {a}".strip()))

    for symptom in Symptom.objects.filter(user_id=user_id).order_by("-created_at")[:12]:
        flags: List[str] = []
        if symptom.is_new:
            flags.append("new")
        if symptom.is_worsening:
            flags.append("worsening")
        flag_text = f" ({', '.join(flags)})" if flags else ""
        contexts.append(("history", f"Symptom: {symptom.name}, severity {symptom.severity}{flag_text}"))

    for question in Question.objects.filter(user_id=user_id).order_by("-created_at")[:12]:
        label = "Unanswered question" if not question.is_answered else "Question"
        contexts.append(("history", f"{label}: {question.text}"))

    for medication in Medication.objects.filter(user_id=user_id, status="active").order_by("-updated_at")[:12]:
        parts = [f"Medication: {medication.name}"]
        if medication.dose:
            parts.append(f"dose {medication.dose}")
        if medication.frequency:
            parts.append(f"frequency {medication.frequency}")
        if medication.route:
            parts.append(f"route {medication.route}")
        contexts.append(("medication", ", ".join(parts)))

    recent_recordings = Recording.objects.filter(user_id=user_id, status="extracted").order_by("-updated_at")[:8]
    for rec in recent_recordings:
        entities = rec.extracted_entities if isinstance(rec.extracted_entities, dict) else {}
        for inst in entities.get("instructions") or []:
            if isinstance(inst, dict):
                text = (inst.get("text") or "").strip()
            else:
                text = str(inst or "").strip()
            if text:
                contexts.append(("visit_extract", f"Provider instruction: {text}"))

    for text in extra_contexts or []:
        t = str(text or "").strip()
        if t:
            contexts.append(("manual", t))

    return _dedupe_contexts(contexts)


def sync_user_preference_vectors(
    *,
    user_id: int,
    extra_contexts: Sequence[str] | None = None,
) -> Dict[str, Any]:
    contexts = build_user_contexts(user_id, extra_contexts=extra_contexts)
    if not contexts:
        return {
            "ok": False,
            "message": "no_contexts",
            "contexts_created": 0,
            "steering_created": 0,
        }

    derived = derive_user_steering([c[1] for c in contexts])
    if not derived:
        return {
            "ok": False,
            "message": "derivation_empty",
            "contexts_created": 0,
            "steering_created": 0,
        }

    alias = _vectors_alias()
    created_contexts = 0
    created_steering = 0
    with transaction.atomic(using=alias):
        PreferenceContext.objects.using(alias).filter(user_id=user_id).delete()
        SteeringVector.objects.using(alias).filter(user_id=user_id).delete()

        for (source, content), d in zip(contexts, derived):
            ctx = PreferenceContext.objects.using(alias).create(
                user_id=user_id,
                source=source,
                content=content,
                semantic_vec=d.semantic_vec,
                representation_meta={
                    "pca_explained_variance": d.pca_explained_variance,
                    "umap_silhouette": d.umap_silhouette,
                    "umap_meaningful": d.umap_meaningful,
                    "extract_layer": d.layer,
                    "diagnostics": d.diagnostics,
                },
            )
            created_contexts += 1

            SteeringVector.objects.using(alias).create(
                user_id=user_id,
                context_id=ctx.id,
                layer=d.layer,
                steering_vec=d.steering_vec,
                norm=d.norm,
            )
            created_steering += 1

    return {
        "ok": True,
        "message": "synced",
        "contexts_created": created_contexts,
        "steering_created": created_steering,
        "layer": int(derived[0].layer),
    }


def get_user_vector_status(*, user_id: int) -> Dict[str, Any]:
    alias = _vectors_alias()
    context_qs = PreferenceContext.objects.using(alias).filter(user_id=user_id)
    steering_qs = SteeringVector.objects.using(alias).filter(user_id=user_id)
    latest_ctx = context_qs.order_by("-created_at").first()
    latest_sv = steering_qs.order_by("-created_at").first()

    semantic_dim = 0
    if latest_ctx is not None:
        raw = latest_ctx.semantic_vec
        if isinstance(raw, list):
            semantic_dim = len(raw)
        else:
            try:
                semantic_dim = len(raw)  # pgvector can come back as array-like
            except Exception:
                semantic_dim = 0

    return {
        "enabled": True,
        "db_alias": alias,
        "contexts": context_qs.count(),
        "steering_vectors": steering_qs.count(),
        "semantic_dim": semantic_dim,
        "latest_context_at": latest_ctx.created_at if latest_ctx else None,
        "latest_steering_at": latest_sv.created_at if latest_sv else None,
        "latest_layer": int(latest_sv.layer) if latest_sv else None,
    }
