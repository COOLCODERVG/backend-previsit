"""Voice-transcript → structured-clinical-data pipeline.

Wraps **AWS Comprehend Medical** (HIPAA-eligible). Three calls:

* ``DetectEntitiesV2`` — produces medical NER + relation extraction in one
  go. We use it for symptoms, conditions, anatomy, and provider instructions.
* ``InferRxNorm`` — normalizes medication mentions to RxNorm CUIs, with the
  candidate list ranked by score.
* ``InferICD10CM`` — normalizes condition mentions to ICD-10-CM codes.

Confidence thresholds determine whether a downstream record is auto-confirmed
or queued for user review (``MedicationReconciliationService``).

The transcript text is processed in 20k-char chunks because Comprehend Medical
rejects requests above its limit; for typical 30-minute appointments a single
call is plenty.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_MED_AUTO_THRESHOLD = float(os.environ.get("CM_AUTO_CONFIRM_THRESHOLD", "0.85"))
COMPREHEND_MAX_BYTES = 20000


def _client():
    import boto3

    return boto3.client(
        "comprehendmedical",
        region_name=(os.environ.get("AWS_REGION") or "us-east-1").strip(),
    )


@dataclass
class MedicationCandidate:
    name: str
    rxnorm_code: str = ""
    rxnorm_description: str = ""
    dose: str = ""
    frequency: str = ""
    route: str = ""
    duration: str = ""
    confidence: float = 0.0
    text_span: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    def auto_confirm(self) -> bool:
        return self.confidence >= DEFAULT_MED_AUTO_THRESHOLD


@dataclass
class SymptomCandidate:
    name: str
    severity: str = ""
    confidence: float = 0.0
    text_span: str = ""


@dataclass
class ConditionCandidate:
    name: str
    icd10_code: str = ""
    icd10_description: str = ""
    confidence: float = 0.0
    text_span: str = ""


@dataclass
class InstructionCandidate:
    text: str
    confidence: float = 0.0


@dataclass
class ExtractionResult:
    medications: List[MedicationCandidate] = field(default_factory=list)
    symptoms: List[SymptomCandidate] = field(default_factory=list)
    conditions: List[ConditionCandidate] = field(default_factory=list)
    instructions: List[InstructionCandidate] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "medications": [m.__dict__ for m in self.medications],
            "symptoms": [s.__dict__ for s in self.symptoms],
            "conditions": [c.__dict__ for c in self.conditions],
            "instructions": [i.__dict__ for i in self.instructions],
            "codes": {
                "rxnorm": [m.rxnorm_code for m in self.medications if m.rxnorm_code],
                "icd10": [c.icd10_code for c in self.conditions if c.icd10_code],
            },
        }


def _chunk_text(text: str, max_bytes: int = COMPREHEND_MAX_BYTES) -> List[str]:
    if not text:
        return []
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(encoded):
        end = min(start + max_bytes, len(encoded))
        # Decode safely on a UTF-8 boundary.
        slice_bytes = encoded[start:end]
        while end > start and slice_bytes:
            try:
                chunks.append(slice_bytes.decode("utf-8"))
                break
            except UnicodeDecodeError:
                end -= 1
                slice_bytes = encoded[start:end]
        start = end
    return chunks


def _attribute(entity: Dict[str, Any], typ: str) -> str:
    """Pull a typed attribute (e.g. DOSAGE, ROUTE_OR_MODE) off a Comprehend entity."""
    for attr in entity.get("Attributes") or []:
        if (attr.get("Type") or "").upper() == typ:
            return (attr.get("Text") or "").strip()
    return ""


def extract_from_transcript(transcript_text: str) -> ExtractionResult:
    """Run the full extraction pipeline on a single transcript."""
    text = (transcript_text or "").strip()
    if not text:
        return ExtractionResult()

    try:
        client = _client()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Comprehend Medical client unavailable: %s", exc)
        return ExtractionResult(raw={"error": str(exc)})

    result = ExtractionResult()
    raw_pages: List[Dict[str, Any]] = []

    for chunk in _chunk_text(text):
        try:
            entities = client.detect_entities_v2(Text=chunk).get("Entities") or []
        except Exception as exc:  # noqa: BLE001
            logger.exception("DetectEntitiesV2 failed: %s", exc)
            entities = []

        try:
            rx = client.infer_rx_norm(Text=chunk).get("Entities") or []
        except Exception as exc:  # noqa: BLE001
            logger.info("InferRxNorm failed: %s", exc)
            rx = []

        try:
            icd = client.infer_icd10_cm(Text=chunk).get("Entities") or []
        except Exception as exc:  # noqa: BLE001
            logger.info("InferICD10CM failed: %s", exc)
            icd = []

        raw_pages.append({"entities": entities, "rxnorm": rx, "icd10": icd})

        for ent in rx:
            concepts = ent.get("RxNormConcepts") or []
            top = concepts[0] if concepts else {}
            result.medications.append(
                MedicationCandidate(
                    name=(ent.get("Text") or "").strip(),
                    rxnorm_code=(top.get("Code") or "").strip(),
                    rxnorm_description=(top.get("Description") or "").strip(),
                    dose=_attribute(ent, "DOSAGE"),
                    frequency=_attribute(ent, "FREQUENCY"),
                    route=_attribute(ent, "ROUTE_OR_MODE"),
                    duration=_attribute(ent, "DURATION"),
                    confidence=float(ent.get("Score") or 0.0),
                    text_span=(ent.get("Text") or "").strip(),
                    raw=ent,
                )
            )

        for ent in icd:
            concepts = ent.get("ICD10CMConcepts") or []
            top = concepts[0] if concepts else {}
            result.conditions.append(
                ConditionCandidate(
                    name=(ent.get("Text") or "").strip(),
                    icd10_code=(top.get("Code") or "").strip(),
                    icd10_description=(top.get("Description") or "").strip(),
                    confidence=float(ent.get("Score") or 0.0),
                    text_span=(ent.get("Text") or "").strip(),
                )
            )

        for ent in entities:
            category = (ent.get("Category") or "").upper()
            etype = (ent.get("Type") or "").upper()
            text_span = (ent.get("Text") or "").strip()
            score = float(ent.get("Score") or 0.0)

            if category == "MEDICAL_CONDITION":
                # Already covered by ICD10 path with normalization; only add
                # un-coded ones here so we don't double-count.
                if not any(c.text_span == text_span for c in result.conditions):
                    result.conditions.append(
                        ConditionCandidate(name=text_span, confidence=score, text_span=text_span)
                    )

            if etype in {"SIGN", "SYMPTOM"} or category == "MEDICAL_CONDITION":
                # Many providers record symptoms as conditions; for the user
                # we surface them as symptoms when not coded.
                if etype in {"SIGN", "SYMPTOM"}:
                    result.symptoms.append(
                        SymptomCandidate(
                            name=text_span,
                            severity=_attribute(ent, "QUALITY"),
                            confidence=score,
                            text_span=text_span,
                        )
                    )

            if category == "PROTECTED_HEALTH_INFORMATION":
                # Don't pass PHI mentions back as "instructions"; they should
                # never appear in downstream summaries.
                continue

            if etype in {"INSTRUCTION", "TREATMENT"} or category == "TEST_TREATMENT_PROCEDURE":
                result.instructions.append(
                    InstructionCandidate(text=text_span, confidence=score)
                )

    result.raw = {"chunks": raw_pages}
    return result


# --------------------------------------------------------------------------- #
# Medication reconciliation                                                   #
# --------------------------------------------------------------------------- #

@dataclass
class ReconciliationDelta:
    introduced: List[Any] = field(default_factory=list)  # Medication
    modified: List[Any] = field(default_factory=list)
    discontinued: List[Any] = field(default_factory=list)
    needs_review: List[Any] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.introduced or self.modified or self.discontinued)

    def summary(self) -> str:
        parts = []
        if self.introduced:
            parts.append(f"{len(self.introduced)} new")
        if self.modified:
            parts.append(f"{len(self.modified)} modified")
        if self.discontinued:
            parts.append(f"{len(self.discontinued)} discontinued")
        return ", ".join(parts) or "no changes"


def reconcile_from_voice(
    *,
    user,
    candidates: List[MedicationCandidate],
    auto_threshold: Optional[float] = None,
) -> ReconciliationDelta:
    """Apply detected medication candidates to the user's stateful list.

    * **Match by RxNorm code first**, then by lowercase name as a fallback.
    * Auto-confirm at high confidence; otherwise stash as ``pending_review``.
    * Every state transition emits a ``MedicationEvent`` so the reconciliation
      audit trail is preserved (longitudinal versioning).
    """
    from medications.models import Medication, MedicationEvent

    threshold = float(auto_threshold if auto_threshold is not None else DEFAULT_MED_AUTO_THRESHOLD)
    delta = ReconciliationDelta()

    if not candidates:
        return delta

    existing = list(Medication.objects.filter(user=user))

    def _find_match(c: MedicationCandidate) -> Optional[Any]:
        if c.rxnorm_code:
            for m in existing:
                if m.rxnorm_code and m.rxnorm_code == c.rxnorm_code:
                    return m
        cl = c.name.lower().strip()
        for m in existing:
            if m.name.lower().strip() == cl:
                return m
        return None

    for cand in candidates:
        verification = "auto_confirmed" if cand.confidence >= threshold else "pending_review"
        match = _find_match(cand)

        if match is None:
            med = Medication.objects.create(
                user=user,
                name=cand.name,
                rxnorm_code=cand.rxnorm_code,
                dose=cand.dose,
                frequency=cand.frequency,
                route=cand.route,
                status="active",
                source="voice_extract",
                verification=verification,
                confidence=cand.confidence,
            )
            MedicationEvent.objects.create(
                medication=med,
                event_type="created",
                payload={
                    "source": "voice_extract",
                    "rxnorm": cand.rxnorm_code,
                    "confidence": cand.confidence,
                    "verification": verification,
                },
            )
            (delta.introduced if verification == "auto_confirmed" else delta.needs_review).append(med)
            existing.append(med)
            continue

        # Detect modifications.
        changes: Dict[str, Any] = {}
        for field_name in ("dose", "frequency", "route"):
            new_val = (getattr(cand, field_name) or "").strip()
            if new_val and new_val != (getattr(match, field_name) or "").strip():
                changes[field_name] = new_val
                setattr(match, field_name, new_val)

        if cand.rxnorm_code and not match.rxnorm_code:
            changes["rxnorm_code"] = cand.rxnorm_code
            match.rxnorm_code = cand.rxnorm_code

        if changes:
            match.status = "modified"
            match.confidence = max(match.confidence or 0.0, cand.confidence)
            if verification == "auto_confirmed":
                match.verification = "auto_confirmed"
            match.save()
            MedicationEvent.objects.create(
                medication=match,
                event_type="modified",
                payload={"source": "voice_extract", "changes": changes, "confidence": cand.confidence},
            )
            (delta.modified if verification == "auto_confirmed" else delta.needs_review).append(match)

    return delta
