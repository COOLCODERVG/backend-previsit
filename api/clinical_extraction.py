"""Voice-transcript → structured-clinical-data pipeline.

Uses the backend LLM service configured by LLM_INFERENCE_URL and LLM_MODEL_ID.
Structured entity extraction is performed by the local Ollama/LLM inference
service instead of external medical NLP APIs.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .llm_client import call_llama_transcript_extraction

logger = logging.getLogger(__name__)

DEFAULT_MED_AUTO_THRESHOLD = float(os.environ.get("CM_AUTO_CONFIRM_THRESHOLD", "0.85"))


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


def extract_from_transcript(transcript_text: str) -> ExtractionResult:
    """Run the full extraction pipeline on a single transcript."""
    text = (transcript_text or "").strip()
    if not text:
        return ExtractionResult()

    result = ExtractionResult()
    try:
        extracted = call_llama_transcript_extraction(
            transcript=text,
            steering_vectors=[],
            timeout_seconds=45,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Transcript extraction failed: %s", exc)
        return ExtractionResult(raw={"error": str(exc)})

    if not extracted or not isinstance(extracted, dict):
        return ExtractionResult(raw={"error": "LLM transcript extraction returned invalid data", "payload": extracted})

    result.raw = {"llm_output": extracted}

    for med in extracted.get("medications") or []:
        if not isinstance(med, dict):
            continue
        result.medications.append(
            MedicationCandidate(
                name=str(med.get("name") or "").strip(),
                dose=str(med.get("dose") or "").strip(),
                frequency=str(med.get("frequency") or "").strip(),
                route=str(med.get("route") or "").strip(),
                rxnorm_code=str(med.get("rxnorm_code") or "").strip(),
                rxnorm_description=str(med.get("rxnorm_description") or "").strip(),
                confidence=float(med.get("confidence") or 0.0),
                text_span=str(med.get("text_span") or med.get("name") or "").strip(),
                raw=med,
            )
        )

    for sym in extracted.get("symptoms") or []:
        if not isinstance(sym, dict):
            continue
        result.symptoms.append(
            SymptomCandidate(
                name=str(sym.get("name") or "").strip(),
                severity=str(sym.get("severity") or "").strip(),
                confidence=float(sym.get("confidence") or 0.0),
                text_span=str(sym.get("text_span") or sym.get("name") or "").strip(),
            )
        )

    for cond in extracted.get("conditions") or []:
        if not isinstance(cond, dict):
            continue
        result.conditions.append(
            ConditionCandidate(
                name=str(cond.get("name") or "").strip(),
                icd10_code=str(cond.get("icd10_code") or "").strip(),
                confidence=float(cond.get("confidence") or 0.0),
                text_span=str(cond.get("text_span") or cond.get("name") or "").strip(),
            )
        )

    for instr in extracted.get("instructions") or []:
        if not isinstance(instr, dict):
            continue
        result.instructions.append(
            InstructionCandidate(
                text=str(instr.get("text") or "").strip(),
                confidence=float(instr.get("confidence") or 0.0),
            )
        )

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
