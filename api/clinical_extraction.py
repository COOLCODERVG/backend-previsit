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

from .llm_client import call_llama_transcript_extraction, call_llama_visit_summary

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
# Post-visit AI summary (patient-facing follow-up brief)                      #
# --------------------------------------------------------------------------- #

def _coerce_str_list(value: Any) -> List[str]:
    """Best-effort normalization of an LLM field that should be a list of
    strings, but might come back as a single string, a list of dicts, or
    missing entirely. Never raises."""
    if not value:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    out.append(text)
            elif isinstance(item, dict):
                # Some models wrap list items as {"text": "..."} objects.
                text = str(item.get("text") or item.get("description") or item.get("name") or "").strip()
                if text:
                    out.append(text)
        return out
    return []


def _coerce_action_items(value: Any) -> List[Dict[str, Any]]:
    """Normalize action items into [{"text": str, "completed": bool}, ...],
    tolerant of the LLM returning plain strings or partially-shaped dicts."""
    items = []
    for text in _coerce_str_list(value):
        items.append({"text": text, "completed": False})
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict) and item.get("completed") is not None:
                text = str(item.get("text") or "").strip()
                for existing in items:
                    if existing["text"] == text:
                        existing["completed"] = bool(item.get("completed"))
    return items


def _coerce_medication_changes(value: Any) -> List[Dict[str, str]]:
    """Normalize medication-change entries into a flexible schema — any of
    name/dosage/frequency/duration/change_type may be missing."""
    if not isinstance(value, list):
        return []
    out: List[Dict[str, str]] = []
    for item in value:
        if isinstance(item, dict):
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            out.append({
                "name": name,
                "dosage": str(item.get("dosage") or item.get("dose") or "").strip(),
                "frequency": str(item.get("frequency") or "").strip(),
                "duration": str(item.get("duration") or "").strip(),
                "change_type": str(item.get("change_type") or "").strip(),
            })
        elif isinstance(item, str) and item.strip():
            out.append({"name": item.strip(), "dosage": "", "frequency": "", "duration": "", "change_type": ""})
    return out


def generate_visit_summary(transcript_text: str) -> Dict[str, Any]:
    """Generate a structured, patient-facing post-visit summary from a raw
    visit transcript using the same Ollama/SyniVia LLM pipeline used
    elsewhere in the app.

    The returned schema is intentionally flexible/tolerant: any section may
    be an empty list/string if the model didn't mention it, or if the LLM
    call failed outright (in which case ``_meta.error`` is populated instead
    of raising, so callers can persist a "failed" status without crashing
    the recording pipeline).
    """
    empty: Dict[str, Any] = {
        "summary": "",
        "action_items": [],
        "medication_changes": [],
        "tests_ordered": [],
        "follow_ups": [],
        "upcoming_appointments": [],
        "doctor_instructions": [],
        "lifestyle_recommendations": [],
        "warnings": [],
        "questions_for_next_visit": [],
    }

    text = (transcript_text or "").strip()
    if not text:
        return {**empty, "_meta": {"error": "empty_transcript"}}

    try:
        raw = call_llama_visit_summary(transcript=text)
    except Exception as exc:  # noqa: BLE001
        logger.exception("call_llama_visit_summary failed: %s", exc)
        return {**empty, "_meta": {"error": str(exc)}}

    if not isinstance(raw, dict) or not str(raw.get("summary") or "").strip():
        return {**empty, "_meta": {"error": "llm_returned_no_usable_summary"}}

    return {
        "summary": str(raw.get("summary") or "").strip(),
        "action_items": _coerce_action_items(raw.get("action_items")),
        "medication_changes": _coerce_medication_changes(raw.get("medication_changes")),
        "tests_ordered": _coerce_str_list(raw.get("tests_ordered")),
        "follow_ups": _coerce_str_list(raw.get("follow_ups")),
        "upcoming_appointments": _coerce_str_list(raw.get("upcoming_appointments")),
        "doctor_instructions": _coerce_str_list(raw.get("doctor_instructions")),
        "lifestyle_recommendations": _coerce_str_list(raw.get("lifestyle_recommendations")),
        "warnings": _coerce_str_list(raw.get("warnings")),
        "questions_for_next_visit": _coerce_str_list(raw.get("questions_for_next_visit")),
        "_meta": {"error": None},
    }


# --------------------------------------------------------------------------- #
# Structured-field merge (symptoms / notes)                                   #
# --------------------------------------------------------------------------- #

def merge_extraction_into_appointment(*, user, appointment, result: "ExtractionResult") -> Dict[str, int]:
    """Merge voice-extracted symptoms/conditions/instructions into the SAME
    appointment sections (Symptom, Note) that text-entered data uses.

    This is strictly additive/append-only: existing user-entered rows are
    never modified or deleted. Duplicate detection is a simple case-insensitive
    match on `name` (symptoms) / `content` (notes) scoped to this appointment,
    so re-running extraction (or combining voice + text) doesn't create
    duplicate rows.

    Medication candidates are handled separately by `reconcile_from_voice`
    (medications are a user-level list, not appointment-scoped).
    """
    from .models import Symptom, Note

    created = {"symptoms": 0, "notes": 0}

    existing_symptom_names = {
        s.name.strip().lower()
        for s in Symptom.objects.filter(appointment=appointment, user=user)
        if s.name
    }
    for sym in result.symptoms:
        name = (sym.name or "").strip()
        if not name or name.lower() in existing_symptom_names:
            continue
        severity_int = 5
        try:
            if sym.severity:
                severity_int = max(1, min(10, int(round(float(sym.severity)))))
        except Exception:
            severity_int = 5
        note_text = (
            f"Detected from voice recording (confidence {sym.confidence:.2f})"
            if sym.confidence
            else "Detected from voice recording"
        )
        Symptom.objects.create(
            user=user,
            appointment=appointment,
            name=name,
            severity=severity_int,
            is_new=True,
            is_worsening=False,
            notes=note_text,
        )
        existing_symptom_names.add(name.lower())
        created["symptoms"] += 1

    existing_note_contents = {
        (n.content or "").strip().lower()
        for n in Note.objects.filter(appointment=appointment, user=user)
    }

    def _add_note(title: str, content: str, category: str) -> None:
        cleaned = (content or "").strip()
        if not cleaned or cleaned.lower() in existing_note_contents:
            return
        Note.objects.create(user=user, appointment=appointment, title=title, content=cleaned, category=category)
        existing_note_contents.add(cleaned.lower())
        created["notes"] += 1

    for instr in result.instructions:
        _add_note("Voice note", instr.text, "voice_extracted")

    for cond in result.conditions:
        label = (cond.name or "").strip()
        if not label:
            continue
        suffix = f" (ICD-10: {cond.icd10_code})" if cond.icd10_code else ""
        _add_note("Condition mentioned", label + suffix, "voice_extracted")

    return created

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
