from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

_VAGUE_MARKERS = {
    'nope', 'no', 'none', 'n/a', 'na', 'nothing', 'no questions', 'not sure',
    'idk', "dont know", "don't know", 'no q', 'no qs', '-', '.',
}


def _is_vague(text: Any) -> bool:
    value = str(text or "").strip().lower().strip('.! ')
    if len(value) < 3:
        return True
    return value in _VAGUE_MARKERS


def _severity_word(level: int) -> str:
    if level >= 9:
        return "severe"
    if level >= 7:
        return "significant"
    if level >= 4:
        return "moderate"
    if level >= 1:
        return "mild"
    return "unspecified"


def _symptom_meta_line(s: Dict[str, Any], esc) -> str:
    """Short muted meta line for a symptom card: duration/timing only (no
    prose sentence) e.g. "Lasting 3 days · worse at night"."""
    duration = str(s.get("duration") or "").strip()
    timing = str(s.get("timing") or "").strip()
    bits = []
    if duration:
        bits.append(f"Lasting {esc(duration)}")
    if timing:
        bits.append(esc(timing))
    return " \u00b7 ".join(bits)


_NOTE_CATEGORY_LABELS = {
    "medication": "Current Medications",
    "lifestyle": "Lifestyle",
    "emotional": "Emotional Wellbeing",
    "recent_changes": "Recent Changes",
    "goal": "Visit Goal",
    "general": "Other Notes",
}

# Categories shown in the "Things I Want My Provider to Know" section — the
# visit goal is intentionally excluded here since it's already surfaced in
# the Visit Overview section (avoid duplication across sections).
_THINGS_TO_KNOW_ORDER = ("medication", "lifestyle", "emotional", "recent_changes", "general")


def _classify_note(n: Dict[str, Any]) -> str:
    category = str(n.get("category") or "").strip().lower()
    if category in _NOTE_CATEGORY_LABELS and category != "general":
        return category
    text = f"{n.get('title') or ''} {n.get('content') or ''}".lower()
    if any(k in text for k in ("medic", "pill", "dose", "tablet", "prescri", "take medicine")):
        return "medication"
    if any(k in text for k in ("feel", "feeling", "anxious", "worried", "scared", "stressed", "mood", "emotion", "overwhelm")):
        return "emotional"
    if any(k in text for k in ("routine", "check-up", "checkup", "follow-up", "purpose", "goal", "reason")):
        return "goal"
    if any(k in text for k in ("diet", "exercise", "sleep", "alcohol", "smoking", "smoke", "caffeine", "activity", "routine change")):
        return "lifestyle"
    if any(k in text for k in ("recently", "new since", "changed", "started", "stopped", "began", "since last visit", "last week", "last month")):
        return "recent_changes"
    return "general"


def _note_display_text(n: Dict[str, Any], category: str, esc) -> str:
    content = str(n.get("content") or "").strip()
    if category == "emotional":
        return f"Patient reported feeling {esc(content.lower().rstrip('.'))}."
    return esc(content)


def render_summary_html(summary_payload: Dict[str, Any], ai_guidance: Optional[Dict[str, Any]] = None) -> str:
    appointment = summary_payload.get("appointment") or {}
    personalization = summary_payload.get("personalization_profile") or {}
    symptoms = summary_payload.get("symptoms") or []
    questions = summary_payload.get("questions") or []
    notes = summary_payload.get("notes") or []

    def esc(s: Any) -> str:
        return (
            str(s or "")
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    def severity_meter(raw_severity: Any) -> str:
        """Renders severity as a 10-segment filled/hollow dot row plus the raw number.
        Kept as text glyphs (not divs) so it degrades gracefully in any WeasyPrint version."""
        try:
            level = max(0, min(10, int(raw_severity)))
        except (TypeError, ValueError):
            level = 0
        filled = "\u25CF" * level
        empty = "\u25CB" * (10 - level)
        return (
            f'<span class="meter">'
            f'<span class="meter-filled">{filled}</span>'
            f'<span class="meter-empty">{empty}</span>'
            f'</span>'
            f'<span class="meter-value">{level}/10</span>'
        )

    def empty_state(label: str) -> str:
        return f'<p class="empty">{esc(label)}</p>'

    # ---- Visit Overview (replaces the old "General"/focus-only block) ------
    # Appointment Reason + Primary Visit Goal, both sourced directly from
    # patient-entered data (appointment notes / personalization profile) —
    # never invented. The LLM's `primary_goal` (a rephrasing, not new facts)
    # is preferred when available; otherwise we fall back to the raw
    # patient-entered reason so the section is never empty without cause.
    visit_reason = appointment.get("notes") or personalization.get("main_reason") or ""
    primary_goal = (ai_guidance or {}).get("primary_goal") or personalization.get("main_reason") or visit_reason

    appointment_date = esc(appointment.get("appointment_date") or "")
    appointment_time = esc(appointment.get("appointment_time") or "")
    specialty = esc(appointment.get("specialty") or "")
    doctor_name = esc(appointment.get("doctor_name") or "Healthcare Provider")

    visit_reason_html = esc(visit_reason) if visit_reason else '<span class="empty-inline">Not specified</span>'
    primary_goal_html = esc(primary_goal) if primary_goal else '<span class="empty-inline">Not specified</span>'

    # ---- AI Executive Summary (NEW, first major content section) -----------
    # 3-5 bullets, strictly from patient-provided information (LLM may only
    # summarize/reorganize/rephrase — see content rules). Falls back to a
    # short deterministic summary built from the same source facts if the
    # LLM call failed, so the section is never silently empty.
    executive_summary: List[str] = list((ai_guidance or {}).get("executive_summary") or [])
    if not executive_summary:
        symptoms_by_severity = sorted(symptoms, key=lambda s: -(int(s.get("severity") or 0)))
        if symptoms_by_severity:
            top = symptoms_by_severity[0]
            sev = top.get("severity")
            bit = f"Primary concern is {'a new ' if top.get('is_new') else ''}{esc(top.get('name', 'symptom')).lower()}"
            if sev:
                bit += f" rated {sev}/10"
            executive_summary.append(bit + ".")
        if personalization.get("biggest_concern"):
            executive_summary.append(f"Patient's biggest concern: {esc(personalization.get('biggest_concern'))}.")
        if any(_classify_note(n) == "medication" for n in notes):
            executive_summary.append("Medication use was mentioned.")
        if any(_classify_note(n) == "emotional" for n in notes):
            executive_summary.append("Patient shared how they're feeling about this visit.")
        if not executive_summary:
            executive_summary.append("No additional visit details were provided.")
    executive_summary_html = "".join(f"<li>{esc(v)}</li>" for v in executive_summary[:5])

    # ---- Symptoms (structured cards, sorted by severity, no long prose) ----
    symptoms_sorted = sorted(symptoms, key=lambda s: -(int(s.get("severity") or 0)))
    if symptoms_sorted:
        rows = []
        for s in symptoms_sorted:
            tags = []
            if s.get("is_new"):
                tags.append("NEW")
            elif s.get("is_worsening"):
                tags.append("WORSENING")
            else:
                tags.append("ONGOING")
            tag_html = f'<span class="tag">{" · ".join(tags)}</span>'

            meta_line = _symptom_meta_line(s, esc)
            meta_html = f'<div class="entry-meta">{meta_line}</div>' if meta_line else ""

            note_text = str(s.get("notes") or "").strip()
            note_html = f'<p class="entry-note">{esc(note_text)}</p>' if note_text else ""

            rows.append(
                f"""
                <div class="entry">
                  <div class="entry-head">
                    <span class="entry-title">{esc(s.get('name', 'Symptom'))}</span>
                    {tag_html}
                  </div>
                  <div class="entry-severity">{severity_meter(s.get('severity', 0))}</div>
                  {meta_html}
                  {note_html}
                </div>
                """
            )
        symptoms_html = "".join(rows)
    else:
        symptoms_html = empty_state("No symptoms were added.")

    # ---- Questions for Provider ---------------------------------------------
    # Vague placeholder answers ("Nope", "N/A") are filtered out; if nothing
    # substantive remains we show a clear empty-state instead of an
    # empty/useless section.
    all_pending = [q.get("text", "") for q in questions if not q.get("is_answered")]
    pending_questions = [q for q in all_pending if not _is_vague(q)]
    if pending_questions:
        items = "".join(
            f'<li><span class="q-index">{i:02d}</span><span class="q-text">{esc(q)}</span></li>'
            for i, q in enumerate(pending_questions, start=1)
        )
        questions_html = f'<ol class="question-list">{items}</ol>'
    else:
        questions_html = empty_state("No questions added.")

    # Suggested discussion topics: LLM-suggested (never assumes a diagnosis),
    # clearly labelled and visually distinct from the patient's own questions.
    suggested_questions: List[str] = list((ai_guidance or {}).get("suggested_questions") or [])
    suggested_questions_html = ""
    if suggested_questions:
        items = "".join(f"<li>{esc(v)}</li>" for v in suggested_questions[:5])
        suggested_questions_html = f"""
        <div class="suggested-topics">
          <div class="suggested-topics-label">Suggested discussion topics</div>
          <ul>{items}</ul>
        </div>
        """

    # ---- Things I Want My Provider to Know (replaces "Patient Notes") ------
    # Grouped bullets by category; only categories with actual content are
    # shown (never invented), and wording avoids repeating what's already
    # covered verbatim in Symptoms / Visit Overview above.
    if notes:
        grouped: Dict[str, List[str]] = {}
        for n in notes:
            category = _classify_note(n)
            if category == "goal":
                continue  # already covered by Visit Overview — avoid duplication
            grouped.setdefault(category, []).append(_note_display_text(n, category, esc))

        rows = []
        for category in _THINGS_TO_KNOW_ORDER:
            entries = grouped.get(category)
            if not entries:
                continue
            label = _NOTE_CATEGORY_LABELS[category]
            items_html = "".join(f"<li>{text}</li>" for text in entries)
            rows.append(
                f"""
                <div class="know-group">
                  <div class="know-group-label">{esc(label)}</div>
                  <ul>{items_html}</ul>
                </div>
                """
            )
        things_to_know_html = "".join(rows) or empty_state("No patient notes provided.")
    else:
        things_to_know_html = empty_state("No patient notes provided.")

    # ---- Visit Preparation (renamed from "AI Visit Preparation Summary") ---
    # Organizes the patient's OWN information only — never new medical facts.
    visit_prep_topics: List[str] = list((ai_guidance or {}).get("visit_prep_topics") or [])
    if not visit_prep_topics:
        visit_prep_topics = pending_questions[:5] or [
            f"Discuss {s.get('name')}" for s in symptoms_sorted[:3]
        ]
    prep_mention_items = []
    if personalization.get("biggest_concern"):
        prep_mention_items.append(personalization.get("biggest_concern"))
    for n in notes:
        if _classify_note(n) == "medication":
            prep_mention_items.append(n.get("content") or "")
    prep_reminders = [str(v) for v in (personalization.get("prepared_items") or [])]

    prep_topics_html = (
        "".join(f"<li>{esc(v)}</li>" for v in visit_prep_topics if v)
        or '<li class="empty-inline">No specific topics suggested</li>'
    )
    prep_mention_html = (
        "".join(f"<li>{esc(v)}</li>" for v in prep_mention_items if v)
        or '<li class="empty-inline">No additional context provided</li>'
    )
    prep_reminders_html = (
        "".join(f"<li>{esc(v)}</li>" for v in prep_reminders if v)
        or '<li class="empty-inline">No reminders added</li>'
    )

    # ---- Provider Snapshot (renamed from "Provider Summary") ---------------
    # A concise, one-minute handoff — only patient-provided facts, no
    # inference, assessment, or recommendations.
    provider_symptom_rows = []
    for s in symptoms_sorted:
        bits = [esc(s.get("name", "symptom"))]
        if s.get("severity"):
            bits.append(f"severity {s.get('severity')}/10")
        if s.get("duration"):
            bits.append(f"duration {esc(s.get('duration'))}")
        provider_symptom_rows.append(f"<li>{', '.join(bits)}</li>")
    provider_symptoms_html = "".join(provider_symptom_rows) or '<li class="empty-inline">None reported</li>'

    goal_text = {
        'clear_diagnosis': 'Wants a clear diagnosis',
        'next_steps_plan': 'Wants next steps or a treatment plan',
        'tests_or_referrals': 'Wants tests or referrals',
        'heard_understood': 'Wants to feel heard and understood',
    }.get(personalization.get("appointment_outcome") or "", "") or primary_goal or "Not specified"

    provider_questions_html = (
        "".join(f"<li>{esc(q)}</li>" for q in all_pending)
        or '<li class="empty-inline">No questions provided</li>'
    )

    provider_med_rows = [
        f"<li>{esc(n.get('content'))}</li>" for n in notes if _classify_note(n) == "medication" and n.get("content")
    ]
    provider_meds_html = "".join(provider_med_rows) or '<li class="empty-inline">None mentioned</li>'

    provider_emotional_rows = [
        f"<li>{esc(n.get('content'))}</li>" for n in notes if _classify_note(n) == "emotional" and n.get("content")
    ]
    provider_emotional_html = "".join(provider_emotional_rows) or '<li class="empty-inline">None mentioned</li>'

    provider_notes_rows = [
        f"<li>{esc(n.get('content'))}</li>"
        for n in notes
        if _classify_note(n) not in ("medication", "emotional", "goal") and n.get("content")
    ]
    provider_notes_html = "".join(provider_notes_rows) or '<li class="empty-inline">None provided</li>'

    generated_at = esc(datetime.utcnow().strftime("%B %d, %Y \u00b7 %H:%M UTC"))

    return f"""
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <style>
      @page {{
        size: A4;
        margin: 22mm 20mm 20mm 20mm;
        @bottom-center {{
          content: "";
        }}
      }}
      :root {{
        --ink: #0E0E0E;
        --charcoal: #363636;
        --graphite: #6E6E6C;
        --hairline: #DCDCDA;
        --mist: #F4F4F2;
        --paper: #FFFFFF;
      }}
      * {{ margin: 0; padding: 0; box-sizing: border-box; }}
      html, body {{ background: var(--paper); }}
      body {{
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-size: 9.5px;
        color: var(--charcoal);
        line-height: 1.65;
      }}

      /* ---------- Masthead ---------- */
      .masthead {{
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        padding-bottom: 14px;
        border-bottom: 2.5px solid var(--ink);
      }}
      .brand {{
        font-size: 9px;
        font-weight: 700;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: var(--ink);
        margin-bottom: 10px;
      }}
      .doc-title {{
        font-family: Georgia, 'Times New Roman', serif;
        font-size: 30px;
        font-weight: 400;
        color: var(--ink);
        letter-spacing: -0.3px;
      }}
      .masthead-right {{
        text-align: right;
      }}
      .provider-name {{
        font-family: Georgia, 'Times New Roman', serif;
        font-size: 14px;
        color: var(--ink);
        margin-bottom: 2px;
      }}
      .provider-specialty {{
        font-size: 9.5px;
        color: var(--graphite);
      }}

      /* ---------- Metadata strip ---------- */
      .meta-strip {{
        display: flex;
        margin: 16px 0 26px;
      }}
      .meta-item {{
        flex: 1;
        padding: 0 16px;
        border-left: 1px solid var(--hairline);
      }}
      .meta-item:first-child {{ padding-left: 0; border-left: none; }}
      .meta-label {{
        font-size: 7.5px;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        color: var(--graphite);
        margin-bottom: 3px;
      }}
      .meta-value {{
        font-size: 12px;
        font-weight: 600;
        color: var(--ink);
      }}

      /* ---------- Visit Overview (Appointment Reason + Primary Goal) ---------- */
      .overview {{
        display: flex;
        gap: 28px;
        padding: 16px 0 18px 16px;
        border-left: 3px solid var(--ink);
        margin-bottom: 24px;
      }}
      .overview-col {{ flex: 1; }}
      .focus-label {{
        font-size: 8px;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        color: var(--graphite);
        margin-bottom: 6px;
      }}
      .focus-body {{
        font-size: 12px;
        color: var(--ink);
        line-height: 1.7;
      }}
      .empty-inline {{ color: var(--graphite); font-style: italic; }}

      /* ---------- AI Executive Summary ---------- */
      .exec-summary {{
        background: var(--mist);
        border-radius: 6px;
        padding: 16px 18px;
        margin-bottom: 4px;
      }}
      .exec-summary ul {{ list-style: none; }}
      .exec-summary li {{
        font-size: 11px;
        color: var(--ink);
        line-height: 1.65;
        padding: 5px 0 5px 16px;
        position: relative;
      }}
      .exec-summary li::before {{
        content: "\2022";
        position: absolute;
        left: 0;
        color: var(--ink);
        font-weight: 700;
      }}

      /* ---------- Section scaffolding ---------- */
      section {{ margin-bottom: 26px; }}
      .section-head {{
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        border-bottom: 1px solid var(--ink);
        padding-bottom: 6px;
        margin-bottom: 14px;
        break-after: avoid;
        page-break-after: avoid;
      }}
      .section-title {{
        font-family: Georgia, 'Times New Roman', serif;
        font-size: 15px;
        font-weight: 400;
        color: var(--ink);
      }}
      .section-count {{
        font-size: 8.5px;
        color: var(--graphite);
        letter-spacing: 0.3px;
      }}
      .empty {{
        font-size: 10px;
        color: var(--graphite);
        font-style: italic;
        padding: 4px 0 2px;
      }}

      /* ---------- Repeating entries (symptoms) ---------- */
      .entry {{
        padding: 12px 0;
        border-bottom: 1px solid var(--hairline);
        break-inside: avoid;
        page-break-inside: avoid;
      }}
      .entry:last-child {{ border-bottom: none; }}
      .entry-head {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 6px;
      }}
      .entry-title {{
        font-size: 11.5px;
        font-weight: 700;
        color: var(--ink);
      }}
      .tag {{
        font-size: 7.5px;
        font-weight: 700;
        letter-spacing: 0.8px;
        color: var(--ink);
        border: 1px solid var(--ink);
        border-radius: 3px;
        padding: 2px 6px;
      }}
      .entry-severity {{ margin: 4px 0 6px; }}
      .meter {{ letter-spacing: 1.5px; font-size: 9px; }}
      .meter-filled {{ color: var(--ink); }}
      .meter-empty {{ color: var(--hairline); }}
      .meter-value {{
        font-size: 8.5px;
        color: var(--graphite);
        margin-left: 8px;
        font-weight: 600;
      }}
      .entry-meta {{
        font-size: 9px;
        color: var(--graphite);
        margin-bottom: 4px;
      }}
      .entry-note {{
        font-size: 10px;
        color: var(--charcoal);
        line-height: 1.6;
      }}

      /* ---------- Question list ---------- */
      .question-list {{ list-style: none; }}
      .question-list li {{
        display: flex;
        gap: 12px;
        padding: 9px 0;
        border-bottom: 1px solid var(--hairline);
        break-inside: avoid;
        page-break-inside: avoid;
      }}
      .question-list li:last-child {{ border-bottom: none; }}
      .q-index {{
        font-size: 9px;
        font-weight: 700;
        color: var(--graphite);
        min-width: 16px;
      }}
      .q-text {{
        font-size: 10.5px;
        color: var(--ink);
        line-height: 1.6;
      }}
      .suggested-topics {{
        margin-top: 14px;
        padding: 12px 14px;
        background: var(--mist);
        border-radius: 6px;
      }}
      .suggested-topics-label {{
        font-size: 8px;
        font-weight: 700;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        color: var(--graphite);
        margin-bottom: 6px;
      }}
      .suggested-topics ul {{ padding-left: 15px; }}
      .suggested-topics li {{ font-size: 10px; color: var(--charcoal); margin-bottom: 3px; }}

      /* ---------- Things I Want My Provider to Know ---------- */
      .know-group {{
        padding: 10px 0;
        border-bottom: 1px solid var(--hairline);
      }}
      .know-group:last-child {{ border-bottom: none; }}
      .know-group-label {{
        font-size: 9.5px;
        font-weight: 700;
        color: var(--ink);
        margin-bottom: 5px;
      }}
      .know-group ul {{ padding-left: 15px; }}
      .know-group li {{ font-size: 10px; color: var(--charcoal); margin-bottom: 3px; line-height: 1.6; }}

      /* ---------- AI Visit Preparation Summary ---------- */
      .ai-prep {{
        background: var(--mist);
        border-radius: 6px;
        padding: 16px 18px;
      }}
      .ai-prep-subhead {{
        font-size: 9px;
        font-weight: 700;
        letter-spacing: 0.6px;
        text-transform: uppercase;
        color: var(--graphite);
        margin: 10px 0 4px;
      }}
      .ai-prep-subhead:first-child {{ margin-top: 0; }}
      .ai-prep ul {{ padding-left: 15px; margin-bottom: 6px; }}
      .ai-prep li {{ font-size: 10px; color: var(--charcoal); margin-bottom: 3px; }}
      .ai-prep-note {{
        font-size: 8px;
        color: var(--graphite);
        font-style: italic;
        margin-top: 10px;
        padding-top: 8px;
        border-top: 1px solid var(--hairline);
      }}

      /* ---------- Provider Summary ---------- */
      .provider-summary {{
        border: 1.5px solid var(--ink);
        border-radius: 6px;
        padding: 16px 18px;
      }}
      .provider-summary-label {{
        font-size: 8px;
        font-weight: 700;
        letter-spacing: 0.6px;
        text-transform: uppercase;
        color: var(--ink);
        background: var(--mist);
        display: inline-block;
        padding: 3px 8px;
        border-radius: 3px;
        margin-bottom: 12px;
      }}
      .provider-subhead {{
        font-size: 10.5px;
        font-weight: 700;
        color: var(--ink);
        margin: 12px 0 4px;
      }}
      .provider-subhead:first-of-type {{ margin-top: 0; }}
      .provider-summary ul {{ padding-left: 15px; margin-bottom: 6px; }}
      .provider-summary li {{ font-size: 10px; color: var(--charcoal); margin-bottom: 3px; }}

      /* ---------- Disclaimer ---------- */
      .disclaimer {{
        margin-top: 32px;
        padding: 14px 0 0;
        border-top: 1px solid var(--ink);
      }}
      .disclaimer-title {{
        font-size: 8.5px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: var(--ink);
        margin-bottom: 6px;
      }}
      .disclaimer-text {{
        font-size: 8.5px;
        color: var(--graphite);
        line-height: 1.7;
      }}
      .disclaimer-text strong {{ color: var(--charcoal); }}

      /* ---------- Footer ---------- */
      .footer {{
        margin-top: 26px;
        padding-top: 12px;
        border-top: 1px solid var(--hairline);
        display: flex;
        justify-content: space-between;
        font-size: 7.5px;
        color: var(--graphite);
        letter-spacing: 0.3px;
      }}
      .footer strong {{ color: var(--ink); }}
    </style>
  </head>
  <body>

    <div class="masthead">
      <div>
        <div class="brand">SyniVia</div>
        <div class="doc-title">Visit Summary</div>
      </div>
      <div class="masthead-right">
        <div class="provider-name">{doctor_name}</div>
        {f'<div class="provider-specialty">{specialty}</div>' if specialty else ''}
      </div>
    </div>

    <div class="meta-strip">
      <div class="meta-item">
        <div class="meta-label">Date</div>
        <div class="meta-value">{appointment_date if appointment_date else "—"}</div>
      </div>
      <div class="meta-item">
        <div class="meta-label">Time</div>
        <div class="meta-value">{appointment_time if appointment_time else "—"}</div>
      </div>
      <div class="meta-item">
        <div class="meta-label">Generated</div>
        <div class="meta-value">{generated_at}</div>
      </div>
    </div>

    <div class="overview">
      <div class="overview-col">
        <div class="focus-label">Appointment Reason</div>
        <div class="focus-body">{visit_reason_html}</div>
      </div>
      <div class="overview-col">
        <div class="focus-label">Goal</div>
        <div class="focus-body">{primary_goal_html}</div>
      </div>
    </div>

    <section>
      <div class="section-head">
        <span class="section-title">AI Executive Summary</span>
      </div>
      <div class="exec-summary">
        <ul>{executive_summary_html}</ul>
      </div>
    </section>

    <section>
      <div class="section-head">
        <span class="section-title">Symptoms &amp; Complaints</span>
        <span class="section-count">{len(symptoms)} recorded</span>
      </div>
      {symptoms_html}
    </section>

    <section>
      <div class="section-head">
        <span class="section-title">Questions for Provider</span>
        <span class="section-count">{len(pending_questions)} pending</span>
      </div>
      {questions_html}
      {suggested_questions_html}
    </section>

    <section>
      <div class="section-head">
        <span class="section-title">Things I Want My Provider to Know</span>
      </div>
      {things_to_know_html}
    </section>

    <section>
      <div class="section-head">
        <span class="section-title">Visit Preparation</span>
      </div>
      <div class="ai-prep">
        <div class="ai-prep-subhead">Important topics to mention</div>
        <ul>{prep_mention_html}</ul>
        <div class="ai-prep-subhead">Suggested discussion order</div>
        <ul>{prep_topics_html}</ul>
        <div class="ai-prep-subhead">Helpful reminders</div>
        <ul>{prep_reminders_html}</ul>
        <div class="ai-prep-note">
          This section only helps organize your conversation with your provider. It is not medical advice and does not diagnose any condition.
        </div>
      </div>
    </section>

    <section>
      <div class="section-head">
        <span class="section-title">Provider Snapshot</span>
      </div>
      <div class="provider-summary">
        <div class="provider-summary-label">Based only on patient-provided information</div>
        <div class="provider-subhead">Chief concerns</div>
        <ul>{provider_symptoms_html}</ul>
        <div class="provider-subhead">Patient goals</div>
        <p class="entry-note">{esc(goal_text)}</p>
        <div class="provider-subhead">Questions</div>
        <ul>{provider_questions_html}</ul>
        <div class="provider-subhead">Medication mentions</div>
        <ul>{provider_meds_html}</ul>
        <div class="provider-subhead">Emotional concerns</div>
        <ul>{provider_emotional_html}</ul>
        <div class="provider-subhead">Relevant notes</div>
        <ul>{provider_notes_html}</ul>
      </div>
    </section>

    <div class="disclaimer">
      <div class="disclaimer-title">Important Disclaimer</div>
      <div class="disclaimer-text">
        This document is generated by <strong>SyniVia</strong> for organizational and record-keeping purposes only.
        <strong>SyniVia is not a medical diagnostic tool and does not provide medical advice.</strong>
        This summary must not be used as a substitute for professional clinical judgment or diagnosis.
        Always consult a qualified healthcare provider regarding any medical concerns.
        <strong>SyniVia is not liable for any errors, omissions, or consequences arising from the use of this document.</strong>
      </div>
    </div>

    <div class="footer">
      <span><strong>SyniVia</strong> &nbsp;Medical Visit Preparation &amp; Summary</span>
      <span>Confidential &nbsp;·&nbsp; Intended for patient use only</span>
    </div>

  </body>
</html>
"""


def html_to_pdf_bytes(html: str) -> bytes:
    try:
        from weasyprint import HTML  # type: ignore
    except Exception as exc:
        raise RuntimeError("WeasyPrint is not available. Install system deps and `WeasyPrint` Python package.") from exc
    return HTML(string=html).write_pdf()