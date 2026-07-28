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


def _symptom_narrative(s: Dict[str, Any], esc) -> str:
    """Turns a raw symptom row into a short prose sentence instead of a
    'Name Severity/10' dump, e.g.:
    "The patient reported a new headache with moderate severity (5/10).
    Additional context should be discussed with the provider."
    """
    name = str(s.get("name") or "symptom").strip().lower()
    try:
        severity = int(s.get("severity") or 0)
    except (TypeError, ValueError):
        severity = 0
    is_new = bool(s.get("is_new"))
    is_worsening = bool(s.get("is_worsening"))
    duration = str(s.get("duration") or "").strip()
    timing = str(s.get("timing") or "").strip()
    notes = str(s.get("notes") or "").strip()

    onset = "newly reported" if is_new else "an ongoing"
    sentence = f"The patient reported {onset} {name}"
    if severity:
        sentence += f" with {_severity_word(severity)} severity ({severity}/10)"
    if duration:
        sentence += f", lasting {duration}"
    if timing:
        sentence += f" ({timing})"
    if is_worsening:
        sentence += ", and it has been getting worse"
    sentence += "."
    if notes:
        sentence += f' Patient notes: "{esc(notes)}."'
    else:
        sentence += " Additional context should be discussed with the provider."
    return sentence


_NOTE_CATEGORY_LABELS = {
    "medication": "Medication",
    "emotional": "Emotional State",
    "goal": "Visit Goal",
    "general": "General",
}


def _classify_note(n: Dict[str, Any]) -> str:
    category = str(n.get("category") or "").strip().lower()
    if category in _NOTE_CATEGORY_LABELS and category != "general":
        return category
    text = f"{n.get('title') or ''} {n.get('content') or ''}".lower()
    if any(k in text for k in ("medic", "pill", "dose", "tablet", "prescri", "take medicine")):
        return "medication"
    if any(k in text for k in ("feel", "feeling", "anxious", "worried", "scared", "stressed", "mood", "emotion")):
        return "emotional"
    if any(k in text for k in ("routine", "check-up", "checkup", "follow-up", "purpose", "goal", "reason")):
        return "goal"
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

    focus_header = (ai_guidance or {}).get("focus_header") or "Visit Focus"
    focus_summary = (ai_guidance or {}).get("focus_summary") or (personalization.get("main_reason") or "")
    discussion_points: List[str] = (ai_guidance or {}).get("discussion_points") or []

    appointment_date = esc(appointment.get("appointment_date") or "")
    appointment_time = esc(appointment.get("appointment_time") or "")
    specialty = esc(appointment.get("specialty") or "")
    doctor_name = esc(appointment.get("doctor_name") or "Healthcare Provider")

    # ---- Focus card -----------------------------------------------------
    focus_body = esc(focus_summary) if focus_summary else '<span class="empty-inline">No focus area specified</span>'
    discussion_html = ""
    if discussion_points:
        items = "".join(f"<li>{esc(p)}</li>" for p in discussion_points)
        discussion_html = f'<ul class="discuss-list">{items}</ul>'

    # ---- Symptoms ---------------------------------------------------------
    # Renders each symptom as a short prose narrative (see _symptom_narrative)
    # rather than a raw "Name Severity/10" dump.
    if symptoms:
        rows = []
        for s in symptoms:
            tags = []
            if s.get("is_new"):
                tags.append("NEW")
            if s.get("is_worsening"):
                tags.append("WORSENING")
            tag_html = f'<span class="tag">{" · ".join(tags)}</span>' if tags else ""

            narrative_html = f'<p class="entry-note">{_symptom_narrative(s, esc)}</p>'

            rows.append(
                f"""
                <div class="entry">
                  <div class="entry-head">
                    <span class="entry-title">{esc(s.get('name', 'Symptom'))}</span>
                    {tag_html}
                  </div>
                  <div class="entry-severity">{severity_meter(s.get('severity', 0))}</div>
                  {narrative_html}
                </div>
                """
            )
        symptoms_html = "".join(rows)
    else:
        symptoms_html = empty_state("No symptoms recorded")

    # ---- Questions ----------------------------------------------------------
    # Vague placeholder answers ("Nope", "N/A") are filtered out; if nothing
    # substantive remains we show an actionable suggestion instead of an
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
        questions_html = (
            '<p class="empty">No specific questions were added. Consider asking about symptoms, '
            'possible causes, treatment options, and next steps.</p>'
        )

    # ---- Notes ----------------------------------------------------------------
    # Notes are grouped into meaningful categories (medication / emotional /
    # visit goal / general) instead of listed as a raw title+content dump.
    if notes:
        grouped: Dict[str, List[str]] = {}
        for n in notes:
            category = _classify_note(n)
            grouped.setdefault(category, []).append(_note_display_text(n, category, esc))

        rows = []
        for category in ("medication", "emotional", "goal", "general"):
            entries = grouped.get(category)
            if not entries:
                continue
            label = _NOTE_CATEGORY_LABELS[category]
            items_html = "".join(f'<p class="entry-note">{text}</p>' for text in entries)
            rows.append(
                f"""
                <div class="entry">
                  <span class="entry-title">{esc(label)}</span>
                  {items_html}
                </div>
                """
            )
        notes_html = "".join(rows)
    else:
        notes_html = empty_state("No notes available")

    # ---- AI Visit Preparation Summary ---------------------------------------
    # Organizational-only guidance: what the patient wants to discuss, useful
    # info to mention, and suggested topics. Never a diagnosis or medical advice.
    ai_discuss = (ai_guidance or {}).get("focus_summary") or (
        f"Discuss {symptoms[0].get('name')} and any other symptoms listed above." if symptoms
        else "Discuss the reason for this visit and any concerns you have."
    )
    ai_mention_items = []
    if personalization.get("biggest_concern"):
        ai_mention_items.append(personalization.get("biggest_concern"))
    for n in notes[:3]:
        if _classify_note(n) == "medication":
            ai_mention_items.append(n.get("content") or "")
    ai_topics = list((ai_guidance or {}).get("discussion_points") or [])
    if not ai_topics:
        ai_topics = pending_questions[:5] or [
            f"Ask about {s.get('name')}" for s in symptoms[:3]
        ]

    ai_mention_html = (
        "".join(f"<li>{esc(v)}</li>" for v in ai_mention_items if v)
        or '<li class="empty-inline">No additional context provided</li>'
    )
    ai_topics_html = (
        "".join(f"<li>{esc(v)}</li>" for v in ai_topics if v)
        or '<li class="empty-inline">No specific topics suggested</li>'
    )

    # ---- Provider Summary (separate, doctor-facing section) ------------------
    # Only patient-provided facts — no inference or hallucinated clinical
    # content — clearly labelled per PART 3 of the export spec.
    provider_symptom_rows = []
    for s in symptoms:
        bits = [esc(s.get("name", "symptom"))]
        if s.get("severity"):
            bits.append(f"severity {s.get('severity')}/10")
        if s.get("duration"):
            bits.append(f"duration {esc(s.get('duration'))}")
        if s.get("timing"):
            bits.append(f"timing {esc(s.get('timing'))}")
        provider_symptom_rows.append(f"<li>{', '.join(bits)}</li>")
    provider_symptoms_html = "".join(provider_symptom_rows) or '<li class="empty-inline">None reported</li>'

    goal_text = {
        'clear_diagnosis': 'Wants a clear diagnosis',
        'next_steps_plan': 'Wants next steps or a treatment plan',
        'tests_or_referrals': 'Wants tests or referrals',
        'heard_understood': 'Wants to feel heard and understood',
    }.get(personalization.get("appointment_outcome") or "", "") or personalization.get("main_reason") or "Not specified"

    provider_questions_html = (
        "".join(f"<li>{esc(q)}</li>" for q in all_pending)
        or '<li class="empty-inline">No questions provided</li>'
    )

    provider_notes_rows = []
    for n in notes:
        category = _classify_note(n)
        if category == "medication":
            provider_notes_rows.append(f"<li>Medication mentioned: {esc(n.get('content'))}</li>")
        elif n.get("content"):
            provider_notes_rows.append(f"<li>{esc(n.get('content'))}</li>")
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

      /* ---------- Focus block (signature accent) ---------- */
      .focus {{
        display: flex;
        gap: 14px;
        padding: 16px 0 18px 16px;
        border-left: 3px solid var(--ink);
        margin-bottom: 30px;
      }}
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
      .discuss-list {{
        margin-top: 10px;
        padding-left: 15px;
      }}
      .discuss-list li {{
        font-size: 10px;
        color: var(--charcoal);
        margin-bottom: 4px;
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

      /* ---------- Repeating entries (symptoms / notes) ---------- */
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

    <div class="focus">
      <div>
        <div class="focus-label">{esc(focus_header)}</div>
        <div class="focus-body">{focus_body}</div>
        {discussion_html}
      </div>
    </div>

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
    </section>

    <section>
      <div class="section-head">
        <span class="section-title">Patient Notes</span>
        <span class="section-count">{len(notes)} note{'s' if len(notes) != 1 else ''}</span>
      </div>
      {notes_html}
    </section>

    <section>
      <div class="section-head">
        <span class="section-title">AI Visit Preparation Summary</span>
      </div>
      <div class="ai-prep">
        <div class="ai-prep-subhead">What you want to discuss</div>
        <p class="entry-note">{esc(ai_discuss)}</p>
        <div class="ai-prep-subhead">Information that may be useful to mention</div>
        <ul>{ai_mention_html}</ul>
        <div class="ai-prep-subhead">Suggested topics to bring up</div>
        <ul>{ai_topics_html}</ul>
        <div class="ai-prep-note">
          This section only helps organize your conversation with your provider. It is not medical advice and does not diagnose any condition.
        </div>
      </div>
    </section>

    <section>
      <div class="section-head">
        <span class="section-title">Provider Summary</span>
      </div>
      <div class="provider-summary">
        <div class="provider-summary-label">Based only on patient-provided information</div>
        <div class="provider-subhead">Patient-reported concerns</div>
        <ul>{provider_symptoms_html}</ul>
        <div class="provider-subhead">Patient goals</div>
        <p class="entry-note">{esc(goal_text)}</p>
        <div class="provider-subhead">Patient questions</div>
        <ul>{provider_questions_html}</ul>
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