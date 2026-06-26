from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional, Tuple


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

    def li(items):
        return "".join(f"<li>{esc(x)}</li>" for x in items)

    focus_header = (ai_guidance or {}).get("focus_header") or "Visit Focus"
    focus_summary = (ai_guidance or {}).get("focus_summary") or (personalization.get("main_reason") or "")

    appointment_date = esc(appointment.get("appointment_date") or "")
    appointment_time = esc(appointment.get("appointment_time") or "")
    specialty = esc(appointment.get("specialty") or "")
    doctor_name = esc(appointment.get("doctor_name") or "Healthcare Provider")

    # Pre-build dynamic HTML blocks to avoid backslash-in-f-string issues
    discussion_html = (
        f"<ul style='margin-top: 10px;'>{li((ai_guidance or {}).get('discussion_points') or [])}</ul>"
        if (ai_guidance or {}).get("discussion_points") else ""
    )

    focus_body = esc(focus_summary) if focus_summary else '<span class="empty">No focus area specified</span>'

    symptom_items = [f"{s.get('name', 'Symptom')} (severity {s.get('severity', '0')}/10)" for s in symptoms]
    symptoms_html = (
        f"<div class='content-card'><ul>{li(symptom_items)}</ul></div>"
        if symptoms else "<div class='content-card'><span class='empty'>No symptoms recorded</span></div>"
    )

    pending_questions = [q.get("text", "Question") for q in questions if not q.get("is_answered")]
    questions_html = (
        f"<div class='content-card'><ul>{li(pending_questions)}</ul></div>"
        if pending_questions else "<div class='content-card'><span class='empty'>No outstanding questions</span></div>"
    )

    note_items = [f"{n.get('title', 'Note')}: {n.get('content', '')}" for n in notes]
    notes_html = (
        f"<div class='content-card'><ul>{li(note_items)}</ul></div>"
        if notes else "<div class='content-card'><span class='empty'>No notes available</span></div>"
    )

    generated_at = esc(datetime.utcnow().strftime("%B %d, %Y at %H:%M UTC"))

    return f"""
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <style>
      @page {{
        size: A4;
        margin: 20mm 20mm 20mm 20mm;
      }}
      * {{
        margin: 0;
        padding: 0;
        box-sizing: border-box;
      }}
      body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', 'Inter', sans-serif;
        font-size: 10px;
        color: #111111;
        line-height: 1.6;
        background: #ffffff;
      }}

      /* Header Section */
      .header {{
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 28px;
        padding-bottom: 20px;
        border-bottom: 3px solid #111111;
      }}
      .header-content h1 {{
        font-size: 28px;
        font-weight: 700;
        color: #111111;
        margin-bottom: 4px;
        letter-spacing: -0.5px;
      }}
      .header-badge {{
        display: inline-block;
        background: #f0f0f0;
        color: #111111;
        padding: 6px 12px;
        border-radius: 8px;
        font-size: 9px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
      }}
      .header-meta {{
        font-size: 9px;
        color: #555555;
        line-height: 1.5;
      }}

      /* Metadata Bar */
      .metadata {{
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 16px;
        margin-bottom: 24px;
        padding: 14px 14px;
        background: #f5f5f5;
        border-radius: 12px;
      }}
      .meta-item {{
        display: flex;
        flex-direction: column;
      }}
      .meta-label {{
        font-size: 8px;
        font-weight: 600;
        color: #555555;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        margin-bottom: 4px;
      }}
      .meta-value {{
        font-size: 11px;
        font-weight: 600;
        color: #111111;
      }}

      /* Section Headers */
      h2 {{
        font-size: 14px;
        font-weight: 700;
        color: #111111;
        margin-top: 24px;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 2px solid #111111;
        text-transform: uppercase;
        letter-spacing: 0.3px;
      }}

      /* Cards and Content Areas */
      .content-card {{
        background: #f5f5f5;
        border: 1px solid #d4d4d4;
        border-radius: 12px;
        padding: 14px 14px;
        margin-bottom: 16px;
      }}
      .focus-card {{
        background: #f0f0f0;
        border: 1px solid #cccccc;
        border-left: 4px solid #111111;
      }}
      .card-title {{
        font-size: 11px;
        font-weight: 700;
        color: #111111;
        text-transform: uppercase;
        letter-spacing: 0.3px;
        margin-bottom: 6px;
      }}
      .card-content {{
        font-size: 10px;
        color: #333333;
        line-height: 1.7;
      }}

      /* Lists */
      ul {{
        margin: 8px 0 0 20px;
        list-style: none;
      }}
      li {{
        margin-bottom: 6px;
        font-size: 10px;
        color: #333333;
        position: relative;
        padding-left: 12px;
        line-height: 1.6;
      }}
      li:before {{
        content: "\25AA";
        position: absolute;
        left: 0;
        color: #111111;
        font-weight: bold;
        font-size: 8px;
      }}

      /* Empty State */
      .empty {{
        color: #888888;
        font-style: italic;
        font-size: 10px;
      }}

      /* Disclaimer Section */
      .disclaimer {{
        margin-top: 28px;
        padding: 14px 14px;
        background: #f5f5f5;
        border: 1.5px solid #cccccc;
        border-radius: 12px;
        border-left: 4px solid #111111;
      }}
      .disclaimer-title {{
        font-size: 10px;
        font-weight: 700;
        color: #111111;
        text-transform: uppercase;
        letter-spacing: 0.3px;
        margin-bottom: 6px;
      }}
      .disclaimer-text {{
        font-size: 9px;
        color: #444444;
        line-height: 1.6;
      }}

      /* Footer */
      .footer {{
        margin-top: 32px;
        padding-top: 16px;
        border-top: 2px solid #cccccc;
        text-align: center;
        font-size: 8px;
        color: #888888;
      }}
      .footer-brand {{
        font-weight: 700;
        color: #111111;
        margin-bottom: 4px;
      }}
      .footer-legal {{
        font-size: 7px;
        color: #aaaaaa;
        margin-top: 6px;
      }}
    </style>
  </head>
  <body>
    <!-- Header -->
    <div class="header">
      <div class="header-content">
        <div class="header-badge">Medical Summary</div>
        <h1>Visit Summary</h1>
        <div class="header-meta">
          Prepared by <span style="font-weight: 600; color: #111111;">SyniVia</span>
        </div>
      </div>
    </div>

    <!-- Metadata Bar -->
    <div class="metadata">
      <div class="meta-item">
        <span class="meta-label">Date</span>
        <span class="meta-value">{appointment_date if appointment_date else "—"}</span>
      </div>
      <div class="meta-item">
        <span class="meta-label">Time</span>
        <span class="meta-value">{appointment_time if appointment_time else "—"}</span>
      </div>
      <div class="meta-item">
        <span class="meta-label">Specialty</span>
        <span class="meta-value">{specialty if specialty else "—"}</span>
      </div>
    </div>

    <!-- Focus Section -->
    <div class="content-card focus-card">
      <div class="card-title">{esc(focus_header)}</div>
      <div class="card-content">{focus_body}</div>
      {discussion_html}
    </div>

    <!-- Symptoms -->
    <h2>Symptoms &amp; Complaints</h2>
    {symptoms_html}

    <!-- Questions -->
    <h2>Questions for Provider</h2>
    {questions_html}

    <!-- Notes -->
    <h2>Clinical Notes</h2>
    {notes_html}

    <!-- Disclaimer -->
    <div class="disclaimer">
      <div class="disclaimer-title">&#9888; Important Disclaimer</div>
      <div class="disclaimer-text">
        This document is generated by <strong>SyniVia</strong> for organizational and record-keeping purposes only.
        <strong>SyniVia is not a medical diagnostic tool and does not provide medical advice.</strong>
        This summary must not be used as a substitute for professional clinical judgment or diagnosis.
        Always consult with a qualified healthcare provider regarding any medical concerns.
        <strong>SyniVia is not liable for any errors, omissions, or consequences arising from the use of this document.</strong>
      </div>
    </div>

    <!-- Footer -->
    <div class="footer">
      <div class="footer-brand">SyniVia</div>
      <div>Medical Visit Preparation &amp; Summary</div>
      <div class="footer-legal">
        Generated on {generated_at} |
        This document is confidential and intended for patient use only.
      </div>
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