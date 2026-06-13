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

    return f"""
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <style>
      body {{ font-family: -apple-system, Segoe UI, Arial, sans-serif; font-size: 12px; color: #111; }}
      h1 {{ font-size: 18px; margin: 0 0 8px 0; }}
      h2 {{ font-size: 13px; margin: 16px 0 6px 0; }}
      .muted {{ color: #555; }}
      .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 10px; margin-top: 8px; }}
      ul {{ margin: 6px 0 0 18px; }}
    </style>
  </head>
  <body>
    <h1>NeuraVia · Pre-Visit Summary</h1>
    <div class="muted">
      Appointment: {esc(appointment.get("appointment_date"))} {esc(appointment.get("appointment_time"))}
      · Specialty: {esc(appointment.get("specialty"))}
    </div>

    <div class="card">
      <h2>{esc(focus_header)}</h2>
      <div>{esc(focus_summary) or "—"}</div>
      {f"<ul>{li((ai_guidance or {}).get('discussion_points') or [])}</ul>" if (ai_guidance or {}).get("discussion_points") else ""}
    </div>

    <h2>Symptoms</h2>
    <ul>{li([f"{s.get('name')} (severity {s.get('severity')}/10)" for s in symptoms]) or "<li>—</li>"}</ul>

    <h2>Questions</h2>
    <ul>{li([q.get('text') for q in questions if not q.get('is_answered')]) or "<li>—</li>"}</ul>

    <h2>Notes</h2>
    <ul>{li([f"{n.get('title')}: {n.get('content')}" for n in notes]) or "<li>—</li>"}</ul>

    <div class="muted" style="margin-top:16px;">
      Generated at {esc(datetime.utcnow().isoformat())} UTC · Not medical advice.
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

