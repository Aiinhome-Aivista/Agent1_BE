"""
alert_models.py

Two tables:
  - alert_config  : role → escalation delay (minutes), read by the scheduler
  - email_log     : one row per email sent (initial or escalation)
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean
from app.core.database import Base


class AlertConfig(Base):
    """
    Configuration table — defines how long to wait before notifying each role.

    Rows (seeded on startup):
      role="DataOps Engineer",   delay_minutes=0   ← notified immediately on failure
      role="Data Engineer",      delay_minutes=2   ← notified 2 min after initial if no new run
      role="Data Platform Lead", delay_minutes=2   ← notified 2 min after initial if no new run

    Change delay_minutes in the DB row to change behaviour WITHOUT touching code.
    """
    __tablename__ = "alert_config"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    role           = Column(String(64), unique=True, nullable=False)
    delay_minutes  = Column(Integer, nullable=False, default=0)
    description    = Column(Text, nullable=True)
    updated_at     = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class EmailLog(Base):
    """
    Audit log — one row written every time the scheduler sends an email.

    email_type values:
      "initial"     → first email to DataOps Engineers
      "escalation"  → follow-up email to Data Engineer + Data Platform Lead

    Fields:
      pipeline_id       — pipelines.id FK (integer)
      pipeline_name     — human-readable name
      run_id            — pipeline_runs.id of the failed run that triggered this cycle
      email_type        — "initial" | "escalation"
      recipient_email   — who received this specific email
      recipient_role    — e.g. "DataOps Engineer"
      sent_at           — when the email was dispatched
      cycle_number      — which retry cycle (1 = first detection, 2 = still failing after 5 min…)
      new_run_found     — True if a new pipeline run was detected before escalation fired
                          (populated on escalation rows only)
      initial_sent_at   — copy of the initial email timestamp for easy querying on escalation rows
      escalation_sent_at— copy of escalation timestamp for easy querying
      status            — "sent" | "failed" | "skipped"
      notes             — free text (e.g. "SMTP error", "No recipients found")
      created_at
    """
    __tablename__ = "email_log"

    id                  = Column(Integer, primary_key=True, autoincrement=True)

    # Pipeline context
    pipeline_id         = Column(Integer, nullable=False, index=True)
    pipeline_name       = Column(String(512), nullable=False)
    run_id              = Column(Integer, nullable=True, index=True)   # the failed run

    # Email metadata
    email_type          = Column(String(16), nullable=False)           # "initial" | "escalation"
    recipient_email     = Column(String(255), nullable=False)
    recipient_role      = Column(String(64),  nullable=False)

    # Timestamps
    sent_at             = Column(DateTime, default=datetime.utcnow, index=True)
    initial_sent_at     = Column(DateTime, nullable=True)              # when cycle's initial mail went out
    escalation_sent_at  = Column(DateTime, nullable=True)              # when escalation went out

    # Loop metadata
    cycle_number        = Column(Integer, default=1)                   # retry iteration
    new_run_found       = Column(Boolean, nullable=True)               # for escalation rows

    # Result
    status              = Column(String(16), default="sent")           # "sent" | "failed" | "skipped"
    notes               = Column(Text, nullable=True)

    created_at          = Column(DateTime, default=datetime.utcnow)