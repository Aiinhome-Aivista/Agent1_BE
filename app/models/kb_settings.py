"""
Knowledge-base settings (singleton row).

Holds the *daily refresh* schedule the operator configures from the UI. The
brief asks for "knowledge base daily fixed time we can set this from the SQL
table" — so the schedule lives in SQL (this table) rather than an env var, and
the scheduler reads it each minute.

Stored as a single row (id == 1). `daily_refresh_time` is "HH:MM" in 24h,
interpreted in the server's UTC clock (documented in the integration guide).
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text
from sqlalchemy.orm import Session

from app.core.database import Base


class KBSettings(Base):
    __tablename__ = "kb_settings"

    id                   = Column(Integer, primary_key=True, default=1)
    daily_refresh_enabled = Column(Boolean, default=True)
    # "HH:MM" 24h, UTC. The scheduler fires once when the clock first matches.
    daily_refresh_time   = Column(String(5), default="02:00")

    # Bookkeeping so the UI can show the last run and prevent double-fires.
    last_run_at          = Column(DateTime, nullable=True)
    last_run_date        = Column(String(10), nullable=True)   # "YYYY-MM-DD" guard
    last_run_summary     = Column(Text, nullable=True)         # JSON string

    updated_at           = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


def get_or_create_settings(db: Session) -> "KBSettings":
    row = db.query(KBSettings).filter(KBSettings.id == 1).first()
    if row is None:
        row = KBSettings(id=1, daily_refresh_enabled=True, daily_refresh_time="02:00")
        db.add(row)
        db.commit()
        db.refresh(row)
    return row
