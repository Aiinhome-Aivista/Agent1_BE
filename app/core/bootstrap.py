"""
Bootstrap helpers run during application startup.
Currently: ensures a default demo user exists so anyone can sign in
immediately without registering. The credentials are configurable via env
vars (DEMO_USER_EMAIL / DEMO_USER_PASSWORD); defaults shown in the login
page hint.
This is a convenience for demos / local dev. In production, set
DEMO_USER_PASSWORD to something else (or remove the seed entirely).
"""
from __future__ import annotations

import logging
import os

from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.core.security import hash_password
from app.models import User
from app.models.user import UserRole

logger = logging.getLogger(__name__)

DEFAULT_EMAIL    = os.environ.get("DEMO_USER_EMAIL",    "demo@example.com")
DEFAULT_PASSWORD = os.environ.get("DEMO_USER_PASSWORD", "demo1234")
DEFAULT_NAME     = "Demo User"


def ensure_demo_user() -> None:
    """Create the demo user on first boot. Idempotent: re-running is a no-op."""
    db: Session = SessionLocal()
    try:
        existing = db.query(User).filter(User.email == DEFAULT_EMAIL).first()
        if existing:
            return

        user = User(
            email           = DEFAULT_EMAIL,
            full_name       = DEFAULT_NAME,
            hashed_password = hash_password(DEFAULT_PASSWORD),
            is_active       = True,
            is_admin        = True,
            role            = UserRole.data_engineer,
        )
        db.add(user)
        db.commit()
        logger.info(
            "Seeded demo user: email=%s password=%s (CHANGE FOR PRODUCTION)",
            DEFAULT_EMAIL, DEFAULT_PASSWORD,
        )
    finally:
        db.close()


# ── NEW: alert config seeding ─────────────────────────────────────────────────

def ensure_alert_config() -> None:
    """
    Insert the three default alert-config rows if they don't already exist.
    Idempotent: safe to call on every startup.

    The pipeline alert scheduler reads delay_minutes from this table at
    runtime, so an admin can change the value in the DB and it takes effect
    on the next alert cycle — no code change or restart required.
    """
    from app.models.alert_models import AlertConfig   # local import avoids circular deps

    defaults = [
        {
            "role":          "DataOps Engineer",
            "delay_minutes": 0,
            "description":   "Notified immediately when a pipeline fails",
        },
        {
            "role":          "Data Engineer",
            "delay_minutes": 2,
            "description":   "Escalation: notified 2 min after initial alert if no new run found",
        },
        {
            "role":          "Data Platform Lead",
            "delay_minutes": 2,
            "description":   "Escalation: notified 2 min after initial alert if no new run found",
        },
    ]

    db: Session = SessionLocal()
    try:
        for row in defaults:
            exists = db.query(AlertConfig).filter(AlertConfig.role == row["role"]).first()
            if not exists:
                db.add(AlertConfig(**row))
                logger.info("Seeded alert_config row: role=%s delay=%d min",
                            row["role"], row["delay_minutes"])
        db.commit()
    finally:
        db.close()