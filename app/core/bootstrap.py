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

# pyrefly: ignore [missing-import]
from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.core.security import hash_password
from app.models import User
from app.models.user import UserRole


logger = logging.getLogger(__name__)


DEFAULT_EMAIL = os.environ.get("DEMO_USER_EMAIL", "demo@example.com")
DEFAULT_PASSWORD = os.environ.get("DEMO_USER_PASSWORD", "demo1234")
DEFAULT_NAME = "Demo User"


def ensure_demo_user() -> None:
    """Create the demo user on first boot. Idempotent: re-running is a no-op."""
    db: Session = SessionLocal()
    try:
        existing = db.query(User).filter(User.email == DEFAULT_EMAIL).first()
        if existing:
            return
        user = User(
            email=DEFAULT_EMAIL,
            full_name=DEFAULT_NAME,
            hashed_password=hash_password(DEFAULT_PASSWORD),
            is_active=True,
            is_admin=True,
            role=UserRole.data_engineer,
        )
        db.add(user)
        db.commit()
        logger.info(
            "Seeded demo user: email=%s password=%s (CHANGE FOR PRODUCTION)",
            DEFAULT_EMAIL, DEFAULT_PASSWORD,
        )
    finally:
        db.close()
