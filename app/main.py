"""
FastAPI application factory.
- Mounts REST API + WebSocket
- Starts an APScheduler that periodically syncs all connectors so failures are
  detected and analysed within ~1 minute even if no user is currently looking
  at the UI.
"""
import asyncio
import logging
from contextlib import asynccontextmanager

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import api_router, websocket_router
from app.core.bootstrap import (
    DEFAULT_EMAIL,
    DEFAULT_PASSWORD,
    ensure_demo_user,
    ensure_alert_config,          # ← NEW import
)
from app.core.config import settings
from app.core.database import Base, engine
from app.services.sync_service import sync_all_connectors
from app.services.pipeline_alert_service import run_pipeline_alert_scheduler  # ← NEW import

# Side-effect import: ensures every model registers with Base.metadata
# before create_all() runs at startup. Reading __all__ keeps linters happy.
from app import models as _models_registry
_REGISTERED_MODELS = _models_registry.__all__

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("pipeline-monitor")

scheduler = AsyncIOScheduler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Auto-create tables on startup (includes the two new alert tables).
    # In production prefer Alembic migrations, but this gives a one-command
    # demo experience.
    Base.metadata.create_all(bind=engine)

    # Seed default demo user so anyone can sign in instantly.
    ensure_demo_user()

    # Seed default alert-config rows (DataOps Engineer=0 min, others=2 min).
    ensure_alert_config()                                    # ← NEW

    scheduler.add_job(
        sync_all_connectors,
        trigger="interval",
        seconds=settings.PIPELINE_SYNC_INTERVAL,
        id="periodic_sync",
        max_instances=1,
        coalesce=True,
    )
    scheduler.start()
    logger.info("Scheduler started: sync every %ds", settings.PIPELINE_SYNC_INTERVAL)

    # Run an initial sync once on boot so the dashboard is populated quickly.
    asyncio.create_task(sync_all_connectors())

    # Start the pipeline failure alert scheduler (initial + escalation emails). ← NEW
    asyncio.create_task(run_pipeline_alert_scheduler())      # ← NEW

    try:
        yield
    finally:
        scheduler.shutdown(wait=False)


app = FastAPI(
    title="Pipeline Monitor",
    description=(
        "Monitor & auto-diagnose ADF, Databricks, and Git pipelines with "
        "real-time updates and Mistral-powered fix suggestions."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.FRONTEND_URL,
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:3004",
        "http://localhost:3002",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"name": "Pipeline Monitor", "version": "1.0.0", "status": "ok"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/api/v1/demo")
def demo_info():
    """
    Public endpoint that exposes the seeded demo credentials so the login
    page can offer one-click sign-in. Safe to expose because this user only
    exists in dev/demo deployments.
    """
    return {
        "email":   DEFAULT_EMAIL,
        "password": DEFAULT_PASSWORD,
        "enabled": True,
    }


app.include_router(api_router)
app.include_router(websocket_router)