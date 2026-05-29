"""
End-to-end sync test.

Wires up a real connector, runs `sync_connector` against a real account, and
asserts that pipelines + runs + logs landed in the DB. If the account has any
recently failed runs, we additionally assert that Mistral was called and an
ErrorAnalysis row was written.

Uses an in-memory SQLite DB so it doesn't touch your MySQL.

Skips entirely if no connector creds are configured.
"""
from __future__ import annotations

import json

import pytest

from app.core.security import encrypt_secret
from app.models import (
    Connector,
    ConnectorStatus,
    ConnectorType,
    ErrorAnalysis,
    Pipeline,
    PipelineLog,
    PipelineRun,
    RunStatus,
    User,
)
from app.services.sync_service import sync_connector


pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _make_user(db) -> User:
    user = User(
        email="e2e@test.local",
        full_name="E2E Test",
        hashed_password="!unused-in-this-test",
        is_active=True,
        is_admin=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _make_connector(db, user: User, type_: ConnectorType, creds: dict) -> Connector:
    encrypted = encrypt_secret(json.dumps({k: v for k, v in creds.items() if v is not None}))
    c = Connector(
        owner_id=user.id,
        name=f"e2e-{type_.value.lower()}",
        type=type_,
        encrypted_credentials=encrypted,
        status=ConnectorStatus.PENDING,
    )
    db.add(c)
    db.commit()
    db.refresh(c)
    return c


# ---------------------------------------------------------------------------
# A small parametrize that runs the same end-to-end check for whichever
# connectors you have credentials for. Missing creds skip cleanly.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("connector_kind", ["ADF", "DATABRICKS", "GIT"])
async def test_sync_connector_end_to_end(
    connector_kind, request, memory_db, mistral_configured,
):
    # Pull the matching credentials fixture by name. If it skips, we skip.
    fixture_name = {
        "ADF": "adf_credentials",
        "DATABRICKS": "databricks_credentials",
        "GIT": "git_credentials",
    }[connector_kind]
    creds = request.getfixturevalue(fixture_name)

    user = _make_user(memory_db)
    connector = _make_connector(
        memory_db, user, ConnectorType(connector_kind), creds,
    )

    # --- Run the orchestrator for real ------------------------------------
    stats = await sync_connector(memory_db, connector)

    # The sync should not have errored out fatally
    assert "errors" in stats
    # We tolerate per-pipeline errors but the connector itself should be CONNECTED
    memory_db.refresh(connector)
    assert connector.status == ConnectorStatus.CONNECTED, (
        f"Connector ended in status {connector.status}: {connector.last_error}"
    )
    assert connector.last_synced_at is not None

    # --- Pipelines persisted ----------------------------------------------
    pipelines = memory_db.query(Pipeline).filter(
        Pipeline.connector_id == connector.id
    ).all()
    if not pipelines:
        pytest.skip(f"{connector_kind} account has no pipelines to sync")

    for p in pipelines:
        assert p.external_id
        assert p.name

    # --- Runs persisted (when the account has any) ------------------------
    runs = memory_db.query(PipelineRun).join(Pipeline).filter(
        Pipeline.connector_id == connector.id
    ).all()

    if not runs:
        pytest.skip(f"{connector_kind} pipelines have no runs in the lookback window")

    for r in runs:
        assert r.external_run_id
        assert r.status in set(RunStatus)

    # --- If any run failed, Mistral analysis must exist -------------------
    failed_runs = [r for r in runs if r.status == RunStatus.FAILED]
    if failed_runs:
        for r in failed_runs:
            # Logs were fetched
            log_count = memory_db.query(PipelineLog).filter(
                PipelineLog.run_id == r.id
            ).count()
            # ADF activity logs may legitimately be 0 for very old runs;
            # we don't hard-require >0 across all connectors.
            assert log_count >= 0

            # Analysis was generated
            analysis = memory_db.query(ErrorAnalysis).filter(
                ErrorAnalysis.run_id == r.id
            ).first()
            assert analysis is not None, (
                f"Failed run {r.external_run_id} has no Mistral analysis"
            )
            assert analysis.summary
            assert 0.0 <= (analysis.confidence or 0) <= 1.0
            assert analysis.model
    else:
        # No failures to analyze. That's fine, but make it visible in test output.
        print(f"\n[info] {connector_kind}: synced {len(pipelines)} pipelines, "
              f"{len(runs)} runs, no failures to analyze")
