"""
Real integration tests for the Mistral diagnose-and-fix service.

We feed `analyze_failure` realistic synthetic logs from the three connector
types and assert that:
  - the call returns the contract dict
  - `summary` and `root_cause` are non-trivial strings
  - `confidence` is a float in [0, 1]
  - the model name comes back

We deliberately use synthetic-but-plausible logs so the tests don't depend on
your real account having a recent failure. The point is that Mistral is being
called for real and producing a usable JSON shape.

Costs ~3-5 cents per run depending on the model. To run only the cheapest
case set:  TEST_MISTRAL_QUICK=1

Requires: MISTRAL_API_KEY
"""
from __future__ import annotations

import os

import pytest

from app.services.mistral_service import mistral_service


pytestmark = [pytest.mark.integration, pytest.mark.slow]


# ---------------------------------------------------------------------------
# Synthetic but plausible failure scenarios for each connector type
# ---------------------------------------------------------------------------
ADF_FAILURE = {
    "pipeline_name": "etl_daily_orders",
    "connector_type": "ADF",
    "error_message": (
        "ErrorCode=AzureSqlOperationFailed,Type=Microsoft.DataTransfer.Common"
        ".Shared.HybridDeliveryException, Message=A database operation failed."
        " Cannot insert duplicate key row in object 'dbo.orders' with unique "
        "index 'IX_orders_id'."
    ),
    "logs": [
        {
            "timestamp": "2026-05-09T03:14:22Z", "level": "INFO",
            "source": "CopyOrders", "message": "Activity start",
        },
        {
            "timestamp": "2026-05-09T03:14:25Z", "level": "INFO",
            "source": "CopyOrders",
            "message": "Source: AzureBlobStorage, sink: AzureSqlDatabase",
        },
        {
            "timestamp": "2026-05-09T03:14:48Z", "level": "ERROR",
            "source": "CopyOrders",
            "message": (
                "Failure happened on 'Sink' side. ErrorCode=AzureSqlOperation"
                "Failed. Cannot insert duplicate key in 'dbo.orders'."
            ),
        },
    ],
    "metadata": {"activities": ["CopyOrders"], "folder": "etl"},
}


DATABRICKS_FAILURE = {
    "pipeline_name": "spark_aggregation_job",
    "connector_type": "DATABRICKS",
    "error_message": "Run failed with error: java.lang.OutOfMemoryError: GC overhead limit exceeded",
    "logs": [
        {
            "timestamp": "2026-05-09T01:05:00Z", "level": "INFO",
            "source": "agg_task", "message": "Reading 240GB parquet from s3://lake/events",
        },
        {
            "timestamp": "2026-05-09T01:17:21Z", "level": "WARNING",
            "source": "agg_task", "message": "Executor lost: 4-2c5e9. Reason: container killed by YARN for exceeding memory limits",
        },
        {
            "timestamp": "2026-05-09T01:17:41Z", "level": "ERROR",
            "source": "stderr",
            "message": (
                "java.lang.OutOfMemoryError: GC overhead limit exceeded\n"
                "  at org.apache.spark.sql.execution.aggregate.HashAggregate"
                "Exec.doExecute(HashAggregateExec.scala:152)"
            ),
        },
    ],
    "metadata": {"tasks": ["agg_task"], "creator_user_name": "data@team.com"},
}


GIT_FAILURE = {
    "pipeline_name": "ops/.github/workflows/ci.yml",
    "connector_type": "GIT",
    "error_message": None,  # GitHub Actions reports per-step status, not a top-level message
    "logs": [
        {
            "timestamp": "2026-05-09T09:00:01Z", "level": "INFO",
            "source": "build", "message": "Setup Node.js 18.x",
        },
        {
            "timestamp": "2026-05-09T09:00:30Z", "level": "INFO",
            "source": "build", "message": "Run npm ci",
        },
        {
            "timestamp": "2026-05-09T09:01:12Z", "level": "ERROR",
            "source": "build/Run npm ci",
            "message": (
                "npm ERR! code ERESOLVE\n"
                "npm ERR! ERESOLVE could not resolve\n"
                "npm ERR! While resolving: app@1.0.0\n"
                "npm ERR! Found: react@18.2.0\n"
                "npm ERR! Could not resolve dependency: peer react@\"^17.0.0\" "
                "from react-modal-dialog@4.1.2\n"
                "Process completed with exit code 1."
            ),
        },
    ],
    "metadata": {"repo": "ops", "workflow_id": 12345, "html_url": "https://example"},
}


def _assert_valid_response(result: dict, scenario_name: str) -> None:
    """Common contract assertions for any analyze_failure response."""
    # Shape contract
    for key in ("summary", "root_cause", "suggested_fix", "fix_patch",
                "confidence", "model", "raw_response"):
        assert key in result, f"missing key {key!r} in response"

    # Substance: a non-trivial diagnosis must come back
    assert isinstance(result["summary"], str) and len(result["summary"]) >= 5, \
        f"[{scenario_name}] summary too short: {result['summary']!r}"

    assert isinstance(result["root_cause"], str) and len(result["root_cause"]) >= 10, \
        f"[{scenario_name}] root_cause too short: {result['root_cause']!r}"

    # Confidence in [0, 1]
    conf = result["confidence"]
    assert isinstance(conf, (int, float))
    assert 0.0 <= float(conf) <= 1.0, f"[{scenario_name}] confidence out of range: {conf}"

    # We asked for it; it should report which model answered
    assert result["model"], f"[{scenario_name}] model name missing"

    # Should not be the canned 'analysis unavailable' error path
    assert result["summary"] != "LLM analysis unavailable", \
        f"[{scenario_name}] Mistral call failed: {result['raw_response']}"


class TestMistralAnalyzesADF:
    def test_diagnoses_duplicate_key_failure(self, mistral_configured):
        result = mistral_service.analyze_failure(**{
            k: ADF_FAILURE[k] for k in
            ("pipeline_name", "connector_type", "error_message", "logs", "metadata")
        })
        _assert_valid_response(result, "ADF")
        # The diagnosis should mention the actual problem in some form.
        # We're loose about exact wording — any of these signals counts.
        text = (result["summary"] + " " + result["root_cause"] + " "
                + result["suggested_fix"]).lower()
        assert any(w in text for w in ("duplicate", "unique", "primary key",
                                       "constraint", "idempot")), \
            f"Diagnosis didn't reference the duplicate-key issue: {text}"


class TestMistralAnalyzesDatabricks:
    def test_diagnoses_oom_failure(self, mistral_configured):
        if os.environ.get("TEST_MISTRAL_QUICK") == "1":
            pytest.skip("Skipped under TEST_MISTRAL_QUICK")
        result = mistral_service.analyze_failure(**{
            k: DATABRICKS_FAILURE[k] for k in
            ("pipeline_name", "connector_type", "error_message", "logs", "metadata")
        })
        _assert_valid_response(result, "DATABRICKS")
        text = (result["summary"] + " " + result["root_cause"] + " "
                + result["suggested_fix"]).lower()
        assert any(w in text for w in ("memory", "oom", "executor", "heap",
                                       "cluster size", "instance")), \
            f"Diagnosis didn't reference the OOM/memory issue: {text}"


class TestMistralAnalyzesGitHubActions:
    def test_diagnoses_npm_peer_dep_failure(self, mistral_configured):
        if os.environ.get("TEST_MISTRAL_QUICK") == "1":
            pytest.skip("Skipped under TEST_MISTRAL_QUICK")
        result = mistral_service.analyze_failure(**{
            k: GIT_FAILURE[k] for k in
            ("pipeline_name", "connector_type", "error_message", "logs", "metadata")
        })
        _assert_valid_response(result, "GIT")
        text = (result["summary"] + " " + result["root_cause"] + " "
                + result["suggested_fix"]).lower()
        assert any(w in text for w in ("peer", "dependency", "react",
                                       "eresolve", "legacy-peer-deps", "version")), \
            f"Diagnosis didn't reference the peer-dep issue: {text}"


class TestMistralReturnsActionableFix:
    """
    The whole point of the system is 'show how to solve the error'. Verify
    that for at least one realistic failure, Mistral returns a non-empty
    suggested_fix that reads like instructions, not an apology.
    """
    def test_suggested_fix_is_actionable(self, mistral_configured):
        result = mistral_service.analyze_failure(**{
            k: ADF_FAILURE[k] for k in
            ("pipeline_name", "connector_type", "error_message", "logs", "metadata")
        })
        fix = result["suggested_fix"]
        assert fix and len(fix) > 30, f"suggested_fix too thin: {fix!r}"
        # Should contain at least one imperative-like cue
        cues = ("add", "use", "set", "change", "remove", "configure",
                "update", "ensure", "modify", "merge", "drop", "deduplicate",
                "switch", "alter")
        assert any(c in fix.lower() for c in cues), \
            f"suggested_fix doesn't read like instructions: {fix!r}"
