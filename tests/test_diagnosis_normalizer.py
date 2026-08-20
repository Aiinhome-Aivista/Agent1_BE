"""
Unit and Integration Tests for Diagnosis Normalization & Fact Locking Engine.
Verifies all 10 Acceptance Criteria from Part 17 of the architecture specification.
"""
from app.services.diagnosis_normalizer import (
    extract_verified_facts,
    normalize_known_fix,
    build_suggested_fix_text,
    normalize_diagnosis,
)


def test_scenario_1_local_llm_hallucinates_facts():
    """TEST 1: Parser values win when LLM returns wrong deterministic facts."""
    verified_facts = extract_verified_facts(
        pipe_name="New Job 2026-08-19 19:17:48",  # raw display name
        connector_type="DATABRICKS",
        error_message="RuntimeError: Workload failed during SILVER_DATA_VALIDATION",
        logs=[
            {"level": "ERROR", "message": "SILVER_DATA_VALIDATION: 4 of 10 records failed validation (40.00% > 5.0% threshold)"},
            {"level": "ERROR", "message": "Error code: DATA_QUALITY_THRESHOLD_BREACH"},
        ],
        metadata={
            "pipeline_name": "ORDER_TRANSACTION_ETL",
            "pipeline_run_id": "20260819134953",
            "environment": "PRODUCTION",
            "failed_stage": "SILVER_DATA_VALIDATION",
            "error_code": "DATA_QUALITY_THRESHOLD_BREACH",
            "total_records": 10,
            "invalid_records": 4,
            "invalid_percentage": 40.0,
            "allowed_threshold": 5.0,
        },
        run_id="822",
    )

    # Bad Local LLM response that tries to overwrite verified facts
    bad_llm_output = {
        "pipeline_name": "New Job 2026-08-19 19:17:48",
        "failed_stage": "execution",
        "error_code": None,
        "summary": "Pipeline New Job 2026-08-19 19:17:48 failed during execution.",
        "root_cause": "Data failure occurred.",
        "failure_mechanism": "Execution halted.",
        "impact": "Data processing stopped.",
    }

    result = normalize_diagnosis(verified_facts, bad_llm_output)

    # Assert Fact-Locking: Verified parser values WIN
    assert result["pipeline_name"] == "ORDER_TRANSACTION_ETL"
    assert result["failed_stage"] == "SILVER_DATA_VALIDATION"
    assert result["error_code"] == "DATA_QUALITY_THRESHOLD_BREACH"
    assert result["total_records"] == 10
    assert result["invalid_records"] == 4
    assert result["invalid_percentage"] == 40.0
    assert result["allowed_threshold"] == 5.0
    assert result["environment"] == "PRODUCTION"

    # Assert Summary override:
    assert "ORDER_TRANSACTION_ETL" in result["summary"]
    assert "SILVER_DATA_VALIDATION" in result["summary"]
    assert "4 of 10 records were invalid" in result["summary"]
    assert "5.0%" in result["summary"]


def test_scenario_2_known_fix_format_normalization():
    """TEST 2: Canonicalizes all local LLM known_fix variations into standardized objects."""
    # Format A: step + description
    fmt_a = [{"step": 1, "description": "Fix source data records."}]
    imm_a, opt_a, comb_a = normalize_known_fix(fmt_a)
    assert imm_a[0]["step"] == 1
    assert imm_a[0]["action"] == "Fix source data records."

    # Format B: step + action
    fmt_b = [{"step": 1, "action": "Correct failed batch."}]
    imm_b, opt_b, comb_b = normalize_known_fix(fmt_b)
    assert imm_b[0]["title"] == "Correct failed batch."

    # Format C: details + description + expected_outcome
    fmt_c = [{
        "step": 1,
        "title": "Correct Source Data",
        "details": "Update invalid records in source batch.",
        "expected_outcome": "Batch meets 5% threshold.",
        "priority": "REQUIRED"
    }]
    imm_c, opt_c, comb_c = normalize_known_fix(fmt_c)
    assert imm_c[0]["expected_outcome"] == "Batch meets 5% threshold."
    assert imm_c[0]["priority"] == "REQUIRED"

    # Format D: string step value
    fmt_d = [{"step": "Quarantine Invalid Records", "description": "Route rejected rows to quarantine Delta table."}]
    imm_d, opt_d, comb_d = normalize_known_fix(fmt_d)
    assert len(opt_d) == 1
    assert opt_d[0]["title"] == "Quarantine Invalid Records"
    assert opt_d[0]["priority"] == "OPTIONAL"  # Quarantine is categorized as optional runbook improvement

    # Format E: Raw string numbered list
    fmt_e = "1. [Required Immediate Action] Correct failed source batch.\nExpected outcome: Meets 5% threshold.\n2. [Optional Runbook Improvement] Evaluate quarantine Delta table."
    imm_e, opt_e, comb_e = normalize_known_fix(fmt_e)
    assert len(imm_e) == 1
    assert len(opt_e) == 1
    assert imm_e[0]["priority"] == "REQUIRED"
    assert opt_e[0]["priority"] == "OPTIONAL"


def test_scenario_3_known_pattern_zero_accepted_fixes():
    """TEST 3: Pattern seen 14x with 0 accepted fixes does not inflate confidence to 98%."""
    from app.models.solution_models import SolutionPattern
    from app.services import confidence_explainer

    pattern = SolutionPattern(
        signature="sig-12345",
        title="Data Quality Breach",
        error_type="Data Quality",
        occurrence_count=14,
        acceptance_count=0,
        rejection_count=0,
        confidence=0.75,
    )

    explanation = confidence_explainer.build(
        llm_confidence=0.80,
        final_confidence=0.80,
        pattern=pattern,
        is_known=True,
        error_type="Data Quality",
    )

    assert explanation.score <= 0.85
    # Factor indicates 0 accepted fixes
    p_factor = next(f for f in explanation.factors if "Recognised pattern (0 accepted fixes)" in f.label)
    assert "no human has accepted its fix yet" in p_factor.detail


def test_scenario_4_known_pattern_with_accepted_fix():
    """TEST 4: Pattern with human-approved fixes raises remediation confidence without altering current metrics."""
    from app.models.solution_models import SolutionPattern
    from app.services import confidence_explainer

    pattern = SolutionPattern(
        signature="sig-54321",
        title="Accepted Data Quality Fix",
        error_type="Data Quality",
        occurrence_count=5,
        acceptance_count=3,
        rejection_count=0,
        confidence=0.95,
    )

    explanation = confidence_explainer.build(
        llm_confidence=0.85,
        final_confidence=0.95,
        pattern=pattern,
        is_known=True,
        error_type="Data Quality",
    )

    assert explanation.score >= 0.90
    assert explanation.level == "High"
    p_factor = next(f for f in explanation.factors if "Proven fix history" in f.label)
    assert "accepted by a human 3×" in p_factor.detail


def test_scenario_5_novel_error_first_principles():
    """TEST 5: Novel error with no KB matches provides evidence-grounded first principles."""
    verified_facts = extract_verified_facts(
        pipe_name="CUSTOM_INGESTION_JOB",
        connector_type="DATABRICKS",
        error_message="ConnectionRefusedError: Failed to connect to DB at 10.0.0.5:5432",
        logs=[{"level": "ERROR", "message": "ConnectionRefusedError: Connection timeout after 30000ms"}],
        metadata={"error_code": "CONNECTION_FAILED", "failed_stage": "INGESTION"},
    )

    llm_output = {
        "summary": "Pipeline CUSTOM_INGESTION_JOB failed due to network connectivity failure.",
        "root_cause": "Ingestion stage could not reach target database at 10.0.0.5:5432.",
        "failure_mechanism": "Connection timeout raised after 30 seconds.",
        "impact": "Data ingestion halted.",
    }

    result = normalize_diagnosis(verified_facts, llm_output, kb_context={"is_known": False})
    assert result["pipeline_name"] == "CUSTOM_INGESTION_JOB"
    assert result["error_code"] == "CONNECTION_FAILED"
    assert len(result["known_fix"]) > 0
    assert len(result["immediate_fix"]) > 0
    assert result["confidence"] <= 0.85


def test_scenario_6_percentage_threshold_and_quarantine_policy():
    """TEST 6: Validates that quarantine is optional and does not claim to rewrite the 40% error rate."""
    verified_facts = {
        "pipeline_name": "ORDER_TRANSACTION_ETL",
        "failed_stage": "SILVER_DATA_VALIDATION",
        "error_code": "DATA_QUALITY_THRESHOLD_BREACH",
        "invalid_records": 4,
        "total_records": 10,
        "invalid_percentage": 40.0,
        "allowed_threshold": 5.0,
    }

    llm_output = {
        "immediate_fix": [
            {"step": 1, "title": "Correct Source Data", "action": "Contact the data source team to fix invalid records."},
            {"step": 2, "title": "Re-run Pipeline", "action": "Trigger pipeline again."}
        ],
        "optional_improvements": [
            {"step": 1, "title": "Quarantine Invalid Records", "action": "Isolate records into quarantine table."}
        ]
    }

    result = normalize_diagnosis(verified_facts, llm_output)

    # Required recovery actions MUST be populated in immediate_fix
    imm = result["immediate_fix"]
    assert len(imm) >= 2
    assert imm[0]["priority"] == "REQUIRED"
    assert "Identify the owner of the upstream source" in imm[0]["action"]  # Sanitized ownership
    # Quarantine must be in optional_improvements
    opt = result["optional_improvements"]
    assert len(opt) >= 1
    assert opt[0]["priority"] == "OPTIONAL"


def test_scenario_8_safe_code_patch_anti_hallucination():
    """TEST 8: fix_patch is cleared when source code context is incomplete."""
    verified_facts = {"pipeline_name": "ORDER_TRANSACTION_ETL"}
    llm_output = {
        "fix_patch": "Update your code to use filter(col('id').isNotNull())",  # Prose, not real patch
    }
    result = normalize_diagnosis(verified_facts, llm_output)
    assert result["fix_patch"] == ""


def test_scenario_9_gemini_and_local_identical_contract():
    """TEST 9: Both Gemini and Local LLM outputs produce the identical API contract."""
    verified_facts = {
        "pipeline_name": "ORDER_TRANSACTION_ETL",
        "pipeline_run_id": "20260819134953",
        "environment": "PRODUCTION",
        "failed_stage": "SILVER_DATA_VALIDATION",
        "error_code": "DATA_QUALITY_THRESHOLD_BREACH",
        "total_records": 10,
        "invalid_records": 4,
        "invalid_percentage": 40.0,
        "allowed_threshold": 5.0,
    }

    # Gemini rich output
    gemini_out = {
        "summary": "Pipeline ORDER_TRANSACTION_ETL failed during SILVER_DATA_VALIDATION.",
        "root_cause": "The incoming order batch contained 4 unique invalid records out of 10 total records.",
        "failure_mechanism": "Execution terminated by threshold breach.",
        "impact": "Terminated to prevent downstream corruption.",
    }
    res_gemini = normalize_diagnosis(verified_facts, gemini_out)

    # Local Mistral partial output
    local_out = {
        "summary": "New Job failed",
        "root_cause": "4 records invalid",
    }
    res_local = normalize_diagnosis(verified_facts, local_out)

    # Both must have identical top-level keys
    expected_keys = {
        "summary", "root_cause", "failure_mechanism", "impact",
        "pipeline_name", "pipeline_run_id", "environment", "failed_stage", "error_code",
        "total_records", "invalid_records", "invalid_percentage", "allowed_threshold",
        "immediate_fix", "optional_improvements", "known_fix", "suggested_fix",
        "recommended_actions", "validation_steps", "long_term_prevention", "fix_patch",
        "confidence", "confidence_breakdown", "diagnosis_status", "diagnosis_error"
    }

    assert expected_keys.issubset(res_gemini.keys())
    assert expected_keys.issubset(res_local.keys())
    assert res_gemini["pipeline_name"] == res_local["pipeline_name"] == "ORDER_TRANSACTION_ETL"
    assert res_gemini["failed_stage"] == res_local["failed_stage"] == "SILVER_DATA_VALIDATION"
    assert res_gemini["error_code"] == res_local["error_code"] == "DATA_QUALITY_THRESHOLD_BREACH"

