import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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


def test_scenario_10_databricks_patient_data_validation_runtime_error():
    """TEST 10: Dynamic metric and violation parsing for patient data validation runtime error."""
    error_msg = (
        "RuntimeError: DATA_QUALITY_THRESHOLD_BREACH: Patient data validation failed. "
        "4 of 10 records are invalid (40.0%), exceeding the allowed threshold of 5.0%. "
        "Violations: missing_patient_id=1, invalid_age=1, invalid_diagnosis_code=1, duplicate_patient_id=1."
    )
    facts = extract_verified_facts(
        pipe_name="PATIENT_DATA_QUALITY_ETL_JOB",
        connector_type="DATABRICKS",
        error_message=error_msg,
        metadata={"task_key": "PATIENT_DATA_QUALITY_ETL"},
    )

    assert facts["pipeline_name"] == "PATIENT_DATA_QUALITY_ETL_JOB"
    assert facts["error_code"] == "DATA_QUALITY_THRESHOLD_BREACH"
    assert facts["invalid_records"] == 4
    assert facts["total_records"] == 10
    assert facts["invalid_percentage"] == 40.0
    assert facts["allowed_threshold"] == 5.0
    assert facts["validation_violations_total"] == 4
    assert facts["validation_failures"].get("Missing Patient Id") == 1
    assert facts["validation_failures"].get("Invalid Age") == 1
    assert facts["validation_failures"].get("Invalid Diagnosis Code") == 1
    assert facts["validation_failures"].get("Duplicate Patient Id") == 1

    result = normalize_diagnosis(facts, {})
    assert "4 of 10 records were invalid (40.0%)" in result["summary"]
    assert "5.0%" in result["summary"]
    assert "4 unique invalid records out of 10 total records" in result["root_cause"]
    assert "40.0%" in result["root_cause"]
    assert "5.0%" in result["root_cause"]
    assert len(result["immediate_fix"]) >= 3
    assert result["confidence"] >= 0.80


def test_scenario_11_mode_b_wrapper_error_no_hallucination():
    """TEST 11: Mode B returns honest insufficient-telemetry response without template artifacts."""
    facts = extract_verified_facts(
        pipe_name="PATIENT_DATA_QUALITY_ETL_JOB",
        connector_type="DATABRICKS",
        error_message="Task PATIENT_DATA_QUALITY_ETL failed with message: Workload failed, see run output for details.",
        logs=[{"level": "ERROR", "message": "Workload failed, see run output for details."}],
        metadata={},
    )

    assert facts["invalid_records"] is None
    assert facts["total_records"] is None
    assert facts["error_code"] is None

    # Suppose an imperfect local LLM produced partial broken template output
    bad_output = {
        "summary": "Pipeline failed during execution.",
        "root_cause": "The incoming batch contained unique invalid records out of total records, producing a rate exceeding configured threshold%.",
        "failure_mechanism": "The execution stage detected an invalid-record rate of %, exceeding % and triggering error code...",
    }

    result = normalize_diagnosis(facts, bad_output)
    assert "generic wrapper error" in result["summary"]
    assert "Detailed task run output has not yet been retrieved" in result["summary"]
    assert "The root cause cannot be determined" in result["root_cause"]
    assert "Detailed task run output is required" in result["root_cause"]
    assert result["immediate_fix"][0]["title"] == "Retrieve Detailed Task Run Output"

    # Strict check: NEVER contain malformed placeholders
    for field in ("summary", "root_cause", "failure_mechanism", "impact"):
        val = result[field]
        assert "null/null" not in val
        assert "threshold%" not in val
        assert "rate of %" not in val
        assert "triggering error code" not in val


def test_scenario_12_databricks_connector_task_get_logs_mock():
    """TEST 12: DatabricksConnector.get_logs calls task-level get-output for multi-task runs."""
    from unittest.mock import MagicMock
    from app.connectors.databricks import DatabricksConnector

    connector = DatabricksConnector({
        "workspace_url": "https://adb-123456789.12.azuredatabricks.net",
        "personal_access_token": "dapi_test_token",
    })

    def mock_get(path, params=None):
        params = params or {}
        if path == "/api/2.1/jobs/runs/get":
            return {
                "run_id": 1001,
                "start_time": 1700000000000,
                "tasks": [
                    {
                        "task_key": "PATIENT_DATA_QUALITY_ETL",
                        "run_id": 2002,
                        "state": {
                            "result_state": "FAILED",
                            "life_cycle_state": "TERMINATED",
                            "state_message": "Workload failed, see run output for details.",
                        },
                        "start_time": 1700000000000,
                        "end_time": 1700000010000,
                    }
                ]
            }
        elif path == "/api/2.1/jobs/runs/get-output":
            if params.get("run_id") == 2002:
                return {
                    "error": (
                        "RuntimeError: DATA_QUALITY_THRESHOLD_BREACH: Patient data validation failed. "
                        "4 of 10 records are invalid (40.0%), exceeding the allowed threshold of 5.0%. "
                        "Violations: missing_patient_id=1, invalid_age=1, invalid_diagnosis_code=1, duplicate_patient_id=1."
                    ),
                    "error_trace": "Traceback (most recent call last):\n  File 'patient_validation.py', line 45",
                }
            return {}
        return {}

    connector._get = MagicMock(side_effect=mock_get)
    logs = connector.get_logs("500", "1001")

    assert len(logs) >= 2
    err_log = next(l for l in logs if "DATA_QUALITY_THRESHOLD_BREACH" in l.message)
    assert err_log.level == "ERROR"
    assert err_log.source == "PATIENT_DATA_QUALITY_ETL"
    assert "4 of 10 records are invalid" in err_log.message


def test_scenario_13_recovery_success_criteria_calculation_p0_1():
    """P0-1: Recovery success criteria mathematical calculations."""
    facts = extract_verified_facts(
        pipe_name="PATIENT_DEMOGRAPHICS_ETL",
        connector_type="databricks",
        error_message="RuntimeError: DATA_QUALITY_THRESHOLD_BREACH: 4 of 10 records are invalid (40.0%), exceeding threshold of 5.0%.",
        logs=[{"level": "ERROR", "message": "Violations: missing_patient_id=2, invalid_age=2"}],
        metadata={},
    )
    crit = facts["recovery_success_criteria"]
    assert crit is not None
    assert crit["total_records"] == 10
    assert crit["threshold_percentage"] == 5.0
    assert crit["allowed_invalid_count"] == 0
    assert crit["comparison_operator"] == "<="
    assert "at most 0 invalid records are allowed" in crit["message"]

    # Test larger batch (100 records, 5.0% threshold -> 5 allowed)
    facts_100 = extract_verified_facts(
        pipe_name="PATIENT_DEMOGRAPHICS_ETL",
        connector_type="databricks",
        error_message="RuntimeError: DATA_QUALITY_THRESHOLD_BREACH: 15 of 100 records are invalid (15.0%), exceeding threshold of 5.0%.",
        logs=[],
        metadata={},
    )
    crit_100 = facts_100["recovery_success_criteria"]
    assert crit_100["allowed_invalid_count"] == 5

    # Test missing telemetry -> clear graceful message
    facts_missing = extract_verified_facts(
        pipe_name="PATIENT_DEMOGRAPHICS_ETL",
        connector_type="databricks",
        error_message="Workload failed, see run output for details.",
        logs=[],
        metadata={},
    )
    crit_missing = facts_missing["recovery_success_criteria"]
    assert crit_missing["allowed_invalid_count"] is None
    assert "cannot be calculated" in crit_missing["message"]


def test_scenario_14_verified_root_cause_vs_inferred_contributing_cause_p0_2():
    """P0-2: Strict separation of verified root cause and inferred contributing cause."""
    facts = extract_verified_facts(
        pipe_name="EHR_VALIDATION_PIPELINE",
        connector_type="databricks",
        error_message=(
            "RuntimeError: DATA_QUALITY_THRESHOLD_BREACH: 4 of 10 unique records failed validation (40.0%), "
            "exceeding threshold of 5.0%. Violations: missing_patient_id=1, invalid_diagnosis_code=1, invalid_age=1, duplicate_patient_id=1."
        ),
        logs=[],
        metadata={},
    )
    res = normalize_diagnosis(facts, {})
    assert "verified_root_cause" in res
    assert "inferred_contributing_cause" in res
    assert "recovery_success_criteria" in res

    v_rc = res["verified_root_cause"]
    assert "4 of 10 unique records failed validation" in v_rc
    assert "40.0%" in v_rc
    assert "5.0%" in v_rc
    assert "missing_patient_id" in v_rc.lower() or "missing patient id" in v_rc.lower()

    i_cc = res["inferred_contributing_cause"]
    assert "upstream validation at the source boundary" in i_cc
    assert "did not intercept" in i_cc


def test_scenario_15_medical_failure_diagnoses_regression():
    """Test regression on diverse medical Databricks failed pipelines."""
    # Medical Pipeline 1: Patient Vital Signs ETL
    facts_vitals = extract_verified_facts(
        pipe_name="PATIENT_VITALS_ETL",
        connector_type="databricks",
        error_message="RuntimeError: DATA_QUALITY_THRESHOLD_BREACH: 3 of 20 records are invalid (15.0%), exceeding threshold of 2.0%.",
        logs=[{"level": "ERROR", "message": "Violations: invalid_blood_pressure=2, missing_patient_id=1."}],
        metadata={},
    )
    res_vitals = normalize_diagnosis(facts_vitals, {})
    assert res_vitals["verified_facts"]["invalid_records"] == 3
    assert res_vitals["verified_facts"]["total_records"] == 20
    assert res_vitals["recovery_success_criteria"]["allowed_invalid_count"] == 0  # floor(20 * 0.02) = 0
    assert len(res_vitals["immediate_fix"]) >= 3
    assert res_vitals["immediate_fix"][0]["priority"] == "REQUIRED"

    # Medical Pipeline 2: Clinical Dosage & Prescription Pipeline
    facts_dosage = extract_verified_facts(
        pipe_name="CLINICAL_PRESCRIPTION_ETL",
        connector_type="databricks",
        error_message="RuntimeError: DATA_QUALITY_THRESHOLD_BREACH: 6 of 50 records are invalid (12.0%), exceeding threshold of 4.0%.",
        logs=[{"level": "ERROR", "message": "Violations: invalid_medication_dosage=4, duplicate_medical_record=2."}],
        metadata={},
    )
    res_dosage = normalize_diagnosis(facts_dosage, {})
    assert res_dosage["verified_facts"]["invalid_records"] == 6
    assert res_dosage["verified_facts"]["total_records"] == 50
    assert res_dosage["recovery_success_criteria"]["allowed_invalid_count"] == 2  # floor(50 * 0.04) = 2


if __name__ == "__main__":
    tests = [
        test_scenario_1_local_llm_hallucinates_facts,
        test_scenario_2_known_fix_format_normalization,
        test_scenario_3_known_pattern_zero_accepted_fixes,
        test_scenario_4_known_pattern_with_accepted_fix,
        test_scenario_5_novel_error_first_principles,
        test_scenario_6_percentage_threshold_and_quarantine_policy,
        test_scenario_8_safe_code_patch_anti_hallucination,
        test_scenario_9_gemini_and_local_identical_contract,
        test_scenario_10_databricks_patient_data_validation_runtime_error,
        test_scenario_11_mode_b_wrapper_error_no_hallucination,
        test_scenario_12_databricks_connector_task_get_logs_mock,
        test_scenario_13_recovery_success_criteria_calculation_p0_1,
        test_scenario_14_verified_root_cause_vs_inferred_contributing_cause_p0_2,
        test_scenario_15_medical_failure_diagnoses_regression,
    ]

    passed = 0
    failed = 0
    print("=" * 60)
    print("RUNNING AGENTIC OPS DIAGNOSIS NORMALIZER & DATABRICKS TEST SUITE")
    print("=" * 60)
    for t in tests:
        try:
            t()
            print(f"  [PASS] {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    print(f"RESULTS: {passed} PASSED, {failed} FAILED (TOTAL {len(tests)})")
    print("=" * 60)
    if failed > 0:
        raise SystemExit(1)



