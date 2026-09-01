import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from app.services.diagnosis_normalizer import (
    extract_verified_facts,
    normalize_diagnosis,
    normalize_known_fix,
    compute_blast_radius,
    build_root_cause_classification,
)
from app.services import confidence_explainer


def test_pharmacy_inventory_reconciliation_generic_accuracy():
    """Test Case 1: Job 12 Pharmacy Inventory Reconciliation with multiple failure categories."""
    logs = [
        {"level": "INFO", "message": "Starting batch reconciliation for PHARMACY_INVENTORY_RECONCILIATION_ETL_JOB run 91023"},
        {"level": "ERROR", "message": "RuntimeError: DATA_QUALITY_THRESHOLD_BREACH: 9 of 12 unique records failed validation (75.0%), exceeding threshold of 25.0%."},
        {"level": "ERROR", "message": "Violations: Inventory Reconciliation Mismatch=5, Critical Stock Shortage=4, Reserved Stock Exceeds Physical=2, Duplicate Inventory Record=2."},
        {"level": "ERROR", "message": "Affected records: ['INV001', 'INV002', 'INV003', 'INV004', 'INV005', 'INV006', 'INV007', 'INV008', 'INV009', 'INV002', 'INV005']"},
    ]
    facts = extract_verified_facts(
        pipe_name="PHARMACY_INVENTORY_RECONCILIATION_ETL_JOB",
        connector_type="databricks",
        error_message="RuntimeError: DATA_QUALITY_THRESHOLD_BREACH: 9 of 12 unique records failed validation (75.0%), exceeding threshold of 25.0%.",
        logs=logs,
        metadata={"pipeline_name": "PHARMACY_INVENTORY_RECONCILIATION_ETL_JOB", "environment": "PRODUCTION"},
        run_id="91023",
    )

    # 1. Fact integrity & category violations explanation
    assert facts["total_records"] == 12
    assert facts["invalid_records"] == 9
    assert facts["invalid_percentage"] == 75.0
    assert facts["allowed_threshold"] == 25.0
    assert facts["validation_violations_total"] == 13
    assert facts["category_violation_explanation"] is not None
    assert "single record may violate multiple validation rules" in facts["category_violation_explanation"].lower()

    # 2. Raw vs Unique affected IDs
    assert len(facts["affected_ids_raw"]) == 11
    assert len(facts["affected_ids_unique"]) == 9
    assert "INV002" in facts["affected_ids_duplicates"]

    # 3. Recovery target math
    crit = facts["recovery_success_criteria"]
    assert crit["allowed_invalid_count"] == 3  # floor(12 * 25% = 3.0)
    assert crit["records_to_resolve"] == 6    # 9 - 3 = 6
    assert "at most 3 invalid records are allowed" in crit["message"]

    # 4. Normalization & Evidence Tiers
    res = normalize_diagnosis(facts, {})
    rc_class = res["root_cause_classification"]
    assert rc_class is not None
    assert rc_class["tier_a_verified_fact"] is not None
    assert "9 of 12 unique records" in rc_class["tier_a_verified_fact"]["statement"]
    assert rc_class["tier_b_deterministic_inference"] is not None
    assert "75.0%" in rc_class["tier_b_deterministic_inference"]["calculation"]
    assert rc_class["tier_c_hypothesis"] is not None
    assert len(rc_class["tier_c_hypothesis"]) > 0
    assert "may indicate" in rc_class["tier_c_hypothesis"][0]["statement"].lower()

    # 5. Fix items & Next Best Action
    imm_fixes = res["immediate_fix"]
    assert len(imm_fixes) >= 3
    assert imm_fixes[0]["step"] == 1
    assert "P0" in imm_fixes[0]["title"]
    assert "Verified" in imm_fixes[0]["evidence_source"]
    assert imm_fixes[1]["what_to_inspect"] is not None
    assert imm_fixes[1]["what_to_fix"] is not None
    assert imm_fixes[1]["evidence"] is not None
    assert imm_fixes[1]["why"] is not None

    # Next Best Action (Part 10)
    assert res.get("next_best_action") is not None
    assert "P1" in res["next_best_action"]["title"]
    assert res["next_best_action"]["target_step"] == 2

    # 6. Structured Impact
    assert res["impact_data"] is not None
    assert res["impact_data"]["records_affected"] == 9
    assert not str(res["impact"]).startswith("{")

    # 7. Confidence Explanation checklist
    conf_exp = confidence_explainer.build(
        llm_confidence=0.90,
        final_confidence=res["confidence"],
        pattern=None,
        is_known=False,
        error_type="Data Quality",
        facts=facts,
    )
    conf_dict = conf_exp.to_dict()
    assert len(conf_dict["evidence_available"]) >= 4
    assert len(conf_dict["evidence_missing"]) >= 1
    print("[PASS] TEST 1 PASSED: PHARMACY_INVENTORY_RECONCILIATION")


def test_financial_transaction_settlement_pipeline():
    """Test Case 2: Financial transaction batch failure with 0% tolerance."""
    logs = [
        {"level": "ERROR", "message": "RuntimeError: DATA_QUALITY_THRESHOLD_BREACH: 2 of 500 records are invalid (0.4%), exceeding threshold of 0.0%."},
        {"level": "ERROR", "message": "Violations: negative_settlement_amount=1, unverified_beneficiary_bic=1"},
        {"level": "ERROR", "message": "Affected records: ['TXN-9901', 'TXN-9988']"},
    ]
    facts = extract_verified_facts(
        pipe_name="FINANCIAL_SETTLEMENT_BATCH_JOB",
        connector_type="databricks",
        error_message="RuntimeError: DATA_QUALITY_THRESHOLD_BREACH: 2 of 500 records are invalid (0.4%), exceeding threshold of 0.0%.",
        logs=logs,
        metadata={"pipeline_name": "FINANCIAL_SETTLEMENT_BATCH_JOB"},
        run_id="4401",
    )
    assert facts["total_records"] == 500
    assert facts["invalid_records"] == 2
    assert facts["allowed_threshold"] == 0.0
    crit = facts["recovery_success_criteria"]
    assert crit["allowed_invalid_count"] == 0
    assert crit["records_to_resolve"] == 2

    res = normalize_diagnosis(facts, {})
    assert "2 of 500 unique records failed validation" in res["verified_root_cause"]
    print("[PASS] TEST 2 PASSED: FINANCIAL_SETTLEMENT_BATCH_JOB")


def test_delta_concurrency_error_pipeline():
    """Test Case 3: Delta Lake concurrent modification conflict."""
    logs = [
        {"level": "ERROR", "message": "ConcurrentAppendException: Files were added to partition [date=2026-08-31] by concurrent update."},
    ]
    facts = extract_verified_facts(
        pipe_name="STREAMING_EVENT_INGESTION",
        connector_type="databricks",
        error_message="ConcurrentAppendException: DELTA_CONCURRENT_MODIFICATION: Files were added by concurrent update.",
        logs=logs,
        metadata={"pipeline_name": "STREAMING_EVENT_INGESTION"},
        run_id="5502",
    )
    assert facts["error_code"] == "DELTA_CONCURRENT_MODIFICATION"
    res = normalize_diagnosis(facts, {})
    assert "DELTA_CONCURRENT_MODIFICATION" in res["summary"]
    assert len(res["immediate_fix"]) >= 1
    print("[PASS] TEST 3 PASSED: DELTA_CONCURRENT_MODIFICATION")


def test_mode_b_missing_telemetry_honest_output():
    """Test Case 4: Mode B missing telemetry wrapper error without hallucinations."""
    facts = extract_verified_facts(
        pipe_name="GENERIC_BATCH_JOB",
        connector_type="databricks",
        error_message="Workload failed, see run output for details.",
        logs=[],
        metadata={"pipeline_name": "GENERIC_BATCH_JOB"},
        run_id="7701",
    )
    res = normalize_diagnosis(facts, {}, is_telemetry_missing=True)
    assert "cannot be determined" in res["root_cause"]
    assert "Retrieve Detailed Task Run Output" in res["immediate_fix"][0]["title"]
    assert res["confidence"] <= 0.45
    print("[PASS] TEST 4 PASSED: MODE_B_MISSING_TELEMETRY")


if __name__ == "__main__":
    test_pharmacy_inventory_reconciliation_generic_accuracy()
    test_financial_transaction_settlement_pipeline()
    test_delta_concurrency_error_pipeline()
    test_mode_b_missing_telemetry_honest_output()
    print("\nALL GENERIC PIPELINE VERIFICATION TESTS PASSED PERFECTLY!")
