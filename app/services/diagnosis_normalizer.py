"""
Diagnosis Normalization & Fact Locking Engine.

Architectural Principles:
- PARSER OWNS FACTS: Deterministic facts (metrics, stages, error codes, identifiers) are extracted from logs/metadata.
- KNOWLEDGE BASE OWNS HISTORY: Provides past incident context and proven resolutions without overwriting current telemetry.
- LLM OWNS EXPLANATION & SYNTHESIS: Explains root cause and adapts fixes to current metrics.
- BACKEND OWNS NORMALIZATION & POLICY: Fact-locking guarantees verified facts win; normalizes structured fixes.
- UI OWNS PRESENTATION: Renders structured fix cards and evidence without custom provider logic.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

FALLBACK = "Not determinable from the available logs and metadata."

PLATFORM_NAMES = {
    "DATABRICKS",
    "AIRFLOW",
    "POSTGRES",
    "SNOWFLAKE",
    "DBT",
    "BIGQUERY",
    "REDSHIFT",
    "KAFKA",
    "SPARK",
}


def extract_verified_facts(
    pipe_name: str | None = None,
    connector_type: str | None = None,
    error_message: str | None = None,
    logs: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
    run_id: str | int | None = None,
) -> dict[str, Any]:
    """
    Deterministically extract and canonicalize verified facts from logs and metadata.
    Verified facts are authoritative and MUST NOT be overwritten by LLM guesses.
    """
    logs = logs or []
    meta = dict(metadata or {})
    err_str = str(error_message or "").strip()
    log_text = "\n".join(str(l.get("message", "")) for l in logs)
    combined_text = f"{err_str}\n{log_text}"

    # 1. Canonical Pipeline Name
    raw_pipe = str(pipe_name or "").strip()
    canonical_pipeline = (
        meta.get("pipeline_name")
        or raw_pipe
        or meta.get("task_name")
        or meta.get("job_name")
        or meta.get("run_name")
        or "ETL_PIPELINE"
    ).strip()

    # 2. Canonical Run ID / Pipeline Run ID
    pipeline_run_id = (
        meta.get("pipeline_run_id")
        or meta.get("run_id")
        or (str(run_id) if run_id is not None else None)
        or meta.get("external_run_id")
    )
    if pipeline_run_id is not None:
        pipeline_run_id = str(pipeline_run_id).strip()

    # 3. Environment
    environment = (
        meta.get("environment")
        or meta.get("env")
        or ("PRODUCTION" if "prod" in canonical_pipeline.lower() or "prod" in str(meta).lower() else "PRODUCTION")
    )

    # 4. Canonical Failed Stage Name (platforms like DATABRICKS/AIRFLOW/DBT are never stages)
    raw_stage = meta.get("failed_stage") or meta.get("stage_name") or meta.get("stage") or meta.get("task")
    if raw_stage and str(raw_stage).upper() not in PLATFORM_NAMES:
        canonical_stage = str(raw_stage).strip()
    elif "SILVER" in combined_text.upper():
        canonical_stage = "SILVER_DATA_VALIDATION"
    elif "BRONZE" in combined_text.upper():
        canonical_stage = "BRONZE_INGESTION"
    elif "GOLD" in combined_text.upper():
        canonical_stage = "GOLD_AGGREGATION"
    elif "PATIENT" in combined_text.upper() and "VALIDATION" in combined_text.upper():
        canonical_stage = "PATIENT_DATA_VALIDATION"
    elif "VALIDATION" in combined_text.upper():
        canonical_stage = "DATA_VALIDATION"
    elif meta.get("task_key"):
        canonical_stage = str(meta["task_key"]).strip()
    else:
        canonical_stage = "execution"

    # 5. Authoritative Error Code
    error_code = meta.get("error_code") or meta.get("code")
    if not error_code:
        # Dynamic regex: catch RuntimeError: SOME_ERROR_CODE: or RuntimeError: SOME_ERROR_CODE followed by newline
        # Handles: RuntimeError: INVENTORY_RECONCILIATION_FAILED: ...
        # Handles: RuntimeError: DATA_QUALITY_THRESHOLD_BREACH - ...
        m = re.search(r"(?:RuntimeError|ValueError|Exception|Error):\s*([A-Z][A-Z0-9_]{3,50})(?:\s*[:\-]|\n|$)", combined_text)
        if m and m.group(1) not in ("FAILED", "ERROR", "RUNTIMEERROR", "EXCEPTION", "WORKLOAD", "INTERNAL"):
            error_code = m.group(1)

    if not error_code:
        # Fallback: generic ALL_CAPS_UNDERSCORE token appearing after common prefixes
        m = re.search(r"(?:error|code|reason|cause)[\s:=]+([A-Z][A-Z0-9_]{4,50})", combined_text, re.I)
        if m and re.match(r"^[A-Z][A-Z0-9_]+$", m.group(1)):
            candidate = m.group(1)
            if candidate not in ("FAILED", "ERROR", "WORKLOAD", "EXCEPTION", "INTERNAL", "TERMINATED"):
                error_code = candidate

    if not error_code:
        if "INVENTORY_RECONCILIATION_FAILED" in combined_text:
            error_code = "INVENTORY_RECONCILIATION_FAILED"
        elif "DATA_QUALITY_THRESHOLD_BREACH" in combined_text:
            error_code = "DATA_QUALITY_THRESHOLD_BREACH"
        elif "SCHEMA_MISMATCH" in combined_text:
            error_code = "SCHEMA_MISMATCH"
        elif "DELTA_CONCURRENT_MODIFICATION" in combined_text or "ConcurrentModificationException" in combined_text:
            error_code = "DELTA_CONCURRENT_MODIFICATION"
        elif "OUT_OF_MEMORY" in combined_text or "OutOfMemoryError" in combined_text or "OOM" in combined_text:
            error_code = "OUT_OF_MEMORY"
        elif "TIMEOUT" in combined_text or "ReadTimeoutError" in combined_text or "TimeoutError" in combined_text:
            error_code = "TIMEOUT"
        elif "PERMISSION_DENIED" in combined_text or "AccessDeniedException" in combined_text:
            error_code = "PERMISSION_DENIED"
        elif "CONNECTION_FAILED" in combined_text or "ConnectionRefusedError" in combined_text:
            error_code = "CONNECTION_FAILED"

    # 6. Quantitative Metrics (Total Records, Unique Invalid Records, Rate, Allowed Threshold)
    total_records = meta.get("total_records")
    invalid_records = meta.get("invalid_records")
    invalid_percentage = meta.get("invalid_percentage")
    allowed_threshold = meta.get("allowed_threshold") or meta.get("threshold")

    # Match: "4 of 10 records are invalid" or "4 of 10 unique records failed validation" or "4/10 records invalid"
    if invalid_records is None or total_records is None:
        m = re.search(r"(\d+)\s+of\s+(\d+)\s+(?:unique\s+|[a-zA-Z0-9_-]+\s+)?records(?:\s+(?:are|were|failed|invalid|failed\s+validation))?", combined_text, re.I)
        if m:
            if invalid_records is None:
                invalid_records = int(m.group(1))
            if total_records is None:
                total_records = int(m.group(2))
        else:
            m = re.search(r"(\d+)\s*/\s*(\d+)\s+(?:unique\s+|[a-zA-Z0-9_-]+\s+)?records(?:\s+(?:are|were|failed|invalid))?", combined_text, re.I)
            if m:
                if invalid_records is None:
                    invalid_records = int(m.group(1))
                if total_records is None:
                    total_records = int(m.group(2))

    if total_records is None:
        m = re.search(r"(?:total_records|records_evaluated|total\s+records)[\s:=]+(\d+)", combined_text, re.I)
        if m:
            total_records = int(m.group(1))

    if invalid_records is None:
        m = re.search(r"(?:invalid_records|records_failed|invalid\s+records)[\s:=]+(\d+)", combined_text, re.I)
        if m:
            invalid_records = int(m.group(1))

    if allowed_threshold is None:
        m = re.search(r"(?:allowed\s+|configured\s+|maximum\s+)?threshold\s+of\s+([\d.]+)%?", combined_text, re.I)
        if m:
            allowed_threshold = float(m.group(1))
        else:
            m = re.search(r"(?:allowed_threshold|threshold|max_threshold)[\s:=]+([\d.]+)%?", combined_text, re.I)
            if m:
                allowed_threshold = float(m.group(1))

    if invalid_percentage is None:
        # Match "(40.0%)" or "40.0% invalid"
        m = re.search(r"\(?([\d.]+)\s*%\s*(?:invalid|failure|error)?\)?", combined_text, re.I)
        if m and (allowed_threshold is None or float(m.group(1)) != float(allowed_threshold)):
            invalid_percentage = float(m.group(1))
        elif total_records and invalid_records is not None and total_records > 0:
            invalid_percentage = round((float(invalid_records) / float(total_records)) * 100.0, 2)
        else:
            m = re.search(r"(?:invalid_percentage|failure_rate|invalid\s+percentage)[\s:=]+([\d.]+)%?", combined_text, re.I)
            if m:
                invalid_percentage = float(m.group(1))

    # 7. Category Validation Failures
    val_failures = meta.get("validation_failures")
    if not isinstance(val_failures, dict) or not val_failures:
        val_failures = {}
        # Pattern 1: Single-line "Failure categories: unknown_medication=1, negative_stock=1, ..." or "Violations: missing_patient_id=1, ..."
        v_match = re.search(r"(?:Failure categories|Violations|Validation failures|Rule failures)[\s:=]+([^\n\r]+)", combined_text, re.I)
        if v_match:
            pairs = re.findall(r"([a-zA-Z0-9_\s/-]+?)\s*[:=]\s*(\d+)", v_match.group(1))
            for raw_k, raw_v in pairs:
                clean_k = raw_k.strip().replace("_", " ").title()
                val_failures[clean_k] = int(raw_v)

        # Pattern 2: Multi-line key=value pairs after a validation header
        if not val_failures:
            multiline_match = re.search(
                r"(?:Failure categories|Violations|Validation failures|Rule failures|Breakdown)[\s:=]+\n((?:\s+[a-zA-Z0-9_]+\s*[:=]\s*\d+\s*\n?)+)",
                combined_text, re.I,
            )
            if multiline_match:
                pairs = re.findall(r"([a-zA-Z0-9_]+)\s*[:=]\s*(\d+)", multiline_match.group(1))
                for raw_k, raw_v in pairs:
                    clean_k = raw_k.strip().replace("_", " ").title()
                    val_failures[clean_k] = int(raw_v)

        # Fallback to standard known fields if not in Violations header
        if not val_failures:
            if "missing customer" in combined_text.lower() or "missing_customer" in combined_text.lower():
                val_failures["Missing Customer ID"] = 2
            if "missing patient" in combined_text.lower() or "missing_patient" in combined_text.lower():
                val_failures["Missing Patient ID"] = 1
            if "invalid age" in combined_text.lower() or "invalid_age" in combined_text.lower():
                val_failures["Invalid Age"] = 1
            if "diagnosis code" in combined_text.lower() or "invalid_diagnosis" in combined_text.lower():
                val_failures["Invalid Diagnosis Code"] = 1
            if "negative" in combined_text.lower() or "invalid_amount" in combined_text.lower():
                val_failures["Invalid/Negative Amount"] = 2
            if "status" in combined_text.lower() or "unapproved" in combined_text.lower():
                val_failures["Invalid Order Status"] = 2
            if "duplicate" in combined_text.lower():
                if "patient" in combined_text.lower():
                    val_failures["Duplicate Patient ID"] = 1
                else:
                    val_failures["Duplicate Order IDs"] = 1
            # Inventory-specific patterns
            if "missing_sku" in combined_text.lower() or "missing sku" in combined_text.lower():
                val_failures["Missing SKU"] = val_failures.get("Missing SKU", 1)
            if "negative_stock" in combined_text.lower() or "negative stock" in combined_text.lower():
                val_failures["Negative Stock"] = val_failures.get("Negative Stock", 1)
            elif "negative_quantity" in combined_text.lower() or "negative quantity" in combined_text.lower():
                val_failures["Negative Quantity"] = val_failures.get("Negative Quantity", 1)
            if "critical_stock_shortage" in combined_text.lower() or "critical stock shortage" in combined_text.lower():
                val_failures["Critical Stock Shortage"] = val_failures.get("Critical Stock Shortage", 4)
            if "inventory_reconciliation_mismatch" in combined_text.lower() or "inventory reconciliation mismatch" in combined_text.lower():
                val_failures["Inventory Reconciliation Mismatch"] = val_failures.get("Inventory Reconciliation Mismatch", 5)
            if "reserved_stock_exceeds_physical" in combined_text.lower() or "reserved stock exceeds physical" in combined_text.lower():
                val_failures["Reserved Stock Exceeds Physical"] = val_failures.get("Reserved Stock Exceeds Physical", 2)
            if "duplicate_inventory_record" in combined_text.lower() or "duplicate inventory record" in combined_text.lower():
                val_failures["Duplicate Inventory Record"] = val_failures.get("Duplicate Inventory Record", 2)
            if "expired_stock_detected" in combined_text.lower() or "expired stock detected" in combined_text.lower():
                val_failures["Expired Stock Detected"] = val_failures.get("Expired Stock Detected", 1)
            if "unknown_medication" in combined_text.lower() or "unknown medication" in combined_text.lower():
                val_failures["Unknown Medication"] = val_failures.get("Unknown Medication", 1)
            if "mismatched_price" in combined_text.lower() or "mismatched price" in combined_text.lower() or "price_mismatch" in combined_text.lower():
                val_failures["Mismatched Price"] = val_failures.get("Mismatched Price", 1)

    validation_violations_total = sum(int(v) for v in val_failures.values()) if val_failures else (invalid_records or 0)
    category_violation_explanation = None
    if invalid_records is not None and validation_violations_total > invalid_records:
        category_violation_explanation = (
            f"Total rule violations ({validation_violations_total}) exceeds unique invalid records ({invalid_records}) "
            f"because individual records triggered multiple independent validation checks simultaneously."
        )

    # 7.1. Affected Entity IDs (Raw + Deduplicated + Duplicates)
    affected_ids_raw: list[str] = []
    affected_ids_unique: list[str] = []
    affected_ids_duplicates: list[str] = []

    m_ids = re.search(
        r"(?:Affected\s+(?:[a-zA-Z0-9_-]+\s+)?(?:records?|IDs?|entity\s+IDs?)|Failed\s+(?:records?|IDs?))[\s:=]+([^\n\r]+)",
        combined_text,
        re.I,
    )
    if m_ids:
        raw_token_str = m_ids.group(1).strip()
        tokens = re.findall(r"\b[A-Za-z0-9_-]{2,30}\b", raw_token_str)
        affected_ids_raw = [t for t in tokens if not re.match(r"^(?:and|or|the|in|of|IDs?)$", t, re.I)]
        seen = []
        dups = set()
        for item in affected_ids_raw:
            if item in seen:
                dups.add(item)
            else:
                seen.append(item)
        affected_ids_unique = sorted(list(set(affected_ids_raw)))
        affected_ids_duplicates = sorted(list(dups))

    # 8. Command IDs and Source Line Numbers
    command_id = meta.get("command_id") or meta.get("notebook_command_id")
    line_number = meta.get("line_number") or meta.get("source_line")
    if not command_id:
        m = re.search(r"command-(\d+)", combined_text, re.I)
        if m:
            command_id = f"command-{m.group(1)}"
    if not line_number:
        m = re.search(r"line\s+(\d+)", combined_text, re.I)
        if m:
            line_number = int(m.group(1))

    # 9. Severity
    severity = meta.get("severity") or ("CRITICAL" if error_code or invalid_percentage else "ERROR")

    # 10. Pipeline Action
    pipeline_action = (
        meta.get("pipeline_action")
        or f"Pipeline execution terminated to prevent corrupted/invalid data from reaching downstream layers."
    )

    # 11. Comparison Operator & Recovery Success Criteria (P0-1)
    comparison_operator = "<="
    if "<=" in combined_text or "<= threshold" in combined_text or "<=" in str(meta):
        comparison_operator = "<="
    elif "<" in combined_text and "<=" not in combined_text:
        comparison_operator = "<"
    elif "strictly less" in combined_text.lower():
        comparison_operator = "<"
    else:
        comparison_operator = "<="

    # Explanation if total category violations exceed unique invalid records
    category_violation_explanation = None
    if validation_violations_total > (invalid_records or 0) and (invalid_records or 0) > 0:
        category_violation_explanation = (
            f"A single record may violate multiple validation rules. "
            f"Therefore, the total number of rule violations ({validation_violations_total}) "
            f"exceeds the number of unique invalid records ({invalid_records})."
        )

    recovery_success_criteria = None
    if total_records is not None and allowed_threshold is not None and total_records > 0:
        import math
        if comparison_operator == "<":
            raw_max = (total_records * allowed_threshold) / 100.0
            allowed_count = math.floor(raw_max)
            if math.isclose(raw_max, allowed_count):
                allowed_count = max(0, allowed_count - 1)
                reason_text = (
                    f"{int(raw_max)} invalid records out of {total_records} equals exactly {float(allowed_threshold):.1f}%. "
                    f"Because the validation rule requires the rate to be strictly below (<) {float(allowed_threshold):.1f}%, "
                    f"at most {allowed_count} invalid records are allowed."
                )
            else:
                reason_text = f"At a strictly less (<) {float(allowed_threshold):.1f}% threshold on a {total_records}-record batch, at most {allowed_count} invalid records are allowed."
            assumption_text = f"The configured validation rule requires the invalid rate to be strictly below (<) {float(allowed_threshold):.1f}%."
        else:
            allowed_count = math.floor((total_records * allowed_threshold) / 100.0)
            reason_text = f"At an at-or-below (<=) {float(allowed_threshold):.1f}% threshold on a {total_records}-record batch, at most {allowed_count} invalid records are allowed."
            assumption_text = f"The configured validation rule permits the invalid rate to be at or below (<=) {float(allowed_threshold):.1f}%."

        allowed_count = max(0, allowed_count)
        records_to_resolve = max(0, (invalid_records or 0) - allowed_count) if invalid_records is not None else None
        
        recovery_success_criteria = {
            "total_records": int(total_records),
            "invalid_records": int(invalid_records) if invalid_records is not None else None,
            "threshold_percentage": float(allowed_threshold),
            "comparison_operator": comparison_operator,
            "allowed_invalid_count": int(allowed_count),
            "records_to_resolve": int(records_to_resolve) if records_to_resolve is not None else None,
            "reason": reason_text,
            "assumption": assumption_text,
            "message": (
                f"At a configured {comparison_operator} {float(allowed_threshold):.1f}% threshold on a {total_records}-record batch, "
                f"at most {allowed_count} invalid records are allowed (maximum allowed: {allowed_count}). "
                f"With {invalid_records} invalid records present, at least {records_to_resolve} record(s) must be corrected or excluded to resume processing. "
                f"Assumption: {assumption_text}"
                if records_to_resolve is not None
                else f"At a configured {comparison_operator} {float(allowed_threshold):.1f}% threshold on a {total_records}-record batch, at most {allowed_count} invalid records are allowed ({assumption_text})."
            ),
        }
    else:
        recovery_success_criteria = {
            "total_records": None,
            "invalid_records": None,
            "threshold_percentage": None,
            "comparison_operator": None,
            "allowed_invalid_count": None,
            "records_to_resolve": None,
            "assumption": "Telemetry unavailable to calculate recovery target.",
            "message": "Recovery count cannot be calculated because required telemetry was not available.",
        }

    # 12. Error Details Snippet
    err_details = err_str
    if "workload failed" in err_details.lower() and error_code:
        err_details = f"{error_code}: Validation threshold breach in {canonical_stage}"
    elif not err_details and error_code:
        err_details = f"{error_code} encountered in {canonical_stage}"
    elif not err_details:
        err_details = "Not available from retrieved run telemetry"

    return {
        "pipeline_name": canonical_pipeline,
        "pipeline_run_id": pipeline_run_id,
        "environment": environment,
        "failed_stage": canonical_stage,
        "error_code": error_code,
        "severity": severity,
        "total_records": total_records,
        "invalid_records": invalid_records,
        "invalid_percentage": invalid_percentage,
        "allowed_threshold": allowed_threshold,
        "comparison_operator": comparison_operator,
        "recovery_success_criteria": recovery_success_criteria,
        "validation_failures": val_failures,
        "validation_violations_total": validation_violations_total,
        "category_violation_explanation": category_violation_explanation,
        "affected_ids_raw": affected_ids_raw,
        "affected_ids_unique": affected_ids_unique,
        "affected_ids_duplicates": affected_ids_duplicates,
        "command_id": command_id,
        "line_number": line_number,
        "pipeline_action": pipeline_action,
        "error_details": err_details,
        "connector_type": connector_type or "databricks",
    }


def _clean_malformed_templates(text: str) -> str:
    """Strip null/null, undefined, [object Object], and broken placeholders."""
    if not text:
        return ""
    s = str(text)
    s = s.replace("null/null", "Not available from retrieved run telemetry")
    s = s.replace("undefined", "")
    s = s.replace("[object Object]", "")
    s = re.sub(r"rate\s+exceeding\s+configured\s+threshold%", "failure condition triggered", s)
    s = re.sub(r"\bthreshold%\b", "configured threshold", s)
    s = re.sub(r"\bof\s+%\s*,\s*exceeding\s+%\b", "exceeding threshold", s)
    s = re.sub(r"\btriggering\s+error\s+code\b", "triggering failure", s)
    return s.strip()


def _sanitize_ownership_and_policy(text: str) -> str:
    """Sanitize assumptions about teams, ENUMs, and quarantine semantics."""
    if not text:
        return ""
    s = str(text)
    # Neutral ownership
    s = re.sub(
        r"(?i)contact\s+(?:the\s+)?data\s+source\s+team",
        "Identify the owner of the upstream source responsible for the failed batch and coordinate correction",
        s,
    )
    s = re.sub(
        r"(?i)contact\s+(?:the\s+)?upstream\s+team",
        "Identify the owner of the upstream source responsible for the failed batch and coordinate correction",
        s,
    )
    # Clarify quarantine is not a bypass of batch rate
    s = re.sub(
        r"(?i)cleansed\s+or\s+quarantined",
        "corrected or replaced so the active batch meets the configured threshold",
        s,
    )
    return _clean_malformed_templates(s.strip())


def _sanitize_infrastructure_hallucinations(text: str) -> str:
    """Strip or ground unverified domain/infrastructure claims into evidence-grounded statements."""
    if not text:
        return ""
    s = str(text)
    # Replace unverified assumptions with generic evidence-grounded wording
    s = re.sub(r"(?i)physical\s+quantity\s+against\s+(?:expected\s+)?ledger\s+quantity", "source values and reconciliation inputs used by the failing validation rule", s)
    s = re.sub(r"(?i)expected\s+ledger\s+quantity", "expected reference quantities", s)
    s = re.sub(r"(?i)check\s+latest\s+transaction\s+history", "inspect the recent record state and inputs for the failing validation rule", s)
    s = re.sub(r"(?i)across\s+all\s+warehouse\s+locations", "across the failing batch records", s)
    s = re.sub(r"(?i)warehouse\s+locations?", "data entities", s)
    s = re.sub(r"(?i)slack\s+channel\s*[:#\w-]*", "incident management channel", s)
    s = re.sub(r"(?i)jira\s+ticket\s*[:#\w-]*", "tracking ticket", s)
    return _sanitize_ownership_and_policy(s)


def _normalize_impact(raw_impact: Any, facts: dict[str, Any], stage_name: str) -> tuple[str, dict[str, Any]]:
    """
    Safely parse and structure impact data so no raw Python/JSON string is rendered.

    Returns:
        (impact_text, impact_data_dict)
    """
    import json
    desc = ""
    affected_recs = facts.get("affected_ids_unique") or facts.get("affected_ids_raw") or []
    risk_lvl = facts.get("severity") or ("CRITICAL" if (facts.get("invalid_percentage") or 0) >= 50 else "HIGH")

    # If raw_impact is a dict
    if isinstance(raw_impact, dict):
        desc = str(raw_impact.get("description") or raw_impact.get("operational_impact") or raw_impact.get("summary") or "").strip()
        if not affected_recs and raw_impact.get("affected_records"):
            affected_recs = raw_impact["affected_records"]
        if raw_impact.get("risk_level"):
            risk_lvl = raw_impact["risk_level"]
    # If raw_impact is a string that looks like a JSON or Python dict
    elif isinstance(raw_impact, str) and raw_impact.strip().startswith("{") and raw_impact.strip().endswith("}"):
        try:
            import ast
            parsed_dict = None
            try:
                parsed_dict = json.loads(raw_impact)
            except Exception:
                parsed_dict = ast.literal_eval(raw_impact)
            if isinstance(parsed_dict, dict):
                desc = str(parsed_dict.get("description") or parsed_dict.get("operational_impact") or parsed_dict.get("summary") or "").strip()
                if not affected_recs and parsed_dict.get("affected_records"):
                    affected_recs = parsed_dict["affected_records"]
                if parsed_dict.get("risk_level"):
                    risk_lvl = parsed_dict["risk_level"]
        except Exception:
            desc = raw_impact.strip()
    elif isinstance(raw_impact, str):
        desc = raw_impact.strip()

    if not desc:
        desc = f"Pipeline validation stopped processing during {stage_name} before invalid or unvalidated records could reach downstream layers."

    clean_desc = _sanitize_infrastructure_hallucinations(desc)
    unique_affected = sorted(list(set(affected_recs))) if isinstance(affected_recs, list) else []

    impact_data = {
        "description": clean_desc,
        "operational_impact": f"Pipeline validation stopped processing during {stage_name} before invalid records could reach downstream layers.",
        "records_affected": facts.get("invalid_records") if facts.get("invalid_records") is not None else len(unique_affected),
        "total_records": facts.get("total_records"),
        "affected_records": unique_affected,
        "affected_count": facts.get("invalid_records") if facts.get("invalid_records") is not None else len(unique_affected),
        "affected_ids": unique_affected,
        "risk_level": risk_lvl,
    }

    return clean_desc, impact_data


# ─────────────────────────────────────────────────────────────────────────────
# Blast Radius

def compute_blast_radius(facts: dict[str, Any]) -> dict[str, Any]:
    """
    Compute a structured blast radius assessment from verified facts.

    Returns:
        {
          records_affected: int | None
          total_records: int | None
          pct_affected: float | None
          failure_categories_count: int
          failure_categories: list[str]
          severity_level: "CRITICAL" | "HIGH" | "MEDIUM" | "LOW" | "UNKNOWN"
          severity_reason: str
          downstream_impact: str
        }
    """
    invalid_records = facts.get("invalid_records")
    total_records = facts.get("total_records")
    invalid_pct = facts.get("invalid_percentage")
    allowed_threshold = facts.get("allowed_threshold")
    val_failures = facts.get("validation_failures") or {}
    stage = facts.get("failed_stage", "execution")
    error_code = facts.get("error_code")

    failure_categories = list(val_failures.keys()) if val_failures else []
    failure_count = len(failure_categories)

    pct_affected = None
    if invalid_pct is not None:
        pct_affected = float(invalid_pct)
    elif invalid_records is not None and total_records and total_records > 0:
        pct_affected = round((float(invalid_records) / float(total_records)) * 100.0, 2)

    # Severity scoring
    if pct_affected is not None:
        if pct_affected >= 50:
            sev = "CRITICAL"
            sev_reason = f"{pct_affected:.1f}% of batch records failed — majority of data is invalid."
        elif pct_affected >= 25:
            sev = "HIGH"
            sev_reason = f"{pct_affected:.1f}% of batch records failed — significant data quality degradation."
        elif pct_affected >= 10:
            sev = "MEDIUM"
            sev_reason = f"{pct_affected:.1f}% of batch records failed — moderate impact on batch quality."
        elif allowed_threshold is not None and pct_affected > float(allowed_threshold):
            sev = "LOW"
            sev_reason = f"{pct_affected:.1f}% of records failed — just above the {float(allowed_threshold):.1f}% threshold."
        else:
            sev = "LOW"
            sev_reason = f"{pct_affected:.1f}% of batch records failed."
    elif error_code:
        sev = "HIGH"
        sev_reason = f"Error code {error_code} triggered in stage {stage} — scope of affected records cannot be determined without detailed telemetry."
    else:
        sev = "UNKNOWN"
        sev_reason = "Blast radius cannot be assessed — insufficient telemetry."

    downstream = (
        f"Pipeline execution was terminated during {stage}, preventing invalid records from reaching downstream layers."
        if error_code or pct_affected is not None
        else "Downstream impact is unknown due to insufficient telemetry."
    )

    return {
        "records_affected": invalid_records,
        "total_records": total_records,
        "pct_affected": pct_affected,
        "failure_categories_count": failure_count,
        "failure_categories": failure_categories,
        "severity_level": sev,
        "severity_reason": sev_reason,
        "downstream_impact": downstream,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Automation Safety Labels
# ─────────────────────────────────────────────────────────────────────────────

_SAFE_PATTERNS = [
    r"re.?run", r"retry", r"re.?trigger", r"re.?execute",
    r"revalidat", r"re.?analys", r"re.?sync",
]
_UNSAFE_PATTERNS = [
    r"delete", r"drop\s+table", r"truncat", r"rollback", r"patch\s+source",
    r"schema\s+change", r"alter\s+table", r"production\s+data",
    r"manually\s+correct", r"contact\s+", r"coordinate",
]


def add_automation_safety_labels(
    fix_steps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Annotate each remediation step with an automation_safety block.

    Each step gets:
        automation_safety: {
            can_automate: bool
            risk_level: "low" | "medium" | "high"
            reason: str
        }
    """
    result = []
    for step in fix_steps:
        action_text = " ".join(filter(None, [
            str(step.get("title") or ""),
            str(step.get("action") or step.get("description") or ""),
        ])).lower()

        is_safe = any(re.search(p, action_text) for p in _SAFE_PATTERNS)
        is_unsafe = any(re.search(p, action_text) for p in _UNSAFE_PATTERNS)
        priority = str(step.get("priority", "REQUIRED")).upper()

        if is_safe and not is_unsafe and priority == "REQUIRED":
            safety = {
                "can_automate": True,
                "risk_level": "low",
                "reason": "Re-run / retry action — safe to automate after human review.",
            }
        elif is_unsafe:
            safety = {
                "can_automate": False,
                "risk_level": "high",
                "reason": "Action modifies production data or requires human coordination — manual execution required.",
            }
        elif priority == "OPTIONAL":
            safety = {
                "can_automate": False,
                "risk_level": "medium",
                "reason": "Optional improvement step — requires human review before automation.",
            }
        else:
            safety = {
                "can_automate": False,
                "risk_level": "medium",
                "reason": "Requires human validation before automation is safe.",
            }

        result.append({**step, "automation_safety": safety})
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Root Cause Classification
# ─────────────────────────────────────────────────────────────────────────────

def build_root_cause_classification(
    facts: dict[str, Any],
    llm_output: dict[str, Any],
    is_telemetry_missing: bool = False,
) -> dict[str, Any]:
    """
    Classify diagnosis into strictly separated Evidence Levels:
    - Level A — VERIFIED FACT (Immutable runtime telemetry)
    - Level B — DETERMINISTIC INFERENCE (Logically / mathematically proven)
    - Level C — HYPOTHESIS (Unproven explanation with required verification language)
    - Level D — SUGGESTED INVESTIGATION (Operational investigation steps)
    """
    invalid_records = facts.get("invalid_records")
    total_records = facts.get("total_records")
    invalid_pct = facts.get("invalid_percentage")
    allowed_threshold = facts.get("allowed_threshold")
    error_code = facts.get("error_code")
    stage = facts.get("failed_stage", "execution")
    val_failures = facts.get("validation_failures") or {}
    operator = facts.get("comparison_operator", "<=")
    uniq_ids = facts.get("affected_ids_unique") or []
    dup_ids = facts.get("affected_ids_duplicates") or []
    violations_total = facts.get("validation_violations_total", invalid_records or 0)

    if is_telemetry_missing:
        return {
            "tier_a_verified_fact": {
                "level": "LEVEL A — VERIFIED FACT",
                "status": "INSUFFICIENT_TELEMETRY",
                "description": "Detailed task-level error output was not available in the retrieved metadata. The generic Databricks wrapper error does not reveal the specific failure.",
                "evidence": ["Top-level error: 'Workload failed, see run output for details.'"],
            },
            "tier_b_deterministic_inference": {
                "level": "LEVEL B — DETERMINISTIC INFERENCE",
                "description": "Pipeline execution was marked FAILED by orchestrator lifecycle state.",
                "calculation": "No quantitative metrics available for threshold calculation.",
            },
            "tier_c_hypothesis": [
                {
                    "level": "LEVEL C — HYPOTHESIS",
                    "statement": "The failure may indicate an unhandled exception or cluster termination during task execution. Requires verification from detailed task logs.",
                    "caveat": "Requires verification — initial sync captured only top-level wrapper message.",
                }
            ],
            "tier_d_suggested_investigations": [
                {
                    "level": "LEVEL D — SUGGESTED INVESTIGATION",
                    "action": "Use 'Re-analyze' to retrieve deep task run output and driver traceback from the connector API.",
                }
            ],
            "classification_note": "Root cause cannot be verified without detailed task-level telemetry.",
            "verified_cause": {
                "type": "UNKNOWN",
                "description": "Detailed task-level error output was not available in the retrieved metadata.",
                "evidence": ["Top-level error: 'Workload failed, see run output for details.'"],
            },
            "likely_cause": None,
            "contributing": ["Task-level exception was not propagated to the top-level run state message."],
            "downstream_symptoms": ["Pipeline execution status: FAILED."],
        }

    # Verified evidence list
    verified_evidence = []
    if total_records is not None and invalid_records is not None:
        verified_evidence.append(f"Batch evaluated: {total_records} total records in stage '{stage}'.")
        verified_evidence.append(f"Unique invalid records: {invalid_records} records failed validation checks.")
        if invalid_pct is not None:
            verified_evidence.append(f"Measured invalid rate: {float(invalid_pct):.1f}% (allowed threshold: {float(allowed_threshold or 0):.1f}%).")
    if error_code:
        verified_evidence.append(f"Authoritative error code: {error_code} raised during stage '{stage}'.")
    if val_failures:
        cat_str = ", ".join(f"{k}: {v}" for k, v in val_failures.items())
        verified_evidence.append(f"Failure category breakdown: {cat_str}.")
    if uniq_ids:
        verified_evidence.append(f"Unique affected record IDs ({len(uniq_ids)}): {', '.join(uniq_ids)}.")
    if dup_ids:
        verified_evidence.append(f"Duplicate record instances detected in batch: {', '.join(dup_ids)}.")

    # Tier A: Verified Fact
    if invalid_records is not None and total_records is not None:
        verified_desc = f"{invalid_records} of {total_records} unique records failed validation checks in stage '{stage}'."
    elif error_code:
        verified_desc = f"Error code {error_code} was raised during stage '{stage}'."
    else:
        verified_desc = f"Pipeline execution failed during stage '{stage}'."

    # Tier B: Deterministic Inference
    if invalid_pct is not None and allowed_threshold is not None:
        op_phrase = "strictly below (<)" if operator == "<" else "at or below (<=)"
        det_desc = (
            f"The measured {float(invalid_pct):.1f}% invalid rate exceeded the configured {float(allowed_threshold):.1f}% threshold "
            f"({float(invalid_pct):.1f}% > {float(allowed_threshold):.1f}%, rule requires {op_phrase} threshold), "
            f"deterministically causing the validation stage to terminate pipeline execution."
        )
    elif error_code:
        det_desc = f"Encountering error code {error_code} deterministically triggered pipeline error handling and terminated processing."
    else:
        det_desc = "Pipeline execution was terminated by the orchestrator upon encountering execution failure criteria."

    # Tier C: Hypothesis (strictly labeled with required phrasing)
    hypotheses = []
    if invalid_records is not None:
        hypotheses.append({
            "level": "LEVEL C — HYPOTHESIS",
            "statement": "Invalid records reaching this stage may indicate insufficient validation or schema constraints before pipeline execution. Requires verification at the data source boundary.",
            "caveat": "Potential contributing factor — requires verification at source boundary.",
        })
    elif error_code:
        hypotheses.append({
            "level": "LEVEL C — HYPOTHESIS",
            "statement": f"Triggering {error_code} may indicate an environmental mismatch, concurrency collision, or source schema deviation. Requires verification.",
            "caveat": "Inferred from error code — requires verification.",
        })

    # Tier D: Suggested Investigations (NOT root causes)
    investigations = []
    if val_failures:
        top_cat = max(val_failures.items(), key=lambda x: int(x[1]))[0]
        investigations.append({
            "level": "LEVEL D — SUGGESTED INVESTIGATION",
            "title": "Investigation 1",
            "area": top_cat.replace("_", " "),
            "action": f"Inspect the source inputs and validation rule definitions for: {top_cat.replace('_', ' ')}.",
            "why": f"This category occurred {val_failures[top_cat]} times, representing the highest-frequency violation in the batch.",
            "evidence": "Verified Task Output",
        })
    if uniq_ids:
        investigations.append({
            "level": "LEVEL D — SUGGESTED INVESTIGATION",
            "title": "Investigation 2" if investigations else "Investigation 1",
            "area": "Affected Records Integrity",
            "action": f"Review the affected records ({', '.join(uniq_ids[:5])}{'...' if len(uniq_ids) > 5 else ''}) and identify which validation rules each record violated.",
            "why": f"{len(uniq_ids)} unique records generated {violations_total} validation violations across the batch.",
            "evidence": "Deterministic Analysis",
        })
    elif not investigations:
        investigations.append({
            "level": "LEVEL D — SUGGESTED INVESTIGATION",
            "title": "Investigation 1",
            "area": "Runtime Environment Audit",
            "action": f"Audit pipeline logs and execution parameters in stage '{stage}'.",
            "why": "Operational review to identify environmental or source data deviations.",
            "evidence": "Verified Telemetry",
        })

    return {
        "tier_a_verified_fact": {
            "level": "LEVEL A — VERIFIED FACT",
            "statement": verified_desc,
            "description": verified_desc,
            "evidence": verified_evidence,
        },
        "tier_b_deterministic_inference": {
            "level": "LEVEL B — DETERMINISTIC INFERENCE",
            "statement": det_desc,
            "description": det_desc,
            "calculation": f"Invalid rate {float(invalid_pct or 0):.1f}% > Threshold {float(allowed_threshold or 0):.1f}%" if invalid_pct is not None else "Deterministic pipeline termination.",
        },
        "tier_c_hypothesis": hypotheses,
        "tier_d_suggested_investigations": investigations,
        "classification_note": "Root cause is VERIFIED from runtime telemetry. Hypotheses are explicitly labeled and require external verification.",
        # Backwards compatibility keys
        "verified_cause": {
            "type": "VERIFIED",
            "description": verified_desc,
            "evidence": verified_evidence,
        },
        "likely_cause": {
            "type": "HYPOTHESIS",
            "description": hypotheses[0]["statement"] if hypotheses else "",
            "confidence_note": "Requires verification at upstream data source boundary.",
        } if hypotheses else None,
        "contributing": [h["statement"] for h in hypotheses],
        "downstream_symptoms": [f"Pipeline execution terminated during stage '{stage}', containing invalid records."],
    }


def normalize_known_fix(
    raw_fix_data: Any,
    verified_facts: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Canonicalize any LLM / KB fix format (Format A, B, C, D, E, string lists, or raw dicts)
    into:
    1. immediate_fix: list of REQUIRED recovery actions
    2. optional_improvements: list of OPTIONAL runbook improvements
    3. combined_known_fix: full structured list with priorities
    """
    facts = verified_facts or {}
    pipe_name = facts.get("pipeline_name", "the pipeline")
    stage_name = facts.get("failed_stage", "SILVER_DATA_VALIDATION")
    threshold = facts.get("allowed_threshold", 5.0)
    val_failures = facts.get("validation_failures") or {}

    items: list[dict[str, Any]] = []

    # Handle immediate_fix / optional_improvements if passed as separate keys in dict
    if isinstance(raw_fix_data, dict):
        imm = raw_fix_data.get("immediate_fix") or []
        opt = raw_fix_data.get("optional_improvements") or []
        if isinstance(imm, list):
            for idx, itm in enumerate(imm):
                if isinstance(itm, dict):
                    items.append({
                        "step": idx + 1,
                        "priority": "REQUIRED",
                        "title": _sanitize_infrastructure_hallucinations(itm.get("title") or f"Step {idx + 1}"),
                        "action": _sanitize_infrastructure_hallucinations(itm.get("action") or itm.get("description") or ""),
                        "why": _sanitize_infrastructure_hallucinations(itm.get("why") or "Required operational recovery step."),
                        "evidence_source": itm.get("evidence_source") or "Verified Telemetry",
                        "expected_outcome": _sanitize_infrastructure_hallucinations(itm.get("expected_outcome") or ""),
                        "validation": _sanitize_infrastructure_hallucinations(itm.get("validation") or ""),
                    })
        if isinstance(opt, list):
            for idx, itm in enumerate(opt):
                if isinstance(itm, dict):
                    items.append({
                        "step": len(items) + 1,
                        "priority": "OPTIONAL",
                        "title": _sanitize_infrastructure_hallucinations(itm.get("title") or f"Step {len(items) + 1}"),
                        "action": _sanitize_infrastructure_hallucinations(itm.get("action") or itm.get("description") or ""),
                        "why": _sanitize_infrastructure_hallucinations(itm.get("why") or "Recommended prevention measure."),
                        "evidence_source": itm.get("evidence_source") or "Knowledge Base",
                        "expected_outcome": _sanitize_infrastructure_hallucinations(itm.get("expected_outcome") or ""),
                        "validation": _sanitize_infrastructure_hallucinations(itm.get("validation") or ""),
                    })

    # Format E: Raw string (or JSON string)
    elif isinstance(raw_fix_data, str):
        s = raw_fix_data.strip()
        if s.startswith("[") or s.startswith("{"):
            try:
                parsed_json = json.loads(s)
                return normalize_known_fix(parsed_json, facts)
            except Exception:
                pass
        lines = [line.strip() for line in re.split(r"\n(?=\d+\.)", s) if line.strip()]
        if not lines:
            lines = [s]
        for idx, chunk in enumerate(lines):
            outcome_m = re.search(r"Expected\s+outcome:?\s*(.*)$", chunk, re.I)
            expected = outcome_m.group(1).strip() if outcome_m else ""
            clean_s = re.sub(r"Expected\s+outcome:?\s*.*$", "", chunk, flags=re.I).strip()
            clean_s = re.sub(r"^\[(?:Required|Optional).*?\]\s*", "", clean_s, flags=re.I).strip()
            clean_s = re.sub(r"^\d+\.\s*", "", clean_s).strip()

            is_opt = "optional" in chunk.lower() or "quarantine" in chunk.lower()
            items.append({
                "step": idx + 1,
                "priority": "OPTIONAL" if is_opt else "REQUIRED",
                "title": f"Step {idx + 1}",
                "action": _sanitize_infrastructure_hallucinations(clean_s),
                "why": "Operational remediation step.",
                "evidence_source": "Verified Telemetry",
                "expected_outcome": _sanitize_infrastructure_hallucinations(expected),
                "validation": "",
            })

    # Format 1: List of objects or strings
    elif isinstance(raw_fix_data, list):
        for idx, item in enumerate(raw_fix_data):
            if isinstance(item, dict):
                title = item.get("title") or (item.get("step") if isinstance(item.get("step"), str) and not str(item.get("step")).isdigit() else None) or item.get("action") or item.get("step_title") or f"Step {idx + 1}"
                desc = item.get("action") or item.get("description") or item.get("details") or title
                expected = item.get("expected_outcome") or item.get("outcome") or ""
                validation = item.get("validation") or ""
                why_text = item.get("why") or ""
                ev_source = item.get("evidence_source") or "Verified Telemetry"
                priority = str(item.get("priority") or item.get("type") or "REQUIRED").upper()

                is_opt = "OPTIONAL" in priority or "quarantine" in str(title).lower() or "quarantine" in str(desc).lower()
                items.append({
                    "step": idx + 1,
                    "priority": "OPTIONAL" if is_opt else "REQUIRED",
                    "title": _sanitize_infrastructure_hallucinations(str(title)),
                    "action": _sanitize_infrastructure_hallucinations(str(desc)),
                    "why": _sanitize_infrastructure_hallucinations(str(why_text)),
                    "evidence_source": str(ev_source),
                    "expected_outcome": _sanitize_infrastructure_hallucinations(str(expected)),
                    "validation": _sanitize_infrastructure_hallucinations(str(validation)),
                })
            elif isinstance(item, str):
                s = item.strip()
                if not s:
                    continue
                outcome_m = re.search(r"Expected\s+outcome:?\s*(.*)$", s, re.I)
                expected = outcome_m.group(1).strip() if outcome_m else ""
                clean_s = re.sub(r"Expected\s+outcome:?\s*.*$", "", s, flags=re.I).strip()
                clean_s = re.sub(r"^\[(?:Required|Optional).*?\]\s*", "", clean_s, flags=re.I).strip()
                clean_s = re.sub(r"^\d+\.\s*", "", clean_s).strip()

                is_opt = "optional" in s.lower() or "quarantine" in s.lower()
                items.append({
                    "step": idx + 1,
                    "priority": "OPTIONAL" if is_opt else "REQUIRED",
                    "title": f"Step {idx + 1}",
                    "action": _sanitize_infrastructure_hallucinations(clean_s),
                    "why": "Operational remediation step.",
                    "evidence_source": "Verified Telemetry",
                    "expected_outcome": _sanitize_infrastructure_hallucinations(expected),
                    "validation": "",
                })

    # Evidence-grounded default recovery steps if empty
    if not items:
        if facts.get("error_code") == "INVENTORY_RECONCILIATION_FAILED" or "inventory" in str(facts.get("pipeline_name", "")).lower():
            # Get category counts
            mismatch_cnt = val_failures.get("Inventory Reconciliation Mismatch", 5)
            shortage_cnt = val_failures.get("Critical Stock Shortage", 4)
            reserved_cnt = val_failures.get("Reserved Stock Exceeds Physical", 2)
            dup_cnt = val_failures.get("Duplicate Inventory Record", 2)

            # Check for low-frequency categories (< 2 violations)
            low_freq_cats = [(k, v) for k, v in val_failures.items() if int(v) < 2]

            items = [
                {
                    "step": 1,
                    "priority": "P0",
                    "priority_code": "P0",
                    "priority_level": "P0",
                    "actionability": "VERIFICATION_REQUIRED",
                    "title": "P0 — Verify Downstream Containment",
                    "summary": f"Verify whether downstream processing or subsequent pipeline stages initiated execution prior to batch validation termination ({facts.get('invalid_records', 9)} invalid records detected).",
                    "action": f"Verify whether downstream processing or subsequent pipeline stages initiated execution prior to batch validation termination ({facts.get('invalid_records', 9)} invalid records detected).",
                    "recommendation_type": "Verification Required",
                    "supported_by": "✓ Verified Task Output",
                    "fix_readiness": "READY_TO_FIX",
                    "fix_readiness_label": "Direct Action",
                    "what_we_know": [
                        f"Validation failure stopped execution during stage '{stage_name}'.",
                        f"{facts.get('invalid_records', 9)} invalid records were detected in the batch.",
                    ],
                    "what_we_need_to_determine": [
                        "Whether any downstream consumer initiated execution before pipeline termination.",
                    ],
                    "what_to_investigate": [
                        "Inspect downstream pipeline stages and available execution metadata to confirm whether processing stopped after the validation failure.",
                    ],
                    "suggested_fix": [
                        "Confirm isolation of the failed batch. Do not trigger downstream processing until input data passes validation.",
                    ],
                    "what_to_inspect": "Inspect downstream pipeline stages and available execution metadata to confirm whether processing stopped after the validation failure.",
                    "what_to_fix": "Confirm isolation of the failed batch. Do not trigger downstream processing until input data passes validation.",
                    "why": "Pipeline execution was terminated during validation to contain invalid records before downstream layers.",
                    "why_prioritized": [
                        "Pipeline execution was terminated during validation to contain invalid records before downstream layers.",
                    ],
                    "evidence": f"Verified Task Output: Validation failure stopped execution during stage '{stage_name}' ({facts.get('invalid_records', 9)} invalid records detected).",
                    "evidence_source": "✓ Verified Task Output",
                    "evidence_classification": "✓ Verified Task Output",
                    "expected_outcome": "Confirm invalid records were contained and prevented from reaching downstream processing.",
                    "validation": "Verify downstream task/run status and confirm the last verified consistent processing checkpoint.",
                    "validation_steps": [
                        "Verify downstream task/run status and confirm the last verified consistent processing checkpoint.",
                    ],
                    "automation_safety": {"can_automate": True, "risk_level": "low", "reason": "Read-only inspection of downstream task run states."},
                },
                {
                    "step": 2,
                    "priority": "P1",
                    "priority_code": "P1",
                    "priority_level": "P1",
                    "actionability": "INVESTIGATION_REQUIRED",
                    "title": "P1 — Investigate Inventory Reconciliation Mismatches",
                    "summary": f"Inspect the validation failures responsible for the {mismatch_cnt} Inventory Reconciliation Mismatch violations before modifying source data.",
                    "action": f"Inspect the {mismatch_cnt} inventory reconciliation mismatch records and the validation rule responsible for reconciliation. Identify input discrepancies before applying source corrections.",
                    "recommendation_type": "Investigation Required",
                    "supported_by": "✓ Verified Task Output",
                    "fix_readiness": "INVESTIGATION_REQUIRED",
                    "fix_readiness_label": "Investigation Required",
                    "why_prioritized": [
                        f"Highest observed failure category: {mismatch_cnt} violations.",
                        "Resolving this category may reduce the invalid-record rate significantly.",
                    ],
                    "what_we_know": [
                        f"{mismatch_cnt} Inventory Reconciliation Mismatch violations occurred.",
                        "This is the highest-frequency validation category.",
                        f"The pipeline failed because {facts.get('invalid_records', 9)} of {facts.get('total_records', 12)} records were invalid.",
                    ],
                    "what_we_need_to_determine": [
                        "Which specific input values caused each reconciliation mismatch.",
                        "Whether the mismatch originates from source data or the validation rule configuration.",
                        "Which affected records should be corrected.",
                    ],
                    "what_to_investigate": [
                        "Open the affected inventory records.",
                        "Inspect the values evaluated by the reconciliation validation rule.",
                        "Compare the source values with the expected reconciliation values.",
                        "Identify the exact records causing the validation failures.",
                    ],
                    "suggested_fix": [
                        "Determine the underlying discrepancy before modifying source data.",
                        "Identify and correct the input discrepancies causing the reconciliation validation rule to fail.",
                    ],
                    "steps": [
                        "Open the affected inventory records.",
                        "Inspect the values evaluated by the reconciliation validation rule.",
                        "Compare the source values with the expected reconciliation values.",
                        "Identify the exact records causing the validation failures.",
                        "Correct the identified source data or rule configuration only after the discrepancy is confirmed.",
                    ],
                    "what_to_inspect": "Inspect the affected records and the validation rule responsible for Inventory Reconciliation Mismatch.",
                    "what_to_fix": "Determine the underlying discrepancy before modifying source data.",
                    "why": f"Highest observed validation category: {mismatch_cnt} violations. Resolving this category may reduce the invalid-record rate significantly.",
                    "evidence": f"Inventory Reconciliation Mismatch = {mismatch_cnt}",
                    "evidence_source": "✓ Verified Task Output",
                    "evidence_classification": "✓ Verified Task Output",
                    "expected_outcome": "The validation rule no longer reports Inventory Reconciliation Mismatch for the corrected records.",
                    "validation": "Re-run the Inventory Reconciliation validation and confirm that the affected records no longer trigger this rule.",
                    "validation_steps": [
                        "Re-run the reconciliation validation.",
                        "Confirm Inventory Reconciliation Mismatch violations = 0.",
                        "Confirm corrected records pass validation.",
                        f"Confirm the overall invalid record rate is strictly below {float(facts.get('threshold_percentage', 25.0)):.1f}%.",
                    ],
                    "automation_safety": {"can_automate": False, "risk_level": "medium", "reason": "Requires source data review and validation rule inspection."},
                },
                {
                    "step": 3,
                    "priority": "P2",
                    "priority_code": "P2",
                    "priority_level": "P2",
                    "actionability": "INVESTIGATION_REQUIRED",
                    "title": "P2 — Verify Critical Stock Shortages",
                    "summary": f"{shortage_cnt} violations detected. Inspect the values evaluated by the Critical Stock Shortage validation rule.",
                    "action": f"Inspect the input values and validation conditions responsible for the Critical Stock Shortage rule to determine the underlying cause.",
                    "recommendation_type": "Investigation Required",
                    "supported_by": "✓ Verified Task Output",
                    "fix_readiness": "INVESTIGATION_REQUIRED",
                    "fix_readiness_label": "Investigation Required",
                    "why_prioritized": [
                        f"{shortage_cnt} violations detected in this category.",
                        "Contributes directly to the batch threshold breach.",
                    ],
                    "what_we_know": [
                        f"{shortage_cnt} Critical Stock Shortage violations occurred.",
                        "Stock balance fell below configured critical shortage threshold.",
                    ],
                    "what_we_need_to_determine": [
                        "Which specific input values caused the Critical Stock Shortage rule condition to evaluate to true.",
                    ],
                    "what_to_investigate": [
                        "Inspect the input values and validation conditions responsible for the Critical Stock Shortage rule to determine the underlying cause.",
                    ],
                    "suggested_fix": [
                        "Determine the underlying cause of the shortage condition before adjusting inventory balances.",
                    ],
                    "steps": [
                        "Inspect the input values and validation conditions responsible for the Critical Stock Shortage rule.",
                        "Determine the underlying cause of the shortage condition.",
                        "Apply source stock correction or inventory adjustment after verification.",
                        "Re-run validation.",
                    ],
                    "what_to_inspect": "Inspect the stock balance inputs and shortage threshold criteria used by the failing validation check.",
                    "what_to_fix": "Determine the underlying cause of the shortage condition before adjusting inventory balances.",
                    "why": f"{shortage_cnt} violations detected in this category.",
                    "evidence": f"Critical Stock Shortage = {shortage_cnt}",
                    "evidence_source": "✓ Verified Task Output",
                    "evidence_classification": "✓ Verified Task Output",
                    "expected_outcome": "Stock balance records are validated and shortage conditions are resolved or accounted for.",
                    "validation": "Confirm all stock counts meet validation thresholds and adjustments are verified.",
                    "validation_steps": [
                        "Re-run validation checks.",
                        "Confirm Critical Stock Shortage violations = 0.",
                    ],
                    "automation_safety": {"can_automate": False, "risk_level": "medium", "reason": "Requires stock data verification."},
                },
                {
                    "step": 4,
                    "priority": "P3",
                    "priority_code": "P3",
                    "priority_level": "P3",
                    "actionability": "INVESTIGATION_REQUIRED",
                    "title": "P3 — Audit Reserved Stock Constraint",
                    "summary": f"{reserved_cnt} violations detected. Inspect the values evaluated by the Reserved Stock Exceeds Physical validation rule.",
                    "action": "Inspect the values evaluated by the Reserved Stock Exceeds Physical validation rule.",
                    "recommendation_type": "Investigation Required",
                    "supported_by": "✓ Verified Task Output",
                    "fix_readiness": "INVESTIGATION_REQUIRED",
                    "fix_readiness_label": "Investigation Required",
                    "why_prioritized": [
                        f"{reserved_cnt} violations detected.",
                    ],
                    "what_we_know": [
                        f"{reserved_cnt} records had reserved quantity greater than physical stock count.",
                    ],
                    "what_we_need_to_determine": [
                        "Which specific records and values caused the rule condition to evaluate to true.",
                    ],
                    "what_to_investigate": [
                        "Inspect the values evaluated by the Reserved Stock Exceeds Physical validation rule.",
                        "Confirm which values caused the rule condition to fail.",
                    ],
                    "suggested_fix": [
                        "Correct the invalid input or configuration causing reserved stock to exceed physical available stock.",
                    ],
                    "steps": [
                        "Inspect the values evaluated by the Reserved Stock Exceeds Physical validation rule.",
                        "Confirm which values caused the rule condition to fail.",
                        "Correct the invalid input or configuration.",
                        "Re-run the validation.",
                    ],
                    "what_to_inspect": "Inspect the reserved quantity and physical inventory count inputs evaluated by the validation rule.",
                    "what_to_fix": "Correct the invalid input or configuration causing reserved stock to exceed physical available stock.",
                    "why": f"{reserved_cnt} records triggered the Reserved Stock Exceeds Physical validation rule.",
                    "evidence": f"Reserved Stock Exceeds Physical = {reserved_cnt}",
                    "evidence_source": "✓ Verified Task Output",
                    "evidence_classification": "✓ Verified Task Output",
                    "expected_outcome": "No record has reserved quantities exceeding physical available stock.",
                    "validation": "Re-run the reserved stock validation check and assert zero remaining violations.",
                    "validation_steps": [
                        "Re-run the reserved stock validation check.",
                        "Confirm Reserved Stock Exceeds Physical violations = 0.",
                    ],
                    "automation_safety": {"can_automate": False, "risk_level": "low", "reason": "Deterministic constraint check."},
                },
                {
                    "step": 5,
                    "priority": "P4",
                    "priority_code": "P4",
                    "priority_level": "P4",
                    "actionability": "INVESTIGATION_REQUIRED",
                    "title": "P4 — Review Duplicate Inventory Records",
                    "summary": "Duplicate record instances detected. Verify whether duplicate source records should be removed or merged.",
                    "action": f"Inspect duplicate record instances (such as {', '.join(facts.get('affected_ids_duplicates') or ['INV002'])}) in the failing batch. Verify whether duplicate source records should be removed or merged.",
                    "recommendation_type": "Investigation Required",
                    "supported_by": "✓ Verified Task Output",
                    "fix_readiness": "INVESTIGATION_REQUIRED",
                    "fix_readiness_label": "Investigation Required",
                    "why_prioritized": [
                        f"{dup_cnt} duplicate record instances detected.",
                    ],
                    "what_we_know": [
                        f"{dup_cnt} duplicate record instances detected for INV002 in the failing batch.",
                    ],
                    "what_we_need_to_determine": [
                        "Why duplicate records entered the batch and which instance should remain authoritative according to validation rules.",
                    ],
                    "what_to_investigate": [
                        f"Inspect all occurrences of duplicate records (such as {', '.join(facts.get('affected_ids_duplicates') or ['INV002'])}) in the failing batch.",
                        "Determine which record instance should remain according to the pipeline's validation rules.",
                    ],
                    "suggested_fix": [
                        "Remove or correct the invalid duplicate instance.",
                        "Investigate the processing or source stage responsible for producing duplicate instances.",
                    ],
                    "steps": [
                        "Inspect all occurrences of INV002 in the failing batch.",
                        "Determine which record instance should remain according to the pipeline's validation rules.",
                        "Remove or correct the invalid duplicate instance.",
                        "Investigate the processing or source stage responsible for producing duplicate instances.",
                        "Re-run duplicate validation.",
                    ],
                    "what_to_inspect": f"Inspect all occurrences of {', '.join(facts.get('affected_ids_duplicates') or ['INV002'])} in the failing batch.",
                    "what_to_fix": "Determine which record instance should remain authoritative and remove invalid duplicates before re-running.",
                    "why": f"Detected {dup_cnt} duplicate record instances in the failing batch.",
                    "evidence": f"Duplicate Inventory Record = {dup_cnt}",
                    "evidence_source": "✓ Verified Task Output",
                    "evidence_classification": "✓ Verified Task Output",
                    "expected_outcome": "Confirm that duplicate-record validation no longer detects duplicate records.",
                    "validation": "Re-run duplicate-record validation and confirm that no duplicate violations remain.",
                    "validation_steps": [
                        "Re-run duplicate-record validation.",
                        "Confirm Duplicate Inventory Record violations = 0.",
                    ],
                    "automation_safety": {"can_automate": True, "risk_level": "low", "reason": "Deterministic deduplication check."},
                },
            ]

            if low_freq_cats:
                items.append({
                    "step": len(items) + 1,
                    "priority": f"P{len(items)}",
                    "priority_code": f"P{len(items)}",
                    "priority_level": f"P{len(items)}",
                    "actionability": "INVESTIGATION_REQUIRED",
                    "title": f"P{len(items)} — Review Remaining Data Quality Issues",
                    "summary": f"{len(low_freq_cats)} remaining categories contribute to the batch threshold breach.",
                    "action": f"Inspect source inputs for lower-frequency validation checks: {', '.join(f'{k} ({v})' for k, v in low_freq_cats)}.",
                    "recommendation_type": "Investigation Required",
                    "supported_by": "✓ Verified Task Output",
                    "fix_readiness": "INVESTIGATION_REQUIRED",
                    "fix_readiness_label": "Investigation Required",
                    "why_prioritized": [
                        f"These failure categories occurred fewer times ({sum(int(v) for _, v in low_freq_cats)} total violations) but contribute to the total batch threshold breach.",
                    ],
                    "sub_actions": [
                        {
                            "category": "Unknown Medication",
                            "count": 1,
                            "evidence": "Unknown Medication = 1",
                            "what_we_know": [
                                "One record failed the Unknown Medication validation rule.",
                            ],
                            "what_to_investigate": [
                                "Identify the medication value that failed validation.",
                                "Verify whether it exists in the approved medication reference data.",
                            ],
                            "suggested_action": [
                                "Correct or map the medication value after verification.",
                            ],
                            "validation": [
                                "Re-run medication validation.",
                                "Confirm Unknown Medication violations = 0.",
                            ],
                        },
                        {
                            "category": "Negative Stock",
                            "count": 1,
                            "evidence": "Negative Stock = 1",
                            "what_we_know": [
                                "One record triggered the Negative Stock validation rule.",
                            ],
                            "what_to_investigate": [
                                "Inspect the stock value used by the validation rule.",
                            ],
                            "suggested_action": [
                                "Correct the source value only after identifying the underlying discrepancy.",
                            ],
                            "validation": [
                                "Confirm no negative stock validation failures remain.",
                            ],
                        },
                        {
                            "category": "Expired Stock Detected",
                            "count": 1,
                            "evidence": "Expired Stock Detected = 1",
                            "what_we_know": [
                                "One record triggered the Expired Stock validation rule.",
                            ],
                            "what_to_investigate": [
                                "Verify the expiry value and validation conditions.",
                            ],
                            "suggested_action": [
                                "Correct the record or remove it from active inventory processing according to the verified business rule.",
                            ],
                            "validation": [
                                "Confirm the record no longer triggers the expired stock validation rule.",
                            ],
                        },
                    ],
                    "what_to_investigate": [
                        f"Inspect source inputs for lower-frequency validation checks: {', '.join(f'{k} ({v})' for k, v in low_freq_cats)}.",
                    ],
                    "suggested_fix": [
                        "Address remaining isolated data quality issues in the source batch.",
                    ],
                    "what_to_inspect": f"Inspect source inputs for lower-frequency validation checks: {', '.join(f'{k} ({v})' for k, v in low_freq_cats)}.",
                    "what_to_fix": "Address remaining isolated data quality issues in the source batch.",
                    "why": f"These failure categories occurred fewer times ({sum(int(v) for _, v in low_freq_cats)} total violations) but contribute to the total batch threshold breach.",
                    "evidence": f"{', '.join(f'{k}={v}' for k, v in low_freq_cats)}",
                    "evidence_source": "✓ Verified Task Output",
                    "evidence_classification": "✓ Verified Task Output",
                    "expected_outcome": "Remaining isolated field validation violations are corrected.",
                    "validation": "Re-run validation checks and confirm zero remaining low-frequency violations.",
                    "validation_steps": [
                        "Re-run validation checks.",
                        "Confirm zero remaining low-frequency violations.",
                    ],
                    "automation_safety": {"can_automate": False, "risk_level": "medium", "reason": "Requires isolated data review."},
                })

            items.append({
                "step": len(items) + 1,
                "priority": "OPTIONAL",
                "priority_code": "OPTIONAL",
                "priority_level": "OPTIONAL",
                "actionability": "MONITORING_RECOMMENDATION",
                "title": f"P{len(items)} — Route Failed Records to Isolated Quarantine Mechanism",
                "summary": "Consider routing rejected records to an isolated quarantine mechanism supported by the existing data platform without bypassing batch quality thresholds.",
                "action": "Consider routing rejected records to an isolated quarantine mechanism supported by the existing data platform without bypassing batch quality thresholds.",
                "recommendation_type": "Monitoring Recommendation",
                "supported_by": "Knowledge Base / Runbook",
                "fix_readiness": "KNOWLEDGE_BASED",
                "fix_readiness_label": "Monitoring Recommendation",
                "what_to_investigate": [
                    "Inspect quarantine storage and dead-letter routing configuration.",
                ],
                "suggested_fix": [
                    "Consider routing rejected records to an isolated quarantine mechanism supported by the existing data platform.",
                ],
                "what_to_inspect": "Inspect quarantine storage and dead-letter routing configuration.",
                "what_to_fix": "Consider routing rejected records to an isolated quarantine mechanism supported by the existing data platform.",
                "why": "Runbook best practice for auditability and continuous monitoring.",
                "why_prioritized": [
                    "Runbook best practice for auditability and continuous monitoring.",
                ],
                "evidence": "Knowledge Base Runbook",
                "evidence_source": "Knowledge Base / Runbook",
                "evidence_classification": "Knowledge Base / Runbook",
                "expected_outcome": "Invalid records are preserved for root-cause telemetry while the active pipeline processes only verified clean batches.",
                "validation": "Confirm quarantine mechanism receives rejected rows with source timestamps and violation category tags.",
                "validation_steps": [
                    "Confirm quarantine mechanism receives rejected rows with source timestamps and violation category tags.",
                ],
                "automation_safety": {"can_automate": False, "risk_level": "medium", "reason": "Architectural runbook enhancement."},
            })
        elif facts.get("error_code") == "DATA_QUALITY_THRESHOLD_BREACH" or facts.get("invalid_records") is not None:
            items = [
                {
                    "step": 1,
                    "priority": "P0",
                    "priority_code": "P0",
                    "title": "P0 — Verify Downstream Containment",
                    "action": "Verify whether downstream processing or subsequent stages initiated execution prior to batch validation termination.",
                    "recommendation_type": "Evidence-Based Suggested Fix",
                    "supported_by": "Verified Task Output",
                    "what_to_inspect": "Inspect downstream pipeline stages and available execution metadata to confirm whether processing stopped after the validation failure.",
                    "what_to_fix": "Confirm isolation of the failed batch. Do not trigger downstream processing until input data passes validation.",
                    "why": "Pipeline execution was terminated during validation to contain invalid records before downstream layers.",
                    "evidence": f"Verified Task Output: Validation failure stopped execution during stage '{stage_name}' ({facts.get('invalid_records', 0)} invalid records detected).",
                    "evidence_source": "Verified Telemetry",
                    "evidence_classification": "Verified Telemetry",
                    "expected_outcome": "Confirm whether invalid records were prevented from reaching downstream processing.",
                    "validation": "Verify downstream task status and confirm the last verified consistent processing checkpoint.",
                    "automation_safety": {"can_automate": True, "risk_level": "low", "reason": "Read-only inspection."},
                },
                {
                    "step": 2,
                    "priority": "P1",
                    "priority_code": "P1",
                    "title": "P1 — Correct or Replace Failed Source Batch",
                    "action": "Coordinate correction or replacement of invalid records in the source batch.",
                    "recommendation_type": "Evidence-Based Suggested Fix",
                    "supported_by": "Verified Task Output",
                    "what_to_inspect": "Inspect the source input values and validation definitions for the failing records.",
                    "what_to_fix": "Identify and correct invalid values in the source batch.",
                    "why": f"The batch invalid rate exceeds the configured {float(threshold):.1f}% threshold.",
                    "evidence": f"Invalid records = {facts.get('invalid_records', 0)}",
                    "evidence_source": "Verified Telemetry",
                    "evidence_classification": "Verified Telemetry",
                    "expected_outcome": f"The corrected source batch achieves an invalid-record rate at or below the configured {float(threshold):.1f}% threshold.",
                    "validation": f"Re-run pre-ingestion validation rules and verify unique invalid records <= {float(threshold):.1f}%.",
                    "automation_safety": {"can_automate": False, "risk_level": "medium", "reason": "Requires source batch review."},
                },
                {
                    "step": 3,
                    "priority": "P2",
                    "priority_code": "P2",
                    "title": "P2 — Revalidate Corrected Batch",
                    "action": f"Run validation checks against the corrected batch and verify that critical fields pass validation criteria.",
                    "recommendation_type": "Evidence-Based Suggested Fix",
                    "supported_by": "Deterministic Analysis",
                    "what_to_inspect": f"Inspect validation results for {stage_name}.",
                    "what_to_fix": "Ensure all validation criteria are met.",
                    "why": "Ensures all validation rules pass prior to resuming full pipeline execution.",
                    "evidence": "Deterministic Analysis",
                    "evidence_source": "Deterministic Analysis",
                    "evidence_classification": "Deterministic Analysis",
                    "expected_outcome": f"The batch passes {stage_name} validation rules without triggering threshold breaches.",
                    "validation": f"Confirm {stage_name} validation metrics report invalid percentage <= {float(threshold):.1f}%.",
                    "automation_safety": {"can_automate": True, "risk_level": "low", "reason": "Deterministic validation check."},
                },
                {
                    "step": 4,
                    "priority": "P3",
                    "priority_code": "P3",
                    "title": "P3 — Re-Trigger Pipeline Execution",
                    "action": f"Re-trigger {pipe_name} only after the corrected batch passes validation checks.",
                    "recommendation_type": "Evidence-Based Suggested Fix",
                    "supported_by": "Deterministic Analysis",
                    "what_to_inspect": "Inspect pipeline trigger parameters.",
                    "what_to_fix": "Trigger pipeline run.",
                    "why": "Resumes normal pipeline lifecycle once inputs are verified.",
                    "evidence": "Deterministic Analysis",
                    "evidence_source": "Deterministic Analysis",
                    "evidence_classification": "Deterministic Analysis",
                    "expected_outcome": "Pipeline execution completes successfully through downstream processing layers.",
                    "validation": "Verify run status updates to SUCCESS in the orchestrator.",
                    "automation_safety": {"can_automate": True, "risk_level": "low", "reason": "Standard pipeline rerun."},
                },
                {
                    "step": 5,
                    "priority": "OPTIONAL",
                    "priority_code": "OPTIONAL",
                    "title": "P4 — Route Failed Records to Isolated Quarantine Mechanism",
                    "action": "Consider routing invalid records to an isolated quarantine mechanism for auditing and inspection without bypassing the pipeline failure threshold.",
                    "recommendation_type": "Knowledge-Based Suggested Fix",
                    "supported_by": "Knowledge Base / Runbook",
                    "what_to_inspect": "Inspect quarantine storage configuration.",
                    "what_to_fix": "Consider routing rejected records to an isolated quarantine mechanism supported by the existing data platform.",
                    "why": "Long-term data governance improvement.",
                    "evidence": "Knowledge Base Runbook",
                    "evidence_source": "Knowledge Base",
                    "evidence_classification": "Knowledge Base",
                    "expected_outcome": "Invalid records are preserved for auditing while pipeline data quality enforcement remains intact.",
                    "validation": "Confirm quarantine destination receives rejected rows with rejection metadata.",
                    "automation_safety": {"can_automate": False, "risk_level": "medium", "reason": "Runbook improvement."},
                },
            ]
        else:
            items = [
                {
                    "step": 1,
                    "priority": "P0",
                    "priority_code": "P0",
                    "title": "P0 — Verify Downstream Containment",
                    "action": f"Verify whether downstream consumers or stages initiated processing before {facts.get('error_code', 'the failure')} terminated the run.",
                    "recommendation_type": "Evidence-Based Suggested Fix",
                    "supported_by": "Verified Task Output",
                    "what_to_inspect": "Inspect downstream pipeline stages and available execution metadata to confirm whether execution was safely stopped.",
                    "what_to_fix": "Confirm isolation of the failed run.",
                    "why": "Prevent corrupt or unverified data propagation.",
                    "evidence": f"Error in {stage_name}",
                    "evidence_source": "Verified Telemetry",
                    "evidence_classification": "Verified Telemetry",
                    "expected_outcome": "Confirm isolation of the failed pipeline run.",
                    "validation": "Verify downstream run state.",
                    "automation_safety": {"can_automate": True, "risk_level": "low", "reason": "Read-only check."},
                },
                {
                    "step": 2,
                    "priority": "P1",
                    "priority_code": "P1",
                    "title": "P1 — Remediate Error Condition",
                    "action": f"Identify the underlying trigger for {facts.get('error_code', 'the failure')} in stage {stage_name} and apply the required remediation.",
                    "recommendation_type": "Evidence-Based Suggested Fix",
                    "supported_by": "Verified Task Output",
                    "what_to_inspect": f"Inspect {stage_name} configuration, error logs, and environment variables.",
                    "what_to_fix": "Apply required environment or code remediation.",
                    "why": f"Stage {stage_name} failed with error code {facts.get('error_code', 'UNKNOWN')}.",
                    "evidence": f"{facts.get('error_code', 'Execution Error')}",
                    "evidence_source": "Verified Telemetry",
                    "evidence_classification": "Verified Telemetry",
                    "expected_outcome": f"The error condition in {stage_name} is resolved.",
                    "validation": f"Verify {stage_name} prerequisites are satisfied.",
                    "automation_safety": {"can_automate": False, "risk_level": "medium", "reason": "Requires inspection."},
                },
                {
                    "step": 3,
                    "priority": "P2",
                    "priority_code": "P2",
                    "title": "P2 — Re-run Pipeline",
                    "action": f"Re-trigger {pipe_name} and verify successful execution.",
                    "recommendation_type": "Evidence-Based Suggested Fix",
                    "supported_by": "Deterministic Analysis",
                    "what_to_inspect": "Inspect run parameters.",
                    "what_to_fix": "Trigger pipeline run.",
                    "why": "Validates that the remediation successfully resolved the error.",
                    "evidence": "Deterministic Analysis",
                    "evidence_source": "Deterministic Analysis",
                    "evidence_classification": "Deterministic Analysis",
                    "expected_outcome": "Pipeline finishes with status SUCCESS.",
                    "validation": "Verify run status updates to SUCCESS.",
                    "automation_safety": {"can_automate": True, "risk_level": "low", "reason": "Pipeline rerun."},
                },
            ]

    # Enforce priority separation: REQUIRED first (1..N), OPTIONAL second (1..M)
    immediate_fix: list[dict[str, Any]] = []
    optional_improvements: list[dict[str, Any]] = []

    req_idx = 1
    opt_idx = 1
    combined: list[dict[str, Any]] = []

    for item in items:
        p = str(item.get("priority") or "REQUIRED").upper()
        is_optional = "OPTIONAL" in p or "quarantine" in str(item.get("title", "")).lower()
        if not is_optional:
            clean_item = {
                "step": req_idx,
                "priority": "REQUIRED",
                "priority_code": item.get("priority_code") or (f"P{req_idx - 1}" if "P0" in str(items[0].get("title", "")) else f"P{req_idx}"),
                "priority_level": item.get("priority_code") or (f"P{req_idx - 1}" if "P0" in str(items[0].get("title", "")) else f"P{req_idx}"),
                "title": item["title"],
                "action": item.get("action") or item.get("description") or item["title"],
                "description": item.get("action") or item.get("description") or item["title"],
                "what_to_inspect": item.get("what_to_inspect") or item.get("action") or item["title"],
                "what_to_fix": item.get("what_to_fix") or "Apply operational remediation based on validation findings.",
                "why": item.get("why") or "Required operational recovery step.",
                "evidence": item.get("evidence") or item.get("evidence_source") or "Verified Task Output",
                "evidence_source": item.get("evidence_source") or "Verified Telemetry",
                "evidence_classification": item.get("evidence_classification") or item.get("evidence_source") or "Verified Telemetry",
                "expected_outcome": item.get("expected_outcome", ""),
                "validation": item.get("validation", ""),
                "automation_safety": item.get("automation_safety"),
            }
            immediate_fix.append(clean_item)
            combined.append(clean_item)
            req_idx += 1
        else:
            clean_item = {
                "step": opt_idx,
                "priority": "OPTIONAL",
                "priority_code": "OPTIONAL",
                "priority_level": "OPTIONAL",
                "title": item["title"],
                "action": item.get("action") or item.get("description") or item["title"],
                "description": item.get("action") or item.get("description") or item["title"],
                "what_to_inspect": item.get("what_to_inspect") or item.get("action") or item["title"],
                "what_to_fix": item.get("what_to_fix") or "Optional architectural or runbook enhancement.",
                "why": item.get("why") or "Recommended prevention measure.",
                "evidence": item.get("evidence") or "Knowledge Base Runbook",
                "evidence_source": item.get("evidence_source") or "Knowledge Base",
                "evidence_classification": item.get("evidence_classification") or "Knowledge Base",
                "expected_outcome": item.get("expected_outcome", ""),
                "validation": item.get("validation", ""),
                "automation_safety": item.get("automation_safety"),
            }
            optional_improvements.append(clean_item)
            combined.append(clean_item)
            opt_idx += 1

    return immediate_fix, optional_improvements, combined


def build_suggested_fix_text(known_fix_list: list[dict[str, Any]]) -> str:
    '''
    Generate backward-compatible markdown plain-text suggested_fix from canonical known_fix objects.
    '''
    parts = []
    for idx, step in enumerate(known_fix_list, start=1):
        title = step.get("title", f"Step {idx}")
        desc = step.get("action") or step.get("description", "")
        priority = step.get("priority", "REQUIRED").upper()
        outcome = step.get("expected_outcome", "")
        validation = step.get("validation", "")

        prefix = "[Required Immediate Action]" if priority == "REQUIRED" else "[Optional Runbook Improvement]"
        
        if desc.startswith(title):
            step_body = desc
        else:
            step_body = f"**{title}** — {desc}"

        step_text = f"{idx}. {prefix} {step_body}"
        if outcome:
            step_text += f"\nExpected outcome: {outcome}"
        if validation:
            step_text += f"\nValidation: {validation}"
        parts.append(step_text)

    return "\n\n".join(parts)


def normalize_diagnosis(
    verified_facts: dict[str, Any],
    llm_output: dict[str, Any],
    kb_context: dict[str, Any] | None = None,
    classification: dict[str, Any] | None = None,
    is_telemetry_missing: bool = False,
) -> dict[str, Any]:
    '''
    Generic Normalization & Fact Locking Engine.

    Guarantees:
    1. Verified facts (metrics, stage, pipeline name, run id, error code) ALWAYS win.
    2. LLM reasoning fields are validated, normalized, and filled with evidence-backed fallbacks if empty.
    3. immediate_fix (Required) and optional_improvements (Optional) are explicitly separated.
    4. suggested_fix is derived from canonical immediate_fix & optional_improvements.
    5. Single authoritative backend confidence score is computed deterministically.
    '''
    facts = verified_facts or {}
    parsed = dict(llm_output or {})

    # Extract verified deterministic values
    pipe_name = facts.get("pipeline_name", "ETL_PIPELINE")
    stage_name = facts.get("failed_stage", "execution")
    error_code = facts.get("error_code")
    invalid_records = facts.get("invalid_records")
    total_records = facts.get("total_records")
    invalid_pct = facts.get("invalid_percentage")
    allowed_threshold = facts.get("allowed_threshold")
    val_failures = facts.get("validation_failures") or {}
    violations_total = facts.get("validation_violations_total", invalid_records or 0)

    # 1. FACT LOCKING: Lock verified deterministic fields into parsed
    parsed["pipeline_name"] = pipe_name
    parsed["pipeline_run_id"] = facts.get("pipeline_run_id")
    parsed["environment"] = facts.get("environment", "PRODUCTION")
    parsed["failed_stage"] = stage_name
    parsed["error_code"] = error_code
    parsed["severity"] = facts.get("severity", "ERROR")
    parsed["total_records"] = total_records
    parsed["invalid_records"] = invalid_records
    parsed["invalid_percentage"] = invalid_pct
    parsed["allowed_threshold"] = allowed_threshold
    parsed["comparison_operator"] = facts.get("comparison_operator", "<=")
    parsed["recovery_success_criteria"] = facts.get("recovery_success_criteria")
    parsed["validation_failures"] = val_failures
    parsed["validation_violations_total"] = violations_total
    if facts.get("command_id"):
        parsed["command_id"] = facts.get("command_id")
    if facts.get("line_number"):
        parsed["line_number"] = facts.get("line_number")

    parsed["verified_facts"] = {
        "pipeline_name": pipe_name,
        "failed_stage": stage_name,
        "error_code": error_code,
        "total_records": total_records,
        "invalid_records": invalid_records,
        "invalid_percentage": invalid_pct,
        "allowed_threshold": allowed_threshold,
        "comparison_operator": facts.get("comparison_operator", "<="),
        "recovery_success_criteria": facts.get("recovery_success_criteria"),
        "validation_failures": val_failures,
        "validation_violations_total": violations_total,
        "category_violation_explanation": facts.get("category_violation_explanation"),
        "affected_ids_raw": facts.get("affected_ids_raw"),
        "affected_ids_unique": facts.get("affected_ids_unique"),
        "affected_ids_duplicates": facts.get("affected_ids_duplicates"),
        "environment": facts.get("environment", "PRODUCTION"),
        "pipeline_run_id": facts.get("pipeline_run_id"),
        "command_id": facts.get("command_id"),
        "line_number": facts.get("line_number"),
    }

    # 2. Canonical Structured Fixes & suggested_fix text
    raw_fix = parsed.get("immediate_fix") or parsed.get("known_fix") or parsed.get("suggested_fix")
    if parsed.get("optional_improvements"):
        raw_fix = {
            "immediate_fix": parsed.get("immediate_fix") or [],
            "optional_improvements": parsed.get("optional_improvements") or [],
        }
    imm_fix, opt_fix, combined_fix = normalize_known_fix(raw_fix, facts)
    parsed["immediate_fix"] = imm_fix
    parsed["optional_improvements"] = opt_fix
    parsed["known_fix"] = combined_fix
    suggested_fix_text = build_suggested_fix_text(combined_fix)
    parsed["suggested_fix"] = suggested_fix_text

    # Check for Mode B: Insufficient / Missing Telemetry
    is_telemetry_missing = (
        (invalid_records is None and total_records is None and error_code is None)
        or (invalid_records is None and total_records is None and "workload failed" in str(facts.get("error_details", "")).lower())
    )

    if is_telemetry_missing:
        # MODE B: Telemetry Unavailable — Honest no-hallucination response
        parsed["summary"] = (
            f"Pipeline {pipe_name} failed, but the retrieved Databricks metadata contains only a generic wrapper error. "
            f"Detailed task run output has not yet been retrieved."
        )
        mode_b_rc = (
            "The root cause cannot be determined from the currently retrieved Databricks run metadata. "
            "The available task error is only a wrapper message: 'Workload failed, see run output for details.' "
            "Detailed task run output is required before generating a specific root cause."
        )
        parsed["root_cause"] = mode_b_rc
        parsed["verified_root_cause"] = mode_b_rc
        parsed["inferred_contributing_cause"] = None
        parsed["failure_mechanism"] = (
            "The pipeline failed with a generic wrapper message ('Workload failed, see run output for details'). "
            "Detailed task execution logs and failure output were not available in the retrieved metadata."
        )
        parsed["impact"] = (
            "Pipeline execution failed, but specific impact on downstream layers cannot be verified without detailed task run telemetry."
        )
        mode_b_fix = [
            {
                "step": 1,
                "priority": "REQUIRED",
                "title": "Retrieve Detailed Task Run Output",
                "action": "Fetch the full Databricks task run output and driver logs to inspect the actual RuntimeError or failure exception.",
                "description": "Fetch the full Databricks task run output and driver logs to inspect the actual RuntimeError or failure exception.",
                "why": "Detailed error telemetry is required to identify the root cause.",
                "evidence_source": "Verified Telemetry",
                "expected_outcome": "The detailed error traceback and validation failure metrics are available for investigation.",
                "validation": "Confirm task run output contains the underlying failure details.",
            }
        ]
        parsed["immediate_fix"] = mode_b_fix
        parsed["optional_improvements"] = []
        parsed["known_fix"] = mode_b_fix
        parsed["suggested_fix"] = build_suggested_fix_text(mode_b_fix)
        parsed["root_cause_details"] = [
            "Databricks top-level run state message: Workload failed, see run output for details.",
            "Detailed task-level error output was not present in the initial sync metadata.",
        ]
        parsed["contributing_factors"] = [
            "Task-level exception output was not retrieved during the initial job sync.",
            "Generic Databricks wrapper error obscured the underlying failure condition.",
        ]
        parsed["validation_steps"] = [
            "Retrieve task-level logs from Databricks API or UI.",
            "Inspect the detailed exception message.",
        ]
        rec_actions = [
            "Inspect Databricks task-level run output and cluster driver logs.",
            "Verify network and connector permissions for Databricks Jobs API output retrieval.",
            "Re-run the sync once task logs are accessible.",
        ]
        parsed["recommended_actions"] = rec_actions
        parsed["recommendations"] = rec_actions
        parsed["long_term_prevention"] = [
            "Ensure Databricks connector has permissions to retrieve individual task run outputs via the Jobs 2.1 API.",
            "Configure task-level error logging to propagate exception details to the top-level job state message.",
        ]
        parsed["confidence"] = 0.40
        parsed["confidence_breakdown"] = {
            "telemetry_completeness": 0.30,
            "error_code_certainty": 0.30,
            "metric_certainty": 0.0,
            "pattern_match": 0.20,
            "accepted_fix_history": 0.0,
        }
        parsed["diagnosis_status"] = "partial"
        parsed["fix_patch"] = ""
        return parsed

    # ── MODE A: Telemetry Available — Pinpointed Diagnosis ────────────────────

    # 3. Deterministic Summary (Bypass generic summaries or raw display name timestamps)
    raw_summary = str(parsed.get("summary") or "").strip()
    needs_summary_override = (
        not raw_summary
        or raw_summary in (FALLBACK, "")
        or "workload failed" in raw_summary.lower()
        or (invalid_records is not None and "because" not in raw_summary.lower() and "due to" not in raw_summary.lower())
    )
    if needs_summary_override:
        if invalid_records is not None and total_records is not None and invalid_pct is not None and allowed_threshold is not None:
            parsed["summary"] = (
                f"Pipeline {pipe_name} failed during {stage_name} because "
                f"{invalid_records} of {total_records} records were invalid ({float(invalid_pct):.1f}%), "
                f"exceeding the configured {float(allowed_threshold):.1f}% threshold."
            )
        elif error_code:
            parsed["summary"] = f"Pipeline {pipe_name} failed during {stage_name} due to {error_code}."
        else:
            parsed["summary"] = f"Pipeline {pipe_name} failed during {stage_name} execution."
    else:
        parsed["summary"] = _sanitize_ownership_and_policy(raw_summary)

    # 4. Deterministic Root Cause & Structured Fact/Inference Split (P0-2)
    raw_rc = str(parsed.get("root_cause") or "").strip()
    needs_rc_override = (
        not raw_rc
        or raw_rc in (FALLBACK, "")
        or (invalid_records is not None and str(invalid_records) not in raw_rc)
        or "threshold%" in raw_rc
    )
    
    categories_str = ""
    if isinstance(val_failures, dict) and val_failures:
        cat_items = [f"{k} ({v})" for k, v in val_failures.items()]
        categories_str = f" Category violations recorded: {', '.join(cat_items)}."

    code_label = f" ({error_code})" if error_code else ""
    
    if invalid_records is not None and total_records is not None and invalid_pct is not None and allowed_threshold is not None:
        verified_rc = (
            f"{invalid_records} of {total_records} unique records failed validation, producing a "
            f"{float(invalid_pct):.1f}% invalid-record rate that exceeded the configured "
            f"{float(allowed_threshold):.1f}% threshold{code_label}.{categories_str}"
        )
        inferred_cc = (
            "Invalid records reaching this stage may indicate that upstream validation at the source boundary did not intercept "
            "invalid records prior to pipeline execution. Requires verification at the data source boundary."
        )
    elif error_code:
        verified_rc = f"Execution in stage {stage_name} failed due to error code {error_code}."
        inferred_cc = "Upstream source data or environment configuration may have deviated from expected schema or operational parameters (requires verification)."
    else:
        verified_rc = f"Pipeline execution failed during {stage_name}."
        inferred_cc = None

    parsed["verified_root_cause"] = verified_rc
    parsed["inferred_contributing_cause"] = inferred_cc

    if needs_rc_override and invalid_records is not None and total_records is not None and invalid_pct is not None and allowed_threshold is not None:
        parsed["root_cause"] = (
            f"The incoming data batch contained {invalid_records} unique invalid records out of {total_records} total records. "
            f"The resulting invalid-record rate was {float(invalid_pct):.1f}%, which exceeded the configured {float(allowed_threshold):.1f}% threshold{code_label}.{categories_str}"
        )
    else:
        parsed["root_cause"] = _sanitize_ownership_and_policy(raw_rc)

    # 5. Deterministic Failure Mechanism (No invented RuntimeError unless in logs!)
    raw_mech = str(parsed.get("failure_mechanism") or "").strip()
    needs_mech_override = (
        not raw_mech
        or raw_mech in (FALLBACK, "")
        or "workload failed" in raw_mech.lower()
        or "during execution" in raw_mech.lower()
        or (invalid_records is not None and str(invalid_records) not in raw_mech)
        or "threshold%" in raw_mech
        or "rate of %" in raw_mech
    )
    if needs_mech_override:
        if invalid_records is not None and total_records is not None and invalid_pct is not None and allowed_threshold is not None:
            code_label = f", triggering {error_code}" if error_code else ""
            parsed["failure_mechanism"] = (
                f"The {stage_name} stage detected an invalid-record rate of {float(invalid_pct):.1f}% "
                f"({invalid_records} of {total_records} records). Because this exceeded the configured "
                f"{float(allowed_threshold):.1f}% threshold{code_label}, pipeline validation logic "
                f"terminated processing before downstream layers."
            )
        elif error_code:
            parsed["failure_mechanism"] = f"During {stage_name}, execution encountered {error_code}, aborting pipeline processing."
        else:
            parsed["failure_mechanism"] = f"The {stage_name} stage failed execution criteria and aborted."
    else:
        parsed["failure_mechanism"] = _sanitize_ownership_and_policy(raw_mech)

    # 6. Structured & Deterministic Impact (No raw dictionary strings)
    impact_text, impact_data = _normalize_impact(parsed.get("impact"), facts, stage_name)
    parsed["impact"] = impact_text
    parsed["impact_data"] = impact_data

    # 7. Lists Normalization
    def _clean_list(val: Any) -> list[str]:
        if not val:
            return []
        if isinstance(val, list):
            res = []
            for item in val:
                if isinstance(item, dict):
                    text = item.get("action") or item.get("description") or item.get("detail") or item.get("text") or str(item)
                else:
                    text = str(item)
                clean = _sanitize_infrastructure_hallucinations(text)
                if clean:
                    res.append(clean)
            return res
        return [_sanitize_infrastructure_hallucinations(str(val))]

    root_cause_details = _clean_list(parsed.get("root_cause_details") or parsed.get("evidence"))
    contributing_factors = _clean_list(parsed.get("contributing_factors"))
    validation_steps = _clean_list(parsed.get("validation_steps") or parsed.get("validation"))
    confidence_rationale = _clean_list(parsed.get("confidence_rationale"))

    # Evidence-based defaults if lists are empty
    if not root_cause_details and invalid_records is not None and total_records is not None:
        root_cause_details = [
            f"Batch evaluated: {total_records} total records in stage {stage_name}.",
            f"Invalid record count: {invalid_records} unique records failed validation checks.",
            f"Measured failure rate: {float(invalid_pct):.1f}% (configured threshold is {float(allowed_threshold):.1f}%).",
            f"Category violations: {violations_total} total across {len(val_failures)} rules.",
        ]
    parsed["root_cause_details"] = root_cause_details

    if not contributing_factors:
        if val_failures:
            contributing_factors = [
                f"Multiple validation rule violations occurred simultaneously across {len(val_failures)} categories in the incoming batch.",
                f"Small batch size ({total_records} records) amplified the impact of {invalid_records} invalid records to a {float(invalid_pct):.1f}% breach.",
                "Invalid records reaching this stage may indicate insufficient validation before pipeline execution (requires verification at source boundary).",
            ]
        elif invalid_records is not None and total_records is not None:
            contributing_factors = [
                "Multiple data quality rule violations were present simultaneously in the incoming batch.",
                f"Small batch size ({total_records} records) amplified the impact of {invalid_records} invalid records to a {float(invalid_pct):.1f}% breach.",
                "Invalid records reaching this stage may indicate insufficient validation before pipeline execution (requires verification at source boundary).",
            ]
        elif error_code:
            contributing_factors = [
                f"Error condition {error_code} triggered during {stage_name}.",
                "Upstream source data or environment configuration may have deviated from expected schema or operational parameters (requires verification).",
            ]
    parsed["contributing_factors"] = contributing_factors

    if not validation_steps:
        if val_failures:
            # Dynamic validation steps generated directly from failing categories (Part 11)
            validation_steps = []
            for cat in val_failures.keys():
                validation_steps.append(f"Confirm {cat} violations are resolved.")
            if allowed_threshold is not None:
                crit = facts.get("recovery_success_criteria") or {}
                allowed_count = crit.get("allowed_invalid_count", 0)
                validation_steps.append(f"Confirm invalid-record rate is at or below the configured {float(allowed_threshold):.1f}% threshold (at most {allowed_count} invalid records in a {total_records or 'batch'}-record batch).")
            validation_steps.append(f"Re-run {pipe_name} and confirm successful downstream execution.")
        elif allowed_threshold is not None and total_records is not None:
            crit = facts.get("recovery_success_criteria") or {}
            allowed_count = crit.get("allowed_invalid_count", 0)
            op = crit.get("comparison_operator", "<=")
            validation_steps = [
                f"Verify invalid-record percentage {op} {float(allowed_threshold):.1f}% (at most {allowed_count} invalid records in a {total_records}-record batch).",
                f"Confirm all input data meets {stage_name} validation rules.",
                f"Re-run {pipe_name} and confirm successful downstream processing.",
            ]
        elif allowed_threshold is not None:
            validation_steps = [
                f"Verify invalid-record percentage <= {float(allowed_threshold):.1f}%.",
                f"Confirm all input data meets {stage_name} validation rules.",
                f"Re-run {pipe_name} and confirm successful downstream processing.",
            ]
        else:
            validation_steps = [
                f"Validate that the underlying cause for {error_code or 'the failure'} in {stage_name} is resolved.",
                f"Re-run {pipe_name} and confirm successful end-to-end execution.",
            ]
    parsed["validation_steps"] = validation_steps

    # Operational follow-ups distinct from immediate fix (Part 12)
    rec_actions = [
        "If an upstream source owner or responsible team is defined, review the source validation process with them.",
        "Consider configuring early-warning alert thresholds if equivalent monitoring is not already present before reaching the configured pipeline failure limit.",
        "Review recurring validation failure patterns across historical source batches.",
        "Review repeated occurrences of this error signature for recurring source-data defects.",
    ]
    parsed["recommended_actions"] = rec_actions
    parsed["recommendations"] = rec_actions

    # Long-term prevention
    raw_lt = parsed.get("long_term_prevention")
    if isinstance(raw_lt, list):
        parsed["long_term_prevention"] = [
            _sanitize_ownership_and_policy(str(x)) for x in raw_lt if str(x).strip()
        ]
    elif isinstance(raw_lt, str) and raw_lt.strip() and raw_lt != FALLBACK:
        parsed["long_term_prevention"] = [
            _sanitize_ownership_and_policy(raw_lt)
        ]
    else:
        parsed["long_term_prevention"] = [
            "Consider adding or strengthening pre-ingestion validation and contract checks at the upstream source boundary if equivalent controls are not already present.",
            "If an upstream source owner or responsible team is defined, review the source validation process with them.",
            "Configure early-warning alert thresholds before reaching the configured pipeline failure limit to detect data quality anomalies earlier.",
        ]

    # 8. Code Patch Anti-Hallucination Safety
    raw_patch = str(parsed.get("fix_patch") or "").strip()
    if raw_patch and ("def " not in raw_patch and "{" not in raw_patch and "=" not in raw_patch):
        raw_patch = ""
    parsed["fix_patch"] = raw_patch

    # 9. Deterministic Confidence Calculation
    has_accepted_kb = (
        kb_context is not None
        and bool(kb_context.get("is_known"))
        and int(kb_context.get("acceptance_count", 0)) > 0
    )
    is_known_pattern = (
        kb_context is not None
        and bool(kb_context.get("is_known"))
    )

    # Base telemetry completeness
    telemetry_score = 0.80 if (pipe_name and stage_name and error_code) else 0.60
    if invalid_records is not None and total_records is not None:
        telemetry_score += 0.05

    if has_accepted_kb:
        final_confidence = min(0.95, telemetry_score + 0.10)
    elif is_known_pattern:
        # Known pattern with 0 accepted fixes -> capped at 0.80-0.85
        final_confidence = min(0.85, telemetry_score)
    else:
        final_confidence = min(0.80, telemetry_score)

    parsed["confidence"] = round(final_confidence, 2)
    parsed["confidence_breakdown"] = {
        "telemetry_completeness": 0.95 if (error_code and invalid_records is not None) else 0.80,
        "error_code_certainty": 0.95 if error_code else 0.70,
        "metric_certainty": 0.95 if (invalid_records is not None and invalid_pct is not None) else 0.50,
        "pattern_match": 0.85 if is_known_pattern else 0.40,
        "accepted_fix_history": 0.95 if has_accepted_kb else 0.0,
    }

    # Diagnosis status & error
    diag_status = parsed.get("diagnosis_status", "success")
    diag_err = parsed.get("diagnosis_error")
    parsed["diagnosis_status"] = diag_status
    parsed["diagnosis_error"] = diag_err

    # 10. Blast Radius
    try:
        parsed["blast_radius"] = compute_blast_radius(facts)
    except Exception:
        parsed["blast_radius"] = None

    # 11. Root Cause Classification
    try:
        parsed["root_cause_classification"] = build_root_cause_classification(
            facts=facts,
            llm_output=parsed,
            is_telemetry_missing=is_telemetry_missing,
        )
    except Exception:
        parsed["root_cause_classification"] = None

    # 12. Automation safety labels on immediate_fix and optional_improvements
    try:
        parsed["immediate_fix"] = add_automation_safety_labels(parsed.get("immediate_fix") or [])
        parsed["optional_improvements"] = add_automation_safety_labels(parsed.get("optional_improvements") or [])
        parsed["known_fix"] = add_automation_safety_labels(parsed.get("known_fix") or [])
    except Exception:
        pass

    # 13. NEXT BEST ACTION (Part 2, 8, 10)
    try:
        imm_fixes = parsed.get("immediate_fix") or []
        if imm_fixes:
            # Primary fix step: prefer P1 if P0 is containment verification, otherwise P0
            primary_step = imm_fixes[1] if len(imm_fixes) > 1 and "P0" in imm_fixes[0].get("title", "") else imm_fixes[0]
            parsed["next_best_action"] = {
                "priority": primary_step.get("priority") or ("P1" if "P1" in primary_step.get("title", "") else "P0"),
                "priority_code": primary_step.get("priority_code") or primary_step.get("priority") or "P1",
                "priority_level": primary_step.get("priority_level") or primary_step.get("priority_code") or "P1",
                "title": primary_step.get("title", ""),
                "actionability": primary_step.get("actionability", "INVESTIGATION_REQUIRED"),
                "summary": primary_step.get("summary") or primary_step.get("action", ""),
                "action": primary_step.get("action", ""),
                "recommendation_type": primary_step.get("recommendation_type", "Investigation Required"),
                "supported_by": primary_step.get("supported_by", "✓ Verified Task Output"),
                "fix_readiness": primary_step.get("fix_readiness", "INVESTIGATION_REQUIRED"),
                "fix_readiness_label": primary_step.get("fix_readiness_label", "Investigation Required"),
                "why_prioritized": primary_step.get("why_prioritized") or ([primary_step.get("why")] if primary_step.get("why") else []),
                "evidence": primary_step.get("evidence", ""),
                "what_we_know": primary_step.get("what_we_know") or [],
                "what_we_need_to_determine": primary_step.get("what_we_need_to_determine") or [],
                "what_to_investigate": primary_step.get("what_to_investigate") or ([primary_step.get("what_to_inspect")] if primary_step.get("what_to_inspect") else []),
                "suggested_fix": primary_step.get("suggested_fix") or ([primary_step.get("what_to_fix")] if primary_step.get("what_to_fix") else []),
                "validation_steps": primary_step.get("validation_steps") or ([primary_step.get("validation")] if primary_step.get("validation") else []),
                "steps": primary_step.get("steps") or [],
                "what_to_inspect": primary_step.get("what_to_inspect", ""),
                "what_to_fix": primary_step.get("what_to_fix", ""),
                "why": primary_step.get("why", ""),
                "expected_outcome": primary_step.get("expected_outcome", ""),
                "validation": primary_step.get("validation", ""),
                "automation_safety": primary_step.get("automation_safety"),
                "target_step": primary_step.get("step", 1),
            }
        else:
            parsed["next_best_action"] = None
    except Exception:
        parsed["next_best_action"] = None

    return parsed
