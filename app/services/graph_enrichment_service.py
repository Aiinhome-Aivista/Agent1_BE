"""
Graph Enrichment Service
─────────────────────────
Runs AFTER a runbook is uploaded + chunked + embedded into Chroma.
Asks the LLM to distill the runbook into structured entities and writes
them into ArangoDB so the diagnosis flow can later traverse:

    error in logs  →  matching patterns  →  fix_actions  →  ranked by
                                            past incident success rate

This module also records the resolution of an incident back into the
graph (which fixes did we actually apply, and did they work) so the
recommendation quality improves over time.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.services import arango_service as graph
from app.services.llm_service import llm_service

logger = logging.getLogger(__name__)


# _EXTRACTION_PROMPT = """You are extracting a structured knowledge graph from
# an operational runbook. Read the runbook text below and emit a SINGLE JSON
# object (no markdown fences) of this exact shape:

# {
#   "components":     [string, ...],              // 1-5 services this runbook touches
#                                                 // (e.g. "Azure Data Factory", "Databricks",
#                                                 //  "AWS Glue", "Git", "Spark", "S3", "Kafka")
#   "error_patterns": [                           // 1-6 distilled error signatures
#     {
#       "signature": string,                      // <= 80 chars human-readable
#       "keywords":  [string, ...],               // 3-8 lowercase tokens that would
#                                                 // appear in a real log for this error
#       "sample":    string                       // <= 200 chars representative log excerpt,
#                                                 // empty string if not in the doc
#     }
#   ],
#   "fix_actions":    [string, ...],              // 1-10 atomic remediation steps,
#                                                 // each a single self-contained instruction.
#                                                 // Strip ordinals ("1.", "Step 2:") from the start.
#   "pattern_fix_map": [                          // which fixes apply to which patterns
#     { "pattern_index": int, "fix_indexes": [int, ...] }
#   ]
# }

# Rules:
# - Use ONLY information present in the runbook. Do not invent steps.
# - Keywords MUST be lowercase, no whitespace inside a single keyword.
# - Keep signatures stable across re-extractions of the same doc
#   (e.g. "OOM in Spark executor" not "the spark executor crashed today").
# - If a section cannot be extracted, return an empty list for it.
# """
_EXTRACTION_PROMPT = """You are an information-extraction engine. Your output
feeds a graph database that powers FUTURE pipeline-failure diagnoses: when a
new incident occurs, the diagnosis service tokenizes the live error log,
matches those tokens against the `keywords` of patterns you emit here,
traverses the graph to linked fix_actions, and ranks them by historical
success. So your job is NOT to summarize the runbook for a human reader —
it is to distill it into stable, machine-matchable nodes that will be useful
months from now against logs you cannot see.

═══════════════════════════════════════════════════════════════════
OUTPUT SHAPE (JSON only — no prose, no markdown fences, no extra keys)
═══════════════════════════════════════════════════════════════════
{
  "components": [string, ...],
  "error_patterns": [
    {"signature": string, "keywords": [string, ...], "sample": string}
  ],
  "fix_actions": [string, ...],
  "pattern_fix_map": [
    {"pattern_index": int, "fix_indexes": [int, ...]}
  ]
}

═══════════════════════════════════════════════════════════════════
FIELD QUALITY BAR
═══════════════════════════════════════════════════════════════════

▸ components  (1-5 items)
   Use the CONTROLLED vocabulary below. If the runbook mentions a tool
   under any alias, normalize to the canonical name.
     "Azure Data Factory"   ← also: ADF, ADFv2, Data Factory
     "Databricks"           ← also: ADB, Azure Databricks
     "AWS Glue"             ← also: Glue, GlueETL
     "Spark"                ← only when generic; prefer Databricks/EMR if specific
     "Git"                  ← also: GitHub, GitLab, Bitbucket
     "Airflow"              ← also: MWAA, Composer
     "Kafka"                ← also: MSK; Event Hubs stays "Event Hubs"
     "S3"                   ← also: Amazon S3
     "ADLS"                 ← Azure Data Lake Storage Gen2
     "Snowflake", "BigQuery", "Redshift"
   If the tool isn't in this list, use the runbook's exact name. Do NOT
   invent components that aren't mentioned in the runbook.

▸ error_patterns  (1-6 items)
   Each pattern is ONE DISTINCT FAILURE MODE the runbook addresses.
   Multiple symptoms of the same root cause = ONE pattern, not several.

   signature  — Short, stable noun-phrase naming the failure mode.
                Format: "<failure-noun> in <where>" or "<failure> on <op>".
                Good:  "OOM in Spark executor"
                       "Schema drift on join"
                       "Linked service auth failure in ADF"
                       "Git authentication denied"
                Bad:   "the executor crashed"        ← not a noun phrase
                       "various memory issues"       ← vague
                       "Spark job failed yesterday"  ← unstable wording
                CRITICAL: the same failure mode must always produce the same
                signature wording, even across re-runs of this extraction.

   keywords   — 3-8 lowercase tokens that would LITERALLY appear in the
                error log when this failure occurs. They are matched
                against tokens extracted from future logs, so they must
                look like what a JVM/Python/CLI actually emits — not
                English prose words from the runbook.
                For "OOM in Spark executor":
                  ["outofmemoryerror","executor","heap","gc","oom","killed"]
                NOT: ["memory","problem","spark","issue"]  ← won't match real logs
                Rules:
                  • lowercase
                  • no whitespace inside a token (use underscores: "key_error")
                  • prefer real exception/error class names over English words
                  • prefer specific tokens ("nullpointerexception") over generic ("error")
                  • skip stopwords: the, is, error, failed, info, warning

   sample     — A representative log excerpt copied VERBATIM from the
                runbook, ≤200 chars. Empty string if the runbook has no
                concrete log example. Do NOT fabricate one.

▸ fix_actions  (1-10 items)
   Each item is one ATOMIC, SELF-CONTAINED remediation. A reader should
   be able to execute it without scrolling back to other steps.
   Good:  "Increase spark.executor.memory to 8g in cluster config"
          "Run `git config --global credential.helper store` then re-push"
          "Verify the Key Vault reference on the linked service is unexpired"
   Bad:   "Try increasing memory"                ← non-specific
          "See Appendix B"                       ← references external content
          "Step 3: now restart the cluster"      ← contains ordinal
          "Open a JIRA ticket"                   ← not a technical remediation
          "Restart cluster, then check logs"     ← two actions in one
   Rules:
     • Strip ordinals ("1.", "Step 2:", "- ", "•") from the start.
     • One sentence per action, ≤200 chars.
     • Include the runbook's specific parameter values ("set X to 8g"),
       not the abstract version ("raise X").
     • EXCLUDE verification, monitoring, ticketing, and communication
       steps. Only actions that change system state.
     • EXCLUDE pure prerequisites (e.g. "ensure you have admin role")
       unless they themselves remediate the failure.

▸ pattern_fix_map
   Tells the graph which fixes address which patterns. M:N is fine: a
   fix can address multiple patterns, a pattern can have multiple fixes.
     pattern_index — 0-based index into error_patterns above
     fix_indexes   — 0-based indexes into fix_actions above
   If the runbook says "for X do A, B; for Y do A, C":
     [{"pattern_index":0,"fix_indexes":[0,1]},
      {"pattern_index":1,"fix_indexes":[0,2]}]
   If the runbook has one pattern with several fixes:
     [{"pattern_index":0,"fix_indexes":[0,1,2,3]}]
   If you cannot infer the mapping with confidence, return [] — the
   system will fall back to all-to-all linking automatically.

═══════════════════════════════════════════════════════════════════
WORKED EXAMPLE
═══════════════════════════════════════════════════════════════════
Input runbook (excerpt):
  Title: Spark OOM Triage  ·  Category: Databricks

  When a Databricks job fails with executor OOM (you'll see
  `java.lang.OutOfMemoryError: Java heap space` or "Container killed
  by YARN for exceeding memory limits"), do the following:
    1. Increase spark.executor.memory from 4g to 8g in the cluster config.
    2. Set spark.sql.shuffle.partitions to 400.
    3. Restart the cluster.
    4. Re-run the failed job.
  If the failure is instead a DRIVER OOM ("OutOfMemoryError" in the
  driver logs, not executor), collect a thread dump first via `jstack`,
  then increase spark.driver.memory to 16g.

Expected output:
{
  "components": ["Databricks", "Spark"],
  "error_patterns": [
    {
      "signature": "OOM in Spark executor",
      "keywords": ["outofmemoryerror","heap","executor","container","killed","yarn"],
      "sample": "java.lang.OutOfMemoryError: Java heap space"
    },
    {
      "signature": "OOM in Spark driver",
      "keywords": ["outofmemoryerror","driver","heap","jstack"],
      "sample": ""
    }
  ],
  "fix_actions": [
    "Increase spark.executor.memory from 4g to 8g in the cluster config",
    "Set spark.sql.shuffle.partitions to 400",
    "Restart the cluster",
    "Re-run the failed job",
    "Collect a thread dump via jstack",
    "Increase spark.driver.memory to 16g"
  ],
  "pattern_fix_map": [
    {"pattern_index": 0, "fix_indexes": [0, 1, 2, 3]},
    {"pattern_index": 1, "fix_indexes": [4, 5, 2, 3]}
  ]
}

═══════════════════════════════════════════════════════════════════
HARD RULES
═══════════════════════════════════════════════════════════════════
1. Use ONLY information present in the runbook. If a section is absent,
   return an empty list for it. Do NOT pad with generic SRE advice.
2. Deduplicate. If the runbook says the same thing two different ways,
   emit it ONCE.
3. Output MUST be valid JSON: no markdown fences (no ```json), no
   trailing commas, no comments, no prose before or after.
4. Output MUST match the exact shape above. No extra keys.
5. If the runbook has no actionable content (e.g. it's a policy doc),
   emit: {"components":[],"error_patterns":[],"fix_actions":[],"pattern_fix_map":[]}
"""

def _truncate(text: str, max_chars: int = 12_000) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    head = text[: int(max_chars * 0.6)]
    tail = text[-int(max_chars * 0.4):]
    return head + "\n\n[…middle truncated…]\n\n" + tail


def enrich_runbook(
    runbook_id: str | int,
    title: str,
    category: str,
    text: str,
    *,
    risk_level: str | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    """
    Extract entities from the runbook text and persist them into the graph.

    Returns a small summary {nodes_created, edges_created, patterns, fixes}
    that the API can surface to the UI. On any failure (Arango offline, LLM
    error, bad JSON) returns a summary with `enabled: False` and the rest
    of the pipeline keeps working.
    """
    if not graph.is_enabled():
        logger.info("Graph disabled — skipping enrichment for runbook %s", runbook_id)
        return {"enabled": False, "reason": "arango_disabled"}

    snippet = _truncate(text)
    if not snippet.strip():
        return {"enabled": True, "reason": "empty_text", "patterns": 0, "fixes": 0}

    # ── 1. Ask the LLM for structured extraction ────────────────────
    try:
        parsed = llm_service.chat_json(
            messages=[
                {"role": "system", "content": _EXTRACTION_PROMPT},
                {"role": "user",   "content": f"Runbook title: {title}\n"
                                              f"Category: {category}\n\n"
                                              f"--- DOCUMENT ---\n{snippet}"},
            ],
            # extraction wants stricter / more deterministic output
            temperature=0.1,
            max_tokens=1500,
        )
    except Exception as e:
        logger.warning("LLM extraction failed for runbook %s: %s", runbook_id, e)
        return {"enabled": True, "reason": f"llm_error: {e}", "patterns": 0, "fixes": 0}

    if not isinstance(parsed, dict) or not parsed:
        return {"enabled": True, "reason": "llm_returned_no_json",
                "patterns": 0, "fixes": 0}

    components     = _clean_list(parsed.get("components"))
    raw_patterns   = parsed.get("error_patterns") or []
    raw_fixes      = _clean_list(parsed.get("fix_actions"))
    pattern_fix    = parsed.get("pattern_fix_map") or []

    # ── 2. Upsert the central runbook node ──────────────────────────
    rb_vid = graph.upsert_runbook(
        runbook_id=runbook_id, title=title, category=category,
        risk_level=risk_level, description=description,
    )

    # ── 3. Components ───────────────────────────────────────────────
    comp_vids: list[str] = []
    for comp in components[:5]:
        vid = graph.upsert_component(comp, category=category)
        comp_vids.append(vid)
        graph.link_runbook_to_component(rb_vid, vid)

    # ── 4. Error patterns ───────────────────────────────────────────
    pattern_vids: list[str] = []
    for raw in raw_patterns[:6]:
        if not isinstance(raw, dict):
            continue
        sig = (raw.get("signature") or "").strip()
        if not sig:
            continue
        kws = [str(k).strip().lower() for k in (raw.get("keywords") or [])
               if isinstance(k, (str, int)) and str(k).strip()]
        kws = list(dict.fromkeys(kws))[:8]
        sample = (raw.get("sample") or "")[:500]
        vid = graph.upsert_error_pattern(signature=sig, sample=sample, keywords=kws)
        pattern_vids.append(vid)
        graph.link_runbook_to_pattern(rb_vid, vid)

    # ── 5. Fix actions ──────────────────────────────────────────────
    fix_vids: list[str] = []
    for order, fix in enumerate(raw_fixes[:10]):
        cleaned = _strip_ordinal(fix)
        if not cleaned:
            continue
        vid = graph.upsert_fix_action(cleaned, kind="manual")
        fix_vids.append(vid)
        graph.link_runbook_to_fix(rb_vid, vid, order=order)

    # ── 6. pattern → fix edges (from LLM's map, fall back to all-to-all) ─
    edges_created = 0
    used_explicit_map = False
    for entry in pattern_fix:
        if not isinstance(entry, dict):
            continue
        pi = entry.get("pattern_index")
        fis = entry.get("fix_indexes") or []
        if not isinstance(pi, int) or not isinstance(fis, list):
            continue
        if not (0 <= pi < len(pattern_vids)):
            continue
        for fi in fis:
            if isinstance(fi, int) and 0 <= fi < len(fix_vids):
                graph.link_pattern_to_fix(pattern_vids[pi], fix_vids[fi], weight=1.0)
                edges_created += 1
                used_explicit_map = True

    # If the model didn't give us a map, conservatively connect each
    # pattern to every fix in the same runbook with a low weight.
    if not used_explicit_map and pattern_vids and fix_vids:
        for p in pattern_vids:
            for f in fix_vids:
                graph.link_pattern_to_fix(p, f, weight=0.5)
                edges_created += 1

    summary = {
        "enabled":       True,
        "runbook_vid":   rb_vid,
        "components":    len(comp_vids),
        "patterns":      len(pattern_vids),
        "fixes":         len(fix_vids),
        "pattern_fix_edges": edges_created,
    }
    logger.info("Graph enrichment for runbook %s: %s", runbook_id, summary)
    return summary


# ──────────────────────────────────────────────────────────────────────
# Incident outcome → graph
# ──────────────────────────────────────────────────────────────────────


def record_incident_outcome(
    *,
    incident_id: str | int,
    pipeline_name: str,
    summary: str,
    confidence: float,
    matched_pattern_signatures: list[str] | None = None,
    applied_fixes: list[str] | None = None,
    success: bool = True,
) -> dict[str, Any]:
    """
    Called after an incident is resolved. Writes:
      - the incident vertex
      - edges to the patterns it matched
      - edges to the fixes that were applied, with a success flag

    These edges are what `recommend_fixes_for_query` uses to RANK
    suggestions on future incidents.
    """
    if not graph.is_enabled():
        return {"enabled": False}

    inc_vid = graph.upsert_incident(incident_id=incident_id,
                                    pipeline_name=pipeline_name,
                                    summary=summary, confidence=confidence)
    # Pipeline-uses-component nothing to infer here; that's done in
    # ingestion / connector layer.

    pattern_edges = 0
    for sig in (matched_pattern_signatures or []):
        # Look up the pattern by slug (signature stable)
        pat_vid = f"error_patterns/{graph._slug(sig)}"
        if graph._upsert_edge("incident_matched", inc_vid, pat_vid,
                              {"score": 1.0}):
            pattern_edges += 1

    fix_edges = 0
    for fix_text in (applied_fixes or []):
        fix_vid = f"fix_actions/{graph._slug(fix_text)}"
        # NOTE: if this fix was never seen in a runbook before, the vertex
        # won't exist. Create it on-the-fly so we still learn from the
        # operator's manual remediation.
        try:
            graph.upsert_fix_action(fix_text, kind="incident-derived")
        except Exception:
            pass
        if graph._upsert_edge("incident_used_fix", inc_vid, fix_vid,
                              {"success": bool(success)}):
            fix_edges += 1

    return {"enabled": True, "incident_vid": inc_vid,
            "pattern_edges": pattern_edges, "fix_edges": fix_edges}


# ──────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────


_ORDINAL_RE = re.compile(r"^\s*(?:\d{1,2}[.)]|[-*•]|step\s*\d+\s*[:.\-])\s*",
                         re.IGNORECASE)


def _strip_ordinal(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return _ORDINAL_RE.sub("", s.strip()).strip()


def _clean_list(x: Any) -> list[str]:
    if not isinstance(x, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in x:
        if not isinstance(item, (str, int, float)):
            continue
        s = str(item).strip()
        if not s or s.lower() in seen:
            continue
        seen.add(s.lower())
        out.append(s)
    return out
