"""
ArangoDB layer for the runbook knowledge graph.

──────────────────────────────────────────────────────────────────────
What we model
──────────────────────────────────────────────────────────────────────

Vertices:
  runbooks            — uploaded SOPs (mirrors the SQL Runbook row by id)
  error_patterns      — distilled error signatures (e.g. "KeyError on join")
  fix_actions         — atomic remediation steps extracted from runbooks
  components          — services / tools (ADF, Databricks, Glue, Git, S3...)
  incidents           — resolved pipeline incidents
  pipelines           — the user's monitored pipelines

Edges (all directed unless noted):
  runbook_addresses     runbooks         → error_patterns
  runbook_has_step      runbooks         → fix_actions     (attrs: order)
  runbook_applies_to    runbooks         → components
  pattern_uses_fix      error_patterns   → fix_actions     (attrs: weight)
  incident_matched      incidents        → error_patterns  (attrs: score)
  incident_used_fix     incidents        → fix_actions     (attrs: success)
  pattern_similar_to    error_patterns   → error_patterns  (attrs: cosine)
  pipeline_uses         pipelines        → components

Why a graph (vs just Chroma):
  Chroma is great at "find me text chunks similar to this error log".
  But a fix recommendation often depends on STRUCTURE Chroma can't
  represent cleanly:

    "Among fixes whose pattern matches THIS error, which ones have
     been used by past incidents on THIS component with the highest
     success rate?"

  That query traverses 3 edges. In a graph DB it's one AQL statement
  and a few ms. In Chroma it's impossible without a join layer.

  The diagnosis flow now uses BOTH:
    1. Chroma RAG → similar past incidents + raw runbook chunks (semantic)
    2. Arango     → ranked fix_actions via graph traversal (structural)

──────────────────────────────────────────────────────────────────────
Defensive init
──────────────────────────────────────────────────────────────────────

init_arango() is idempotent: it creates DB / collections / graph only
if missing, and is safe to call on every app startup. If Arango is
unreachable the service degrades silently — all read helpers return
empty results so the diagnosis flow falls back to Chroma-only.
"""
from __future__ import annotations

import hashlib
import logging
import re
import time
from typing import Any, Iterable

from app.core.config import settings

logger = logging.getLogger(__name__)

# Lazy import — don't blow up at import time if python-arango isn't installed.
try:
    from arango import ArangoClient            # python-arango
    from arango.exceptions import ArangoError  # type: ignore
    _ARANGO_AVAILABLE = True
except ImportError:  # pragma: no cover
    ArangoClient = None  # type: ignore
    ArangoError = Exception  # type: ignore
    _ARANGO_AVAILABLE = False


# ── Collection names ──────────────────────────────────────────────────

VERTEX_COLLECTIONS = (
    "runbooks",
    "error_patterns",
    "fix_actions",
    "components",
    "incidents",
    "pipelines",
)

EDGE_COLLECTIONS = (
    "runbook_addresses",
    "runbook_has_step",
    "runbook_applies_to",
    "pattern_uses_fix",
    "incident_matched",
    "incident_used_fix",
    "pattern_similar_to",
    "pipeline_uses",
)

GRAPH_NAME = "runbook_graph"

_EDGE_DEFINITIONS = [
    {"edge_collection": "runbook_addresses",
     "from_vertex_collections": ["runbooks"],
     "to_vertex_collections":   ["error_patterns"]},
    {"edge_collection": "runbook_has_step",
     "from_vertex_collections": ["runbooks"],
     "to_vertex_collections":   ["fix_actions"]},
    {"edge_collection": "runbook_applies_to",
     "from_vertex_collections": ["runbooks"],
     "to_vertex_collections":   ["components"]},
    {"edge_collection": "pattern_uses_fix",
     "from_vertex_collections": ["error_patterns"],
     "to_vertex_collections":   ["fix_actions"]},
    {"edge_collection": "incident_matched",
     "from_vertex_collections": ["incidents"],
     "to_vertex_collections":   ["error_patterns"]},
    {"edge_collection": "incident_used_fix",
     "from_vertex_collections": ["incidents"],
     "to_vertex_collections":   ["fix_actions"]},
    {"edge_collection": "pattern_similar_to",
     "from_vertex_collections": ["error_patterns"],
     "to_vertex_collections":   ["error_patterns"]},
    {"edge_collection": "pipeline_uses",
     "from_vertex_collections": ["pipelines"],
     "to_vertex_collections":   ["components"]},
]


# ── Module-level connection ───────────────────────────────────────────

_db = None                  # the StandardDatabase handle
_enabled: bool = False      # flips True after a successful init


def _slug(text: str, maxlen: int = 80) -> str:
    """Stable, AQL-safe key from arbitrary text."""
    if not text:
        return "unknown"
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", text.strip().lower())
    s = re.sub(r"_+", "_", s).strip("_")
    if len(s) > maxlen:
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
        s = f"{s[:maxlen]}_{digest}"
    return s or "unknown"


def is_enabled() -> bool:
    return _enabled


def get_db():
    """Return the live db handle or None if Arango isn't ready."""
    return _db if _enabled else None


# ── Bootstrap ─────────────────────────────────────────────────────────


def init_arango() -> bool:
    """
    Connect to ArangoDB and ensure schema. Returns True if the graph is
    ready to use; False (and logs a warning) if anything went wrong —
    in which case the rest of the app should fall back to Chroma-only.
    """
    global _db, _enabled

    if not _ARANGO_AVAILABLE:
        logger.warning("python-arango not installed; graph features disabled. "
                       "Install with: pip install python-arango")
        return False

    if not getattr(settings, "ARANGO_ENABLED", True):
        logger.info("ARANGO_ENABLED=False; graph features disabled by config.")
        return False

    url       = getattr(settings, "ARANGO_URL",      "http://localhost:8529")
    user      = getattr(settings, "ARANGO_USER",     "root")
    password  = getattr(settings, "ARANGO_PASSWORD", "")
    dbname    = getattr(settings, "ARANGO_DB",       "pipeline_graph")

    try:
        client = ArangoClient(hosts=url)
        sys_db = client.db("_system", username=user, password=password)

        if not sys_db.has_database(dbname):
            sys_db.create_database(dbname)
            logger.info("ArangoDB: created database '%s'", dbname)

        _db = client.db(dbname, username=user, password=password)

        for name in VERTEX_COLLECTIONS:
            if not _db.has_collection(name):
                _db.create_collection(name)
                logger.info("ArangoDB: created vertex collection '%s'", name)

        for name in EDGE_COLLECTIONS:
            if not _db.has_collection(name):
                _db.create_collection(name, edge=True)
                logger.info("ArangoDB: created edge collection '%s'", name)

        if not _db.has_graph(GRAPH_NAME):
            _db.create_graph(GRAPH_NAME, edge_definitions=_EDGE_DEFINITIONS)
            logger.info("ArangoDB: created graph '%s'", GRAPH_NAME)

        _enabled = True
        logger.info("ArangoDB ready at %s db=%s", url, dbname)
        return True

    except Exception as exc:
        _enabled = False
        _db = None
        logger.warning("ArangoDB init failed (%s); falling back to Chroma-only", exc)
        return False


# ── Write helpers ─────────────────────────────────────────────────────


def _upsert_vertex(collection: str, key: str, doc: dict[str, Any]) -> str:
    """Insert-or-update a vertex by deterministic key. Returns _id."""
    if not _enabled:
        return ""
    col = _db.collection(collection)
    doc = {**doc, "_key": key, "updated_at": int(time.time())}
    if col.has(key):
        col.update({"_key": key, **doc})
    else:
        doc["created_at"] = doc["updated_at"]
        col.insert(doc)
    return f"{collection}/{key}"


def _upsert_edge(collection: str, from_id: str, to_id: str,
                 attrs: dict[str, Any] | None = None) -> str:
    """Idempotent edge upsert keyed by (from, to)."""
    if not _enabled or not from_id or not to_id:
        return ""
    col = _db.collection(collection)
    key = hashlib.sha1(f"{from_id}->{to_id}".encode()).hexdigest()[:24]
    doc = {"_key": key, "_from": from_id, "_to": to_id, **(attrs or {})}
    if col.has(key):
        col.update(doc)
    else:
        col.insert(doc)
    return f"{collection}/{key}"


# Public surface — one function per kind of node/edge --------------------

def upsert_component(name: str, category: str | None = None) -> str:
    return _upsert_vertex("components", _slug(name),
                          {"name": name, "category": category or ""})


def upsert_runbook(runbook_id: str | int, title: str, category: str,
                   risk_level: str | None = None,
                   description: str | None = None) -> str:
    return _upsert_vertex("runbooks", _slug(f"rb_{runbook_id}"),
                          {"runbook_id": str(runbook_id),
                           "title": title,
                           "category": category,
                           "risk_level": risk_level or "Medium",
                           "description": (description or "")[:1000]})


def upsert_error_pattern(signature: str, sample: str | None = None,
                         keywords: list[str] | None = None) -> str:
    return _upsert_vertex("error_patterns", _slug(signature),
                          {"signature": signature,
                           "sample": (sample or "")[:500],
                           "keywords": keywords or []})


def upsert_fix_action(text: str, kind: str = "manual") -> str:
    return _upsert_vertex("fix_actions", _slug(text),
                          {"text": text[:500], "kind": kind})


def upsert_pipeline(name: str) -> str:
    return _upsert_vertex("pipelines", _slug(name), {"name": name})


def upsert_incident(incident_id: str | int, pipeline_name: str,
                    summary: str | None = None,
                    confidence: float | None = None) -> str:
    return _upsert_vertex("incidents", _slug(f"inc_{incident_id}"),
                          {"incident_id": str(incident_id),
                           "pipeline_name": pipeline_name,
                           "summary": (summary or "")[:500],
                           "confidence": float(confidence or 0.0)})


def link_runbook_to_pattern(runbook_vid: str, pattern_vid: str) -> str:
    return _upsert_edge("runbook_addresses", runbook_vid, pattern_vid)


def link_runbook_to_fix(runbook_vid: str, fix_vid: str, order: int = 0) -> str:
    return _upsert_edge("runbook_has_step", runbook_vid, fix_vid, {"order": order})


def link_runbook_to_component(runbook_vid: str, component_vid: str) -> str:
    return _upsert_edge("runbook_applies_to", runbook_vid, component_vid)


def link_pattern_to_fix(pattern_vid: str, fix_vid: str, weight: float = 1.0) -> str:
    return _upsert_edge("pattern_uses_fix", pattern_vid, fix_vid, {"weight": weight})


def link_incident_to_pattern(incident_vid: str, pattern_vid: str,
                             score: float = 1.0) -> str:
    return _upsert_edge("incident_matched", incident_vid, pattern_vid, {"score": score})


def link_incident_to_fix(incident_vid: str, fix_vid: str,
                         success: bool = True) -> str:
    return _upsert_edge("incident_used_fix", incident_vid, fix_vid,
                        {"success": bool(success)})


# ── Read / traversal helpers — the part the diagnosis flow uses ──────


def recommend_fixes_for_query(
    query: str,
    *,
    component: str | None = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Given a free-text error / log snippet (and optionally a component name),
    walk the graph to recommend the top-K fix_actions, ranked by:

      1. # of error_patterns matched by query keywords  (broader = better)
      2. # of past incidents that used the fix and succeeded
      3. weighted edge from pattern → fix

    Returns:
      [{
        "fix": <str>,
        "fix_id": <_id>,
        "score": <float>,
        "matched_patterns": [<signature>, ...],
        "supporting_runbooks": [<title>, ...],
        "success_count": <int>,
        "failure_count": <int>,
      }, ...]

    Falls back to [] if Arango is offline.
    """
    if not _enabled:
        return []

    keywords = _extract_keywords(query)
    if not keywords:
        return []

    # AQL: find patterns whose `keywords` overlap with the query keywords,
    # then traverse to fixes, then aggregate incident outcomes.
    aql = """
    LET kws = @kws
    LET matched_patterns = (
      FOR p IN error_patterns
        LET hits = LENGTH(INTERSECTION(p.keywords, kws))
        FILTER hits > 0
        RETURN { p, hits }
    )

    LET fix_scores = (
      FOR mp IN matched_patterns
        FOR v, e IN 1..1 OUTBOUND mp.p._id pattern_uses_fix
          COLLECT fix_id = v._id, fix_text = v.text
              AGGREGATE base = SUM(mp.hits * (e.weight || 1.0))
          RETURN { fix_id, fix_text, base, patterns: UNIQUE(matched_patterns[*].p.signature) }
    )

    FOR fs IN fix_scores
      // outcomes: how often have incidents used this fix?
      LET successes = LENGTH(
        FOR i, ie IN 1..1 INBOUND fs.fix_id incident_used_fix
          FILTER ie.success == true
          RETURN 1
      )
      LET failures = LENGTH(
        FOR i, ie IN 1..1 INBOUND fs.fix_id incident_used_fix
          FILTER ie.success == false
          RETURN 1
      )
      // supporting runbooks — keep the full vertex so we can also
      // traverse to components for the optional filter
      LET runbook_vs = (
        FOR rb, re IN 1..1 INBOUND fs.fix_id runbook_has_step
          RETURN rb
      )
      LET runbook_titles = runbook_vs[*].title
      // optional component filter: does ANY supporting runbook apply
      // to a component whose name matches @component (case-insensitive)?
      LET applies_here = @component == null ? true : (
        LENGTH(
          FOR rb IN runbook_vs
            FOR c, ce IN 1..1 OUTBOUND rb._id runbook_applies_to
              FILTER LOWER(c.name) == LOWER(@component)
              LIMIT 1
              RETURN 1
        ) > 0
      )
      FILTER applies_here
      LET score = fs.base + successes * 2 - failures
      SORT score DESC
      LIMIT @top_k
      RETURN {
        fix_id: fs.fix_id,
        fix:    fs.fix_text,
        score:  score,
        matched_patterns:    fs.patterns,
        supporting_runbooks: UNIQUE(runbook_titles),
        success_count:       successes,
        failure_count:       failures
      }
    """
    try:
        cur = _db.aql.execute(aql, bind_vars={
            "kws": keywords,
            "component": component,
            "top_k": int(top_k),
        })
        return list(cur)
    except Exception as e:
        logger.warning("graph recommend_fixes_for_query failed: %s", e)
        return []


def get_runbook_subgraph(runbook_id: str | int) -> dict[str, Any]:
    """For visualization: return nodes + edges around one runbook."""
    if not _enabled:
        return {"nodes": [], "edges": []}
    vid = f"runbooks/{_slug(f'rb_{runbook_id}')}"
    aql = """
    LET center = DOCUMENT(@vid)
    FILTER center != null
    LET neighbours = (
      FOR v, e IN 1..1 ANY @vid GRAPH @graph
        RETURN { v, e }
    )
    RETURN {
      center: center,
      neighbours: neighbours
    }
    """
    try:
        cur = _db.aql.execute(aql, bind_vars={"vid": vid, "graph": GRAPH_NAME})
        result = next(iter(cur), None)
        if not result:
            return {"nodes": [], "edges": []}

        nodes_by_id: dict[str, dict[str, Any]] = {}
        edges: list[dict[str, Any]] = []
        c = result["center"]
        nodes_by_id[c["_id"]] = {"id": c["_id"], "label": c.get("title") or c.get("name") or c["_key"],
                                  "type": c["_id"].split("/")[0]}
        for n in result["neighbours"]:
            v, e = n["v"], n["e"]
            nodes_by_id[v["_id"]] = {
                "id": v["_id"],
                "label": v.get("title") or v.get("name") or v.get("text") or v.get("signature") or v["_key"],
                "type": v["_id"].split("/")[0],
            }
            edges.append({
                "from": e["_from"],
                "to":   e["_to"],
                "label": e["_id"].split("/")[0],
            })
        return {"nodes": list(nodes_by_id.values()), "edges": edges}
    except Exception as e:
        logger.warning("graph get_runbook_subgraph failed: %s", e)
        return {"nodes": [], "edges": []}


def graph_stats() -> dict[str, Any]:
    """Counts per collection — used by the Metrics page and the UI badge."""
    if not _enabled:
        return {"enabled": False, "counts": {}}
    counts: dict[str, int] = {}
    try:
        for name in VERTEX_COLLECTIONS + EDGE_COLLECTIONS:
            counts[name] = _db.collection(name).count()
    except Exception as e:
        logger.warning("graph_stats failed: %s", e)
    return {"enabled": True, "counts": counts}


# ── Keyword extraction ───────────────────────────────────────────────


_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "on", "in", "at", "to", "of", "for", "and", "or", "but", "if",
    "this", "that", "with", "as", "by", "from", "it", "its",
    "error", "exception", "failed", "failure", "trace", "traceback",
    "line", "file", "info", "warn", "warning", "debug",
}


def _extract_keywords(text: str, max_kw: int = 12) -> list[str]:
    """
    Cheap keyword extractor: alphanum tokens of length >= 3, lowercased,
    minus stopwords, deduplicated, ordered by first appearance. The graph
    matches keywords against error_pattern.keywords lists.
    """
    if not text:
        return []
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", text)
    seen: set[str] = set()
    out: list[str] = []
    for tok in tokens:
        low = tok.lower()
        if low in _STOPWORDS or low in seen:
            continue
        seen.add(low)
        out.append(low)
        if len(out) >= max_kw:
            break
    return out
