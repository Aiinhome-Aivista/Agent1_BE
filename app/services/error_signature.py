"""
Error signature utilities.

The known-vs-new classification (requirement #1) needs a *stable* fingerprint
for an error so that two occurrences of "the same" failure map to the same
SolutionPattern row even though the raw text differs in run ids, timestamps,
row counts, file line numbers, etc.

`normalize()` strips out the volatile bits; `compute_signature()` hashes the
normalized text. `guess_error_type()` does a light classification used for
support-group routing and code-gen prompting.

This is deliberately dependency-free and deterministic.
"""
from __future__ import annotations

import hashlib
import re

# Order matters: apply the most specific replacements first.
_SCRUBBERS: list[tuple[re.Pattern[str], str]] = [
    # ISO timestamps & dates
    (re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?"), "<TS>"),
    (re.compile(r"\d{4}-\d{2}-\d{2}"), "<DATE>"),
    (re.compile(r"\b\d{2}:\d{2}:\d{2}\b"), "<TIME>"),
    # UUIDs / hex blobs / hashes
    (re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"), "<UUID>"),
    (re.compile(r"\b[0-9a-fA-F]{12,}\b"), "<HEX>"),
    # file:line references → keep the file, drop the line number
    (re.compile(r"(line)\s+\d+", re.IGNORECASE), r"\1 <N>"),
    (re.compile(r"(:)\d+(\b)"), r"\1<N>\2"),
    # absolute-ish paths → keep just the basename
    (re.compile(r"(/[\w.\-]+)+/([\w.\-]+)"), r"<PATH>/\2"),
    # standalone numbers (row counts, attempt ids, exit codes)
    (re.compile(r"\b\d[\d,_.]*\b"), "<N>"),
    # collapse whitespace
    (re.compile(r"\s+"), " "),
]

# Lightweight error-type detector. Extend freely.
_TYPE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bKeyError\b"), "KeyError"),
    (re.compile(r"\b(schema|column).*(not found|drift|mismatch|changed)", re.IGNORECASE), "SchemaDrift"),
    (re.compile(r"\b(timed?\s?out|timeout)\b", re.IGNORECASE), "Timeout"),
    (re.compile(r"\b(connection|connect).*(refused|reset|failed)", re.IGNORECASE), "ConnectionError"),
    (re.compile(r"\b(permission|denied|unauthor|forbidden|403|401)\b", re.IGNORECASE), "AuthError"),
    (re.compile(r"\b(out of memory|oom|memoryerror)\b", re.IGNORECASE), "OutOfMemory"),
    (re.compile(r"\b(rate\s?limit|429|throttl)", re.IGNORECASE), "RateLimit"),
    (re.compile(r"\b(null|none).*(pointer|type|value)", re.IGNORECASE), "NullValue"),
    (re.compile(r"\b(file ?not ?found|no such file|404)\b", re.IGNORECASE), "NotFound"),
    (re.compile(r"\b(dependency|module|import).*(error|not found)", re.IGNORECASE), "DependencyError"),
]


def normalize(text: str) -> str:
    """Strip volatile tokens so equivalent errors collapse to one string."""
    s = (text or "").strip()
    if not s:
        return ""
    # Use only the most relevant lines: prefer ERROR/CRITICAL/exception lines.
    lines = [ln for ln in s.splitlines() if ln.strip()]
    salient = [
        ln for ln in lines
        if re.search(r"error|exception|failed|critical|traceback", ln, re.IGNORECASE)
    ]
    candidate = "\n".join(salient[-8:]) if salient else "\n".join(lines[-8:])
    candidate = candidate.lower()
    for pat, repl in _SCRUBBERS:
        candidate = pat.sub(repl, candidate)
    return candidate.strip()[:1000]


def compute_signature(text: str, *, component: str | None = None) -> str:
    """Return a short stable hash for the normalized error.

    `component` is folded in so the *same* generic error on two different
    connectors doesn't accidentally share a solution (e.g. a Timeout on
    Databricks vs on ADF may need different fixes).
    """
    norm = normalize(text)
    if not norm:
        return ""
    basis = f"{(component or '').lower().strip()}|{norm}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:32]


def guess_error_type(text: str) -> str:
    s = text or ""
    for pat, label in _TYPE_PATTERNS:
        if pat.search(s):
            return label
    return "Unknown"


def short_title(text: str, *, fallback: str = "Pipeline failure") -> str:
    """Build a compact human title from the first salient error line."""
    for ln in (text or "").splitlines():
        if re.search(r"error|exception|failed", ln, re.IGNORECASE):
            return ln.strip()[:120] or fallback
    first = (text or "").strip().splitlines()
    return (first[0][:120] if first else fallback) or fallback
