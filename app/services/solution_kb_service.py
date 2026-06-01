"""
Solution Knowledge Base service.

This is the brain of the learning loop. It owns:

  classify(...)             → given an error, is it a KNOWN or NEW error?
                              Returns the matched SolutionPattern (or None) plus
                              a decision the incident pipeline acts on.

  record_occurrence(...)    → upsert a pattern when an incident is diagnosed
                              (creates a NEW pattern or bumps an existing one).

  reinforce(...)            → called when a human ACCEPTS or REJECTS a fix.
                              Increments counters and recomputes confidence.
                              Accepting the same fix repeatedly drives confidence
                              up (requirement #3).

  attach_llm_fix(...)       → store an LLM-generated code change on a pattern.

  ingest_merged_pr(...)     → store a human-merged PR's diff as a HUMAN_PR fix
                              and mark the pattern KNOWN, so the SAME error can
                              be auto-fixed next time (requirement #1 detailed
                              + requirement #2).

Confidence model
----------------
We blend the model's per-incident confidence with the *track record* of human
acceptances using a Wilson-style shrink toward the acceptance rate:

    base       = llm_confidence (0..1)
    accepts, rejects = pattern counters
    trials     = accepts + rejects
    accept_rate = accepts / trials               (when trials > 0)
    # weight grows with evidence, capped so a single accept can't hit 1.0
    w          = trials / (trials + 2)
    confidence = (1 - w) * base + w * accept_rate

So with zero human signal we defer to the model; after several accepted fixes
for the same signature the confidence converges toward ~1.0.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from app.models.solution_models import (
    FixOrigin, SolutionFix, SolutionPattern, SolutionStatus,
)
from app.services import error_signature as sig
from app.core.config import settings

logger = logging.getLogger(__name__)

# A pattern is treated as "known and trusted enough to auto-write code" when
# confidence clears this bar AND it has at least one accepted code fix.
AUTO_FIX_CONFIDENCE = 0.7


@dataclass
class Classification:
    is_known: bool
    pattern: SolutionPattern | None
    signature: str
    error_type: str
    # True when the agent should write code + raise a PR rather than just notify.
    auto_fix: bool
    reason: str


def _recompute_confidence(pattern: SolutionPattern, llm_confidence: float) -> float:
    base = max(0.0, min(1.0, float(llm_confidence or 0.0)))
    accepts = int(pattern.acceptance_count or 0)
    rejects = int(pattern.rejection_count or 0)
    trials = accepts + rejects
    if trials == 0:
        return round(base, 4)
    accept_rate = accepts / trials
    w = trials / (trials + 2)
    blended = (1.0 - w) * base + w * accept_rate
    return round(max(0.0, min(1.0, blended)), 4)


class SolutionKBService:
    # ── classification (known vs new) ──────────────────────────────────
    def classify(
        self,
        db: Session,
        *,
        error_text: str,
        component: str | None,
        llm_confidence: float = 0.0,
    ) -> Classification:
        signature = sig.compute_signature(error_text, component=component)
        error_type = sig.guess_error_type(error_text)

        if not signature:
            return Classification(
                is_known=False, pattern=None, signature="",
                error_type=error_type, auto_fix=False,
                reason="empty error text — cannot fingerprint",
            )

        pattern = (
            db.query(SolutionPattern)
            .filter(SolutionPattern.signature == signature)
            .first()
        )

        if not pattern:
            return Classification(
                is_known=False, pattern=None, signature=signature,
                error_type=error_type, auto_fix=False,
                reason="no matching signature in KB — new error type",
            )

        # Known signature. Decide whether we trust it enough to auto-fix.
        auto_fix = (
            pattern.status == SolutionStatus.KNOWN
            and (pattern.confidence or 0.0) >= AUTO_FIX_CONFIDENCE
            and pattern.acceptance_count >= 1
            and any(f.has_code for f in (pattern.fixes or []))
        )
        reason = (
            f"known signature (seen {pattern.occurrence_count}x, "
            f"{pattern.acceptance_count} accepted, confidence "
            f"{(pattern.confidence or 0):.0%})"
        )
        return Classification(
            is_known=True, pattern=pattern, signature=signature,
            error_type=error_type, auto_fix=auto_fix, reason=reason,
        )

    # ── upsert on diagnosis ────────────────────────────────────────────
    def record_occurrence(
        self,
        db: Session,
        *,
        error_text: str,
        component: str | None,
        category: str,
        root_cause: str,
        fix_summary: str,
        fix_steps: list[str],
        llm_confidence: float,
        support_group: str | None = None,
    ) -> SolutionPattern:
        signature = sig.compute_signature(error_text, component=component)
        error_type = sig.guess_error_type(error_text)
        pattern = (
            db.query(SolutionPattern)
            .filter(SolutionPattern.signature == signature)
            .first()
            if signature else None
        )

        if pattern is None:
            pattern = SolutionPattern(
                signature=signature or f"adhoc-{datetime.utcnow().timestamp()}",
                title=sig.short_title(error_text),
                error_excerpt=(error_text or "")[:2000],
                category=(category or "General"),
                component=component,
                error_type=error_type,
                support_group=support_group,
                root_cause=root_cause or "",
                fix_summary=fix_summary or "",
                fix_steps=fix_steps or [],
                occurrence_count=1,
                acceptance_count=0,
                rejection_count=0,
                status=SolutionStatus.PROPOSED,
            )
            pattern.confidence = _recompute_confidence(pattern, llm_confidence)
            db.add(pattern)
            db.commit()
            db.refresh(pattern)
            logger.info("KB: created NEW pattern #%s (%s)", pattern.id, error_type)
            self._mirror_to_vectors(pattern)
            return pattern

        # Existing pattern — bump occurrence and refresh the diagnosis text if
        # the new one is richer. Don't clobber a human-curated root cause.
        pattern.occurrence_count = (pattern.occurrence_count or 0) + 1
        pattern.last_seen_at = datetime.utcnow()
        if root_cause and len(root_cause) > len(pattern.root_cause or ""):
            pattern.root_cause = root_cause
        if fix_steps and not pattern.fix_steps:
            pattern.fix_steps = fix_steps
        if support_group and not pattern.support_group:
            pattern.support_group = support_group
        pattern.confidence = _recompute_confidence(pattern, llm_confidence)
        db.commit()
        db.refresh(pattern)
        logger.info(
            "KB: matched pattern #%s now seen %sx (conf %.2f)",
            pattern.id, pattern.occurrence_count, pattern.confidence,
        )
        return pattern

    # ── reinforcement on human decision ────────────────────────────────
    def reinforce(
        self,
        db: Session,
        *,
        signature: str | None = None,
        pattern_id: int | None = None,
        accepted: bool,
        llm_confidence: float | None = None,
    ) -> SolutionPattern | None:
        pattern = self._resolve(db, signature=signature, pattern_id=pattern_id)
        if pattern is None:
            return None

        if accepted:
            pattern.acceptance_count = (pattern.acceptance_count or 0) + 1
            pattern.last_accepted_at = datetime.utcnow()
            # First acceptance promotes a PROPOSED pattern to KNOWN.
            if pattern.status == SolutionStatus.PROPOSED:
                pattern.status = SolutionStatus.KNOWN
        else:
            pattern.rejection_count = (pattern.rejection_count or 0) + 1

        base = llm_confidence if llm_confidence is not None else (pattern.confidence or 0.0)
        pattern.confidence = _recompute_confidence(pattern, base)
        db.commit()
        db.refresh(pattern)
        logger.info(
            "KB: reinforced pattern #%s accepted=%s → conf %.2f (%s accepts / %s rejects)",
            pattern.id, accepted, pattern.confidence,
            pattern.acceptance_count, pattern.rejection_count,
        )
        self._mirror_to_vectors(pattern)
        return pattern

    # ── code fixes ─────────────────────────────────────────────────────
    def attach_llm_fix(
        self,
        db: Session,
        *,
        pattern_id: int,
        file_path: str | None,
        new_content: str | None,
        diff: str | None,
        explanation: str,
        language: str | None = None,
    ) -> SolutionFix:
        fix = SolutionFix(
            pattern_id=pattern_id,
            origin=FixOrigin.LLM,
            file_path=file_path,
            new_content=new_content,
            diff=diff,
            explanation=explanation or "",
            language=language,
        )
        db.add(fix)
        db.commit()
        db.refresh(fix)
        return fix

    def ingest_merged_pr(
        self,
        db: Session,
        *,
        signature: str | None = None,
        pattern_id: int | None = None,
        pr_url: str,
        pr_number: int | None,
        merged_by: str | None,
        diff: str,
        file_path: str | None = None,
        new_content: str | None = None,
        explanation: str = "",
    ) -> SolutionPattern | None:
        """Close the loop: a human merged a PR for this error. Store the real
        change so the SAME signature becomes auto-fixable next time."""
        pattern = self._resolve(db, signature=signature, pattern_id=pattern_id)
        if pattern is None:
            return None

        fix = SolutionFix(
            pattern_id=pattern.id,
            origin=FixOrigin.HUMAN_PR,
            file_path=file_path,
            new_content=new_content,
            diff=diff,
            explanation=explanation or f"Ingested from merged PR {pr_url}",
            pr_url=pr_url,
            pr_number=pr_number,
            merged_by=merged_by,
        )
        db.add(fix)

        # A merged human PR is a strong positive signal.
        pattern.acceptance_count = (pattern.acceptance_count or 0) + 1
        pattern.last_accepted_at = datetime.utcnow()
        pattern.status = SolutionStatus.KNOWN
        pattern.confidence = _recompute_confidence(pattern, pattern.confidence or 0.6)
        db.commit()
        db.refresh(pattern)
        logger.info(
            "KB: ingested merged PR for pattern #%s → now auto-fixable=%s",
            pattern.id, pattern.is_auto_fixable,
        )
        self._mirror_to_vectors(pattern)
        return pattern

    # ── stats for the metrics page ─────────────────────────────────────
    def stats(self, db: Session) -> dict[str, Any]:
        total = db.query(SolutionPattern).count()
        known = db.query(SolutionPattern).filter(
            SolutionPattern.status == SolutionStatus.KNOWN
        ).count()
        auto = sum(
            1 for p in db.query(SolutionPattern).all() if p.is_auto_fixable
        )
        accepts = db.query(SolutionFix).filter(
            SolutionFix.origin == FixOrigin.HUMAN_PR
        ).count()
        return {
            "patterns_total": total,
            "patterns_known": known,
            "patterns_auto_fixable": auto,
            "human_prs_ingested": accepts,
        }

    # ── internals ──────────────────────────────────────────────────────
    def _resolve(
        self, db: Session, *, signature: str | None, pattern_id: int | None,
    ) -> SolutionPattern | None:
        if pattern_id:
            return db.query(SolutionPattern).filter(
                SolutionPattern.id == pattern_id
            ).first()
        if signature:
            return db.query(SolutionPattern).filter(
                SolutionPattern.signature == signature
            ).first()
        return None

    def _mirror_to_vectors(self, pattern: SolutionPattern) -> None:
        """Best-effort: keep the vector incidents collection enriched with the
        latest solution narrative so RAG retrieval surfaces it (requirement
        #2: continuously enrich the KB). Reuses the existing incidents
        collection rather than adding a new one, so no schema change."""
        try:
            from app.services.rag_service import rag_service  # noqa: PLC0415
            narrative_fix = pattern.fix_summary or "\n".join(pattern.fix_steps or [])
            rag_service.store_incident(
                incident_id=f"kb-{pattern.id}",
                pipeline_name=pattern.title,
                error_log=pattern.error_excerpt or pattern.title,
                root_cause=pattern.root_cause,
                suggested_fix=narrative_fix,
                confidence=pattern.confidence,
                risk_tier="Low" if (pattern.confidence or 0) >= 0.7 else "Medium",
            )
        except Exception:
            logger.debug("KB vector mirror failed for pattern #%s", pattern.id, exc_info=True)


solution_kb_service = SolutionKBService()
