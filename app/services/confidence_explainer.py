"""
Confidence explainer.

The UI shows a single "Confidence 90%" number. The brief asks: "why the
confidence level is high — it will be explained in detail." This service turns
the inputs that drive the score into a transparent breakdown the UI can render:

  - LLM self-confidence              (the model's own certainty)
  - Knowledge-base track record      (how often this signature was seen +
                                      how many times a human accepted the fix)
  - Runbook / RAG match strength     (did uploaded docs directly cover it)
  - Verified code fix present        (an accepted, committed fix exists)
  - Model-supplied rationale         (free-text reasons from the LLM)

Each factor carries a short label, a detail string, and a 0..1 contribution
weight so the UI can show bars. `level` buckets the final score.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from app.models.solution_models import SolutionPattern


@dataclass
class Factor:
    label: str
    detail: str
    contribution: float           # 0..1 — how strongly this lifts confidence
    polarity: str = "positive"    # "positive" | "negative" | "neutral"


@dataclass
class ConfidenceExplanation:
    score: float
    level: str                    # "High" | "Medium" | "Low"
    headline: str
    factors: list[Factor] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "level": self.level,
            "headline": self.headline,
            "factors": [asdict(f) for f in self.factors],
        }


def _level(score: float) -> str:
    if score >= 0.7:
        return "High"
    if score >= 0.4:
        return "Medium"
    return "Low"


def build(
    *,
    llm_confidence: float,
    final_confidence: float,
    pattern: SolutionPattern | None,
    is_known: bool,
    error_type: str | None = None,
    runbook_top_similarity: float | None = None,
    llm_rationale: list[str] | None = None,
) -> ConfidenceExplanation:
    factors: list[Factor] = []

    # 1. LLM self-confidence.
    factors.append(Factor(
        label="Model certainty",
        detail=f"The diagnosis model rated its own analysis at {llm_confidence:.0%}.",
        contribution=max(0.0, min(1.0, llm_confidence)),
        polarity="positive" if llm_confidence >= 0.5 else "neutral",
    ))

    # 2. Knowledge-base track record (the big lever for "why so high").
    if pattern is not None:
        acc = pattern.acceptance_count or 0
        rej = pattern.rejection_count or 0
        occ = pattern.occurrence_count or 1
        if acc > 0:
            factors.append(Factor(
                label="Proven fix history",
                detail=(
                    f"This exact error has been seen {occ}× and its fix was "
                    f"accepted by a human {acc}× (rejected {rej}×). Each "
                    f"acceptance raises confidence."
                ),
                contribution=min(1.0, acc / (acc + 2)),
                polarity="positive",
            ))
        else:
            factors.append(Factor(
                label="Recognised pattern",
                detail=(
                    f"The error signature matches a known pattern seen {occ}×, "
                    f"but no human has accepted its fix yet — confidence is "
                    f"capped until one does."
                ),
                contribution=min(0.5, occ / (occ + 4)),
                polarity="neutral",
            ))
        if rej > acc and rej > 0:
            factors.append(Factor(
                label="Past rejections",
                detail=f"This fix was rejected {rej}× — that pulls confidence down.",
                contribution=min(1.0, rej / (rej + 2)),
                polarity="negative",
            ))
    elif is_known:
        factors.append(Factor(
            label="Recognised pattern",
            detail="The error matches a known signature in the knowledge base.",
            contribution=0.5, polarity="positive",
        ))
    else:
        factors.append(Factor(
            label="New error type",
            detail=(
                "No matching signature in the knowledge base yet — this is a "
                "first occurrence, so the agent only proposes analysis until a "
                "human-approved fix teaches it."
            ),
            contribution=0.2, polarity="neutral",
        ))

    # 3. Verified code fix present.
    if pattern is not None and any(getattr(f, "has_code", False) for f in (pattern.fixes or [])):
        has_human = any(
            getattr(f, "origin", None)
            and str(getattr(f.origin, "value", f.origin)) == "HUMAN_PR"
            for f in (pattern.fixes or [])
        )
        factors.append(Factor(
            label="Verified code fix on file",
            detail=(
                "A concrete code change is attached"
                + (" (ingested from a merged human PR)" if has_human else "")
                + " and can be reused / raised as a PR automatically."
            ),
            contribution=0.85 if has_human else 0.6,
            polarity="positive",
        ))

    # 4. Runbook / RAG match.
    if runbook_top_similarity is not None and runbook_top_similarity > 0:
        factors.append(Factor(
            label="Runbook coverage",
            detail=(
                f"Uploaded runbooks matched this error at "
                f"{runbook_top_similarity:.0%} similarity, grounding the fix in "
                f"documented procedure."
            ),
            contribution=max(0.0, min(1.0, runbook_top_similarity)),
            polarity="positive" if runbook_top_similarity >= 0.5 else "neutral",
        ))

    # 5. Model-supplied free-text reasons.
    for r in (llm_rationale or [])[:4]:
        r = (r or "").strip()
        if r:
            factors.append(Factor(
                label="Model reasoning",
                detail=r[:240],
                contribution=0.0,
                polarity="neutral",
            ))

    level = _level(final_confidence)
    if level == "High":
        headline = (
            "High confidence: the diagnosis is backed by a recognised pattern "
            "with an accepted fix and/or strong runbook coverage."
        )
    elif level == "Medium":
        headline = (
            "Medium confidence: the model is reasonably sure but the fix has "
            "limited acceptance history — review before applying."
        )
    else:
        headline = (
            "Low confidence: this looks new or contested. Treat the suggestion "
            "as a starting point and approve a fix to teach the agent."
        )

    return ConfidenceExplanation(
        score=final_confidence, level=level, headline=headline, factors=factors,
    )
