"""
confidence_scorer.py

Combines extractor/linker evidence into a richer confidence estimate + prose
reason. This stays lightweight/heuristic for now so we can iterate quickly
before introducing learned models.
"""

from __future__ import annotations

from typing import Dict, List, Any, Tuple


class ConfidenceScorer:
    """Heuristic confidence aggregator for ticker mentions."""

    def __init__(
        self,
        finance_term_bonus: float = 0.02,
        max_finance_bonus: float = 0.10,
        peer_bonus: float = 0.05,
        context_bonus: float = 0.04,
        boost_bonus: float = 0.03,
    ):
        self.finance_term_bonus = finance_term_bonus
        self.max_finance_bonus = max_finance_bonus
        self.peer_bonus = peer_bonus
        self.context_bonus = context_bonus
        self.boost_bonus = boost_bonus

    # ------------------------------------------------------------------
    def score(self, ticker: str, base_score: float, evidence: Dict[str, Any]) -> Tuple[float, str]:
        """Return a calibrated confidence + short explanation string."""
        score = base_score or 0.0
        notes: List[str] = []
        evidence = evidence or {}
        extractor_meta = evidence.get("extractor") or {}
        linker_meta = evidence.get("linker") or {}
        boosts = evidence.get("boosts") or []

        # finance verbs/keywords near the ticker
        finance_terms = extractor_meta.get("finance_terms") or []
        if finance_terms:
            bonus = min(len(finance_terms) * self.finance_term_bonus, self.max_finance_bonus)
            score += bonus
            notes.append(f"finance terms: {self._short_list(finance_terms)} (+{bonus:.2f})")

        # explicit context keywords
        context_terms = extractor_meta.get("context_terms") or linker_meta.get("matched_terms") or []
        if context_terms:
            score += self.context_bonus
            notes.append(f"context keywords: {self._short_list(context_terms)} (+{self.context_bonus:.2f})")

        # peer ticker support
        if extractor_meta.get("peer_hit"):
            peer_symbol = extractor_meta.get("peer_ticker")
            score += self.peer_bonus
            peer_label = f" via {peer_symbol}" if peer_symbol else ""
            notes.append(f"peer ticker context{peer_label} (+{self.peer_bonus:.2f})")

        # boosts applied downstream
        if boosts:
            bonus = min(len(boosts) * self.boost_bonus, self.boost_bonus * 3)
            score += bonus
            notes.append(f"boosts: {self._short_list(boosts)} (+{bonus:.2f})")

        # default reason if nothing triggered
        if not notes and extractor_meta.get("accept_reason"):
            notes.append(extractor_meta["accept_reason"])

        final_score = self._clamp(score)
        summary = "; ".join(notes) if notes else "baseline acceptance"
        return round(final_score, 3), summary

    # ------------------------------------------------------------------
    @staticmethod
    def _short_list(items: List[str], limit: int = 3) -> str:
        """Return comma-separated subset for compact logging."""
        unique = []
        for item in items:
            if item not in unique:
                unique.append(item)
        if not unique:
            return ""
        display = unique[:limit]
        if len(unique) > limit:
            display.append("…")
        return ", ".join(display)

    @staticmethod
    def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
        return max(lower, min(upper, value))

