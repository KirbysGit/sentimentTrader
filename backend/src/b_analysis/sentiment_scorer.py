"""
purpose:
  provide a sentiment helper for Stage 2 so we can tag every reddit post with a
  quick polarity score. defaults to a lexicon, but optionally upgrades to
  FinBERT (or any Hugging Face finance sentiment model) when transformers is
  available.

what this does:
  - lazily initializes a HF text-classification pipeline if enabled
  - otherwise falls back to the dependency-free lexicon counts
  - returns a float sentiment score (positive → +1, neutral → 0, negative → -1)
"""

from typing import Optional

from src.utils.config import (
    POSITIVE_SENTIMENT_WORDS,
    NEGATIVE_SENTIMENT_WORDS,
)


class SentimentScorer:
    """
    Finance-aware sentiment engine with optional FinBERT backend.

    Usage:
        scorer = SentimentScorer(use_transformer=True)
        score = scorer.score("NVDA earnings crushed expectations.")

    Notes:
        - Setting `use_transformer=True` requires `transformers` and `torch`
          to be installed (`pip install torch transformers`).
        - If transformers cannot be imported or the model download fails,
          the scorer automatically falls back to the lexicon strategy.
    """

    TRANSFORMER_MODEL = "ProsusAI/finbert"

    def __init__(self, use_transformer: bool = True):
        self.positive = set(POSITIVE_SENTIMENT_WORDS)
        self.negative = set(NEGATIVE_SENTIMENT_WORDS)
        self.use_transformer = use_transformer
        self._pipeline = None  # lazy-init to avoid startup penalty

    # ------------------------------------------------------------------
    def score(self, text: str) -> float:
        """Return sentiment score for the provided text."""
        if not text:
            return 0.0

        text = text.strip()
        if not text:
            return 0.0

        if self.use_transformer:
            score = self._score_with_transformer(text)
            if score is not None:
                return score

        # fallback lexicon score
        words = text.lower().split()
        pos = sum(1 for w in words if w in self.positive)
        neg = sum(1 for w in words if w in self.negative)
        return float(pos - neg)

    # ------------------------------------------------------------------
    def _score_with_transformer(self, text: str) -> Optional[float]:
        """Run FinBERT if available, otherwise return None."""
        try:
            pipeline = self._get_pipeline()
        except Exception:
            return None

        if pipeline is None:
            return None

        try:
            result = pipeline(text, truncation=True)
        except Exception:
            return None

        if not result:
            return None

        # FinBERT returns [{'label': 'positive|neutral|negative', 'score': prob}]
        top = result[0]
        label = top.get("label", "").lower()
        score = float(top.get("score", 0.0))

        if label == "positive":
            return score
        if label == "negative":
            return -score
        return 0.0  # neutral

    # ------------------------------------------------------------------
    def _get_pipeline(self):
        """Lazy import transformers and build the HF pipeline."""
        if self._pipeline is not None:
            return self._pipeline

        try:
            from transformers import pipeline  # type: ignore
        except Exception:
            self._pipeline = None
            return None

        try:
            self._pipeline = pipeline(
                task="text-classification",
                model=self.TRANSFORMER_MODEL,
                tokenizer=self.TRANSFORMER_MODEL,
                return_all_scores=False,
            )
        except Exception:
            self._pipeline = None

        return self._pipeline
