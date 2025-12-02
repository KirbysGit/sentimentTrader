"""Standalone sentiment scoring helper for Reddit posts."""


class SentimentScorer:
    """
    Lightweight lexicon-based sentiment engine. Keeps logic isolated so we can
    swap in FinBERT or ensemble scoring later without touching processors.
    """

    def __init__(self):
        self.positive = {
            "up", "bull", "bullish", "gain", "green",
            "beat", "pump", "moon", "mooning",
            "strong", "win", "positive", "profit",
            "surge", "soar", "rocket", "pump"
        }

        self.negative = {
            "down", "bear", "bearish", "loss",
            "dump", "crash", "bad", "miss",
            "weak", "negative", "red", "selloff",
            "plunge", "tank", "bleed"
        }

    def score(self, text: str) -> float:
        """
        Return a simple sentiment score (positive count - negative count).
        """
        if not text:
            return 0.0

        words = text.lower().split()
        pos = sum(1 for w in words if w in self.positive)
        neg = sum(1 for w in words if w in self.negative)
        return float(pos - neg)
