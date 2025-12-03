"""
purpose:
  provide a tiny, dependency-free sentiment helper for Stage 2 so we can
  tag every reddit post with a quick polarity score.

what this does:
  - tokenizes cleaned post text
  - counts occurrences of finance slang stored in config-level lexicons
  - returns (positive hits - negative hits) as a lightweight score

how it fits:
  RedditDataProcessor imports this class, feeds cleaned text, and
  stores the returned float alongside tickers + engagement.
  Later we can swap in FinBERT without touching the processor API.

dependencies:
  - `src.utils.config` for the positive/negative lexicon sets
"""

from src.utils.config import (
    POSITIVE_SENTIMENT_WORDS,
    NEGATIVE_SENTIMENT_WORDS,
)


class SentimentScorer:
    """
    Lightweight lexicon-based sentiment engine. Keeps logic isolated so we can
    swap in FinBERT or ensemble scoring later without touching processors.
    """

    def __init__(self):
        self.positive = set(POSITIVE_SENTIMENT_WORDS)
        self.negative = set(NEGATIVE_SENTIMENT_WORDS)

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
