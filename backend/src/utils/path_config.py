from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Assumes this file is in src/utils/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Main directories
DATA_DIR = PROJECT_ROOT / "data"
SRC_DIR = PROJECT_ROOT / "src"

# Data subdirectories
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
# Other directories (created on-demand when needed)
DEBUG_DIR = DATA_DIR / "debug"
REFERENCES_DIR = DATA_DIR / "references"
MODEL_DIR = DATA_DIR / "models"
TICKERS_DIR = DATA_DIR / "tickers"
TICKER_GENERAL_DIR = TICKERS_DIR / "ticker_general"
TICKER_SENTIMENT_DIR = TICKERS_DIR / "ticker_sentiment"
REDDIT_DATA_DIR = RAW_DIR / "reddit_data"
STOCK_DATA_DIR = RAW_DIR / "stock_data"
PROCESSED_REDDIT_DIR = PROCESSED_DIR / "reddit_data"
RESULTS_DIR = PROJECT_ROOT / "results"

# Reference files paths
VALID_TICKERS_FILE = REFERENCES_DIR / "valid_tickers.csv"
ENTITY_CACHE_FILE = REFERENCES_DIR / "entity_cache.json"

# Note: Other directories (debug, references, models, tickers, results) 
# are created on-demand when needed by the pipeline components

# Only auto-create essential directories (raw and processed)
# Other directories are created on-demand when needed
_REQUIRED_DIRECTORIES = [
    RAW_DIR,
    PROCESSED_DIR
]

def _ensure_directories_exist():
    """Silently create required directories if they don't exist."""
    for directory in _REQUIRED_DIRECTORIES:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {str(e)}")

# Create only essential directories when module is imported
_ensure_directories_exist()
