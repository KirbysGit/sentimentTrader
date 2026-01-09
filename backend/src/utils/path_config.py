from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# assumes we're in 
proj_root = Path(__file__).resolve().parents[3]
backend_dir = proj_root / "backend"

# Main directories
data_dir = backend_dir / "data"
src_dir = backend_dir / "src"
tickers_dir = backend_dir / "tickers"

env_path = proj_root / ".env"

# Data subdirectories
raw_dir = data_dir / "raw"
processed_dir = data_dir / "processed"
reasoning_dir = data_dir / "reasoning"

# data subdirectory.
seen_path = data_dir / "seen_post_ids.json"
last_seen_created_utc_path = data_dir / "last_seen_created_utc.json"

raw_reddit_dir = raw_dir / "reddit"
raw_stocks_dir = raw_dir / "stocks"
raw_stocktwits_dir = raw_dir / "stocktwits"
processed_reddit_by_day_dir = processed_dir / "reddit" / "by_day"
processed_metrics_dir = processed_dir / "metrics"
processed_stocktwits_by_day_dir = processed_dir / "stocktwits" / "by_day"

# Note: Other directories (debug, references, models, tickers, results) 
# are created on-demand when needed by the pipeline components

# Only auto-create essential directories (raw and processed)
# Other directories are created on-demand when needed
_REQUIRED_DIRECTORIES = [
    raw_dir,
    processed_dir,
    raw_reddit_dir,
    raw_stocks_dir,
    raw_stocktwits_dir,
    processed_reddit_by_day_dir,
    processed_metrics_dir,
    processed_stocktwits_by_day_dir,
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
