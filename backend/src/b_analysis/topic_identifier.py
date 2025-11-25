import re
import logging
import pandas as pd
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from tqdm import tqdm
from colorama import Fore, Style, init
from src.utils.path_config import RAW_DIR, DEBUG_DIR, REFERENCES_DIR, VALID_TICKERS_FILE, PROCESSED_REDDIT_DIR
import requests
import numpy as np
from pathlib import Path

# Initialize colorama
init()

logger = logging.getLogger(__name__)

# Constants for filtering
POTENTIALLY_AMBIGUOUS_TICKERS = {
    'SEE', 'OPEN', 'REAL', 'DAY', 'TOP', 'KEY', 'TRUE', 'SAFE', 'GAIN', 'LOT', 'TURN',
    'MORE', 'LIKE', 'JUST', 'CASH', 'PROP', 'STEP', 'ELSE', 'JUNE', 'NEXT', 'GOOD',
    'BEST', 'WELL', 'FAST', 'FREE', 'LIVE', 'PLAY', 'STAY', 'MOVE', 'MIND', 'LIFE',
    'PEAK', 'FUND', 'HUGE', 'NICE', 'EASY', 'BEAT', 'HOPE', 'CARE', 'MAIN', 'RIDE'
}

class TopicIdentifier:
    def __init__(self, data_dir=None):
        """Initialize the Topic Identifier."""
        print(f"\n{Fore.CYAN}Initializing Topic Identifier...{Style.RESET_ALL}")
        self.data_dir = data_dir or REFERENCES_DIR  # Update to use references directory
        self.ticker_pattern = re.compile(r'\$([A-Z]{1,5})')
        
        # Set up ticker analysis directory
        self.ticker_analysis_dir = RAW_DIR / "ticker_analysis"
        self.ticker_analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracking dictionaries
        self.ticker_mentions = defaultdict(int)
        self.ticker_engagement = defaultdict(float)
        self.ticker_confidence = defaultdict(list)
        self.ticker_contexts = defaultdict(list)
        self.ticker_min_confidence = defaultdict(float)
        self.ticker_max_confidence = defaultdict(float)
        
        # Common words to filter out
        self.common_words = {
            'THE', 'AND', 'FOR', 'ARE', 'WAS', 'YOU', 'HAS', 'HAD', 'HIS', 'HER', 'ITS', 'OUR', 'THEIR',
            'FROM', 'THIS', 'THAT', 'WITH', 'WHICH', 'WHEN', 'WHERE', 'WHAT', 'WHY', 'HOW', 'WHO'
        }
        
        # Common finance terms that might be mistaken for tickers
        self.finance_terms = {
            'STOCK', 'MARKET', 'BUY', 'SELL', 'HOLD', 'SHORT', 'LONG', 'CALL', 'PUT', 'OPTION',
            'PRICE', 'SHARE', 'TRADE', 'TRADING', 'INVEST', 'INVESTING', 'MONEY', 'CASH', 'GAIN',
            'LOSS', 'PROFIT', 'LOSS', 'RISK', 'SAFE', 'BEAR', 'BULL', 'PUMP', 'DUMP', 'MOON'
        }
        
        # Load valid tickers from NASDAQ and NYSE
        self.valid_tickers = self._load_valid_tickers()
        
        # Track ambiguous ticker stats
        self.ambiguous_stats = defaultdict(lambda: {
            'total_mentions': 0,
            'high_confidence_mentions': 0,
            'contexts': [],
            'subreddits': set()
        })
        
        print(f"{Fore.GREEN}✓ Topic Identifier initialized successfully{Style.RESET_ALL}\n")
    
    def _load_valid_tickers(self) -> set:
        """Load valid stock tickers from NASDAQ and NYSE."""
        print(f"{Fore.CYAN}Loading valid tickers from exchanges...{Style.RESET_ALL}")
        try:
            # Try to load from local cache first
            if VALID_TICKERS_FILE.exists():
                print(f"{Fore.YELLOW}Loading tickers from cache at {VALID_TICKERS_FILE}...{Style.RESET_ALL}")
                df = pd.read_csv(VALID_TICKERS_FILE)
                tickers = set(df['Symbol'].str.upper().tolist())
                print(f"{Fore.GREEN}✓ Loaded {len(tickers)} tickers from cache{Style.RESET_ALL}")
                return tickers

            # If no cache, try to download from NASDAQ
            print(f"{Fore.YELLOW}Downloading NASDAQ tickers...{Style.RESET_ALL}")
            nasdaq_url = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=10000"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(nasdaq_url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                tickers = set()
                
                # Extract tickers from response
                for row in data.get('data', {}).get('table', {}).get('rows', []):
                    symbol = row.get('symbol', '').upper()
                    if len(symbol) <= 5 and symbol.isalpha():
                        tickers.add(symbol)
                
                # Save to cache in references directory
                REFERENCES_DIR.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({'Symbol': list(tickers)}).to_csv(VALID_TICKERS_FILE, index=False)
                print(f"{Fore.GREEN}✓ Downloaded and cached {len(tickers)} tickers to {VALID_TICKERS_FILE}{Style.RESET_ALL}")
                return tickers
            
            raise Exception("Failed to download ticker list")
            
        except Exception as e:
            print(f"{Fore.RED}✗ Error loading valid tickers: {str(e)}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Using fallback ticker list...{Style.RESET_ALL}")
            return {
                'NVDA', 'AMD', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'INTC', 'IBM',
                'JPM', 'BAC', 'WFC', 'GS', 'V', 'MA', 'PYPL', 'SQ', 'COIN', 'HOOD',
                'NFLX', 'DIS', 'WMT', 'TGT', 'COST', 'HD', 'LOW', 'MCD', 'SBUX', 'PEP',
                'KO', 'PG', 'JNJ', 'PFE', 'MRK', 'ABBV', 'UNH', 'CVS', 'WBA', 'LLY',
                'XOM', 'CVX', 'COP', 'BP', 'SHEL', 'PBR', 'BABA', 'JD', 'PDD', 'BIDU',
                'NIO', 'XPEV', 'LI', 'RIVN', 'LCID', 'F', 'GM', 'TM', 'HMC', 'TSM',
                'ASML', 'QCOM', 'AVGO', 'TXN', 'MU', 'LRCX', 'AMAT', 'KLAC', 'ADI', 'MRVL'
            }
    
    def extract_tickers(self, text: str) -> List[str]:
        """Extract ticker symbols from text with enhanced filtering."""
        if pd.isna(text):
            return []
        
        # Find $TICKER format
        tickers = self.ticker_pattern.findall(text.upper())
        
        # Find standalone tickers (3-5 capital letters)
        standalone = re.findall(r'\b[A-Z]{3,5}\b', text.upper())
        tickers.extend(standalone)
        
        # Filter out common words and finance terms
        tickers = [
            t for t in tickers 
            if self._validate_ticker(t)
        ]
        
        return list(set(tickers))
    
    def _load_latest_ticker_analysis(self, analysis_type="daily"):
        """Load the most recent ticker analysis results."""
        try:
            analysis_files = list(self.ticker_analysis_dir.glob(f"ticker_analysis_{analysis_type}_*.csv"))
            if not analysis_files:
                return None
            
            # Get most recent file
            latest_file = max(analysis_files, key=lambda x: x.stat().st_mtime)
            
            # Read CSV file
            df = pd.read_csv(latest_file)
            
            # Verify data is from today
            file_date = datetime.strptime(latest_file.stem.split('_')[-1], '%Y%m%d').date()
            if file_date == datetime.now().date():
                print(f"{Fore.GREEN}✓ Loaded ticker analysis from {latest_file.name}{Style.RESET_ALL}")
                
                # Convert to expected format for compatibility
                return {
                    'scores': dict(zip(df['ticker'], df['total_engagement'])),
                    'mentions': dict(zip(df['ticker'], df['mentions'])),
                    'sentiment': dict(zip(df['ticker'], df['avg_confidence'])),
                    'debug_info': {
                        ticker: {
                            'mentions': row['mentions'],
                            'engagement': row['total_engagement'],
                            'avg_confidence': row['avg_confidence'],
                            'sentiment_mean': row['avg_confidence'],
                            'sentiment_std': 0,  # Not tracked in CSV format
                            'confidence_classes': {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}  # Not tracked in CSV format
                        }
                        for ticker, row in df.iterrows()
                    }
                }
            else:
                print(f"{Fore.YELLOW}Warning: Most recent ticker analysis is from {file_date}{Style.RESET_ALL}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading ticker analysis: {str(e)}")
            return None
    
    def _validate_ticker(self, ticker: str) -> bool:
        """Enhanced ticker validation."""
        if not ticker or len(ticker) < 2:  # Prevent single-letter tickers
            return False
            
        if ticker in self.common_words or ticker in self.finance_terms:
            return False
            
        if ticker in POTENTIALLY_AMBIGUOUS_TICKERS and len(ticker) < 3:
            return False
            
        return ticker in self.valid_tickers
    
    def calculate_ticker_scores(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate scores for each ticker based on mentions, engagement, and confidence class."""
        if df.empty:
            return {}
        
        print(f"\n{Fore.CYAN}Pipeline Health Dashboard{Style.RESET_ALL}")
        print(f"Posts: {len(df)} | Relevant: {sum(df['is_relevant'])} | Date Range: {df['date'].min()} to {df['date'].max()}")
        
        # Filter for relevant posts with ticker mentions
        relevant_df = df[df['is_relevant'] & df['tickers'].notna()]
        if relevant_df.empty:
            print(f"{Fore.YELLOW}No relevant ticker mentions found{Style.RESET_ALL}")
            return {}
        
        ticker_data = []
        ticker_mentions = Counter()
        ticker_engagement = Counter()
        ticker_confidence = defaultdict(list)
        ticker_contexts = defaultdict(list)
        
        for _, row in tqdm(relevant_df.iterrows(), total=len(relevant_df), desc=f"{Fore.YELLOW}Processing posts{Style.RESET_ALL}"):
            for ticker in row['tickers']:
                # Enhanced ticker validation
                if not self._validate_ticker(ticker):
                    continue
                
                # Track mentions and confidence
                ticker_mentions[ticker] += 1
                confidence = row.get('ticker_confidence', 0)
                ticker_confidence[ticker].append(confidence)
                
                # Track contexts (limit to first 200 chars)
                context = row.get('cleaned_text', '')[:200]
                if context and context not in ticker_contexts[ticker]:
                    ticker_contexts[ticker].append(context)
                
                # Calculate weighted engagement based on confidence class
                confidence_class = row.get('confidence_class', 'LOW')
                class_multiplier = {
                    'HIGH': 1.5,    # Increased multiplier for HIGH confidence
                    'MEDIUM': 1.0,  # Base multiplier for MEDIUM confidence
                    'LOW': 0.5      # Reduced multiplier for LOW confidence
                }.get(confidence_class, 0.5)
                
                engagement = (
                    row.get('score', 0) + 
                    row.get('num_comments', 0)
                ) * confidence * class_multiplier
                
                ticker_engagement[ticker] += engagement
        
        # Prepare data for CSV
        for ticker in ticker_mentions.keys():
            # Skip tickers with only 1 mention
            if ticker_mentions[ticker] < 2:
                logger.warning(f"Ticker {ticker} excluded: only {ticker_mentions[ticker]} mention(s)")
                continue
                
            # Check for sentiment files
            daily_file = Path(PROCESSED_REDDIT_DIR) / f"{ticker}_daily_sentiment.csv"
            detailed_file = Path(PROCESSED_REDDIT_DIR) / f"{ticker}_detailed_sentiment.csv"
            
            if not (daily_file.exists() or detailed_file.exists()):
                logger.warning(f"Ticker {ticker} excluded: no sentiment files found")
                continue
            
            confidences = ticker_confidence[ticker]
            contexts = ticker_contexts[ticker]
            
            data = {
                'ticker': ticker,
                'mentions': ticker_mentions[ticker],
                'total_engagement': ticker_engagement[ticker],
                'avg_confidence': np.mean(confidences) if confidences else 0,
                'passed_filters': True,  # Since we already filtered in _validate_ticker
                'is_common_word': ticker.upper() in self.common_words,
                'is_valid_ticker': ticker in self.valid_tickers,
                'is_ambiguous': ticker in POTENTIALLY_AMBIGUOUS_TICKERS,  # Direct check against the set
                'example_contexts': '; '.join(contexts[:3]),  # Take up to 3 example contexts
                'min_confidence': min(confidences) if confidences else 0,
                'max_confidence': max(confidences) if confidences else 0
            }
            ticker_data.append(data)
        
        # Convert to DataFrame and save as CSV
        df_output = pd.DataFrame(ticker_data)
        if df_output.empty:
            logger.warning("No tickers met the minimum requirements for trending analysis")
            return {}
            
        df_output = df_output.sort_values('total_engagement', ascending=False)
        
        # Save to CSV
        timestamp = datetime.now().strftime('%Y%m%d')
        csv_file = self.ticker_analysis_dir / f"ticker_analysis_daily_{timestamp}.csv"
        df_output.to_csv(csv_file, index=False)
        print(f"{Fore.GREEN}✓ Saved ticker analysis to {csv_file}{Style.RESET_ALL}")
        
        # Return scores for compatibility with existing code
        return dict(zip(df_output['ticker'], df_output['total_engagement']))
    
    def identify_trending_topics(self, df: pd.DataFrame, min_mentions: int = 2) -> Dict[str, float]:
        """Identify trending tickers from processed Reddit posts."""
        try:
            if df.empty:
                return {}
            
            # Try to load cached analysis first if df is None
            if df is None:
                cached_analysis = self._load_latest_ticker_analysis()
                if cached_analysis:
                    return cached_analysis['scores']
            
            # Calculate ticker scores
            ticker_scores = self.calculate_ticker_scores(df)
            
            # Filter out tickers with low mentions or missing sentiment
            filtered_scores = {}
            for ticker, score in ticker_scores.items():
                # Check for sentiment files
                daily_file = Path(PROCESSED_REDDIT_DIR) / f"{ticker}_daily_sentiment.csv"
                detailed_file = Path(PROCESSED_REDDIT_DIR) / f"{ticker}_detailed_sentiment.csv"
                
                if daily_file.exists() or detailed_file.exists():
                    filtered_scores[ticker] = score
                else:
                    logger.warning(f"Ticker {ticker} excluded from trending: no sentiment files found")
            
            # Sort by score
            trending = dict(sorted(
                filtered_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
            
            print(f"\n{Fore.CYAN}Trending Analysis Complete:{Style.RESET_ALL}")
            print(f"{Fore.GREEN}✓ Identified {len(trending)} trending tickers with sentiment data{Style.RESET_ALL}")
            return trending
            
        except Exception as e:
            print(f"{Fore.RED}✗ Error identifying trending topics: {str(e)}{Style.RESET_ALL}")
            return {}
    
    def get_trending_tickers(self, df: pd.DataFrame = None, top_n: int = 10) -> List[str]:
        """Get the top N trending tickers from processed data or cache, ensuring they have sentiment files."""
        # Try to get trending tickers from data or cache
        if df is None:
            cached_analysis = self._load_latest_ticker_analysis()
            if cached_analysis:
                trending = cached_analysis['scores']
            else:
                print(f"{Fore.RED}No cached ticker analysis found and no data provided{Style.RESET_ALL}")
                return []
        else:
            trending = self.identify_trending_topics(df)
        
        # Filter tickers based on sentiment file existence and validity
        valid_tickers = []
        for ticker in trending.keys():
            if not self._validate_ticker(ticker):
                continue
                
            # Check for sentiment files
            daily_file = Path(PROCESSED_REDDIT_DIR) / f"{ticker}_daily_sentiment.csv"
            detailed_file = Path(PROCESSED_REDDIT_DIR) / f"{ticker}_detailed_sentiment.csv"
            
            if daily_file.exists() or detailed_file.exists():
                valid_tickers.append(ticker)
            else:
                logger.warning(f"Ticker {ticker} excluded from trending: no sentiment files found")
        
        # Sort by engagement score and return top N
        sorted_tickers = sorted(
            valid_tickers,
            key=lambda t: trending[t],
            reverse=True
        )
        
        selected_tickers = sorted_tickers[:top_n]
        if selected_tickers:
            print(f"\n{Fore.GREEN}Selected {len(selected_tickers)} trending tickers with sentiment data{Style.RESET_ALL}")
        else:
            print(f"\n{Fore.YELLOW}No trending tickers found with sentiment data{Style.RESET_ALL}")
        
        return selected_tickers

def main():
    """Test the TopicIdentifier."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    print(f"\n{Fore.CYAN}=== Reddit Trending Ticker Analysis ==={Style.RESET_ALL}")
    
    # Initialize identifier
    identifier = TopicIdentifier()
    
    # Load and process test data
    test_file = RAW_DIR / "reddit_data" / "processed_reddit.csv"
    if test_file.exists():
        df = pd.read_csv(test_file)
        trending = identifier.identify_trending_topics(df)
        
        if trending:
            print(f"\n{Fore.CYAN}Top Trending Tickers:{Style.RESET_ALL}")
            for ticker, score in list(trending.items())[:10]:
                print(f"{Fore.GREEN}{ticker}: {score:.2f}{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}No trending tickers found{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}No processed data found for testing{Style.RESET_ALL}")

if __name__ == "__main__":
    main() 