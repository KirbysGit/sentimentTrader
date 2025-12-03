"""
purpose:
  pull historical OHLCV data for every ticker that survived stage 2 so the
  pipeline can tie reddit sentiment to actual price action.

what this module does:
  - accepts a ticker list + date range from upstream configs
  - filters out blocked/non-equity symbols via config blacklist
  - looks up each ticker (and its alias chain) through yfinance with retries
  - validates/cleans the dataframe (columns, gaps, basic features)
  - saves both raw CSV + metadata JSON into `RAW_DIR/stock_data`

how it fits:
  stage 3/4 (market data). After RedditDataProcessor identifies tickers,
  StockDataCollector ensures we have price history for backtesting and
  modeling. diagnostics from earlier stages drives which symbols we fetch.

future expansion:
  - add intraday intervals or multi-source fallbacks (Polygon, Tiingo, etc.)
  - capture adjusted close, dividends, splits for factor modeling
  - maintain a cache of already-downloaded tickers to skip duplicates
"""

import json
import time
import logging
import pandas as pd
import yfinance as yf
from datetime import datetime
from typing import Optional, Dict, List
from colorama import Fore, Style
from src.utils.path_config import RAW_DIR
from src.utils.config import STOCK_DATA_BLACKLIST
from src.utils.ticker_aliases import get_alias_chain
import os

# setup logging.
logger = logging.getLogger(__name__)

# stock data collector class.
class StockDataCollector:

    # -----------------------------------------------------------------------------------------------

    """collects historical stock data with enhanced error handling and validation."""
    
    def __init__(self, data_dir=None):
        """initialize the stock data collector."""
        self.data_dir = data_dir or (RAW_DIR / "stock_data")
        os.makedirs(self.data_dir, exist_ok=True)
        self.symbols: List[str] = []
        self.start_date: Optional[str] = None
        self.end_date: Optional[str] = None
        self.available_data: Dict[str, pd.DataFrame] = {}

    # -----------------------------------------------------------------------------------------------

    def configure(self, symbols: List[str], start_date: str, end_date: str) -> None:
        """
        configure tickers + date range supplied from upstream pipeline stages.
        """
        unique_symbols = list({s.upper() for s in symbols if s})
        if not unique_symbols:
            raise ValueError("stock data collector.configure(): no symbols provided.")

        filtered = [
            s for s in unique_symbols
            if s not in STOCK_DATA_BLACKLIST and s.isalpha() and 1 <= len(s) <= 5
        ]
        dropped = sorted(set(unique_symbols) - set(filtered))
        if not filtered:
            raise ValueError("stock data collector.configure(): all symbols filtered out.")

        if dropped:
            logger.info(f"[stock data collector] skipping blocked/non-equity symbols: {dropped}")

        self.symbols = filtered
        self.start_date = start_date
        self.end_date = end_date

        logger.debug(
            f"[stock data collector] configured for {len(self.symbols)} tickers "
            f"({self.start_date} → {self.end_date})"
        )
    
    # -----------------------------------------------------------------------------------------------

    def collect_data(self) -> Dict[str, pd.DataFrame]:
        """collect historical data for all symbols with retry mechanism"""

        if not self.symbols:
            raise ValueError("stock data collector: symbols not configured. call configure() first.")
        if not self.start_date or not self.end_date:
            raise ValueError("stock data collector: start/end dates not configured. call configure() first.")

        all_data = {}
        self.available_data = {}
        print(f"{Fore.CYAN}===== stage 3: stock data collection ====={Style.RESET_ALL}")
        print(
            f"{Fore.CYAN}fetching data for {len(self.symbols)} tickers "
            f"(window: {self.start_date} → {self.end_date}){Style.RESET_ALL}"
        )
        print(f"{Fore.CYAN}{self.symbols}{Style.RESET_ALL}")

        failed_symbols = []

        # iterate through each symbol.
        for symbol in self.symbols:
            # get the alias chain.
            alias_chain = get_alias_chain(symbol)
            # initialize the dataframe and source symbol.
            df = None
            source_symbol = None

            # iterate through each candidate.
            for candidate in alias_chain:
                # fetch the data with retry.
                candidate_df = self._fetch_with_retry(candidate)
                if candidate_df is None or candidate_df.empty:
                    continue

                # validate and clean the data.
                candidate_df = self._validate_and_clean_data(candidate_df, candidate)
                if len(candidate_df) < 60:
                    continue

                # set the dataframe and source symbol.
                df = candidate_df
                source_symbol = candidate
                break

            if df is None:
                print(f"{Fore.RED}✗ {symbol}: no usable data (aliases tried: {alias_chain}){Style.RESET_ALL}")
                failed_symbols.append(symbol)
                continue

            # set the dataframe and available data.
            all_data[symbol] = df
            self.available_data[symbol] = df
            if source_symbol and source_symbol != symbol:
                print(
                    f"{Fore.GREEN}✓ {symbol}: collected via {source_symbol} ({len(df)} days){Style.RESET_ALL}"
                )
            else:
                print(f"{Fore.GREEN}✓ {symbol}: collected ({len(df)} days){Style.RESET_ALL}")
            self._save_data(df, symbol, source_symbol=source_symbol)

        if failed_symbols:
            print(
                f"{Fore.YELLOW}⚠ no usable data for {len(failed_symbols)} tickers: {failed_symbols}{Style.RESET_ALL}"
            )
        else:
            print(f"{Fore.GREEN}✓ all configured tickers collected successfully{Style.RESET_ALL}")
        
        return all_data
    
    # -----------------------------------------------------------------------------------------------
    
    def _fetch_with_retry(self, symbol: str, max_retries: int = 3) -> Optional[pd.DataFrame]:
        """fetch data with retry mechanism."""
        for attempt in range(max_retries):
            try:
                stock = yf.Ticker(symbol)
                # fetch the data.
                df = stock.history(start=self.start_date, end=self.end_date, interval='1d')

                # check if the dataframe is empty.
                if df.empty:
                    # log the error.
                    logger.debug(f"Attempt {attempt + 1}: empty data for {symbol}")
                    time.sleep(2 ** attempt)
                    continue

                return df

            except Exception as e:
                logger.debug(f"Attempt {attempt + 1} failed for {symbol}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                continue

        return None
    
    # -----------------------------------------------------------------------------------------------

    def _validate_and_clean_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """validate & clean the collected data."""
        df = df.copy()
        
        # reset index to make date a column.
        df.reset_index(inplace=True)
        
        # ensure all required columns exist.
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns for {symbol}: {missing_columns}")
        
        # remove any duplicate dates.
        df.drop_duplicates(subset=['Date'], keep='last', inplace=True)
        
        # sort by date.
        df.sort_values('Date', inplace=True)
        
        # handle missing values.
        for col in ['Open', 'High', 'Low', 'Close']:
            if df[col].isnull().any():
                logger.debug(f"Filling Missing {col} Values for {symbol}")
                df[col] = df[col].ffill().bfill()
        
        # fill missing volume with 0.
        df['Volume'] = df['Volume'].fillna(0)
        
        # add basic metrics.
        df['Returns'] = df['Close'].pct_change()
        df['High-Low'] = df['High'] - df['Low']
        df['High-PrevClose'] = df['High'] - df['Close'].shift(1)
        df['Low-PrevClose'] = df['Low'] - df['Close'].shift(1)
        
        # calculate trading days between dates.
        df['trading_gap'] = df['Date'].diff().dt.days
        
        # log any gaps in trading days.
        gaps = df[df['trading_gap'] > 1]
        if not gaps.empty:
            logger.debug(f"Found {len(gaps)} trading gaps for {symbol}")
        
        return df
    
    # -----------------------------------------------------------------------------------------------

    def _save_data(self, df: pd.DataFrame, symbol: str, source_symbol: Optional[str] = None) -> None:
        """save the collected data."""
        output_file = self.data_dir / f"{symbol}_stock_data.csv"
        df.to_csv(output_file, index=False)
        
        # save metadata.
        metadata = {
            'symbol': symbol,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'source_symbol': (source_symbol or symbol),
            'rows': len(df),
            'trading_days': len(df),
            'data_columns': df.columns.tolist(),
            'collection_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_version': '1.0'
        }
        
        metadata_file = self.data_dir / f"{symbol}_stock_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        logger.debug(f"stock data saved to: {output_file}")
        logger.debug(f"metadata saved to: {metadata_file}")

    def get_available_tickers(self) -> List[str]:
        """return list of tickers that produced valid stock data."""
        return list(self.available_data.keys())