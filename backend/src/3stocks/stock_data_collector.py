# src / data_collection / stock_data_collector.py

# Description : This file contains the StockDataCollector class, which is used to collect historical stock data.

# Imports.
import json
import time
import logging
import pandas as pd
import yfinance as yf
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime, timedelta
from src.utils.path_config import RAW_DIR
import os

# Setup Logging.
logger = logging.getLogger(__name__)

# Stock Data Collector Class.
class StockDataCollector:

    # -----------------------------------------------------------------------------------------------

    """Collects Historical Stock Data with Enhanced Error Handling and Validation."""
    
    def __init__(self, data_dir=None):
        """Initialize the Stock Data Collector."""
        self.data_dir = data_dir or (RAW_DIR / "stock_data")
        os.makedirs(self.data_dir, exist_ok=True)
    
    # -----------------------------------------------------------------------------------------------

    def collect_data(self) -> Dict[str, pd.DataFrame]:
        """Collect Historical Data for All Symbols w/ Retry Mechanism"""

        # Initialize Dictionary to Store All Data.
        all_data = {}

        # Collect Data for Each Symbol.
        for symbol in self.symbols:
            try:
                logger.info(f"Collecting Data for {symbol} from {self.start_date} to {self.end_date}")
                df = self._fetch_with_retry(symbol)
                
                if df is not None and not df.empty:
                    # Validate and Clean the Data.
                    df = self._validate_and_clean_data(df, symbol)
                    
                    if len(df) >= 60:
                        all_data[symbol] = df
                        self._save_data(df, symbol)
                        logger.info(f"Successfully Collected {len(df)} Days of Data for {symbol}")
                    else:
                        logger.error(f"Insufficient Data for {symbol}: Only {len(df)} Days Available")
                else:
                    logger.error(f"No Data Retrieved for {symbol}")
                
            except Exception as e:
                logger.error(f"Error Collecting Data for {symbol}: {str(e)}")
                continue
        
        return all_data
    
    # -----------------------------------------------------------------------------------------------
    
    def _fetch_with_retry(self, symbol: str, max_retries: int = 3) -> Optional[pd.DataFrame]:
        """Fetch Data with Retry Mechanism."""
        for attempt in range(max_retries):
            try:
                stock = yf.Ticker(symbol)
                df = stock.history(start=self.start_date, end=self.end_date, interval='1d')
                
                if df.empty:
                    logger.warning(f"Attempt {attempt + 1}: Empty Data Received for {symbol}")
                    time.sleep(2 ** attempt)
                    continue
                
                return df
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} Failed for {symbol}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                continue
        
        return None
    
    # -----------------------------------------------------------------------------------------------

    def _validate_and_clean_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Validate and Clean the Collected Data."""
        df = df.copy()
        
        # Reset Index to Make Date a Column.
        df.reset_index(inplace=True)
        
        # Ensure All Required Columns Exist.
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns for {symbol}: {missing_columns}")
        
        # Remove Any Duplicate Dates.
        df.drop_duplicates(subset=['Date'], keep='last', inplace=True)
        
        # Sort by Date.
        df.sort_values('Date', inplace=True)
        
        # Handle Missing Values.
        for col in ['Open', 'High', 'Low', 'Close']:
            if df[col].isnull().any():
                logger.warning(f"Filling Missing {col} Values for {symbol}")
                df[col] = df[col].ffill().bfill()
        
        # Fill Missing Volume with 0.
        df['Volume'] = df['Volume'].fillna(0)
        
        # Add Basic Metrics.
        df['Returns'] = df['Close'].pct_change()
        df['High-Low'] = df['High'] - df['Low']
        df['High-PrevClose'] = df['High'] - df['Close'].shift(1)
        df['Low-PrevClose'] = df['Low'] - df['Close'].shift(1)
        
        # Calculate Trading Days Between Dates.
        df['trading_gap'] = df['Date'].diff().dt.days
        
        # Log Any Gaps in Trading Days.
        gaps = df[df['trading_gap'] > 1]
        if not gaps.empty:
            logger.warning(f"Found {len(gaps)} Trading Gaps for {symbol}")
            for _, row in gaps.iterrows():
                logger.warning(f"Gap of {row['trading_gap']} days before {row['Date']}")
        
        return df
    
    # -----------------------------------------------------------------------------------------------

    def _save_data(self, df: pd.DataFrame, symbol: str) -> None:
        """Save the Collected Data."""
        output_file = self.data_dir / f"{symbol}_stock_data.csv"
        df.to_csv(output_file, index=False)
        
        # Save Metadata.
        metadata = {
            'symbol': symbol,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'rows': len(df),
            'trading_days': len(df),
            'data_columns': df.columns.tolist(),
            'collection_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_version': '1.0'
        }
        
        metadata_file = self.data_dir / f"{symbol}_stock_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        logger.info(f"Stock data saved to: {output_file}")
        logger.info(f"Metadata saved to: {metadata_file}")

    def fetch_stock_data(self, ticker, start_date, end_date):
        """
        Fetch stock data for a given ticker and date range.
        
        Args:
            ticker (str): Stock ticker symbol
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            pd.DataFrame: Stock data with OHLCV and additional metrics
        """
        try:
            # Fetch data from Yahoo Finance
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            
            if df.empty:
                logger.warning(f"No data found for {ticker}")
                return None
                
            # Calculate additional metrics
            df['Daily_Return'] = df['Close'].pct_change()
            df['Volume_MA'] = df['Volume'].rolling(window=5).mean()
            df['Price_MA'] = df['Close'].rolling(window=5).mean()
            df['RSI'] = self._calculate_rsi(df['Close'])
            
            # Reset index to make Date a column
            df = df.reset_index()
            df.rename(columns={'index': 'Date'}, inplace=True)
            
            # Add ticker column
            df['Ticker'] = ticker
            
            logger.info(f"Successfully collected data for {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            return None
            
    def _calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))


# -----------------------------------------------------------------------------------------------

def main():
    """Main function to test the StockDataCollector."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Initialize collector
    collector = StockDataCollector()
    
    # Test with multiple tickers
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    test_tickers = ['NVDA', 'TSLA']
    for ticker in test_tickers:
        df = collector.fetch_stock_data(
            ticker,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        if df is not None:
            logger.info(f"Successfully fetched {len(df)} rows for {ticker}")
        else:
            logger.warning(f"Failed to fetch data for {ticker}")

if __name__ == "__main__":
    main() 