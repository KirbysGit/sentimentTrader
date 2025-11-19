# src.modeling.feature_builder

"""
Feature Builder Module for Stock Price Movement Prediction

This module generates clean, well-structured feature sets combining Reddit sentiment
and financial indicators for predictive modeling of stock price movements.
"""


# Imports.
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from colorama import Fore, Style, init
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# Initialize Colorama.
init()

# Configure logging.
logger = logging.getLogger(__name__)

# Import Path Configs.
from src.utils.path_config import (
    PROCESSED_DIR,
    STOCK_DATA_DIR,
    TICKER_GENERAL_DIR,
    TICKER_SENTIMENT_DIR,
    PROCESSED_REDDIT_DIR
)

# Feature Builder Class.
class FeatureBuilder:

    # Initialize the FeatureBuilder w/ Necessary Paths & Configs.
    def __init__(self):
        # Set Up Paths.
        self.feature_sets_dir = PROCESSED_DIR / "feature_sets"
        self.feature_sets_dir.mkdir(parents=True, exist_ok=True)
        
        # Feature Configs.
        self.lag_periods = [1, 3, 7]  # Days to Lag.
        self.rolling_periods = [3, 7]  # Days for Rolling Averages.
        
        # Required Columns.
        self.reddit_cols = [
            'overall_sentiment',
            'comment_count',
            'reddit_mentions',
            'reddit_engagement',
            'avg_confidence'
        ]
        
        # Stock Columns.
        self.stock_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Add Helper Function for Date Conversion.
        self._to_naive_datetime = lambda series: (
            pd.to_datetime(series).dt.tz_localize(None).dt.floor('D')
            if pd.api.types.is_datetime64_any_dtype(series)
            else pd.to_datetime(series, utc=True).dt.tz_localize(None).dt.floor('D')
        )
        
        # Print Success Message.
        print(f"{Fore.GREEN}✓ Feature Builder initialized{Style.RESET_ALL}")
    
    # Load Reddit Data.
    def _load_reddit_data(self, ticker: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        try:
            # Load Processed Reddit Data.
            processed_file = PROCESSED_REDDIT_DIR / f"{ticker}_detailed_sentiment.csv"
            daily_file = PROCESSED_REDDIT_DIR / f"{ticker}_daily_sentiment.csv"
            
            # If NO Reddit Data Files Exist, Return None.   
            if not processed_file.exists() and not daily_file.exists():
                logger.warning(f"Missing Reddit data files for {ticker}")
                return None, None
            
            processed_df = None
            daily_df = None
            
            # Load processed data if available
            if processed_file.exists():
                processed_df = pd.read_csv(processed_file)
                # Convert date columns and ensure timezone naive
                if 'created_utc' in processed_df.columns:
                    processed_df['Date'] = self._to_naive_datetime(processed_df['created_utc'])
                elif 'date' in processed_df.columns:
                    processed_df['Date'] = self._to_naive_datetime(processed_df['date'])
            
            # Load daily data if available
            if daily_file.exists():
                daily_df = pd.read_csv(daily_file)
                if 'Date' in daily_df.columns:
                    daily_df['Date'] = self._to_naive_datetime(daily_df['Date'])
                elif 'date' in daily_df.columns:
                    daily_df['Date'] = self._to_naive_datetime(daily_df['date'])
                    daily_df.drop('date', axis=1, inplace=True)
            
            return processed_df, daily_df
            
        except Exception as e:
            logger.error(f"Error loading Reddit data for {ticker}: {str(e)}")
            return None, None
    
    def _load_stock_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load and validate stock price data for a given ticker."""
        try:
            stock_file = STOCK_DATA_DIR / f"{ticker}_raw.csv"
            
            if not stock_file.exists():
                logger.warning(f"Missing stock data file for {ticker}")
                return None
            
            stock_df = pd.read_csv(stock_file)
            
            # Ensure date column exists and is datetime (timezone naive)
            if 'Date' not in stock_df.columns and 'date' in stock_df.columns:
                stock_df['Date'] = stock_df['date']
                stock_df.drop('date', axis=1, inplace=True)
            elif 'Date' not in stock_df.columns:
                logger.error(f"No Date column in stock data for {ticker}")
                return None
            
            stock_df['Date'] = self._to_naive_datetime(stock_df['Date'])
            
            return stock_df
            
        except Exception as e:
            logger.error(f"Error loading stock data for {ticker}: {str(e)}")
            return None
    
    def _load_ticker_analysis(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load ticker analysis data if available."""
        try:
            # Find most recent analysis file
            analysis_files = list(TICKER_GENERAL_DIR.glob("ticker_analysis_daily_*.csv"))
            if not analysis_files:
                return None
            
            latest_file = max(analysis_files, key=lambda x: x.stat().st_mtime)
            df = pd.read_csv(latest_file)
            
            # Extract date from filename (format: ticker_analysis_daily_YYYYMMDD.csv)
            date_str = latest_file.stem.split('_')[-1]
            file_date = pd.to_datetime(date_str, format='%Y%m%d').floor('D')
            
            # Filter for specific ticker and create a copy to avoid SettingWithCopyWarning
            ticker_data = df[df['ticker'] == ticker].copy()
            if ticker_data.empty:
                return None
            
            # Add the date from filename as the Date column
            ticker_data['Date'] = file_date
            
            return ticker_data
            
        except Exception as e:
            logger.error(f"Error loading ticker analysis: {str(e)}")
            return None
    
    def _load_ticker_sentiment(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load ticker-specific sentiment data from the latest sentiment file."""
        try:
            # Find most recent sentiment file
            sentiment_files = list(TICKER_SENTIMENT_DIR.glob("ticker_sentiment_*.csv"))
            if not sentiment_files:
                logger.warning(f"No ticker sentiment files found")
                return None
            
            latest_file = max(sentiment_files, key=lambda x: x.stat().st_mtime)
            df = pd.read_csv(latest_file)
            
            # Extract date from filename (format: ticker_sentiment_YYYYMMDD.csv)
            date_str = latest_file.stem.split('_')[-1]
            file_date = pd.to_datetime(date_str, format='%Y%m%d').floor('D')
            
            # Filter for specific ticker and create a copy to avoid SettingWithCopyWarning
            ticker_data = df[df['ticker'] == ticker].copy()
            if ticker_data.empty:
                logger.warning(f"No sentiment data found for {ticker}")
                return None
            
            # Add the date from filename as the Date column
            ticker_data['Date'] = file_date
            
            # Convert other date columns to datetime
            date_columns = ['first_mention_date', 'last_mention_date']
            for col in date_columns:
                if col in ticker_data.columns:
                    ticker_data[col] = pd.to_datetime(ticker_data[col]).dt.floor('D')
            
            return ticker_data
            
        except Exception as e:
            logger.error(f"Error loading ticker sentiment data: {str(e)}")
            return None
    
    def _create_lag_features(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Create lagged features for specified columns efficiently using pd.concat."""
        # Create all lag features at once
        lag_dfs = []
        for col in columns:
            col_lags = [df[col].shift(lag).rename(f"{col}_lag_{lag}") 
                       for lag in self.lag_periods]
            lag_dfs.extend(col_lags)
        
        # Combine original DataFrame with all lag features
        if lag_dfs:
            return pd.concat([df] + lag_dfs, axis=1)
        return df
    
    def _create_rolling_features(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Create rolling average features for specified columns efficiently using pd.concat."""
        # Create all rolling features at once
        rolling_dfs = []
        for col in columns:
            rolling_means = [df[col].rolling(window=window).mean()
                           .rename(f"{col}_rolling_mean_{window}")
                           for window in self.rolling_periods]
            rolling_dfs.extend(rolling_means)
        
        # Combine original DataFrame with all rolling features
        if rolling_dfs:
            return pd.concat([df] + rolling_dfs, axis=1)
        return df
    
    def _create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create next-day percentage price change target."""
        target_df = df['Close'].pct_change().shift(-1).rename('Close_pct_change_t+1')
        return pd.concat([df, target_df], axis=1)
    
    def _merge_and_clean(self, daily_sentiment: pd.DataFrame, stock_data: pd.DataFrame,
                         ticker_analysis: Optional[pd.DataFrame] = None,
                         ticker_sentiment: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Merge and clean all data sources, ensuring consistent date handling."""
        try:
            # Create copies to avoid modifying original dataframes
            daily_sentiment = daily_sentiment.copy() if daily_sentiment is not None else None
            stock_data = stock_data.copy()
            
            def standardize_date_column(df: pd.DataFrame, source_name: str) -> Optional[pd.DataFrame]:
                """Standardize date column in DataFrame and log the date range."""
                if df is None:
                    return None
                
                # Find the date column
                date_cols = ['Date', 'date', 'created_utc']
                found_col = next((col for col in date_cols if col in df.columns), None)
                
                if found_col is None:
                    logger.warning(f"No date column found in {source_name}")
                    return None
                
                # Convert to datetime and ensure timezone naive
                df['Date'] = self._to_naive_datetime(df[found_col])
                
                # Log date range
                if not df.empty:
                    min_date = df['Date'].min()
                    max_date = df['Date'].max()
                    logger.info(f"{source_name} date range: {min_date} to {max_date}")
                
                # Drop other date columns except special ones
                special_dates = ['first_mention_date', 'last_mention_date']
                for col in df.columns:
                    if col in date_cols and col != 'Date' and col not in special_dates:
                        df.drop(col, axis=1, inplace=True)
                
                return df.sort_values('Date')
            
            # Standardize date columns in all DataFrames
            stock_data = standardize_date_column(stock_data, "Stock Data")
            if stock_data is None:
                raise ValueError("Stock data is required but date standardization failed")
            
            daily_sentiment = standardize_date_column(daily_sentiment, "Daily Sentiment")
            ticker_analysis = standardize_date_column(ticker_analysis, "Ticker Analysis")
            ticker_sentiment = standardize_date_column(ticker_sentiment, "Ticker Sentiment")
            
            # Start with stock data as the base
            df = stock_data
            
            # Merge with daily sentiment if available
            if daily_sentiment is not None and not daily_sentiment.empty:
                df = pd.merge(
                    df,
                    daily_sentiment,
                    on='Date',
                    how='left'
                )
                logger.info(f"After merging daily sentiment: {len(df)} rows")
            
            # Add ticker analysis if available
            if ticker_analysis is not None and not ticker_analysis.empty:
                df = pd.merge(
                    df,
                    ticker_analysis,
                    on='Date',
                    how='left'
                )
                logger.info(f"After merging ticker analysis: {len(df)} rows")
            
            # Add ticker sentiment if available
            if ticker_sentiment is not None and not ticker_sentiment.empty:
                df = pd.merge(
                    df,
                    ticker_sentiment,
                    on='Date',
                    how='left'
                )
                logger.info(f"After merging ticker sentiment: {len(df)} rows")
            
            # Final chronological sort and index reset
            df = df.sort_values('Date').reset_index(drop=True)
            
            # Log final date range
            if not df.empty:
                logger.info(f"Final dataset date range: {df['Date'].min()} to {df['Date'].max()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error in merge_and_clean: {str(e)}")
            raise  # Re-raise the exception to be handled by the caller
    
    def generate_features_for_ticker(self, 
                                   ticker: str, 
                                   save_output: bool = True,
                                   normalize: bool = False) -> Optional[pd.DataFrame]:
        """Generate feature set for a single ticker."""
        print(f"\n{Fore.CYAN}Generating features for {ticker}...{Style.RESET_ALL}")
        
        try:
            # Load data
            processed_reddit, daily_sentiment = self._load_reddit_data(ticker)
            stock_data = self._load_stock_data(ticker)
            ticker_analysis = self._load_ticker_analysis(ticker)
            ticker_sentiment = self._load_ticker_sentiment(ticker)
            
            # Stock data is required
            if stock_data is None:
                logger.error(f"Missing required stock data for {ticker}")
                return None
            
            # Merge available data sources
            df = self._merge_and_clean(
                daily_sentiment, 
                stock_data, 
                ticker_analysis,
                ticker_sentiment
            )
            
            if df.empty:
                logger.warning(f"No data after merging for {ticker}")
                return None
            
            # Create lag features for available columns
            sentiment_cols = [col for col in df.columns if 'sentiment' in col.lower()]
            price_cols = ['Close', 'Volume']
            engagement_cols = [col for col in df.columns if 'engagement' in col.lower()]
            confidence_cols = [col for col in df.columns if 'confidence' in col.lower()]
            
            feature_cols = []
            # Only add columns that exist in the DataFrame
            feature_cols.extend([col for col in price_cols if col in df.columns])
            feature_cols.extend([col for col in sentiment_cols if col in df.columns])
            feature_cols.extend([col for col in engagement_cols if col in df.columns])
            feature_cols.extend([col for col in confidence_cols if col in df.columns])
            
            if feature_cols:
                df = self._create_lag_features(df, feature_cols)
                df = self._create_rolling_features(df, feature_cols)
            
            # Create target variable
            df = self._create_target_variable(df)
            
            # Add ticker column
            df['Ticker'] = ticker
            
            # Normalize features if requested
            if normalize:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
            
            # Save output if requested
            if save_output:
                output_file = self.feature_sets_dir / f"{ticker}_features.csv"
                df.to_csv(output_file, index=False)
                print(f"{Fore.GREEN}✓ Saved features to {output_file}{Style.RESET_ALL}")
                
                # Print feature summary
                print(f"\n{Fore.CYAN}Feature Summary for {ticker}:{Style.RESET_ALL}")
                print(f"• Total rows: {len(df)}")
                print(f"• Feature categories:")
                print(f"  - Sentiment features: {len([c for c in df.columns if 'sentiment' in c.lower()])}")
                print(f"  - Price features: {len([c for c in df.columns if c in price_cols])}")
                print(f"  - Engagement features: {len([c for c in df.columns if 'engagement' in c.lower()])}")
                print(f"  - Confidence features: {len([c for c in df.columns if 'confidence' in c.lower()])}")
                print(f"  - Total features: {len(df.columns)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error generating features for {ticker}: {str(e)}")
            return None
    
    def generate_features_for_tickers(self, 
                                    tickers: List[str], 
                                    save_output: bool = True,
                                    normalize: bool = False) -> Dict[str, pd.DataFrame]:
        """Generate feature sets for multiple tickers."""
        results = {}
        for ticker in tickers:
            df = self.generate_features_for_ticker(ticker, save_output, normalize)
            if df is not None:
                results[ticker] = df
        return results

def main():
    """CLI entry point for testing."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Test tickers
    test_tickers = ['AAPL', 'GOOGL', 'MSFT']
    
    # Initialize builder
    builder = FeatureBuilder()
    
    # Generate features
    results = builder.generate_features_for_tickers(
        tickers=test_tickers,
        save_output=True,
        normalize=True
    )
    
    # Print summary
    print(f"\n{Fore.CYAN}Feature Generation Summary:{Style.RESET_ALL}")
    for ticker, df in results.items():
        if df is not None:
            print(f"{Fore.GREEN}✓ {ticker}: {len(df)} rows, {len(df.columns)} features{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}✗ {ticker}: Failed to generate features{Style.RESET_ALL}")

if __name__ == "__main__":
    main() 