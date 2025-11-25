"""
Predictor Module for Stock Price Movement Prediction

This module handles loading trained models and making predictions on the latest data.
It integrates with the pipeline to provide buy/sell signals based on trained models.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
from colorama import Fore, Style, init

# Initialize colorama
init()

# Import path configs
from src.utils.path_config import MODEL_DIR, RESULTS_DIR, PROCESSED_DIR

class Predictor:
    """Handles predictions using trained models on latest feature data."""
    
    def __init__(self, confidence_threshold: float = 0.6):
        """Initialize the predictor.
        
        Args:
            confidence_threshold: Minimum confidence required for a strong signal
        """
        self.models_path = MODEL_DIR
        self.features_path = PROCESSED_DIR / "feature_sets"
        self.results_path = RESULTS_DIR / "predictions"
        self.confidence_threshold = confidence_threshold
        
        # Create necessary directories
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        print(f"{Fore.GREEN}✓ Predictor initialized{Style.RESET_ALL}")
    
    def _load_model(self, ticker: str) -> Tuple[Optional[object], Optional[List[str]]]:
        """Load the trained model and its feature list for a ticker.
        
        Returns:
            Tuple of (model, feature_list) or (None, None) if loading fails
        """
        try:
            model_file = self.models_path / f"{ticker}_model.pkl"
            if not model_file.exists():
                print(f"{Fore.RED}✗ No model found for {ticker}{Style.RESET_ALL}")
                return None, None
                
            artifact = joblib.load(model_file)
            if isinstance(artifact, dict) and 'model' in artifact and 'features' in artifact:
                return artifact['model'], artifact['features']
            else:
                # Handle legacy models that were saved without features
                print(f"{Fore.YELLOW}⚠ Legacy model format detected for {ticker}{Style.RESET_ALL}")
                return artifact, None
            
        except Exception as e:
            print(f"{Fore.RED}Error loading model for {ticker}: {str(e)}{Style.RESET_ALL}")
            return None, None
    
    def _load_features(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load the latest feature set for a ticker."""
        try:
            feature_file = self.features_path / f"{ticker}_features.csv"
            if not feature_file.exists():
                print(f"{Fore.RED}✗ No feature set found for {ticker}{Style.RESET_ALL}")
                return None
                
            df = pd.read_csv(feature_file)
            df['Date'] = pd.to_datetime(df['Date'])
            return df
            
        except Exception as e:
            print(f"{Fore.RED}Error loading features for {ticker}: {str(e)}{Style.RESET_ALL}")
            return None
    
    def _prepare_features(self, df: pd.DataFrame, required_features: List[str]) -> pd.DataFrame:
        """Prepare features for prediction, ensuring all required features are present.
        
        Args:
            df: DataFrame containing the latest data
            required_features: List of features required by the model
            
        Returns:
            DataFrame with exactly the required features
        """
        # Create a copy to avoid modifying the original
        features = pd.DataFrame(index=df.index)
        
        # Add required features, filling missing ones with 0
        for col in required_features:
            if col in df.columns:
                features[col] = df[col]
            else:
                print(f"{Fore.YELLOW}⚠ Missing feature '{col}' - using 0{Style.RESET_ALL}")
                features[col] = 0
                
        return features
    
    def _get_signal_emoji(self, prediction: int, confidence: float) -> str:
        """Get an appropriate emoji for the prediction signal."""
        if prediction == 1:
            return "🟢" if confidence >= self.confidence_threshold else "🔼"
        else:
            return "🔴" if confidence >= self.confidence_threshold else "🔽"
    
    def predict(self, ticker: str, save_results: bool = True) -> Optional[Dict]:
        """Make a prediction for a single ticker.
        
        Args:
            ticker: The ticker symbol to predict
            save_results: Whether to save the prediction results
            
        Returns:
            Dictionary containing prediction details or None if prediction fails
        """
        try:
            # Load model and features
            model, required_features = self._load_model(ticker)
            df = self._load_features(ticker)
            
            if model is None or df is None:
                return None
            
            # Get latest data point
            latest = df.sort_values('Date').iloc[-1:]
            
            # Prepare features ensuring consistency with training
            if required_features is not None:
                features = self._prepare_features(latest, required_features)
            else:
                # Fallback for legacy models - use all numeric columns except Date and target
                features = latest.drop(columns=['Date', 'Ticker', 'Close_pct_change_t+1'], errors='ignore')
                features = features.select_dtypes(include=[np.number])
            
            # Simple imputation
            features = features.ffill().fillna(0)
            
            # Make prediction
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0][1]
            
            # Get signal strength
            signal = self._get_signal_emoji(prediction, probability)
            direction = "BUY" if prediction == 1 else "SELL"
            strength = "Strong" if probability >= self.confidence_threshold else "Weak"
            
            # Format the prediction message
            print(f"\n{Fore.CYAN}Prediction for {ticker}:{Style.RESET_ALL}")
            print(f"• Signal: {signal} {strength} {direction}")
            print(f"• Confidence: {probability:.2%}")
            print(f"• Date: {latest['Date'].iloc[0]}")
            
            # Prepare results
            result = {
                'ticker': ticker,
                'date': latest['Date'].iloc[0],
                'prediction': int(prediction),
                'confidence': float(probability),
                'signal': direction,
                'strength': strength,
                'timestamp': datetime.now()
            }
            
            # Save results if requested
            if save_results:
                output_file = self.results_path / f"{ticker}_prediction.csv"
                pd.DataFrame([result]).to_csv(output_file, index=False)
                print(f"{Fore.GREEN}✓ Saved prediction to {output_file}{Style.RESET_ALL}")
            
            return result
            
        except Exception as e:
            print(f"{Fore.RED}Error predicting for {ticker}: {str(e)}{Style.RESET_ALL}")
            return None
    
    def predict_all(self, tickers: List[str], save_summary: bool = True) -> pd.DataFrame:
        """Make predictions for multiple tickers.
        
        Args:
            tickers: List of ticker symbols to predict
            save_summary: Whether to save a summary of all predictions
            
        Returns:
            DataFrame containing all prediction results
        """
        results = []
        
        print(f"\n{Fore.CYAN}Running predictions for {len(tickers)} tickers...{Style.RESET_ALL}")
        
        for ticker in tickers:
            result = self.predict(ticker)
            if result:
                results.append(result)
        
        if not results:
            print(f"{Fore.YELLOW}No predictions were made{Style.RESET_ALL}")
            return pd.DataFrame()
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(results)
        
        if save_summary:
            # Save with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            summary_file = self.results_path / f"prediction_summary_{timestamp}.csv"
            summary_df.to_csv(summary_file, index=False)
            
            # Print summary table
            print(f"\n{Fore.CYAN}Prediction Summary:{Style.RESET_ALL}")
            print(f"{'Ticker':<6} {'Signal':<6} {'Strength':<8} {'Confidence':<10}")
            print("-" * 32)
            
            for _, row in summary_df.iterrows():
                signal = "🟢" if row['prediction'] == 1 else "🔴"
                print(f"{row['ticker']:<6} {signal} {row['signal']:<4} {row['strength']:<8} {row['confidence']:.2%}")
        
        return summary_df

def main():
    """CLI entry point for predictor."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Make predictions for stock tickers')
    parser.add_argument('--tickers', type=str, help='Path to file containing tickers (one per line)')
    parser.add_argument('--confidence', type=float, default=0.6, 
                      help='Confidence threshold for strong signals')
    parser.add_argument('--save-summary', action='store_true', 
                      help='Save a summary of all predictions')
    
    args = parser.parse_args()
    
    # Load tickers
    if args.tickers:
        with open(args.tickers) as f:
            test_tickers = [line.strip() for line in f if line.strip()]
    else:
        test_tickers = ['AAPL', 'GOOGL', 'MSFT']  # Default test tickers
    
    # Initialize predictor and run predictions
    predictor = Predictor(confidence_threshold=args.confidence)
    predictor.predict_all(test_tickers, save_summary=args.save_summary)

if __name__ == "__main__":
    main() 