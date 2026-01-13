# src.modeling.model_trainer

"""
Model Trainer Module for Stock Price Movement Prediction

This module handles training, evaluation, and persistence of predictive models
for stock price movement based on Reddit sentiment and technical indicators.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from colorama import Fore, Style

# Configure logging
logger = logging.getLogger(__name__)

# Import Path Configs
from src.utils.path_config import PROCESSED_DIR, MODEL_DIR, RESULTS_DIR

class ModelTrainer:
    """Handles training and evaluation of predictive models for stock price movements."""
    
    def __init__(self, 
                 model_type: str = 'xgboost',
                 test_size: float = 0.2,
                 n_splits: int = 5,
                 classification_threshold: float = 0.001,
                 lookback_window_days: int = 60,
                 min_rows_required: int = 30,
                 min_class_balance: float = 0.1,
                 verbose: bool = False,
                 summary_path: Optional[str] = None):
        """Initialize the ModelTrainer.
        
        Args:
            model_type: Type of model to use ('xgboost' or 'random_forest')
            test_size: Fraction of data to use for testing
            n_splits: Number of splits for time series cross-validation
            classification_threshold: Threshold for converting price changes to binary labels
            lookback_window_days: Number of days to look back for feature generation
            min_rows_required: Minimum number of rows required for training
            min_class_balance: Minimum proportion required for minority class
            verbose: Whether to print detailed training information
            summary_path: Path to save training summary (CSV or JSON)
        """
        # Configuration
        self.verbose = verbose
        self.summary_path = Path(summary_path) if summary_path else None
        
        # Set up paths
        self.model_dir = MODEL_DIR
        self.results_dir = RESULTS_DIR
        self.feature_sets_dir = PROCESSED_DIR / "feature_sets"
        
        # Create necessary directories
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Model configuration
        self.model_type = model_type.lower()
        self.test_size = test_size
        self.n_splits = n_splits
        self.classification_threshold = classification_threshold
        self.lookback_window_days = lookback_window_days
        self.min_rows_required = min_rows_required
        self.min_class_balance = min_class_balance
        
        # Default model parameters
        self.model_params = {
            'xgboost': {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'learning_rate': 0.1,
                'max_depth': 5,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'base_score': 0.5
            },
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 5,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'class_weight': 'balanced'
            }
        }
        
        # Training summary for all tickers
        self.training_summary = {}
        
        print(f"{Fore.GREEN}✓ Model Trainer initialized{Style.RESET_ALL}")
    
    def _load_feature_set(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load feature set for a given ticker."""
        try:
            feature_file = self.feature_sets_dir / f"{ticker}_features.csv"
            if not feature_file.exists():
                logger.warning(f"No feature file found for {ticker}")
                return None
            
            df = pd.read_csv(feature_file)
            df['Date'] = pd.to_datetime(df['Date'])
            return df
            
        except Exception as e:
            logger.error(f"Error loading features for {ticker}: {str(e)}")
            return None
    
    def _validate_feature_data(self, df: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
        """Validate feature data efficiently."""
        try:
            if df is None or df.empty:
                logger.error(f"{ticker}: Feature data is empty")
                return None
            
            # Check for target column
            if 'Close_pct_change_t+1' not in df.columns:
                logger.error(f"{ticker}: Missing target column 'Close_pct_change_t+1'")
                return None
            
            # Efficiently identify and drop constant features
            feature_cols = [col for col in df.columns if col not in ['Date', 'Ticker', 'Close_pct_change_t+1']]
            if not feature_cols:
                logger.error(f"{ticker}: No feature columns found")
                return None
            
            # Use nunique() for efficient constant feature detection
            nunique = df[feature_cols].nunique()
            constant_features = nunique[nunique <= 1].index.tolist()
            
            if constant_features:
                if self.verbose:
                    logger.info(f"{ticker}: Dropping {len(constant_features)} constant features")
                df = df.drop(columns=constant_features)
                
            # Log constant features to debug
            if constant_features and self.verbose:
                logger.debug(f"{ticker} constant features: {constant_features}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error validating features for {ticker}: {str(e)}")
            return None
    
    def _prepare_data(self, df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """Prepare features and target for modeling with validation."""
        try:
            # Drop non-feature columns
            drop_cols = ['Date', 'Ticker', 'Close_pct_change_t+1']
            feature_cols = [col for col in df.columns if col not in drop_cols]
            
            # Prepare features
            X = df[feature_cols].copy()
            
            # Modern fillna syntax
            X = X.ffill().fillna(0)
            
            # Create binary target (1 for price increase, 0 for decrease)
            y = (df['Close_pct_change_t+1'] > self.classification_threshold).astype(int)
            
            # Validate class balance
            unique_labels, counts = np.unique(y, return_counts=True)
            if len(unique_labels) < 2:
                logger.error(f"Only one class ({unique_labels[0]}) present in target labels")
                return None, None
            
            # Log class distribution
            total = len(y)
            for label, count in zip(unique_labels, counts):
                percentage = (count / total) * 100
                logger.info(f"Class {label}: {count} samples ({percentage:.2f}%)")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            return None, None
    
    def _create_model(self) -> Union[xgb.XGBClassifier, RandomForestClassifier]:
        """Create a new model instance based on configuration."""
        if self.model_type == 'xgboost':
            return xgb.XGBClassifier(**self.model_params['xgboost'])
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(**self.model_params['random_forest'])
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _evaluate_predictions(self, y_true: pd.Series, y_pred: np.ndarray, 
                            y_prob: np.ndarray) -> Dict:
        """Calculate performance metrics for predictions."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_prob),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        return metrics
    
    def _analyze_feature_data(self, df: pd.DataFrame, ticker: str) -> Dict:
        """Analyze feature data with compact output."""
        feature_cols = [col for col in df.columns if col not in ['Date', 'Ticker', 'Close_pct_change_t+1']]
        
        # Calculate feature statistics efficiently
        stats = {
            'n_rows': len(df),
            'n_features': len(feature_cols),
            'date_range': (df['Date'].min(), df['Date'].max()),
            'class_balance': None
        }
        
        # Calculate class balance
        y = (df['Close_pct_change_t+1'] > self.classification_threshold).astype(int)
        class_counts = y.value_counts()
        stats['class_balance'] = class_counts.min() / len(y)
        
        # Print compact summary
        print(f"\n{Fore.CYAN}Dataset Summary for {ticker}:{Style.RESET_ALL}")
        print(f"• Samples: {stats['n_rows']} ({stats['date_range'][0]} to {stats['date_range'][1]})")
        print(f"• Features: {stats['n_features']}")
        print(f"• Class Balance: {stats['class_balance']:.2%}")
        
        # Only print detailed feature analysis in verbose mode
        if self.verbose:
            variances = df[feature_cols].var().sort_values(ascending=False)
            print("\n• Top 10 Features by Variance:")
            for feat, var in variances.head(10).items():
                print(f"  - {feat}: {var:.4f}")
        
        return stats
    
    def _save_model_results(self, ticker: str, model, metrics: Dict, feature_importance: pd.DataFrame) -> None:
        """Save the trained model and results to disk.
        
        Args:
            ticker: The ticker symbol
            model: The trained model instance
            metrics: Dictionary of model performance metrics
            feature_importance: DataFrame of feature importance scores
            
        The method saves:
        - A model artifact containing both the model and its feature names
        - Performance metrics in JSON format
        - Feature importance scores in CSV format
        """
        try:
            # Save model artifact (model + features)
            model_path = self.model_dir / f"{ticker}_model.pkl"
            model_artifact = {
                'model': model,
                'features': list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else list(feature_importance.index)
            }
            joblib.dump(model_artifact, model_path)
            logger.info(f"Saved model artifact for {ticker} to {model_path}")

            # Save metrics
            metrics_path = self.results_dir / f"{ticker}_metrics.json"
            pd.DataFrame([metrics]).to_json(metrics_path, orient='records', indent=2)
            logger.info(f"Saved metrics for {ticker} to {metrics_path}")

            # Save feature importance
            fi_path = self.results_dir / f"{ticker}_feature_importance.csv"
            feature_importance.to_csv(fi_path)
            logger.info(f"Saved feature importance for {ticker} to {fi_path}")

        except Exception as e:
            logger.error(f"Failed to save model results for {ticker}: {str(e)}")
    
    def train_and_evaluate(self, ticker: str, save_results: bool = True) -> Dict:
        """Train and evaluate a model with streamlined output."""
        try:
            df = self._load_feature_set(ticker)
            df = self._validate_feature_data(df, ticker)
            if df is None:
                self.training_summary[ticker] = {'status': 'failed', 'reason': 'invalid_data'}
                return None
            
            stats = self._analyze_feature_data(df, ticker)
            
            # Validation checks
            if stats['n_rows'] < self.min_rows_required:
                print(f"{Fore.YELLOW}⚠ {ticker}: Insufficient rows ({stats['n_rows']} < {self.min_rows_required}){Style.RESET_ALL}")
                self.training_summary[ticker] = {'status': 'skipped', 'reason': 'insufficient_rows'}
                return None
            
            if stats['class_balance'] < self.min_class_balance:
                print(f"{Fore.YELLOW}⚠ {ticker}: Poor class balance ({stats['class_balance']:.2%}){Style.RESET_ALL}")
                self.training_summary[ticker] = {'status': 'skipped', 'reason': 'class_imbalance'}
                return None
            
            X, y = self._prepare_data(df)
            if X is None or y is None:
                return None
            
            # Training with compact output
            valid_folds = 0
            fold_metrics = []
            feature_importance = pd.DataFrame(0, index=X.columns, columns=['importance'])
            
            # Cross-validation with minimal output
            tscv = TimeSeriesSplit(n_splits=self.n_splits)
            for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                    continue
                
                valid_folds += 1
                model = self._create_model()
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
                
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'f1': f1_score(y_test, y_pred, zero_division=0),
                    'roc_auc': roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else np.nan
                }
                
                fold_metrics.append(metrics)
                
                if hasattr(model, 'feature_importances_'):
                    feature_importance['importance'] += model.feature_importances_
            
            if valid_folds < 2:
                print(f"{Fore.YELLOW}⚠ {ticker}: Insufficient valid folds ({valid_folds}){Style.RESET_ALL}")
                self.training_summary[ticker] = {'status': 'skipped', 'reason': 'insufficient_folds'}
                return None
            
            # Calculate final metrics
            avg_metrics = {
                metric: np.mean([m[metric] for m in fold_metrics])
                for metric in ['accuracy', 'f1', 'roc_auc']
            }
            
            # Get top features
            feature_importance['importance'] /= valid_folds
            top_features = feature_importance.sort_values('importance', ascending=False).head(5)
            
            # Print compact summary
            print(f"{Fore.GREEN}✓ {ticker:<5} → ", end='')
            print(f"Acc: {avg_metrics['accuracy']:.2f} | F1: {avg_metrics['f1']:.2f} | AUC: {avg_metrics['roc_auc']:.2f}")
            print(f"  Samples: {len(y)} | Features: {len(X.columns)} | Folds: {valid_folds}/{self.n_splits}")
            print(f"  Top Features: {', '.join(top_features.index)}")
            
            # Save results
            if save_results:
                self._save_model_results(ticker, model, avg_metrics, feature_importance)
            
            # Update training summary
            result = {
                'status': 'success',
                'metrics': avg_metrics,
                'n_samples': len(y),
                'n_features': len(X.columns),
                'valid_folds': valid_folds,
                'top_features': top_features.index.tolist(),
                'class_balance': stats['class_balance']
            }
            
            self.training_summary[ticker] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error training model for {ticker}: {str(e)}")
            self.training_summary[ticker] = {'status': 'failed', 'reason': str(e)}
            return None
    
    def save_training_summary(self) -> None:
        """Save training summary to file if path is specified."""
        if not self.summary_path:
            return
        
        try:
            summary_data = []
            for ticker, result in self.training_summary.items():
                if result['status'] == 'success':
                    summary_data.append({
                        'ticker': ticker,
                        'status': result['status'],
                        'accuracy': result['metrics']['accuracy'],
                        'f1': result['metrics']['f1'],
                        'roc_auc': result['metrics']['roc_auc'],
                        'samples': result['n_samples'],
                        'features': result['n_features'],
                        'valid_folds': result['valid_folds'],
                        'top_features': ','.join(result['top_features']),
                        'class_balance': result['class_balance']
                    })
                else:
                    summary_data.append({
                        'ticker': ticker,
                        'status': result['status'],
                        'reason': result.get('reason', 'unknown')
                    })
            
            df = pd.DataFrame(summary_data)
            
            if self.summary_path.suffix == '.csv':
                df.to_csv(self.summary_path, index=False)
            elif self.summary_path.suffix == '.json':
                df.to_json(self.summary_path, orient='records', indent=2)
            else:
                logger.warning(f"Unsupported file format: {self.summary_path.suffix}")
                
            print(f"\n{Fore.GREEN}✓ Saved training summary to {self.summary_path}{Style.RESET_ALL}")
            
        except Exception as e:
            logger.error(f"Error saving training summary: {str(e)}")

    def get_training_summary(self) -> str:
        """Generate a formatted summary of all training results."""
        summary = f"\n{Fore.CYAN}=== Training Summary ==={Style.RESET_ALL}\n"
        
        # Group tickers by status
        status_groups = {'success': [], 'skipped': [], 'failed': []}
        for ticker, result in self.training_summary.items():
            status_groups[result['status']].append((ticker, result))
        
        # Print successful models
        if status_groups['success']:
            summary += f"\n{Fore.GREEN}Successful Models:{Style.RESET_ALL}\n"
            for ticker, result in status_groups['success']:
                metrics = result['metrics']
                summary += f"• {ticker:<5} → Acc: {metrics['accuracy']:.2f} | F1: {metrics['f1']:.2f} | AUC: {metrics['roc_auc']:.2f}\n"
        
        # Print skipped models
        if status_groups['skipped']:
            summary += f"\n{Fore.YELLOW}Skipped Models:{Style.RESET_ALL}\n"
            for ticker, result in status_groups['skipped']:
                summary += f"• {ticker:<5} → Reason: {result['reason']}\n"
        
        # Print failed models
        if status_groups['failed']:
            summary += f"\n{Fore.RED}Failed Models:{Style.RESET_ALL}\n"
            for ticker, result in status_groups['failed']:
                summary += f"• {ticker:<5} → Error: {result['reason']}\n"
        
        return summary

def main():
    """CLI entry point with enhanced options."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train stock price movement prediction models')
    parser.add_argument('--tickers', type=str, help='Path to file containing tickers (one per line)')
    parser.add_argument('--lookback-days', type=int, default=60, help='Number of days to look back')
    parser.add_argument('--min-rows', type=int, default=30, help='Minimum rows required for training')
    parser.add_argument('--min-class-balance', type=float, default=0.1, help='Minimum class balance')
    parser.add_argument('--verbose', action='store_true', help='Print detailed output')
    parser.add_argument('--summary-path', type=str, help='Path to save training summary (CSV or JSON)')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Load tickers
    if args.tickers:
        with open(args.tickers) as f:
            test_tickers = [line.strip() for line in f if line.strip()]
    else:
        test_tickers = ['AAPL', 'GOOGL', 'MSFT']
    
    # Initialize trainer
    trainer = ModelTrainer(
        model_type='xgboost',
        lookback_window_days=args.lookback_days,
        min_rows_required=args.min_rows,
        min_class_balance=args.min_class_balance,
        verbose=args.verbose,
        summary_path=args.summary_path
    )
    
    # Train models
    for ticker in test_tickers:
        trainer.train_and_evaluate(ticker, save_results=True)
    
    # Print and save summary
    print(trainer.get_training_summary())
    trainer.save_training_summary()

if __name__ == "__main__":
    main() 