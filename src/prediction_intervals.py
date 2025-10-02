"""
Prediction intervals and uncertainty quantification for retail demand forecasting.
Implements quantile regression and conformal prediction methods.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import QuantileRegressor
import warnings
warnings.filterwarnings('ignore')

# Model imports
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

logger = logging.getLogger(__name__)


class QuantileRegressionModel:
    """Quantile regression model for prediction intervals."""
    
    def __init__(self, quantiles: List[float] = [0.1, 0.5, 0.9]):
        """
        Initialize quantile regression model.
        
        Args:
            quantiles: List of quantiles to predict
        """
        self.quantiles = quantiles
        self.models = {}
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit quantile regression models for each quantile."""
        for quantile in self.quantiles:
            logger.info(f"Fitting quantile regression for quantile {quantile}")
            
            # Use QuantileRegressor for each quantile
            model = QuantileRegressor(quantile=quantile, alpha=0.0)
            model.fit(X, y)
            
            self.models[quantile] = model
        
        self.is_fitted = True
        logger.info("Quantile regression models fitted successfully")
    
    def predict(self, X: pd.DataFrame) -> Dict[float, np.ndarray]:
        """Predict quantiles for given features."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        predictions = {}
        for quantile, model in self.models.items():
            pred = model.predict(X)
            predictions[quantile] = np.maximum(pred, 0)  # Ensure non-negative
        
        return predictions


class LightGBMQuantileModel:
    """LightGBM-based quantile regression for prediction intervals."""
    
    def __init__(self, quantiles: List[float] = [0.1, 0.5, 0.9]):
        """
        Initialize LightGBM quantile model.
        
        Args:
            quantiles: List of quantiles to predict
        """
        self.quantiles = quantiles
        self.models = {}
        self.is_fitted = False
        
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not available")
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit LightGBM models for each quantile."""
        for quantile in self.quantiles:
            logger.info(f"Fitting LightGBM quantile regression for quantile {quantile}")
            
            # LightGBM parameters for quantile regression
            params = {
                'objective': 'quantile',
                'metric': 'quantile',
                'alpha': quantile,
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1
            }
            
            # Create dataset
            train_data = lgb.Dataset(X, label=y)
            
            # Train model
            model = lgb.train(
                params,
                train_data,
                num_boost_round=1000,
                valid_sets=[train_data],
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
            )
            
            self.models[quantile] = model
        
        self.is_fitted = True
        logger.info("LightGBM quantile regression models fitted successfully")
    
    def predict(self, X: pd.DataFrame) -> Dict[float, np.ndarray]:
        """Predict quantiles using LightGBM models."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        predictions = {}
        for quantile, model in self.models.items():
            pred = model.predict(X)
            predictions[quantile] = np.maximum(pred, 0)  # Ensure non-negative
        
        return predictions


class ConformalPrediction:
    """Conformal prediction for uncertainty quantification."""
    
    def __init__(self, confidence_level: float = 0.9):
        """
        Initialize conformal prediction.
        
        Args:
            confidence_level: Confidence level for prediction intervals
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.calibration_scores = None
        self.quantile_threshold = None
        self.is_fitted = False
    
    def fit(self, y_true: pd.Series, y_pred: np.ndarray):
        """
        Fit conformal prediction using calibration data.
        
        Args:
            y_true: True values from calibration set
            y_pred: Predictions from calibration set
        """
        # Calculate conformity scores (absolute residuals)
        conformity_scores = np.abs(y_true - y_pred)
        
        # Calculate quantile threshold
        self.quantile_threshold = np.quantile(conformity_scores, 1 - self.alpha)
        
        self.is_fitted = True
        logger.info(f"Conformal prediction fitted with threshold: {self.quantile_threshold:.4f}")
    
    def predict_intervals(self, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate prediction intervals using conformal prediction.
        
        Args:
            y_pred: Point predictions
        
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if not self.is_fitted:
            raise ValueError("Conformal prediction must be fitted before making predictions")
        
        lower_bound = y_pred - self.quantile_threshold
        upper_bound = y_pred + self.quantile_threshold
        
        # Ensure non-negative bounds
        lower_bound = np.maximum(lower_bound, 0)
        
        return lower_bound, upper_bound


class BootstrapPredictionIntervals:
    """Bootstrap-based prediction intervals."""
    
    def __init__(self, n_bootstrap: int = 100, confidence_level: float = 0.9):
        """
        Initialize bootstrap prediction intervals.
        
        Args:
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for intervals
        """
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.bootstrap_predictions = None
        self.is_fitted = False
    
    def fit(self, model, X: pd.DataFrame, y: pd.Series):
        """
        Fit bootstrap models.
        
        Args:
            model: Base model to bootstrap
            X: Training features
            y: Training target
        """
        bootstrap_predictions = []
        
        for i in range(self.n_bootstrap):
            logger.info(f"Bootstrap iteration {i+1}/{self.n_bootstrap}")
            
            # Bootstrap sample
            n_samples = len(X)
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            
            X_bootstrap = X.iloc[bootstrap_indices]
            y_bootstrap = y.iloc[bootstrap_indices]
            
            # Fit model on bootstrap sample
            bootstrap_model = type(model)(**model.get_params()) if hasattr(model, 'get_params') else model
            bootstrap_model.fit(X_bootstrap, y_bootstrap)
            
            # Store model for later prediction
            bootstrap_predictions.append(bootstrap_model)
        
        self.bootstrap_models = bootstrap_predictions
        self.is_fitted = True
        logger.info("Bootstrap models fitted successfully")
    
    def predict_intervals(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate prediction intervals using bootstrap.
        
        Args:
            X: Features for prediction
        
        Returns:
            Tuple of (mean_prediction, lower_bound, upper_bound)
        """
        if not self.is_fitted:
            raise ValueError("Bootstrap models must be fitted before making predictions")
        
        # Get predictions from all bootstrap models
        all_predictions = []
        for model in self.bootstrap_models:
            pred = model.predict(X)
            all_predictions.append(pred)
        
        all_predictions = np.array(all_predictions)
        
        # Calculate statistics
        mean_prediction = np.mean(all_predictions, axis=0)
        lower_bound = np.quantile(all_predictions, self.alpha / 2, axis=0)
        upper_bound = np.quantile(all_predictions, 1 - self.alpha / 2, axis=0)
        
        # Ensure non-negative bounds
        lower_bound = np.maximum(lower_bound, 0)
        mean_prediction = np.maximum(mean_prediction, 0)
        
        return mean_prediction, lower_bound, upper_bound


class PredictionIntervalGenerator:
    """Main class for generating prediction intervals."""
    
    def __init__(self, config: Dict):
        """Initialize prediction interval generator."""
        self.config = config
        self.prediction_config = config['prediction_intervals']
        self.confidence_levels = self.prediction_config['confidence_levels']
        self.method = self.prediction_config['method']
        
        # Initialize methods
        self.quantile_model = None
        self.conformal_prediction = None
        self.bootstrap_intervals = None
    
    def fit(self, model, X: pd.DataFrame, y: pd.Series, X_cal: pd.DataFrame = None, y_cal: pd.Series = None):
        """
        Fit prediction interval methods.
        
        Args:
            model: Base forecasting model
            X: Training features
            y: Training target
            X_cal: Calibration features (for conformal prediction)
            y_cal: Calibration target (for conformal prediction)
        """
        logger.info(f"Fitting prediction intervals using method: {self.method}")
        
        if self.method == 'quantile_regression':
            self._fit_quantile_regression(X, y)
        
        elif self.method == 'conformal_prediction':
            self._fit_conformal_prediction(model, X, y, X_cal, y_cal)
        
        elif self.method == 'bootstrap':
            self._fit_bootstrap_intervals(model, X, y)
        
        else:
            raise ValueError(f"Unknown prediction interval method: {self.method}")
        
        logger.info("Prediction intervals fitted successfully")
    
    def _fit_quantile_regression(self, X: pd.DataFrame, y: pd.Series):
        """Fit quantile regression models."""
        # Calculate quantiles for all confidence levels
        all_quantiles = []
        for confidence in self.confidence_levels:
            alpha = 1 - confidence
            all_quantiles.extend([alpha / 2, 0.5, 1 - alpha / 2])
        
        all_quantiles = sorted(list(set(all_quantiles)))
        
        # Use LightGBM quantile regression if available
        if LIGHTGBM_AVAILABLE:
            self.quantile_model = LightGBMQuantileModel(all_quantiles)
        else:
            self.quantile_model = QuantileRegressionModel(all_quantiles)
        
        self.quantile_model.fit(X, y)
    
    def _fit_conformal_prediction(self, model, X: pd.DataFrame, y: pd.Series, 
                                 X_cal: pd.DataFrame, y_cal: pd.Series):
        """Fit conformal prediction."""
        if X_cal is None or y_cal is None:
            # Use last portion of data for calibration
            cal_size = int(len(X) * 0.2)
            X_cal = X.iloc[-cal_size:]
            y_cal = y.iloc[-cal_size:]
            X_train = X.iloc[:-cal_size]
            y_train = y.iloc[:-cal_size]
        else:
            X_train = X
            y_train = y
        
        # Fit base model on training data
        model.fit(X_train, y_train)
        
        # Get predictions on calibration data
        y_pred_cal = model.predict(X_cal)
        
        # Fit conformal prediction
        self.conformal_prediction = ConformalPrediction()
        self.conformal_prediction.fit(y_cal, y_pred_cal)
    
    def _fit_bootstrap_intervals(self, model, X: pd.DataFrame, y: pd.Series):
        """Fit bootstrap prediction intervals."""
        self.bootstrap_intervals = BootstrapPredictionIntervals(
            n_bootstrap=100,
            confidence_level=max(self.confidence_levels)
        )
        self.bootstrap_intervals.fit(model, X, y)
    
    def predict_intervals(self, X: pd.DataFrame) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Generate prediction intervals for all confidence levels.
        
        Args:
            X: Features for prediction
        
        Returns:
            Dictionary containing intervals for each confidence level
        """
        intervals = {}
        
        if self.method == 'quantile_regression':
            intervals = self._predict_quantile_intervals(X)
        
        elif self.method == 'conformal_prediction':
            intervals = self._predict_conformal_intervals(X)
        
        elif self.method == 'bootstrap':
            intervals = self._predict_bootstrap_intervals(X)
        
        return intervals
    
    def _predict_quantile_intervals(self, X: pd.DataFrame) -> Dict[str, Dict[str, np.ndarray]]:
        """Generate quantile-based prediction intervals."""
        quantile_predictions = self.quantile_model.predict(X)
        intervals = {}
        
        for confidence in self.confidence_levels:
            alpha = 1 - confidence
            lower_quantile = alpha / 2
            upper_quantile = 1 - alpha / 2
            median_quantile = 0.5
            
            intervals[f'confidence_{confidence}'] = {
                'lower': quantile_predictions[lower_quantile],
                'upper': quantile_predictions[upper_quantile],
                'median': quantile_predictions[median_quantile]
            }
        
        return intervals
    
    def _predict_conformal_intervals(self, X: pd.DataFrame) -> Dict[str, Dict[str, np.ndarray]]:
        """Generate conformal prediction intervals."""
        # This would need the base model predictions
        # For now, return empty dict
        return {}
    
    def _predict_bootstrap_intervals(self, X: pd.DataFrame) -> Dict[str, Dict[str, np.ndarray]]:
        """Generate bootstrap prediction intervals."""
        mean_pred, lower_bound, upper_bound = self.bootstrap_intervals.predict_intervals(X)
        
        intervals = {}
        for confidence in self.confidence_levels:
            intervals[f'confidence_{confidence}'] = {
                'lower': lower_bound,
                'upper': upper_bound,
                'median': mean_pred
            }
        
        return intervals


def calculate_interval_coverage(y_true: pd.Series, intervals: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, float]:
    """
    Calculate coverage of prediction intervals.
    
    Args:
        y_true: True values
        intervals: Prediction intervals
    
    Returns:
        Dictionary of coverage rates for each confidence level
    """
    coverage = {}
    
    for conf_name, interval_dict in intervals.items():
        lower = interval_dict['lower']
        upper = interval_dict['upper']
        
        # Calculate coverage
        covered = (y_true >= lower) & (y_true <= upper)
        coverage_rate = covered.mean()
        
        coverage[conf_name] = coverage_rate
    
    return coverage


if __name__ == "__main__":
    # Example usage
    import yaml
    
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = pd.DataFrame(np.random.randn(n_samples, n_features), 
                    columns=[f'feature_{i}' for i in range(n_features)])
    y = pd.Series(np.random.poisson(10, n_samples))
    
    # Test quantile regression
    quantile_model = QuantileRegressionModel([0.1, 0.5, 0.9])
    quantile_model.fit(X, y)
    
    # Make predictions
    predictions = quantile_model.predict(X.iloc[:10])
    
    print("Quantile Predictions:")
    for quantile, pred in predictions.items():
        print(f"Quantile {quantile}: {pred[:5]}")
    
    # Test prediction interval generator
    generator = PredictionIntervalGenerator(config)
    
    # Create a simple model for testing
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    
    # Fit prediction intervals
    generator.fit(model, X, y)
    
    # Generate intervals
    intervals = generator.predict_intervals(X.iloc[:10])
    
    print("\nPrediction Intervals:")
    for conf_name, interval_dict in intervals.items():
        print(f"{conf_name}:")
        print(f"  Lower: {interval_dict['lower'][:3]}")
        print(f"  Upper: {interval_dict['upper'][:3]}")
        print(f"  Median: {interval_dict['median'][:3]}")

