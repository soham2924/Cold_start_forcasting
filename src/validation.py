"""
Time-aware validation framework for retail demand forecasting.
Implements proper time series cross-validation with gap handling.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from sklearn.model_selection import BaseCrossValidator
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class TimeAwareSplit(BaseCrossValidator):
    """Time-aware cross-validation splitter for time series data."""
    
    def __init__(self, n_splits: int = 5, test_size: float = 0.2, gap_days: int = 7):
        """
        Initialize time-aware splitter.
        
        Args:
            n_splits: Number of cross-validation splits
            test_size: Proportion of data to use for testing
            gap_days: Number of days to leave as gap between train and test
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap_days = gap_days
    
    def split(self, X: pd.DataFrame, y: pd.Series = None, groups: pd.Series = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test splits for time series data."""
        n_samples = len(X)
        test_size_samples = int(n_samples * self.test_size)
        gap_samples = int(self.gap_days * 24)  # Assuming daily data
        
        splits = []
        
        for i in range(self.n_splits):
            # Calculate test end position
            test_end = n_samples - (i * test_size_samples // self.n_splits)
            test_start = test_end - test_size_samples
            
            # Calculate train end position (with gap)
            train_end = test_start - gap_samples
            train_start = max(0, train_end - test_size_samples * 2)  # Use 2x test size for training
            
            # Ensure valid ranges
            if train_start >= train_end or test_start >= test_end:
                continue
            
            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)
            
            splits.append((train_indices, test_indices))
        
        return splits
    
    def get_n_splits(self, X: pd.DataFrame = None, y: pd.Series = None, groups: pd.Series = None) -> int:
        """Return the number of splitting iterations."""
        return self.n_splits


class TimeSeriesValidator:
    """Comprehensive time series validation framework."""
    
    def __init__(self, config: Dict):
        """Initialize validator with configuration."""
        self.config = config
        self.cv_config = config['model']['cv']
        self.n_splits = self.cv_config['n_splits']
        self.test_size = self.cv_config['test_size']
        self.gap_days = self.cv_config['gap_days']
        
        # Initialize splitter
        self.splitter = TimeAwareSplit(
            n_splits=self.n_splits,
            test_size=self.test_size,
            gap_days=self.gap_days
        )
    
    def validate_model(self, model, X: pd.DataFrame, y: pd.Series, 
                      city: str = None, sku_id: str = None) -> Dict[str, Any]:
        """
        Perform comprehensive model validation.
        
        Args:
            model: Model to validate
            X: Feature matrix
            y: Target variable
            city: City name for logging
            sku_id: SKU ID for logging
        
        Returns:
            Dictionary containing validation results
        """
        logger.info(f"Starting validation for {city}-{sku_id}")
        
        # Perform cross-validation
        cv_results = self._cross_validate(model, X, y)
        
        # Perform holdout validation
        holdout_results = self._holdout_validate(model, X, y)
        
        # Calculate validation summary
        validation_summary = self._calculate_validation_summary(cv_results, holdout_results)
        
        # Add metadata
        validation_summary['metadata'] = {
            'city': city,
            'sku_id': sku_id,
            'n_splits': self.n_splits,
            'test_size': self.test_size,
            'gap_days': self.gap_days,
            'total_samples': len(X)
        }
        
        logger.info(f"Validation completed for {city}-{sku_id}")
        
        return validation_summary
    
    def _cross_validate(self, model, X: pd.DataFrame, y: pd.Series) -> Dict[str, List[float]]:
        """Perform cross-validation."""
        cv_scores = {
            'mae': [],
            'mse': [],
            'rmse': [],
            'mape': [],
            'smape': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(self.splitter.split(X, y)):
            logger.info(f"Cross-validation fold {fold + 1}/{self.n_splits}")
            
            # Split data
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Fit model
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_val)
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_val, y_pred)
            
            # Store scores
            for metric, value in metrics.items():
                cv_scores[metric].append(value)
        
        return cv_scores
    
    def _holdout_validate(self, model, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Perform holdout validation."""
        # Use last portion of data for holdout
        holdout_size = int(len(X) * self.test_size)
        gap_size = int(self.gap_days * 24)  # Assuming daily data
        
        # Split data
        train_end = len(X) - holdout_size - gap_size
        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        X_holdout = X.iloc[-holdout_size:]
        y_holdout = y.iloc[-holdout_size:]
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_holdout)
        
        # Calculate metrics
        holdout_metrics = self._calculate_metrics(y_holdout, y_pred)
        
        return holdout_metrics
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        # Ensure predictions are non-negative
        y_pred = np.maximum(y_pred, 0)
        
        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mape': self._calculate_mape(y_true, y_pred),
            'smape': self._calculate_smape(y_true, y_pred)
        }
        
        return metrics
    
    def _calculate_mape(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error."""
        # Avoid division by zero
        mask = y_true != 0
        if mask.sum() == 0:
            return 0.0
        
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    def _calculate_smape(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate Symmetric Mean Absolute Percentage Error."""
        numerator = np.abs(y_true - y_pred)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        
        # Avoid division by zero
        mask = denominator != 0
        if mask.sum() == 0:
            return 0.0
        
        return np.mean(numerator[mask] / denominator[mask]) * 100
    
    def _calculate_validation_summary(self, cv_results: Dict[str, List[float]], 
                                    holdout_results: Dict[str, float]) -> Dict[str, Any]:
        """Calculate validation summary statistics."""
        summary = {}
        
        # Cross-validation summary
        cv_summary = {}
        for metric, scores in cv_results.items():
            cv_summary[metric] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'scores': scores
            }
        
        summary['cross_validation'] = cv_summary
        
        # Holdout validation
        summary['holdout_validation'] = holdout_results
        
        # Overall performance score (weighted combination)
        cv_mape = cv_summary['mape']['mean']
        holdout_mape = holdout_results['mape']
        overall_score = 0.7 * cv_mape + 0.3 * holdout_mape
        
        summary['overall_performance_score'] = overall_score
        
        # Performance grade
        if overall_score < 10:
            grade = 'A'
        elif overall_score < 20:
            grade = 'B'
        elif overall_score < 30:
            grade = 'C'
        else:
            grade = 'D'
        
        summary['performance_grade'] = grade
        
        return summary
    
    def validate_by_city_sku(self, model, data: pd.DataFrame, target_city: str) -> Dict[str, Any]:
        """
        Validate model for each city-SKU combination.
        
        Args:
            model: Model to validate
            data: Full dataset with features and target
            target_city: Target city for forecasting
        
        Returns:
            Dictionary containing validation results for each city-SKU
        """
        validation_results = {}
        
        # Get unique city-SKU combinations
        city_sku_combinations = data.groupby(['city', 'sku_id']).size().reset_index()
        
        for _, row in city_sku_combinations.iterrows():
            city = row['city']
            sku_id = row['sku_id']
            
            # Filter data for this city-SKU combination
            mask = (data['city'] == city) & (data['sku_id'] == sku_id)
            city_sku_data = data[mask].copy()
            
            if len(city_sku_data) < 20:  # Skip if insufficient data
                logger.warning(f"Insufficient data for {city}-{sku_id}: {len(city_sku_data)} samples")
                continue
            
            # Prepare features and target
            feature_columns = [col for col in city_sku_data.columns 
                             if col not in ['date', 'city', 'sku_id', 'units_sold']]
            X = city_sku_data[feature_columns]
            y = city_sku_data['units_sold']
            
            # Validate model
            try:
                validation_result = self.validate_model(model, X, y, city, sku_id)
                validation_results[f"{city}_{sku_id}"] = validation_result
                
            except Exception as e:
                logger.error(f"Validation failed for {city}-{sku_id}: {str(e)}")
                continue
        
        return validation_results
    
    def generate_validation_report(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        report = {
            'summary': {},
            'by_city': {},
            'by_sku': {},
            'overall_metrics': {}
        }
        
        # Calculate overall metrics
        all_scores = []
        city_scores = {}
        sku_scores = {}
        
        for key, result in validation_results.items():
            city, sku_id = key.split('_', 1)
            score = result['overall_performance_score']
            grade = result['performance_grade']
            
            all_scores.append(score)
            
            # By city
            if city not in city_scores:
                city_scores[city] = []
            city_scores[city].append(score)
            
            # By SKU
            if sku_id not in sku_scores:
                sku_scores[sku_id] = []
            sku_scores[sku_id].append(score)
        
        # Overall summary
        report['summary'] = {
            'total_combinations': len(validation_results),
            'mean_score': np.mean(all_scores),
            'std_score': np.std(all_scores),
            'min_score': np.min(all_scores),
            'max_score': np.max(all_scores),
            'grade_distribution': self._calculate_grade_distribution(validation_results)
        }
        
        # By city summary
        for city, scores in city_scores.items():
            report['by_city'][city] = {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'count': len(scores)
            }
        
        # By SKU summary
        for sku_id, scores in sku_scores.items():
            report['by_sku'][sku_id] = {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'count': len(scores)
            }
        
        return report
    
    def _calculate_grade_distribution(self, validation_results: Dict[str, Any]) -> Dict[str, int]:
        """Calculate grade distribution."""
        grades = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        
        for result in validation_results.values():
            grade = result['performance_grade']
            grades[grade] += 1
        
        return grades


class ModelSelectionValidator:
    """Model selection and hyperparameter tuning validator."""
    
    def __init__(self, config: Dict):
        """Initialize model selection validator."""
        self.config = config
        self.validator = TimeSeriesValidator(config)
    
    def select_best_model(self, models: Dict[str, Any], X: pd.DataFrame, y: pd.Series) -> Tuple[str, Any]:
        """
        Select the best model based on validation performance.
        
        Args:
            models: Dictionary of model names and model instances
            X: Feature matrix
            y: Target variable
        
        Returns:
            Tuple of (best_model_name, best_model)
        """
        best_score = float('inf')
        best_model_name = None
        best_model = None
        
        for model_name, model in models.items():
            logger.info(f"Validating model: {model_name}")
            
            try:
                validation_result = self.validator.validate_model(model, X, y)
                score = validation_result['overall_performance_score']
                
                logger.info(f"{model_name} validation score: {score:.4f}")
                
                if score < best_score:
                    best_score = score
                    best_model_name = model_name
                    best_model = model
                    
            except Exception as e:
                logger.error(f"Validation failed for {model_name}: {str(e)}")
                continue
        
        logger.info(f"Best model: {best_model_name} with score: {best_score:.4f}")
        
        return best_model_name, best_model


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
    
    # Test validation framework
    validator = TimeSeriesValidator(config)
    
    # Create a simple model for testing
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    
    # Validate model
    validation_result = validator.validate_model(model, X, y)
    
    print("Validation Results:")
    print(f"Overall Performance Score: {validation_result['overall_performance_score']:.4f}")
    print(f"Performance Grade: {validation_result['performance_grade']}")
    
    print("\nCross-Validation Results:")
    for metric, stats in validation_result['cross_validation'].items():
        print(f"{metric.upper()}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    print("\nHoldout Validation Results:")
    for metric, value in validation_result['holdout_validation'].items():
        print(f"{metric.upper()}: {value:.4f}")

