"""
Model implementations for retail demand forecasting with transfer learning.
Includes LightGBM, XGBoost, CatBoost, and ensemble models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
import joblib
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

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

logger = logging.getLogger(__name__)


class BaseModel:
    """Base class for all forecasting models."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.feature_columns = None
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Fit the model."""
        raise NotImplementedError
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        raise NotImplementedError
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance."""
        raise NotImplementedError
    
    def save_model(self, path: str):
        """Save the model."""
        joblib.dump(self, path)
    
    def load_model(self, path: str):
        """Load the model."""
        return joblib.load(path)


class LightGBMModel(BaseModel):
    """LightGBM implementation for demand forecasting."""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not available")
        
        # Default parameters
        self.params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': self.config.get('random_seed', 42)
        }
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Fit LightGBM model."""
        self.feature_columns = X.columns.tolist()
        
        # Create LightGBM dataset
        train_data = lgb.Dataset(X, label=y)
        
        # Train model
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        
        self.is_fitted = True
        logger.info("LightGBM model fitted successfully")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with LightGBM."""
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from LightGBM."""
        if not self.is_fitted:
            return {}
        
        importance = self.model.feature_importance(importance_type='gain')
        return dict(zip(self.feature_columns, importance))


class XGBoostModel(BaseModel):
    """XGBoost implementation for demand forecasting."""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not available")
        
        # Default parameters
        self.params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.9,
            'random_state': self.config.get('random_seed', 42)
        }
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Fit XGBoost model."""
        self.feature_columns = X.columns.tolist()
        
        # Create XGBoost DMatrix
        dtrain = xgb.DMatrix(X, label=y)
        
        # Train model
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=1000,
            evals=[(dtrain, 'train')],
            early_stopping_rounds=100,
            verbose_eval=False
        )
        
        self.is_fitted = True
        logger.info("XGBoost model fitted successfully")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with XGBoost."""
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from XGBoost."""
        if not self.is_fitted:
            return {}
        
        importance = self.model.get_score(importance_type='gain')
        return importance


class CatBoostModel(BaseModel):
    """CatBoost implementation for demand forecasting."""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost is not available")
        
        # Default parameters
        self.params = {
            'iterations': 1000,
            'learning_rate': 0.05,
            'depth': 6,
            'l2_leaf_reg': 3,
            'random_seed': self.config.get('random_seed', 42),
            'verbose': False
        }
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Fit CatBoost model."""
        self.feature_columns = X.columns.tolist()
        
        # Train model
        self.model = CatBoostRegressor(**self.params)
        self.model.fit(X, y, eval_set=(X, y), early_stopping_rounds=100, verbose=False)
        
        self.is_fitted = True
        logger.info("CatBoost model fitted successfully")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with CatBoost."""
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from CatBoost."""
        if not self.is_fitted:
            return {}
        
        importance = self.model.get_feature_importance()
        return dict(zip(self.feature_columns, importance))


class EnsembleModel(BaseModel):
    """Ensemble model combining multiple base models."""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.models = {}
        self.weights = config['model']['ensemble']['weights']
        self.model_names = config['model']['ensemble']['models']
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Fit all ensemble models."""
        self.feature_columns = X.columns.tolist()
        
        for model_name in self.model_names:
            if model_name == 'lightgbm' and LIGHTGBM_AVAILABLE:
                self.models[model_name] = LightGBMModel(self.config)
            elif model_name == 'xgboost' and XGBOOST_AVAILABLE:
                self.models[model_name] = XGBoostModel(self.config)
            elif model_name == 'catboost' and CATBOOST_AVAILABLE:
                self.models[model_name] = CatBoostModel(self.config)
            else:
                logger.warning(f"Model {model_name} is not available, skipping...")
                continue
            
            # Fit the model
            self.models[model_name].fit(X, y, **kwargs)
            logger.info(f"Fitted {model_name} model")
        
        self.is_fitted = True
        logger.info(f"Ensemble model fitted with {len(self.models)} models")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions."""
        predictions = []
        weights = []
        
        for i, (model_name, model) in enumerate(self.models.items()):
            pred = model.predict(X)
            predictions.append(pred)
            weights.append(self.weights[i])
        
        # Weighted average
        predictions = np.array(predictions)
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize weights
        
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        return ensemble_pred
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get ensemble feature importance."""
        all_importance = {}
        
        for i, (model_name, model) in enumerate(self.models.items()):
            importance = model.get_feature_importance()
            weight = self.weights[i]
            
            for feature, imp in importance.items():
                if feature not in all_importance:
                    all_importance[feature] = 0
                all_importance[feature] += imp * weight
        
        return all_importance


class TransferLearningModel:
    """Transfer learning model for cold-start forecasting."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.source_model = None
        self.target_model = None
        self.transfer_config = config['transfer_learning']
    
    def fit_source_model(self, X_source: pd.DataFrame, y_source: pd.Series):
        """Fit model on source cities data."""
        logger.info("Fitting source model on all cities except target...")
        
        # Create ensemble model for source
        self.source_model = EnsembleModel(self.config)
        self.source_model.fit(X_source, y_source)
        
        logger.info("Source model fitted successfully")
    
    def fit_target_model(self, X_target: pd.DataFrame, y_target: pd.Series):
        """Fit model on target city data with transfer learning."""
        logger.info("Fitting target model with transfer learning...")
        
        # Create target model
        self.target_model = EnsembleModel(self.config)
        
        # Get source predictions for target data
        source_predictions = self.source_model.predict(X_target)
        
        # Create transfer learning features
        X_transfer = X_target.copy()
        X_transfer['source_prediction'] = source_predictions
        X_transfer['prediction_residual'] = y_target - source_predictions
        
        # Fit target model
        self.target_model.fit(X_transfer, y_target)
        
        logger.info("Target model fitted with transfer learning")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using transfer learning."""
        # Get source predictions
        source_pred = self.source_model.predict(X)
        
        # Create transfer features
        X_transfer = X.copy()
        X_transfer['source_prediction'] = source_pred
        X_transfer['prediction_residual'] = 0  # No actual residual for new predictions
        
        # Get target predictions
        target_pred = self.target_model.predict(X_transfer)
        
        # Combine predictions with weights
        source_weight = self.transfer_config['source_cities_weight']
        target_weight = self.transfer_config['target_city_weight']
        
        final_pred = source_weight * source_pred + target_weight * target_pred
        
        return final_pred
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get combined feature importance."""
        source_importance = self.source_model.get_feature_importance()
        target_importance = self.target_model.get_feature_importance()
        
        # Combine importances
        combined_importance = {}
        all_features = set(source_importance.keys()) | set(target_importance.keys())
        
        for feature in all_features:
            source_imp = source_importance.get(feature, 0)
            target_imp = target_importance.get(feature, 0)
            
            source_weight = self.transfer_config['source_cities_weight']
            target_weight = self.transfer_config['target_city_weight']
            
            combined_importance[feature] = source_weight * source_imp + target_weight * target_imp
        
        return combined_importance


class ModelEvaluator:
    """Model evaluation utilities."""
    
    @staticmethod
    def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Evaluate model predictions."""
        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100,
            'smape': 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))
        }
        
        return metrics
    
    @staticmethod
    def time_series_cv_score(model, X: pd.DataFrame, y: pd.Series, 
                           cv_splits: int = 5) -> Dict[str, float]:
        """Perform time series cross-validation."""
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Fit model
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_val)
            
            # Calculate metrics
            metrics = ModelEvaluator.evaluate_predictions(y_val, y_pred)
            scores.append(metrics)
        
        # Average scores
        avg_scores = {}
        for metric in scores[0].keys():
            avg_scores[metric] = np.mean([score[metric] for score in scores])
        
        return avg_scores


def create_model(config: Dict) -> BaseModel:
    """Factory function to create models."""
    model_type = config['model']['primary_model']
    
    if model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
        return LightGBMModel(config)
    elif model_type == 'xgboost' and XGBOOST_AVAILABLE:
        return XGBoostModel(config)
    elif model_type == 'catboost' and CATBOOST_AVAILABLE:
        return CatBoostModel(config)
    elif model_type == 'ensemble':
        return EnsembleModel(config)
    else:
        # Default to LightGBM if available, otherwise XGBoost
        if LIGHTGBM_AVAILABLE:
            return LightGBMModel(config)
        elif XGBOOST_AVAILABLE:
            return XGBoostModel(config)
        else:
            raise ValueError("No suitable model available")


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
    y = pd.Series(np.random.randn(n_samples))
    
    # Test different models
    models_to_test = ['lightgbm', 'xgboost', 'catboost', 'ensemble']
    
    for model_type in models_to_test:
        try:
            config['model']['primary_model'] = model_type
            model = create_model(config)
            
            # Fit and predict
            model.fit(X, y)
            predictions = model.predict(X)
            
            # Evaluate
            metrics = ModelEvaluator.evaluate_predictions(y, predictions)
            
            print(f"\n{model_type.upper()} Results:")
            for metric, value in metrics.items():
                print(f"  {metric.upper()}: {value:.4f}")
                
        except Exception as e:
            print(f"Error with {model_type}: {str(e)}")

