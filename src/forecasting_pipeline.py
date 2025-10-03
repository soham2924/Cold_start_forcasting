import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import yaml
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from .data_loader import DataLoader
from .feature_engineering import FeatureEngineer
from .models import TransferLearningModel, create_model
from .validation import TimeSeriesValidator, ModelSelectionValidator
from .prediction_intervals import PredictionIntervalGenerator
from .explainability import DriverAttribution

logger = logging.getLogger(__name__)


class ForecastingPipeline:
    """Main forecasting pipeline for retail demand forecasting."""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.data_loader = DataLoader(config_path)
        self.feature_engineer = FeatureEngineer(self.config)
        self.validator = TimeSeriesValidator(self.config)
        self.model_selector = ModelSelectionValidator(self.config)
        self.prediction_intervals = PredictionIntervalGenerator(self.config)
        self.driver_attribution = DriverAttribution(self.config)

        self.data = {}
        self.features = None
        self.model = None
        self.forecasts = {}
        self.validation_results = {}
        self.explainability_results = {}

        self._setup_logging()
        
        logger.info("Forecasting pipeline initialized successfully")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO'))
        log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.FileHandler(log_config.get('file', 'logs/forecasting.log')),
                logging.StreamHandler()
            ]
        )
    
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load all required data."""
        logger.info("Loading data...")
        
        self.data = self.data_loader.load_all_data()
        
        # Validate data availability
        target_city = self.config['data']['target_city']
        validation = self.data_loader.validate_data_availability(target_city)
        
        if not validation['target_city_exists']:
            raise ValueError(f"Target city {target_city} not found in data")
        
        if not validation['has_sufficient_data']:
            logger.warning(f"Limited data available for target city {target_city}")
        
        logger.info(f"Data loaded successfully. Quality score: {validation['data_quality_score']:.2f}")
        
        return self.data
    
    def engineer_features(self) -> pd.DataFrame:
        """Engineer features for forecasting."""
        logger.info("Engineering features...")
        
        # Start with sales data
        sales_data = self.data['sales'].copy()
        
        # Engineer features
        self.features = self.feature_engineer.engineer_all_features(
            sales_data, 
            self.data, 
            fit_scalers=True
        )
        
        logger.info(f"Feature engineering completed. Shape: {self.features.shape}")
        
        return self.features
    
    def prepare_training_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Prepare training data for transfer learning."""
        logger.info("Preparing training data for transfer learning...")
        
        target_city = self.config['data']['target_city']
        
        # Split data into source (other cities) and target (Jaipur)
        source_data = self.features[self.features['city'] != target_city].copy()
        target_data = self.features[self.features['city'] == target_city].copy()
        
        # Prepare features and target
        feature_columns = [col for col in self.features.columns 
                         if col not in ['date', 'city', 'sku_id', 'units_sold']]
        
        # Source data
        X_source = source_data[feature_columns]
        y_source = source_data['units_sold']
        
        # Target data
        X_target = target_data[feature_columns]
        y_target = target_data['units_sold']
        
        logger.info(f"Source data: {X_source.shape}, Target data: {X_target.shape}")
        
        return X_source, y_source, X_target, y_target
    
    def train_model(self) -> TransferLearningModel:
        """Train transfer learning model."""
        logger.info("Training transfer learning model...")
        
        # Prepare training data
        X_source, y_source, X_target, y_target = self.prepare_training_data()
        
        # Create transfer learning model
        self.model = TransferLearningModel(self.config)
        
        # Train source model
        self.model.fit_source_model(X_source, y_source)
        
        # Train target model with transfer learning
        self.model.fit_target_model(X_target, y_target)
        
        logger.info("Transfer learning model trained successfully")
        
        return self.model
    
    def validate_model(self) -> Dict[str, Any]:
        """Validate the trained model."""
        logger.info("Validating model...")
        
        # Prepare validation data
        X_source, y_source, X_target, y_target = self.prepare_training_data()
        
        # Validate on target city data
        validation_results = self.validator.validate_by_city_sku(
            self.model, 
            self.features[self.features['city'] == self.config['data']['target_city']], 
            self.config['data']['target_city']
        )
        
        # Generate validation report
        validation_report = self.validator.generate_validation_report(validation_results)
        
        self.validation_results = {
            'detailed_results': validation_results,
            'summary_report': validation_report
        }
        
        logger.info(f"Model validation completed. Overall performance: {validation_report['summary']['mean_score']:.4f}")
        
        return self.validation_results
    
    def generate_forecasts(self) -> Dict[str, Any]:
        """Generate forecasts for the next 13 weeks."""
        logger.info("Generating forecasts...")
        
        target_city = self.config['data']['target_city']
        forecast_horizon = self.config['data']['forecast_horizon_weeks']
        
        # Get all SKUs for target city
        target_data = self.features[self.features['city'] == target_city]
        sku_list = target_data['sku_id'].unique()
        
        forecasts = {}
        
        for sku_id in sku_list:
            logger.info(f"Forecasting for SKU: {sku_id}")
            
            # Get historical data for this SKU
            sku_data = target_data[target_data['sku_id'] == sku_id].copy()
            
            if len(sku_data) < 7:  # Need at least 1 week of data
                logger.warning(f"Insufficient data for SKU {sku_id}")
                continue
            
            # Generate future dates
            last_date = sku_data['date'].max()
            future_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=forecast_horizon * 7,
                freq='D'
            )
            
            # Create future features
            future_features = self._create_future_features(
                sku_data, future_dates, target_city, sku_id
            )
            
            # Make predictions
            predictions = self.model.predict(future_features)
            
            # Generate prediction intervals
            prediction_intervals = self._generate_prediction_intervals(future_features)
            
            # Store forecasts
            forecasts[sku_id] = {
                'dates': future_dates,
                'predictions': predictions,
                'prediction_intervals': prediction_intervals,
                'sku_id': sku_id,
                'city': target_city
            }
        
        self.forecasts = forecasts
        logger.info(f"Forecasts generated for {len(forecasts)} SKUs")
        
        return self.forecasts
    
    def _create_future_features(self, historical_data: pd.DataFrame, 
                               future_dates: pd.DatetimeIndex, 
                               city: str, sku_id: str) -> pd.DataFrame:
        """Create features for future dates."""
        # Start with future dates
        future_data = pd.DataFrame({
            'date': future_dates,
            'city': city,
            'sku_id': sku_id,
            'units_sold': 0  # Placeholder
        })
        
        # Engineer features for future dates
        future_features = self.feature_engineer.engineer_all_features(
            future_data, 
            self.data, 
            fit_scalers=False  # Use existing scalers
        )
        
        # Remove target column
        feature_columns = [col for col in future_features.columns 
                         if col not in ['date', 'city', 'sku_id', 'units_sold']]
        
        return future_features[feature_columns]
    
    def _generate_prediction_intervals(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Generate prediction intervals for forecasts."""
        try:
            # Fit prediction intervals
            X_source, y_source, X_target, y_target = self.prepare_training_data()
            self.prediction_intervals.fit(self.model, X_target, y_target)
            
            # Generate intervals
            intervals = self.prediction_intervals.predict_intervals(X)
            
            return intervals
            
        except Exception as e:
            logger.warning(f"Failed to generate prediction intervals: {str(e)}")
            return {}
    
    def analyze_drivers(self) -> Dict[str, Any]:
        """Analyze drivers of demand."""
        logger.info("Analyzing demand drivers...")
        
        # Prepare data for driver analysis
        X_source, y_source, X_target, y_target = self.prepare_training_data()
        
        # Analyze drivers
        driver_analysis = self.driver_attribution.analyze_drivers(
            self.model, 
            pd.concat([X_source, X_target]), 
            pd.concat([y_source, y_target])
        )
        
        # Generate driver report
        driver_report = self.driver_attribution.generate_driver_report(driver_analysis)
        
        self.explainability_results = {
            'driver_analysis': driver_analysis,
            'driver_report': driver_report
        }
        
        logger.info("Driver analysis completed")
        
        return self.explainability_results
    
    def save_results(self):
        """Save all results to files."""
        logger.info("Saving results...")
        
        output_path = Path(self.config['data']['output_path'])
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save forecasts
        self._save_forecasts(output_path)
        
        # Save model
        self._save_model(output_path)
        
        # Save validation results
        self._save_validation_results(output_path)
        
        # Save explainability results
        self._save_explainability_results(output_path)
        
        logger.info("Results saved successfully")
    
    def _save_forecasts(self, output_path: Path):
        """Save forecasts to CSV."""
        forecast_data = []
        
        for sku_id, forecast in self.forecasts.items():
            for i, date in enumerate(forecast['dates']):
                row = {
                    'date': date,
                    'city': forecast['city'],
                    'sku_id': sku_id,
                    'forecast': forecast['predictions'][i]
                }
                
                # Add prediction intervals
                for conf_name, intervals in forecast['prediction_intervals'].items():
                    row[f'lower_{conf_name}'] = intervals['lower'][i]
                    row[f'upper_{conf_name}'] = intervals['upper'][i]
                
                forecast_data.append(row)
        
        # Save to CSV
        forecast_df = pd.DataFrame(forecast_data)
        forecast_df.to_csv(output_path / 'forecasts.csv', index=False)
        
        logger.info(f"Forecasts saved to {output_path / 'forecasts.csv'}")
    
    def _save_model(self, output_path: Path):
        """Save trained model."""
        model_path = output_path / 'trained_model.pkl'
        joblib.dump(self.model, model_path)
        
        logger.info(f"Model saved to {model_path}")
    
    def _save_validation_results(self, output_path: Path):
        """Save validation results."""
        validation_path = output_path / 'validation_results.pkl'
        joblib.dump(self.validation_results, validation_path)
        
        logger.info(f"Validation results saved to {validation_path}")
    
    def _save_explainability_results(self, output_path: Path):
        """Save explainability results."""
        explainability_path = output_path / 'explainability_results.pkl'
        joblib.dump(self.explainability_results, explainability_path)
        
        logger.info(f"Explainability results saved to {explainability_path}")
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        logger.info("Starting full forecasting pipeline...")
        
        try:
            # Step 1: Load data
            self.load_data()
            
            # Step 2: Engineer features
            self.engineer_features()
            
            # Step 3: Train model
            self.train_model()
            
            # Step 4: Validate model
            self.validate_model()
            
            # Step 5: Generate forecasts
            self.generate_forecasts()
            
            # Step 6: Analyze drivers
            self.analyze_drivers()
            
            # Step 7: Save results
            self.save_results()
            
            logger.info("Full forecasting pipeline completed successfully")
            
            return {
                'status': 'success',
                'forecasts': self.forecasts,
                'validation_results': self.validation_results,
                'explainability_results': self.explainability_results
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }


if __name__ == "__main__":

    pipeline = ForecastingPipeline()

    results = pipeline.run_full_pipeline()
    
    if results['status'] == 'success':
        print("Pipeline completed successfully!")
        print(f"Generated forecasts for {len(results['forecasts'])} SKUs")
    else:
        print(f"Pipeline failed: {results['error']}")

