"""
Feature engineering module for retail demand forecasting.
Creates time features, lag features, rolling window features, and external regressors.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Handles feature engineering for retail demand forecasting."""
    
    def __init__(self, config: Dict):
        """Initialize FeatureEngineer with configuration."""
        self.config = config
        self.feature_config = config['features']
        self.scalers = {}
        self.encoders = {}
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features from date column."""
        df = df.copy()
        
        # Ensure date column is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        else:
            raise ValueError("Date column not found")
        
        # Extract time features
        time_features = self.feature_config['time_features']
        
        if 'year' in time_features:
            df['year'] = df['date'].dt.year
        
        if 'month' in time_features:
            df['month'] = df['date'].dt.month
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        if 'week_of_year' in time_features:
            df['week_of_year'] = df['date'].dt.isocalendar().week
            df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
            df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
        
        if 'day_of_week' in time_features:
            df['day_of_week'] = df['date'].dt.dayofweek
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        if 'is_weekend' in time_features:
            df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)
        
        if 'quarter' in time_features:
            df['quarter'] = df['date'].dt.quarter
            df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
            df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
        
        # Additional time features
        df['days_since_start'] = (df['date'] - df['date'].min()).dt.days
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        df['is_quarter_start'] = df['date'].dt.is_quarter_start.astype(int)
        df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)
        
        logger.info(f"Created {len([col for col in df.columns if col not in ['date', 'city', 'sku_id', 'units_sold']])} time features")
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str = 'units_sold') -> pd.DataFrame:
        """Create lag features for the target variable."""
        if not self.feature_config['lags']['enabled']:
            return df
        
        df = df.copy()
        lag_periods = self.feature_config['lags']['periods']
        
        # Sort by city, sku, and date
        df = df.sort_values(['city', 'sku_id', 'date'])
        
        for lag in lag_periods:
            df[f'{target_col}_lag_{lag}'] = df.groupby(['city', 'sku_id'])[target_col].shift(lag)
        
        # Create lag ratios
        for lag in [1, 7, 14]:
            if f'{target_col}_lag_{lag}' in df.columns:
                df[f'{target_col}_lag_{lag}_ratio'] = df[target_col] / (df[f'{target_col}_lag_{lag}'] + 1e-8)
        
        logger.info(f"Created {len(lag_periods)} lag features")
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, target_col: str = 'units_sold') -> pd.DataFrame:
        """Create rolling window features."""
        if not self.feature_config['rolling_windows']['enabled']:
            return df
        
        df = df.copy()
        windows = self.feature_config['rolling_windows']['windows']
        functions = self.feature_config['rolling_windows']['functions']
        
        # Sort by city, sku, and date and reset index
        df = df.sort_values(['city', 'sku_id', 'date']).reset_index(drop=True)
        
        for window in windows:
            for func in functions:
                # Create rolling features with proper index handling
                rolling_result = df.groupby(['city', 'sku_id'])[target_col].rolling(window=window, min_periods=1)
                
                if func == 'mean':
                    df[f'{target_col}_rolling_{window}_mean'] = rolling_result.mean().values
                elif func == 'std':
                    df[f'{target_col}_rolling_{window}_std'] = rolling_result.std().values
                elif func == 'min':
                    df[f'{target_col}_rolling_{window}_min'] = rolling_result.min().values
                elif func == 'max':
                    df[f'{target_col}_rolling_{window}_max'] = rolling_result.max().values
        
        # Create rolling ratios
        for window in [7, 14]:
            if f'{target_col}_rolling_{window}_mean' in df.columns:
                df[f'{target_col}_vs_rolling_{window}_mean'] = df[target_col] / (df[f'{target_col}_rolling_{window}_mean'] + 1e-8)
        
        logger.info(f"Created rolling features for windows: {windows}")
        
        return df
    
    def create_external_regressors(self, df: pd.DataFrame, external_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Add external regressors to the dataset."""
        df = df.copy()
        regressors = self.feature_config['external_regressors']
        
        # Merge weather data
        if 'weather' in external_data and not external_data['weather'].empty:
            weather_cols = [col for col in regressors if col in external_data['weather'].columns]
            if weather_cols:
                df = df.merge(
                    external_data['weather'][['city', 'date'] + weather_cols],
                    on=['city', 'date'],
                    how='left'
                )
                logger.info(f"Added weather features: {weather_cols}")
        
        # Merge holiday data
        if 'holidays' in external_data and not external_data['holidays'].empty:
            # Create holiday features
            df['is_holiday'] = 0
            df['holiday_type'] = 'None'
            
            for _, holiday in external_data['holidays'].iterrows():
                mask = df['date'] == holiday['date']
                if 'city' in holiday and holiday['city'] != 'All':
                    mask = mask & (df['city'] == holiday['city'])
                
                df.loc[mask, 'is_holiday'] = 1
                if 'holiday_type' in holiday:
                    df.loc[mask, 'holiday_type'] = holiday['holiday_type']
            
            logger.info("Added holiday features")
        
        # Merge promotional data
        if 'promos' in external_data and not external_data['promos'].empty:
            promo_cols = [col for col in regressors if col in external_data['promos'].columns]
            if promo_cols:
                df = df.merge(
                    external_data['promos'][['city', 'sku_id', 'date'] + promo_cols],
                    on=['city', 'sku_id', 'date'],
                    how='left'
                )
                logger.info(f"Added promo features: {promo_cols}")
        
        # Fill missing values for external regressors
        for col in regressors:
            if col in df.columns:
                if col in ['is_holiday']:
                    df[col] = df[col].fillna(0)
                elif col in ['holiday_type']:
                    df[col] = df[col].fillna('None')
                else:
                    df[col] = df[col].fillna(df[col].median())
        
        return df
    
    def create_cross_sectional_features(self, df: pd.DataFrame, target_col: str = 'units_sold') -> pd.DataFrame:
        """Create cross-sectional features (city and SKU level aggregations)."""
        df = df.copy()
        
        # City-level features
        city_stats = df.groupby(['city', 'date'])[target_col].agg(['mean', 'std', 'sum']).reset_index()
        city_stats.columns = ['city', 'date', 'city_mean_sales', 'city_std_sales', 'city_total_sales']
        df = df.merge(city_stats, on=['city', 'date'], how='left')
        
        # SKU-level features across cities
        sku_stats = df.groupby(['sku_id', 'date'])[target_col].agg(['mean', 'std', 'sum']).reset_index()
        sku_stats.columns = ['sku_id', 'date', 'sku_mean_sales', 'sku_std_sales', 'sku_total_sales']
        df = df.merge(sku_stats, on=['sku_id', 'date'], how='left')
        
        # City-SKU interaction features
        df['city_sku_ratio'] = df[target_col] / (df['city_mean_sales'] + 1e-8)
        df['sku_city_ratio'] = df[target_col] / (df['sku_mean_sales'] + 1e-8)
        
        logger.info("Created cross-sectional features")
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features."""
        df = df.copy()
        
        categorical_columns = ['city', 'sku_id', 'holiday_type']
        
        for col in categorical_columns:
            if col in df.columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.encoders[col].fit_transform(df[col].astype(str))
                else:
                    # Handle unseen categories
                    unique_values = df[col].astype(str).unique()
                    known_values = self.encoders[col].classes_
                    unseen_values = set(unique_values) - set(known_values)
                    
                    if unseen_values:
                        # Add unseen values to encoder
                        all_values = np.concatenate([known_values, list(unseen_values)])
                        self.encoders[col] = LabelEncoder()
                        self.encoders[col].fit(all_values)
                    
                    df[f'{col}_encoded'] = self.encoders[col].transform(df[col].astype(str))
        
        return df
    
    def scale_features(self, df: pd.DataFrame, fit_scalers: bool = True) -> pd.DataFrame:
        """Scale numerical features."""
        df = df.copy()
        
        # Identify numerical columns to scale
        numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_columns = ['date', 'units_sold', 'city_encoded', 'sku_id_encoded', 'holiday_type_encoded']
        scale_columns = [col for col in numerical_columns if col not in exclude_columns]
        
        for col in scale_columns:
            if fit_scalers:
                if col not in self.scalers:
                    self.scalers[col] = StandardScaler()
                    df[f'{col}_scaled'] = self.scalers[col].fit_transform(df[[col]])
                else:
                    df[f'{col}_scaled'] = self.scalers[col].transform(df[[col]])
            else:
                if col in self.scalers:
                    df[f'{col}_scaled'] = self.scalers[col].transform(df[[col]])
        
        return df
    
    def engineer_all_features(self, df: pd.DataFrame, external_data: Dict[str, pd.DataFrame] = None, 
                            fit_scalers: bool = True) -> pd.DataFrame:
        """Apply all feature engineering steps."""
        logger.info("Starting feature engineering...")
        
        # Start with time features
        df = self.create_time_features(df)
        
        # Add lag features
        df = self.create_lag_features(df)
        
        # Add rolling features
        df = self.create_rolling_features(df)
        
        # Add external regressors
        if external_data:
            df = self.create_external_regressors(df, external_data)
        
        # Add cross-sectional features
        df = self.create_cross_sectional_features(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df)
        
        # Scale features
        df = self.scale_features(df, fit_scalers=fit_scalers)
        
        # Remove original date column and keep only engineered features
        # Keep only numeric columns for model training
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_columns = [col for col in numeric_columns if col not in ['date', 'city', 'sku_id', 'units_sold']]
        
        # Create final dataset with only numeric features
        df_final = df[['date', 'city', 'sku_id', 'units_sold'] + feature_columns].copy()
        
        # Convert all numeric columns to float to avoid dtype issues
        for col in feature_columns:
            if col in df_final.columns:
                df_final[col] = df_final[col].astype(float)
        
        logger.info(f"Feature engineering completed. Total features: {len(feature_columns)}")
        logger.info(f"Feature columns: {feature_columns}")
        
        return df_final
    
    def get_feature_importance_names(self) -> List[str]:
        """Get list of feature names for importance analysis."""
        # This would be populated after feature engineering
        return [
            'year', 'month_sin', 'month_cos', 'week_sin', 'week_cos',
            'day_sin', 'day_cos', 'is_weekend', 'quarter_sin', 'quarter_cos',
            'units_sold_lag_1', 'units_sold_lag_7', 'units_sold_lag_14',
            'units_sold_rolling_7_mean', 'units_sold_rolling_14_mean',
            'price', 'promo_discount', 'temperature', 'humidity', 'precipitation',
            'is_holiday', 'city_mean_sales', 'sku_mean_sales'
        ]


if __name__ == "__main__":
    # Example usage
    import yaml
    
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # Create sample data
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    cities = ['Mumbai', 'Delhi', 'Jaipur']
    skus = ['SKU001', 'SKU002', 'SKU003']
    
    data = []
    for date in dates:
        for city in cities:
            for sku in skus:
                data.append({
                    'date': date,
                    'city': city,
                    'sku_id': sku,
                    'units_sold': np.random.poisson(10)
                })
    
    df = pd.DataFrame(data)
    
    # Initialize feature engineer
    fe = FeatureEngineer(config)
    
    # Engineer features
    df_features = fe.engineer_all_features(df)
    
    print(f"Original shape: {df.shape}")
    print(f"Features shape: {df_features.shape}")
    print(f"Feature columns: {[col for col in df_features.columns if col not in ['date', 'city', 'sku_id', 'units_sold']]}")

