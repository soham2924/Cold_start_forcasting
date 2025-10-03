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
    
    def __init__(self, config: Dict):
        self.config = config
        self.feature_config = config['features']
        self.scalers = {}
        self.encoders = {}
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        else:
            raise ValueError("Date column not found")

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

        df['days_since_start'] = (df['date'] - df['date'].min()).dt.days
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        df['is_quarter_start'] = df['date'].dt.is_quarter_start.astype(int)
        df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)
        
        logger.info(f"Created {len([col for col in df.columns if col not in ['date', 'city', 'sku_id', 'units_sold']])} time features")
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str = 'units_sold') -> pd.DataFrame:
        if not self.feature_config['lags']['enabled']:
            return df
        
        df = df.copy()
        lag_periods = self.feature_config['lags']['periods']

        df = df.sort_values(['city', 'sku_id', 'date'])
        
        for lag in lag_periods:
            df[f'{target_col}_lag_{lag}'] = df.groupby(['city', 'sku_id'])[target_col].shift(lag)

        for lag in [1, 7, 14]:
            if f'{target_col}_lag_{lag}' in df.columns:
                df[f'{target_col}_lag_{lag}_ratio'] = df[target_col] / (df[f'{target_col}_lag_{lag}'] + 1e-8)
        
        logger.info(f"Created {len(lag_periods)} lag features")
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, target_col: str = 'units_sold') -> pd.DataFrame:
        if not self.feature_config['rolling_windows']['enabled']:
            return df
        
        df = df.copy()
        windows = self.feature_config['rolling_windows']['windows']
        functions = self.feature_config['rolling_windows']['functions']
        
        df = df.sort_values(['city', 'sku_id', 'date']).reset_index(drop=True)
        
        for window in windows:
            for func in functions:
                rolling_result = df.groupby(['city', 'sku_id'])[target_col].rolling(window=window, min_periods=1)
                
                if func == 'mean':
                    df[f'{target_col}_rolling_{window}_mean'] = rolling_result.mean().values
                elif func == 'std':
                    df[f'{target_col}_rolling_{window}_std'] = rolling_result.std().values
                elif func == 'min':
                    df[f'{target_col}_rolling_{window}_min'] = rolling_result.min().values
                elif func == 'max':
                    df[f'{target_col}_rolling_{window}_max'] = rolling_result.max().values

        for window in [7, 14]:
            if f'{target_col}_rolling_{window}_mean' in df.columns:
                df[f'{target_col}_vs_rolling_{window}_mean'] = df[target_col] / (df[f'{target_col}_rolling_{window}_mean'] + 1e-8)
        
        logger.info(f"Created rolling features for windows: {windows}")
        
        return df
    
    def create_external_regressors(self, df: pd.DataFrame, external_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        df = df.copy()
        regressors = self.feature_config['external_regressors']
    
        if 'weather' in external_data and not external_data['weather'].empty:
            weather_cols = [col for col in regressors if col in external_data['weather'].columns]
            if weather_cols:
                df = df.merge(
                    external_data['weather'][['city', 'date'] + weather_cols],
                    on=['city', 'date'],
                    how='left'
                )
                logger.info(f"Added weather features: {weather_cols}")
        
        if 'holidays' in external_data and not external_data['holidays'].empty:
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
        if 'promos' in external_data and not external_data['promos'].empty:
            promo_cols = [col for col in regressors if col in external_data['promos'].columns]
            if promo_cols:
                df = df.merge(
                    external_data['promos'][['city', 'sku_id', 'date'] + promo_cols],
                    on=['city', 'sku_id', 'date'],
                    how='left'
                )
                logger.info(f"Added promo features: {promo_cols}")
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
        df = df.copy()

        city_stats = df.groupby(['city', 'date'])[target_col].agg(['mean', 'std', 'sum']).reset_index()
        city_stats.columns = ['city', 'date', 'city_mean_sales', 'city_std_sales', 'city_total_sales']
        df = df.merge(city_stats, on=['city', 'date'], how='left')

        sku_stats = df.groupby(['sku_id', 'date'])[target_col].agg(['mean', 'std', 'sum']).reset_index()
        sku_stats.columns = ['sku_id', 'date', 'sku_mean_sales', 'sku_std_sales', 'sku_total_sales']
        df = df.merge(sku_stats, on=['sku_id', 'date'], how='left')

        df['city_sku_ratio'] = df[target_col] / (df['city_mean_sales'] + 1e-8)
        df['sku_city_ratio'] = df[target_col] / (df['sku_mean_sales'] + 1e-8)
        
        logger.info("Created cross-sectional features")
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        categorical_columns = ['city', 'sku_id', 'holiday_type']
        
        for col in categorical_columns:
            if col in df.columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.encoders[col].fit_transform(df[col].astype(str))
                else:
                    unique_values = df[col].astype(str).unique()
                    known_values = self.encoders[col].classes_
                    unseen_values = set(unique_values) - set(known_values)
                    
                    if unseen_values:
                        all_values = np.concatenate([known_values, list(unseen_values)])
                        self.encoders[col] = LabelEncoder()
                        self.encoders[col].fit(all_values)
                    
                    df[f'{col}_encoded'] = self.encoders[col].transform(df[col].astype(str))
        
        return df
    
    def scale_features(self, df: pd.DataFrame, fit_scalers: bool = True) -> pd.DataFrame:
        df = df.copy()
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
        logger.info("Starting feature engineering...")

        df = self.create_time_features(df)
        df = self.create_lag_features(df)

        df = self.create_rolling_features(df)

        if external_data:
            df = self.create_external_regressors(df, external_data)

        df = self.create_cross_sectional_features(df)

        df = self.encode_categorical_features(df)

        df = self.scale_features(df, fit_scalers=fit_scalers)
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_columns = [col for col in numeric_columns if col not in ['date', 'city', 'sku_id', 'units_sold']]

        df_final = df[['date', 'city', 'sku_id', 'units_sold'] + feature_columns].copy()

        for col in feature_columns:
            if col in df_final.columns:
                df_final[col] = df_final[col].astype(float)
        
        logger.info(f"Feature engineering completed. Total features: {len(feature_columns)}")
        logger.info(f"Feature columns: {feature_columns}")
        
        return df_final
    
    def get_feature_importance_names(self) -> List[str]:
        return [
            'year', 'month_sin', 'month_cos', 'week_sin', 'week_cos',
            'day_sin', 'day_cos', 'is_weekend', 'quarter_sin', 'quarter_cos',
            'units_sold_lag_1', 'units_sold_lag_7', 'units_sold_lag_14',
            'units_sold_rolling_7_mean', 'units_sold_rolling_14_mean',
            'price', 'promo_discount', 'temperature', 'humidity', 'precipitation',
            'is_holiday', 'city_mean_sales', 'sku_mean_sales'
        ]


if __name__ == "__main__":
    import yaml
    
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
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
    fe = FeatureEngineer(config)
    df_features = fe.engineer_all_features(df)
    
    print(f"Original shape: {df.shape}")
    print(f"Features shape: {df_features.shape}")
    print(f"Feature columns: {[col for col in df_features.columns if col not in ['date', 'city', 'sku_id', 'units_sold']]}")

