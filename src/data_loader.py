import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import yaml

logger = logging.getLogger(__name__)

class DataLoader:
    
    def __init__(self, config_path: str = "config.yaml"):

        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.data_config = self.config['data']
        self.input_path = Path(self.data_config['input_path'])
        self.target_city = self.data_config['target_city']
        
        self._create_directories()
    
    def _create_directories(self):

        directories = [
            self.data_config['output_path'],
            self.data_config['plots_path'],
            self.data_config['reports_path']
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

        Path("logs").mkdir(exist_ok=True)
    
    def load_sales_data(self) -> pd.DataFrame:
        file_path = self.input_path / self.data_config['sales_data']
        df = pd.read_csv(file_path)
            
        logger.info(f"Loaded sales data: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")

        df = self._preprocess_sales_data(df)
            
        return df
            
    
    def _preprocess_sales_data(self, df: pd.DataFrame) -> pd.DataFrame:

        if 'week_start' in df.columns:
            df['date'] = pd.to_datetime(df['week_start'])
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        elif 'Date' in df.columns:
            df['date'] = pd.to_datetime(df['Date'])
        else:
            raise ValueError("Date column not found in sales data")

        if 'market' in df.columns:
            df['city'] = df['market']
        if 'units' in df.columns:
            df['units_sold'] = df['units']

        required_columns = ['city', 'sku_id', 'units_sold']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
 
        df = df.sort_values(['city', 'date'])

        df = df.drop_duplicates(subset=['city', 'sku_id', 'date'])
        df['units_sold'] = df['units_sold'].fillna(0)
        
        return df
    
    def load_weather_data(self) -> pd.DataFrame:
        file_path = self.input_path / self.data_config['weather_data']
        df = pd.read_csv(file_path)
            
        logger.info(f"Loaded weather data: {df.shape}")

        df = self._preprocess_weather_data(df)
            
        return df

    def _preprocess_weather_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'week_start' in df.columns:
            df['date'] = pd.to_datetime(df['week_start'])
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        elif 'Date' in df.columns:
            df['date'] = pd.to_datetime(df['Date'])
        if 'market' in df.columns:
            df['city'] = df['market']
        if 'city' not in df.columns:
            raise ValueError("City column not found in weather data")
        df = df.sort_values(['city', 'date'])
        
        return df
    
    def load_holiday_data(self) -> pd.DataFrame:
        file_path = self.input_path / self.data_config['holiday_data']
        df = pd.read_csv(file_path)
            
        logger.info(f"Loaded holiday data: {df.shape}")
        df = self._preprocess_holiday_data(df)
        return df
    
    def _preprocess_holiday_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'week_start' in df.columns:
            df['date'] = pd.to_datetime(df['week_start'])
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        elif 'Date' in df.columns:
            df['date'] = pd.to_datetime(df['Date'])
        if 'city' not in df.columns:
            df['city'] = 'All'  
        df = df.sort_values('date')
        
        return df
    
    def load_promo_data(self) -> pd.DataFrame:
        file_path = self.input_path / self.data_config['promo_data']
        df = pd.read_csv(file_path)
            
        logger.info(f"Loaded promo data: {df.shape}")
        df = self._preprocess_promo_data(df)
            
        return df
    
    def _preprocess_promo_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'week_start' in df.columns:
            df['date'] = pd.to_datetime(df['week_start'])
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        elif 'Date' in df.columns:
            df['date'] = pd.to_datetime(df['Date'])
        if 'market' in df.columns:
            df['city'] = df['market']

        required_columns = ['city', 'sku_id']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in promo data: {missing_columns}")
        df = df.sort_values(['city', 'sku_id', 'date'])
        
        return df
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        data = {}

        data['sales'] = self.load_sales_data()

        data['weather'] = self.load_weather_data()
        data['holidays'] = self.load_holiday_data()
        data['promos'] = self.load_promo_data()
        
        logger.info("All data loaded successfully")
        return data
    
    def get_city_list(self) -> List[str]:
        sales_data = self.load_sales_data()
        return sorted(sales_data['city'].unique().tolist())
    
    def get_sku_list(self) -> List[str]:
        sales_data = self.load_sales_data()
        return sorted(sales_data['sku_id'].unique().tolist())
    
    def get_date_range(self) -> Tuple[datetime, datetime]:
        sales_data = self.load_sales_data()
        return sales_data['date'].min(), sales_data['date'].max()
    
    def validate_data_availability(self, target_city: str = None) -> Dict[str, bool]:
        if target_city is None:
            target_city = self.target_city
        
        sales_data = self.load_sales_data()
        
        validation = {
            'target_city_exists': target_city in sales_data['city'].unique(),
            'has_sufficient_data': False,
            'data_quality_score': 0.0
        }
        
        if validation['target_city_exists']:
            city_data = sales_data[sales_data['city'] == target_city]
            weeks_of_data = len(city_data['date'].unique()) / 7
            validation['has_sufficient_data'] = weeks_of_data >= 2  # At least 2 weeks
            validation['data_quality_score'] = min(weeks_of_data / 13, 1.0)  # Normalize to 0-1
        
        return validation


if __name__ == "__main__":
    loader = DataLoader()

    data = loader.load_all_data()
    print("Data Summary:")
    for name, df in data.items():
        if not df.empty:
            print(f"{name}: {df.shape}")
        else:
            print(f"{name}: No data available")
    validation = loader.validate_data_availability()
    print(f"\nTarget city validation: {validation}")

