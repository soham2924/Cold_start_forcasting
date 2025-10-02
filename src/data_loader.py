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
        
        # Create output directories
        self._create_directories()
    
    def _create_directories(self):

        directories = [
            self.data_config['output_path'],
            self.data_config['plots_path'],
            self.data_config['reports_path']
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Create logs directory
        Path("logs").mkdir(exist_ok=True)
    
    def load_sales_data(self) -> pd.DataFrame:

        try:
            file_path = self.input_path / self.data_config['sales_data']
            df = pd.read_csv(file_path)
            
            logger.info(f"Loaded sales data: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")
            
            # Basic preprocessing
            df = self._preprocess_sales_data(df)
            
            return df
            
        except FileNotFoundError:
            logger.error(f"Sales data file not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading sales data: {str(e)}")
            raise
    
    def _preprocess_sales_data(self, df: pd.DataFrame) -> pd.DataFrame:

        if 'week_start' in df.columns:
            df['date'] = pd.to_datetime(df['week_start'])
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        elif 'Date' in df.columns:
            df['date'] = pd.to_datetime(df['Date'])
        else:
            raise ValueError("Date column not found in sales data")
        
        # Map column names to expected format
        if 'market' in df.columns:
            df['city'] = df['market']
        if 'units' in df.columns:
            df['units_sold'] = df['units']
        
        # Ensure required columns exist
        required_columns = ['city', 'sku_id', 'units_sold']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Sort by date and city
        df = df.sort_values(['city', 'date'])
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['city', 'sku_id', 'date'])
        
        # Fill missing values
        df['units_sold'] = df['units_sold'].fillna(0)
        
        return df
    
    def load_weather_data(self) -> pd.DataFrame:
        """Load weather data."""
        try:
            file_path = self.input_path / self.data_config['weather_data']
            df = pd.read_csv(file_path)
            
            logger.info(f"Loaded weather data: {df.shape}")
            
            # Preprocess weather data
            df = self._preprocess_weather_data(df)
            
            return df
            
        except FileNotFoundError:
            logger.warning(f"Weather data file not found: {file_path}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading weather data: {str(e)}")
            raise
    
    def _preprocess_weather_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess weather data."""
        # Convert date column - handle different column names
        if 'week_start' in df.columns:
            df['date'] = pd.to_datetime(df['week_start'])
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        elif 'Date' in df.columns:
            df['date'] = pd.to_datetime(df['Date'])
        
        # Map column names to expected format
        if 'market' in df.columns:
            df['city'] = df['market']
        
        # Ensure city column exists
        if 'city' not in df.columns:
            raise ValueError("City column not found in weather data")
        
        # Sort by date and city
        df = df.sort_values(['city', 'date'])
        
        return df
    
    def load_holiday_data(self) -> pd.DataFrame:
        """Load holiday data."""
        try:
            file_path = self.input_path / self.data_config['holiday_data']
            df = pd.read_csv(file_path)
            
            logger.info(f"Loaded holiday data: {df.shape}")
            
            # Preprocess holiday data
            df = self._preprocess_holiday_data(df)
            
            return df
            
        except FileNotFoundError:
            logger.warning(f"Holiday data file not found: {file_path}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading holiday data: {str(e)}")
            raise
    
    def _preprocess_holiday_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess holiday data."""
        # Convert date column - handle different column names
        if 'week_start' in df.columns:
            df['date'] = pd.to_datetime(df['week_start'])
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        elif 'Date' in df.columns:
            df['date'] = pd.to_datetime(df['Date'])
        
        # Ensure required columns exist
        if 'city' not in df.columns:
            df['city'] = 'All'  # Assume national holidays if city not specified
        
        # Sort by date
        df = df.sort_values('date')
        
        return df
    
    def load_promo_data(self) -> pd.DataFrame:
        """Load promotional data."""
        try:
            file_path = self.input_path / self.data_config['promo_data']
            df = pd.read_csv(file_path)
            
            logger.info(f"Loaded promo data: {df.shape}")
            
            # Preprocess promo data
            df = self._preprocess_promo_data(df)
            
            return df
            
        except FileNotFoundError:
            logger.warning(f"Promo data file not found: {file_path}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading promo data: {str(e)}")
            raise
    
    def _preprocess_promo_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess promotional data."""
        # Convert date column - handle different column names
        if 'week_start' in df.columns:
            df['date'] = pd.to_datetime(df['week_start'])
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        elif 'Date' in df.columns:
            df['date'] = pd.to_datetime(df['Date'])
        
        # Map column names to expected format
        if 'market' in df.columns:
            df['city'] = df['market']
        
        # Ensure required columns exist
        required_columns = ['city', 'sku_id']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in promo data: {missing_columns}")
        
        # Sort by date, city, and sku
        df = df.sort_values(['city', 'sku_id', 'date'])
        
        return df
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Load all available data sources."""
        data = {}
        
        # Load sales data (required)
        data['sales'] = self.load_sales_data()
        
        # Load optional data sources
        data['weather'] = self.load_weather_data()
        data['holidays'] = self.load_holiday_data()
        data['promos'] = self.load_promo_data()
        
        logger.info("All data loaded successfully")
        return data
    
    def get_city_list(self) -> List[str]:
        """Get list of all cities in the sales data."""
        sales_data = self.load_sales_data()
        return sorted(sales_data['city'].unique().tolist())
    
    def get_sku_list(self) -> List[str]:
        """Get list of all SKU IDs in the sales data."""
        sales_data = self.load_sales_data()
        return sorted(sales_data['sku_id'].unique().tolist())
    
    def get_date_range(self) -> Tuple[datetime, datetime]:
        """Get the date range of available data."""
        sales_data = self.load_sales_data()
        return sales_data['date'].min(), sales_data['date'].max()
    
    def validate_data_availability(self, target_city: str = None) -> Dict[str, bool]:
        """Validate data availability for target city."""
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
    # Example usage
    loader = DataLoader()
    
    # Load all data
    data = loader.load_all_data()
    
    # Print summary
    print("Data Summary:")
    for name, df in data.items():
        if not df.empty:
            print(f"{name}: {df.shape}")
        else:
            print(f"{name}: No data available")
    
    # Validate target city
    validation = loader.validate_data_availability()
    print(f"\nTarget city validation: {validation}")

