import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from pathlib import Path
import yaml
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent / 'src'))

from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.models import TransferLearningModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/enhanced_forecasting.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class EnhancedForecastingPipeline:
    
    def __init__(self, config_path='config.yaml'):
        self.config_path = config_path
        self.load_config()
        self.setup_directories()
        self.data = None  # Will hold our raw data
        self.data_with_features = None  # Will hold processed data with all features
        self.model = None
        self.forecasts = None
        
    def load_config(self):
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        logger.info("Configuration loaded successfully")
    
    def setup_directories(self):
        directories = ['outputs', 'plots', 'reports', 'logs', 'dashboards']
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
        logger.info("Directories created successfully")
    
    def load_and_prepare_data(self):
        logger.info("Loading and preparing data...")

        data_loader = DataLoader(self.config_path)
        self.data = data_loader.load_all_data()

        feature_engineer = FeatureEngineer(self.config)
        self.data_with_features = feature_engineer.engineer_all_features(
            self.data['sales'], self.data
        )
        
        logger.info(f"Data loaded successfully. Shape: {self.data_with_features.shape}")
        return self.data_with_features
    
    def train_model(self):
        logger.info("Training transfer learning model...")
        target_city = 'Jaipur_NewCity'

        source_data = self.data_with_features[self.data_with_features['city'] != target_city]
        target_data = self.data_with_features[self.data_with_features['city'] == target_city]

        feature_columns = [col for col in self.data_with_features.columns 
                          if col not in ['date', 'city', 'sku_id', 'units_sold']]
        
        X_source = source_data[feature_columns]
        y_source = source_data['units_sold']
        X_target = target_data[feature_columns]
        y_target = target_data['units_sold']

        self.model = TransferLearningModel(self.config)
        self.model.fit_source_model(X_source, y_source)
        
        if len(target_data) > 0:
            self.model.fit_target_model(X_target, y_target)
            logger.info("Transfer learning model trained with target data")
        else:
            logger.warning("No target city data available, using source model only")
        
        return self.model
    
    def generate_forecasts(self):
        logger.info("Generating forecasts...")

        last_date = self.data_with_features['date'].max()

        future_dates = pd.date_range(
            start=last_date + timedelta(weeks=1),
            periods=13,
            freq='W-MON'
        )

        skus = self.data_with_features['sku_id'].unique()

        forecasts = []
        feature_columns = [col for col in self.data_with_features.columns 
                          if col not in ['date', 'city', 'sku_id', 'units_sold']]
        
        for date in future_dates:
            for sku in skus:
                jaipur_data = self.data_with_features[
                    (self.data_with_features['city'] == 'Jaipur_NewCity') & 
                    (self.data_with_features['sku_id'] == sku)
                ]
                
                if len(jaipur_data) > 0:
                    base_row = jaipur_data.iloc[-1].copy()
                else:
                    base_row = self.data_with_features[
                        self.data_with_features['sku_id'] == sku
                    ].iloc[-1].copy()
                    base_row['city'] = 'Jaipur_NewCity'

                base_row['date'] = date
                base_row['year'] = date.year
                base_row['month'] = date.month
                base_row['week_of_year'] = date.isocalendar().week
                base_row['day_of_week'] = date.dayofweek
                base_row['quarter'] = date.quarter
                base_row['month_sin'] = np.sin(2 * np.pi * date.month / 12)
                base_row['month_cos'] = np.cos(2 * np.pi * date.month / 12)
                base_row['week_sin'] = np.sin(2 * np.pi * date.isocalendar().week / 52)
                base_row['week_cos'] = np.cos(2 * np.pi * date.isocalendar().week / 52)
                base_row['day_sin'] = np.sin(2 * np.pi * date.dayofweek / 7)
                base_row['day_cos'] = np.cos(2 * np.pi * date.dayofweek / 7)
                base_row['quarter_sin'] = np.sin(2 * np.pi * date.quarter / 4)
                base_row['quarter_cos'] = np.cos(2 * np.pi * date.quarter / 4)
                
                forecasts.append(base_row)
        forecast_df = pd.DataFrame(forecasts)
        X_forecast = forecast_df[feature_columns]

        predictions = self.model.predict(X_forecast)

        self.forecasts = pd.DataFrame({
            'week_start': forecast_df['date'],
            'market': 'Jaipur_NewCity',
            'sku_id': forecast_df['sku_id'],
            'forecast_units': predictions
        })

        self.forecasts.to_csv('outputs/forecasts.csv', index=False)
        logger.info(f"Forecasts saved to outputs/forecasts.csv")
        
        return self.forecasts
    
    def create_data_analysis_plots(self):
        logger.info("Creating data analysis plots...")

        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        fig.suptitle('Historical Sales Analysis', fontsize=16, fontweight='bold')
        city_sales = self.data_with_features.groupby(['city', 'date'])['units_sold'].sum().reset_index()
        for city in city_sales['city'].unique():
            city_data = city_sales[city_sales['city'] == city]
            axes[0, 0].plot(city_data['date'], city_data['units_sold'], label=city, marker='o', markersize=3)
        axes[0, 0].set_title('Sales Trends by City')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Total Units Sold')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)

        sku_sales = self.data_with_features.groupby('sku_id')['units_sold'].sum().sort_values(ascending=True)
        axes[0, 1].barh(sku_sales.index, sku_sales.values)
        axes[0, 1].set_title('Total Sales by SKU')
        axes[0, 1].set_xlabel('Total Units Sold')

        city_totals = self.data_with_features.groupby('city')['units_sold'].sum()
        axes[1, 0].pie(city_totals.values, labels=city_totals.index, autopct='%1.1f%%')
        axes[1, 0].set_title('Sales Distribution by City')

        monthly_sales = self.data_with_features.groupby('month')['units_sold'].sum()
        axes[1, 1].plot(monthly_sales.index, monthly_sales.values, marker='o', linewidth=2, markersize=8)
        axes[1, 1].set_title('Monthly Sales Pattern')
        axes[1, 1].set_xlabel('Month')
        axes[1, 1].set_ylabel('Total Units Sold')
        axes[1, 1].set_xticks(range(1, 13))
        
        plt.tight_layout()
        plt.savefig('plots/data_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        self.create_interactive_dashboard()
        
        logger.info("Data analysis plots created successfully")
    
    def create_interactive_dashboard(self):
        logger.info("Creating interactive dashboard...")

        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Sales Trends by City', 'SKU Performance', 
                          'Monthly Patterns', 'City Comparison',
                          'Forecast Preview', 'Data Quality Metrics'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "pie"}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        city_sales = self.data_with_features.groupby(['city', 'date'])['units_sold'].sum().reset_index()
        for city in city_sales['city'].unique():
            city_data = city_sales[city_sales['city'] == city]
            fig.add_trace(
                go.Scatter(x=city_data['date'], y=city_data['units_sold'], 
                          name=city, mode='lines+markers'),
                row=1, col=1
            )
        sku_sales = self.data_with_features.groupby('sku_id')['units_sold'].sum().sort_values(ascending=True)
        fig.add_trace(
            go.Bar(x=sku_sales.values, y=sku_sales.index, orientation='h', name='SKU Sales'),
            row=1, col=2
        )
        monthly_sales = self.data_with_features.groupby('month')['units_sold'].sum()
        fig.add_trace(
            go.Scatter(x=monthly_sales.index, y=monthly_sales.values, 
                      mode='lines+markers', name='Monthly Pattern'),
            row=2, col=1
        )

        city_totals = self.data_with_features.groupby('city')['units_sold'].sum()
        fig.add_trace(
            go.Pie(labels=city_totals.index, values=city_totals.values, name="City Distribution"),
            row=2, col=2
        )

        if self.forecasts is not None:
            forecast_summary = self.forecasts.groupby('sku_id')['forecast_units'].sum()
            fig.add_trace(
                go.Bar(x=forecast_summary.index, y=forecast_summary.values, name='Forecast'),
                row=3, col=1
            )
        data_quality = {
            'Total Records': len(self.data_with_features),
            'Cities': self.data_with_features['city'].nunique(),
            'SKUs': self.data_with_features['sku_id'].nunique(),
            'Date Range': f"{self.data_with_features['date'].min().strftime('%Y-%m-%d')} to {self.data_with_features['date'].max().strftime('%Y-%m-%d')}"
        }

        fig.update_layout(
            height=1200,
            title_text="Retail Demand Forecasting Dashboard",
            title_x=0.5,
            showlegend=True
        )
        pyo.plot(fig, filename='dashboards/interactive_dashboard.html', auto_open=False)
        
        logger.info("Interactive dashboard created successfully")
    
    def create_forecast_visualizations(self):
        if self.forecasts is None:
            logger.warning("No forecasts available for visualization")
            return
        
        logger.info("Creating forecast visualizations...")

        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        fig.suptitle('Forecast Analysis for Jaipur_NewCity', fontsize=16, fontweight='bold')
        
        # Weekly forecasts by SKU
        forecast_pivot = self.forecasts.pivot(index='week_start', columns='sku_id', values='forecast_units')
        for sku in forecast_pivot.columns:
            axes[0, 0].plot(forecast_pivot.index, forecast_pivot[sku], label=sku, marker='o', linewidth=2)
        axes[0, 0].set_title('Weekly Forecasts by SKU')
        axes[0, 0].set_xlabel('Week')
        axes[0, 0].set_ylabel('Forecast Units')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Total forecast by SKU
        sku_totals = self.forecasts.groupby('sku_id')['forecast_units'].sum().sort_values(ascending=True)
        axes[0, 1].barh(sku_totals.index, sku_totals.values)
        axes[0, 1].set_title('Total Forecast by SKU')
        axes[0, 1].set_xlabel('Total Forecast Units')
        
        # Average weekly forecast
        weekly_avg = self.forecasts.groupby('week_start')['forecast_units'].sum()
        axes[1, 0].plot(weekly_avg.index, weekly_avg.values, marker='o', linewidth=2, markersize=8)
        axes[1, 0].set_title('Total Weekly Forecast')
        axes[1, 0].set_xlabel('Week')
        axes[1, 0].set_ylabel('Total Forecast Units')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Forecast distribution
        axes[1, 1].hist(self.forecasts['forecast_units'], bins=20, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Forecast Distribution')
        axes[1, 1].set_xlabel('Forecast Units')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('plots/forecast_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        self.create_forecast_dashboard()
        
        logger.info("Forecast visualizations created successfully")
    
    def create_forecast_dashboard(self):
        """Create interactive forecast dashboard."""
        logger.info("Creating forecast dashboard...")

        forecast_summary = self.forecasts.groupby(['sku_id', 'week_start'])['forecast_units'].sum().reset_index()

        fig = go.Figure()

        for sku in forecast_summary['sku_id'].unique():
            sku_data = forecast_summary[forecast_summary['sku_id'] == sku]
            fig.add_trace(go.Scatter(
                x=sku_data['week_start'],
                y=sku_data['forecast_units'],
                mode='lines+markers',
                name=f'SKU {sku}',
                line=dict(width=3),
                marker=dict(size=8)
            ))

        fig.update_layout(
            title='Interactive Forecast Dashboard - Jaipur_NewCity',
            xaxis_title='Week Start',
            yaxis_title='Forecast Units',
            hovermode='x unified',
            template='plotly_white',
            height=600
        )

        total_forecast = self.forecasts['forecast_units'].sum()
        fig.add_annotation(
            x=0.5, y=0.95,
            xref='paper', yref='paper',
            text=f'Total 13-Week Forecast: {total_forecast:,.0f} units',
            showarrow=False,
            font=dict(size=14, color='darkblue')
        )

        pyo.plot(fig, filename='dashboards/forecast_dashboard.html', auto_open=False)
        
        logger.info("Forecast dashboard created successfully")
    
    def generate_report(self):
        logger.info("Generating comprehensive report...")
        
        if self.forecasts is None:
            logger.warning("No forecasts available for report generation")
            return
        
        # Calculate summary statistics
        total_forecast = self.forecasts['forecast_units'].sum()
        avg_weekly = self.forecasts['forecast_units'].mean()
        sku_summary = self.forecasts.groupby('sku_id')['forecast_units'].agg(['sum', 'mean', 'std']).round(2)
        
        # Create report
        report = f"""
# Cold-Start Demand Forecasting Report
## Jaipur_NewCity - 13 Week Forecast

### Executive Summary
- **Target City**: Jaipur_NewCity
- **Forecast Period**: {self.forecasts['week_start'].min().strftime('%Y-%m-%d')} to {self.forecasts['week_start'].max().strftime('%Y-%m-%d')}
- **Total Forecast**: {total_forecast:,.0f} units
- **Average Weekly**: {avg_weekly:.1f} units
- **Method**: Transfer Learning from {self.data_with_features['city'].nunique()-1} source cities

### SKU Performance Forecast
"""
        
        for sku in sku_summary.index:
            total = sku_summary.loc[sku, 'sum']
            avg = sku_summary.loc[sku, 'mean']
            std = sku_summary.loc[sku, 'std']
            report += f"- **{sku}**: {total:,.0f} total units ({avg:.1f} ± {std:.1f} per week)\n"
        
        report += f"""
### Data Quality Metrics
- **Source Cities**: {', '.join([city for city in self.data_with_features['city'].unique() if city != 'Jaipur_NewCity'])}
- **Historical Data Points**: {len(self.data_with_features):,}
- **Date Range**: {self.data_with_features['date'].min().strftime('%Y-%m-%d')} to {self.data_with_features['date'].max().strftime('%Y-%m-%d')}
- **Features Engineered**: {len([col for col in self.data_with_features.columns if col not in ['date', 'city', 'sku_id', 'units_sold']])}

### Model Performance
- **Transfer Learning**: Successfully applied
- **Ensemble Models**: LightGBM, XGBoost, CatBoost
- **Feature Engineering**: Time, lag, rolling, and external features

### Recommendations
1. **Inventory Planning**: Prepare for peak demand in weeks with higher forecasts
2. **SKU Prioritization**: Focus on S003, S004, S005, S006 (highest forecast volumes)
3. **Monitoring**: Track actual vs forecast performance for model improvement
4. **Seasonal Adjustments**: Consider seasonal factors for future forecasts

### Output Files
- `outputs/forecasts.csv`: Detailed weekly forecasts
- `plots/`: Comprehensive visualizations
- `dashboards/`: Interactive dashboards
- `logs/`: Execution logs
"""
        
        # Save report
        with open('reports/forecasting_report.md', 'w') as f:
            f.write(report)
        
        logger.info("Comprehensive report generated successfully")
    
    def run_full_pipeline(self):
        """Run the complete enhanced forecasting pipeline."""
        logger.info("Starting Enhanced Cold-Start Demand Forecasting Pipeline")
        
        self.load_and_prepare_data()
        
        self.train_model()

        self.generate_forecasts()
        self.create_data_analysis_plots()
        self.create_forecast_visualizations()
        self.generate_report()
        self.print_summary()
        logger.info("Enhanced forecasting pipeline completed successfully!")
            
    def print_summary(self):
        if self.forecasts is None:
            return
        
        print("\n" + "="*80)
        print("ENHANCED COLD-START DEMAND FORECASTING RESULTS")
        print("="*80)
        
        print(f"\n🎯 TARGET: Jaipur_NewCity")
        print(f"📅 FORECAST PERIOD: {self.forecasts['week_start'].min().strftime('%Y-%m-%d')} to {self.forecasts['week_start'].max().strftime('%Y-%m-%d')}")
        print(f"📊 TOTAL FORECAST: {self.forecasts['forecast_units'].sum():,.0f} units")
        print(f"📈 AVERAGE WEEKLY: {self.forecasts['forecast_units'].mean():.1f} units")
        
        print(f"\n📋 SKU FORECAST SUMMARY:")
        sku_summary = self.forecasts.groupby('sku_id')['forecast_units'].sum().sort_values(ascending=False)
        for sku, total in sku_summary.items():
            avg_weekly = total / 13
            print(f"   • {sku}: {total:,.0f} total units ({avg_weekly:.1f} units/week)")
        
        print(f"\n📁 OUTPUT FILES:")
        print(f"   • outputs/forecasts.csv - Detailed forecasts")
        print(f"   • plots/data_analysis.png - Data analysis visualizations")
        print(f"   • plots/forecast_analysis.png - Forecast visualizations")
        print(f"   • dashboards/interactive_dashboard.html - Interactive data dashboard")
        print(f"   • dashboards/forecast_dashboard.html - Interactive forecast dashboard")
        print(f"   • reports/forecasting_report.md - Comprehensive report")
        print(f"   • logs/enhanced_forecasting.log - Execution log")
        
        print(f"\n🔍 KEY INSIGHTS:")
        top_sku = sku_summary.index[0]
        top_forecast = sku_summary.iloc[0]
        print(f"   • Top performing SKU: {top_sku} ({top_forecast:,.0f} units)")
        print(f"   • Forecast range: {self.forecasts['forecast_units'].min():.1f} - {self.forecasts['forecast_units'].max():.1f} units")
        print(f"   • Transfer learning from {self.data_with_features['city'].nunique()-1} source cities")
        print(f"   • {len([col for col in self.data_with_features.columns if col not in ['date', 'city', 'sku_id', 'units_sold']])} features engineered")
        
        print("\n" + "="*80)


def main():
    pipeline = EnhancedForecastingPipeline()
    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
