"""
Visualization and plotting utilities for retail demand forecasting.
Creates comprehensive plots for forecasts, validation, and explainability.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Advanced plotting imports
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import bokeh.plotting as bk
    from bokeh.models import HoverTool, ColumnDataSource
    from bokeh.layouts import column, row
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ForecastVisualizer:
    """Visualization utilities for forecasting results."""
    
    def __init__(self, config: Dict):
        """Initialize forecast visualizer."""
        self.config = config
        self.plots_path = Path(config['data']['plots_path'])
        self.plots_path.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting parameters
        self.figsize = (12, 8)
        self.dpi = 300
        
        logger.info("Forecast visualizer initialized")
    
    def plot_forecasts(self, forecasts: Dict[str, Any], historical_data: pd.DataFrame = None, 
                      save_plots: bool = True) -> List[str]:
        """
        Plot forecasts for all SKUs.
        
        Args:
            forecasts: Dictionary containing forecast results
            historical_data: Historical data for context
            save_plots: Whether to save plots to disk
        
        Returns:
            List of saved plot file paths
        """
        logger.info("Creating forecast plots...")
        
        saved_plots = []
        
        for sku_id, forecast in forecasts.items():
            try:
                # Create individual forecast plot
                plot_path = self._plot_single_forecast(
                    sku_id, forecast, historical_data, save_plots
                )
                
                if plot_path:
                    saved_plots.append(plot_path)
                    
            except Exception as e:
                logger.warning(f"Failed to plot forecast for SKU {sku_id}: {str(e)}")
                continue
        
        # Create summary plots
        summary_plots = self._create_summary_plots(forecasts, save_plots)
        saved_plots.extend(summary_plots)
        
        logger.info(f"Created {len(saved_plots)} forecast plots")
        
        return saved_plots
    
    def _plot_single_forecast(self, sku_id: str, forecast: Dict[str, Any], 
                             historical_data: pd.DataFrame = None, 
                             save_plot: bool = True) -> Optional[str]:
        """Plot forecast for a single SKU."""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Plot historical data if available
        if historical_data is not None:
            sku_historical = historical_data[
                (historical_data['sku_id'] == sku_id) & 
                (historical_data['city'] == forecast['city'])
            ]
            
            if not sku_historical.empty:
                ax.plot(sku_historical['date'], sku_historical['units_sold'], 
                       label='Historical', color='blue', alpha=0.7, linewidth=2)
        
        # Plot forecast
        forecast_dates = forecast['dates']
        predictions = forecast['predictions']
        
        ax.plot(forecast_dates, predictions, 
               label='Forecast', color='red', linewidth=2, linestyle='--')
        
        # Plot prediction intervals if available
        if 'prediction_intervals' in forecast:
            self._plot_prediction_intervals(ax, forecast_dates, forecast['prediction_intervals'])
        
        # Customize plot
        ax.set_title(f'Demand Forecast - {sku_id} ({forecast["city"]})', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Units Sold', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        if save_plot:
            plot_path = self.plots_path / f'forecast_{sku_id}.png'
            plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            return str(plot_path)
        else:
            plt.show()
            return None
    
    def _plot_prediction_intervals(self, ax, dates: pd.DatetimeIndex, 
                                  intervals: Dict[str, Any]):
        """Plot prediction intervals."""
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        
        for i, (conf_name, interval_data) in enumerate(intervals.items()):
            if i >= len(colors):
                break
                
            lower = interval_data['lower']
            upper = interval_data['upper']
            
            # Extract confidence level from name
            conf_level = conf_name.split('_')[-1]
            
            ax.fill_between(dates, lower, upper, 
                           alpha=0.3, color=colors[i],
                           label=f'{conf_level} Confidence Interval')
    
    def _create_summary_plots(self, forecasts: Dict[str, Any], 
                             save_plots: bool = True) -> List[str]:
        """Create summary plots for all forecasts."""
        saved_plots = []
        
        # Plot 1: All forecasts on one plot
        plot_path = self._plot_all_forecasts(forecasts, save_plots)
        if plot_path:
            saved_plots.append(plot_path)
        
        # Plot 2: Forecast distribution
        plot_path = self._plot_forecast_distribution(forecasts, save_plots)
        if plot_path:
            saved_plots.append(plot_path)
        
        # Plot 3: Forecast by week
        plot_path = self._plot_forecast_by_week(forecasts, save_plots)
        if plot_path:
            saved_plots.append(plot_path)
        
        return saved_plots
    
    def _plot_all_forecasts(self, forecasts: Dict[str, Any], 
                           save_plot: bool = True) -> Optional[str]:
        """Plot all forecasts on a single plot."""
        fig, ax = plt.subplots(figsize=(15, 10), dpi=self.dpi)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(forecasts)))
        
        for i, (sku_id, forecast) in enumerate(forecasts.items()):
            ax.plot(forecast['dates'], forecast['predictions'], 
                   label=sku_id, color=colors[i], linewidth=2, alpha=0.8)
        
        ax.set_title('All SKU Forecasts - Jaipur', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Units Sold', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_plot:
            plot_path = self.plots_path / 'all_forecasts.png'
            plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            return str(plot_path)
        else:
            plt.show()
            return None
    
    def _plot_forecast_distribution(self, forecasts: Dict[str, Any], 
                                   save_plot: bool = True) -> Optional[str]:
        """Plot distribution of forecast values."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=self.dpi)
        
        # Collect all forecast values
        all_forecasts = []
        sku_names = []
        
        for sku_id, forecast in forecasts.items():
            all_forecasts.extend(forecast['predictions'])
            sku_names.extend([sku_id] * len(forecast['predictions']))
        
        # Plot 1: Overall distribution
        ax1.hist(all_forecasts, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title('Distribution of All Forecasts', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Forecasted Units Sold', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Box plot by SKU
        forecast_df = pd.DataFrame({'SKU': sku_names, 'Forecast': all_forecasts})
        sns.boxplot(data=forecast_df, x='SKU', y='Forecast', ax=ax2)
        ax2.set_title('Forecast Distribution by SKU', fontsize=14, fontweight='bold')
        ax2.set_xlabel('SKU ID', fontsize=12)
        ax2.set_ylabel('Forecasted Units Sold', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = self.plots_path / 'forecast_distribution.png'
            plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            return str(plot_path)
        else:
            plt.show()
            return None
    
    def _plot_forecast_by_week(self, forecasts: Dict[str, Any], 
                              save_plot: bool = True) -> Optional[str]:
        """Plot forecast aggregated by week."""
        fig, ax = plt.subplots(figsize=(12, 8), dpi=self.dpi)
        
        # Aggregate forecasts by week
        weekly_forecasts = {}
        
        for sku_id, forecast in forecasts.items():
            forecast_df = pd.DataFrame({
                'date': forecast['dates'],
                'forecast': forecast['predictions']
            })
            forecast_df['week'] = forecast_df['date'].dt.isocalendar().week
            forecast_df['year'] = forecast_df['date'].dt.year
            
            weekly_agg = forecast_df.groupby(['year', 'week'])['forecast'].sum().reset_index()
            weekly_agg['date'] = pd.to_datetime(weekly_agg[['year', 'week']].assign(day=1))
            
            weekly_forecasts[sku_id] = weekly_agg
        
        # Plot weekly forecasts
        colors = plt.cm.tab10(np.linspace(0, 1, len(weekly_forecasts)))
        
        for i, (sku_id, weekly_data) in enumerate(weekly_forecasts.items()):
            ax.plot(weekly_data['date'], weekly_data['forecast'], 
                   label=sku_id, color=colors[i], marker='o', linewidth=2)
        
        ax.set_title('Weekly Forecast Aggregation - Jaipur', fontsize=16, fontweight='bold')
        ax.set_xlabel('Week', fontsize=12)
        ax.set_ylabel('Total Units Sold', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_plot:
            plot_path = self.plots_path / 'weekly_forecasts.png'
            plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            return str(plot_path)
        else:
            plt.show()
            return None


class ValidationVisualizer:
    """Visualization utilities for validation results."""
    
    def __init__(self, config: Dict):
        """Initialize validation visualizer."""
        self.config = config
        self.plots_path = Path(config['data']['plots_path'])
        self.plots_path.mkdir(parents=True, exist_ok=True)
        
        self.figsize = (12, 8)
        self.dpi = 300
    
    def plot_validation_results(self, validation_results: Dict[str, Any], 
                               save_plots: bool = True) -> List[str]:
        """Plot validation results."""
        logger.info("Creating validation plots...")
        
        saved_plots = []
        
        # Plot 1: Performance by city-SKU
        plot_path = self._plot_performance_by_combination(validation_results, save_plots)
        if plot_path:
            saved_plots.append(plot_path)
        
        # Plot 2: Metric distribution
        plot_path = self._plot_metric_distribution(validation_results, save_plots)
        if plot_path:
            saved_plots.append(plot_path)
        
        # Plot 3: Performance heatmap
        plot_path = self._plot_performance_heatmap(validation_results, save_plots)
        if plot_path:
            saved_plots.append(plot_path)
        
        return saved_plots
    
    def _plot_performance_by_combination(self, validation_results: Dict[str, Any], 
                                        save_plot: bool = True) -> Optional[str]:
        """Plot performance by city-SKU combination."""
        fig, ax = plt.subplots(figsize=(15, 8), dpi=self.dpi)
        
        # Extract performance scores
        combinations = []
        scores = []
        grades = []
        
        for key, result in validation_results['detailed_results'].items():
            combinations.append(key)
            scores.append(result['overall_performance_score'])
            grades.append(result['performance_grade'])
        
        # Create bar plot
        bars = ax.bar(range(len(combinations)), scores, 
                     color=[self._grade_to_color(grade) for grade in grades])
        
        ax.set_title('Model Performance by City-SKU Combination', fontsize=16, fontweight='bold')
        ax.set_xlabel('City-SKU Combination', fontsize=12)
        ax.set_ylabel('Performance Score (MAPE)', fontsize=12)
        ax.set_xticks(range(len(combinations)))
        ax.set_xticklabels(combinations, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Add grade labels on bars
        for i, (bar, grade) in enumerate(zip(bars, grades)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   grade, ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = self.plots_path / 'validation_performance.png'
            plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            return str(plot_path)
        else:
            plt.show()
            return None
    
    def _grade_to_color(self, grade: str) -> str:
        """Convert performance grade to color."""
        color_map = {'A': 'green', 'B': 'yellow', 'C': 'orange', 'D': 'red'}
        return color_map.get(grade, 'gray')
    
    def _plot_metric_distribution(self, validation_results: Dict[str, Any], 
                                 save_plot: bool = True) -> Optional[str]:
        """Plot distribution of validation metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=self.dpi)
        axes = axes.ravel()
        
        metrics = ['mae', 'rmse', 'mape', 'smape']
        
        for i, metric in enumerate(metrics):
            # Collect metric values
            values = []
            for result in validation_results['detailed_results'].values():
                if metric in result['cross_validation']:
                    values.append(result['cross_validation'][metric]['mean'])
            
            if values:
                axes[i].hist(values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                axes[i].set_title(f'Distribution of {metric.upper()}', fontsize=14, fontweight='bold')
                axes[i].set_xlabel(metric.upper(), fontsize=12)
                axes[i].set_ylabel('Frequency', fontsize=12)
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = self.plots_path / 'metric_distribution.png'
            plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            return str(plot_path)
        else:
            plt.show()
            return None
    
    def _plot_performance_heatmap(self, validation_results: Dict[str, Any], 
                                 save_plot: bool = True) -> Optional[str]:
        """Plot performance heatmap."""
        # Extract data for heatmap
        data = []
        cities = set()
        skus = set()
        
        for key, result in validation_results['detailed_results'].items():
            city, sku = key.split('_', 1)
            cities.add(city)
            skus.add(sku)
            data.append((city, sku, result['overall_performance_score']))
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=['City', 'SKU', 'Score'])
        pivot_df = df.pivot(index='City', columns='SKU', values='Score')
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8), dpi=self.dpi)
        sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap='RdYlGn_r', 
                   ax=ax, cbar_kws={'label': 'Performance Score (MAPE)'})
        
        ax.set_title('Performance Heatmap by City and SKU', fontsize=16, fontweight='bold')
        ax.set_xlabel('SKU ID', fontsize=12)
        ax.set_ylabel('City', fontsize=12)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = self.plots_path / 'performance_heatmap.png'
            plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            return str(plot_path)
        else:
            plt.show()
            return None


class ExplainabilityVisualizer:
    """Visualization utilities for explainability results."""
    
    def __init__(self, config: Dict):
        """Initialize explainability visualizer."""
        self.config = config
        self.plots_path = Path(config['data']['plots_path'])
        self.plots_path.mkdir(parents=True, exist_ok=True)
        
        self.figsize = (12, 8)
        self.dpi = 300
    
    def plot_explainability_results(self, explainability_results: Dict[str, Any], 
                                   save_plots: bool = True) -> List[str]:
        """Plot explainability results."""
        logger.info("Creating explainability plots...")
        
        saved_plots = []
        
        # Plot 1: Feature importance
        plot_path = self._plot_feature_importance(explainability_results, save_plots)
        if plot_path:
            saved_plots.append(plot_path)
        
        # Plot 2: Driver categories
        plot_path = self._plot_driver_categories(explainability_results, save_plots)
        if plot_path:
            saved_plots.append(plot_path)
        
        # Plot 3: Method comparison
        plot_path = self._plot_method_comparison(explainability_results, save_plots)
        if plot_path:
            saved_plots.append(plot_path)
        
        return saved_plots
    
    def _plot_feature_importance(self, explainability_results: Dict[str, Any], 
                                save_plot: bool = True) -> Optional[str]:
        """Plot feature importance."""
        driver_analysis = explainability_results['driver_analysis']
        
        if 'driver_ranking' not in driver_analysis:
            return None
        
        top_drivers = driver_analysis['driver_ranking']['top_drivers'][:15]
        
        if not top_drivers:
            return None
        
        features, importances = zip(*top_drivers)
        
        fig, ax = plt.subplots(figsize=(12, 8), dpi=self.dpi)
        
        bars = ax.barh(range(len(features)), importances, color='skyblue')
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel('Feature Importance', fontsize=12)
        ax.set_title('Top 15 Feature Importance', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, importance) in enumerate(zip(bars, importances)):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{importance:.4f}', ha='left', va='center')
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = self.plots_path / 'feature_importance.png'
            plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            return str(plot_path)
        else:
            plt.show()
            return None
    
    def _plot_driver_categories(self, explainability_results: Dict[str, Any], 
                               save_plot: bool = True) -> Optional[str]:
        """Plot driver categories."""
        driver_report = explainability_results['driver_report']
        
        if 'top_drivers' not in driver_report or 'driver_categories' not in driver_report['top_drivers']:
            return None
        
        categories = driver_report['top_drivers']['driver_categories']
        
        # Count features in each category
        category_counts = {cat: len(features) for cat, features in categories.items()}
        
        fig, ax = plt.subplots(figsize=(10, 8), dpi=self.dpi)
        
        categories_list = list(category_counts.keys())
        counts = list(category_counts.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(categories_list)))
        
        wedges, texts, autotexts = ax.pie(counts, labels=categories_list, autopct='%1.1f%%',
                                         colors=colors, startangle=90)
        
        ax.set_title('Driver Categories Distribution', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = self.plots_path / 'driver_categories.png'
            plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            return str(plot_path)
        else:
            plt.show()
            return None
    
    def _plot_method_comparison(self, explainability_results: Dict[str, Any], 
                               save_plot: bool = True) -> Optional[str]:
        """Plot comparison of explainability methods."""
        driver_analysis = explainability_results['driver_analysis']
        
        if 'method_comparison' not in driver_analysis:
            return None
        
        method_comparison = driver_analysis['method_comparison']
        
        if not method_comparison:
            return None
        
        # Extract correlations
        methods = []
        correlations = []
        
        for comparison, data in method_comparison.items():
            methods.append(comparison.replace('_vs_', ' vs '))
            correlations.append(data['correlation'])
        
        fig, ax = plt.subplots(figsize=(12, 6), dpi=self.dpi)
        
        bars = ax.bar(methods, correlations, color='lightcoral')
        ax.set_title('Method Agreement (Correlation)', fontsize=16, fontweight='bold')
        ax.set_ylabel('Correlation Coefficient', fontsize=12)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, corr in zip(bars, correlations):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{corr:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_plot:
            plot_path = self.plots_path / 'method_comparison.png'
            plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            return str(plot_path)
        else:
            plt.show()
            return None


class InteractiveVisualizer:
    """Interactive visualization using Plotly and Bokeh."""
    
    def __init__(self, config: Dict):
        """Initialize interactive visualizer."""
        self.config = config
        self.plots_path = Path(config['data']['plots_path'])
        self.plots_path.mkdir(parents=True, exist_ok=True)
    
    def create_interactive_forecast_plot(self, forecasts: Dict[str, Any], 
                                        historical_data: pd.DataFrame = None) -> Optional[str]:
        """Create interactive forecast plot using Plotly."""
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available for interactive plots")
            return None
        
        fig = make_subplots(
            rows=len(forecasts), cols=1,
            subplot_titles=list(forecasts.keys()),
            vertical_spacing=0.05
        )
        
        for i, (sku_id, forecast) in enumerate(forecasts.items(), 1):
            # Add historical data if available
            if historical_data is not None:
                sku_historical = historical_data[
                    (historical_data['sku_id'] == sku_id) & 
                    (historical_data['city'] == forecast['city'])
                ]
                
                if not sku_historical.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=sku_historical['date'],
                            y=sku_historical['units_sold'],
                            mode='lines',
                            name=f'{sku_id} Historical',
                            line=dict(color='blue', width=2)
                        ),
                        row=i, col=1
                    )
            
            # Add forecast
            fig.add_trace(
                go.Scatter(
                    x=forecast['dates'],
                    y=forecast['predictions'],
                    mode='lines',
                    name=f'{sku_id} Forecast',
                    line=dict(color='red', width=2, dash='dash')
                ),
                row=i, col=1
            )
            
            # Add prediction intervals if available
            if 'prediction_intervals' in forecast:
                for conf_name, intervals in forecast['prediction_intervals'].items():
                    conf_level = conf_name.split('_')[-1]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=forecast['dates'],
                            y=intervals['upper'],
                            mode='lines',
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo='skip'
                        ),
                        row=i, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=forecast['dates'],
                            y=intervals['lower'],
                            mode='lines',
                            line=dict(width=0),
                            fill='tonexty',
                            fillcolor=f'rgba(255,0,0,0.2)',
                            name=f'{conf_level} Confidence',
                            hoverinfo='skip'
                        ),
                        row=i, col=1
                    )
        
        fig.update_layout(
            title='Interactive Demand Forecasts - Jaipur',
            height=300 * len(forecasts),
            showlegend=True
        )
        
        # Save interactive plot
        plot_path = self.plots_path / 'interactive_forecasts.html'
        fig.write_html(str(plot_path))
        
        logger.info(f"Interactive forecast plot saved to {plot_path}")
        
        return str(plot_path)


if __name__ == "__main__":
    # Example usage
    import yaml
    
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # Create sample data for testing
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    
    # Sample forecasts
    forecasts = {
        'SKU001': {
            'dates': pd.date_range('2024-01-01', periods=91, freq='D'),
            'predictions': np.random.poisson(10, 91),
            'city': 'Jaipur',
            'sku_id': 'SKU001'
        },
        'SKU002': {
            'dates': pd.date_range('2024-01-01', periods=91, freq='D'),
            'predictions': np.random.poisson(15, 91),
            'city': 'Jaipur',
            'sku_id': 'SKU002'
        }
    }
    
    # Test forecast visualizer
    forecast_viz = ForecastVisualizer(config)
    saved_plots = forecast_viz.plot_forecasts(forecasts)
    
    print(f"Created {len(saved_plots)} forecast plots")
    
    # Test interactive visualizer
    if PLOTLY_AVAILABLE:
        interactive_viz = InteractiveVisualizer(config)
        interactive_plot = interactive_viz.create_interactive_forecast_plot(forecasts)
        if interactive_plot:
            print(f"Interactive plot saved to {interactive_plot}")

