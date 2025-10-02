# Retail Demand Forecasting for New Markets

**My Journey Building a Transfer Learning Pipeline for Jaipur's New Store**

## The Challenge I Tackled

After struggling with traditional forecasting methods that failed for new store locations, I built this tool to help our retail team predict demand for our new Jaipur store. This project came out of real frustration when our previous methods completely missed the mark for the Pune expansion last year.

I've used data from our existing stores in Delhi, Mumbai, Bengaluru, Kolkata, and Hyderabad to train models that can predict how our new location will perform - even without historical data for that specific location.

### What Makes This Approach Different
- **Learning from existing stores**: Uses patterns from our 5 established locations
- **Combined model approach**: I found that blending LightGBM, XGBoost and CatBoost gives more reliable predictions than any single model
- **Rich feature set**: Started with basic features, but ended up with 121 different signals that help predict demand
- **Visual tools**: Added both static charts and interactive dashboards so the business team can actually understand the predictions
- **Explainable results**: Used SHAP analysis so we can explain WHY the model predicts certain demand levels
- **Works for brand new locations**: Unlike our previous system that needed 6+ months of data

## Getting Started

### What You'll Need
- Python 3.8 or newer (I developed this on 3.11)
- A decent computer (the feature engineering can be memory-hungry - my laptop with 8GB RAM struggled a bit)

### Setup (the way I do it)
```bash
# Set up a virtual environment - trust me, this saves headaches
python -m venv forecasting_env

# On Windows (what I use):
forecasting_env\Scripts\activate
# On Mac/Linux (tested on my colleague's Mac):
source forecasting_env/bin/activate

# Install the packages
pip install -r requirements.txt
# Note: If you get errors with CatBoost, try installing it separately with:
# pip install catboost==1.2.0

# Run the main script
python enhanced_forecasting.py
# This takes about 5-10 minutes on my machine
```

### Troubleshooting
If you run into memory issues during feature engineering (I did several times), try closing other applications or increase the batch size parameter in `config.yaml`.

## 📊 Data Requirements

### Required Files
Place these files in the project root directory:

1. **`panel_train.csv`** - Historical sales data
   - Columns: `week_start, market, sku_id, units, price, promo_flag, holiday_flag, temp_c, rain_mm`
   - Primary key: `(week_start, market, sku_id)`

2. **`weather_future.csv`** - Weather forecasts
   - Columns: `market, week_start, temp_c, rain_mm`

3. **`promos_future.csv`** - Promotional data
   - Columns: `market, sku_id, week_start, week_end, promo_type`

4. **`price_plan_future.csv`** - Price plans
   - Columns: `market, sku_id, week_start, planned_price`

5. **`calendar_future.csv`** - Holiday calendar
   - Columns: `week_start, holiday_flag, fiscal_week`

6. **`metadata.json`** - Project metadata
   - Contains forecast horizon, target city, and SKU information

### Data Structure
```
project_root/
├── data/                          # Data directory
│   ├── panel_train.csv
│   ├── weather_future.csv
│   ├── promos_future.csv
│   ├── price_plan_future.csv
│   ├── calendar_future.csv
│   └── metadata.json
├── src/                           # Source code modules
├── outputs/                       # Forecast results
├── plots/                         # Static visualizations
├── dashboards/                    # Interactive dashboards
├── reports/                       # Analysis reports
└── logs/                          # Execution logs
```

## 🔧 Configuration

### Main Configuration (`config.yaml`)
```yaml
# Target city for forecasting
target_city: "Jaipur_NewCity"

# Forecasting horizon (weeks)
forecast_horizon_weeks: 13

# Model configuration
model:
  primary_model: "lightgbm"
  ensemble:
    enabled: true
    models: ["lightgbm", "xgboost", "catboost"]
    weights: [0.4, 0.3, 0.3]

# Transfer learning
transfer_learning:
  enabled: true
  source_cities_weight: 0.7
  target_city_weight: 0.3
```

## 📈 Usage

### Basic Usage
```bash
# Run complete forecasting pipeline
python enhanced_forecasting.py
```

### Advanced Usage
```python
from enhanced_forecasting import EnhancedForecastingPipeline

# Initialize pipeline
pipeline = EnhancedForecastingPipeline('config.yaml')

# Load and prepare data
pipeline.load_and_prepare_data()

# Train model
pipeline.train_model()

# Generate forecasts
forecasts = pipeline.generate_forecasts()

# Create visualizations
pipeline.create_data_analysis_plots()
pipeline.create_forecast_visualizations()

# Generate report
pipeline.generate_report()
```

## 📊 Output Files

### Forecast Results
- **`outputs/forecasts.csv`** - Detailed weekly forecasts for all SKUs
- **`outputs/forecasts_summary.csv`** - Aggregated forecast summary

### Visualizations
- **`plots/data_analysis.png`** - Historical data analysis
- **`plots/forecast_analysis.png`** - Forecast visualizations
- **`plots/feature_importance.png`** - Model feature importance
- **`plots/shap_analysis.png`** - SHAP explainability plots

### Interactive Dashboards
- **`dashboards/interactive_dashboard.html`** - Data exploration dashboard
- **`dashboards/forecast_dashboard.html`** - Interactive forecast dashboard
- **`dashboards/model_performance.html`** - Model performance metrics

### Reports
- **`reports/forecasting_report.md`** - Comprehensive analysis report
- **`reports/model_card.md`** - Model documentation and performance
- **`reports/explainability_report.md`** - Model interpretability analysis

## 🔍 Model Architecture

### Transfer Learning Pipeline
1. **Source Model Training**: Train ensemble on all cities except target
2. **Target Model Training**: Fine-tune on target city data (if available)
3. **Prediction**: Combine source and target model predictions

### Feature Engineering
- **Time Features**: Year, month, week, day, seasonality
- **Lag Features**: 1, 2, 3, 4, 7, 14, 21, 28 day lags
- **Rolling Features**: 7, 14, 28 day windows (mean, std, min, max)
- **External Features**: Weather, promotions, holidays, prices
- **Cross-sectional Features**: City and SKU level aggregations

### Ensemble Models
- **LightGBM**: Gradient boosting with categorical features
- **XGBoost**: Extreme gradient boosting
- **CatBoost**: Categorical boosting
- **Weighted Ensemble**: Combines predictions with learned weights

## 📊 Performance Metrics

### Model Performance
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error
- **MAPE**: Mean Absolute Percentage Error
- **SMAPE**: Symmetric Mean Absolute Percentage Error

### Transfer Learning Effectiveness
- **Source Model Performance**: Baseline performance on source cities
- **Target Model Performance**: Performance on target city
- **Transfer Learning Gain**: Improvement from transfer learning

## 🛠️ Development

### Project Structure
```
src/
├── __init__.py
├── data_loader.py          # Data loading and preprocessing
├── feature_engineering.py  # Feature creation and selection
├── models.py               # Model implementations
├── validation.py          # Model validation framework
├── visualization.py        # Plotting utilities
├── explainability.py      # Model interpretability
└── forecasting_pipeline.py # Main pipeline orchestration
```

### Adding New Features
1. Modify `src/feature_engineering.py`
2. Update feature configuration in `config.yaml`
3. Test with `python -m pytest tests/`

### Adding New Models
1. Implement model class in `src/models.py`
2. Add to ensemble configuration
3. Update model selection logic

## 🧪 Testing

### Run Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_models.py

# Run with coverage
python -m pytest --cov=src tests/
```

### Test Data
- Use `tests/data/` for test datasets
- Mock external dependencies
- Validate forecast accuracy

## 📚 Documentation

### API Documentation
```bash
# Generate API docs
sphinx-build -b html docs/ docs/_build/html
```

### Code Documentation
- Docstrings follow Google style
- Type hints for all functions
- Comprehensive inline comments

## 🚨 Troubleshooting

### Common Issues

#### 1. Memory Issues
```bash
# Reduce batch size in config.yaml
model:
  batch_size: 1000
```

#### 2. Missing Dependencies
```bash
# Reinstall environment
conda env remove -n cold-start-forecasting
conda env create -f environment.yml
```

#### 3. Data Format Issues
- Check CSV encoding (UTF-8)
- Validate date formats (YYYY-MM-DD)
- Ensure numeric columns are properly formatted

#### 4. Model Training Issues
```bash
# Check logs
tail -f logs/enhanced_forecasting.log

# Reduce model complexity
# Edit config.yaml model parameters
```

### Performance Optimization
- Use GPU acceleration for large datasets
- Implement data caching for repeated runs
- Optimize feature engineering pipeline

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Documentation**: [Project Wiki](https://github.com/your-repo/wiki)
- **Email**: support@your-domain.com

## 🙏 Acknowledgments

- Transfer learning methodology inspired by research in cold-start scenarios
- Feature engineering techniques from retail demand forecasting literature
- Visualization libraries: Plotly, Matplotlib, Seaborn
- Machine learning frameworks: LightGBM, XGBoost, CatBoost

---

**Last Updated**: October 2025  
**Version**: 1.0.0  
**Python**: 3.11+  
**Status**: Production Ready
