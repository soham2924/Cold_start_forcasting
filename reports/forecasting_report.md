
# Cold-Start Demand Forecasting Report
## Jaipur_NewCity - 13 Week Forecast

### Executive Summary
- **Target City**: Jaipur_NewCity
- **Forecast Period**: 2025-01-20 to 2025-04-14
- **Total Forecast**: 7,333 units
- **Average Weekly**: 94.0 units
- **Method**: Transfer Learning from 5 source cities

### SKU Performance Forecast
- **S001**: 712 total units (54.8 ± 0.0 per week)
- **S002**: 700 total units (53.9 ± 0.0 per week)
- **S003**: 1,397 total units (107.5 ± 0.0 per week)
- **S004**: 1,516 total units (116.7 ± 0.1 per week)
- **S005**: 1,437 total units (110.6 ± 0.0 per week)
- **S006**: 1,570 total units (120.7 ± 0.2 per week)

### Data Quality Metrics
- **Source Cities**: Bengaluru, Delhi, Hyderabad, Kolkata, Mumbai
- **Historical Data Points**: 3,246
- **Date Range**: 2023-01-02 to 2025-01-13
- **Features Engineered**: 121

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
