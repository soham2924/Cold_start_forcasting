import sys
import logging
from pathlib import Path
import argparse
import yaml
from datetime import datetime

sys.path.append(str(Path(__file__).parent / 'src'))

from src.forecasting_pipeline import ForecastingPipeline
from src.visualization import ForecastVisualizer, ValidationVisualizer, ExplainabilityVisualizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/forecasting.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Retail Demand Forecasting Pipeline')
    parser.add_argument('--config', type=str, default='config.yaml', 
                       help='Path to configuration file')
    parser.add_argument('--data-path', type=str, default='data/', 
                       help='Path to data directory')
    parser.add_argument('--output-path', type=str, default='outputs/', 
                       help='Path to output directory')
    parser.add_argument('--plots-path', type=str, default='plots/', 
                       help='Path to plots directory')
    parser.add_argument('--skip-validation', action='store_true', 
                       help='Skip model validation')
    parser.add_argument('--skip-explainability', action='store_true', 
                       help='Skip explainability analysis')
    parser.add_argument('--skip-plots', action='store_true', 
                       help='Skip plot generation')
    
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    
    config['data']['input_path'] = args.data_path
    config['data']['output_path'] = args.output_path
    config['data']['plots_path'] = args.plots_path

    Path('logs').mkdir(exist_ok=True)
    Path(args.data_path).mkdir(parents=True, exist_ok=True)
    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    Path(args.plots_path).mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting Retail Demand Forecasting Pipeline")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Output path: {args.output_path}")
    
    try:

        pipeline = ForecastingPipeline(args.config)

        results = pipeline.run_full_pipeline()
        
        if results['status'] == 'success':
            logger.info("Pipeline completed successfully!")

            if not args.skip_plots:
                logger.info("Generating visualizations...")

                forecast_viz = ForecastVisualizer(config)
                forecast_plots = forecast_viz.plot_forecasts(results['forecasts'])
                logger.info(f"Created {len(forecast_plots)} forecast plots")
                if not args.skip_validation:
                    validation_viz = ValidationVisualizer(config)
                    validation_plots = validation_viz.plot_validation_results(results['validation_results'])
                    logger.info(f"Created {len(validation_plots)} validation plots")

                if not args.skip_explainability:
                    explainability_viz = ExplainabilityVisualizer(config)
                    explainability_plots = explainability_viz.plot_explainability_results(results['explainability_results'])
                    logger.info(f"Created {len(explainability_plots)} explainability plots")
            print_summary(results)
            
        else:
            logger.error(f"Pipeline failed: {results['error']}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        sys.exit(1)


def print_summary(results):
    print("\n" + "="*60)
    print("RETAIL DEMAND FORECASTING PIPELINE SUMMARY")
    print("="*60)

    forecasts = results['forecasts']
    print(f"\n📊 FORECASTS GENERATED:")
    print(f"   • Total SKUs forecasted: {len(forecasts)}")
    print(f"   • Forecast horizon: 13 weeks")
    print(f"   • Target city: Jaipur")

    if 'validation_results' in results:
        validation = results['validation_results']['summary_report']['summary']
        print(f"\n✅ MODEL VALIDATION:")
        print(f"   • Total combinations tested: {validation['total_combinations']}")
        print(f"   • Mean performance score: {validation['mean_score']:.2f}")
        print(f"   • Performance grade distribution: {validation['grade_distribution']}")

    if 'explainability_results' in results:
        driver_report = results['explainability_results']['driver_report']
        print(f"\n🔍 DRIVER ANALYSIS:")
        print(f"   • Total features analyzed: {driver_report['summary']['total_features_analyzed']}")
        print(f"   • Top drivers identified: {driver_report['summary']['top_drivers_count']}")
        print(f"   • Methods used: {driver_report['summary']['methods_used']}")
        
        if driver_report['insights']:
            print(f"\n💡 KEY INSIGHTS:")
            for insight in driver_report['insights'][:3]: 
                print(f"   • {insight}")

    print(f"\n📁 OUTPUT FILES:")
    print(f"   • forecasts.csv - Forecast results")
    print(f"   • trained_model.pkl - Trained model")
    print(f"   • validation_results.pkl - Validation results")
    print(f"   • explainability_results.pkl - Explainability results")
    print(f"   • plots/ - Visualization plots")
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"   1. Review forecasts.csv for predictions")
    print(f"   2. Check plots/ directory for visualizations")
    print(f"   3. Analyze explainability results for insights")
    print(f"   4. Use trained model for future predictions")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
