import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import warnings
warnings.filterwarnings('ignore')

import shap
SHAP_AVAILABLE = True

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

try:
    import eli5
    ELI5_AVAILABLE = True
except ImportError:
    ELI5_AVAILABLE = False

logger = logging.getLogger(__name__)


class SHAPExplainer:
    
    def __init__(self, model, X_background: pd.DataFrame):

        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is not available")
        
        self.model = model
        self.X_background = X_background
        self.explainer = self._create_explainer()
        self.is_fitted = False
    
    def _create_explainer(self):
        model_type = type(self.model).__name__.lower()
        
        if 'lightgbm' in model_type or 'lgb' in model_type:
            return shap.TreeExplainer(self.model)
        elif 'xgboost' in model_type or 'xgb' in model_type:
            return shap.TreeExplainer(self.model)
        elif 'catboost' in model_type:
            return shap.TreeExplainer(self.model)
        elif 'ensemble' in model_type:
            return shap.Explainer(self._ensemble_predict, self.X_background)
        else:
            return shap.KernelExplainer(self._model_predict, self.X_background)
    
    def _model_predict(self, X):
        if hasattr(self.model, 'predict'):
            return self.model.predict(X)
        else:
            return self.model.predict(X.values)
    
    def _ensemble_predict(self, X):
        return self.model.predict(X)
    
    def explain_instance(self, X_instance: pd.DataFrame) -> Dict[str, Any]:
        shap_values = self.explainer.shap_values(X_instance)

        feature_names = X_instance.columns.tolist()
        explanation = {
            'shap_values': shap_values[0] if len(shap_values.shape) > 1 else shap_values,
            'feature_names': feature_names,
            'feature_importance': dict(zip(feature_names, np.abs(shap_values[0] if len(shap_values.shape) > 1 else shap_values))),
            'base_value': self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0,
            'prediction': self._model_predict(X_instance)[0] if len(X_instance) == 1 else self._model_predict(X_instance)
        }
        
        return explanation
    
    def explain_global(self, X_sample: pd.DataFrame, max_samples: int = 100) -> Dict[str, Any]:
        if len(X_sample) > max_samples:
            X_sample = X_sample.sample(max_samples, random_state=42)
        shap_values = self.explainer.shap_values(X_sample)
        feature_importance = np.mean(np.abs(shap_values), axis=0)
        feature_names = X_sample.columns.tolist()
        global_explanation = {
            'feature_importance': dict(zip(feature_names, feature_importance)),
            'feature_importance_ranked': sorted(
                zip(feature_names, feature_importance), 
                key=lambda x: x[1], 
                reverse=True
            ),
            'shap_values': shap_values,
            'feature_names': feature_names
        }
        
        return global_explanation


class LIMEExplainer:
    def __init__(self, model, X_background: pd.DataFrame):
        if not LIME_AVAILABLE:
            raise ImportError("LIME is not available")
        
        self.model = model
        self.X_background = X_background
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            self.X_background.values,
            feature_names=self.X_background.columns,
            mode='regression',
            discretize_continuous=True
        )
    
    def explain_instance(self, X_instance: pd.DataFrame) -> Dict[str, Any]:
        explanation = self.explainer.explain_instance(
            X_instance.values[0],
            self._model_predict,
            num_features=len(X_instance.columns)
        )
        feature_importance = {}
        for feature, importance in explanation.as_list():
            feature_importance[feature] = importance
        lime_explanation = {
            'feature_importance': feature_importance,
            'prediction': explanation.predicted_value,
            'explanation': explanation.as_list(),
            'explanation_html': explanation.as_html()
        }
        
        return lime_explanation
    
    def _model_predict(self, X):
        if hasattr(self.model, 'predict'):
            return self.model.predict(X)
        else:
            return self.model.predict(X)


class ELI5Explainer:
    def __init__(self, model):
        if not ELI5_AVAILABLE:
            raise ImportError("ELI5 is not available")
        
        self.model = model
    
    def explain_global(self, X_sample: pd.DataFrame) -> Dict[str, Any]
        feature_importance = eli5.explain_weights(self.model, feature_names=X_sample.columns.tolist())
        importance_dict = {}
        if hasattr(feature_importance, 'feature_importances_'):
            importance_dict = dict(zip(X_sample.columns, feature_importance.feature_importances_))
        # Create explanation dictionary
        eli5_explanation = {
            'feature_importance': importance_dict,
            'explanation_html': eli5.format_as_html(feature_importance),
            'explanation_text': eli5.format_as_text(feature_importance)
        }
        
        return eli5_explanation


class DriverAttribution:
    def __init__(self, config: Dict):
        self.config = config
        self.explainability_config = config['explainability']
        self.top_features = self.explainability_config['top_features']
        self.methods = self.explainability_config['methods']
    
    def analyze_drivers(self, model, X: pd.DataFrame, y: pd.Series = None, 
                       X_sample: pd.DataFrame = None) -> Dict[str, Any]:
        logger.info("Starting driver attribution analysis...")
        
        driver_analysis = {
            'global_analysis': {},
            'feature_importance': {},
            'driver_ranking': {},
            'method_comparison': {}
        }
        if X_sample is None:
            X_sample = X.sample(min(100, len(X)), random_state=42)
        for method in self.methods:
            try:
                if method == 'shap' and SHAP_AVAILABLE:
                    explainer = SHAPExplainer(model, X_sample)
                    global_explanation = explainer.explain_global(X_sample)
                    driver_analysis['global_analysis']['shap'] = global_explanation
                    driver_analysis['feature_importance']['shap'] = global_explanation['feature_importance']
                
                elif method == 'lime' and LIME_AVAILABLE:
                    explainer = LIMEExplainer(model, X_sample)
                    lime_explanations = []
                    for i in range(min(10, len(X_sample))):
                        explanation = explainer.explain_instance(X_sample.iloc[[i]])
                        lime_explanations.append(explanation)
                    
                    driver_analysis['global_analysis']['lime'] = lime_explanations
                
                elif method == 'feature_importance':
                    if hasattr(model, 'feature_importances_'):
                        importance = model.feature_importances_
                        feature_names = X.columns.tolist()
                        driver_analysis['feature_importance']['built_in'] = dict(zip(feature_names, importance))
                    elif hasattr(model, 'get_feature_importance'):
                        driver_analysis['feature_importance']['built_in'] = model.get_feature_importance()
                
                elif method == 'eli5' and ELI5_AVAILABLE:
                    explainer = ELI5Explainer(model)
                    eli5_explanation = explainer.explain_global(X_sample)
                    driver_analysis['global_analysis']['eli5'] = eli5_explanation
                    driver_analysis['feature_importance']['eli5'] = eli5_explanation['feature_importance']
                
            except Exception as e:
                logger.warning(f"Failed to apply {method} explainability: {str(e)}")
                continue

        driver_analysis['driver_ranking'] = self._create_driver_ranking(driver_analysis['feature_importance'])
        driver_analysis['method_comparison'] = self._compare_methods(driver_analysis['feature_importance'])
        
        logger.info("Driver attribution analysis completed")
        
        return driver_analysis
    
    def _create_driver_ranking(self, feature_importance_dict: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        if not feature_importance_dict:
            return {}

        all_features = set()
        for method, importance in feature_importance_dict.items():
            all_features.update(importance.keys())

        feature_scores = {}
        for feature in all_features:
            scores = []
            for method, importance in feature_importance_dict.items():
                if feature in importance:
                    scores.append(importance[feature])
            
            if scores:
                feature_scores[feature] = np.mean(scores)
        ranked_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)

        top_features = ranked_features[:self.top_features]
        
        return {
            'top_drivers': top_features,
            'all_drivers': ranked_features,
            'driver_scores': feature_scores
        }
    
    def _compare_methods(self, feature_importance_dict: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        if len(feature_importance_dict) < 2:
            return {}
        
        methods = list(feature_importance_dict.keys())
        comparison = {}

        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                importance1 = feature_importance_dict[method1]
                importance2 = feature_importance_dict[method2]
                common_features = set(importance1.keys()) & set(importance2.keys())
                
                if len(common_features) > 1:
                    scores1 = [importance1[f] for f in common_features]
                    scores2 = [importance2[f] for f in common_features]
                    
                    correlation = np.corrcoef(scores1, scores2)[0, 1]
                    comparison[f'{method1}_vs_{method2}'] = {
                        'correlation': correlation,
                        'common_features': len(common_features)
                    }
        
        return comparison
    
    def explain_prediction(self, model, X_instance: pd.DataFrame) -> Dict[str, Any]:
        prediction_explanation = {
            'prediction': model.predict(X_instance)[0] if len(X_instance) == 1 else model.predict(X_instance),
            'feature_contributions': {},
            'method_explanations': {}
        }
        if 'shap' in self.methods and SHAP_AVAILABLE:
            try:
                explainer = SHAPExplainer(model, X_instance)
                shap_explanation = explainer.explain_instance(X_instance)
                prediction_explanation['method_explanations']['shap'] = shap_explanation
                prediction_explanation['feature_contributions']['shap'] = shap_explanation['feature_importance']
            except Exception as e:
                logger.warning(f"SHAP local explanation failed: {str(e)}")
        if 'lime' in self.methods and LIME_AVAILABLE:
            try:
                explainer = LIMEExplainer(model, X_instance)
                lime_explanation = explainer.explain_instance(X_instance)
                prediction_explanation['method_explanations']['lime'] = lime_explanation
                prediction_explanation['feature_contributions']['lime'] = lime_explanation['feature_importance']
            except Exception as e:
                logger.warning(f"LIME local explanation failed: {str(e)}")
        
        return prediction_explanation
    
    def generate_driver_report(self, driver_analysis: Dict[str, Any]) -> Dict[str, Any]:
        report = {
            'summary': {},
            'top_drivers': {},
            'method_agreement': {},
            'insights': []
        }

        if 'driver_ranking' in driver_analysis:
            top_drivers = driver_analysis['driver_ranking'].get('top_drivers', [])
            report['summary'] = {
                'total_features_analyzed': len(driver_analysis['driver_ranking'].get('all_drivers', [])),
                'top_drivers_count': len(top_drivers),
                'methods_used': len(driver_analysis.get('feature_importance', {}))
            }

        if 'driver_ranking' in driver_analysis:
            top_drivers = driver_analysis['driver_ranking'].get('top_drivers', [])
            report['top_drivers'] = {
                'primary_drivers': top_drivers[:5],
                'secondary_drivers': top_drivers[5:10] if len(top_drivers) > 5 else [],
                'driver_categories': self._categorize_drivers(top_drivers)
            }

        if 'method_comparison' in driver_analysis:
            report['method_agreement'] = driver_analysis['method_comparison']
        report['insights'] = self._generate_insights(driver_analysis)
        
        return report
    
    def _categorize_drivers(self, top_drivers: List[Tuple[str, float]]) -> Dict[str, List[str]]:
        categories = {
            'time_features': [],
            'lag_features': [],
            'external_regressors': [],
            'cross_sectional': [],
            'other': []
        }
        
        for feature, importance in top_drivers:
            if any(time_feat in feature.lower() for time_feat in ['year', 'month', 'week', 'day', 'quarter']):
                categories['time_features'].append(feature)
            elif 'lag' in feature.lower():
                categories['lag_features'].append(feature)
            elif any(ext_feat in feature.lower() for ext_feat in ['price', 'promo', 'weather', 'holiday']):
                categories['external_regressors'].append(feature)
            elif any(cs_feat in feature.lower() for cs_feat in ['city', 'sku', 'mean', 'std']):
                categories['cross_sectional'].append(feature)
            else:
                categories['other'].append(feature)
        
        return categories
    
    def _generate_insights(self, driver_analysis: Dict[str, Any]) -> List[str]:
        insights = []

        if 'driver_ranking' in driver_analysis:
            top_drivers = driver_analysis['driver_ranking'].get('top_drivers', [])
            if top_drivers:
                top_feature, top_importance = top_drivers[0]
                insights.append(f"Primary demand driver: {top_feature} (importance: {top_importance:.4f})")
        if 'method_comparison' in driver_analysis:
            comparisons = driver_analysis['method_comparison']
            if comparisons:
                avg_correlation = np.mean([comp['correlation'] for comp in comparisons.values()])
                if avg_correlation > 0.8:
                    insights.append("High agreement between explainability methods indicates reliable driver identification")
                elif avg_correlation < 0.5:
                    insights.append("Low agreement between methods suggests complex feature interactions")
        if 'driver_ranking' in driver_analysis:
            categories = self._categorize_drivers(driver_analysis['driver_ranking'].get('top_drivers', []))
            dominant_category = max(categories.items(), key=lambda x: len(x[1]))
            if dominant_category[1]:
                insights.append(f"Dominant driver category: {dominant_category[0]} ({len(dominant_category[1])} features)")
        
        return insights


if __name__ == "__main__":
    import yaml
    
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = pd.DataFrame(np.random.randn(n_samples, n_features), 
                    columns=[f'feature_{i}' for i in range(n_features)])
    y = pd.Series(np.random.poisson(10, n_samples))

    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    driver_attribution = DriverAttribution(config)

    driver_analysis = driver_attribution.analyze_drivers(model, X, y)
    
    print("Driver Analysis Results:")
    print(f"Top drivers: {driver_analysis['driver_ranking']['top_drivers'][:5]}")

    report = driver_attribution.generate_driver_report(driver_analysis)
    print(f"\nInsights: {report['insights']}")

    X_instance = X.iloc[[0]]
    prediction_explanation = driver_attribution.explain_prediction(model, X_instance)
    print(f"\nPrediction: {prediction_explanation['prediction']}")

