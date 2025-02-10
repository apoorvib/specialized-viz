import unittest
import pandas as pd
import numpy as np
from specialized_viz.timeseries.forecasting import TimeseriesForecasting, ForecastConfig

class TestTimeseriesForecasting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data used across test methods"""
        # Create test data with known patterns
        dates = pd.date_range(start='2023-01-01', periods=1000, freq='D')
        
        # Create trend component
        trend = np.linspace(0, 10, 1000)
        
        # Create multiple seasonal components
        daily = np.sin(np.linspace(0, 20*np.pi, 1000))  # Daily pattern
        weekly = 2 * np.sin(np.linspace(0, 20*np.pi/7, 1000))  # Weekly pattern
        monthly = 3 * np.sin(np.linspace(0, 20*np.pi/30, 1000))  # Monthly pattern
        
        # Add noise
        noise = np.random.normal(0, 0.5, 1000)
        
        # Combine components
        values = trend + daily + weekly + monthly + noise
        
        # Create exogenous variables
        cls.test_data = pd.DataFrame({
            'target': values,
            'exog1': np.random.randn(1000),
            'exog2': values + np.random.normal(0, 0.1, 1000),  # Correlated with target
            'categorical': np.random.choice(['A', 'B', 'C'], 1000)
        }, index=dates)
        
        # Create forecaster instances
        cls.forecaster = TimeseriesForecasting(cls.test_data)
        
        # Create forecaster with custom config
        custom_config = ForecastConfig(
            forecast_horizon=60,
            train_test_split=0.7,
            cv_folds=3,
            features_lag=[1, 2, 3, 7],
            features_window=[7, 14],
            model_type='lstm',
            use_exogenous=True
        )
        cls.custom_forecaster = TimeseriesForecasting(cls.test_data, custom_config)

    def test_initialization(self):
        """Test initialization and configuration"""
        # Test default initialization
        self.assertIsInstance(self.forecaster, TimeseriesForecasting)
        self.assertIsInstance(self.forecaster.config, ForecastConfig)
        
        # Test with invalid data
        with self.assertRaises(ValueError):
            TimeseriesForecasting(pd.DataFrame())
        
        # Test with non-datetime index
        wrong_index_data = pd.DataFrame({'value': range(10)})
        with self.assertRaises(ValueError):
            TimeseriesForecasting(wrong_index_data)
        
        # Test with invalid config
        invalid_config = ForecastConfig(forecast_horizon=-1)
        with self.assertRaises(ValueError):
            TimeseriesForecasting(self.test_data, invalid_config)

    def test_create_features(self):
        """Test feature creation functionality"""
        features = self.forecaster.create_features('target')
        
        # Test feature structure
        self.assertIsInstance(features, pd.DataFrame)
        
        # Test lag features
        for lag in self.forecaster.config.features_lag:
            self.assertIn(f'lag_{lag}', features.columns)
            
        # Test rolling window features
        for window in self.forecaster.config.features_window:
            self.assertIn(f'rolling_mean_{window}', features.columns)
            self.assertIn(f'rolling_std_{window}', features.columns)
            
        # Test date features
        date_features = ['month', 'day_of_week', 'day_of_year', 'week_of_year']
        for feature in date_features:
            self.assertTrue(any(col.startswith(feature) for col in features.columns))
            
        # Test cyclical encoding
        cyclical_features = ['month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos']
        for feature in cyclical_features:
            self.assertIn(feature, features.columns)
            
        # Test with different feature configurations
        custom_features = self.custom_forecaster.create_features('target')
        self.assertNotEqual(len(features.columns), len(custom_features.columns))

    def test_ensemble_forecast(self):
        """Test ensemble forecasting"""
        features = self.forecaster.create_features('target')
        target = self.test_data['target'][features.index]
        
        result = self.forecaster.ensemble_forecast(features.dropna(), target)
        
        # Test result structure
        self.assertIsInstance(result, dict)
        self.assertIn('predictions', result)
        self.assertIn('weights', result)
        self.assertIn('model_predictions', result)
        
        # Test predictions
        self.assertEqual(len(result['predictions']), len(target))
        
        # Test weights
        weights = result['weights']
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=5)
        self.assertTrue(all(w >= 0 for w in weights.values()))
        
        # Test individual model predictions
        for model_name, preds in result['model_predictions'].items():
            self.assertEqual(len(preds), len(target))

    def test_probabilistic_forecast(self):
        """Test probabilistic forecasting"""
        features = self.forecaster.create_features('target')
        target = self.test_data['target'][features.index]
        
        result = self.forecaster.probabilistic_forecast(features.dropna(), target)
        
        # Test result structure
        self.assertIsInstance(result, dict)
        self.assertIn('median', result)
        self.assertIn('quantiles', result)
        self.assertIn('lower_bound', result)
        self.assertIn('upper_bound', result)
        
        # Test quantile properties
        for q in result['quantiles']:
            self.assertTrue(0 <= q <= 1)
            
        # Test prediction intervals
        self.assertTrue(all(result['lower_bound'] <= result['upper_bound']))
        self.assertTrue(all(result['median'] >= result['lower_bound']))
        self.assertTrue(all(result['median'] <= result['upper_bound']))

    def test_online_learning(self):
        """Test online learning capabilities"""
        features = self.forecaster.create_features('target')
        target = self.test_data['target'][features.index]
        
        self.forecaster.online_learning(
            features.dropna(), 
            target,
            window_size=50
        )
        
        # Test performance logging
        self.assertTrue(hasattr(self.forecaster, 'online_performance'))
        performance = self.forecaster.online_performance
        
        # Test performance metrics
        self.assertIn('actual', performance.columns)
        self.assertIn('predicted', performance.columns)
        self.assertIn('error', performance.columns)
        self.assertIn('drift_level', performance.columns)
        
        # Test drift detection
        self.assertTrue(all(performance['drift_level'] >= 0))

    def test_adaptation_to_regime(self):
        """Test regime adaptation"""
        features = self.forecaster.create_features('target')
        target = self.test_data['target'][features.index]
        
        # Create regime indicators
        regime_indicators = pd.DataFrame({
            'regime': pd.qcut(target, q=3, labels=['Low', 'Medium', 'High'])
        }, index=target.index)
        
        result = self.forecaster.adapt_to_regime(
            features.dropna(),
            target,
            regime_indicators
        )
        
        # Test result structure
        self.assertIsInstance(result, dict)
        self.assertIn('regime_models', result)
        self.assertIn('regime_predictions', result)
        self.assertIn('combined_predictions', result)
        
        # Test regime-specific models
        self.assertEqual(
            len(result['regime_models']),
            len(regime_indicators['regime'].unique())
        )
        
        # Test combined predictions
        self.assertEqual(
            len(result['combined_predictions']),
            len(target)
        )

    def test_feature_selection(self):
        """Test feature selection methods"""
        features = self.forecaster.create_features('target')
        target = self.test_data['target'][features.index]
        
        # Test importance-based selection
        selected_features = self.forecaster.feature_selection(
            features.dropna(),
            target,
            method='importance'
        )
        self.assertIsInstance(selected_features, pd.DataFrame)
        self.assertTrue(len(selected_features.columns) <= len(features.columns))
        
        # Test correlation-based selection
        corr_features = self.forecaster.feature_selection(
            features.dropna(),
            target,
            method='correlation'
        )
        self.assertIsInstance(corr_features, pd.DataFrame)
        self.assertTrue(len(corr_features.columns) <= len(features.columns))

    def test_prophet_forecast(self):
        """Test Prophet forecasting"""
        features = self.forecaster.create_features('target')
        target = self.test_data['target'][features.index]
        
        result = self.forecaster.prophet_forecast(features.dropna(), target)
        
        # Test result structure
        self.assertIsInstance(result, dict)
        self.assertIn('forecast', result)
        self.assertIn('lower_bound', result)
        self.assertIn('upper_bound', result)
        
        # Test forecast properties
        self.assertEqual(
            len(result['forecast']),
            self.forecaster.config.forecast_horizon
        )
        
        # Test prediction intervals
        self.assertTrue(all(result['lower_bound'] <= result['upper_bound']))

    def test_var_forecast(self):
        """Test VAR forecasting"""
        features = self.forecaster.create_features('target')
        target = self.test_data['target'][features.index]
        
        result = self.forecaster.var_forecast(features.dropna(), target)
        
        # Test result structure
        self.assertIsInstance(result, dict)
        self.assertIn('forecast', result)
        self.assertIn('model', result)
        self.assertIn('feature_forecasts', result)
        
        # Test forecast length
        self.assertEqual(
            len(result['forecast']),
            self.forecaster.config.forecast_horizon
        )

    def test_lstm_forecast(self):
        """Test LSTM forecasting"""
        features = self.forecaster.create_features('target')
        target = self.test_data['target'][features.index]
        
        result = self.forecaster.lstm_forecast(features.dropna(), target)
        
        # Test result structure
        self.assertIsInstance(result, dict)
        self.assertIn('forecast', result)
        self.assertIn('model', result)
        
        # Test forecast length
        self.assertEqual(
            len(result['forecast']),
            self.forecaster.config.forecast_horizon
        )

    def test_nbeats_forecast(self):
        """Test N-BEATS forecasting"""
        features = self.forecaster.create_features('target')
        target = self.test_data['target'][features.index]
        
        result = self.forecaster.nbeats_forecast(features.dropna(), target)
        
        # Test result structure
        self.assertIsInstance(result, dict)
        self.assertIn('forecast', result)
        self.assertIn('model', result)
        
        # Test forecast length
        self.assertEqual(
            len(result['forecast']),
            self.forecaster.config.forecast_horizon
        )

    def test_seasonal_forecast(self):
        """Test seasonal forecasting"""
        features = self.forecaster.create_features('target')
        target = self.test_data['target'][features.index]
        
        result = self.forecaster.seasonal_forecast(features.dropna(), target)
        
        # Test result structure
        self.assertIsInstance(result, dict)
        self.assertIn('forecast', result)
        self.assertIn('components', result)
        self.assertIn('model', result)
        
        # Test components
        self.assertIn('trend', result['components'])
        self.assertIn('seasonal', result['components'])
        self.assertIn('resid', result['components'])
        
        # Test forecast length
        self.assertEqual(
            len(result['forecast']),
            self.forecaster.config.forecast_horizon
        )

    def test_ets_forecast(self):
        """Test ETS forecasting"""
        features = self.forecaster.create_features('target')
        target = self.test_data['target'][features.index]
        
        result = self.forecaster.ets_forecast(features.dropna(), target)
        
        # Test result structure
        self.assertIsInstance(result, dict)
        self.assertIn('forecast', result)
        self.assertIn('model', result)
        self.assertIn('params', result)
        
        # Test forecast length
        self.assertEqual(
            len(result['forecast']),
            self.forecaster.config.forecast_horizon
        )

if __name__ == '__main__':
    unittest.main()