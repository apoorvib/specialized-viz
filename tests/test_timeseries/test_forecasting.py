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
        
    def test_cross_validation(self):
        """Test cross-validation with different strategies"""
        features = self.forecaster.create_features('target')
        target = self.test_data['target'][features.index]
        
        # Test default cross-validation
        cv_metrics = self.forecaster.cross_validate(features.dropna(), target)
        self.assertIsInstance(cv_metrics, dict)
        self.assertIn('mse', cv_metrics)
        self.assertIn('mae', cv_metrics)
        self.assertIn('r2', cv_metrics)
        
        # Test with different number of folds
        cv_metrics_3fold = self.forecaster.cross_validate(
            features.dropna(), target, n_splits=3
        )
        self.assertNotEqual(
            cv_metrics['mse_std'],
            cv_metrics_3fold['mse_std']
        )
        
        # Test with expanding window
        cv_metrics_expanding = self.forecaster.cross_validate(
            features.dropna(), target, expanding=True
        )
        self.assertIsInstance(cv_metrics_expanding, dict)

    def test_evaluate_forecast(self):
        """Test comprehensive forecast evaluation metrics"""
        # Create some test predictions
        target = self.test_data['target']
        predictions = target + np.random.normal(0, 0.1, len(target))
        
        metrics = self.forecaster.evaluate_forecast(predictions, target)
        
        # Test basic metrics
        self.assertIn('mse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('r2', metrics)
        self.assertIn('mape', metrics)
        
        # Test directional accuracy
        self.assertIn('directional_accuracy', metrics)
        self.assertTrue(0 <= metrics['directional_accuracy'] <= 100)
        
        # Test scale-independent metrics
        self.assertIn('mase', metrics)
        self.assertIn('nrmse', metrics)
        
        # Test with perfect predictions
        perfect_metrics = self.forecaster.evaluate_forecast(target, target)
        self.assertAlmostEqual(perfect_metrics['mse'], 0)
        self.assertAlmostEqual(perfect_metrics['mae'], 0)
        
        # Test with completely wrong predictions
        wrong_predictions = -target
        wrong_metrics = self.forecaster.evaluate_forecast(wrong_predictions, target)
        self.assertLess(wrong_metrics['r2'], perfect_metrics['r2'])

    def test_generate_scenarios(self):
        """Test scenario generation with different parameters"""
        features = self.forecaster.create_features('target')
        
        # Test with default parameters
        scenarios = self.forecaster.generate_scenarios(features.dropna())
        self.assertIsInstance(scenarios, dict)
        self.assertIn('scenarios', scenarios)
        self.assertIn('statistics', scenarios)
        
        # Test different number of scenarios
        scenarios_50 = self.forecaster.generate_scenarios(
            features.dropna(), n_scenarios=50
        )
        self.assertEqual(len(scenarios_50['scenarios']), 50)
        
        # Test scenario statistics
        stats = scenarios['statistics']
        self.assertIn('median', stats)
        self.assertIn('mean', stats)
        self.assertIn('std', stats)
        self.assertIn('q10', stats)
        self.assertIn('q90', stats)
        
        # Test reproducibility with seed
        scenarios1 = self.forecaster.generate_scenarios(
            features.dropna(), random_state=42
        )
        scenarios2 = self.forecaster.generate_scenarios(
            features.dropna(), random_state=42
        )
        pd.testing.assert_frame_equal(
            scenarios1['scenarios'],
            scenarios2['scenarios']
        )

    def test_prepare_time_series(self):
        """Test time series preparation functionality"""
        series = self.test_data['target']
        
        # Test basic preparation
        prepared = self.forecaster._prepare_time_series(series)
        self.assertIsInstance(prepared, pd.Series)
        self.assertIsInstance(prepared.index, pd.DatetimeIndex)
        
        # Test with missing values
        series_with_nan = series.copy()
        series_with_nan.iloc[10:15] = np.nan
        prepared_nan = self.forecaster._prepare_time_series(series_with_nan)
        self.assertTrue(prepared_nan.notna().all())
        
        # Test with non-datetime index
        wrong_index = series.reset_index()['target']
        with self.assertRaises(ValueError):
            self.forecaster._prepare_time_series(wrong_index)
        
        # Test frequency inference
        prepared_freq = self.forecaster._prepare_time_series(series)
        self.assertIsNotNone(prepared_freq.index.freq)

    def test_set_forecast_index(self):
        """Test forecast index generation"""
        last_date = pd.Timestamp('2023-12-31')
        
        # Test daily frequency
        daily_index = self.forecaster._set_forecast_index(
            last_date, steps=5
        )
        self.assertEqual(len(daily_index), 5)
        self.assertEqual(daily_index.freq, 'B')
        
        # Test weekly frequency
        weekly_index = self.forecaster._set_forecast_index(
            last_date, steps=5, freq='W'
        )
        self.assertEqual(len(weekly_index), 5)
        self.assertEqual(weekly_index.freq, 'W')
        
        # Test monthly frequency
        monthly_index = self.forecaster._set_forecast_index(
            last_date, steps=5, freq='M'
        )
        self.assertEqual(len(monthly_index), 5)
        self.assertEqual(monthly_index.freq, 'M')
        
        # Test with invalid frequency
        with self.assertRaises(ValueError):
            self.forecaster._set_forecast_index(
                last_date, steps=5, freq='invalid'
            )

    def test_train_test_split(self):
        """Test data splitting functionality"""
        features = self.forecaster.create_features('target')
        target = self.test_data['target'][features.index]
        
        # Test default split
        result = self.forecaster.train_forecast_model(
            features.dropna(),
            target
        )
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        
        # Test different split ratios
        result_80_20 = self.forecaster.train_forecast_model(
            features.dropna(),
            target,
            test_size=0.2
        )
        result_70_30 = self.forecaster.train_forecast_model(
            features.dropna(),
            target,
            test_size=0.3
        )
        self.assertNotEqual(
            result_80_20[1]['test_mse'],
            result_70_30[1]['test_mse']
        )

    def test_make_forecast(self):
        """Test forecast generation with different parameters"""
        features = self.forecaster.create_features('target')
        target = self.test_data['target'][features.index]
        
        # Train model
        model, _ = self.forecaster.train_forecast_model(
            features.dropna(),
            target
        )
        
        # Test basic forecast
        forecast = self.forecaster.make_forecast(
            model,
            features.dropna()
        )
        self.assertIsInstance(forecast, dict)
        self.assertIn('forecast', forecast)
        
        # Test with uncertainty intervals
        forecast_with_intervals = self.forecaster.make_forecast(
            model,
            features.dropna(),
            uncertainty=True
        )
        self.assertIn('lower', forecast_with_intervals)
        self.assertIn('upper', forecast_with_intervals)
        
        # Test different confidence levels
        forecast_90 = self.forecaster.make_forecast(
            model,
            features.dropna(),
            uncertainty=True,
            confidence=0.90
        )
        forecast_99 = self.forecaster.make_forecast(
            model,
            features.dropna(),
            uncertainty=True,
            confidence=0.99
        )
        self.assertTrue(
            (forecast_99['upper'] - forecast_99['lower']).mean() >
            (forecast_90['upper'] - forecast_90['lower']).mean()
        )

    def test_residual_analysis(self):
        """Test residual analysis functionality"""
        # Generate some predictions
        features = self.forecaster.create_features('target')
        target = self.test_data['target'][features.index]
        predictions = target + np.random.normal(0, 0.1, len(target))
        
        # Test residual analysis
        residuals = self.forecaster.evaluate_residuals(target, predictions)
        
        # Test basic statistics
        self.assertIn('mean', residuals)
        self.assertIn('std', residuals)
        self.assertIn('skew', residuals)
        self.assertIn('kurtosis', residuals)
        
        # Test Ljung-Box test
        self.assertIn('ljung_box_p', residuals)
        
        # Test with perfect predictions
        perfect_residuals = self.forecaster.evaluate_residuals(
            target,
            target
        )
        self.assertAlmostEqual(perfect_residuals['mean'], 0)
        self.assertAlmostEqual(perfect_residuals['std'], 0)

    def test_process_train_data(self):
        """Test training data processing"""
        series = self.test_data['target']
        
        # Test basic processing
        processed = self.forecaster._process_train_data(series)
        self.assertIsInstance(processed, pd.Series)
        self.assertTrue(processed.index.is_monotonic_increasing)
        
        # Test with missing values
        series_with_nan = series.copy()
        series_with_nan.iloc[10:15] = np.nan
        processed_nan = self.forecaster._process_train_data(series_with_nan)
        self.assertTrue(processed_nan.notna().all())
        
        # Test with irregular index
        irregular_series = series.copy()
        irregular_series.index = pd.date_range(
            '2023-01-01',
            periods=len(series),
            freq='H'
        )
        processed_irregular = self.forecaster._process_train_data(
            irregular_series
        )
        self.assertIsNotNone(processed_irregular.index.freq)

    def test_combined_seasonal_forecast(self):
        """Test combined seasonal forecasting"""
        features = self.forecaster.create_features('target')
        target = self.test_data['target'][features.index]
        
        result = self.forecaster.combined_seasonal_forecast(
            features.dropna(),
            target
        )
        
        # Test result structure
        self.assertIsInstance(result, dict)
        self.assertIn('forecast', result)
        self.assertIn('lower_bound', result)
        self.assertIn('upper_bound', result)
        self.assertIn('component_forecasts', result)
        self.assertIn('models', result)
        
        # Test component forecasts
        self.assertIn('seasonal', result['component_forecasts'])
        self.assertIn('sarima', result['component_forecasts'])
        self.assertIn('ets', result['component_forecasts'])
        
        # Test forecast combinations
        self.assertEqual(
            len(result['forecast']),
            self.forecaster.config.forecast_horizon
        )
        self.assertTrue(
            all(result['lower_bound'] <= result['forecast'])
        )
        self.assertTrue(
            all(result['forecast'] <= result['upper_bound'])
        )

if __name__ == '__main__':
    unittest.main()