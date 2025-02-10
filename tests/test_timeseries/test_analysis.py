import unittest
import pandas as pd
import numpy as np
from specialized_viz.timeseries.analysis import TimeseriesAnalysis, TimeseriesConfig

class TestTimeseriesAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data used across test methods"""
        # Create sample timeseries data with known patterns
        dates = pd.date_range(start='2023-01-01', periods=1000, freq='D')
        
        # Create trend
        trend = np.linspace(0, 10, 1000)
        
        # Create seasonality with multiple periods
        daily = np.sin(np.linspace(0, 20*np.pi, 1000))  # Daily pattern
        weekly = 2 * np.sin(np.linspace(0, 20*np.pi/7, 1000))  # Weekly pattern
        monthly = 3 * np.sin(np.linspace(0, 20*np.pi/30, 1000))  # Monthly pattern
        
        # Create cyclical component
        cycle = 5 * np.sin(np.linspace(0, 4*np.pi, 1000))
        
        # Add noise
        noise = np.random.normal(0, 0.5, 1000)
        
        # Combine components
        values = trend + daily + weekly + monthly + cycle + noise
        
        # Create main test DataFrame
        cls.test_data = pd.DataFrame({
            'value': values,
            'volume': np.random.randint(1000, 5000, 1000),
            'categorical': np.random.choice(['A', 'B', 'C'], 1000)
        }, index=dates)
        
        # Add some missing values
        cls.test_data.loc[dates[10:15], 'value'] = np.nan
        
        # Add some anomalies
        cls.test_data.loc[dates[100], 'value'] = values[100] + 100
        cls.test_data.loc[dates[200], 'value'] = values[200] - 100
        
        # Create analyzer instance with default config
        cls.analyzer = TimeseriesAnalysis(cls.test_data)
        
        # Create analyzer instance with custom config
        custom_config = TimeseriesConfig(
            decomposition_method='multiplicative',
            seasonal_periods=[7, 30, 90],
            trend_window=30,
            cycle_max_period=365
        )
        cls.custom_analyzer = TimeseriesAnalysis(cls.test_data, custom_config)

    def test_initialization_and_validation(self):
        """Test initialization and data validation"""
        # Test with valid data
        self.assertIsInstance(self.analyzer, TimeseriesAnalysis)
        
        # Test with non-datetime index
        wrong_index_data = pd.DataFrame({
            'value': range(10)
        })
        with self.assertRaises(ValueError):
            TimeseriesAnalysis(wrong_index_data)
        
        # Test with empty DataFrame
        with self.assertRaises(ValueError):
            TimeseriesAnalysis(pd.DataFrame())
        
        # Test with invalid config
        invalid_config = TimeseriesConfig(seasonal_periods=[-1])
        with self.assertRaises(ValueError):
            TimeseriesAnalysis(self.test_data, invalid_config)

    def test_seasonality_analysis(self):
        """Test seasonality analysis functionality"""
        result = self.analyzer.analyze_seasonality('value')
        
        # Test basic structure
        self.assertIsInstance(result, dict)
        self.assertIn('seasonal_strength_5', result)
        self.assertIn('seasonal_changes', result)
        self.assertIn('holiday_effects', result)
        
        # Test seasonal strength calculation
        for period in self.analyzer.config.seasonal_periods:
            key = f'seasonal_strength_{period}'
            self.assertIn(key, result)
            self.assertTrue(0 <= result[key] <= 1)
        
        # Test with different decomposition methods
        multiplicative_result = self.custom_analyzer.analyze_seasonality('value')
        self.assertNotEqual(
            result['seasonal_strength_7'],
            multiplicative_result['seasonal_strength_7']
        )
        
        # Test with insufficient data
        short_data = self.test_data.iloc[:10].copy()
        short_analyzer = TimeseriesAnalysis(short_data)
        with self.assertWarns(Warning):
            short_analyzer.analyze_seasonality('value')

    def test_anomaly_detection(self):
        """Test anomaly detection with various methods"""
        # Test all methods
        methods = ['iqr', 'zscore', 'isolation_forest', 'dbscan', 'lof']
        anomalies = self.analyzer.detect_anomalies('value', methods=methods)
        
        # Check structure
        self.assertIsInstance(anomalies, dict)
        for method in methods:
            self.assertIn(method, anomalies)
            self.assertIsInstance(anomalies[method], pd.Series)
        
        # Verify known anomalies are detected
        known_anomaly_dates = [
            pd.Timestamp('2023-04-10'),  # index 100
            pd.Timestamp('2023-07-19')   # index 200
        ]
        
        for method in methods:
            detected = anomalies[method]
            for date in known_anomaly_dates:
                self.assertTrue(detected[date], 
                              f"Known anomaly at {date} not detected by {method}")
        
        # Test threshold sensitivity
        high_threshold = self.analyzer.detect_anomalies(
            'value', 
            methods=['iqr'], 
            threshold=0.5
        )
        low_threshold = self.analyzer.detect_anomalies(
            'value', 
            methods=['iqr'], 
            threshold=0.01
        )
        self.assertTrue(low_threshold['iqr'].sum() > high_threshold['iqr'].sum())
        
        # Test with missing values
        na_data = self.test_data.copy()
        na_data.loc[na_data.index[300:310], 'value'] = np.nan
        na_analyzer = TimeseriesAnalysis(na_data)
        na_anomalies = na_analyzer.detect_anomalies('value')
        self.assertTrue(all(pd.isna(na_anomalies['iqr'][300:310])))

    def test_cycle_analysis(self):
        """Test cycle analysis functionality"""
        cycles = self.analyzer.analyze_cycles('value')
        
        # Test structure
        self.assertIsInstance(cycles, dict)
        self.assertIn('fourier', cycles)
        self.assertIn('wavelet', cycles)
        self.assertIn('periodicity', cycles)
        self.assertIn('phase', cycles)
        
        # Test Fourier analysis
        fourier = cycles['fourier']
        self.assertIn('dominant_frequencies', fourier)
        self.assertIn('amplitudes', fourier)
        self.assertEqual(
            len(fourier['dominant_frequencies']),
            len(fourier['amplitudes'])
        )
        
        # Test wavelet analysis
        wavelet = cycles['wavelet']
        self.assertIn('dominant_scales', wavelet)
        self.assertIn('power_spectrum', wavelet)
        
        # Test periodicity detection
        periodicity = cycles['periodicity']
        self.assertIn('periodic_lengths', periodicity)
        self.assertIn('correlation_strength', periodicity)
        self.assertIn('autocorrelation', periodicity)
        
        # Test with different cycle_max_period
        custom_cycles = self.custom_analyzer.analyze_cycles('value')
        self.assertNotEqual(
            len(cycles['periodicity']['periodic_lengths']),
            len(custom_cycles['periodicity']['periodic_lengths'])
        )
        
        # Test with noisy data
        noisy_data = self.test_data.copy()
        noisy_data['value'] += np.random.normal(0, 10, len(noisy_data))
        noisy_analyzer = TimeseriesAnalysis(noisy_data)
        noisy_cycles = noisy_analyzer.analyze_cycles('value')
        self.assertLess(
            max(noisy_cycles['periodicity']['correlation_strength']),
            max(cycles['periodicity']['correlation_strength'])
        )

    def test_causality_tests(self):
        """Test causality analysis"""
        # Create second series with known relationship
        self.test_data['lagged_value'] = self.test_data['value'].shift(1)
        
        causality = self.analyzer.causality_tests('value', 'lagged_value')
        
        # Test structure
        self.assertIsInstance(causality, dict)
        self.assertIn('granger', causality)
        self.assertIn('cross_correlation', causality)
        self.assertIn('transfer_entropy', causality)
        self.assertIn('mutual_information', causality)
        
        # Test Granger causality
        granger = causality['granger']
        self.assertIn('1_causes_2', granger)
        self.assertIn('2_causes_1', granger)
        
        # Test cross-correlation
        cross_corr = causality['cross_correlation']
        self.assertIn('lags', cross_corr)
        self.assertIn('correlations', cross_corr)
        self.assertIn('max_correlation_lag', cross_corr)
        
        # Test with independent series
        random_series = pd.Series(
            np.random.randn(len(self.test_data)),
            index=self.test_data.index
        )
        ind_causality = self.analyzer.causality_tests('value', random_series.name)
        self.assertLess(
            ind_causality['mutual_information'],
            causality['mutual_information']
        )

    def test_stationarity_analysis(self):
        """Test stationarity analysis"""
        stationarity = self.analyzer.analyze_stationarity('value')
        
        # Test structure
        self.assertIsInstance(stationarity, dict)
        self.assertIn('adf_test', stationarity)
        self.assertIn('kpss_test', stationarity)
        self.assertIn('rolling_stats', stationarity)
        
        # Test ADF test results
        adf = stationarity['adf_test']
        self.assertIn('statistic', adf)
        self.assertIn('p_value', adf)
        self.assertIn('critical_values', adf)
        self.assertIn('is_stationary', adf)
        
        # Test KPSS test results
        kpss = stationarity['kpss_test']
        self.assertIn('statistic', kpss)
        self.assertIn('p_value', kpss)
        self.assertIn('critical_values', kpss)
        self.assertIn('is_trend_stationary', kpss)
        
        # Test with stationary data
        stationary_data = pd.DataFrame({
            'value': np.random.randn(1000)
        }, index=self.test_data.index)
        stationary_analyzer = TimeseriesAnalysis(stationary_data)
        stationary_results = stationary_analyzer.analyze_stationarity('value')
        self.assertTrue(stationary_results['adf_test']['is_stationary'])
        
        # Test with trending data
        trend_data = pd.DataFrame({
            'value': np.linspace(0, 100, 1000)
        }, index=self.test_data.index)
        trend_analyzer = TimeseriesAnalysis(trend_data)
        trend_results = trend_analyzer.analyze_stationarity('value')
        self.assertFalse(trend_results['adf_test']['is_stationary'])

    def test_structural_breaks(self):
        """Test structural break detection"""
        breaks = self.analyzer.detect_structural_breaks('value')
        
        # Test structure
        self.assertIsInstance(breaks, dict)
        self.assertIn('cusum_test', breaks)
        self.assertIn('break_points', breaks)
        self.assertIn('multiple_breaks', breaks)
        
        # Test CUSUM test
        cusum = breaks['cusum_test']
        self.assertIn('statistic', cusum)
        self.assertIn('p_value', cusum)
        self.assertIn('has_break', cusum)
        
        # Test break points
        self.assertIsInstance(breaks['break_points'], list)
        for point in breaks['break_points']:
            self.assertIn('index', point)
            self.assertIn('statistic', point)
            self.assertIn('p_value', point)
        
        # Test with data containing known breaks
        break_data = self.test_data.copy()
        break_data.loc['2023-06-01':, 'value'] += 50
        break_analyzer = TimeseriesAnalysis(break_data)
        break_results = break_analyzer.detect_structural_breaks('value')
        self.assertTrue(break_results['cusum_test']['has_break'])
        
        # Test with stable data
        stable_data = pd.DataFrame({
            'value': np.random.randn(1000)
        }, index=self.test_data.index)
        stable_analyzer = TimeseriesAnalysis(stable_data)
        stable_results = stable_analyzer.detect_structural_breaks('value')
        self.assertFalse(stable_results['cusum_test']['has_break'])

    def test_initialization(self):
        """Test analyzer initialization"""
        self.assertIsInstance(self.analyzer, TimeseriesAnalysis)
        self.assertIsInstance(self.analyzer.config, TimeseriesConfig)
        self.assertTrue(hasattr(self.analyzer, 'data'))

    def test_analyze_seasonality(self):
        """Test seasonality analysis"""
        seasonality = self.analyzer.analyze_seasonality('value')
        self.assertIsInstance(seasonality, dict)
        self.assertIn('seasonal_strength_5', seasonality)
        self.assertIn('seasonal_changes', seasonality)
        self.assertIn('holiday_effects', seasonality)

    def test_detect_anomalies(self):
        """Test anomaly detection with different methods"""
        anomalies = self.analyzer.detect_anomalies('value')
        self.assertIsInstance(anomalies, dict)
        for method in ['iqr', 'zscore', 'isolation_forest', 'dbscan', 'lof']:
            self.assertIn(method, anomalies)
            self.assertIsInstance(anomalies[method], pd.Series)
            # Verify known anomalies are detected
            self.assertTrue(anomalies[method]['2023-02-01'])
            self.assertTrue(anomalies[method]['2023-03-01'])

    def test_analyze_cycles(self):
        """Test cycle analysis"""
        cycles = self.analyzer.analyze_cycles('value')
        self.assertIsInstance(cycles, dict)
        self.assertIn('fourier', cycles)
        self.assertIn('wavelet', cycles)
        self.assertIn('periodicity', cycles)
        self.assertIn('phase', cycles)

    def test_causality_tests(self):
        """Test causality analysis"""
        causality = self.analyzer.causality_tests('value', 'trend')
        self.assertIsInstance(causality, dict)
        self.assertIn('granger', causality)
        self.assertIn('cross_correlation', causality)
        self.assertIn('transfer_entropy', causality)
        self.assertIn('mutual_information', causality)

    def test_analyze_stationarity(self):
        """Test stationarity analysis"""
        stationarity = self.analyzer.analyze_stationarity('value')
        self.assertIsInstance(stationarity, dict)
        self.assertIn('adf_test', stationarity)
        self.assertIn('kpss_test', stationarity)
        self.assertIn('rolling_stats', stationarity)

    def test_detect_structural_breaks(self):
        """Test structural break detection"""
        breaks = self.analyzer.detect_structural_breaks('value')
        self.assertIsInstance(breaks, dict)
        self.assertIn('cusum_test', breaks)
        self.assertIn('break_points', breaks)
        self.assertIn('multiple_breaks', breaks)

    def test_analyze_nonlinearity(self):
        """Test nonlinearity analysis"""
        nonlinearity = self.analyzer.analyze_nonlinearity('value')
        self.assertIsInstance(nonlinearity, dict)
        self.assertIn('terasvirta_test', nonlinearity)
        self.assertIn('bds_test', nonlinearity)
        self.assertIn('correlation_dimension', nonlinearity)
        self.assertIn('lyapunov_exponent', nonlinearity)

    def test_decompose(self):
        """Test time series decomposition"""
        decomposition = self.analyzer.decompose('value')
        self.assertIsInstance(decomposition, dict)
        self.assertIn('trend', decomposition)
        self.assertIn('seasonal', decomposition)
        self.assertIn('residual', decomposition)
        self.assertIn('original', decomposition)

    def test_analyze_trend(self):
        """Test trend analysis"""
        trend = self.analyzer.analyze_trend('value')
        self.assertIsInstance(trend, dict)
        self.assertIn('trend_line', trend)
        self.assertIn('slope', trend)
        self.assertIn('r_squared', trend)
        self.assertIn('mann_kendall_stats', trend)

    def test_detect_change_points(self):
        """Test change point detection"""
        # Test CUSUM method
        cusum_changes = self.analyzer.detect_change_points('value', method='cusum')
        self.assertIsInstance(cusum_changes, pd.Series)
        
        # Test Pettitt method
        pettitt_changes = self.analyzer.detect_change_points('value', method='pettitt')
        self.assertIsInstance(pettitt_changes, pd.Series)

    def test_config_validation(self):
        """Test configuration validation"""
        # Test with invalid decomposition method
        with self.assertRaises(ValueError):
            invalid_config = TimeseriesConfig(decomposition_method='invalid')
            TimeseriesAnalysis(self.test_data, invalid_config)

    def test_seasonal_periods(self):
        """Test different seasonal periods"""
        for period in [5, 7, 12, 30]:
            result = self.analyzer.decompose('value', period=period)
            self.assertIsInstance(result, dict)
            self.assertEqual(len(result['seasonal']), len(self.test_data))

    def test_missing_data_handling(self):
        """Test handling of missing data"""
        # Create data with missing values
        data_with_nan = self.test_data.copy()
        data_with_nan.loc['2023-01-15':'2023-01-20', 'value'] = np.nan
        
        analyzer = TimeseriesAnalysis(data_with_nan)
        result = analyzer.analyze_seasonality('value')
        self.assertIsInstance(result, dict)

    def test_config_initialization(self):
        """Test TimeseriesConfig initialization and validation"""
        # Test default configuration
        config = TimeseriesConfig()
        self.assertEqual(config.decomposition_method, 'additive')
        self.assertEqual(config.trend_window, 20)
        self.assertEqual(config.forecast_horizon, 30)
        self.assertEqual(config.seasonal_periods, [5, 21, 63, 252])
        
        # Test custom configuration
        custom_config = TimeseriesConfig(
            decomposition_method='multiplicative',
            trend_window=30,
            seasonal_periods=[7, 14, 28],
            anomaly_threshold=0.2
        )
        self.assertEqual(custom_config.decomposition_method, 'multiplicative')
        self.assertEqual(custom_config.trend_window, 30)
        self.assertEqual(custom_config.seasonal_periods, [7, 14, 28])
        self.assertEqual(custom_config.anomaly_threshold, 0.2)
        
        # Test invalid configuration
        with self.assertRaises(ValueError):
            TimeseriesConfig(decomposition_method='invalid')
        with self.assertRaises(ValueError):
            TimeseriesConfig(trend_window=-1)
        with self.assertRaises(ValueError):
            TimeseriesConfig(seasonal_periods=[0])
        with self.assertRaises(ValueError):
            TimeseriesConfig(anomaly_threshold=2.0)

    def test_nonlinearity_analysis(self):
        """Test nonlinearity analysis functionality"""
        result = self.analyzer.analyze_nonlinearity('value')
        
        # Test structure
        self.assertIsInstance(result, dict)
        self.assertIn('terasvirta_test', result)
        self.assertIn('bds_test', result)
        self.assertIn('correlation_dimension', result)
        self.assertIn('lyapunov_exponent', result)
        
        # Test Terasvirta test
        terasvirta = result['terasvirta_test']
        self.assertIn('statistic', terasvirta)
        self.assertIn('p_value', terasvirta)
        self.assertIn('is_nonlinear', terasvirta)
        
        # Test BDS test
        bds = result['bds_test']
        self.assertIn('statistics', bds)
        self.assertIn('dimensions', bds)
        self.assertIn('is_independent', bds)
        
        # Test with linear data
        linear_data = pd.DataFrame({
            'value': np.linspace(0, 100, 1000)
        }, index=self.test_data.index)
        linear_analyzer = TimeseriesAnalysis(linear_data)
        linear_result = linear_analyzer.analyze_nonlinearity('value')
        self.assertFalse(linear_result['terasvirta_test']['is_nonlinear'])
        
        # Test with nonlinear data (logistic map)
        x = np.zeros(1000)
        x[0] = 0.5
        r = 3.9  # Chaos parameter
        for i in range(1, 1000):
            x[i] = r * x[i-1] * (1 - x[i-1])
        nonlinear_data = pd.DataFrame({
            'value': x
        }, index=self.test_data.index)
        nonlinear_analyzer = TimeseriesAnalysis(nonlinear_data)
        nonlinear_result = nonlinear_analyzer.analyze_nonlinearity('value')
        self.assertTrue(nonlinear_result['terasvirta_test']['is_nonlinear'])

    def test_helper_methods(self):
        """Test internal helper methods"""
        # Test _calculate_atr
        atr = self.analyzer._calculate_atr(self.test_data, window=14)
        self.assertIsInstance(atr, pd.Series)
        self.assertEqual(len(atr), len(self.test_data))
        self.assertTrue(all(x >= 0 for x in atr.dropna()))
        
        # Test _check_stable_statistics
        is_stable = self.analyzer._check_stable_statistics(self.test_data['value'])
        self.assertIsInstance(is_stable, bool)
        
        # Test with clearly unstable data
        unstable_data = pd.concat([
            pd.Series(np.ones(500)),
            pd.Series(np.ones(500) * 1000)
        ])
        is_unstable = self.analyzer._check_stable_statistics(unstable_data)
        self.assertFalse(is_unstable)
        
        # Test _detect_chow_breaks
        breaks = self.analyzer._detect_chow_breaks(self.test_data['value'])
        self.assertIsInstance(breaks, list)
        for break_point in breaks:
            self.assertIn('index', break_point)
            self.assertIn('statistic', break_point)
            self.assertIn('p_value', break_point)
        
        # Test _detect_multiple_breaks
        multiple_breaks = self.analyzer._detect_multiple_breaks(self.test_data['value'])
        self.assertIsInstance(multiple_breaks, list)
        self.assertTrue(all('index' in b and 'type' in b for b in multiple_breaks))

    def test_large_dataset_performance(self):
        """Test performance with large datasets"""
        # Create large dataset (100K points)
        dates = pd.date_range('2020-01-01', periods=100000, freq='5min')
        large_data = pd.DataFrame({
            'value': np.random.randn(100000),
            'volume': np.random.randint(1000, 5000, 100000)
        }, index=dates)
        
        large_analyzer = TimeseriesAnalysis(large_data)
        
        # Test basic analysis performance
        import time
        
        # Test seasonality analysis
        start_time = time.time()
        large_analyzer.analyze_seasonality('value')
        season_time = time.time() - start_time
        self.assertLess(season_time, 30)  # Should complete within 30 seconds
        
        # Test anomaly detection performance
        start_time = time.time()
        large_analyzer.detect_anomalies('value')
        anomaly_time = time.time() - start_time
        self.assertLess(anomaly_time, 30)
        
        # Test structural breaks performance
        start_time = time.time()
        large_analyzer.detect_structural_breaks('value')
        breaks_time = time.time() - start_time
        self.assertLess(breaks_time, 30)

    def test_statistical_distributions(self):
        """Test statistical distribution analysis"""
        # Test with normal distribution
        normal_data = pd.DataFrame({
            'value': np.random.normal(0, 1, 1000)
        }, index=self.test_data.index)
        normal_analyzer = TimeseriesAnalysis(normal_data)
        normal_stats = normal_analyzer.analyze_stationarity('value')
        self.assertTrue(normal_stats['adf_test']['is_stationary'])
        
        # Test with uniform distribution
        uniform_data = pd.DataFrame({
            'value': np.random.uniform(0, 1, 1000)
        }, index=self.test_data.index)
        uniform_analyzer = TimeseriesAnalysis(uniform_data)
        uniform_stats = uniform_analyzer.analyze_stationarity('value')
        self.assertTrue(uniform_stats['adf_test']['is_stationary'])
        
        # Test with exponential distribution
        exp_data = pd.DataFrame({
            'value': np.random.exponential(1, 1000)
        }, index=self.test_data.index)
        exp_analyzer = TimeseriesAnalysis(exp_data)
        exp_stats = exp_analyzer.analyze_stationarity('value')
        self.assertTrue(exp_stats['adf_test']['is_stationary'])
        
        # Test with lognormal distribution
        lognorm_data = pd.DataFrame({
            'value': np.random.lognormal(0, 1, 1000)
        }, index=self.test_data.index)
        lognorm_analyzer = TimeseriesAnalysis(lognorm_data)
        lognorm_stats = lognorm_analyzer.analyze_stationarity('value')
        self.assertFalse(lognorm_stats['adf_test']['is_stationary'])

    def test_advanced_decomposition(self):
        """Test advanced decomposition methods"""
        # Test multiplicative decomposition
        mult_config = TimeseriesConfig(decomposition_method='multiplicative')
        mult_analyzer = TimeseriesAnalysis(self.test_data, mult_config)
        mult_decomp = mult_analyzer.decompose('value')
        
        self.assertIsInstance(mult_decomp, dict)
        self.assertTrue(all(x > 0 for x in mult_decomp['trend'].dropna()))
        self.assertTrue(all(x > 0 for x in mult_decomp['seasonal'].dropna()))
        
        # Test with zero values (should raise error for multiplicative)
        zero_data = self.test_data.copy()
        zero_data.loc[zero_data.index[0], 'value'] = 0
        with self.assertRaises(ValueError):
            mult_analyzer = TimeseriesAnalysis(zero_data, mult_config)
            mult_analyzer.decompose('value')
        
        # Test with different seasonal periods
        custom_periods = TimeseriesConfig(seasonal_periods=[7, 30, 90])
        period_analyzer = TimeseriesAnalysis(self.test_data, custom_periods)
        period_decomp = period_analyzer.decompose('value')
        
        self.assertNotEqual(
            period_decomp['seasonal'].std(),
            mult_decomp['seasonal'].std()
        )

    def test_extreme_cases(self):
        """Test handling of extreme cases"""
        # Test with constant data
        const_data = pd.DataFrame({
            'value': np.ones(1000)
        }, index=self.test_data.index)
        const_analyzer = TimeseriesAnalysis(const_data)
        
        # Seasonality should be zero
        const_seasonal = const_analyzer.analyze_seasonality('value')
        self.assertAlmostEqual(const_seasonal['seasonal_strength_5'], 0)
        
        # No anomalies should be detected
        const_anomalies = const_analyzer.detect_anomalies('value')
        self.assertEqual(const_anomalies['iqr'].sum(), 0)
        
        # Test with all missing data
        na_data = pd.DataFrame({
            'value': np.nan * np.ones(1000)
        }, index=self.test_data.index)
        with self.assertRaises(ValueError):
            TimeseriesAnalysis(na_data)
        
        # Test with infinities
        inf_data = pd.DataFrame({
            'value': np.inf * np.ones(1000)
        }, index=self.test_data.index)
        with self.assertRaises(ValueError):
            TimeseriesAnalysis(inf_data)
        
        # Test with very large numbers
        large_data = pd.DataFrame({
            'value': 1e100 * np.ones(1000)
        }, index=self.test_data.index)
        large_analyzer = TimeseriesAnalysis(large_data)
        large_stats = large_analyzer.analyze_stationarity('value')
        self.assertTrue(large_stats['adf_test']['is_stationary'])

if __name__ == '__main__':
    unittest.main()