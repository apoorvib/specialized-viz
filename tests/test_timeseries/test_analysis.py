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

if __name__ == '__main__':
    unittest.main()