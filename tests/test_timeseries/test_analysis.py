import unittest
import pandas as pd
import numpy as np
from specialized_viz.timeseries.analysis import TimeseriesAnalysis, TimeseriesConfig

class TestTimeseriesAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data used across test methods"""
        # Create sample timeseries data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        cls.test_data = pd.DataFrame({
            'value': np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.normal(0, 0.1, 100),
            'trend': np.linspace(0, 10, 100),
            'seasonal': np.tile(np.sin(np.linspace(0, 2*np.pi, 20)), 5),
            'volume': np.random.randint(1000, 5000, 100)
        }, index=dates)
        
        # Add some anomalies
        cls.test_data.loc['2023-02-01', 'value'] = 100
        cls.test_data.loc['2023-03-01', 'value'] = -100
        
        cls.analyzer = TimeseriesAnalysis(cls.test_data)

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