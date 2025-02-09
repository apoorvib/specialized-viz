import unittest
import pandas as pd
import numpy as np
from specialized_viz.candlestick.visualization import CandlestickVisualizer, VisualizationConfig

class TestCandlestickVisualizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data that will be used across test methods"""
        # Create sample OHLCV data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        cls.test_data = pd.DataFrame({
            'Open': np.random.randint(100, 150, 100),
            'High': np.random.randint(120, 170, 100),
            'Low': np.random.randint(80, 130, 100),
            'Close': np.random.randint(90, 160, 100),
            'Volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
        
        # Ensure High is highest and Low is lowest
        cls.test_data['High'] = cls.test_data[['Open', 'High', 'Close']].max(axis=1)
        cls.test_data['Low'] = cls.test_data[['Open', 'Low', 'Close']].min(axis=1)
        
        # Create visualizer instance
        cls.config = VisualizationConfig()
        cls.visualizer = CandlestickVisualizer(cls.test_data, cls.config)

    def test_visualization_config(self):
        """Test VisualizationConfig initialization"""
        config = VisualizationConfig()
        self.assertIsNotNone(config.color_scheme)
        self.assertEqual(config.theme, 'plotly_white')
        self.assertEqual(config.default_height, 800)
        self.assertEqual(config.default_width, 1200)

    def test_create_candlestick_chart_plotly(self):
        """Test creation of Plotly candlestick chart"""
        fig = self.visualizer.create_candlestick_chart(use_plotly=True)
        self.assertIsNotNone(fig)
        self.assertTrue(hasattr(fig, 'data'))
        self.assertTrue(len(fig.data) >= 1)

    def test_create_candlestick_chart_matplotlib(self):
        """Test creation of Matplotlib candlestick chart"""
        fig = self.visualizer.create_candlestick_chart(use_plotly=False)
        self.assertIsNotNone(fig)

    def test_pattern_clustering(self):
        """Test pattern clustering functionality"""
        clustered_data = self.visualizer._cluster_patterns()
        self.assertIsInstance(clustered_data, pd.DataFrame)
        self.assertIn('cluster', clustered_data.columns)

    def test_create_pattern_cluster_chart(self):
        """Test creation of pattern cluster visualization"""
        fig = self.visualizer.create_pattern_cluster_chart()
        self.assertIsNotNone(fig)
        self.assertTrue(hasattr(fig, 'data'))

    def test_create_multi_timeframe_chart(self):
        """Test creation of multi-timeframe chart"""
        # Create weekly and monthly data
        weekly_data = self.test_data.resample('W').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
        monthly_data = self.test_data.resample('M').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
        
        fig = self.visualizer.create_multi_timeframe_chart(weekly_data, monthly_data)
        self.assertIsNotNone(fig)
        self.assertTrue(hasattr(fig, 'data'))

    def test_create_pattern_reliability_chart(self):
        """Test creation of pattern reliability chart"""
        fig = self.visualizer.create_pattern_reliability_chart()
        self.assertIsNotNone(fig)
        self.assertTrue(hasattr(fig, 'data'))

    def test_create_interactive_dashboard(self):
        """Test creation of interactive dashboard"""
        fig = self.visualizer.create_interactive_dashboard()
        self.assertIsNotNone(fig)
        self.assertTrue(hasattr(fig, 'data'))

    def test_market_regime_detection(self):
        """Test market regime detection"""
        regimes = self.visualizer.detect_market_regime()
        self.assertIsInstance(regimes, pd.DataFrame)
        expected_columns = ['volatility_regime', 'trend_regime', 
                          'volume_regime', 'momentum_regime', 'combined_regime']
        for col in expected_columns:
            self.assertIn(col, regimes.columns)

    def test_create_regime_visualization(self):
        """Test creation of regime visualization"""
        fig = self.visualizer.create_regime_visualization()
        self.assertIsNotNone(fig)
        self.assertTrue(hasattr(fig, 'data'))

if __name__ == '__main__':
    unittest.main()