import unittest
import pandas as pd
import numpy as np
from specialized_viz.candlestick.visualization import CandlestickVisualizer, VisualizationConfig
import plotly.graph_objects as go


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
        
        # Create visualizer instance with custom config
        cls.config = VisualizationConfig(
            color_scheme={
                'bullish': '#00ff00',
                'bearish': '#ff0000',
                'neutral': '#0000ff',
                'complex': '#ff00ff',
                'volume_up': '#00ff00',
                'volume_down': '#ff0000',
                'background': '#ffffff',
                'text': '#000000'
            }
        )
        cls.visualizer = CandlestickVisualizer(cls.test_data, cls.config)

    def test_visualization_config(self):
        """Test VisualizationConfig initialization and properties"""
        config = VisualizationConfig()
        self.assertIsNotNone(config.color_scheme)
        self.assertEqual(config.theme, 'plotly_white')
        self.assertEqual(config.default_height, 800)
        self.assertEqual(config.default_width, 1200)
        self.assertEqual(config.pattern_opacity, 0.7)
        self.assertTrue(config.show_grid)

    def test_create_candlestick_chart_plotly(self):
        """Test Plotly candlestick chart creation"""
        fig = self.visualizer.create_candlestick_chart(use_plotly=True)
        self.assertIsInstance(fig, go.Figure)
        self.assertTrue(len(fig.data) >= 1)
        self.assertEqual(fig.data[0].type, 'candlestick')

    def test_create_candlestick_chart_matplotlib(self):
        """Test Matplotlib candlestick chart creation"""
        fig = self.visualizer.create_candlestick_chart(use_plotly=False)
        self.assertIsNotNone(fig)

    def test_create_pattern_cluster_chart(self):
        """Test pattern cluster chart creation"""
        fig = self.visualizer.create_pattern_cluster_chart()
        self.assertIsInstance(fig, go.Figure)
        self.assertTrue(len(fig.data) >= 1)

    def test_create_multi_timeframe_chart(self):
        """Test multi-timeframe chart creation"""
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
        self.assertIsInstance(fig, go.Figure)
        self.assertTrue(len(fig.data) >= 3)

    def test_create_pattern_reliability_chart(self):
        """Test pattern reliability chart creation"""
        fig = self.visualizer.create_pattern_reliability_chart()
        self.assertIsInstance(fig, go.Figure)
        self.assertTrue(len(fig.data) >= 1)

    def test_create_interactive_dashboard(self):
        """Test interactive dashboard creation"""
        fig = self.visualizer.create_interactive_dashboard()
        self.assertIsInstance(fig, go.Figure)
        self.assertTrue(len(fig.data) >= 1)
        self.assertTrue(hasattr(fig.layout, 'updatemenus'))

    def test_detect_market_regime(self):
        """Test market regime detection"""
        regimes = self.visualizer.detect_market_regime()
        self.assertIsInstance(regimes, pd.DataFrame)
        expected_columns = ['volatility_regime', 'trend_regime', 
                          'volume_regime', 'momentum_regime', 'combined_regime']
        for col in expected_columns:
            self.assertIn(col, regimes.columns)

    def test_create_regime_visualization(self):
        """Test regime visualization creation"""
        fig = self.visualizer.create_regime_visualization()
        self.assertIsInstance(fig, go.Figure)
        self.assertTrue(len(fig.data) >= 1)

    def test_add_pattern_correlation_analysis(self):
        """Test pattern correlation analysis"""
        fig = go.Figure()
        fig = self.visualizer.add_pattern_correlation_analysis(fig)
        self.assertIsInstance(fig, go.Figure)
        self.assertTrue(len(fig.data) >= 1)

    def test_add_price_action_confirmation(self):
        """Test price action confirmation"""
        confirmation = self.visualizer.add_price_action_confirmation()
        self.assertIsInstance(confirmation, pd.DataFrame)
        self.assertTrue(all(col in confirmation.columns 
                          for col in ['RSI', 'MACD', 'Volume_MA_Ratio', 'ATR_Ratio']))

    def test_get_pattern_distribution(self):
        """Test pattern distribution calculation"""
        distribution = self.visualizer._get_pattern_distribution()
        self.assertIsInstance(distribution, pd.Series)
        self.assertTrue(len(distribution) > 0)

    def test_get_volume_colors(self):
        """Test volume color generation"""
        colors = self.visualizer._get_volume_colors()
        self.assertIsInstance(colors, list)
        self.assertEqual(len(colors), len(self.test_data))
        self.assertTrue(all(c in [self.config.color_scheme['volume_up'], 
                                self.config.color_scheme['volume_down']] 
                          for c in colors))

    def test_analyze_pattern_sequences(self):
        """Test pattern sequence analysis"""
        sequences = self.visualizer.analyze_pattern_sequences()
        self.assertIsInstance(sequences, pd.DataFrame)
        self.assertTrue(all(col in sequences.columns for col in ['up', 'down', 'total']))

    def test_overlay_custom_indicators(self):
        """Test custom indicator overlay"""
        fig = go.Figure()
        
        def simple_sma(df, period=20):
            return df['Close'].rolling(window=period).mean()
        
        indicators = {'SMA': simple_sma}
        
        fig = self.visualizer.overlay_custom_indicators(fig, indicators)
        self.assertIsInstance(fig, go.Figure)
        self.assertTrue(len(fig.data) >= 1)

    def test_calculate_rsi(self):
        """Test RSI calculation"""
        rsi = self.visualizer._calculate_rsi(self.test_data['Close'])
        self.assertIsInstance(rsi, pd.Series)
        self.assertTrue(all(0 <= x <= 100 for x in rsi.dropna()))

    def test_calculate_macd(self):
        """Test MACD calculation"""
        macd = self.visualizer._calculate_macd(self.test_data['Close'])
        self.assertIsInstance(macd, pd.Series)
        self.assertEqual(len(macd), len(self.test_data))

    def test_get_all_pattern_methods(self):
        """Test retrieving all pattern detection methods"""
        pattern_methods = self.visualizer._get_all_pattern_methods()
        self.assertIsInstance(pattern_methods, dict)
        self.assertTrue(len(pattern_methods) > 0)
        # Verify all methods start with 'detect_'
        self.assertTrue(all(inspect.isfunction(method) for method in pattern_methods.values()))

    def test_create_matplotlib_chart_with_indicators(self):
        """Test matplotlib chart creation with various indicators"""
        # Create Bollinger Bands
        df = self.test_data.copy()
        sma = df['Close'].rolling(window=20).mean()
        std = df['Close'].rolling(window=20).std()
        bollinger_upper = sma + (std * 2)
        bollinger_lower = sma - (std * 2)
        
        fig = self.visualizer._create_matplotlib_chart(
            bollinger_upper=bollinger_upper,
            bollinger_mid=sma,
            bollinger_lower=bollinger_lower,
            pivot_points=[100, 110, 120],
            title="Test Chart"
        )
        self.assertIsNotNone(fig)

    def test_cluster_patterns(self):
        """Test pattern clustering functionality"""
        clustered_data = self.visualizer._cluster_patterns(window_size=10)
        self.assertIsInstance(clustered_data, pd.DataFrame)
        self.assertIn('cluster', clustered_data.columns)
        # Test with different window sizes
        clustered_data_large = self.visualizer._cluster_patterns(window_size=30)
        self.assertEqual(len(clustered_data), len(clustered_data_large))

    def test_create_pattern_reliability_chart_with_filters(self):
        """Test pattern reliability chart with different lookback windows"""
        fig = self.visualizer.create_pattern_reliability_chart(lookback_window=50)
        self.assertIsInstance(fig, go.Figure)
        self.assertTrue(len(fig.data) >= 4)  # Should have at least 4 subplots

    def test_create_pattern_cluster_chart_empty_data(self):
        """Test pattern cluster chart creation with empty data"""
        empty_df = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
        viz = CandlestickVisualizer(empty_df, self.config)
        fig = viz.create_pattern_cluster_chart()
        self.assertIsInstance(fig, go.Figure)

    def test_create_interactive_dashboard_components(self):
        """Test all components of interactive dashboard"""
        fig = self.visualizer.create_interactive_dashboard()
        self.assertIsInstance(fig, go.Figure)
        
        # Test menu components
        self.assertTrue(hasattr(fig.layout, 'updatemenus'))
        self.assertGreaterEqual(len(fig.layout.updatemenus), 1)
        
        # Test subplot structure
        self.assertTrue(hasattr(fig.layout, 'annotations'))
        self.assertGreaterEqual(len(fig.layout.annotations), 3)

    def test_volume_profile_visualization(self):
        """Test volume profile components"""
        colors = self.visualizer._get_volume_colors()
        self.assertEqual(len(colors), len(self.test_data))
        self.assertTrue(all(c in [self.config.color_scheme['volume_up'], 
                                self.config.color_scheme['volume_down']] 
                            for c in colors))

    def test_pattern_buttons_creation(self):
        """Test pattern button creation for dashboard"""
        buttons = self.visualizer._create_pattern_buttons()
        self.assertIsInstance(buttons, list)
        self.assertGreater(len(buttons), 1)  # Should have at least "All Patterns" button
        self.assertEqual(buttons[0]['label'], 'All Patterns')

    def test_add_price_action_confirmation_indicators(self):
        """Test price action confirmation indicators"""
        confirmation = self.visualizer.add_price_action_confirmation()
        self.assertIsInstance(confirmation, pd.DataFrame)
        required_columns = ['RSI', 'MACD', 'Volume_MA_Ratio', 'ATR_Ratio']
        for col in required_columns:
            self.assertIn(col, confirmation.columns)
        # Test value ranges
        self.assertTrue(all(0 <= x <= 100 for x in confirmation['RSI'].dropna()))

    def test_pattern_significance_calculation(self):
        """Test pattern significance calculations"""
        significance = self.visualizer.add_pattern_significance(confidence_level=0.95)
        self.assertIsInstance(significance, pd.DataFrame)
        self.assertTrue('p_value' in significance.columns)
        self.assertTrue('significant' in significance.columns)
        # Test p-values are within valid range
        self.assertTrue(all(0 <= x <= 1 for x in significance['p_value'].dropna()))

    def test_analyze_pattern_sequences_lookback(self):
        """Test pattern sequence analysis with different lookback periods"""
        sequences = self.visualizer.analyze_pattern_sequences(lookback=3)
        self.assertIsInstance(sequences, pd.DataFrame)
        self.assertTrue(all(col in sequences.columns for col in ['up', 'down', 'total']))
        
        # Test with different lookback
        sequences_long = self.visualizer.analyze_pattern_sequences(lookback=5)
        self.assertEqual(sequences.columns.tolist(), sequences_long.columns.tolist())

    def test_overlay_custom_indicators_multiple(self):
        """Test overlaying multiple custom indicators"""
        fig = go.Figure()
        
        # Define multiple custom indicators
        indicators = {
            'SMA20': lambda df: df['Close'].rolling(window=20).mean(),
            'SMA50': lambda df: df['Close'].rolling(window=50).mean(),
            'Momentum': lambda df: df['Close'] - df['Close'].shift(10)
        }
        
        fig = self.visualizer.overlay_custom_indicators(fig, indicators)
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), len(indicators))

    def test_technical_indicators_calculation(self):
        """Test technical indicator calculations"""
        # Test RSI
        rsi = self.visualizer._calculate_rsi(self.test_data['Close'])
        self.assertIsInstance(rsi, pd.Series)
        self.assertTrue(all(0 <= x <= 100 for x in rsi.dropna()))
        
        # Test MACD
        macd = self.visualizer._calculate_macd(self.test_data['Close'])
        self.assertIsInstance(macd, pd.Series)
        self.assertEqual(len(macd), len(self.test_data))

    def test_market_regime_detection_periods(self):
        """Test market regime detection with different periods"""
        regimes = self.visualizer.detect_market_regime(window=10)
        self.assertIsInstance(regimes, pd.DataFrame)
        required_columns = ['volatility_regime', 'trend_regime', 
                        'volume_regime', 'momentum_regime', 'combined_regime']
        for col in required_columns:
            self.assertIn(col, regimes.columns)
        
        # Test with different window
        regimes_long = self.visualizer.detect_market_regime(window=30)
        self.assertEqual(regimes.columns.tolist(), regimes_long.columns.tolist())

if __name__ == '__main__':
    unittest.main()