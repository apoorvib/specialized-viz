import unittest
import pandas as pd
import numpy as np
from specialized_viz.candlestick.patterns import CandlestickPatterns

class TestCandlestickPatterns(unittest.TestCase):
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
        
        cls.patterns = CandlestickPatterns()

    def test_detect_doji(self):
        """Test doji pattern detection"""
        # Create specific doji pattern
        doji_data = self.test_data.copy()
        doji_data.loc['2023-01-05', 'Open'] = 100
        doji_data.loc['2023-01-05', 'Close'] = 100
        doji_data.loc['2023-01-05', 'High'] = 110
        doji_data.loc['2023-01-05', 'Low'] = 90
        
        result = self.patterns.detect_doji(doji_data)
        self.assertTrue(result['2023-01-05'])

    def test_detect_hammer(self):
        """Test hammer pattern detection"""
        # Create specific hammer pattern
        hammer_data = self.test_data.copy()
        hammer_data.loc['2023-01-05', 'Open'] = 100
        hammer_data.loc['2023-01-05', 'Close'] = 105
        hammer_data.loc['2023-01-05', 'High'] = 110
        hammer_data.loc['2023-01-05', 'Low'] = 85
        
        result = self.patterns.detect_hammer(hammer_data)
        self.assertTrue(result['2023-01-05'])

    def test_detect_engulfing(self):
        """Test engulfing pattern detection"""
        # Create bullish engulfing pattern
        engulfing_data = self.test_data.copy()
        # First day (bearish)
        engulfing_data.loc['2023-01-05', 'Open'] = 110
        engulfing_data.loc['2023-01-05', 'Close'] = 100
        # Second day (bullish engulfing)
        engulfing_data.loc['2023-01-06', 'Open'] = 95
        engulfing_data.loc['2023-01-06', 'Close'] = 115
        
        bullish, bearish = self.patterns.detect_engulfing(engulfing_data)
        self.assertTrue(bullish['2023-01-06'])

    def test_pattern_strength(self):
        """Test pattern strength calculation"""
        # Create sample pattern data
        pattern_indices = pd.Series([True, False, True, False], index=self.test_data.index[:4])
        
        strength_scores = self.patterns.calculate_pattern_strength(
            self.test_data, pattern_indices
        )
        self.assertIsInstance(strength_scores, pd.Series)
        self.assertEqual(len(strength_scores), len(self.test_data))

    def test_detect_morning_star(self):
        """Test morning star pattern detection"""
        morning_star_data = self.test_data.copy()
        # First day (bearish)
        morning_star_data.loc['2023-01-05', 'Open'] = 110
        morning_star_data.loc['2023-01-05', 'Close'] = 100
        # Second day (doji)
        morning_star_data.loc['2023-01-06', 'Open'] = 101
        morning_star_data.loc['2023-01-06', 'Close'] = 100
        # Third day (bullish)
        morning_star_data.loc['2023-01-07', 'Open'] = 102
        morning_star_data.loc['2023-01-07', 'Close'] = 112
        
        result = self.patterns.detect_morning_star(morning_star_data)
        self.assertTrue(result['2023-01-07'])

    def test_detect_evening_star(self):
        """Test evening star pattern detection"""
        evening_star_data = self.test_data.copy()
        # First day (bullish)
        evening_star_data.loc['2023-01-05', 'Open'] = 100
        evening_star_data.loc['2023-01-05', 'Close'] = 110
        # Second day (doji)
        evening_star_data.loc['2023-01-06', 'Open'] = 111
        evening_star_data.loc['2023-01-06', 'Close'] = 110
        # Third day (bearish)
        evening_star_data.loc['2023-01-07', 'Open'] = 108
        evening_star_data.loc['2023-01-07', 'Close'] = 98
        
        result = self.patterns.detect_evening_star(evening_star_data)
        self.assertTrue(result['2023-01-07'])

if __name__ == '__main__':
    unittest.main()