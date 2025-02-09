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
        doji_data.loc['2023-01-05', 'Close'] = 100.1  # Small body
        doji_data.loc['2023-01-05', 'High'] = 110
        doji_data.loc['2023-01-05', 'Low'] = 90
        
        result = self.patterns.detect_doji(doji_data)
        self.assertTrue(result['2023-01-05'])
        
        # Test non-doji
        doji_data.loc['2023-01-06', 'Open'] = 100
        doji_data.loc['2023-01-06', 'Close'] = 110  # Large body
        non_doji_result = self.patterns.detect_doji(doji_data)
        self.assertFalse(non_doji_result['2023-01-06'])

    def test_detect_hammer(self):
        """Test hammer pattern detection"""
        hammer_data = self.test_data.copy()
        # Create hammer pattern
        hammer_data.loc['2023-01-05', 'Open'] = 100
        hammer_data.loc['2023-01-05', 'Close'] = 105
        hammer_data.loc['2023-01-05', 'High'] = 106
        hammer_data.loc['2023-01-05', 'Low'] = 85
        
        result = self.patterns.detect_hammer(hammer_data)
        self.assertTrue(result['2023-01-05'])
        
        # Test non-hammer
        hammer_data.loc['2023-01-06', 'High'] = 120  # Long upper shadow
        hammer_data.loc['2023-01-06', 'Open'] = 100
        hammer_data.loc['2023-01-06', 'Close'] = 105
        hammer_data.loc['2023-01-06', 'Low'] = 85
        non_hammer_result = self.patterns.detect_hammer(hammer_data)
        self.assertFalse(non_hammer_result['2023-01-06'])

    def test_detect_engulfing(self):
        """Test engulfing pattern detection"""
        engulfing_data = self.test_data.copy()
        # Create bullish engulfing
        engulfing_data.loc['2023-01-05', 'Open'] = 110
        engulfing_data.loc['2023-01-05', 'Close'] = 100
        engulfing_data.loc['2023-01-06', 'Open'] = 95
        engulfing_data.loc['2023-01-06', 'Close'] = 115
        
        bullish, bearish = self.patterns.detect_engulfing(engulfing_data)
        self.assertTrue(bullish['2023-01-06'])
        self.assertFalse(bearish['2023-01-06'])
        
        # Create bearish engulfing
        engulfing_data.loc['2023-01-07', 'Open'] = 105
        engulfing_data.loc['2023-01-07', 'Close'] = 115
        engulfing_data.loc['2023-01-08', 'Open'] = 120
        engulfing_data.loc['2023-01-08', 'Close'] = 100
        
        bullish2, bearish2 = self.patterns.detect_engulfing(engulfing_data)
        self.assertFalse(bullish2['2023-01-08'])
        self.assertTrue(bearish2['2023-01-08'])

    def test_detect_morning_star(self):
        """Test morning star pattern detection"""
        morning_star_data = self.test_data.copy()
        # Create morning star pattern
        morning_star_data.loc['2023-01-05', 'Open'] = 110
        morning_star_data.loc['2023-01-05', 'Close'] = 100
        morning_star_data.loc['2023-01-06', 'Open'] = 101
        morning_star_data.loc['2023-01-06', 'Close'] = 100
        morning_star_data.loc['2023-01-07', 'Open'] = 102
        morning_star_data.loc['2023-01-07', 'Close'] = 112
        
        result = self.patterns.detect_morning_star(morning_star_data)
        self.assertTrue(result['2023-01-07'])
        
        # Test non-morning star
        morning_star_data.loc['2023-01-08', 'Open'] = 110
        morning_star_data.loc['2023-01-08', 'Close'] = 100
        morning_star_data.loc['2023-01-09', 'Open'] = 101
        morning_star_data.loc['2023-01-09', 'Close'] = 95  # Wrong direction
        morning_star_data.loc['2023-01-10', 'Open'] = 102
        morning_star_data.loc['2023-01-10', 'Close'] = 112
        
        non_ms_result = self.patterns.detect_morning_star(morning_star_data)
        self.assertFalse(non_ms_result['2023-01-10'])

    def test_detect_evening_star(self):
        """Test evening star pattern detection"""
        evening_star_data = self.test_data.copy()
        # Create evening star pattern
        evening_star_data.loc['2023-01-05', 'Open'] = 100
        evening_star_data.loc['2023-01-05', 'Close'] = 110
        evening_star_data.loc['2023-01-06', 'Open'] = 111
        evening_star_data.loc['2023-01-06', 'Close'] = 110
        evening_star_data.loc['2023-01-07', 'Open'] = 108
        evening_star_data.loc['2023-01-07', 'Close'] = 98
        
        result = self.patterns.detect_evening_star(evening_star_data)
        self.assertTrue(result['2023-01-07'])
        
        # Test non-evening star
        evening_star_data.loc['2023-01-08', 'Open'] = 100
        evening_star_data.loc['2023-01-08', 'Close'] = 110
        evening_star_data.loc['2023-01-09', 'Open'] = 111
        evening_star_data.loc['2023-01-09', 'Close'] = 115  # Wrong direction
        evening_star_data.loc['2023-01-10', 'Open'] = 108
        evening_star_data.loc['2023-01-10', 'Close'] = 98
        
        non_es_result = self.patterns.detect_evening_star(evening_star_data)
        self.assertFalse(non_es_result['2023-01-10'])

    def test_detect_three_white_soldiers(self):
        """Test three white soldiers pattern detection"""
        soldiers_data = self.test_data.copy()
        # Create three white soldiers pattern
        for i, day in enumerate(['2023-01-05', '2023-01-06', '2023-01-07']):
            soldiers_data.loc[day, 'Open'] = 100 + (i * 5)
            soldiers_data.loc[day, 'Close'] = 110 + (i * 5)
            soldiers_data.loc[day, 'High'] = 111 + (i * 5)
            soldiers_data.loc[day, 'Low'] = 99 + (i * 5)
        
        result = self.patterns.detect_three_white_soldiers(soldiers_data)
        self.assertTrue(result['2023-01-07'])
        
        # Test non-three white soldiers (different heights)
        for i, day in enumerate(['2023-01-08', '2023-01-09', '2023-01-10']):
            soldiers_data.loc[day, 'Open'] = 100 + (i * 5)
            soldiers_data.loc[day, 'Close'] = 105  # Same closing price
            soldiers_data.loc[day, 'High'] = 111 + (i * 5)
            soldiers_data.loc[day, 'Low'] = 99 + (i * 5)
        
        non_soldiers_result = self.patterns.detect_three_white_soldiers(soldiers_data)
        self.assertFalse(non_soldiers_result['2023-01-10'])

    def test_detect_three_black_crows(self):
        """Test three black crows pattern detection"""
        crows_data = self.test_data.copy()
        # Create three black crows pattern
        for i, day in enumerate(['2023-01-05', '2023-01-06', '2023-01-07']):
            crows_data.loc[day, 'Open'] = 110 - (i * 5)
            crows_data.loc[day, 'Close'] = 100 - (i * 5)
            crows_data.loc[day, 'High'] = 111 - (i * 5)
            crows_data.loc[day, 'Low'] = 99 - (i * 5)
        
        result = self.patterns.detect_three_black_crows(crows_data)
        self.assertTrue(result['2023-01-07'])
        
        # Test non-three black crows (one bullish day)
        crows_data.loc['2023-01-08', 'Open'] = 95
        crows_data.loc['2023-01-08', 'Close'] = 85
        crows_data.loc['2023-01-09', 'Open'] = 90
        crows_data.loc['2023-01-09', 'Close'] = 95  # Bullish day
        crows_data.loc['2023-01-10', 'Open'] = 85
        crows_data.loc['2023-01-10', 'Close'] = 75
        
        non_crows_result = self.patterns.detect_three_black_crows(crows_data)
        self.assertFalse(non_crows_result['2023-01-10'])

    def test_detect_shooting_star(self):
        """Test shooting star pattern detection"""
        star_data = self.test_data.copy()
        # Create shooting star pattern
        star_data.loc['2023-01-05', 'Open'] = 100
        star_data.loc['2023-01-05', 'Close'] = 102
        star_data.loc['2023-01-05', 'High'] = 120
        star_data.loc['2023-01-05', 'Low'] = 98
        
        result = self.patterns.detect_shooting_star(star_data)
        self.assertTrue(result['2023-01-05'])
        
        # Test non-shooting star (long lower shadow)
        star_data.loc['2023-01-06', 'Open'] = 100
        star_data.loc['2023-01-06', 'Close'] = 102
        star_data.loc['2023-01-06', 'High'] = 110
        star_data.loc['2023-01-06', 'Low'] = 80  # Long lower shadow
        
        non_star_result = self.patterns.detect_shooting_star(star_data)
        self.assertFalse(non_star_result['2023-01-06'])

    def test_detect_harami(self):
        """Test harami pattern detection"""
        harami_data = self.test_data.copy()
        # Create bullish harami pattern
        harami_data.loc['2023-01-05', 'Open'] = 110
        harami_data.loc['2023-01-05', 'Close'] = 90
        harami_data.loc['2023-01-06', 'Open'] = 95
        harami_data.loc['2023-01-06', 'Close'] = 100
        
        bullish, bearish = self.patterns.detect_harami(harami_data)
        self.assertTrue(bullish['2023-01-06'])
        
        # Create bearish harami pattern
        harami_data.loc['2023-01-07', 'Open'] = 90
        harami_data.loc['2023-01-07', 'Close'] = 110
        harami_data.loc['2023-01-08', 'Open'] = 105
        harami_data.loc['2023-01-08', 'Close'] = 100
        
        bullish2, bearish2 = self.patterns.detect_harami(harami_data)
        self.assertTrue(bearish2['2023-01-08'])

    def test_detect_rising_falling_three_methods(self):
        """Test rising and falling three methods pattern detection"""
        methods_data = self.test_data.copy()
        
        # Create rising three methods pattern
        methods_data.loc['2023-01-05', 'Open'] = 100
        methods_data.loc['2023-01-05', 'Close'] = 120
        for i, day in enumerate(['2023-01-06', '2023-01-07', '2023-01-08']):
            methods_data.loc[day, 'Open'] = 115 - i
            methods_data.loc[day, 'Close'] = 110 - i
        methods_data.loc['2023-01-09', 'Open'] = 115
        methods_data.loc['2023-01-09', 'Close'] = 125
        
        rising_result = self.patterns.detect_rising_three_methods(methods_data)
        self.assertTrue(rising_result['2023-01-09'])
        
        # Create falling three methods pattern
        methods_data.loc['2023-01-10', 'Open'] = 120
        methods_data.loc['2023-01-10', 'Close'] = 100
        for i, day in enumerate(['2023-01-11', '2023-01-12', '2023-01-13']):
            methods_data.loc[day, 'Open'] = 105 + i
            methods_data.loc[day, 'Close'] = 110 + i
        methods_data.loc['2023-01-14', 'Open'] = 105
        methods_data.loc['2023-01-14', 'Close'] = 95
        
        falling_result = self.patterns.detect_falling_three_methods(methods_data)
        self.assertTrue(falling_result['2023-01-14'])

    def test_detect_abandoned_baby(self):
        """Test abandoned baby pattern detection"""
        baby_data = self.test_data.copy()
        # Create bullish abandoned baby
        baby_data.loc['2023-01-05', 'Open'] = 110
        baby_data.loc['2023-01-05', 'Close'] = 100
        baby_data.loc['2023-01-06', 'Open'] = 95
        baby_data.loc['2023-01-06', 'Close'] = 95
        baby_data.loc['2023-01-07', 'Open'] = 102
        baby_data.loc['2023-01-07', 'Close'] = 112
        
        bullish, bearish = self.patterns.detect_abandoned_baby(baby_data)
        self.assertTrue(bullish['2023-01-07'])
        
        # Create bearish abandoned baby
        baby_data.loc['2023-01-08', 'Open'] = 100
        baby_data.loc['2023-01-08', 'Close'] = 110
        baby_data.loc['2023-01-09', 'Open'] = 115
        baby_data.loc['2023-01-09', 'Close'] = 115
        baby_data.loc['2023-01-10', 'Open'] = 108
        baby_data.loc['2023-01-10', 'Close'] = 98
        
        bullish2, bearish2 = self.patterns.detect_abandoned_baby(baby_data)
        self.assertTrue(bearish2['2023-01-10'])

    def test_detect_unique_three_river_bottom(self):
        """Test unique three river bottom pattern detection"""
        river_data = self.test_data.copy()
        # Create three river bottom pattern
        river_data.loc['2023-01-05', 'Open'] = 110
        river_data.loc['2023-01-05', 'Close'] = 90
        river_data.loc['2023-01-06', 'Open'] = 95
        river_data.loc['2023-01-06', 'Close'] = 92
        river_data.loc['2023-01-06', 'Low'] = 85
        river_data.loc['2023-01-07', 'Open'] = 94
        river_data.loc['2023-01-07', 'Close'] = 95
        river_data.loc['2023-01-07', 'Low'] = 90
        
        result = self.patterns.detect_unique_three_river_bottom(river_data)
        self.assertTrue(result['2023-01-07'])
        
        # Test non-three river bottom (wrong sequence)
        river_data.loc['2023-01-08', 'Open'] = 110
        river_data.loc['2023-01-08', 'Close'] = 90
        river_data.loc['2023-01-09', 'Open'] = 95
        river_data.loc['2023-01-09', 'Close'] = 100  # Wrong direction
        river_data.loc['2023-01-09', 'Low'] = 85
        river_data.loc['2023-01-10', 'Open'] = 94
        river_data.loc['2023-01-10', 'Close'] = 95
        river_data.loc['2023-01-10', 'Low'] = 90
        
        non_river_result = self.patterns.detect_unique_three_river_bottom(river_data)
        self.assertFalse(non_river_result['2023-01-10'])

    def test_detect_concealing_baby_swallow(self):
        """Test concealing baby swallow pattern detection"""
        swallow_data = self.test_data.copy()
        # Create concealing baby swallow pattern
        for day in ['2023-01-05', '2023-01-06']:
            swallow_data.loc[day, 'Open'] = 110
            swallow_data.loc[day, 'Close'] = 100
            swallow_data.loc[day, 'High'] = 110
            swallow_data.loc[day, 'Low'] = 100
        swallow_data.loc['2023-01-07', 'Open'] = 105
        swallow_data.loc['2023-01-07', 'Close'] = 95
        swallow_data.loc['2023-01-07', 'Low'] = 90
        swallow_data.loc['2023-01-08', 'Open'] = 98
        swallow_data.loc['2023-01-08', 'Close'] = 108
        
        result = self.patterns.detect_concealing_baby_swallow(swallow_data)
        self.assertTrue(result['2023-01-08'])
        
        # Test non-concealing baby swallow (wrong sequence)
        for day in ['2023-01-09', '2023-01-10']:
            swallow_data.loc[day, 'Open'] = 110
            swallow_data.loc[day, 'Close'] = 100
            swallow_data.loc[day, 'High'] = 110
            swallow_data.loc[day, 'Low'] = 100
        swallow_data.loc['2023-01-11', 'Open'] = 105
        swallow_data.loc['2023-01-11', 'Close'] = 115  # Wrong direction
        swallow_data.loc['2023-01-11', 'Low'] = 90
        swallow_data.loc['2023-01-12', 'Open'] = 98
        swallow_data.loc['2023-01-12', 'Close'] = 108
        
        non_swallow_result = self.patterns.detect_concealing_baby_swallow(swallow_data)
        self.assertFalse(non_swallow_result['2023-01-12'])

    def test_detect_stick_sandwich(self):
        """Test stick sandwich pattern detection"""
        sandwich_data = self.test_data.copy()
        # Create stick sandwich pattern
        sandwich_data.loc['2023-01-05', 'Open'] = 110
        sandwich_data.loc['2023-01-05', 'Close'] = 100
        sandwich_data.loc['2023-01-06', 'Open'] = 98
        sandwich_data.loc['2023-01-06', 'Close'] = 108
        sandwich_data.loc['2023-01-07', 'Open'] = 110
        sandwich_data.loc['2023-01-07', 'Close'] = 100
        
        result = self.patterns.detect_stick_sandwich(sandwich_data)
        self.assertTrue(result['2023-01-07'])
        
        # Test non-stick sandwich (different closing prices)
        sandwich_data.loc['2023-01-08', 'Open'] = 110
        sandwich_data.loc['2023-01-08', 'Close'] = 100
        sandwich_data.loc['2023-01-09', 'Open'] = 98
        sandwich_data.loc['2023-01-09', 'Close'] = 108
        sandwich_data.loc['2023-01-10', 'Open'] = 110
        sandwich_data.loc['2023-01-10', 'Close'] = 95  # Different closing price
        
        non_sandwich_result = self.patterns.detect_stick_sandwich(sandwich_data)
        self.assertFalse(non_sandwich_result['2023-01-10'])

    def test_detect_upside_downside_gap_three_methods(self):
        """Test upside and downside gap three methods patterns"""
        gap_data = self.test_data.copy()
        
        # Test upside gap three methods
        gap_data.loc['2023-01-05', 'Open'] = 100
        gap_data.loc['2023-01-05', 'Close'] = 110
        gap_data.loc['2023-01-06', 'Open'] = 115
        gap_data.loc['2023-01-06', 'Close'] = 120
        gap_data.loc['2023-01-07', 'Open'] = 118
        gap_data.loc['2023-01-07', 'Close'] = 114
        
        upside_result = self.patterns.detect_upside_gap_three_methods(gap_data)
        self.assertTrue(upside_result['2023-01-07'])
        
        # Test downside gap three methods
        gap_data.loc['2023-01-08', 'Open'] = 110
        gap_data.loc['2023-01-08', 'Close'] = 100
        gap_data.loc['2023-01-09', 'Open'] = 95
        gap_data.loc['2023-01-09', 'Close'] = 90
        gap_data.loc['2023-01-10', 'Open'] = 93
        gap_data.loc['2023-01-10', 'Close'] = 97
        
        downside_result = self.patterns.detect_downside_gap_three_methods(gap_data)
        self.assertTrue(downside_result['2023-01-10'])

    def test_detect_two_rabbits(self):
        """Test two rabbits pattern detection"""
        rabbits_data = self.test_data.copy()
        
        # Test bullish two rabbits
        rabbits_data.loc['2023-01-05', 'Open'] = 100
        rabbits_data.loc['2023-01-05', 'Close'] = 105
        rabbits_data.loc['2023-01-05', 'Low'] = 90
        rabbits_data.loc['2023-01-06', 'Open'] = 103
        rabbits_data.loc['2023-01-06', 'Close'] = 108
        rabbits_data.loc['2023-01-06', 'Low'] = 90
        
        bullish, bearish = self.patterns.detect_two_rabbits(rabbits_data)
        self.assertTrue(bullish['2023-01-06'])
        
        # Test bearish two rabbits
        rabbits_data.loc['2023-01-07', 'Open'] = 110
        rabbits_data.loc['2023-01-07', 'Close'] = 105
        rabbits_data.loc['2023-01-07', 'High'] = 120
        rabbits_data.loc['2023-01-08', 'Open'] = 107
        rabbits_data.loc['2023-01-08', 'Close'] = 102
        rabbits_data.loc['2023-01-08', 'High'] = 120
        
        bullish2, bearish2 = self.patterns.detect_two_rabbits(rabbits_data)
        self.assertTrue(bearish2['2023-01-08'])

    def test_detect_eight_new_price_lines(self):
        """Test eight new price lines pattern detection"""
        lines_data = self.test_data.copy()
        
        # Test bullish eight new price lines
        base_price = 100
        for i, day in enumerate(pd.date_range('2023-01-05', periods=8)):
            lines_data.loc[day, 'High'] = base_price + ((i + 1) * 2)
            lines_data.loc[day, 'Low'] = base_price + (i * 2)
            lines_data.loc[day, 'Open'] = base_price + (i * 2)
            lines_data.loc[day, 'Close'] = base_price + ((i + 1) * 2)
        
        bullish, bearish = self.patterns.detect_eight_new_price_lines(lines_data)
        self.assertTrue(bullish[lines_data.index[-1]])
        
        # Test bearish eight new price lines
        base_price = 120
        for i, day in enumerate(pd.date_range('2023-01-13', periods=8)):
            lines_data.loc[day, 'High'] = base_price - (i * 2)
            lines_data.loc[day, 'Low'] = base_price - ((i + 1) * 2)
            lines_data.loc[day, 'Open'] = base_price - (i * 2)
            lines_data.loc[day, 'Close'] = base_price - ((i + 1) * 2)
        
        bullish2, bearish2 = self.patterns.detect_eight_new_price_lines(lines_data)
        self.assertTrue(bearish2[lines_data.index[-1]])

    def test_detect_three_stars_south(self):
        """Test three stars in the south pattern detection"""
        stars_data = self.test_data.copy()
        
        # Create three stars in the south pattern
        stars_data.loc['2023-01-05', 'Open'] = 110
        stars_data.loc['2023-01-05', 'Close'] = 90
        stars_data.loc['2023-01-05', 'Low'] = 85
        
        stars_data.loc['2023-01-06', 'Open'] = 95
        stars_data.loc['2023-01-06', 'Close'] = 85
        stars_data.loc['2023-01-06', 'Low'] = 87
        
        stars_data.loc['2023-01-07', 'Open'] = 88
        stars_data.loc['2023-01-07', 'Close'] = 85
        stars_data.loc['2023-01-07', 'Low'] = 87
        
        result = self.patterns.detect_three_stars_south(stars_data)
        self.assertTrue(result['2023-01-07'])

    def test_detect_tri_star(self):
        """Test tri-star pattern detection"""
        tri_star_data = self.test_data.copy()
        
        # Create bullish tri-star pattern
        for day in ['2023-01-05', '2023-01-06', '2023-01-07']:
            tri_star_data.loc[day, 'Open'] = 100
            tri_star_data.loc[day, 'Close'] = 101
            tri_star_data.loc[day, 'High'] = 102
            tri_star_data.loc[day, 'Low'] = 99
        
        # Add gaps for tri-star pattern
        tri_star_data.loc['2023-01-06', 'High'] = 98  # Gap down
        tri_star_data.loc['2023-01-06', 'Low'] = 95
        tri_star_data.loc['2023-01-07', 'Low'] = 103  # Gap up
        
        bullish, bearish = self.patterns.detect_tri_star(tri_star_data)
        self.assertTrue(bullish['2023-01-07'])

    def test_detect_ladder_bottom(self):
        """Test ladder bottom pattern detection"""
        ladder_data = self.test_data.copy()
        
        # Create ladder bottom pattern
        for i, day in enumerate(['2023-01-05', '2023-01-06', '2023-01-07']):
            ladder_data.loc[day, 'Open'] = 110 - i
            ladder_data.loc[day, 'Close'] = 100 - i
            ladder_data.loc[day, 'Low'] = 95 + i
            
        ladder_data.loc['2023-01-08', 'Open'] = 98
        ladder_data.loc['2023-01-08', 'Close'] = 108
        
        result = self.patterns.detect_ladder_bottom(ladder_data)
        self.assertTrue(result['2023-01-08'])

    def test_detect_mat_hold(self):
        """Test mat hold pattern detection"""
        mat_data = self.test_data.copy()
        
        # Create mat hold pattern
        mat_data.loc['2023-01-05', 'Open'] = 100
        mat_data.loc['2023-01-05', 'Close'] = 110
        mat_data.loc['2023-01-05', 'High'] = 111
        
        # Gap up and three small bearish days
        for i, day in enumerate(['2023-01-06', '2023-01-07', '2023-01-08']):
            mat_data.loc[day, 'Open'] = 115 + i
            mat_data.loc[day, 'Close'] = 113 + i
            
        # Final bullish day
        mat_data.loc['2023-01-09', 'Open'] = 115
        mat_data.loc['2023-01-09', 'Close'] = 125
        
        result = self.patterns.detect_mat_hold(mat_data)
        self.assertTrue(result['2023-01-09'])

    def test_detect_matching_low(self):
        """Test matching low pattern detection"""
        match_data = self.test_data.copy()
        
        # Create matching low pattern
        match_data.loc['2023-01-05', 'Open'] = 110
        match_data.loc['2023-01-05', 'Close'] = 100
        match_data.loc['2023-01-06', 'Open'] = 110
        match_data.loc['2023-01-06', 'Close'] = 100
        
        result = self.patterns.detect_matching_low(match_data)
        self.assertTrue(result['2023-01-06'])

    def test_detect_kicking_pattern(self):
        """Test kicking pattern detection"""
        kicking_data = self.test_data.copy()
        
        # Create bullish kicking pattern
        kicking_data.loc['2023-01-05', 'Open'] = 110
        kicking_data.loc['2023-01-05', 'Close'] = 100
        kicking_data.loc['2023-01-05', 'High'] = 110
        kicking_data.loc['2023-01-05', 'Low'] = 100
        
        kicking_data.loc['2023-01-06', 'Open'] = 115
        kicking_data.loc['2023-01-06', 'Close'] = 125
        kicking_data.loc['2023-01-06', 'High'] = 125
        kicking_data.loc['2023-01-06', 'Low'] = 115
        
        bullish, bearish = self.patterns.detect_kicking_pattern(kicking_data)
        self.assertTrue(bullish['2023-01-06'])

    def test_breakout_patterns(self):
        """Test breakout patterns detection"""
        patterns = self.patterns.detect_breakout_patterns(self.test_data)
        self.assertIsInstance(patterns, pd.DataFrame)
        self.assertIn('bullish_bb', patterns.columns)
        self.assertIn('bearish_bb', patterns.columns)
        self.assertIn('bullish_channel', patterns.columns)
        self.assertIn('bearish_channel', patterns.columns)
        self.assertIn('volume_confirmed', patterns.columns)

    def test_harmonic_patterns(self):
        """Test harmonic patterns detection"""
        patterns = self.patterns.detect_harmonic_patterns(self.test_data)
        self.assertIsInstance(patterns, pd.DataFrame)
        self.assertIn('gartley', patterns.columns)
        self.assertIn('butterfly', patterns.columns)
        self.assertIn('bat', patterns.columns)

    def test_island_reversal(self):
        """Test island reversal pattern detection"""
        island_data = self.test_data.copy()
        
        # Create bullish island reversal
        island_data.loc['2023-01-05', 'High'] = 110
        island_data.loc['2023-01-05', 'Low'] = 100
        island_data.loc['2023-01-06', 'High'] = 95
        island_data.loc['2023-01-06', 'Low'] = 90
        island_data.loc['2023-01-07', 'High'] = 105
        island_data.loc['2023-01-07', 'Low'] = 100
        
        bullish, bearish = self.patterns.detect_island_reversal(island_data)
        self.assertTrue(bullish['2023-01-07'])

    def test_detect_thrust_pattern(self):
        """Test thrust pattern detection"""
        thrust_data = self.test_data.copy()
        
        # Create bullish thrust pattern
        thrust_data.loc['2023-01-05', 'Open'] = 100
        thrust_data.loc['2023-01-05', 'Close'] = 110
        thrust_data.loc['2023-01-06', 'Open'] = 109
        thrust_data.loc['2023-01-06', 'Close'] = 107
        thrust_data.loc['2023-01-07', 'Open'] = 108
        thrust_data.loc['2023-01-07', 'Close'] = 115
        
        bullish, bearish = self.patterns.detect_thrust_pattern(thrust_data)
        self.assertTrue(bullish['2023-01-07'])
        
        # Create bearish thrust pattern
        thrust_data.loc['2023-01-08', 'Open'] = 110
        thrust_data.loc['2023-01-08', 'Close'] = 100
        thrust_data.loc['2023-01-09', 'Open'] = 101
        thrust_data.loc['2023-01-09', 'Close'] = 103
        thrust_data.loc['2023-01-10', 'Open'] = 102
        thrust_data.loc['2023-01-10', 'Close'] = 95
        
        bullish2, bearish2 = self.patterns.detect_thrust_pattern(thrust_data)
        self.assertTrue(bearish2['2023-01-10'])

    def test_detect_gapping_side_by_side_white_lines(self):
        """Test gapping side-by-side white lines pattern detection"""
        lines_data = self.test_data.copy()
        
        # Create side-by-side white lines pattern
        lines_data.loc['2023-01-05', 'Open'] = 100
        lines_data.loc['2023-01-05', 'Close'] = 110
        lines_data.loc['2023-01-05', 'High'] = 111
        
        lines_data.loc['2023-01-06', 'Open'] = 115
        lines_data.loc['2023-01-06', 'Close'] = 120
        lines_data.loc['2023-01-06', 'Low'] = 114
        
        lines_data.loc['2023-01-07', 'Open'] = 115
        lines_data.loc['2023-01-07', 'Close'] = 120
        
        result = self.patterns.detect_gapping_side_by_side_white_lines(lines_data)
        self.assertTrue(result['2023-01-07'])

        # Test non-matching pattern
        lines_data.loc['2023-01-08', 'Open'] = 100
        lines_data.loc['2023-01-08', 'Close'] = 110
        lines_data.loc['2023-01-09', 'Open'] = 115
        lines_data.loc['2023-01-09', 'Close'] = 120
        lines_data.loc['2023-01-10', 'Open'] = 125  # Different open price
        lines_data.loc['2023-01-10', 'Close'] = 130
        
        non_pattern_result = self.patterns.detect_gapping_side_by_side_white_lines(lines_data)
        self.assertFalse(non_pattern_result['2023-01-10'])

    def test_multi_timeframe_patterns(self):
        """Test multi-timeframe pattern detection"""
        # Create sample data for different timeframes
        daily_data = self.test_data.copy()
        weekly_data = self.test_data.resample('W').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last'
        })
        monthly_data = self.test_data.resample('M').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last'
        })
        
        # Create engulfing patterns in all timeframes
        for df in [daily_data, weekly_data, monthly_data]:
            df.loc[df.index[1], 'Open'] = 110
            df.loc[df.index[1], 'Close'] = 100
            df.loc[df.index[2], 'Open'] = 95
            df.loc[df.index[2], 'Close'] = 115
        
        result = self.patterns.detect_multi_timeframe_patterns(
            daily_data, weekly_data, monthly_data
        )
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue('strong_bullish_mtf' in result.columns)
        self.assertTrue('strong_bearish_mtf' in result.columns)

    def test_detect_pattern_reliability(self):
        """Test pattern reliability calculation with multiple patterns"""
        # Test with hammer pattern
        reliability = self.patterns.detect_pattern_reliability(
            self.test_data,
            self.patterns.detect_hammer,
            lookback_window=20,
            forward_window=5
        )
        self.assertIsInstance(reliability, pd.DataFrame)
        self.assertTrue(all(col in reliability.columns 
                        for col in ['success_rate', 'avg_return', 'risk_reward']))
        
        # Test with engulfing pattern
        reliability_eng = self.patterns.detect_pattern_reliability(
            self.test_data,
            lambda x: self.patterns.detect_engulfing(x)[0],  # Use bullish engulfing
            lookback_window=20,
            forward_window=5
        )
        self.assertIsInstance(reliability_eng, pd.DataFrame)

    def test_calculate_pattern_strength_with_different_patterns(self):
        """Test pattern strength calculation with various patterns"""
        # Test with doji pattern
        doji_pattern = self.patterns.detect_doji(self.test_data)
        strength_doji = self.patterns.calculate_pattern_strength(
            self.test_data, doji_pattern
        )
        self.assertIsInstance(strength_doji, pd.Series)
        
        # Test with hammer pattern
        hammer_pattern = self.patterns.detect_hammer(self.test_data)
        strength_hammer = self.patterns.calculate_pattern_strength(
            self.test_data, hammer_pattern
        )
        self.assertIsInstance(strength_hammer, pd.Series)
        
        # Verify strength values are between 0 and some reasonable upper limit
        self.assertTrue(all(0 <= x <= 10 for x in strength_hammer.dropna()))

    def test_detect_three_line_strike(self):
        """Test three-line strike pattern detection"""
        strike_data = self.test_data.copy()
        
        # Create bullish three-line strike
        # Three declining bearish candles
        for i, day in enumerate(['2023-01-05', '2023-01-06', '2023-01-07']):
            strike_data.loc[day, 'Open'] = 110 - (i * 5)
            strike_data.loc[day, 'Close'] = 100 - (i * 5)
        # Fourth candle reversal
        strike_data.loc['2023-01-08', 'Open'] = 85
        strike_data.loc['2023-01-08', 'Close'] = 115
        
        bullish, bearish = self.patterns.detect_three_line_strike(strike_data)
        self.assertTrue(bullish['2023-01-08'])
        
        # Create bearish three-line strike
        # Three rising bullish candles
        for i, day in enumerate(['2023-01-09', '2023-01-10', '2023-01-11']):
            strike_data.loc[day, 'Open'] = 100 + (i * 5)
            strike_data.loc[day, 'Close'] = 110 + (i * 5)
        # Fourth candle reversal
        strike_data.loc['2023-01-12', 'Open'] = 125
        strike_data.loc['2023-01-12', 'Close'] = 95
        
        bullish2, bearish2 = self.patterns.detect_three_line_strike(strike_data)
        self.assertTrue(bearish2['2023-01-12'])

    def test_calculate_atr(self):
        """Test Average True Range calculation"""
        atr = self.patterns._calculate_atr(self.test_data, window=14)
        self.assertIsInstance(atr, pd.Series)
        self.assertEqual(len(atr), len(self.test_data))
        self.assertTrue(all(x >= 0 for x in atr.dropna()))
    
    def test_detect_breakout_patterns(self):
        """Test breakout patterns detection with volatility adjustment"""
        data = self.test_data.copy()
        
        # Create a breakout setup
        base_price = 100
        for i in range(20):
            data.iloc[i, data.columns.get_loc('Close')] = base_price + i
            data.iloc[i, data.columns.get_loc('High')] = base_price + i + 1
            data.iloc[i, data.columns.get_loc('Low')] = base_price + i - 1
            data.iloc[i, data.columns.get_loc('Volume')] = 1000000

        # Create a volume spike and price breakout
        data.iloc[21, data.columns.get_loc('Close')] = base_price + 30
        data.iloc[21, data.columns.get_loc('Volume')] = 3000000

        breakouts = self.patterns.detect_breakout_patterns(data)
        
        self.assertIsInstance(breakouts, pd.DataFrame)
        self.assertTrue('bullish_bb' in breakouts.columns)
        self.assertTrue('bearish_bb' in breakouts.columns)
        self.assertTrue('bullish_channel' in breakouts.columns)
        self.assertTrue('bearish_channel' in breakouts.columns)
        self.assertTrue('volume_confirmed' in breakouts.columns)
        
        # Test volume confirmation
        self.assertTrue(breakouts['volume_confirmed'].iloc[21])

    def test_detect_harmonic_patterns(self):
        """Test harmonic patterns detection"""
        data = self.test_data.copy()
        
        # Create a Gartley pattern setup
        points = [100, 150, 120, 140, 110]  # Prices that form a Gartley pattern
        for i, price in enumerate(points):
            data.iloc[i, data.columns.get_loc('High')] = price
            data.iloc[i, data.columns.get_loc('Low')] = price - 10
        
        patterns = self.patterns.detect_harmonic_patterns(data)
        
        self.assertIsInstance(patterns, pd.DataFrame)
        self.assertTrue('gartley' in patterns.columns)
        self.assertTrue('butterfly' in patterns.columns)
        self.assertTrue('bat' in patterns.columns)

    def test_detect_volatility_adjusted_patterns(self):
        """Test volatility adjusted patterns detection"""
        data = self.test_data.copy()
        
        # Create a high volatility period
        data.iloc[10:15, data.columns.get_loc('Close')] *= 1.5
        
        # Create a low volatility period
        data.iloc[20:25, data.columns.get_loc('Close')] *= 0.9
        
        patterns = self.patterns.detect_volatility_adjusted_patterns(data)
        
        self.assertIsInstance(patterns, pd.DataFrame)
        self.assertTrue('volatile_bullish_engulfing' in patterns.columns)
        self.assertTrue('low_vol_breakout' in patterns.columns)
        
        # Test pattern detection in both high and low volatility periods
        self.assertTrue(any(patterns['volatile_bullish_engulfing'].iloc[10:15]))
        self.assertTrue(any(patterns['low_vol_breakout'].iloc[20:25]))

    def test_detect_pattern_combinations(self):
        """Test pattern combinations detection"""
        data = self.test_data.copy()
        
        # Create a doji followed by bullish engulfing
        data.loc['2023-01-05', 'Open'] = 100
        data.loc['2023-01-05', 'Close'] = 100.1
        data.loc['2023-01-05', 'High'] = 105
        data.loc['2023-01-05', 'Low'] = 95
        
        data.loc['2023-01-06', 'Open'] = 95
        data.loc['2023-01-06', 'Close'] = 110
        
        combinations = self.patterns.detect_pattern_combinations(data)
        
        self.assertIsInstance(combinations, pd.DataFrame)
        self.assertTrue('strong_bullish' in combinations.columns)
        self.assertTrue('strong_bearish' in combinations.columns)
        
        # Test bullish combination detection
        self.assertTrue(combinations['strong_bullish'].iloc[-1])

if __name__ == '__main__':
    unittest.main()