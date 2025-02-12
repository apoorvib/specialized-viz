import unittest
import pandas as pd
import numpy as np
from specialized_viz.candlestick.patterns import CandlestickPatterns

class TestCandlestickPatterns(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data that will be used across test methods"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        cls.test_data = pd.DataFrame({
            'Open': np.random.uniform(100, 150, 100),  # Use float instead of int
            'High': np.random.uniform(120, 170, 100),
            'Low': np.random.uniform(80, 130, 100),
            'Close': np.random.uniform(90, 160, 100),
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
        # First day - bullish
        evening_star_data.loc['2023-01-08', 'Open'] = 100.0
        evening_star_data.loc['2023-01-08', 'High'] = 110.0
        evening_star_data.loc['2023-01-08', 'Low'] = 99.0
        evening_star_data.loc['2023-01-08', 'Close'] = 110.0
        
        # Second day - doji
        evening_star_data.loc['2023-01-09', 'Open'] = 111.0
        evening_star_data.loc['2023-01-09', 'High'] = 112.0
        evening_star_data.loc['2023-01-09', 'Low'] = 110.0
        evening_star_data.loc['2023-01-09', 'Close'] = 111.1
        
        # Third day - bearish
        evening_star_data.loc['2023-01-10', 'Open'] = 108.0
        evening_star_data.loc['2023-01-10', 'High'] = 109.0
        evening_star_data.loc['2023-01-10', 'Low'] = 98.0
        evening_star_data.loc['2023-01-10', 'Close'] = 98.0
        
        result = self.patterns.detect_evening_star(evening_star_data)
        self.assertTrue(result['2023-01-10'])
    
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
        
        # Create uptrend before the shooting star
        star_data.loc['2023-01-03', 'Close'] = 95.0
        star_data.loc['2023-01-04', 'Close'] = 98.0
        
        # Create shooting star pattern
        star_data.loc['2023-01-05', 'Open'] = 100.0
        star_data.loc['2023-01-05', 'Close'] = 102.0
        star_data.loc['2023-01-05', 'High'] = 120.0
        star_data.loc['2023-01-05', 'Low'] = 98.0
        
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
        """Test rising three methods pattern detection"""
        methods_data = self.test_data.copy()
        
        print("\nDebugging Rising Three Methods:")
        # First day - strong bullish
        methods_data.loc['2023-01-05', 'Open'] = 100.0
        methods_data.loc['2023-01-05', 'Close'] = 110.0
        methods_data.loc['2023-01-05', 'High'] = 111.0
        methods_data.loc['2023-01-05', 'Low'] = 99.0
        print(f"Day 1 (Long Bullish): O:{100.0} H:{111.0} L:{99.0} C:{110.0}")
        
        # Three small bearish days within the range
        for i, day in enumerate(['2023-01-06', '2023-01-07', '2023-01-08']):
            open_price = 108.0 - i
            close_price = 107.0 - i
            high_price = 109.0 - i
            low_price = 106.0 - i
            methods_data.loc[day, 'Open'] = open_price
            methods_data.loc[day, 'Close'] = close_price
            methods_data.loc[day, 'High'] = high_price
            methods_data.loc[day, 'Low'] = low_price
            print(f"Day {i+2} (Small Bearish): O:{open_price} H:{high_price} L:{low_price} C:{close_price}")
        
        # Final bullish breakout
        methods_data.loc['2023-01-09', 'Open'] = 108.0
        methods_data.loc['2023-01-09', 'Close'] = 112.0
        methods_data.loc['2023-01-09', 'High'] = 113.0
        methods_data.loc['2023-01-09', 'Low'] = 107.0
        print(f"Day 5 (Breakout): O:{108.0} H:{113.0} L:{107.0} C:{112.0}")
        
        # Debug pattern conditions
        first_day_body = methods_data.loc['2023-01-05', 'Close'] - methods_data.loc['2023-01-05', 'Open']
        middle_days_bearish = all(methods_data.loc[day, 'Close'] < methods_data.loc[day, 'Open'] 
                                for day in ['2023-01-06', '2023-01-07', '2023-01-08'])
        middle_days_contained = all(
            methods_data.loc[day, 'High'] < methods_data.loc['2023-01-05', 'High'] and
            methods_data.loc[day, 'Low'] > methods_data.loc['2023-01-05', 'Low']
            for day in ['2023-01-06', '2023-01-07', '2023-01-08']
        )
        final_day_breakout = (methods_data.loc['2023-01-09', 'Close'] > methods_data.loc['2023-01-05', 'High'])
        
        print("\nPattern Conditions:")
        print(f"First day strong bullish (body size): {first_day_body:.2f}")
        print(f"Middle days bearish: {middle_days_bearish}")
        print(f"Middle days contained within first day: {middle_days_contained}")
        print(f"Final day breaks above first day high: {final_day_breakout}")
        
        rising_result = self.patterns.detect_rising_three_methods(methods_data)
        print(f"Pattern detected: {rising_result['2023-01-09']}")
        
        self.assertTrue(rising_result['2023-01-09'])
    
    def test_detect_abandoned_baby(self):
        """Test abandoned baby pattern detection"""
        baby_data = self.test_data.copy()
        
        # Debug prints for analysis
        print("\nDebugging Abandoned Baby Pattern:")
        print("Setting up pattern data...")
        
        # First day - bearish
        baby_data.loc['2023-01-05', 'Open'] = 110.0
        baby_data.loc['2023-01-05', 'High'] = 112.0
        baby_data.loc['2023-01-05', 'Low'] = 95.0
        baby_data.loc['2023-01-05', 'Close'] = 100.0
        print(f"Day 1 (Bearish): O:{110.0} H:{112.0} L:{95.0} C:{100.0}")
        
        # Second day - doji with gap down
        baby_data.loc['2023-01-06', 'Open'] = 92.0
        baby_data.loc['2023-01-06', 'High'] = 93.0
        baby_data.loc['2023-01-06', 'Low'] = 91.0
        baby_data.loc['2023-01-06', 'Close'] = 92.0
        print(f"Day 2 (Doji): O:{92.0} H:{93.0} L:{91.0} C:{92.0}")
        
        # Third day - bullish with gap up
        baby_data.loc['2023-01-07', 'Open'] = 96.0
        baby_data.loc['2023-01-07', 'High'] = 108.0
        baby_data.loc['2023-01-07', 'Low'] = 95.0
        baby_data.loc['2023-01-07', 'Close'] = 105.0
        print(f"Day 3 (Bullish): O:{96.0} H:{108.0} L:{95.0} C:{105.0}")
        
        bullish, bearish = self.patterns.detect_abandoned_baby(baby_data)
        
        # Debug the gap conditions
        gap_down = baby_data.loc['2023-01-06', 'High'] < baby_data.loc['2023-01-05', 'Low']
        gap_up = baby_data.loc['2023-01-07', 'Low'] > baby_data.loc['2023-01-06', 'High']
        
        print("\nGap Analysis:")
        print(f"Gap down exists: {gap_down}")
        print(f"Gap up exists: {gap_up}")
        print(f"Pattern detected: {bullish['2023-01-07']}")
        
        self.assertTrue(bullish['2023-01-07'])
        
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
        """Test upside gap three methods pattern detection"""
        gap_data = self.test_data.copy()
        
        print("\nDebugging Upside Gap Three Methods:")
        # First bullish day
        gap_data.loc['2023-01-05', 'Open'] = 100.0
        gap_data.loc['2023-01-05', 'Close'] = 105.0
        gap_data.loc['2023-01-05', 'High'] = 106.0
        gap_data.loc['2023-01-05', 'Low'] = 99.0
        print(f"Day 1 (Bullish): O:{100.0} H:{106.0} L:{99.0} C:{105.0}")
        
        # Second day gaps up
        gap_data.loc['2023-01-06', 'Open'] = 108.0
        gap_data.loc['2023-01-06', 'Close'] = 112.0
        gap_data.loc['2023-01-06', 'High'] = 113.0
        gap_data.loc['2023-01-06', 'Low'] = 107.0
        print(f"Day 2 (Gap Up): O:{108.0} H:{113.0} L:{107.0} C:{112.0}")
        
        # Third day fills gap
        gap_data.loc['2023-01-07', 'Open'] = 110.0
        gap_data.loc['2023-01-07', 'Close'] = 106.0
        gap_data.loc['2023-01-07', 'High'] = 111.0
        gap_data.loc['2023-01-07', 'Low'] = 105.0
        print(f"Day 3 (Fill): O:{110.0} H:{111.0} L:{105.0} C:{106.0}")
        
        # Debug pattern conditions
        first_day_bullish = gap_data.loc['2023-01-05', 'Close'] > gap_data.loc['2023-01-05', 'Open']
        gap_up = gap_data.loc['2023-01-06', 'Low'] > gap_data.loc['2023-01-05', 'High']
        second_day_bullish = gap_data.loc['2023-01-06', 'Close'] > gap_data.loc['2023-01-06', 'Open']
        fills_gap = (gap_data.loc['2023-01-07', 'Open'] < gap_data.loc['2023-01-06', 'Close'] and
                    gap_data.loc['2023-01-07', 'Close'] < gap_data.loc['2023-01-06', 'Open'])
        
        print("\nPattern Conditions:")
        print(f"First day bullish: {first_day_bullish}")
        print(f"Gap up present: {gap_up}")
        print(f"Second day bullish: {second_day_bullish}")
        print(f"Third day fills gap: {fills_gap}")
        
        upside_result = self.patterns.detect_upside_gap_three_methods(gap_data)
        print(f"Pattern detected: {upside_result['2023-01-07']}")
        
        self.assertTrue(upside_result['2023-01-07'])
    
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
        
        print("\nDebugging Eight New Price Lines Pattern:")
        print("Creating sequence of 8 higher highs and lows...")
        
        # Create eight consecutive higher highs and lows
        base_price = 100.0
        dates = []
        for i, day in enumerate(pd.date_range('2023-01-05', periods=8)):
            lines_data.loc[day, 'High'] = base_price + (i + 1) * 2
            lines_data.loc[day, 'Low'] = base_price + i * 2
            lines_data.loc[day, 'Open'] = base_price + i * 2
            lines_data.loc[day, 'Close'] = base_price + (i + 1) * 2
            dates.append(day)
            print(f"Day {i+1} ({day.date()}): High:{base_price + (i + 1) * 2:.1f} "
                f"Low:{base_price + i * 2:.1f} Open:{base_price + i * 2:.1f} "
                f"Close:{base_price + (i + 1) * 2:.1f}")
        
        bullish, bearish = self.patterns.detect_eight_new_price_lines(lines_data)
        
        # Verify sequence
        print("\nSequence Verification:")
        for i in range(1, len(dates)):
            print(f"Day {i+1} vs Day {i}:")
            print(f"High increasing: {lines_data.loc[dates[i], 'High'] > lines_data.loc[dates[i-1], 'High']}")
            print(f"Low increasing: {lines_data.loc[dates[i], 'Low'] > lines_data.loc[dates[i-1], 'Low']}")
        
        print(f"\nPattern detected on {dates[-1].date()}: {bullish[dates[-1]]}")
        print(f"Available dates in result: {bullish.index}")
        
        self.assertTrue(bullish[lines_data.index[-1]])
            
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
        
        print("\nDebugging Tri-Star Pattern:")
        # First doji
        tri_star_data.loc['2023-01-05', 'Open'] = 100.0
        tri_star_data.loc['2023-01-05', 'Close'] = 100.1
        tri_star_data.loc['2023-01-05', 'High'] = 101.0
        tri_star_data.loc['2023-01-05', 'Low'] = 99.0
        print(f"Doji 1: O:{100.0} H:{101.0} L:{99.0} C:{100.1}")
        
        # Second doji at lower level
        tri_star_data.loc['2023-01-06', 'Open'] = 97.0
        tri_star_data.loc['2023-01-06', 'Close'] = 97.1
        tri_star_data.loc['2023-01-06', 'High'] = 98.0
        tri_star_data.loc['2023-01-06', 'Low'] = 96.0
        print(f"Doji 2: O:{97.0} H:{98.0} L:{96.0} C:{97.1}")
        
        # Third doji at higher level
        tri_star_data.loc['2023-01-07', 'Open'] = 102.0
        tri_star_data.loc['2023-01-07', 'Close'] = 102.1
        tri_star_data.loc['2023-01-07', 'High'] = 103.0
        tri_star_data.loc['2023-01-07', 'Low'] = 101.0
        print(f"Doji 3: O:{102.0} H:{103.0} L:{101.0} C:{102.1}")
        
        # Debug each doji condition
        def is_doji(data, date):
            body = abs(data.loc[date, 'Close'] - data.loc[date, 'Open'])
            range_ = data.loc[date, 'High'] - data.loc[date, 'Low']
            return body / range_ < 0.1
        
        doji1 = is_doji(tri_star_data, '2023-01-05')
        doji2 = is_doji(tri_star_data, '2023-01-06')
        doji3 = is_doji(tri_star_data, '2023-01-07')
        gap_down = tri_star_data.loc['2023-01-06', 'High'] < tri_star_data.loc['2023-01-05', 'Low']
        gap_up = tri_star_data.loc['2023-01-07', 'Low'] > tri_star_data.loc['2023-01-06', 'High']
        
        print("\nPattern Conditions:")
        print(f"First doji valid: {doji1}")
        print(f"Second doji valid: {doji2}")
        print(f"Third doji valid: {doji3}")
        print(f"Gap down between first and second: {gap_down}")
        print(f"Gap up between second and third: {gap_up}")
        
        bullish, bearish = self.patterns.detect_tri_star(tri_star_data)
        print(f"Bullish pattern detected: {bullish['2023-01-07']}")
        
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
        
        print("\nDebugging Mat Hold Pattern:")
        
        # First day - strong bullish
        mat_data.loc['2023-01-05', 'Open'] = 100.0
        mat_data.loc['2023-01-05', 'High'] = 110.0
        mat_data.loc['2023-01-05', 'Low'] = 99.0
        mat_data.loc['2023-01-05', 'Close'] = 110.0
        print(f"Day 1 (Strong Bullish): O:{100.0} H:{110.0} L:{99.0} C:{110.0}")
        
        # Three small bearish days
        for i, day in enumerate(['2023-01-06', '2023-01-07', '2023-01-08']):
            open_price = 111.0 - i
            mat_data.loc[day, 'Open'] = open_price
            mat_data.loc[day, 'High'] = open_price + 1
            mat_data.loc[day, 'Low'] = open_price - 1
            mat_data.loc[day, 'Close'] = open_price - 0.5
            print(f"Day {i+2} (Small Bearish): O:{open_price} H:{open_price+1} L:{open_price-1} C:{open_price-0.5}")
        
        # Final bullish day
        mat_data.loc['2023-01-09', 'Open'] = 110.0
        mat_data.loc['2023-01-09', 'High'] = 115.0
        mat_data.loc['2023-01-09', 'Low'] = 109.0
        mat_data.loc['2023-01-09', 'Close'] = 115.0
        print(f"Day 5 (Final Bullish): O:{110.0} H:{115.0} L:{109.0} C:{115.0}")
        
        # Debug pattern conditions
        first_day_bullish = mat_data.loc['2023-01-05', 'Close'] > mat_data.loc['2023-01-05', 'Open']
        gap_up = mat_data.loc['2023-01-06', 'Low'] > mat_data.loc['2023-01-05', 'High']
        three_bearish = all(mat_data.loc[day, 'Close'] < mat_data.loc[day, 'Open'] 
                        for day in ['2023-01-06', '2023-01-07', '2023-01-08'])
        final_bullish = mat_data.loc['2023-01-09', 'Close'] > mat_data.loc['2023-01-09', 'Open']
        breaks_high = mat_data.loc['2023-01-09', 'Close'] > mat_data.loc['2023-01-05', 'High']
        
        print("\nPattern Conditions:")
        print(f"First day bullish: {first_day_bullish}")
        print(f"Gap up after first day: {gap_up}")
        print(f"Three bearish days: {three_bearish}")
        print(f"Final day bullish: {final_bullish}")
        print(f"Breaks above first day high: {breaks_high}")
        
        result = self.patterns.detect_mat_hold(mat_data)
        print(f"Pattern detected: {result['2023-01-09']}")
        
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
        
        print("\nDebugging Volatility Adjusted Patterns:")
        # Setup data for volatile period (10-15)
        base_price = 100.0
        
        # Create a series of prices leading up to volatile period
        for i in range(5, 10):
            data.iloc[i, data.columns.get_loc('Close')] = base_price + i
            data.iloc[i, data.columns.get_loc('Open')] = base_price + i - 0.5
            print(f"Pre-volatile day {i}: O:{base_price + i - 0.5:.1f} C:{base_price + i:.1f}")
        
        # Create volatile bullish engulfing pattern
        for i in range(10, 15):
            if i == 12:  # Create engulfing pattern
                data.iloc[i-1, data.columns.get_loc('Open')] = 115.0
                data.iloc[i-1, data.columns.get_loc('Close')] = 110.0
                data.iloc[i, data.columns.get_loc('Open')] = 109.0
                data.iloc[i, data.columns.get_loc('Close')] = 116.0
                print(f"\nEngulfing Pattern:")
                print(f"Day {i-1} (Bearish): O:115.0 C:110.0")
                print(f"Day {i} (Bullish): O:109.0 C:116.0")
            else:
                data.iloc[i, data.columns.get_loc('Close')] = base_price + i * 1.5
                data.iloc[i, data.columns.get_loc('Open')] = base_price + i * 1.5 - 1
        
        # Calculate and print volatility metrics
        window = 20
        returns = data['Close'].pct_change()
        volatility = returns.rolling(window=window).std()
        
        print("\nVolatility Analysis:")
        print(f"Average volatility: {volatility.mean():.4f}")
        print(f"Volatility during pattern: {volatility.iloc[12]:.4f}")
        
        patterns = self.patterns.detect_volatility_adjusted_patterns(data)
        
        print("\nPattern Detection Results:")
        print(f"Volatile periods found: {patterns['volatile_bullish_engulfing'].sum()}")
        print(f"Pattern detected in target range: {any(patterns['volatile_bullish_engulfing'].iloc[10:15])}")
        
        self.assertTrue(any(patterns['volatile_bullish_engulfing'].iloc[10:15]))
    
    def test_detect_pattern_combinations(self):
        """Test pattern combinations detection"""
        data = self.test_data.copy()
        
        print("\nDebugging Pattern Combinations:")
        # Create doji
        data.loc['2023-01-05', 'Open'] = 102.0
        data.loc['2023-01-05', 'Close'] = 102.1
        data.loc['2023-01-05', 'High'] = 104.0
        data.loc['2023-01-05', 'Low'] = 100.0
        print(f"Doji: O:{102.0} H:{104.0} L:{100.0} C:{102.1}")
        
        # Create bullish engulfing
        data.loc['2023-01-06', 'Open'] = 101.0
        data.loc['2023-01-06', 'Close'] = 106.0
        data.loc['2023-01-06', 'High'] = 107.0
        data.loc['2023-01-06', 'Low'] = 100.0
        print(f"Engulfing: O:{101.0} H:{107.0} L:{100.0} C:{106.0}")
        
        # Debug pattern conditions
        body_size_doji = abs(data.loc['2023-01-05', 'Close'] - data.loc['2023-01-05', 'Open'])
        total_range_doji = data.loc['2023-01-05', 'High'] - data.loc['2023-01-05', 'Low']
        is_doji = body_size_doji / total_range_doji < 0.1
        
        engulfing_body = abs(data.loc['2023-01-06', 'Close'] - data.loc['2023-01-06', 'Open'])
        is_bullish_engulfing = (data.loc['2023-01-06', 'Close'] > data.loc['2023-01-06', 'Open'] and
                            data.loc['2023-01-06', 'Open'] < data.loc['2023-01-05', 'Close'] and
                            data.loc['2023-01-06', 'Close'] > data.loc['2023-01-05', 'Open'])
        
        print("\nPattern Conditions:")
        print(f"Is Doji: {is_doji} (body/range = {body_size_doji/total_range_doji:.3f})")
        print(f"Is Bullish Engulfing: {is_bullish_engulfing}")
        
        combinations = self.patterns.detect_pattern_combinations(data)
        print(f"Pattern detected: {combinations['strong_bullish'].iloc[-1]}")
        
        self.assertTrue(combinations['strong_bullish'].iloc[-1])        
        
    def test_detect_island_reversal(self):
        """Test island reversal pattern detection"""
        island_data = self.test_data.copy()
        
        print("\nDebugging Island Reversal Pattern:")
        # Setup before island
        island_data.loc['2023-01-05', 'Open'] = 100.0
        island_data.loc['2023-01-05', 'High'] = 110.0
        island_data.loc['2023-01-05', 'Low'] = 98.0
        island_data.loc['2023-01-05', 'Close'] = 105.0
        island_data.loc['2023-01-05', 'Volume'] = 1000000
        print(f"Day 1 (Before Island): O:100.0 H:110.0 L:98.0 C:105.0 V:1000000")
        
        # Island formation (gap down)
        island_data.loc['2023-01-06', 'Open'] = 95.0
        island_data.loc['2023-01-06', 'High'] = 96.0
        island_data.loc['2023-01-06', 'Low'] = 93.0
        island_data.loc['2023-01-06', 'Close'] = 94.0
        island_data.loc['2023-01-06', 'Volume'] = 1500000  # Higher volume
        print(f"Day 2 (Island): O:95.0 H:96.0 L:93.0 C:94.0 V:1500000")
        
        # Gap up from island
        island_data.loc['2023-01-07', 'Open'] = 98.0
        island_data.loc['2023-01-07', 'High'] = 102.0
        island_data.loc['2023-01-07', 'Low'] = 97.0
        island_data.loc['2023-01-07', 'Close'] = 101.0
        island_data.loc['2023-01-07', 'Volume'] = 1200000
        print(f"Day 3 (After Island): O:98.0 H:102.0 L:97.0 C:101.0 V:1200000")
        
        # Debug pattern conditions
        gap_down = island_data.loc['2023-01-06', 'High'] < island_data.loc['2023-01-05', 'Low']
        gap_up = island_data.loc['2023-01-07', 'Low'] > island_data.loc['2023-01-06', 'High']
        volume_increase = (island_data.loc['2023-01-06', 'Volume'] > 
                        island_data.loc['2023-01-05', 'Volume'])
        island_bearish = island_data.loc['2023-01-06', 'Close'] < island_data.loc['2023-01-06', 'Open']
        final_bullish = island_data.loc['2023-01-07', 'Close'] > island_data.loc['2023-01-07', 'Open']
        
        print("\nPattern Conditions:")
        print(f"Gap down into island: {gap_down}")
        print(f"Gap up from island: {gap_up}")
        print(f"Increased volume on island: {volume_increase}")
        print(f"Island day bearish: {island_bearish}")
        print(f"Final day bullish: {final_bullish}")
        
        bullish, bearish = self.patterns.detect_island_reversal(island_data)
        print(f"Bullish pattern detected: {bullish['2023-01-07']}")
        
        self.assertTrue(bullish['2023-01-07'])

if __name__ == '__main__':
    unittest.main()