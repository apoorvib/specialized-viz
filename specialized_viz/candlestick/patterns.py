import numpy as np
import pandas as pd

class CandlestickPatterns:
    @staticmethod
    def detect_doji(df, threshold=0.1):
        """Existing doji detection code"""
        body = abs(df['Close'] - df['Open'])
        total_length = df['High'] - df['Low']
        return (body / total_length) < threshold

    @staticmethod
    def detect_hammer(df, body_ratio=0.3, shadow_ratio=2.0):
        """Existing hammer detection code"""
        body = abs(df['Close'] - df['Open'])
        total_length = df['High'] - df['Low']
        upper_shadow = df['High'] - df[['Open', 'Close']].max(axis=1)
        lower_shadow = df[['Open', 'Close']].min(axis=1) - df['Low']
        
        return (
            (body / total_length < body_ratio) &
            (lower_shadow / body > shadow_ratio) &
            (upper_shadow < body)
        )

    @staticmethod
    def detect_engulfing(df):
        """
        Detect bullish and bearish engulfing patterns
        
        Returns:
            tuple: (bullish_engulfing, bearish_engulfing) boolean Series
        """
        prev_body = abs(df['Close'].shift(1) - df['Open'].shift(1))
        curr_body = abs(df['Close'] - df['Open'])
        
        bullish = (
            (df['Open'].shift(1) > df['Close'].shift(1)) &  # Previous red candle
            (df['Close'] > df['Open']) &  # Current green candle
            (df['Open'] < df['Close'].shift(1)) &  # Opens below prev close
            (df['Close'] > df['Open'].shift(1)) &  # Closes above prev open
            (curr_body > prev_body)  # Current body larger than previous
        )
        
        bearish = (
            (df['Close'].shift(1) > df['Open'].shift(1)) &  # Previous green candle
            (df['Open'] > df['Close']) &  # Current red candle
            (df['Open'] > df['Close'].shift(1)) &  # Opens above prev close
            (df['Close'] < df['Open'].shift(1)) &  # Closes below prev open
            (curr_body > prev_body)  # Current body larger than previous
        )
        
        return bullish, bearish

    @staticmethod
    def detect_morning_star(df, doji_threshold=0.1):
        """
        Detect morning star pattern (including doji morning star)
        """
        return (
            (df['Close'].shift(2) < df['Open'].shift(2)) &  # First day: bearish
            (abs(df['Close'].shift(1) - df['Open'].shift(1)) < 
             (df['High'].shift(1) - df['Low'].shift(1)) * doji_threshold) &  # Second day: doji
            (df['Close'] > df['Open']) &  # Third day: bullish
            (df['Close'] > (df['Open'].shift(2) + df['Close'].shift(2)) / 2)  # Closes above first day midpoint
        )

    @staticmethod
    def detect_evening_star(df, doji_threshold=0.1):
        """
        Detect evening star pattern (including doji evening star)
        """
        return (
            (df['Close'].shift(2) > df['Open'].shift(2)) &  # First day: bullish
            (abs(df['Close'].shift(1) - df['Open'].shift(1)) < 
             (df['High'].shift(1) - df['Low'].shift(1)) * doji_threshold) &  # Second day: doji
            (df['Close'] < df['Open']) &  # Third day: bearish
            (df['Close'] < (df['Open'].shift(2) + df['Close'].shift(2)) / 2)  # Closes below first day midpoint
        )

    @staticmethod
    def detect_three_white_soldiers(df, price_threshold=0.1):
        """
        Detect three white soldiers pattern
        """
        return (
            # Three consecutive bullish candles
            (df['Close'] > df['Open']) &
            (df['Close'].shift(1) > df['Open'].shift(1)) &
            (df['Close'].shift(2) > df['Open'].shift(2)) &
            # Each opens within previous body
            (df['Open'] > df['Open'].shift(1)) &
            (df['Open'].shift(1) > df['Open'].shift(2)) &
            # Each closes higher than previous
            (df['Close'] > df['Close'].shift(1)) &
            (df['Close'].shift(1) > df['Close'].shift(2)) &
            # Small upper shadows
            (((df['High'] - df['Close']) / (df['Close'] - df['Open'])) < price_threshold) &
            (((df['High'].shift(1) - df['Close'].shift(1)) / 
              (df['Close'].shift(1) - df['Open'].shift(1))) < price_threshold) &
            (((df['High'].shift(2) - df['Close'].shift(2)) / 
              (df['Close'].shift(2) - df['Open'].shift(2))) < price_threshold)
        )
        
    @staticmethod
    def detect_three_black_crows(df, price_threshold=0.1):
        """
        Detect three black crows pattern (opposite of three white soldiers)
        """
        return (
            # Three consecutive bearish candles
            (df['Close'] < df['Open']) &
            (df['Close'].shift(1) < df['Open'].shift(1)) &
            (df['Close'].shift(2) < df['Open'].shift(2)) &
            # Each opens within previous body
            (df['Open'] < df['Open'].shift(1)) &
            (df['Open'].shift(1) < df['Open'].shift(2)) &
            # Each closes lower than previous
            (df['Close'] < df['Close'].shift(1)) &
            (df['Close'].shift(1) < df['Close'].shift(2)) &
            # Small lower shadows
            (((df['Close'] - df['Low']) / (df['Open'] - df['Close'])) < price_threshold) &
            (((df['Close'].shift(1) - df['Low'].shift(1)) / 
              (df['Open'].shift(1) - df['Close'].shift(1))) < price_threshold) &
            (((df['Close'].shift(2) - df['Low'].shift(2)) / 
              (df['Open'].shift(2) - df['Close'].shift(2))) < price_threshold)
        )

    @staticmethod
    def detect_shooting_star(df, body_ratio=0.3, shadow_ratio=2.0):
        """
        Detect shooting star pattern (inverse of hammer)
        """
        body = abs(df['Close'] - df['Open'])
        total_length = df['High'] - df['Low']
        upper_shadow = df['High'] - df[['Open', 'Close']].max(axis=1)
        lower_shadow = df[['Open', 'Close']].min(axis=1) - df['Low']
        
        return (
            (body / total_length < body_ratio) &  # Small body
            (upper_shadow / body > shadow_ratio) &  # Long upper shadow
            (lower_shadow < body)  # Small lower shadow
        )

    @staticmethod
    def detect_harami(df):
        """
        Detect bullish and bearish harami patterns
        """
        prev_body = abs(df['Close'].shift(1) - df['Open'].shift(1))
        curr_body = abs(df['Close'] - df['Open'])
        
        bullish = (
            (df['Open'].shift(1) > df['Close'].shift(1)) &  # Previous bearish
            (df['Close'] > df['Open']) &  # Current bullish
            (curr_body < prev_body) &  # Current body smaller than previous
            (df['Open'] > df['Close'].shift(1)) &  # Current body inside
            (df['Close'] < df['Open'].shift(1))    # previous body
        )
        
        bearish = (
            (df['Close'].shift(1) > df['Open'].shift(1)) &  # Previous bullish
            (df['Open'] > df['Close']) &  # Current bearish
            (curr_body < prev_body) &  # Current body smaller than previous
            (df['Open'] < df['Close'].shift(1)) &  # Current body inside
            (df['Close'] > df['Open'].shift(1))    # previous body
        )
        
        return bullish, bearish

    @staticmethod
    def detect_piercing_pattern(df, threshold=0.5):
        """
        Detect piercing pattern (bullish reversal)
        """
        prev_body = df['Open'].shift(1) - df['Close'].shift(1)
        prev_mid = (df['Open'].shift(1) + df['Close'].shift(1)) / 2
        
        return (
            (df['Close'].shift(1) < df['Open'].shift(1)) &  # Previous bearish
            (df['Close'] > df['Open']) &  # Current bullish
            (df['Open'] < df['Low'].shift(1)) &  # Opens below previous low
            (df['Close'] > prev_mid) &  # Closes above previous midpoint
            (prev_body > 0)  # Confirms previous bearish candle
        )

    @staticmethod
    def detect_dark_cloud_cover(df, threshold=0.5):
        """
        Detect dark cloud cover pattern (bearish reversal)
        """
        prev_body = df['Close'].shift(1) - df['Open'].shift(1)
        prev_mid = (df['Open'].shift(1) + df['Close'].shift(1)) / 2
        
        return (
            (df['Close'].shift(1) > df['Open'].shift(1)) &  # Previous bullish
            (df['Close'] < df['Open']) &  # Current bearish
            (df['Open'] > df['High'].shift(1)) &  # Opens above previous high
            (df['Close'] < prev_mid) &  # Closes below previous midpoint
            (prev_body > 0)  # Confirms previous bullish candle
        )

    @staticmethod
    def detect_marubozu(df, body_ratio=0.95):
        """
        Detect marubozu (candle with no or very small shadows)
        """
        total_range = df['High'] - df['Low']
        body = abs(df['Close'] - df['Open'])
        
        return body / total_range > body_ratio

    @staticmethod
    def detect_spinning_top(df, body_threshold=0.3, shadow_equal_threshold=0.1):
        """
        Detect spinning top pattern (small body with equal shadows)
        """
        body = abs(df['Close'] - df['Open'])
        total_length = df['High'] - df['Low']
        upper_shadow = df['High'] - df[['Open', 'Close']].max(axis=1)
        lower_shadow = df[['Open', 'Close']].min(axis=1) - df['Low']
        
        return (
            (body / total_length < body_threshold) &  # Small body
            (abs(upper_shadow - lower_shadow) / total_length < shadow_equal_threshold)  # Equal shadows
        )

    @staticmethod
    def detect_tweezer_patterns(df, price_threshold=0.001, body_threshold=0.3):
        """
        Detect tweezer top and bottom patterns
        """
        body = abs(df['Close'] - df['Open'])
        prev_body = abs(df['Close'].shift(1) - df['Open'].shift(1))
        
        tweezer_bottom = (
            ((df['Low'] - df['Low'].shift(1)).abs() < price_threshold) &  # Same low
            ((df['Close'].shift(1) < df['Open'].shift(1))) &  # Previous bearish
            ((df['Close'] > df['Open'])) &  # Current bullish
            ((body / (df['High'] - df['Low']) > body_threshold)) &  # Significant bodies
            ((prev_body / (df['High'].shift(1) - df['Low'].shift(1)) > body_threshold))
        )
        
        tweezer_top = (
            ((df['High'] - df['High'].shift(1)).abs() < price_threshold) &  # Same high
            (df['Close'].shift(1) > df['Open'].shift(1)) &  # Previous bullish
            (df['Close'] < df['Open']) &  # Current bearish
            (body / (df['High'] - df['Low']) > body_threshold) &  # Significant bodies
            (prev_body / (df['High'].shift(1) - df['Low'].shift(1)) > body_threshold)
        )
        
        return tweezer_bottom, tweezer_top

    @staticmethod
    def detect_kicking_pattern(df, gap_threshold=0.01):
        """
        Detect bullish and bearish kicking patterns
        """
        bullish = (
            (df['Open'].shift(1) > df['Close'].shift(1)) &  # Previous bearish marubozu
            (df['Close'] > df['Open']) &  # Current bullish marubozu
            (df['Low'] - df['High'].shift(1) > gap_threshold)  # Gap up
        )
        
        bearish = (
            (df['Close'].shift(1) > df['Open'].shift(1)) &  # Previous bullish marubozu
            (df['Open'] > df['Close']) &  # Current bearish marubozu
            (df['Low'].shift(1) - df['High'] > gap_threshold)  # Gap down
        )
        
        return bullish, bearish

    @staticmethod
    def detect_three_inside(df):
        """
        Detect three inside up/down patterns
        """
        # First detect harami
        bullish_harami, bearish_harami = CandlestickPatterns.detect_harami(df)
        
        three_inside_up = (
            bullish_harami &
            (df['Close'].shift(-1) > df['Close'])  # Third day confirms
        )
        
        three_inside_down = (
            bearish_harami &
            (df['Close'].shift(-1) < df['Close'])  # Third day confirms
        )
        
        return three_inside_up, three_inside_down

    @staticmethod
    def detect_three_outside(df):
        """
        Detect three outside up/down patterns
        """
        # First detect engulfing
        bullish_engulfing, bearish_engulfing = CandlestickPatterns.detect_engulfing(df)
        
        three_outside_up = (
            bullish_engulfing &
            (df['Close'].shift(-1) > df['Close'])  # Third day confirms
        )
        
        three_outside_down = (
            bearish_engulfing &
            (df['Close'].shift(-1) < df['Close'])  # Third day confirms
        )
        
        return three_outside_up, three_outside_down
    
    @staticmethod
    def detect_rising_three_methods(df, threshold=0.01):
        """
        Detect Rising Three Methods (Bullish Continuation)
        Also known as Bullish Rising Three Methods or Bullish Three Methods
        """
        return (
            # First day is a long bullish candle
            (df['Close'].shift(4) > df['Open'].shift(4)) &
            (df['Close'].shift(4) - df['Open'].shift(4) > threshold) &
            
            # Next three days are small bearish candles contained within first day's range
            (df['Open'].shift(3) > df['Close'].shift(3)) &  # Bearish
            (df['Open'].shift(2) > df['Close'].shift(2)) &  # Bearish
            (df['Open'].shift(1) > df['Close'].shift(1)) &  # Bearish
            
            # Small bodies for middle three days
            ((df['Open'].shift(3) - df['Close'].shift(3)) < (df['Close'].shift(4) - df['Open'].shift(4))) &
            ((df['Open'].shift(2) - df['Close'].shift(2)) < (df['Close'].shift(4) - df['Open'].shift(4))) &
            ((df['Open'].shift(1) - df['Close'].shift(1)) < (df['Close'].shift(4) - df['Open'].shift(4))) &
            
            # Last day is a strong bullish candle breaking above first day's high
            (df['Close'] > df['Open']) &
            (df['Close'] > df['High'].shift(4))
        )

    @staticmethod
    def detect_falling_three_methods(df, threshold=0.01):
        """
        Detect Falling Three Methods (Bearish Continuation)
        """
        return (
            # First day is a long bearish candle
            (df['Close'].shift(4) < df['Open'].shift(4)) &
            (df['Open'].shift(4) - df['Close'].shift(4) > threshold) &
            
            # Next three days are small bullish candles contained within first day's range
            (df['Close'].shift(3) > df['Open'].shift(3)) &  # Bullish
            (df['Close'].shift(2) > df['Open'].shift(2)) &  # Bullish
            (df['Close'].shift(1) > df['Open'].shift(1)) &  # Bullish
            
            # Small bodies for middle three days
            ((df['Close'].shift(3) - df['Open'].shift(3)) < (df['Open'].shift(4) - df['Close'].shift(4))) &
            ((df['Close'].shift(2) - df['Open'].shift(2)) < (df['Open'].shift(4) - df['Close'].shift(4))) &
            ((df['Close'].shift(1) - df['Open'].shift(1)) < (df['Open'].shift(4) - df['Close'].shift(4))) &
            
            # Last day is a strong bearish candle breaking below first day's low
            (df['Close'] < df['Open']) &
            (df['Close'] < df['Low'].shift(4))
        )

    @staticmethod
    def detect_abandoned_baby(df, gap_threshold=0.01):
        """
        Detect Abandoned Baby pattern (both bullish and bearish)
        """
        bullish = (
            # First day is bearish
            (df['Close'].shift(2) < df['Open'].shift(2)) &
            # Doji on second day gapped down
            (df['High'].shift(1) < df['Low'].shift(2) - gap_threshold) &
            # Third day is bullish and gaps up
            (df['Close'] > df['Open']) &
            (df['Low'] > df['High'].shift(1) + gap_threshold)
        )
        
        bearish = (
            # First day is bullish
            (df['Close'].shift(2) > df['Open'].shift(2)) &
            # Doji on second day gapped up
            (df['Low'].shift(1) > df['High'].shift(2) + gap_threshold) &
            # Third day is bearish and gaps down
            (df['Close'] < df['Open']) &
            (df['High'] < df['Low'].shift(1) - gap_threshold)
        )
        
        return bullish, bearish

    @staticmethod
    def detect_unique_three_river_bottom(df, threshold=0.01):
        """
        Detect Unique Three River Bottom pattern (Bullish Reversal)
        """
        return (
            # First day is a long bearish candle
            (df['Close'].shift(2) < df['Open'].shift(2)) &
            (df['Open'].shift(2) - df['Close'].shift(2) > threshold) &
            
            # Second day is a bearish candle with lower low but higher close
            (df['Close'].shift(1) < df['Open'].shift(1)) &
            (df['Low'].shift(1) < df['Low'].shift(2)) &
            (df['Close'].shift(1) > df['Close'].shift(2)) &
            
            # Third day is a small bullish or bearish candle with higher low
            (df['Low'] > df['Low'].shift(1)) &
            (abs(df['Close'] - df['Open']) < (df['Open'].shift(2) - df['Close'].shift(2)) * 0.5)
        )

    @staticmethod
    def detect_concealing_baby_swallow(df, threshold=0.01):
        """
        Detect Concealing Baby Swallow pattern (Bullish Reversal)
        """
        return (
            # First two days are bearish marubozu
            (df['Close'].shift(3) < df['Open'].shift(3)) &
            (df['High'].shift(3) - df['Open'].shift(3) < threshold) &
            (df['Close'].shift(3) - df['Low'].shift(3) < threshold) &
            
            (df['Close'].shift(2) < df['Open'].shift(2)) &
            (df['High'].shift(2) - df['Open'].shift(2) < threshold) &
            (df['Close'].shift(2) - df['Low'].shift(2) < threshold) &
            
            # Third day opens within second day's body and makes new low
            (df['Open'].shift(1) < df['Open'].shift(2)) &
            (df['Open'].shift(1) > df['Close'].shift(2)) &
            (df['Low'].shift(1) < df['Low'].shift(2)) &
            
            # Fourth day opens within third day's body and engulfs it
            (df['Open'] < df['Open'].shift(1)) &
            (df['Open'] > df['Close'].shift(1)) &
            (df['Close'] > df['Open'].shift(1))
        )

    @staticmethod
    def detect_three_stars_south(df, threshold=0.01):
        """
        Detect Three Stars in the South pattern (Bullish Reversal)
        """
        return (
            # First day is a long bearish candle
            (df['Close'].shift(2) < df['Open'].shift(2)) &
            (df['Open'].shift(2) - df['Close'].shift(2) > threshold) &
            
            # Second day is a bearish candle with smaller body
            (df['Close'].shift(1) < df['Open'].shift(1)) &
            (df['Open'].shift(1) - df['Close'].shift(1) < df['Open'].shift(2) - df['Close'].shift(2)) &
            (df['Open'].shift(1) <= df['Open'].shift(2)) &
            (df['Low'].shift(1) > df['Low'].shift(2)) &
            
            # Third day is a small bearish candle or doji
            (df['Close'] < df['Open']) &
            (df['Open'] - df['Close'] < df['Open'].shift(1) - df['Close'].shift(1)) &
            (df['Low'] >= df['Low'].shift(1))
        )

    @staticmethod
    def detect_tri_star(df, doji_threshold=0.1):
        """
        Detect Tri-Star pattern (both bullish and bearish)
        """
        # Helper function to detect doji
        is_doji = lambda o, h, l, c: abs(c - o) < (h - l) * doji_threshold
        
        bullish = (
            # Three consecutive doji
            is_doji(df['Open'].shift(2), df['High'].shift(2), df['Low'].shift(2), df['Close'].shift(2)) &
            is_doji(df['Open'].shift(1), df['High'].shift(1), df['Low'].shift(1), df['Close'].shift(1)) &
            is_doji(df['Open'], df['High'], df['Low'], df['Close']) &
            
            # Middle doji gaps down (for bullish)
            (df['High'].shift(1) < df['Low'].shift(2)) &
            # Last doji gaps up
            (df['Low'] > df['High'].shift(1))
        )
        
        bearish = (
            # Three consecutive doji
            is_doji(df['Open'].shift(2), df['High'].shift(2), df['Low'].shift(2), df['Close'].shift(2)) &
            is_doji(df['Open'].shift(1), df['High'].shift(1), df['Low'].shift(1), df['Close'].shift(1)) &
            is_doji(df['Open'], df['High'], df['Low'], df['Close']) &
            
            # Middle doji gaps up (for bearish)
            (df['Low'].shift(1) > df['High'].shift(2)) &
            # Last doji gaps down
            (df['High'] < df['Low'].shift(1))
        )
        
        return bullish, bearish
    
    @staticmethod
    def detect_ladder_bottom(df, threshold=0.01):
        """
        Detect Ladder Bottom pattern (Bullish Reversal)
        """
        return (
            # Three consecutive bearish candles with lower opens
            (df['Close'].shift(3) < df['Open'].shift(3)) &
            (df['Close'].shift(2) < df['Open'].shift(2)) &
            (df['Close'].shift(1) < df['Open'].shift(1)) &
            (df['Open'].shift(2) < df['Open'].shift(3)) &
            (df['Open'].shift(1) < df['Open'].shift(2)) &
            
            # Each with higher lows
            (df['Low'].shift(2) > df['Low'].shift(3)) &
            (df['Low'].shift(1) > df['Low'].shift(2)) &
            
            # Final day is a strong bullish candle
            (df['Close'] > df['Open']) &
            (df['Close'] - df['Open'] > threshold)
        )

    @staticmethod
    def detect_matching_low(df, price_threshold=0.001):
        """
        Detect Matching Low pattern (Bullish)
        """
        return (
            # Two bearish days with matching lows
            (df['Close'].shift(1) < df['Open'].shift(1)) &
            (df['Close'] < df['Open']) &
            (abs(df['Close'] - df['Close'].shift(1)) < price_threshold) &
            # Second day opens near first day's open
            (abs(df['Open'] - df['Open'].shift(1)) < price_threshold * 3)
        )

    @staticmethod
    def detect_mat_hold(df, threshold=0.01):
        """
        Detect Mat Hold pattern (Bullish Continuation)
        """
        return (
            # First day is a strong bullish candle
            (df['Close'].shift(4) > df['Open'].shift(4)) &
            (df['Close'].shift(4) - df['Open'].shift(4) > threshold) &
            
            # Gap up on second day
            (df['Low'].shift(3) > df['High'].shift(4)) &
            
            # Three small bearish candles
            (df['Close'].shift(3) < df['Open'].shift(3)) &
            (df['Close'].shift(2) < df['Open'].shift(2)) &
            (df['Close'].shift(1) < df['Open'].shift(1)) &
            
            # Final bullish candle
            (df['Close'] > df['Open']) &
            (df['Close'] > df['High'].shift(4))
        )

    @staticmethod
    def detect_stick_sandwich(df, price_threshold=0.001):
        """
        Detect Stick Sandwich pattern (Bullish Reversal)
        """
        return (
            # First day bearish
            (df['Close'].shift(2) < df['Open'].shift(2)) &
            # Second day bullish
            (df['Close'].shift(1) > df['Open'].shift(1)) &
            # Third day bearish with close matching first day
            (df['Close'] < df['Open']) &
            (abs(df['Close'] - df['Close'].shift(2)) < price_threshold)
        )

    @staticmethod
    def detect_upside_gap_three_methods(df, gap_threshold=0.01):
        """
        Detect Upside Gap Three Methods (Bullish Continuation)
        """
        return (
            # First day bullish
            (df['Close'].shift(2) > df['Open'].shift(2)) &
            # Second day gaps up and bullish
            (df['Low'].shift(1) > df['High'].shift(2) + gap_threshold) &
            (df['Close'].shift(1) > df['Open'].shift(1)) &
            # Third day opens within gap and closes below second day's open
            (df['Open'] < df['Low'].shift(1)) &
            (df['Open'] > df['High'].shift(2)) &
            (df['Close'] < df['Open'].shift(1))
        )

    @staticmethod
    def detect_downside_gap_three_methods(df, gap_threshold=0.01):
        """
        Detect Downside Gap Three Methods (Bearish Continuation)
        """
        return (
            # First day bearish
            (df['Close'].shift(2) < df['Open'].shift(2)) &
            # Second day gaps down and bearish
            (df['High'].shift(1) < df['Low'].shift(2) - gap_threshold) &
            (df['Close'].shift(1) < df['Open'].shift(1)) &
            # Third day opens within gap and closes above second day's open
            (df['Open'] > df['High'].shift(1)) &
            (df['Open'] < df['Low'].shift(2)) &
            (df['Close'] > df['Open'].shift(1))
        )

    @staticmethod
    def detect_two_rabbits(df, threshold=0.01):
        """
        Detect Two Rabbits pattern (Reversal)
        """
        bullish = (
            # Two long lower shadows
            (df['Close'].shift(1) - df['Low'].shift(1) > threshold) &
            (df['Close'] - df['Low'] > threshold) &
            # Similar lows
            (abs(df['Low'] - df['Low'].shift(1)) < threshold) &
            # Second day closes higher
            (df['Close'] > df['Close'].shift(1))
        )
        
        bearish = (
            # Two long upper shadows
            (df['High'].shift(1) - df['Close'].shift(1) > threshold) &
            (df['High'] - df['Close'] > threshold) &
            # Similar highs
            (abs(df['High'] - df['High'].shift(1)) < threshold) &
            # Second day closes lower
            (df['Close'] < df['Close'].shift(1))
        )
        
        return bullish, bearish

    @staticmethod
    def detect_eight_new_price_lines(df, threshold=0.01):
        """
        Detect Eight New Price Lines pattern (Continuation)
        """
        bullish = (
            # Eight consecutive higher highs and higher lows
            (df['High'] > df['High'].shift(1)) &
            (df['High'].shift(1) > df['High'].shift(2)) &
            (df['High'].shift(2) > df['High'].shift(3)) &
            (df['High'].shift(3) > df['High'].shift(4)) &
            (df['High'].shift(4) > df['High'].shift(5)) &
            (df['High'].shift(5) > df['High'].shift(6)) &
            (df['High'].shift(6) > df['High'].shift(7)) &
            (df['Low'] > df['Low'].shift(1)) &
            (df['Low'].shift(1) > df['Low'].shift(2)) &
            (df['Low'].shift(2) > df['Low'].shift(3)) &
            (df['Low'].shift(3) > df['Low'].shift(4)) &
            (df['Low'].shift(4) > df['Low'].shift(5)) &
            (df['Low'].shift(5) > df['Low'].shift(6)) &
            (df['Low'].shift(6) > df['Low'].shift(7))
        )
        
        bearish = (
            # Eight consecutive lower highs and lower lows
            (df['High'] < df['High'].shift(1)) &
            (df['High'].shift(1) < df['High'].shift(2)) &
            (df['High'].shift(2) < df['High'].shift(3)) &
            (df['High'].shift(3) < df['High'].shift(4)) &
            (df['High'].shift(4) < df['High'].shift(5)) &
            (df['High'].shift(5) < df['High'].shift(6)) &
            (df['High'].shift(6) < df['High'].shift(7)) &
            (df['Low'] < df['Low'].shift(1)) &
            (df['Low'].shift(1) < df['Low'].shift(2)) &
            (df['Low'].shift(2) < df['Low'].shift(3)) &
            (df['Low'].shift(3) < df['Low'].shift(4)) &
            (df['Low'].shift(4) < df['Low'].shift(5)) &
            (df['Low'].shift(5) < df['Low'].shift(6)) &
            (df['Low'].shift(6) < df['Low'].shift(7))
        )
        
        return bullish, bearish

    @staticmethod
    def detect_gapping_side_by_side_white_lines(df, gap_threshold=0.01):
        """
        Detect Gapping Side-by-Side White Lines (Bullish Continuation)
        """
        return (
            # First day bullish
            (df['Close'].shift(2) > df['Open'].shift(2)) &
            # Gap up
            (df['Low'].shift(1) > df['High'].shift(2) + gap_threshold) &
            # Two similar bullish candles after gap
            (df['Close'].shift(1) > df['Open'].shift(1)) &
            (df['Close'] > df['Open']) &
            # Similar opens
            (abs(df['Open'] - df['Open'].shift(1)) < gap_threshold) &
            # Similar closes
            (abs(df['Close'] - df['Close'].shift(1)) < gap_threshold)
        )
        
    # Pattern Modifiers and Strength Indicators
    @staticmethod
    def calculate_pattern_strength(df, pattern_indices, volatility_window=20):
        """
        Calculate pattern strength based on:
        - Relative candle size to historical volatility
        - Volume confirmation
        - Price range relative to recent trading range
        """
        # Calculate historical volatility
        log_returns = np.log(df['Close'] / df['Close'].shift(1))
        historical_volatility = log_returns.rolling(window=volatility_window).std()
        
        # Calculate average true range (ATR)
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift(1)).abs()
        low_close = (df['Low'] - df['Close'].shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=volatility_window).mean()
        
        # Calculate volume strength
        volume_ma = df['Volume'].rolling(window=volatility_window).mean()
        volume_strength = df['Volume'] / volume_ma
        
        strength_scores = pd.Series(0.0, index=df.index)
        
        for idx in pattern_indices[pattern_indices].index:
            # Calculate body size relative to ATR
            body_size = abs(df.loc[idx, 'Close'] - df.loc[idx, 'Open'])
            body_strength = body_size / atr[idx]
            
            # Calculate price movement relative to volatility
            price_movement = abs(df.loc[idx, 'Close'] - df.loc[idx, 'Close'].shift(1)[idx])
            volatility_strength = price_movement / (historical_volatility[idx] * df.loc[idx, 'Close'])
            
            # Combine factors
            strength_scores[idx] = (
                body_strength * 0.4 +
                volatility_strength * 0.4 +
                volume_strength[idx] * 0.2
            )
            
        return strength_scores

    # Complex Multi-Candle Patterns
    @staticmethod
    def detect_three_line_strike(df, threshold=0.01):
        """
        Detect Three-Line Strike pattern (Reversal)
        """
        bullish = (
            # Three declining bearish candles
            (df['Close'].shift(3) < df['Open'].shift(3)) &
            (df['Close'].shift(2) < df['Open'].shift(2)) &
            (df['Close'].shift(1) < df['Open'].shift(1)) &
            (df['Close'].shift(2) < df['Close'].shift(3)) &
            (df['Close'].shift(1) < df['Close'].shift(2)) &
            # Fourth candle opens below third candle and closes above first candle's open
            (df['Open'] < df['Close'].shift(1)) &
            (df['Close'] > df['Open'].shift(3))
        )
        
        bearish = (
            # Three rising bullish candles
            (df['Close'].shift(3) > df['Open'].shift(3)) &
            (df['Close'].shift(2) > df['Open'].shift(2)) &
            (df['Close'].shift(1) > df['Open'].shift(1)) &
            (df['Close'].shift(2) > df['Close'].shift(3)) &
            (df['Close'].shift(1) > df['Close'].shift(2)) &
            # Fourth candle opens above third candle and closes below first candle's open
            (df['Open'] > df['Close'].shift(1)) &
            (df['Close'] < df['Open'].shift(3))
        )
        
        return bullish, bearish

    @staticmethod
    def detect_breakout_patterns(df, window=20, threshold=2.0):
        """
        Detect various breakout patterns with volatility adjustment
        """
        # Calculate Bollinger Bands
        rolling_mean = df['Close'].rolling(window=window).mean()
        rolling_std = df['Close'].rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * threshold)
        lower_band = rolling_mean - (rolling_std * threshold)
        
        # Calculate price channels
        rolling_high = df['High'].rolling(window=window).max()
        rolling_low = df['Low'].rolling(window=window).min()
        
        breakouts = pd.DataFrame(index=df.index)
        
        # Upward breakouts
        breakouts['bullish_bb'] = df['Close'] > upper_band
        breakouts['bullish_channel'] = df['High'] > rolling_high.shift(1)
        
        # Downward breakouts
        breakouts['bearish_bb'] = df['Close'] < lower_band
        breakouts['bearish_channel'] = df['Low'] < rolling_low.shift(1)
        
        # Volume confirmation
        volume_ma = df['Volume'].rolling(window=window).mean()
        breakouts['volume_confirmed'] = df['Volume'] > (volume_ma * 1.5)
        
        return breakouts

    @staticmethod
    def detect_island_reversal(df, gap_threshold=0.01):
        """
        Detect Island Reversal patterns
        """
        bullish = (
            # Gap down into isolated price action
            (df['High'].shift(2) < df['Low'].shift(1) - gap_threshold) &
            # Gap up out of island
            (df['Low'] > df['High'].shift(1) + gap_threshold) &
            # Price and volume characteristics of island
            (df['Volume'].shift(1) > df['Volume'].shift(2)) &
            (df['Close'].shift(1) < df['Open'].shift(1))
        )
        
        bearish = (
            # Gap up into isolated price action
            (df['Low'].shift(2) > df['High'].shift(1) + gap_threshold) &
            # Gap down out of island
            (df['High'] < df['Low'].shift(1) - gap_threshold) &
            # Price and volume characteristics of island
            (df['Volume'].shift(1) > df['Volume'].shift(2)) &
            (df['Close'].shift(1) > df['Open'].shift(1))
        )
        
        return bullish, bearish

    @staticmethod
    def detect_thrust_pattern(df, threshold=0.01):
        """
        Detect Thrust patterns (Continuation)
        """
        bullish = (
            # Initial strong movement
            (df['Close'].shift(2) > df['Open'].shift(2)) &
            (df['Close'].shift(2) - df['Open'].shift(2) > threshold) &
            # Shallow retracement
            (df['Close'].shift(1) < df['Close'].shift(2)) &
            (df['Close'].shift(1) > (df['Close'].shift(2) + df['Open'].shift(2)) / 2) &
            # Thrust day
            (df['Close'] > df['Close'].shift(2))
        )
        
        bearish = (
            # Initial strong movement
            (df['Close'].shift(2) < df['Open'].shift(2)) &
            (df['Open'].shift(2) - df['Close'].shift(2) > threshold) &
            # Shallow retracement
            (df['Close'].shift(1) > df['Close'].shift(2)) &
            (df['Close'].shift(1) < (df['Close'].shift(2) + df['Open'].shift(2)) / 2) &
            # Thrust day
            (df['Close'] < df['Close'].shift(2))
        )
        
        return bullish, bearish

    # Pattern Combinations and Complex Setups
    @staticmethod
    def detect_pattern_combinations(df, lookback_window=5):
        """
        Detect multiple patterns occurring together for stronger signals
        """
        combinations = pd.DataFrame(index=df.index)
        
        # Get individual pattern signals
        doji = CandlestickPatterns.detect_doji(df)
        hammer_bull, hammer_bear = CandlestickPatterns.detect_hammer(df)
        engulfing_bull, engulfing_bear = CandlestickPatterns.detect_engulfing(df)
        
        # Complex Bullish Combinations
        combinations['strong_bullish'] = (
            # Bullish engulfing after a doji
            (doji.shift(1) & engulfing_bull) |
            # Hammer followed by bullish engulfing
            (hammer_bull.shift(1) & engulfing_bull) |
            # Multiple doji at support
            (doji & doji.shift(1) & (df['Close'] < df['Close'].rolling(lookback_window).min()))
        )
        
        # Complex Bearish Combinations
        combinations['strong_bearish'] = (
            # Bearish engulfing after a doji
            (doji.shift(1) & engulfing_bear) |
            # Shooting star followed by bearish engulfing
            (hammer_bear.shift(1) & engulfing_bear) |
            # Multiple doji at resistance
            (doji & doji.shift(1) & (df['Close'] > df['Close'].rolling(lookback_window).max()))
        )
        
        return combinations

    @staticmethod
    def detect_volatility_adjusted_patterns(df, window=20):
        """
        Detect patterns with volatility-based thresholds
        """
        # Calculate volatility metrics
        atr = CandlestickPatterns._calculate_atr(df, window)
        volatility = df['Close'].rolling(window=window).std()
        
        # Adjust thresholds based on volatility
        dynamic_threshold = atr / df['Close'] * 100
        
        patterns = pd.DataFrame(index=df.index)
        
        # Volatile Bullish Engulfing
        patterns['volatile_bullish_engulfing'] = (
            (df['Close'] > df['Open']) &
            (df['Open'] < df['Close'].shift(1)) &
            (df['Close'] > df['Open'].shift(1)) &
            (abs(df['Close'] - df['Open']) > dynamic_threshold * 1.5)
        )
        
        # Low Volatility Breakout
        patterns['low_vol_breakout'] = (
            (df['Close'] > df['High'].rolling(window=window).max()) &
            (volatility < volatility.rolling(window=window).mean() * 0.5)
        )
        
        return patterns

    @staticmethod
    def _calculate_atr(df, window):
        """Helper function to calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift(1))
        low_close = abs(df['Low'] - df['Close'].shift(1))
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(window=window).mean()

    @staticmethod
    def detect_multi_timeframe_patterns(df_daily, df_weekly, df_monthly):
        """
        Detect patterns across multiple timeframes for stronger signals
        """
        patterns = pd.DataFrame(index=df_daily.index)
        
        # Daily patterns
        daily_engulfing_bull, daily_engulfing_bear = CandlestickPatterns.detect_engulfing(df_daily)
        
        # Weekly patterns
        weekly_engulfing_bull, weekly_engulfing_bear = CandlestickPatterns.detect_engulfing(df_weekly)
        
        # Monthly patterns
        monthly_engulfing_bull, monthly_engulfing_bear = CandlestickPatterns.detect_engulfing(df_monthly)
        
        # Combined signals
        patterns['strong_bullish_mtf'] = (
            daily_engulfing_bull &
            weekly_engulfing_bull.reindex(df_daily.index, method='ffill') &
            monthly_engulfing_bull.reindex(df_daily.index, method='ffill')
        )
        
        patterns['strong_bearish_mtf'] = (
            daily_engulfing_bear &
            weekly_engulfing_bear.reindex(df_daily.index, method='ffill') &
            monthly_engulfing_bear.reindex(df_daily.index, method='ffill')
        )
        
        return patterns

    @staticmethod
    def detect_pattern_reliability(df, pattern_func, lookback_window=100, forward_window=20):
        """
        Calculate pattern reliability based on historical performance
        """
        # Get pattern signals
        pattern_signals = pattern_func(df)
        
        reliability_metrics = {
            'success_rate': [],
            'avg_return': [],
            'risk_reward': []
        }
        
        for idx in pattern_signals[pattern_signals].index:
            if idx + forward_window >= len(df):
                continue
                
            # Calculate forward returns
            forward_return = (df['Close'][idx + forward_window] - df['Close'][idx]) / df['Close'][idx]
            
            # Calculate success rate
            historical_signals = pattern_signals[max(0, idx - lookback_window):idx]
            historical_success = sum(
                (df['Close'][i + forward_window] - df['Close'][i]) / df['Close'][i] > 0
                for i in historical_signals[historical_signals].index
                if i + forward_window < len(df)
            ) / max(1, len(historical_signals[historical_signals]))
            
            reliability_metrics['success_rate'].append(historical_success)
            reliability_metrics['avg_return'].append(forward_return)
            
            # Calculate risk-reward ratio
            stop_loss = df['Low'][idx:idx + forward_window].min()
            take_profit = df['High'][idx:idx + forward_window].max()
            risk_reward = abs((take_profit - df['Close'][idx]) / (df['Close'][idx] - stop_loss))
            reliability_metrics['risk_reward'].append(risk_reward)
        
        return pd.DataFrame(reliability_metrics)

    @staticmethod
    def detect_harmonic_patterns(df, tolerance=0.05):
        """
        Detect harmonic price patterns (Gartley, Butterfly, Bat, etc.)
        """
        patterns = pd.DataFrame(index=df.index)
        
        # Helper function for Fibonacci ratios
        def is_fib_ratio(ratio, target, tolerance):
            fib_ratios = {0.382, 0.500, 0.618, 0.786, 1.272, 1.618}
            return any(abs(ratio - fib) < tolerance for fib in fib_ratios)
        
        for i in range(5, len(df)):
            price_swings = [
                df['High'][i-4:i+1].max() - df['Low'][i-4:i+1].min(),
                df['High'][i-3:i+1].max() - df['Low'][i-3:i+1].min(),
                df['High'][i-2:i+1].max() - df['Low'][i-2:i+1].min(),
                df['High'][i-1:i+1].max() - df['Low'][i-1:i+1].min()
            ]
            
            # Calculate ratios between swings
            ratios = [
                price_swings[i] / price_swings[i-1]
                for i in range(1, len(price_swings))
            ]
            
            # Detect Gartley Pattern
            patterns.loc[df.index[i], 'gartley'] = all(
                is_fib_ratio(ratio, target, tolerance)
                for ratio, target in zip(ratios, [0.618, 0.382, 0.786])
            )
            
            # Detect Butterfly Pattern
            patterns.loc[df.index[i], 'butterfly'] = all(
                is_fib_ratio(ratio, target, tolerance)
                for ratio, target in zip(ratios, [0.786, 0.382, 1.618])
            )
            
            # Detect Bat Pattern
            patterns.loc[df.index[i], 'bat'] = all(
                is_fib_ratio(ratio, target, tolerance)
                for ratio, target in zip(ratios, [0.382, 0.382, 1.618])
            )
        
        return patterns