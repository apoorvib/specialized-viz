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
        """Detect three white soldiers pattern"""
        body = abs(df['Close'] - df['Open'])
        
        return (
            # Three consecutive bullish candles
            (df['Close'] > df['Open']) &
            (df['Close'].shift(1) > df['Open'].shift(1)) &
            (df['Close'].shift(2) > df['Open'].shift(2)) &
            
            # Progressive higher closes
            (df['Close'] > df['Close'].shift(1)) &
            (df['Close'].shift(1) > df['Close'].shift(2)) &
            
            # Opens within previous candle's body
            (df['Open'] > df['Open'].shift(1)) &
            (df['Open'] < df['Close'].shift(1)) &
            (df['Open'].shift(1) > df['Open'].shift(2)) &
            (df['Open'].shift(1) < df['Close'].shift(2)) &
            
            # Small upper shadows
            ((df['High'] - df['Close']) / (body + 0.0001) < price_threshold) &
            ((df['High'].shift(1) - df['Close'].shift(1)) / (body.shift(1) + 0.0001) < price_threshold) &
            ((df['High'].shift(2) - df['Close'].shift(2)) / (body.shift(2) + 0.0001) < price_threshold)
        )

    @staticmethod
    def detect_three_black_crows(df, price_threshold=0.1):
        """Detect three black crows pattern"""
        body = abs(df['Close'] - df['Open'])
        
        return (
            # Three consecutive bearish candles
            (df['Close'] < df['Open']) &
            (df['Close'].shift(1) < df['Open'].shift(1)) &
            (df['Close'].shift(2) < df['Open'].shift(2)) &
            
            # Progressive lower closes
            (df['Close'] < df['Close'].shift(1)) &
            (df['Close'].shift(1) < df['Close'].shift(2)) &
            
            # Opens within previous candle's body
            (df['Open'] < df['Open'].shift(1)) &
            (df['Open'] > df['Close'].shift(1)) &
            (df['Open'].shift(1) < df['Open'].shift(2)) &
            (df['Open'].shift(1) > df['Close'].shift(2)) &
            
            # Small lower shadows
            ((df['Close'] - df['Low']) / (body + 0.0001) < price_threshold) &
            ((df['Close'].shift(1) - df['Low'].shift(1)) / (body.shift(1) + 0.0001) < price_threshold) &
            ((df['Close'].shift(2) - df['Low'].shift(2)) / (body.shift(2) + 0.0001) < price_threshold)
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
        """Detect Rising Three Methods (Bullish Continuation)"""
        # Calculate relative body sizes
        body = abs(df['Close'] - df['Open'])
        body_size = body / df['Close']
        
        return (
            # First day is a long bullish candle
            (df['Close'].shift(4) > df['Open'].shift(4)) &
            (body_size.shift(4) > threshold) &
            
            # Next three days are small bearish candles within first day's range
            (df['Open'].shift(3) > df['Close'].shift(3)) &  # Bearish
            (df['Open'].shift(2) > df['Close'].shift(2)) &  # Bearish
            (df['Open'].shift(1) > df['Close'].shift(1)) &  # Bearish
            
            # Small bodies contained within first day's range
            (df['High'].shift(3) < df['High'].shift(4)) &
            (df['Low'].shift(3) > df['Low'].shift(4)) &
            (df['High'].shift(2) < df['High'].shift(4)) &
            (df['Low'].shift(2) > df['Low'].shift(4)) &
            (df['High'].shift(1) < df['High'].shift(4)) &
            (df['Low'].shift(1) > df['Low'].shift(4)) &
            
            # Last day is a strong bullish candle breaking above
            (df['Close'] > df['Open']) &
            (df['Close'] > df['High'].shift(4)) &
            (body_size > threshold)
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
        """Detect Abandoned Baby pattern (both bullish and bearish)"""
        # Calculate body sizes for doji detection
        body = abs(df['Close'] - df['Open'])
        total_length = df['High'] - df['Low']
        is_doji = body / total_length < 0.1
        
        bullish = (
            # First day is bearish
            (df['Close'].shift(2) < df['Open'].shift(2)) &
            # Second day is doji
            is_doji.shift(1) &
            # Gaps between days
            (df['High'].shift(1) < df['Low'].shift(2)) &  # Gap down after first day
            (df['Low'] > df['High'].shift(1)) &  # Gap up after doji
            # Third day is bullish
            (df['Close'] > df['Open'])
        )
        
        bearish = (
            # First day is bullish
            (df['Close'].shift(2) > df['Open'].shift(2)) &
            # Second day is doji
            is_doji.shift(1) &
            # Gaps between days
            (df['Low'].shift(1) > df['High'].shift(2)) &  # Gap up after first day
            (df['High'] < df['Low'].shift(1)) &  # Gap down after doji
            # Third day is bearish
            (df['Close'] < df['Open'])
        )
        
        return pd.Series(bullish), pd.Series(bearish)    
    
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
        """Detect Tri-Star pattern (both bullish and bearish)"""
        def is_doji(opens, highs, lows, closes):
            body = abs(closes - opens)
            total_range = highs - lows
            return body < (total_range * doji_threshold)
        
        # Detect doji for each day
        doji_today = is_doji(df['Open'], df['High'], df['Low'], df['Close'])
        doji_prev = is_doji(df['Open'].shift(1), df['High'].shift(1), 
                            df['Low'].shift(1), df['Close'].shift(1))
        doji_prev2 = is_doji(df['Open'].shift(2), df['High'].shift(2), 
                            df['Low'].shift(2), df['Close'].shift(2))
        
        bullish = (
            doji_today & doji_prev & doji_prev2 &
            (df['Low'].shift(1) < df['Low'].shift(2)) &  # Second doji gaps down
            (df['Low'] > df['High'].shift(1))  # Third doji gaps up
        )
        
        bearish = (
            doji_today & doji_prev & doji_prev2 &
            (df['High'].shift(1) > df['High'].shift(2)) &  # Second doji gaps up
            (df['High'] < df['Low'].shift(1))  # Third doji gaps down
        )
        
        return pd.Series(bullish), pd.Series(bearish)    
    
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
        """Detect Two Rabbits pattern (Reversal)"""
        bullish = (
            # Fix calculation to use actual prices
            (df['Close'].shift(1) - df['Low'].shift(1) > threshold * df['Close'].shift(1)) &  # Relative threshold
            (df['Close'] - df['Low'] > threshold * df['Close']) &
            (abs(df['Low'] - df['Low'].shift(1)) < threshold * df['Low']) &
            (df['Close'] > df['Close'].shift(1))
        )
        
        bearish = (
            (df['High'].shift(1) - df['Close'].shift(1) > threshold * df['Close'].shift(1)) &
            (df['High'] - df['Close'] > threshold * df['Close']) &
            (abs(df['High'] - df['High'].shift(1)) < threshold * df['High']) &
            (df['Close'] < df['Close'].shift(1))
        )
        
        return pd.Series(bullish), pd.Series(bearish)  # Explicitly return Series
      
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
        """Calculate pattern strength based on volatility and volume"""
        # Calculate historical volatility
        log_returns = np.log(df['Close'] / df['Close'].shift(1))
        historical_volatility = log_returns.rolling(window=volatility_window).std()
        
        # Calculate ATR
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
            body_size = abs(df['Close'][idx] - df['Open'][idx])
            body_strength = body_size / atr[idx]
            
            # Calculate price movement relative to volatility
            price_movement = abs(df['Close'][idx] - df['Close'].shift(1).loc[idx])
            volatility_strength = price_movement / (historical_volatility[idx] * df['Close'][idx])
            
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
    
    @staticmethod
    def detect_island_reversal(df, gap_threshold=0.01):
        """Detect Island Reversal patterns"""
        bullish = pd.Series(False, index=df.index)
        bearish = pd.Series(False, index=df.index)
        
        for i in range(2, len(df)):
            # Detailed price level debug
            print(f"\nDetailed price analysis for {df.index[i]}:")
            print(f"Day 1: O:{df['Open'].iloc[i-2]:.2f} H:{df['High'].iloc[i-2]:.2f} "
                f"L:{df['Low'].iloc[i-2]:.2f} C:{df['Close'].iloc[i-2]:.2f}")
            print(f"Island day: O:{df['Open'].iloc[i-1]:.2f} H:{df['High'].iloc[i-1]:.2f} "
                f"L:{df['Low'].iloc[i-1]:.2f} C:{df['Close'].iloc[i-1]:.2f}")
            print(f"Final day: O:{df['Open'].iloc[i]:.2f} H:{df['High'].iloc[i]:.2f} "
                f"L:{df['Low'].iloc[i]:.2f} C:{df['Close'].iloc[i]:.2f}")
            
            # Gap calculations
            gap_down = df['High'].iloc[i-1] < df['Low'].iloc[i-2]
            gap_up = df['Low'].iloc[i] > df['High'].iloc[i-1]
            
            # Volume analysis
            vol_increase = df['Volume'].iloc[i-1] > df['Volume'].iloc[i-2]
            print(f"Volume sequence: {df['Volume'].iloc[i-2]:.0f} -> "
                f"{df['Volume'].iloc[i-1]:.0f} -> {df['Volume'].iloc[i]:.0f}")
            
            # Pattern completion check
            final_bullish = df['Close'].iloc[i] > df['Open'].iloc[i]
            
            bullish.iloc[i] = (gap_down and gap_up and vol_increase and final_bullish)
            
            print(f"Pattern conditions: Gaps({gap_down},{gap_up}), "
                f"Volume({vol_increase}), Final({final_bullish})")
            
        return bullish, bearish

    @staticmethod
    def detect_mat_hold(df, threshold=0.01):
        """Detect Mat Hold pattern"""
        result = pd.Series(False, index=df.index)
        
        for i in range(4, len(df)):
            print(f"\nAnalyzing sequence at {df.index[i]}:")
            
            # First day analysis
            first_day_bullish = df['Close'].iloc[i-4] > df['Open'].iloc[i-4]
            first_day_body = df['Close'].iloc[i-4] - df['Open'].iloc[i-4]
            first_body_size = abs(first_day_body)
            print(f"First day: O:{df['Open'].iloc[i-4]:.2f} C:{df['Close'].iloc[i-4]:.2f} "
                f"Body:{first_body_size:.2f}")
            
            # Three bearish days analysis
            bearish_days = []
            contained_bodies = []
            for j in range(3):
                curr_idx = i - 3 + j
                is_bearish = df['Close'].iloc[curr_idx] < df['Open'].iloc[curr_idx]
                
                # Check if body (not full range) is contained
                curr_high = max(df['Open'].iloc[curr_idx], df['Close'].iloc[curr_idx])
                curr_low = min(df['Open'].iloc[curr_idx], df['Close'].iloc[curr_idx])
                first_high = max(df['Open'].iloc[i-4], df['Close'].iloc[i-4])
                first_low = min(df['Open'].iloc[i-4], df['Close'].iloc[i-4])
                
                is_contained = curr_high <= first_high and curr_low >= first_low
                
                print(f"Day {j+1}: O:{df['Open'].iloc[curr_idx]:.2f} "
                    f"C:{df['Close'].iloc[curr_idx]:.2f} "
                    f"Bearish:{is_bearish} Contained:{is_contained}")
                
                bearish_days.append(is_bearish)
                contained_bodies.append(is_contained)
            
            # Final day analysis
            final_bullish = df['Close'].iloc[i] > df['Open'].iloc[i]
            breaks_high = df['Close'].iloc[i] > df['High'].iloc[i-4]
            print(f"Final day: O:{df['Open'].iloc[i]:.2f} C:{df['Close'].iloc[i]:.2f} "
                f"Bullish:{final_bullish} Breaks high:{breaks_high}")
            
            result.iloc[i] = (first_day_bullish and 
                            all(bearish_days) and 
                            all(contained_bodies) and
                            final_bullish and 
                            breaks_high)
        
        return result

    # Pattern Combinations and Complex Setups
    @staticmethod
    def detect_pattern_combinations(df, lookback_window=5):
        """Detect multiple patterns occurring together for stronger signals"""
        combinations = pd.DataFrame(index=df.index)
        
        # Get individual pattern signals
        doji = CandlestickPatterns.detect_doji(df, threshold=0.1)
        hammer = CandlestickPatterns.detect_hammer(df)
        
        # Calculate bullish engulfing
        bullish_engulfing = (
            (df['Close'] > df['Open']) &
            (df['Open'] < df['Close'].shift(1)) &
            (df['Close'] > df['Open'].shift(1)) &
            (abs(df['Close'] - df['Open']) > abs(df['Close'].shift(1) - df['Open'].shift(1)))
        )
        
        # Define support/resistance levels
        support = df['Low'].rolling(window=lookback_window).min()
        resistance = df['High'].rolling(window=lookback_window).max()
        
        # Complex Bullish Combinations
        combinations['strong_bullish'] = (
            # Doji followed by bullish engulfing
            (doji.shift(1) & bullish_engulfing) |
            # Hammer at support
            (hammer & (df['Low'] <= support * 1.02)) |
            # Multiple doji near support
            (doji & doji.shift(1) & (df['Low'] <= support * 1.02))
        )
        
        return combinations
    
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
        """Calculate pattern reliability based on historical performance"""
        pattern_signals = pattern_func(df)
        
        reliability_metrics = {
            'success_rate': [],
            'avg_return': [],
            'risk_reward': []
        }
        
        for i in range(len(df) - forward_window):
            if not pattern_signals.iloc[i]:
                continue
                
            # Calculate forward returns using integer indexing
            forward_return = (df['Close'].iloc[i + forward_window] - df['Close'].iloc[i]) / df['Close'].iloc[i]
            
            # Calculate historical success rate
            historical_start = max(0, i - lookback_window)
            historical_signals = pattern_signals.iloc[historical_start:i]
            historical_success = sum(
                (df['Close'].iloc[j + forward_window] - df['Close'].iloc[j]) / df['Close'].iloc[j] > 0
                for j in range(historical_start, i)
                if historical_signals.iloc[j - historical_start] and j + forward_window < len(df)
            ) / max(1, len(historical_signals[historical_signals]))
            
            reliability_metrics['success_rate'].append(historical_success)
            reliability_metrics['avg_return'].append(forward_return)
            
            # Calculate risk-reward ratio
            stop_loss = df['Low'].iloc[i:i + forward_window].min()
            take_profit = df['High'].iloc[i:i + forward_window].max()
            risk_reward = abs((take_profit - df['Close'].iloc[i]) / (df['Close'].iloc[i] - stop_loss))
            reliability_metrics['risk_reward'].append(risk_reward)
        
        return pd.DataFrame(reliability_metrics)
    
    @staticmethod
    def detect_shooting_star(df, body_ratio=0.3, shadow_ratio=2.0):
        """Detect shooting star pattern"""
        body = abs(df['Close'] - df['Open'])
        total_length = df['High'] - df['Low']
        upper_shadow = df['High'] - df[['Open', 'Close']].max(axis=1)
        lower_shadow = df[['Open', 'Close']].min(axis=1) - df['Low']
        
        for i in range(len(df)):
            print(f"\nAnalyzing potential shooting star at {df.index[i]}:")
            print(f"Candle structure:")
            print(f"  Open: {df['Open'].iloc[i]:.2f}")
            print(f"  High: {df['High'].iloc[i]:.2f}")
            print(f"  Low: {df['Low'].iloc[i]:.2f}")
            print(f"  Close: {df['Close'].iloc[i]:.2f}")
            
            print(f"Measurements:")
            print(f"  Body size: {body.iloc[i]:.2f}")
            print(f"  Total length: {total_length.iloc[i]:.2f}")
            print(f"  Upper shadow: {upper_shadow.iloc[i]:.2f}")
            print(f"  Lower shadow: {lower_shadow.iloc[i]:.2f}")
            
            print(f"Ratios:")
            print(f"  Body/Total ratio: {(body/total_length).iloc[i]:.3f} (should be < {body_ratio})")
            print(f"  Upper/Body ratio: {(upper_shadow/body).iloc[i]:.3f} (should be > {shadow_ratio})")
            print(f"  Lower shadow vs Body: {lower_shadow.iloc[i]:.2f} vs {body.iloc[i]:.2f}")
        
        return (
            (body / total_length < body_ratio) &
            (upper_shadow / body > shadow_ratio) &
            (lower_shadow < body)
        )

    @staticmethod
    def detect_eight_new_price_lines(df):
        """Detect Eight New Price Lines pattern"""
        bullish = pd.Series(False, index=df.index)
        bearish = pd.Series(False, index=df.index)
        
        for i in range(7, len(df)):
            sequence = df.iloc[i-7:i+1]
            print(f"\nAnalyzing 8-day sequence ending at {df.index[i]}:")
            print("Daily prices:")
            for j, (idx, row) in enumerate(sequence.iterrows()):
                print(f"Day {j+1} ({idx}):")
                print(f"  High: {row['High']:.2f}")
                print(f"  Low: {row['Low']:.2f}")
            
            highs = sequence['High'].values
            lows = sequence['Low'].values
            
            # Check consecutive higher highs and lows
            is_bullish = True
            is_bearish = True
            
            print("\nChecking consecutive relationships:")
            for j in range(1, 8):
                high_increasing = highs[j] > highs[j-1]
                low_increasing = lows[j] > lows[j-1]
                print(f"Days {j}-{j+1}:")
                print(f"  Highs: {highs[j-1]:.2f} -> {highs[j]:.2f} ({high_increasing})")
                print(f"  Lows: {lows[j-1]:.2f} -> {lows[j]:.2f} ({low_increasing})")
                
                if not (high_increasing and low_increasing):
                    is_bullish = False
                    print("  Failed bullish criteria")
            
            bullish.iloc[i] = is_bullish
            if is_bullish:
                print("Found valid bullish sequence!")
        
        return bullish, bearish

    @staticmethod
    def detect_upside_gap_three_methods(df, gap_threshold=0.01):
        """Detect Upside Gap Three Methods pattern"""
        result = pd.Series(False, index=df.index)
        
        for i in range(2, len(df)):
            print(f"\nAnalyzing potential three methods at {df.index[i]}:")
            # First day analysis
            first_bullish = df['Close'].iloc[i-2] > df['Open'].iloc[i-2]
            print(f"First day (i-2):")
            print(f"  O:{df['Open'].iloc[i-2]:.2f} H:{df['High'].iloc[i-2]:.2f} "
                f"L:{df['Low'].iloc[i-2]:.2f} C:{df['Close'].iloc[i-2]:.2f}")
            print(f"  Bullish: {first_bullish}")
            
            # Second day analysis
            gap_up = df['Low'].iloc[i-1] > df['High'].iloc[i-2]
            second_bullish = df['Close'].iloc[i-1] > df['Open'].iloc[i-1]
            print(f"Second day (i-1):")
            print(f"  O:{df['Open'].iloc[i-1]:.2f} H:{df['High'].iloc[i-1]:.2f} "
                f"L:{df['Low'].iloc[i-1]:.2f} C:{df['Close'].iloc[i-1]:.2f}")
            print(f"  Gap up: {gap_up}")
            print(f"  Bullish: {second_bullish}")
            
            # Third day analysis
            opens_in_gap = (df['Open'].iloc[i] > df['High'].iloc[i-2] and 
                        df['Open'].iloc[i] < df['Low'].iloc[i-1])
            closes_below = df['Close'].iloc[i] < df['Open'].iloc[i-1]
            print(f"Third day (i):")
            print(f"  O:{df['Open'].iloc[i]:.2f} H:{df['High'].iloc[i]:.2f} "
                f"L:{df['Low'].iloc[i]:.2f} C:{df['Close'].iloc[i]:.2f}")
            print(f"  Opens in gap: {opens_in_gap}")
            print(f"  Closes below second open: {closes_below}")
            
            result.iloc[i] = (first_bullish and gap_up and second_bullish and 
                            opens_in_gap and closes_below)
            
            if result.iloc[i]:
                print("Valid three methods pattern found!")
        
        return result

    @staticmethod
    def detect_volatility_adjusted_patterns(df, window=20):
        """Detect volatility adjusted patterns"""
        patterns = pd.DataFrame(index=df.index)
        
        # Calculate volatility metrics
        returns = df['Close'].pct_change()
        volatility = returns.rolling(window=window, min_periods=1).std()
        
        print("\nVolatility Analysis:")
        print(f"Rolling {window}-day volatility:")
        for i in range(min(5, len(df))):
            print(f"Day {i}: {volatility.iloc[i]:.4f}")
        
        for i in range(1, len(df)):
            print(f"\nAnalyzing day {df.index[i]}:")
            # Check for engulfing pattern
            prev_bearish = df['Close'].iloc[i-1] < df['Open'].iloc[i-1]
            curr_bullish = df['Close'].iloc[i] > df['Open'].iloc[i]
            engulfs = (df['Open'].iloc[i] < df['Close'].iloc[i-1] and 
                    df['Close'].iloc[i] > df['Open'].iloc[i-1])
            
            print(f"Previous candle: O:{df['Open'].iloc[i-1]:.2f} C:{df['Close'].iloc[i-1]:.2f}")
            print(f"Current candle: O:{df['Open'].iloc[i]:.2f} C:{df['Close'].iloc[i]:.2f}")
            print(f"Bearish->Bullish: {prev_bearish}->{curr_bullish}")
            print(f"Engulfing: {engulfs}")
            
            # Volatility check
            current_vol = volatility.iloc[i]
            move_size = abs(df['Close'].iloc[i] - df['Open'].iloc[i])
            vol_threshold = current_vol * df['Close'].iloc[i] * 1.5
            
            print(f"Move size: {move_size:.2f}")
            print(f"Volatility threshold: {vol_threshold:.2f}")
            
            patterns.loc[df.index[i], 'volatile_bullish_engulfing'] = (
                prev_bearish and curr_bullish and engulfs and
                move_size > vol_threshold
            )
        
        return patterns
    
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