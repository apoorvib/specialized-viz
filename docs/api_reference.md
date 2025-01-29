# API Reference

## Pattern Detection Methods

### Basic Patterns

#### `detect_doji(df, threshold=0.1)`
Detects Doji candlestick patterns.
- Parameters:
  - `df`: DataFrame with OHLCV data
  - `threshold`: Maximum body/shadow ratio (default: 0.1)
- Returns: Boolean Series indicating Doji locations

#### `detect_hammer(df, body_ratio=0.3, shadow_ratio=2.0)`
Detects Hammer and Inverted Hammer patterns.
- Parameters:
  - `df`: DataFrame with OHLCV data
  - `body_ratio`: Maximum body to total length ratio
  - `shadow_ratio`: Minimum shadow to body ratio
- Returns: Tuple of (hammer, inverted_hammer) Boolean Series

[Continue with all pattern detection methods...]

### Complex Patterns

#### `detect_three_line_strike(df, threshold=0.01)`
Detects Three Line Strike patterns.
- Parameters:
  - `df`: DataFrame with OHLCV data
  - `threshold`: Minimum size threshold
- Returns: Tuple of (bullish, bearish) Boolean Series

[Continue with complex patterns...]

### Harmonic Patterns

#### `detect_gartley(df, tolerance=0.05)`
Detects Gartley harmonic patterns.
- Parameters:
  - `df`: DataFrame with OHLCV data
  - `tolerance`: Fibonacci ratio tolerance
- Returns: Tuple of (bullish, bearish) Boolean Series

[Continue with harmonic patterns...]