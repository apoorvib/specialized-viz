# Specialized Data Visualization Library

A comprehensive Python library for financial technical analysis and visualization, specializing in candlestick patterns, market regime detection, and interactive analysis tools.

## Features

### 1. Candlestick Pattern Detection
- Multiple pattern types supported:
  - Basic patterns (Doji, Hammer, Engulfing)
  - Complex patterns (Three Line Strike, Abandoned Baby)
  - Harmonic patterns (Gartley, Butterfly, Bat)
  - Multi-timeframe patterns
- Automatic pattern recognition with configurable parameters
- Pattern reliability metrics and statistical validation

### 2. Market Regime Analysis
- Volatility regime detection
- Trend identification
- Volume analysis
- Combined regime classification
- Momentum and market state analysis

### 3. Interactive Visualization
- Customizable candlestick charts
- Pattern overlays and annotations
- Multi-timeframe views
- Volume analysis
- Pattern clustering visualization
- Interactive pattern filtering

### 4. Advanced Analytics
- Pattern reliability metrics
- Statistical significance testing
- Pattern sequence analysis
- Correlation analysis
- Custom indicator support

## Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Dependencies
```bash
pip install -r requirements.txt
```

## Usage Guide

### Basic Usage

```
from specialized_viz import CandlestickVisualizer
import pandas as pd

# Load your financial data
df = pd.read_csv('your_data.csv')

# Initialize the visualizer
viz = CandlestickVisualizer(df)

# Create basic candlestick chart with pattern detection
fig = viz.create_candlestick_chart()
fig.show()
```

### Advanced Pattern Detection

```
# Detect specific patterns
patterns = viz.detect_pattern_reliability(lookback_window=100)

# Create pattern reliability chart
reliability_fig = viz.create_pattern_reliability_chart()
reliability_fig.show()

# Analyze pattern clusters
cluster_fig = viz.create_pattern_cluster_chart()
cluster_fig.show()
```

### Market Regime Analysis

```
# Detect market regimes
regimes = viz.detect_market_regime(window=20)

# Visualize market regimes
regime_fig = viz.create_regime_visualization()
regime_fig.show()
```

### Multi-timeframe Analysis

```
# Create multi-timeframe chart
mtf_fig = viz.create_multi_timeframe_chart(
    weekly_df=weekly_data,
    monthly_df=monthly_data
)
mtf_fig.show()
```

## Configuration

### Visualization Settings

```
from specialized_viz import VisualizationConfig

# Create custom configuration
config = VisualizationConfig(
    color_scheme={
        'bullish': '#2ecc71',
        'bearish': '#e74c3c',
        'neutral': '#3498db',
        'complex': '#9b59b6'
    },
    theme='plotly_white',
    pattern_opacity=0.7
)

# Initialize visualizer with custom config
viz = CandlestickVisualizer(df, config=config)
```

### Pattern Detection Parameters

```
# Customize pattern detection parameters
custom_patterns = viz.detect_candlestick_patterns(
    doji_threshold=0.1,
    hammer_body_ratio=0.3,
    engulfing_factor=1.5
)
```

## API Reference

### CandlestickVisualizer Class

#### Main Methods:

- `create_candlestick_chart()`: Creates basic candlestick chart
- `create_pattern_reliability_chart()`: Visualizes pattern reliability
- `create_pattern_cluster_chart()`: Shows pattern clustering analysis
- `create_regime_visualization()`: Displays market regime analysis
- `create_interactive_dashboard()`: Creates interactive analysis dashboard

#### Analysis Methods:

- `detect_market_regime()`: Analyzes market conditions
- `add_pattern_correlation_analysis()`: Calculates pattern correlations
- `analyze_pattern_sequences()`: Studies pattern sequences
- `add_price_action_confirmation()`: Adds price action indicators

#### Utility Methods:

- `add_technical_indicators()`: Overlays technical indicators
- `_calculate_atr()`: Calculates Average True Range
- `_get_pattern_distribution()`: Analyzes pattern distribution

### VisualizationConfig Class

Configuration options for customizing visualizations:

- `color_scheme`: Custom color definitions
- `theme`: Plot theme selection
- `default_height/width`: Chart dimensions
- `pattern_opacity`: Pattern visualization opacity
- `show_grid`: Grid display toggle
- `annotation_font_size`: Text annotation size

## Best Practices

1. Data Preparation
   - Ensure your data includes OHLCV columns
   - Clean missing values
   - Use appropriate timeframe data
2. Pattern Detection
   - Start with default parameters
   - Adjust thresholds based on your needs
   - Validate pattern reliability
3. Visualization
   - Use appropriate timeframes for analysis
   - Consider combining multiple indicators
   - Customize colors for clarity

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and feature requests, please use the GitHub issue tracker.