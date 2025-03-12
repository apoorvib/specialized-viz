# Specialized Technical Analysis and Visualization Library

[![PyPI](https://img.shields.io/pypi/v/specialized-viz?color=blue)](https://pypi.org/project/specialized-viz)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A comprehensive Python library for financial technical analysis and visualization, specializing in candlestick patterns, market regime detection, time series analysis, and interactive analysis tools.

## Table of Contents

- Features
  - [Candlestick Pattern Detection](#1-candlestick-pattern-detection)
  - [Market Regime Analysis](#2-market-regime-analysis)
  - [Time Series Analysis & Forecasting](#3-time-series-analysis--forecasting)
  - [Interactive Visualization](#4-interactive-visualization)
- Installation
  - [Prerequisites](#prerequisites)
- Quick Start
  - [Candlestick Analysis](#candlestick-analysis)
  - [Time Series Analysis](#time-series-analysis)
- Core Modules
  - [Candlestick Analysis](#1-candlestick-analysis)
  - [TimeseriesAnalysis](#2-timeseriesanalysis)
  - [TimeseriesVisualizer](#3-timeseriesvisualizer)
  - [TimeseriesForecasting](#4-timeseriesforecasting)
- [Best Practices](#best-practices)
- [Contributing](#contributing)
- [License](#license)
- [Support](#support)



## Features

### 1. Candlestick Pattern Detection

#### Pattern Types

- Basic Patterns
  - Doji, Hammer, Engulfing
  - Morning/Evening Star
  - Marubozu, Harami, Spinning Top
  - Piercing Pattern, Dark Cloud Cover
- Complex Patterns
  - Three Line Strike
  - Abandoned Baby
  - Rising/Falling Three Methods
  - Three White Soldiers, Three Black Crows
  - Three Inside Up/Down, Three Outside Up/Down
  - Island Reversal, Mat Hold
- Harmonic Patterns
  - Gartley
  - Butterfly
  - Bat
- Advanced Patterns
  - Eight New Price Lines
  - Tweezer Tops/Bottoms
  - Kicking Pattern
  - Unique Three River Bottom
  - Concealing Baby Swallow
  - Three Stars in the South
  - Tri-Star
  - Ladder Bottom
  - Matching Low
  - Stick Sandwich
  - Downside Gap Three Methods
  - Two Rabbits
  - Gapping Side-by-Side White Lines
  - Volatility Adjusted Patterns



#### Multi-timeframe Analysis

- Time Aggregation
  - Minute (1m, 5m, 15m, 30m)
  - Hourly (1h, 2h, 4h)
  - Daily (1D)
  - Weekly (1W)
  - Monthly (1M)
- Pattern Verification
  - Cross-timeframe pattern validation
  - Higher timeframe trend alignment
  - Lower timeframe entry signals
  - Volume profile across timeframes
- Market Structure
  - Higher timeframe support/resistance
  - Multi-timeframe momentum analysis
  - Trend strength across timeframes
  - Breakout confirmation
- Analysis Tools
  - Timeframe correlation matrix
  - Multi-timeframe momentum indicators
  - Volume analysis across timeframes
  - Volatility comparison

#### Pattern Analysis

- Automatic pattern recognition
- Configurable parameters
- Pattern reliability metrics
- Statistical validation
- Pattern clustering
- Pattern completion probability
- Pattern breakout prediction
- Historical performance analysis
- Pattern failure analysis
- Cross-timeframe signal confirmation
- Confluence detection with technical indicators

### 2. Market Regime Analysis

- Volatility regime detection
- Trend identification
- Volume analysis
- Combined regime classification
- Momentum and market state analysis
- Regime transition prediction
- Cross-timeframe regime synchronization
- Regime stability assessment
- Regime performance metrics for patterns
- Regime transition driver analysis

### 3. Time Series Analysis & Forecasting

#### Multiple Forecasting Models

- Traditional Methods
  - SARIMA (Seasonal ARIMA)
  - ETS (Error, Trend, Seasonal)
  - Combined Forecasting Approaches
- Advanced Models
  - Prophet
  - VAR (Vector Autoregression)
  - LSTM Neural Networks
  - N-BEATS Neural Network
- Ensemble Methods
  - Random Forest
  - Gradient Boosting
  - Weighted Model Combination

#### Feature Engineering

- Automatic lag feature creation
- Rolling statistics
- Calendar features
- Cyclical encoding
- Interaction features

### 4. Interactive Visualization

- Customizable candlestick charts
- Pattern overlays and annotations
- Multi-timeframe views
- Volume analysis
- Pattern clustering visualization
- Interactive pattern filtering
- Comprehensive dashboards
- Time series decomposition plots
- Feature importance visualization
- Advanced annotation system
- Drawing tools for technical analysis
- Pattern distribution visualization
- Market regime visualization
- Volume profile analysis
- Theme and configuration management
- Cross-timeframe synchronization

### 5. Network Analysis & Integration

- Network Structure
  - Correlation-based networks
  - Pattern influence networks
  - Dynamic network evolution
  - Community detection
- Cross-Module Integration
  - Pattern-network analysis
  - Time series integration
  - Real-time updates
  - Risk assessment
- Visualization
  - Interactive network views
  - Pattern-based coloring
  - Community visualization
  - Dynamic animations

## Installation

### Prerequisites

- Python 3.8+
- pip package manager

```python
# Basic installation
pip install specialized-viz

# Full installation with all models
pip install specialized-viz[all]
```

## Quick Start

### Candlestick Analysis

```python
from specialized_viz import CandlestickVisualizer
import yfinance as yf

# Get sample data
data = yf.download('AAPL', start='2023-01-01')

# Initialize visualizer
viz = CandlestickVisualizer(data)

# Detect and visualize patterns
patterns = viz.detect_patterns()
chart = viz.create_candlestick_chart(patterns=patterns)
chart.show()
```

### Time Series Analysis

```python
from specialized_viz.timeseries import (
    TimeseriesAnalysis,
    TimeseriesConfig,
    TimeseriesVisualizer,
    TimeseriesForecasting
)

# Initialize components
config = TimeseriesConfig(
    decomposition_method='additive',
    seasonal_periods=[5, 21, 63, 252]
)
analyzer = TimeseriesAnalysis(data, config)
viz = TimeseriesVisualizer(analyzer)
forecaster = TimeseriesForecasting(data)

# Basic analysis
decomp_fig = viz.plot_decomposition('Close')
decomp_fig.show()

# Create forecast
features = forecaster.create_features('Close')
forecast = forecaster.seasonal_forecast(features, data['Close'])
```

## Core Modules

### 1. Candlestick Analysis

The Candlestick Analysis module provides comprehensive tools for pattern detection, visualization, and analysis:

#### CandlestickPatterns Class

- Extensive pattern detection methods for over 30 candlestick patterns
- Configurable parameters for pattern sensitivity
- Support for traditional and advanced pattern types
- Multi-timeframe pattern analysis

```python
from specialized_viz.candlestick import CandlestickPatterns

# Create pattern detector instance
detector = CandlestickPatterns()

# Detect specific patterns
doji = detector.detect_doji(data, threshold=0.1)
engulfing_bull, engulfing_bear = detector.detect_engulfing(data)
evening_star = detector.detect_evening_star(data)
```

#### CandlestickVisualizer Class

- Interactive and static chart creation
- Pattern overlays and annotations
- Customizable visualization styles
- Advanced analysis tools:
  - Pattern reliability analysis
  - Pattern clustering
  - Market regime visualization
  - Multi-timeframe analysis
  - Interactive dashboards

```python
from specialized_viz.candlestick import CandlestickVisualizer, VisualizationConfig

# Create custom configuration
config = VisualizationConfig(
    theme='plotly_dark',
    pattern_opacity=0.8,
    show_grid=True
)

# Initialize visualizer with custom config
visualizer = CandlestickVisualizer(data, config)

# Create interactive dashboard
dashboard = visualizer.create_interactive_dashboard()
dashboard.show()

# Analyze pattern reliability
reliability = visualizer.create_pattern_reliability_chart()
reliability.show()
```

#### Market Regime Analysis

- Detect and analyze market regimes
- Volatility and trend analysis
- Regime transition prediction
- Performance metrics by regime

```python
# Analyze market regimes
regimes = visualizer.detect_market_regime(window=20)
regime_chart = visualizer.visualize_market_regimes(regimes)
regime_chart.show()
```

#### Multi-timeframe Analysis

- Synchronize analysis across timeframes
- Confirm patterns across different time horizons
- Identify high-confidence setups

```python
# Create multi-timeframe chart
weekly_data = data.resample('W').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})
monthly_data = data.resample('M').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})

mtf_chart = visualizer.create_multi_timeframe_chart(weekly_data, monthly_data)
mtf_chart.show()
```

### 2. TimeseriesAnalysis

- Time series decomposition
- Pattern detection
- Seasonality analysis
- Change point detection

### 3. TimeseriesVisualizer

- Interactive dashboards
- Correlation analysis
- Distribution evolution
- Feature importance visualization

### 4. TimeseriesForecasting

- Multiple forecasting models
- Feature engineering
- Model evaluation
- Online learning

### 5. Network Analysis

- Network Structure Analysis
  - Correlation networks
  - Pattern-based networks
  - Community detection
  - Network evolution tracking

- Pattern Network Integration
  - Pattern influence mapping
  - Cross-asset pattern propagation
  - Community-based pattern analysis
  - Network-pattern synchronization

- Time Series Network Integration
  - Time-varying network metrics
  - Combined forecasting
  - Temporal pattern analysis
  - Network regime detection

- Interactive Network Visualization
  - Dynamic network views
  - Pattern-colored networks
  - Community visualization
  - Real-time network updates

- Advanced Integration Features
  - Multi-module predictive analytics
  - Cross-module risk assessment
  - Real-time optimization
  - Pattern cascade analysis

### 6. Integration Manager

- Cross-Module Integration
  - Pattern-network coloring
  - Combined indicators
  - Event detection
  - Real-time updates

- Multi-Module Analysis
  - Combined forecasting
  - Risk assessment
  - Pattern propagation
  - Market regime analysis

- Performance Optimization
  - Real-time processing
  - Caching mechanisms
  - Parallel processing
  - Memory management

## Best Practices

1. Data Preparation
   - Use OHLCV data for candlestick analysis
   - Ensure datetime index
   - Handle missing values
   - Use appropriate timeframes
2. Pattern Detection
   - Configure pattern parameters based on timeframe
   - Validate patterns with multiple indicators
   - Consider volume confirmation
3. Model Selection
   - Use seasonal models for periodic data
   - Consider ensemble methods for complex patterns
   - Implement online learning for streaming data
4. Performance Optimization
   - Monitor memory usage
   - Use appropriate window sizes
   - Implement data downsampling when needed

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the <LICENSE> file for details.

## Support

- Documentation: <docs/>
- Issues: [GitHub Issues](https://github.com/username/specialized-viz/issues)
- Examples: <examples/>