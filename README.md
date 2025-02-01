# Specialized Data Visualization Library

A comprehensive Python library for financial technical analysis and visualization, specializing in candlestick patterns, market regime detection, time series analysis, and interactive analysis tools.

![PyPI version](https://img.shields.io/pypi/v/specialized-viz)
![Python versions](https://img.shields.io/pypi/pyversions/specialized-viz)
![License](https://img.shields.io/github/license/apoorvib/specialized-viz)

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [Modules](#modules)
- [API Reference](#api-reference)
- [Best Practices](#best-practices)
- [Contributing](#contributing)
- [License](#license)
- [Support](#support)

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

### 3. Time Series Analysis & Forecasting
- Multiple Forecasting Models:
  - Traditional Methods:
    - ARIMA/SARIMA
    - Exponential Smoothing
    - Linear/Ridge/Lasso Regression
  - Advanced Models:
    - Prophet (Facebook's forecasting tool)
    - VAR (Vector Autoregression)
    - LSTM Neural Networks
    - N-BEATS Neural Network
  - Ensemble Methods:
    - Random Forest
    - Gradient Boosting
    - Model Combination with Optimal Weights
- Feature Engineering:
  - Automatic lag feature creation
  - Rolling statistics
  - Calendar features
  - Cyclical encoding
- Forecast Evaluation:
  - Multiple error metrics
  - Cross-validation
  - Directional accuracy
  - Scale-independent metrics

### 4. Interactive Visualization
- Customizable candlestick charts
- Pattern overlays and annotations
- Multi-timeframe views
- Volume analysis
- Pattern clustering visualization
- Interactive pattern filtering

## Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Basic Installation
```bash
pip install specialized-viz
```

### Full Installation (with all forecasting models)
```bash
pip install specialized-viz[all]
```

Note: Some models like Prophet require additional system dependencies. For Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install python3-dev gcc
```

## Quick Start

```python
from specialized_viz import CandlestickVisualizer, TimeseriesForecasting
import yfinance as yf

# Get sample data
data = yf.download('AAPL', start='2023-01-01')

# Candlestick Analysis
viz = CandlestickVisualizer(data)
patterns = viz.create_candlestick_chart()
patterns.show()

# Time Series Forecasting
ts = TimeseriesForecasting(data)
forecast = ts.prophet_forecast(target=data['Close'])
forecast.show()
```

## Usage Guide

### Basic Time Series Analysis
```python
from specialized_viz import TimeseriesForecasting

# Initialize forecaster
ts = TimeseriesForecasting(data)

# Create features
features = ts.create_features(column='Close')

# Train ensemble model
ensemble_result = ts.ensemble_forecast(features, data['Close'])

# Generate probabilistic forecast
prob_forecast = ts.probabilistic_forecast(features, data['Close'])
```

### Advanced Forecasting Models
```python
# Prophet Model
prophet_forecast = ts.prophet_forecast(features, data['Close'])

# LSTM Neural Network
lstm_forecast = ts.lstm_forecast(features, data['Close'])

# N-BEATS Model
nbeats_forecast = ts.nbeats_forecast(features, data['Close'])

# Vector Autoregression
var_forecast = ts.var_forecast(features, data['Close'])
```

## Best Practices

### 1. Data Preparation
- Ensure your data includes OHLCV columns
- Clean missing values
- Use appropriate timeframe data

### 2. Model Selection
- Start with simple models (ensemble of traditional methods)
- Use Prophet for data with strong seasonality
- Consider LSTM/N-BEATS for complex patterns
- Use VAR for multivariate analysis

### 3. Feature Engineering
- Include domain-specific features
- Consider multiple time horizons
- Test feature importance
- Remove highly correlated features

### 4. Model Evaluation
- Use multiple error metrics
- Consider computational requirements
- Validate on recent data
- Monitor prediction intervals

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