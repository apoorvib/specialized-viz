"""Enhanced time series visualization module for specialized-viz library."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import shap
from .analysis import TimeseriesAnalysis

class TimeseriesVisualizer:
    """Enhanced visualization class for time series analysis."""
    
    def __init__(self, analyzer: TimeseriesAnalysis):
        """Initialize visualizer with analyzer instance."""
        self.analyzer = analyzer
        self.color_scheme = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'tertiary': '#2ca02c',
            'quaternary': '#d62728',
            'background': '#ffffff',
            'grid': '#e0e0e0'
        }
        
    def plot_correlogram(self, column: str) -> go.Figure:
        """Create advanced correlation visualization.
        
        Args:
            column: Name of the column to analyze
            
        Returns:
            Plotly figure with correlation analysis
        """
        series = self.analyzer.data[column]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Autocorrelation Function (ACF)',
                'Partial Autocorrelation (PACF)',
                'Rolling Correlation',
                'Correlation Heatmap'
            )
        )
        
        # ACF Plot
        acf_values = pd.Series(series).autocorr(lag=None)
        fig.add_trace(
            go.Bar(
                x=list(range(len(acf_values))),
                y=acf_values,
                name='ACF'
            ),
            row=1, col=1
        )
        
        # PACF Plot
        from statsmodels.tsa.stattools import pacf
        pacf_values = pacf(series.dropna())
        fig.add_trace(
            go.Bar(
                x=list(range(len(pacf_values))),
                y=pacf_values,
                name='PACF'
            ),
            row=1, col=2
        )
        
        # Rolling Correlation
        rolling_corr = series.rolling(window=30).corr(
            series.shift(1)
        )
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=rolling_corr,
                name='Rolling Correlation'
            ),
            row=2, col=1
        )
        
        # Correlation Heatmap
        if isinstance(self.analyzer.data, pd.DataFrame):
            corr_matrix = self.analyzer.data.corr()
            fig.add_trace(
                go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu'
                ),
                row=2, col=2
            )
            
        fig.update_layout(height=800, width=1200, showlegend=False)
        return fig
        
    def plot_distribution_evolution(self, column: str) -> go.Figure:
        """Create distribution analysis over time.
        
        Args:
            column: Name of the column to analyze
            
        Returns:
            Plotly figure with distribution analysis
        """
        series = self.analyzer.data[column]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Rolling Probability Density',
                'Quantile Plot',
                'Box Plot Over Time',
                'Violin Plot by Period'
            )
        )
        
        # Rolling Probability Density
        from scipy import stats
        windows = np.array_split(series, 4)  # Split into 4 periods
        for i, window in enumerate(windows):
            kernel = stats.gaussian_kde(window.dropna())
            x_range = np.linspace(series.min(), series.max(), 100)
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=kernel(x_range),
                    name=f'Period {i+1}',
                    line=dict(color=px.colors.qualitative.Set3[i])
                ),
                row=1, col=1
            )
            
        # Quantile Plot
        fig.add_trace(
            go.Scatter(
                x=series.sort_values(),
                y=stats.norm.ppf(np.linspace(0.01, 0.99, len(series))),
                mode='markers',
                name='Q-Q Plot'
            ),
            row=1, col=2
        )
        
        # Box Plot Over Time
        df_box = pd.DataFrame({
            'value': series.values,
            'period': pd.qcut(np.arange(len(series)), 4)
        })
        fig.add_trace(
            go.Box(
                x=df_box['period'].astype(str),
                y=df_box['value'],
                name='Box Plot'
            ),
            row=2, col=1
        )
        
        # Violin Plot
        df_violin = pd.DataFrame({
            'value': series.values,
            'period': series.index.month
        })
        fig.add_trace(
            go.Violin(
                x=df_violin['period'].astype(str),
                y=df_violin['value'],
                name='Violin Plot',
                box_visible=True
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=800, width=1200, showlegend=True)
        return fig
        
    def create_animation(self, column: str) -> go.Figure:
        """Create animated visualizations.
        
        Args:
            column: Name of the column to analyze
            
        Returns:
            Plotly figure with animation frames
        """
        series = self.analyzer.data[column]
        
        # Create frames for animation
        frames = []
        window_size = len(series) // 10  # 10 frames
        
        for i in range(0, len(series) - window_size, window_size // 2):
            window = series.iloc[i:i+window_size]
            
            # Trend analysis for window
            X = np.arange(len(window)).reshape(-1, 1)
            y = window.values.reshape(-1, 1)
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X, y)
            trend = model.predict(X)
            
            frame = go.Frame(
                data=[
                    go.Scatter(
                        x=window.index,
                        y=window.values,
                        name='Original',
                        mode='lines'
                    ),
                    go.Scatter(
                        x=window.index,
                        y=trend.flatten(),
                        name='Trend',
                        line=dict(color='red', dash='dash')
                    )
                ],
                name=f'frame{i}'
            )
            frames.append(frame)
            
        # Create base figure
        fig = go.Figure(
            data=frames[0].data,
            frames=frames,
        )
        
        # Add slider and play button
        fig.update_layout(
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [{
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 500, 'redraw': True},
                        'fromcurrent': True,
                        'transition': {'duration': 300}
                    }]
                }]
            }],
            sliders=[{
                'currentvalue': {'prefix': 'Frame: '},
                'steps': [{'args': [[f.name], {'mode': 'immediate'}],
                          'label': str(k),
                          'method': 'animate'} for k, f in enumerate(frames)]
            }]
        )
        
        return fig
        
    def plot_feature_importance(self, 
                              features: pd.DataFrame,
                              target: pd.Series) -> go.Figure:
        """Create feature importance visualization.
        
        Args:
            features: Feature DataFrame
            target: Target series
            
        Returns:
            Plotly figure with feature importance analysis
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'SHAP Values',
                'Feature Correlation',
                'Temporal Importance',
                'Impact Analysis'
            )
        )
        
        # SHAP Values
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(features, target)
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features)
        
        fig.add_trace(
            go.Bar(
                x=features.columns,
                y=np.abs(shap_values).mean(0),
                name='SHAP Values'
            ),
            row=1, col=1
        )
        
        # Feature Correlation
        corr_matrix = features.corr()
        fig.add_trace(
            go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu'
            ),
            row=1, col=2
        )
        
        # Temporal Importance
        importance_over_time = pd.DataFrame(
            index=features.index,
            columns=features.columns
        )
        
        window_size = len(features) // 10
        for i in range(0, len(features) - window_size, window_size):
            window_features = features.iloc[i:i+window_size]
            window_target = target.iloc[i:i+window_size]
            model.fit(window_features, window_target)
            importance_over_time.iloc[i:i+window_size] = model.feature_importances_
            
        for column in features.columns:
            fig.add_trace(
                go.Scatter(
                    x=importance_over_time.index,
                    y=importance_over_time[column],
                    name=column
                ),
                row=2, col=1
            )
            
        # Impact Analysis
        impact_scores = []
        for column in features.columns:
            shuffled = features.copy()
            shuffled[column] = np.random.permutation(shuffled[column])
            model_shuffled = RandomForestRegressor(n_estimators=100, random_state=42)
            model_shuffled.fit(shuffled, target)
            impact = mean_squared_error(target, model.predict(features)) - \
                    mean_squared_error(target, model_shuffled.predict(shuffled))
            impact_scores.append(impact)
            
        fig.add_trace(
            go.Bar(
                x=features.columns,
                y=impact_scores,
                name='Feature Impact'
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=1000, width=1200, showlegend=True)
        return fig
    
    def create_comprehensive_dashboard(self, column: str) -> go.Figure:
        """Create a comprehensive analysis dashboard.
        
        Args:
            column: Name of the column to analyze
            
        Returns:
            Plotly figure with comprehensive dashboard
        """
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                'Time Series Plot',
                'Decomposition',
                'Seasonality Analysis',
                'Anomaly Detection',
                'Cycle Analysis',
                'Distribution',
                'Correlation Analysis',
                'Feature Importance',
                'Forecasting'
            ),
            specs=[[{'secondary_y': True}, {'secondary_y': False}, {'secondary_y': False}],
                  [{'secondary_y': False}, {'secondary_y': False}, {'secondary_y': False}],
                  [{'secondary_y': False}, {'secondary_y': False}, {'secondary_y': False}]]
        )
        
        series = self.analyzer.data[column]
        
        # 1. Time Series Plot with Moving Averages
        fig.add_trace(
            go.Scatter(x=series.index, y=series, name='Original',
                      line=dict(color=self.color_scheme['primary'])),
            row=1, col=1
        )
        
        # Add moving averages
        for window in [7, 30]:
            ma = series.rolling(window=window).mean()
            fig.add_trace(
                go.Scatter(x=ma.index, y=ma, name=f'MA-{window}',
                          line=dict(dash='dash')),
                row=1, col=1
            )
        
        # 2. Decomposition
        components = self.analyzer.decompose(column)
        fig.add_trace(
            go.Scatter(x=components['trend'].index, y=components['trend'],
                      name='Trend', line=dict(color=self.color_scheme['secondary'])),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=components['seasonal'].index, y=components['seasonal'],
                      name='Seasonal', line=dict(color=self.color_scheme['tertiary'])),
            row=1, col=2
        )
        
        # 3. Seasonality Analysis
        seasonality = self.analyzer.analyze_seasonality(column)
        seasonal_strength = pd.Series(seasonality['seasonal_strength_12'])
        fig.add_trace(
            go.Bar(x=seasonal_strength.index, y=seasonal_strength.values,
                  name='Seasonal Strength'),
            row=1, col=3
        )
        
        # 4. Anomaly Detection
        anomalies = self.analyzer.detect_anomalies(column)
        fig.add_trace(
            go.Scatter(x=series.index, y=series, name='Data',
                      line=dict(color='lightgrey')),
            row=2, col=1
        )
        
        for method, anomaly_series in anomalies.items():
            anomaly_points = series[anomaly_series]
            fig.add_trace(
                go.Scatter(x=anomaly_points.index, y=anomaly_points,
                          mode='markers', name=f'Anomalies ({method})',
                          marker=dict(size=8, symbol='x')),
                row=2, col=1
            )
        
        # 5. Cycle Analysis
        cycles = self.analyzer.analyze_cycles(column)
        fig.add_trace(
            go.Scatter(x=np.arange(len(cycles['fourier']['dominant_frequencies'])),
                      y=cycles['fourier']['amplitudes'],
                      name='Frequency Components'),
            row=2, col=2
        )
        
        # 6. Distribution
        fig.add_trace(
            go.Histogram(x=series, nbinsx=30, name='Distribution'),
            row=2, col=3
        )
        
        # 7. Correlation Analysis
        autocorr = pd.Series(series).autocorr(lag=None)
        fig.add_trace(
            go.Bar(x=np.arange(len(autocorr)), y=autocorr,
                  name='Autocorrelation'),
            row=3, col=1
        )
        
        # 8. Feature Importance
        # Create basic features for demonstration
        features = pd.DataFrame({
            'lag_1': series.shift(1),
            'lag_7': series.shift(7),
            'ma_7': series.rolling(7).mean(),
            'std_7': series.rolling(7).std()
        }).dropna()
        
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        target = series[features.index]
        model.fit(features, target)
        
        importance = pd.Series(model.feature_importances_, index=features.columns)
        fig.add_trace(
            go.Bar(x=importance.index, y=importance.values,
                  name='Feature Importance'),
            row=3, col=2
        )
        
        # 9. Forecasting
        from sklearn.model_selection import train_test_split
        train_size = int(len(features) * 0.8)
        train_features = features[:train_size]
        test_features = features[train_size:]
        train_target = target[:train_size]
        test_target = target[train_size:]
        
        model.fit(train_features, train_target)
        predictions = model.predict(test_features)
        
        fig.add_trace(
            go.Scatter(x=test_features.index, y=test_target,
                      name='Actual', line=dict(color=self.color_scheme['primary'])),
            row=3, col=3
        )
        fig.add_trace(
            go.Scatter(x=test_features.index, y=predictions,
                      name='Forecast', line=dict(color=self.color_scheme['secondary'],
                                               dash='dash')),
            row=3, col=3
        )
        
        # Update layout for each subplot
        for i in range(1, 4):
            for j in range(1, 4):
                fig.update_xaxes(title_text="Date", row=i, col=j)
                fig.update_yaxes(title_text="Value", row=i, col=j)
        
        # Overall layout updates
        fig.update_layout(
            height=1200,
            width=1600,
            showlegend=True,
            title_text="Comprehensive Time Series Analysis Dashboard",
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
        
    def plot_decomposition(self, column: str, period: int = None) -> go.Figure:
        """Create decomposition plot showing trend, seasonal, and residual components.
        
        Args:
            column: Name of the column to decompose
            period: Optional seasonal period to use
            
        Returns:
            Plotly figure with decomposition plots
        """
        components = self.analyzer.decompose(column, period=period)
        
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual'),
            vertical_spacing=0.05
        )
        
        # Plot original data
        fig.add_trace(
            go.Scatter(
                x=self.analyzer.data.index,
                y=components['original'],
                name='Original',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Plot trend
        fig.add_trace(
            go.Scatter(
                x=self.analyzer.data.index,
                y=components['trend'],
                name='Trend',
                line=dict(color='red')
            ),
            row=2, col=1
        )
        
        # Plot seasonal component
        fig.add_trace(
            go.Scatter(
                x=self.analyzer.data.index,
                y=components['seasonal'],
                name='Seasonal',
                line=dict(color='green')
            ),
            row=3, col=1
        )
        
        # Plot residual
        fig.add_trace(
            go.Scatter(
                x=self.analyzer.data.index,
                y=components['residual'],
                name='Residual',
                line=dict(color='purple')
            ),
            row=4, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            width=1200,
            title_text="Time Series Decomposition",
            showlegend=True
        )
        
        # Update y-axes titles
        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_yaxes(title_text="Trend", row=2, col=1)
        fig.update_yaxes(title_text="Seasonal", row=3, col=1)
        fig.update_yaxes(title_text="Residual", row=4, col=1)
        
        return fig
        
    def plot_trend_analysis(self, column: str) -> go.Figure:
        """Create trend analysis plot.
        
        Args:
            column: Name of the column to analyze
            
        Returns:
            Plotly figure with trend analysis
        """
        trend_analysis = self.analyzer.analyze_trend(column)
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Trend Analysis', 'Residuals'),
            vertical_spacing=0.15
        )
        
        # Plot original data and trend line
        fig.add_trace(
            go.Scatter(
                x=self.analyzer.data.index,
                y=self.analyzer.data[column],
                name='Original',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=self.analyzer.data.index,
                y=trend_analysis['trend_line'],
                name='Trend Line',
                line=dict(color='red', dash='dash')
            ),
            row=1, col=1
        )
        
        # Add trend statistics as annotations
        fig.add_annotation(
            xref='paper',
            yref='paper',
            x=0.02,
            y=0.98,
            text=f"Slope: {trend_analysis['slope']:.4f}<br>" + \
                f"R²: {trend_analysis['r_squared']:.4f}<br>" + \
                f"Trend: {trend_analysis['mann_kendall_stats']['trend']}<br>" + \
                f"p-value: {trend_analysis['mann_kendall_stats']['p_value']:.4f}",
            showarrow=False,
            bgcolor='rgba(255,255,255,0.8)'
        )
        
        # Plot residuals
        residuals = self.analyzer.data[column] - trend_analysis['trend_line']
        fig.add_trace(
            go.Scatter(
                x=self.analyzer.data.index,
                y=residuals,
                name='Residuals',
                line=dict(color='gray')
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=800,
            width=1200,
            title_text=f"Trend Analysis for {column}",
            showlegend=True
        )
        
        return fig
        
    def plot_change_points(self, column: str, 
                        methods: Optional[list] = None) -> go.Figure:
        """Create change point detection plot.
        
        Args:
            column: Name of the column to analyze
            methods: List of detection methods to use
            
        Returns:
            Plotly figure with change points
        """
        if methods is None:
            methods = ['cusum', 'pettitt']
            
        fig = go.Figure()
        
        # Plot original data
        fig.add_trace(
            go.Scatter(
                x=self.analyzer.data.index,
                y=self.analyzer.data[column],
                name='Original',
                line=dict(color='blue')
            )
        )
        
        colors = {'cusum': 'red', 'pettitt': 'green'}
        
        # Add change points for each method
        for method in methods:
            change_points = self.analyzer.detect_change_points(column, method=method)
            if change_points.any():
                # Plot vertical lines at change points
                for idx in self.analyzer.data.index[change_points]:
                    fig.add_vline(
                        x=idx,
                        line_dash="dash",
                        line_color=colors[method],
                        opacity=0.5,
                        name=f"{method.upper()} Change Point"
                    )
        
        fig.update_layout(
            height=600,
            width=1200,
            title_text=f"Change Point Detection for {column}",
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
    
    def create_interactive_dashboard(self, column: str) -> go.Figure:
        """Create an interactive dashboard with all analyses.
        
        Args:
            column: Name of the column to analyze
            
        Returns:
            Plotly figure with interactive dashboard
        """
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Time Series with Trend',
                'Decomposition',
                'Change Points (CUSUM)',
                'Change Points (Pettitt)',
                'Seasonal Pattern',
                'Residual Analysis'
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # Get all analyses
        trend_analysis = self.analyzer.analyze_trend(column)
        components = self.analyzer.decompose(column)
        cusum_changes = self.analyzer.detect_change_points(column, method='cusum')
        pettitt_changes = self.analyzer.detect_change_points(column, method='pettitt')
        
        # 1. Time Series with Trend
        fig.add_trace(
            go.Scatter(
                x=self.analyzer.data.index,
                y=self.analyzer.data[column],
                name='Original',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=self.analyzer.data.index,
                y=trend_analysis['trend_line'],
                name='Trend',
                line=dict(color='red', dash='dash')
            ),
            row=1, col=1
        )
        
        # 2. Decomposition
        fig.add_trace(
            go.Scatter(
                x=self.analyzer.data.index,
                y=components['trend'],
                name='Trend Component',
                line=dict(color='purple')
            ),
            row=1, col=2
        )
        
        # 3. CUSUM Change Points
        fig.add_trace(
            go.Scatter(
                x=self.analyzer.data.index,
                y=self.analyzer.data[column],
                name='Original',
                line=dict(color='blue'),
                showlegend=False
            ),
            row=2, col=1
        )
        for idx in self.analyzer.data.index[cusum_changes]:
            fig.add_vline(
                x=idx,
                line_dash="dash",
                line_color="red",
                opacity=0.5,
                row=2, col=1
            )
            
        # 4. Pettitt Change Points
        fig.add_trace(
            go.Scatter(
                x=self.analyzer.data.index,
                y=self.analyzer.data[column],
                name='Original',
                line=dict(color='blue'),
                showlegend=False
            ),
            row=2, col=2
        )
        for idx in self.analyzer.data.index[pettitt_changes]:
            fig.add_vline(
                x=idx,
                line_dash="dash",
                line_color="green",
                opacity=0.5,
                row=2, col=2
            )
            
        # 5. Seasonal Pattern
        fig.add_trace(
            go.Scatter(
                x=self.analyzer.data.index,
                y=components['seasonal'],
                name='Seasonal',
                line=dict(color='green')
            ),
            row=3, col=1
        )
        
        # 6. Residual Analysis
        fig.add_trace(
            go.Scatter(
                x=self.analyzer.data.index,
                y=components['residual'],
                name='Residual',
                line=dict(color='gray')
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1000,
            width=1200,
            title_text=f"Time Series Analysis Dashboard for {column}",
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
    
    # fig.update_layout(
    #     height=1200,
    #     width=1600,
    #     showlegend=True,
    #     title_text="Comprehensive Time Series Analysis Dashboard"
    # )
    
    # return fig



"""Time series visualization module for specialized-viz library."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, Optional
from .analysis import TimeseriesAnalysis, TimeseriesConfig

class TimeseriesVisualizer:
    """Visualization class for time series analysis."""
    
    def __init__(self, analyzer: TimeseriesAnalysis):
        """Initialize visualizer with analyzer instance.
        
        Args:
            analyzer: TimeseriesAnalysis instance
        """
        self.analyzer = analyzer
        
    