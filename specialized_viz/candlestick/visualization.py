import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass
from sklearn.cluster import DBSCAN
import inspect
from .patterns import CandlestickPatterns

@dataclass
class VisualizationConfig:
    """Configuration class for visualization settings"""
    color_scheme: Dict[str, str] = None
    theme: str = 'plotly_white'
    default_height: int = 800
    default_width: int = 1200
    pattern_opacity: float = 0.7
    show_grid: bool = True
    annotation_font_size: int = 10
    
    def __post_init__(self):
        if self.color_scheme is None:
            self.color_scheme = {
                'bullish': '#2ecc71',
                'bearish': '#e74c3c',
                'neutral': '#3498db',
                'complex': '#9b59b6',
                'volume_up': '#2ecc71',
                'volume_down': '#e74c3c',
                'background': '#ffffff',
                'text': '#2c3e50'
            }

class CandlestickVisualizer:
    def __init__(self, df: pd.DataFrame, config: Optional[VisualizationConfig] = None):
        """
        Initialize visualizer with financial data and configuration
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            config (VisualizationConfig, optional): Visualization configuration
        """
        self.df = df
        self.patterns = CandlestickPatterns()
        self.config = config or VisualizationConfig()
        self._cached_pattern_results = {}
        
    def _cluster_patterns(self, window_size: int = 20) -> pd.DataFrame:
        """
        Cluster pattern occurrences to identify high-density regions
        
        Args:
            window_size (int): Rolling window size for pattern density
            
        Returns:
            pd.DataFrame: Clustered pattern data
        """
        pattern_methods = self._get_all_pattern_methods()
        pattern_density = pd.DataFrame(index=self.df.index)
        
        # Calculate pattern density
        for pattern_name, pattern_func in pattern_methods.items():
            try:
                result = pattern_func(self.df)
                if isinstance(result, tuple):
                    for sub_pattern, sub_name in zip(result, ['bullish', 'bearish']):
                        density = sub_pattern.rolling(window=window_size).sum()
                        pattern_density[f"{pattern_name}_{sub_name}"] = density
                else:
                    density = result.rolling(window=window_size).sum()
                    pattern_density[pattern_name] = density
            except Exception as e:
                print(f"Error clustering pattern {pattern_name}: {str(e)}")
        
        # Perform DBSCAN clustering
        clustering = DBSCAN(eps=0.5, min_samples=3)
        pattern_density['cluster'] = clustering.fit_predict(
            pattern_density.fillna(0)
        )
        
        return pattern_density

    def create_pattern_cluster_chart(self) -> go.Figure:
        """
        Create a visualization of pattern clusters
        
        Returns:
            go.Figure: Plotly figure with pattern clusters
        """
        clustered_data = self._cluster_patterns()
        
        fig = make_subplots(rows=2, cols=1, 
                           shared_xaxes=True,
                           vertical_spacing=0.05,
                           row_heights=[0.7, 0.3])
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=self.df.index,
                open=self.df['Open'],
                high=self.df['High'],
                low=self.df['Low'],
                close=self.df['Close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Add pattern density heatmap
        cluster_colors = px.colors.qualitative.Set3
        for cluster in sorted(clustered_data['cluster'].unique()):
            if cluster >= 0:  # Ignore noise points (-1)
                cluster_data = clustered_data[clustered_data['cluster'] == cluster]
                fig.add_trace(
                    go.Scatter(
                        x=cluster_data.index,
                        y=cluster_data.mean(axis=1),
                        fill='tozeroy',
                        name=f'Cluster {cluster}',
                        line=dict(color=cluster_colors[cluster % len(cluster_colors)]),
                        opacity=0.3
                    ),
                    row=2, col=1
                )
        
        fig.update_layout(
            title='Pattern Cluster Analysis',
            height=self.config.default_height,
            width=self.config.default_width,
            template=self.config.theme,
            showlegend=True
        )
        
        return fig

    def create_multi_timeframe_chart(self, 
                                   weekly_df: pd.DataFrame, 
                                   monthly_df: pd.DataFrame) -> go.Figure:
        """
        Create a multi-timeframe analysis chart
        
        Args:
            weekly_df (pd.DataFrame): Weekly OHLCV data
            monthly_df (pd.DataFrame): Monthly OHLCV data
            
        Returns:
            go.Figure: Plotly figure with multi-timeframe analysis
        """
        fig = make_subplots(rows=3, cols=1,
                           shared_xaxes=True,
                           vertical_spacing=0.05,
                           subplot_titles=('Daily', 'Weekly', 'Monthly'))
        
        # Add candlestick charts for each timeframe
        timeframe_data = [
            (self.df, 1), (weekly_df, 2), (monthly_df, 3)
        ]
        
        for df, row in timeframe_data:
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name=f'Price'
                ),
                row=row, col=1
            )
            
            # Add volume
            if 'Volume' in df.columns:
                colors = [self.config.color_scheme['volume_up'] if c >= o 
                         else self.config.color_scheme['volume_down']
                         for c, o in zip(df['Close'], df['Open'])]
                
                fig.add_trace(
                    go.Bar(
                        x=df.index,
                        y=df['Volume'],
                        marker_color=colors,
                        name='Volume',
                        opacity=0.5
                    ),
                    row=row, col=1
                )
        
        fig.update_layout(
            height=self.config.default_height * 1.5,
            width=self.config.default_width,
            template=self.config.theme,
            showlegend=False
        )
        
        return fig
    
    def create_pattern_reliability_chart(self, lookback_window: int = 100) -> go.Figure:
        """
        Create a visualization of pattern reliability metrics
        
        Args:
            lookback_window (int): Historical window for reliability calculation
            
        Returns:
            go.Figure: Plotly figure with reliability metrics
        """
        pattern_methods = self._get_all_pattern_methods()
        reliability_data = {}
        
        for pattern_name, pattern_func in pattern_methods.items():
            try:
                pattern_reliability = self.patterns.detect_pattern_reliability(
                    self.df, pattern_func, lookback_window
                )
                if not pattern_reliability.empty:
                    reliability_data[pattern_name] = {
                        'success_rate': pattern_reliability['success_rate'].mean(),
                        'avg_return': pattern_reliability['avg_return'].mean(),
                        'risk_reward': pattern_reliability['risk_reward'].mean()
                    }
            except Exception as e:
                print(f"Error calculating reliability for {pattern_name}: {str(e)}")
        
        # Create subplots for different metrics
        fig = make_subplots(rows=3, cols=1,
                           subplot_titles=('Success Rate', 'Average Return', 'Risk-Reward Ratio'))
        
        patterns = list(reliability_data.keys())
        
        # Success Rate
        fig.add_trace(
            go.Bar(
                x=patterns,
                y=[data['success_rate'] for data in reliability_data.values()],
                name='Success Rate',
                marker_color=self.config.color_scheme['neutral']
            ),
            row=1, col=1
        )
        
        # Average Return
        returns = [data['avg_return'] for data in reliability_data.values()]
        colors = [self.config.color_scheme['bullish'] if r > 0 else 
                 self.config.color_scheme['bearish'] for r in returns]
        fig.add_trace(
            go.Bar(
                x=patterns,
                y=returns,
                name='Avg Return',
                marker_color=colors
            ),
            row=2, col=1
        )
        
        # Risk-Reward Ratio
        fig.add_trace(
            go.Bar(
                x=patterns,
                y=[data['risk_reward'] for data in reliability_data.values()],
                name='Risk-Reward',
                marker_color=self.config.color_scheme['complex']
            ),
            row=3, col=1
        )
        
        fig.update_layout(
            height=self.config.default_height * 1.5,
            width=self.config.default_width,
            template=self.config.theme,
            showlegend=False
        )
        
        return fig

    def create_interactive_dashboard(self) -> go.Figure:
        """
        Create an interactive dashboard with pattern filtering
        
        Returns:
            go.Figure: Plotly figure with interactive controls
        """
        # Create base figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Price Action', 'Pattern Distribution', 
                          'Pattern Reliability', 'Volume Analysis'),
            specs=[[{"colspan": 2}, None],
                  [{}, {}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=self.df.index,
                open=self.df['Open'],
                high=self.df['High'],
                low=self.df['Low'],
                close=self.df['Close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Add pattern distribution
        pattern_counts = self._get_pattern_distribution()
        fig.add_trace(
            go.Pie(
                labels=pattern_counts.index,
                values=pattern_counts.values,
                name='Pattern Distribution'
            ),
            row=2, col=1
        )
        
        # Add volume analysis
        if 'Volume' in self.df.columns:
            fig.add_trace(
                go.Bar(
                    x=self.df.index,
                    y=self.df['Volume'],
                    name='Volume',
                    marker_color=self._get_volume_colors()
                ),
                row=2, col=2
            )
        
        # Add buttons for pattern filtering
        pattern_buttons = self._create_pattern_buttons()
        
        fig.update_layout(
            updatemenus=[{
                'buttons': pattern_buttons,
                'direction': 'down',
                'showactive': True,
                'x': 0.1,
                'y': 1.1
            }],
            height=self.config.default_height * 1.5,
            width=self.config.default_width,
            template=self.config.theme
        )
        
        return fig

    def _get_pattern_distribution(self) -> pd.Series:
        """Calculate pattern distribution"""
        pattern_methods = self._get_all_pattern_methods()
        pattern_counts = {}
        
        for pattern_name, pattern_func in pattern_methods.items():
            try:
                result = pattern_func(self.df)
                if isinstance(result, tuple):
                    for sub_pattern, sub_name in zip(result, ['bullish', 'bearish']):
                        pattern_counts[f"{pattern_name}_{sub_name}"] = sub_pattern.sum()
                else:
                    pattern_counts[pattern_name] = result.sum()
            except Exception as e:
                print(f"Error counting pattern {pattern_name}: {str(e)}")
                
        return pd.Series(pattern_counts)

    def _get_volume_colors(self) -> List[str]:
        """Generate volume bar colors based on price movement"""
        return [self.config.color_scheme['volume_up'] if c >= o 
                else self.config.color_scheme['volume_down']
                for c, o in zip(self.df['Close'], self.df['Open'])]

    def _create_pattern_buttons(self) -> List[dict]:
        """Create buttons for pattern filtering"""
        pattern_methods = self._get_all_pattern_methods()
        buttons = []
        
        # Add 'All Patterns' button
        buttons.append({
            'label': 'All Patterns',
            'method': 'update',
            'args': [{'visible': [True] * len(pattern_methods)}]
        })
        
        # Add individual pattern buttons
        for i, pattern_name in enumerate(pattern_methods.keys()):
            visible = [False] * len(pattern_methods)
            visible[i] = True
            buttons.append({
                'label': pattern_name,
                'method': 'update',
                'args': [{'visible': visible}]
            })
            
        return buttons

    def add_pattern_correlation_analysis(self, fig: go.Figure) -> go.Figure:
        """
        Add pattern correlation analysis to existing figure
        
        Args:
            fig (go.Figure): Existing plotly figure
            
        Returns:
            go.Figure: Updated figure with correlation analysis
        """
        pattern_methods = self._get_all_pattern_methods()
        pattern_results = pd.DataFrame(index=self.df.index)
        
        # Collect pattern results
        for pattern_name, pattern_func in pattern_methods.items():
            try:
                result = pattern_func(self.df)
                if isinstance(result, tuple):
                    for sub_pattern, sub_name in zip(result, ['bullish', 'bearish']):
                        pattern_results[f"{pattern_name}_{sub_name}"] = sub_pattern
                else:
                    pattern_results[pattern_name] = result
            except Exception as e:
                print(f"Error analyzing pattern {pattern_name}: {str(e)}")
        
        # Calculate correlation matrix
        corr_matrix = pattern_results.corr()
        
        # Add correlation heatmap
        fig.add_trace(
            go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.index,
                y=corr_matrix.columns,
                colorscale='RdBu',
                name='Pattern Correlation'
            )
        )
        
        return fig
    
    def detect_market_regime(self, window: int = 20) -> pd.DataFrame:
        """
        Detect market regime using multiple indicators
        
        Args:
            window (int): Lookback window for calculations
            
        Returns:
            pd.DataFrame: Market regime indicators
        """
        regimes = pd.DataFrame(index=self.df.index)
        
        # Calculate volatility regime
        returns = np.log(self.df['Close'] / self.df['Close'].shift(1))
        volatility = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
        regimes['volatility_regime'] = pd.qcut(volatility, q=3, labels=['Low', 'Medium', 'High'])
        
        # Calculate trend regime using multiple indicators
        sma_short = self.df['Close'].rolling(window=20).mean()
        sma_long = self.df['Close'].rolling(window=50).mean()
        regimes['trend_regime'] = np.where(sma_short > sma_long, 'Uptrend',
                                         np.where(sma_short < sma_long, 'Downtrend', 'Sideways'))
        
        # Calculate volume regime
        volume_ma = self.df['Volume'].rolling(window=window).mean()
        regimes['volume_regime'] = pd.qcut(volume_ma, q=3, labels=['Low', 'Medium', 'High'])
        
        # Momentum regime
        momentum = self.df['Close'].pct_change(window)
        regimes['momentum_regime'] = pd.qcut(momentum, q=3, labels=['Weak', 'Neutral', 'Strong'])
        
        # Combined regime
        def get_combined_regime(row):
            if row['trend_regime'] == 'Uptrend' and row['volatility_regime'] == 'Low':
                return 'Strong_Bull'
            elif row['trend_regime'] == 'Downtrend' and row['volatility_regime'] == 'High':
                return 'Strong_Bear'
            elif row['trend_regime'] == 'Sideways' and row['volatility_regime'] == 'Low':
                return 'Range_Bound'
            else:
                return 'Mixed'
                
        regimes['combined_regime'] = regimes.apply(get_combined_regime, axis=1)
        
        return regimes

    def create_regime_visualization(self) -> go.Figure:
        """
        Create visualization of market regimes
        
        Returns:
            go.Figure: Plotly figure with regime analysis
        """
        regimes = self.detect_market_regime()
        
        fig = make_subplots(rows=3, cols=1,
                           subplot_titles=('Price with Regime', 'Volatility Regime', 'Volume Regime'),
                           shared_xaxes=True,
                           vertical_spacing=0.05)
        
        # Price chart with regime overlay
        fig.add_trace(
            go.Candlestick(
                x=self.df.index,
                open=self.df['Open'],
                high=self.df['High'],
                low=self.df['Low'],
                close=self.df['Close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Add regime background colors
        regime_colors = {
            'Strong_Bull': 'rgba(46, 204, 113, 0.2)',
            'Strong_Bear': 'rgba(231, 76, 60, 0.2)',
            'Range_Bound': 'rgba(241, 196, 15, 0.2)',
            'Mixed': 'rgba(149, 165, 166, 0.2)'
        }
        
        for regime in regime_colors:
            mask = regimes['combined_regime'] == regime
            if mask.any():
                fig.add_traces([
                    go.Scatter(
                        x=self.df.index,
                        y=[self.df['High'].max()] * len(self.df),
                        fill='tonexty',
                        mode='none',
                        name=regime,
                        fillcolor=regime_colors[regime],
                        showlegend=True
                    )
                ])
        
        # Volatility regime
        fig.add_trace(
            go.Scatter(
                x=self.df.index,
                y=regimes['volatility_regime'].map({'Low': 1, 'Medium': 2, 'High': 3}),
                name='Volatility Regime',
                line=dict(color='purple')
            ),
            row=2, col=1
        )
        
        # Volume regime
        fig.add_trace(
            go.Bar(
                x=self.df.index,
                y=self.df['Volume'],
                name='Volume',
                marker_color=regimes['volume_regime'].map({
                    'Low': 'lightblue',
                    'Medium': 'blue',
                    'High': 'darkblue'
                })
            ),
            row=3, col=1
        )
        
        fig.update_layout(
            height=self.config.default_height * 1.5,
            width=self.config.default_width,
            template=self.config.theme
        )
        
        return fig

    def add_price_action_confirmation(self) -> pd.DataFrame:
        """
        Add price action confirmation metrics
        
        Returns:
            pd.DataFrame: Price action confirmation indicators
        """
        confirmation = pd.DataFrame(index=self.df.index)
        
        # Price momentum
        confirmation['RSI'] = self._calculate_rsi(self.df['Close'])
        confirmation['MACD'] = self._calculate_macd(self.df['Close'])
        
        # Volume confirmation
        confirmation['Volume_MA_Ratio'] = self.df['Volume'] / self.df['Volume'].rolling(20).mean()
        
        # Volatility confirmation
        atr = self._calculate_atr(self.df)
        confirmation['ATR_Ratio'] = atr / atr.rolling(20).mean()
        
        return confirmation

    def add_pattern_significance(self, confidence_level: float = 0.95) -> pd.DataFrame:
        """
        Calculate statistical significance of pattern occurrences
        
        Args:
            confidence_level (float): Statistical confidence level
            
        Returns:
            pd.DataFrame: Pattern significance metrics
        """
        pattern_methods = self._get_all_pattern_methods()
        significance = pd.DataFrame(index=pattern_methods.keys())
        
        for pattern_name, pattern_func in pattern_methods.items():
            try:
                # Get pattern occurrences
                result = pattern_func(self.df)
                if isinstance(result, tuple):
                    for i, (sub_pattern, sub_name) in enumerate(zip(result, ['bullish', 'bearish'])):
                        # Calculate forward returns
                        forward_returns = self.df['Close'].pct_change(5).shift(-5)
                        pattern_returns = forward_returns[sub_pattern]
                        non_pattern_returns = forward_returns[~sub_pattern]
                        
                        # Perform t-test
                        t_stat, p_value = stats.ttest_ind(pattern_returns.dropna(), 
                                                        non_pattern_returns.dropna())
                        
                        significance.loc[f"{pattern_name}_{sub_name}", 'p_value'] = p_value
                        significance.loc[f"{pattern_name}_{sub_name}", 'significant'] = p_value < (1 - confidence_level)
            except Exception as e:
                print(f"Error calculating significance for {pattern_name}: {str(e)}")
        
        return significance

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices: pd.Series) -> pd.Series:
        """Calculate MACD"""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        return exp1 - exp2

    def analyze_pattern_sequences(self, lookback: int = 5) -> pd.DataFrame:
        """
        Analyze pattern sequences and their predictive power
        
        Args:
            lookback (int): Number of patterns to look back
            
        Returns:
            pd.DataFrame: Pattern sequence analysis
        """
        pattern_methods = self._get_all_pattern_methods()
        sequences = pd.DataFrame(index=self.df.index)
        
        # Generate pattern sequences
        for pattern_name, pattern_func in pattern_methods.items():
            try:
                result = pattern_func(self.df)
                if isinstance(result, tuple):
                    for sub_pattern, sub_name in zip(result, ['bullish', 'bearish']):
                        pattern_key = f"{pattern_name}_{sub_name}"
                        sequences[pattern_key] = sub_pattern.astype(int)
                else:
                    sequences[pattern_name] = result.astype(int)
            except Exception as e:
                print(f"Error analyzing sequence for {pattern_name}: {str(e)}")
        
        # Calculate sequence probabilities
        sequence_probs = {}
        for i in range(len(sequences) - lookback):
            current_seq = tuple(sequences.iloc[i:i+lookback].values.flatten())
            next_move = np.sign(self.df['Close'].iloc[i+lookback+1] - 
                              self.df['Close'].iloc[i+lookback])
            
            if current_seq not in sequence_probs:
                sequence_probs[current_seq] = {'up': 0, 'down': 0, 'total': 0}
            
            sequence_probs[current_seq]['total'] += 1
            if next_move > 0:
                sequence_probs[current_seq]['up'] += 1
            elif next_move < 0:
                sequence_probs[current_seq]['down'] += 1
        
        return pd.DataFrame(sequence_probs).T

    def overlay_custom_indicators(self, fig: go.Figure, 
                                indicators: Dict[str, Callable]) -> go.Figure:
        """
        Overlay custom technical indicators
        
        Args:
            fig (go.Figure): Existing plotly figure
            indicators (Dict[str, Callable]): Dictionary of indicator functions
            
        Returns:
            go.Figure: Updated figure with indicators
        """
        for name, func in indicators.items():
            try:
                indicator_values = func(self.df)
                fig.add_trace(
                    go.Scatter(
                        x=self.df.index,
                        y=indicator_values,
                        name=name,
                        line=dict(width=1)
                    )
                )
            except Exception as e:
                print(f"Error adding indicator {name}: {str(e)}")
        
        return fig
