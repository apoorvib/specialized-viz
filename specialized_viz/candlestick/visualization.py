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
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

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
            
        Raises:
            ValueError: If required columns are missing or data types are invalid
        """
        required_columns = ['Open', 'High', 'Low', 'Close']
        optional_columns = ['Volume']
        
        # Validate required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Validate data types and values
        for col in required_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"Column {col} must contain numeric data")
                
        # Validate price relationships
        invalid_prices = (
            (df['High'] < df['Low']) |
            (df['High'] < df['Open']) |
            (df['High'] < df['Close']) |
            (df['Low'] > df['Open']) |
            (df['Low'] > df['Close'])
        )
        if invalid_prices.any():
            invalid_dates = df.index[invalid_prices].tolist()
            raise ValueError(f"Invalid price relationships found at dates: {invalid_dates}")
        
        # Create a copy of the dataframe to prevent modifications
        self.df = df.copy()
        self.config = config or VisualizationConfig()
        self._cached_pattern_results = {}
        self._pattern_lock = threading.Lock()  # Add thread safety     
        
    def create_candlestick_chart(self, use_plotly: bool = False, **kwargs) -> Union[go.Figure, None]:
        """
        Create a candlestick chart using either Plotly or Matplotlib
        
        Args:
            use_plotly (bool): If True, uses Plotly. If False, uses Matplotlib
            **kwargs: Additional arguments for plotting (figsize, title, etc.)
            
        Returns:
            go.Figure if use_plotly=True, None otherwise
        """
        if use_plotly:
            return self._create_plotly_chart(**kwargs)
        else:
            return self._create_matplotlib_chart(**kwargs)

    def _create_plotly_chart(self, show_volume: bool = True, title: str = 'Candlestick Chart') -> go.Figure:
        """Create interactive Plotly candlestick chart"""
        fig = go.Figure(data=[go.Candlestick(
            x=self.df.index,
            open=self.df['Open'],
            high=self.df['High'],
            low=self.df['Low'],
            close=self.df['Close'],
            name='OHLC'
        )])
        
        if show_volume and 'Volume' in self.df.columns:
            fig.add_trace(go.Bar(
                x=self.df.index,
                y=self.df['Volume'],
                name='Volume',
                yaxis='y2'
            ))
            
            fig.update_layout(
                yaxis2=dict(
                    title='Volume',
                    overlaying='y',
                    side='right'
                )
            )
        
        fig.update_layout(
            title=title,
            yaxis_title='Price',
            xaxis_title='Date',
            template='plotly_white'
        )
        
        return fig
    
    def _ensure_datetime_index(self) -> pd.DataFrame:
        """
        Ensure DataFrame has a datetime index without modifying original data
        
        Returns:
            pd.DataFrame: DataFrame with datetime index
        """
        df_copy = self.df.copy()
        
        if not isinstance(df_copy.index, pd.DatetimeIndex):
            try:
                # Try basic conversion first
                df_copy.index = pd.to_datetime(df_copy.index)
            except (ValueError, TypeError):
                try:
                    # Try with UTC for timestamps
                    df_copy.index = pd.to_datetime(df_copy.index, utc=True)
                except Exception as e:
                    raise ValueError(f"Unable to convert index to datetime: {str(e)}")
        
        return df_copy

    def _create_matplotlib_chart(self,
                            bollinger_bands: Optional[Dict[str, pd.Series]] = None,
                            pivot_points: Optional[List[float]] = None,
                            title: str = 'OHLC Candles with Indicators',
                            figsize: Tuple[int, int] = (14, 7)) -> Figure:
        """
        Create a static Matplotlib candlestick chart with optional indicators
        
        Args:
            bollinger_bands (Optional[Dict[str, pd.Series]]): Dictionary containing 
                'upper', 'middle', and 'lower' Bollinger Band Series
            pivot_points (Optional[List[float]]): List of pivot point price levels
            title (str): Chart title
            figsize (Tuple[int, int]): Figure dimensions (width, height)
        
        Returns:
            matplotlib.figure.Figure: The created figure
            
        Raises:
            ValueError: If date conversion fails or if invalid data is provided
        """
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)
        
        try:
            # Get datetime index without modifying original data
            df = self._ensure_datetime_index()
            
            # Convert to matplotlib dates for plotting
            dates_mdates = mdates.date2num(df.index.to_pydatetime())
            
            # Calculate candle width based on data frequency
            if len(df) > 1:
                avg_time_delta = (dates_mdates[-1] - dates_mdates[0]) / len(dates_mdates)
                candle_width = 0.6 * avg_time_delta
            else:
                candle_width = 0.6
            
            # Plot candlesticks
            for idx, (date, row) in enumerate(df.iterrows()):
                # Determine candle color
                color = self.config.color_scheme['bullish'] if row['Close'] >= row['Open'] else self.config.color_scheme['bearish']
                
                # Plot price range (high-low line)
                ax.vlines(dates_mdates[idx], row['Low'], row['High'], 
                        color=color, linewidth=1)
                
                # Plot candle body
                body_bottom = min(row['Open'], row['Close'])
                body_height = abs(row['Close'] - row['Open'])
                rect = Rectangle((dates_mdates[idx] - candle_width/2, body_bottom),
                            candle_width, body_height,
                            facecolor=color, edgecolor=color, alpha=0.8)
                ax.add_patch(rect)
            
            # Add Bollinger Bands if provided
            if bollinger_bands is not None:
                for band_name, band_data in bollinger_bands.items():
                    if band_name == 'upper':
                        ax.plot(dates_mdates, band_data.values, 
                            color='purple', linestyle='--', 
                            label='Upper BB', alpha=0.7)
                    elif band_name == 'middle':
                        ax.plot(dates_mdates, band_data.values, 
                            color='blue', linestyle='--', 
                            label='Middle BB', alpha=0.7)
                    elif band_name == 'lower':
                        ax.plot(dates_mdates, band_data.values, 
                            color='purple', linestyle='--', 
                            label='Lower BB', alpha=0.7)
                
                # Add band fill
                if all(k in bollinger_bands for k in ['upper', 'lower']):
                    ax.fill_between(dates_mdates, 
                                bollinger_bands['lower'].values,
                                bollinger_bands['upper'].values,
                                color='purple', alpha=0.1)
            
            # Add pivot points if provided
            if pivot_points:
                for level in pivot_points:
                    ax.axhline(y=level, color='blue', 
                            linestyle='--', linewidth=0.8, 
                            alpha=0.7, label=f'Pivot: {level:.2f}')
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.xticks(rotation=45, ha='right')
            
            # Format y-axis
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.2f}'))
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # Set labels and title
            ax.set_title(title, pad=20)
            ax.set_xlabel('Date', labelpad=10)
            ax.set_ylabel('Price', labelpad=10)
            
            # Add legend if we have any indicators
            if bollinger_bands is not None or pivot_points:
                ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1),
                        borderaxespad=0., frameon=True)
            
            # Adjust layout to prevent label cutoff
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            plt.close(fig)  # Clean up in case of error
            raise ValueError(f"Error creating matplotlib chart: {str(e)}")
            
        finally:
            # The caller is responsible for closing the figure after use
            pass
            
    def _cluster_patterns(self, window_size: int = 20) -> pd.DataFrame:
        """
        Cluster pattern occurrences to identify high-density regions
        
        Args:
            window_size (int): Rolling window size for pattern density
            
        Returns:
            pd.DataFrame: Clustered pattern data with density metrics
            
        Raises:
            ValueError: If pattern detection fails or clustering cannot be performed
        """
        pattern_methods = self._get_all_pattern_methods()
        pattern_density = pd.DataFrame(index=self.df.index)
        
        excluded_patterns = {
            'breakout_patterns', 'harmonic_patterns', 'multi_timeframe_patterns',
            'pattern_combinations', 'pattern_reliability', 'volatility_adjusted_patterns'
        }
        
        for pattern_name, pattern_func in pattern_methods.items():
            if pattern_name in excluded_patterns:
                continue
                
            try:
                # Get pattern results from cache or calculate new
                cached_result = self._get_pattern_results(pattern_name)
                if cached_result is not None:
                    result = cached_result
                else:
                    result = pattern_func(self.df)
                    self._set_pattern_results(pattern_name, result)
                
                if isinstance(result, tuple):
                    # Handle bullish/bearish patterns
                    bullish, bearish = self._validate_pattern_signals(result)
                    pattern_density[f"{pattern_name}_bullish"] = self._calculate_density(bullish, window_size)
                    pattern_density[f"{pattern_name}_bearish"] = self._calculate_density(bearish, window_size)
                else:
                    # Handle single signal patterns
                    signal = self._validate_pattern_signals(result)
                    pattern_density[pattern_name] = self._calculate_density(signal, window_size)
                    
            except Exception as e:
                print(f"Error processing pattern {pattern_name}: {str(e)}")
                continue
        
        # Remove columns with no signals
        pattern_density = pattern_density.loc[:, (pattern_density != 0).any()]
        
        if pattern_density.empty:
            return pd.DataFrame({'cluster': [-1]}, index=self.df.index)
        
        # Perform clustering
        try:
            normalized_density = (pattern_density - pattern_density.mean()) / pattern_density.std()
            clustering = DBSCAN(eps=0.5, min_samples=3)
            pattern_density['cluster'] = clustering.fit_predict(normalized_density.fillna(0))
        except Exception as e:
            print(f"Clustering failed: {str(e)}")
            pattern_density['cluster'] = -1
        
        return pattern_density

    def _calculate_density(self, signal: pd.Series, window: int) -> pd.Series:
        """
        Calculate pattern density using rolling window
        
        Args:
            signal (pd.Series): Pattern signal
            window (int): Rolling window size
            
        Returns:
            pd.Series: Pattern density
        """
        return signal.rolling(window=window, min_periods=1).sum()


    def _validate_pattern_signals(self, signals: Union[pd.Series, pd.DataFrame, Tuple]) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
        """
        Validate and convert pattern signals to proper format
        
        Args:
            signals: Pattern detection signals
            
        Returns:
            Validated and converted signals
            
        Raises:
            ValueError: If signals are in invalid format
        """
        if isinstance(signals, tuple):
            bullish, bearish = signals
            return (
                self._convert_to_series(bullish, 'bullish'),
                self._convert_to_series(bearish, 'bearish')
            )
        else:
            return self._convert_to_series(signals, 'signal')

    def _convert_to_series(self, data: Union[pd.Series, pd.DataFrame], name: str) -> pd.Series:
        """
        Convert pattern data to Series format
        
        Args:
            data: Pattern data
            name: Name for the series
            
        Returns:
            pd.Series: Converted and validated data
            
        Raises:
            ValueError: If data cannot be converted to Series
        """
        if isinstance(data, pd.DataFrame):
            if data.empty:
                return pd.Series(0, index=self.df.index, name=name)
            return data.iloc[:, 0].astype(float)
        elif isinstance(data, pd.Series):
            return data.astype(float)
        else:
            raise ValueError(f"Invalid data format for {name}")


    def _get_pattern_results(self, pattern_name: str) -> Optional[Union[pd.Series, Tuple[pd.Series, pd.Series]]]:
        """
        Get cached pattern results in a thread-safe manner
        
        Args:
            pattern_name (str): Name of the pattern to retrieve
            
        Returns:
            Optional[Union[pd.Series, Tuple[pd.Series, pd.Series]]]: Cached pattern results
        """
        with self._pattern_lock:
            return self._cached_pattern_results.get(pattern_name)

    def _set_pattern_results(self, pattern_name: str, results: Union[pd.Series, Tuple[pd.Series, pd.Series]]) -> None:
        """
        Cache pattern results in a thread-safe manner
        
        Args:
            pattern_name (str): Name of the pattern to cache
            results: Pattern detection results
        """
        with self._pattern_lock:
            self._cached_pattern_results[pattern_name] = results
            
    def _clear_pattern_cache(self) -> None:
        """Clear the pattern cache"""
        with self._pattern_lock:
            self._cached_pattern_results.clear()
        
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
        Create visualization of pattern reliability metrics
        
        Args:
            lookback_window (int): Historical window for reliability calculation
                
        Returns:
            go.Figure: Plotly figure with reliability metrics
        """
        pattern_methods = self._get_all_pattern_methods()
        reliability_data = {}
        
        # Filter out patterns that require additional arguments or return DataFrames
        patterns_to_exclude = {
            'breakout_patterns', 'harmonic_patterns', 'multi_timeframe_patterns',
            'pattern_combinations', 'pattern_reliability', 'volatility_adjusted_patterns'
        }
        
        for pattern_name, pattern_func in pattern_methods.items():
            if pattern_name in patterns_to_exclude:
                continue
                
            try:
                # Get pattern signals
                result = pattern_func(self.df)
                # Calculate future returns once
                future_returns = self.df['Close'].pct_change().shift(-1)
                
                if isinstance(result, tuple):
                    # Handle patterns that return bullish/bearish signals
                    bullish, bearish = result
                    
                    # Convert to Series if needed and handle boolean conversion
                    if isinstance(bullish, pd.DataFrame):
                        bullish = bullish.iloc[:, 0]
                    if isinstance(bearish, pd.DataFrame):
                        bearish = bearish.iloc[:, 0]
                    
                    # Convert to numeric and handle NaN values
                    bullish = pd.Series(bullish).astype(float)
                    bearish = pd.Series(bearish).astype(float)
                    
                    # Calculate metrics for bullish patterns
                    bullish_mask = bullish > 0
                    if bullish_mask.any():
                        reliability_data[f"{pattern_name}_bullish"] = {
                            'success_rate': (future_returns[bullish_mask] > 0).mean() * 100,
                            'avg_return': future_returns[bullish_mask].mean() * 100,
                            'occurrences': bullish_mask.sum(),
                            'risk_reward': abs(future_returns[bullish_mask].mean() / 
                                            future_returns[bullish_mask].std()) if future_returns[bullish_mask].std() != 0 else 0
                        }
                    
                    # Calculate metrics for bearish patterns
                    bearish_mask = bearish > 0
                    if bearish_mask.any():
                        reliability_data[f"{pattern_name}_bearish"] = {
                            'success_rate': (future_returns[bearish_mask] < 0).mean() * 100,
                            'avg_return': future_returns[bearish_mask].mean() * -100,
                            'occurrences': bearish_mask.sum(),
                            'risk_reward': abs(future_returns[bearish_mask].mean() / 
                                            future_returns[bearish_mask].std()) if future_returns[bearish_mask].std() != 0 else 0
                        }
                else:
                    # Handle single signal patterns
                    if isinstance(result, pd.DataFrame):
                        result = result.iloc[:, 0]
                    
                    # Convert to numeric and handle NaN values
                    result = pd.Series(result).astype(float)
                    pattern_mask = result > 0
                    
                    if pattern_mask.any():
                        reliability_data[pattern_name] = {
                            'success_rate': (future_returns[pattern_mask] > 0).mean() * 100,
                            'avg_return': future_returns[pattern_mask].mean() * 100,
                            'occurrences': pattern_mask.sum(),
                            'risk_reward': abs(future_returns[pattern_mask].mean() / 
                                            future_returns[pattern_mask].std()) if future_returns[pattern_mask].std() != 0 else 0
                        }
            except Exception as e:
                print(f"Error calculating reliability for {pattern_name}: {str(e)}")
                continue

        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Success Rate (%)', 'Average Return (%)', 
                        'Number of Occurrences', 'Risk-Reward Ratio'),
            vertical_spacing=0.12
        )
        
        # Prepare data for plotting
        patterns = list(reliability_data.keys())
        metrics = {
            'success_rate': [],
            'avg_return': [],
            'occurrences': [],
            'risk_reward': []
        }
        
        for pattern in patterns:
            for metric in metrics:
                metrics[metric].append(reliability_data[pattern][metric])
        
        # Add traces for each metric
        fig.add_trace(
            go.Bar(x=patterns, y=metrics['success_rate'], 
                name='Success Rate',
                marker_color=self.config.color_scheme['neutral']),
            row=1, col=1
        )
        
        colors = [self.config.color_scheme['bullish'] if r > 0 else 
                self.config.color_scheme['bearish'] for r in metrics['avg_return']]
        fig.add_trace(
            go.Bar(x=patterns, y=metrics['avg_return'],
                name='Avg Return',
                marker_color=colors),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(x=patterns, y=metrics['occurrences'],
                name='Occurrences',
                marker_color=self.config.color_scheme['neutral']),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(x=patterns, y=metrics['risk_reward'],
                name='Risk-Reward',
                marker_color=self.config.color_scheme['complex']),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=self.config.default_height * 1.5,
            width=self.config.default_width,
            template=self.config.theme,
            showlegend=False,
            title_text="Pattern Reliability Analysis",
            title_x=0.5
        )
        
        # Update axes
        fig.update_xaxes(tickangle=45, tickfont=dict(size=10))
        fig.update_yaxes(title_text="Percentage", row=1, col=1)
        fig.update_yaxes(title_text="Percentage", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        fig.update_yaxes(title_text="Ratio", row=2, col=2)
        
        return fig
    def create_interactive_dashboard(self) -> go.Figure:
        """Create interactive dashboard with pattern filtering"""
        # Create subplot structure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Price Action with Patterns', 'Pattern Distribution', 
                        'Pattern Occurrences', 'Volume Profile'),
            specs=[[{"colspan": 2}, None],
                [{"type": "pie"}, {"type": "xy"}]],
            vertical_spacing=0.15,
            horizontal_spacing=0.15
        )
        
        # Add candlestick chart (always visible)
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
        if not pattern_counts.empty:
            top_patterns = pattern_counts.sort_values(ascending=False).head(10)
            fig.add_trace(
                go.Pie(
                    labels=top_patterns.index,
                    values=top_patterns.values,
                    name='Pattern Distribution',
                    textinfo='label+percent',
                    textposition='outside',
                    hole=0.4,
                    showlegend=False,
                    pull=[0.1 if i == 0 else 0 for i in range(len(top_patterns))],
                    marker=dict(colors=[
                        self.config.color_scheme['bullish'] if 'bullish' in label else
                        self.config.color_scheme['bearish'] if 'bearish' in label else
                        self.config.color_scheme['neutral'] for label in top_patterns.index
                    ])
                ),
                row=2, col=1
            )
        
        # Add volume profile
        if 'Volume' in self.df.columns:
            fig.add_trace(
                go.Bar(
                    x=self.df.index,
                    y=self.df['Volume'],
                    name='Volume',
                    marker_color=self._get_volume_colors(),
                    opacity=0.7
                ),
                row=2, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=self.df.index,
                    y=self.df['Volume'].rolling(window=20).mean(),
                    name='Volume MA (20)',
                    line=dict(color='rgba(0,0,0,0.5)', width=2)
                ),
                row=2, col=2
            )

        # Store base traces count
        n_base_traces = len(fig.data)
        
        # Get pattern methods
        pattern_methods = {k: v for k, v in self._get_all_pattern_methods().items()
                        if k not in {'breakout_patterns', 'harmonic_patterns', 
                                    'multi_timeframe_patterns', 'pattern_combinations', 
                                    'pattern_reliability', 'volatility_adjusted_patterns'}}
        
        # Add traces for each pattern (initially visible)
        pattern_traces = {}
        for pattern_name, pattern_func in pattern_methods.items():
            try:
                result = pattern_func(self.df)
                if isinstance(result, tuple):
                    bullish, bearish = result
                    pattern_traces[pattern_name] = []
                    
                    # Add bullish markers
                    if pd.Series(bullish).any():
                        trace_idx = len(fig.data)
                        fig.add_trace(
                            go.Scatter(
                                x=self.df.index[bullish],
                                y=self.df['Low'][bullish] * 0.99,
                                mode='markers',
                                marker=dict(
                                    symbol='triangle-up',
                                    size=10,
                                    color=self.config.color_scheme['bullish']
                                ),
                                name=f'{pattern_name} (Bullish)',
                                visible=True  # Initially visible
                            ),
                            row=1, col=1
                        )
                        pattern_traces[pattern_name].append(trace_idx)
                    
                    # Add bearish markers
                    if pd.Series(bearish).any():
                        trace_idx = len(fig.data)
                        fig.add_trace(
                            go.Scatter(
                                x=self.df.index[bearish],
                                y=self.df['High'][bearish] * 1.01,
                                mode='markers',
                                marker=dict(
                                    symbol='triangle-down',
                                    size=10,
                                    color=self.config.color_scheme['bearish']
                                ),
                                name=f'{pattern_name} (Bearish)',
                                visible=True  # Initially visible
                            ),
                            row=1, col=1
                        )
                        pattern_traces[pattern_name].append(trace_idx)
                else:
                    if pd.Series(result).any():
                        trace_idx = len(fig.data)
                        fig.add_trace(
                            go.Scatter(
                                x=self.df.index[result],
                                y=self.df['Low'][result] * 0.99,
                                mode='markers',
                                marker=dict(
                                    symbol='circle',
                                    size=8,
                                    color=self.config.color_scheme['neutral']
                                ),
                                name=pattern_name,
                                visible=True  # Initially visible
                            ),
                            row=1, col=1
                        )
                        pattern_traces[pattern_name] = [trace_idx]
            except Exception as e:
                print(f"Error adding pattern {pattern_name}: {str(e)}")
        
        # Create buttons
        buttons = [{
            'label': 'All Patterns',
            'method': 'update',
            'args': [{
                'visible': [True] * len(fig.data)  # All traces visible
            }]
        }]
        
        # Add pattern-specific buttons
        for pattern_name, trace_indices in pattern_traces.items():
            if trace_indices:  # Only add button if pattern has traces
                visibility = [i < n_base_traces or i in trace_indices for i in range(len(fig.data))]
                buttons.append({
                    'label': pattern_name.replace('_', ' ').title(),
                    'method': 'update',
                    'args': [{'visible': visibility}]
                })
        
        # Update layout
        fig.update_layout(
            height=self.config.default_height * 1.5,
            width=self.config.default_width,
            template=self.config.theme,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=10)
            ),
            margin=dict(t=150, b=50, l=50, r=50),
            updatemenus=[{
                'buttons': buttons,
                'direction': 'down',
                'showactive': True,
                'x': 0.1,
                'y': 1.15,
                'xanchor': 'left',
                'yanchor': 'top',
                'font': {'size': 12}
            }],
            title=dict(
                text="Interactive Pattern Analysis Dashboard",
                y=0.95,
                x=0.5,
                xanchor='center',
                yanchor='top',
                font=dict(size=16)
            )
        )
        
        # Update axes
        fig.update_xaxes(title_text="Date", rangeslider_visible=False, row=1, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=2)
        fig.update_yaxes(title_text="Volume", row=2, col=2)
        
        # Update subplot titles
        for annotation in fig.layout.annotations:
            annotation.font.size = 14
            annotation.font.color = self.config.color_scheme['text']
            annotation.y = annotation.y - 0.02
        
        return fig
    def _get_pattern_distribution(self) -> pd.Series:
        """Calculate pattern distribution"""
        pattern_methods = self._get_all_pattern_methods()
        pattern_counts = {}
        
        # Patterns to exclude (require special handling or additional arguments)
        patterns_to_exclude = {
            'multi_timeframe_patterns',  # Requires df_weekly and df_monthly
            'pattern_combinations',      # Special return type
            'pattern_reliability',       # Requires pattern_func argument
            'breakout_patterns',         # Returns DataFrame
            'harmonic_patterns',         # Returns DataFrame
            'volatility_adjusted_patterns'  # Returns DataFrame
        }
        
        for pattern_name, pattern_func in pattern_methods.items():
            if pattern_name in patterns_to_exclude:
                continue
                
            try:
                result = pattern_func(self.df)
                if isinstance(result, tuple):
                    for sub_pattern, sub_name in zip(result, ['bullish', 'bearish']):
                        if isinstance(sub_pattern, pd.DataFrame):
                            sub_pattern = sub_pattern.iloc[:, 0]
                        pattern_counts[f"{pattern_name}_{sub_name}"] = sub_pattern.astype(int).sum()
                else:
                    if isinstance(result, pd.DataFrame):
                        result = result.iloc[:, 0]
                    pattern_counts[pattern_name] = result.astype(int).sum()
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
        
        # First trace is candlestick, last trace is volume
        base_visibility = [True]  # Candlestick always visible
        if 'Volume' in self.df.columns:
            base_visibility.append(True)  # Volume always visible
        
        # Add 'All Patterns' button
        buttons.append({
            'label': 'All Patterns',
            'method': 'update',
            'args': [{
                'visible': base_visibility
            }]
        })
        
        # Add individual pattern buttons
        for pattern_name, pattern_func in pattern_methods.items():
            try:
                # Get pattern signals
                result = pattern_func(self.df)
                if isinstance(result, tuple):
                    # For patterns that return bullish/bearish signals
                    bullish, bearish = result
                    visibility = base_visibility.copy()  # Start with base visibility
                    
                    # Add trace for this pattern
                    fig_update = {
                        'visible': visibility,
                        'showlegend': True
                    }
                    
                    # Add candlestick trace
                    fig_update['candlestick'] = [{
                        'x': self.df.index,
                        'open': self.df['Open'],
                        'high': self.df['High'],
                        'low': self.df['Low'],
                        'close': self.df['Close'],
                        'type': 'candlestick',
                        'name': 'Price'
                    }]
                    
                    # Add pattern markers
                    if bullish.any():
                        fig_update['scatter'] = [{
                            'x': self.df.index[bullish],
                            'y': self.df['Low'][bullish],
                            'mode': 'markers',
                            'marker': {'symbol': 'triangle-up', 'size': 10, 'color': self.config.color_scheme['bullish']},
                            'name': f'{pattern_name} (Bullish)'
                        }]
                    
                    if bearish.any():
                        fig_update['scatter'].append({
                            'x': self.df.index[bearish],
                            'y': self.df['High'][bearish],
                            'mode': 'markers',
                            'marker': {'symbol': 'triangle-down', 'size': 10, 'color': self.config.color_scheme['bearish']},
                            'name': f'{pattern_name} (Bearish)'
                        })
                    
                    buttons.append({
                        'label': pattern_name,
                        'method': 'update',
                        'args': [fig_update]
                    })
                else:
                    # For single signal patterns
                    visibility = base_visibility.copy()
                    
                    fig_update = {
                        'visible': visibility,
                        'showlegend': True
                    }
                    
                    # Add candlestick trace
                    fig_update['candlestick'] = [{
                        'x': self.df.index,
                        'open': self.df['Open'],
                        'high': self.df['High'],
                        'low': self.df['Low'],
                        'close': self.df['Close'],
                        'type': 'candlestick',
                        'name': 'Price'
                    }]
                    
                    # Add pattern markers
                    if result.any():
                        fig_update['scatter'] = [{
                            'x': self.df.index[result],
                            'y': self.df['Low'][result],
                            'mode': 'markers',
                            'marker': {'symbol': 'circle', 'size': 10, 'color': self.config.color_scheme['neutral']},
                            'name': pattern_name
                        }]
                    
                    buttons.append({
                        'label': pattern_name,
                        'method': 'update',
                        'args': [fig_update]
                    })
                    
            except Exception as e:
                print(f"Error creating button for {pattern_name}: {str(e)}")
                continue
        
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
        """
        Calculate Relative Strength Index safely
        
        Args:
            prices (pd.Series): Price data
            period (int): RSI period
            
        Returns:
            pd.Series: RSI values
        """
        delta = prices.diff()
        
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = pd.Series(0, index=prices.index)  # Initialize with zeros
        valid_denominator = avg_loss != 0
        rs[valid_denominator] = avg_gain[valid_denominator] / avg_loss[valid_denominator]
        
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
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
        # pattern_methods = self._get_all_pattern_methods()
        # sequences = pd.DataFrame(index=self.df.index)
        
        # # Generate pattern sequences
        # for pattern_name, pattern_func in pattern_methods.items():
        #     try:
        #         result = pattern_func(self.df)
        #         if isinstance(result, tuple):
        #             for sub_pattern, sub_name in zip(result, ['bullish', 'bearish']):
        #                 pattern_key = f"{pattern_name}_{sub_name}"
        #                 sequences[pattern_key] = sub_pattern.astype(int)
        #         else:
        #             sequences[pattern_name] = result.astype(int)
        #     except Exception as e:
        #         print(f"Error analyzing sequence for {pattern_name}: {str(e)}")
        
        # # Calculate sequence probabilities
        # sequence_probs = {}
        # for i in range(len(sequences) - lookback):
        #     current_seq = tuple(sequences.iloc[i:i+lookback].values.flatten())
        #     next_move = np.sign(self.df['Close'].iloc[i+lookback+1] - 
        #                       self.df['Close'].iloc[i+lookback])
            
        #     if current_seq not in sequence_probs:
        #         sequence_probs[current_seq] = {'up': 0, 'down': 0, 'total': 0}
            
        #     sequence_probs[current_seq]['total'] += 1
        #     if next_move > 0:
        #         sequence_probs[current_seq]['up'] += 1
        #     elif next_move < 0:
        #         sequence_probs[current_seq]['down'] += 1
        
        # return pd.DataFrame(sequence_probs).T
            # Use vectorized operations for pattern detection
        pattern_methods = self._get_all_pattern_methods()
        for pattern_name, pattern_func in pattern_methods.items():
            try:
                results = pattern_func(self.df)
                if isinstance(results, tuple):
                    sequences[f"{pattern_name}_bullish"] = results[0].astype(int)
                    sequences[f"{pattern_name}_bearish"] = results[1].astype(int)
                else:
                    sequences[pattern_name] = results.astype(int)
            except Exception as e:
                print(f"Error in pattern {pattern_name}: {str(e)}")
                continue
        
        # Use rolling window for sequence generation
        sequence_data = []
        for i in range(lookback, len(sequences)):
            window = sequences.iloc[i-lookback:i]
            next_move = np.sign(self.df['Close'].iloc[i] - self.df['Close'].iloc[i-1])
            sequence_data.append({
                'sequence': tuple(window.values.flatten()),
                'next_move': next_move,
                'date': sequences.index[i]
            })
        
        return pd.DataFrame(sequence_data)

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
    
    def _get_all_pattern_methods(self) -> Dict[str, Callable]:
        """
        Get all pattern detection methods from CandlestickPatterns class
        
        Returns:
            Dict[str, Callable]: Dictionary of pattern names and their detection methods
        """
        pattern_methods = {}
        for name, method in inspect.getmembers(CandlestickPatterns, predicate=inspect.isfunction):
            if name.startswith('detect_'):
                pattern_methods[name.replace('detect_', '')] = method
        return pattern_methods
