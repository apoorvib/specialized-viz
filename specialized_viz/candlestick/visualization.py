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
    # def create_interactive_dashboard(self) -> go.Figure:
    #     """Create interactive dashboard with pattern filtering"""
    #     # Create subplot structure
    #     fig = make_subplots(
    #         rows=2, cols=2,
    #         subplot_titles=('Price Action with Patterns', 'Pattern Distribution', 
    #                     'Pattern Occurrences', 'Volume Profile'),
    #         specs=[[{"colspan": 2}, None],
    #             [{"type": "pie"}, {"type": "xy"}]],
    #         vertical_spacing=0.15,
    #         horizontal_spacing=0.15
    #     )
        
    #     # Add candlestick chart (always visible)
    #     fig.add_trace(
    #         go.Candlestick(
    #             x=self.df.index,
    #             open=self.df['Open'],
    #             high=self.df['High'],
    #             low=self.df['Low'],
    #             close=self.df['Close'],
    #             name='Price'
    #         ),
    #         row=1, col=1
    #     )
        
    #     # Add pattern distribution
    #     pattern_counts = self._get_pattern_distribution()
    #     if not pattern_counts.empty:
    #         top_patterns = pattern_counts.sort_values(ascending=False).head(10)
    #         fig.add_trace(
    #             go.Pie(
    #                 labels=top_patterns.index,
    #                 values=top_patterns.values,
    #                 name='Pattern Distribution',
    #                 textinfo='label+percent',
    #                 textposition='outside',
    #                 hole=0.4,
    #                 showlegend=False,
    #                 pull=[0.1 if i == 0 else 0 for i in range(len(top_patterns))],
    #                 marker=dict(colors=[
    #                     self.config.color_scheme['bullish'] if 'bullish' in label else
    #                     self.config.color_scheme['bearish'] if 'bearish' in label else
    #                     self.config.color_scheme['neutral'] for label in top_patterns.index
    #                 ])
    #             ),
    #             row=2, col=1
    #         )
        
    #     # Add volume profile
    #     if 'Volume' in self.df.columns:
    #         fig.add_trace(
    #             go.Bar(
    #                 x=self.df.index,
    #                 y=self.df['Volume'],
    #                 name='Volume',
    #                 marker_color=self._get_volume_colors(),
    #                 opacity=0.7
    #             ),
    #             row=2, col=2
    #         )
            
    #         fig.add_trace(
    #             go.Scatter(
    #                 x=self.df.index,
    #                 y=self.df['Volume'].rolling(window=20).mean(),
    #                 name='Volume MA (20)',
    #                 line=dict(color='rgba(0,0,0,0.5)', width=2)
    #             ),
    #             row=2, col=2
    #         )

    #     # Store base traces count
    #     n_base_traces = len(fig.data)
        
    #     # Get pattern methods
    #     pattern_methods = {k: v for k, v in self._get_all_pattern_methods().items()
    #                     if k not in {'breakout_patterns', 'harmonic_patterns', 
    #                                 'multi_timeframe_patterns', 'pattern_combinations', 
    #                                 'pattern_reliability', 'volatility_adjusted_patterns'}}
        
    #     # Add traces for each pattern (initially visible)
    #     pattern_traces = {}
    #     for pattern_name, pattern_func in pattern_methods.items():
    #         try:
    #             result = pattern_func(self.df)
    #             if isinstance(result, tuple):
    #                 bullish, bearish = result
    #                 pattern_traces[pattern_name] = []
                    
    #                 # Add bullish markers
    #                 if pd.Series(bullish).any():
    #                     trace_idx = len(fig.data)
    #                     fig.add_trace(
    #                         go.Scatter(
    #                             x=self.df.index[bullish],
    #                             y=self.df['Low'][bullish] * 0.99,
    #                             mode='markers',
    #                             marker=dict(
    #                                 symbol='triangle-up',
    #                                 size=10,
    #                                 color=self.config.color_scheme['bullish']
    #                             ),
    #                             name=f'{pattern_name} (Bullish)',
    #                             visible=True  # Initially visible
    #                         ),
    #                         row=1, col=1
    #                     )
    #                     pattern_traces[pattern_name].append(trace_idx)
                    
    #                 # Add bearish markers
    #                 if pd.Series(bearish).any():
    #                     trace_idx = len(fig.data)
    #                     fig.add_trace(
    #                         go.Scatter(
    #                             x=self.df.index[bearish],
    #                             y=self.df['High'][bearish] * 1.01,
    #                             mode='markers',
    #                             marker=dict(
    #                                 symbol='triangle-down',
    #                                 size=10,
    #                                 color=self.config.color_scheme['bearish']
    #                             ),
    #                             name=f'{pattern_name} (Bearish)',
    #                             visible=True  # Initially visible
    #                         ),
    #                         row=1, col=1
    #                     )
    #                     pattern_traces[pattern_name].append(trace_idx)
    #             else:
    #                 if pd.Series(result).any():
    #                     trace_idx = len(fig.data)
    #                     fig.add_trace(
    #                         go.Scatter(
    #                             x=self.df.index[result],
    #                             y=self.df['Low'][result] * 0.99,
    #                             mode='markers',
    #                             marker=dict(
    #                                 symbol='circle',
    #                                 size=8,
    #                                 color=self.config.color_scheme['neutral']
    #                             ),
    #                             name=pattern_name,
    #                             visible=True  # Initially visible
    #                         ),
    #                         row=1, col=1
    #                     )
    #                     pattern_traces[pattern_name] = [trace_idx]
    #         except Exception as e:
    #             print(f"Error adding pattern {pattern_name}: {str(e)}")
        
    #     # Create buttons
    #     buttons = [{
    #         'label': 'All Patterns',
    #         'method': 'update',
    #         'args': [{
    #             'visible': [True] * len(fig.data)  # All traces visible
    #         }]
    #     }]
        
    #     # Add pattern-specific buttons
    #     for pattern_name, trace_indices in pattern_traces.items():
    #         if trace_indices:  # Only add button if pattern has traces
    #             visibility = [i < n_base_traces or i in trace_indices for i in range(len(fig.data))]
    #             buttons.append({
    #                 'label': pattern_name.replace('_', ' ').title(),
    #                 'method': 'update',
    #                 'args': [{'visible': visibility}]
    #             })
        
    #     # Update layout
    #     fig.update_layout(
    #         height=self.config.default_height * 1.5,
    #         width=self.config.default_width,
    #         template=self.config.theme,
    #         showlegend=True,
    #         legend=dict(
    #             orientation="h",
    #             yanchor="bottom",
    #             y=1.02,
    #             xanchor="right",
    #             x=1,
    #             font=dict(size=10)
    #         ),
    #         margin=dict(t=150, b=50, l=50, r=50),
    #         updatemenus=[{
    #             'buttons': buttons,
    #             'direction': 'down',
    #             'showactive': True,
    #             'x': 0.1,
    #             'y': 1.15,
    #             'xanchor': 'left',
    #             'yanchor': 'top',
    #             'font': {'size': 12}
    #         }],
    #         title=dict(
    #             text="Interactive Pattern Analysis Dashboard",
    #             y=0.95,
    #             x=0.5,
    #             xanchor='center',
    #             yanchor='top',
    #             font=dict(size=16)
    #         )
    #     )
        
    #     # Update axes
    #     fig.update_xaxes(title_text="Date", rangeslider_visible=False, row=1, col=1)
    #     fig.update_yaxes(title_text="Price", row=1, col=1)
    #     fig.update_xaxes(title_text="Date", row=2, col=2)
    #     fig.update_yaxes(title_text="Volume", row=2, col=2)
        
    #     # Update subplot titles
    #     for annotation in fig.layout.annotations:
    #         annotation.font.size = 14
    #         annotation.font.color = self.config.color_scheme['text']
    #         annotation.y = annotation.y - 0.02
        
    #     return fig
    
    def create_interactive_dashboard(self) -> go.Figure:
        """
        Create an interactive dashboard with pattern filtering and analysis
        
        Returns:
            go.Figure: Interactive Plotly dashboard
            
        Raises:
            ValueError: If data preparation or visualization fails
        """
        # Initialize subplot structure
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Price Action with Patterns',
                'Pattern Distribution',
                'Volume Profile',
                'Pattern Reliability',
                'Market Regime',
                'Technical Indicators'
            ),
            specs=[
                [{"colspan": 2}, None],
                [{"type": "pie"}, {"type": "bar"}],
                [{"type": "heatmap"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )

        try:
            # Add main price chart
            self._add_price_chart(fig, row=1, col=1)
            
            # Add pattern distribution
            self._add_pattern_distribution(fig, row=2, col=1)
            
            # Add volume profile
            self._add_volume_profile(fig, row=2, col=2)
            
            # Add pattern correlation heatmap
            self._add_pattern_correlation(fig, row=3, col=1)
            
            # Add technical indicators
            self._add_technical_indicators(fig, row=3, col=2)
            
            # Create pattern filter buttons
            buttons = self._create_pattern_filter_buttons()
            
            # Update layout with controls and styling
            self._update_dashboard_layout(fig, buttons)
            
            return fig
            
        except Exception as e:
            raise ValueError(f"Failed to create dashboard: {str(e)}")

    def _create_pattern_filter_buttons(self) -> List[Dict]:
        """
        Create buttons for pattern filtering
        
        Returns:
            List[Dict]: List of button configurations
        """
        buttons = [{
            'label': 'All Patterns',
            'method': 'update',
            'args': [{
                'visible': [True] * len(self.fig.data)
            }]
        }]
        
        pattern_methods = self._get_safe_pattern_methods()
        base_traces = 2  # Candlestick and volume are always visible
        
        for pattern_name in pattern_methods:
            visibility = [i < base_traces for i in range(len(self.fig.data))]
            pattern_traces = self._get_pattern_trace_indices(pattern_name)
            
            for idx in pattern_traces:
                visibility[idx] = True
                
            buttons.append({
                'label': pattern_name.replace('_', ' ').title(),
                'method': 'update',
                'args': [{
                    'visible': visibility
                }]
            })
        
        return buttons

    def _get_pattern_trace_indices(self, pattern_name: str) -> List[int]:
        """
        Get trace indices for a specific pattern
        
        Args:
            pattern_name (str): Name of the pattern
            
        Returns:
            List[int]: List of trace indices
        """
        indices = []
        for i, trace in enumerate(self.fig.data):
            if trace.name and pattern_name in trace.name:
                indices.append(i)
        return indices



    def _add_price_chart(self, fig: go.Figure, row: int, col: int) -> None:
        """Add main price chart with patterns to dashboard"""
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=self.df.index,
                open=self.df['Open'],
                high=self.df['High'],
                low=self.df['Low'],
                close=self.df['Close'],
                name='Price',
                showlegend=True
            ),
            row=row, col=col
        )
        
        # Add detected patterns
        pattern_methods = self._get_safe_pattern_methods()
        for pattern_name, pattern_data in pattern_methods.items():
            try:
                signals = self._get_pattern_signals(pattern_name)
                if signals:
                    self._add_pattern_markers(fig, signals, pattern_name, row, col)
            except Exception as e:
                print(f"Error adding pattern {pattern_name}: {str(e)}")



    def _add_pattern_markers(self, 
                            fig: go.Figure, 
                            signals: Union[pd.Series, Tuple[pd.Series, pd.Series]], 
                            pattern_name: str,
                            row: int,
                            col: int) -> None:
        """
        Add pattern markers to the chart
        
        Args:
            fig (go.Figure): Plotly figure
            signals: Pattern signals
            pattern_name (str): Name of the pattern
            row (int): Subplot row
            col (int): Subplot column
        """
        if isinstance(signals, tuple):
            bullish, bearish = signals
            
            # Add bullish markers
            if bullish.any():
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
                        showlegend=True
                    ),
                    row=row, col=col
                )
            
            # Add bearish markers
            if bearish.any():
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
                        showlegend=True
                    ),
                    row=row, col=col
                )
        else:
            # Add neutral markers
            if signals.any():
                fig.add_trace(
                    go.Scatter(
                        x=self.df.index[signals],
                        y=self.df['Low'][signals] * 0.99,
                        mode='markers',
                        marker=dict(
                            symbol='circle',
                            size=8,
                            color=self.config.color_scheme['neutral']
                        ),
                        name=pattern_name,
                        showlegend=True
                    ),
                    row=row, col=col
                )


    def _get_pattern_signals(self, pattern_name: str) -> Union[pd.Series, Tuple[pd.Series, pd.Series], None]:
        """
        Get pattern signals with caching
        
        Args:
            pattern_name (str): Name of the pattern
            
        Returns:
            Pattern signals or None if detection fails
        """
        try:
            # Check cache first
            cached_result = self._get_pattern_results(pattern_name)
            if cached_result is not None:
                return cached_result
                
            # Calculate new results
            pattern_func = getattr(self.patterns, f'detect_{pattern_name}')
            result = pattern_func(self.df)
            
            # Cache the results
            self._set_pattern_results(pattern_name, result)
            
            return result
            
        except Exception as e:
            print(f"Error getting signals for {pattern_name}: {str(e)}")
            return None


    def _add_pattern_distribution(self, fig: go.Figure, row: int, col: int) -> None:
        """Add pattern distribution pie chart"""
        pattern_counts = self._get_pattern_distribution()
        if not pattern_counts.empty:
            top_patterns = pattern_counts.nlargest(8)
            
            fig.add_trace(
                go.Pie(
                    labels=top_patterns.index,
                    values=top_patterns.values,
                    textinfo='label+percent',
                    hole=0.4,
                    marker=dict(
                        colors=self._get_pattern_colors(top_patterns.index)
                    ),
                    showlegend=False
                ),
                row=row, col=col
            )

    def add_volume_analysis(self, fig: go.Figure, row: int, col: int) -> go.Figure:
        """
        Add advanced volume analysis
        
        Args:
            fig (go.Figure): Plotly figure
            row (int): Subplot row
            col (int): Subplot column
            
        Returns:
            go.Figure: Updated figure with volume analysis
        """
        if 'Volume' not in self.df.columns:
            return fig
            
        # Calculate volume metrics
        volume_sma = self._calculate_sma(self.df['Volume'], 20)
        volume_std = self.df['Volume'].rolling(window=20).std()
        
        # Identify high volume bars (> 2 standard deviations)
        high_volume = self.df['Volume'] > (volume_sma + 2 * volume_std)
        
        # Color code volume bars
        colors = self._get_volume_colors()
        
        # Add base volume bars
        fig.add_trace(
            go.Bar(
                x=self.df.index,
                y=self.df['Volume'],
                marker_color=colors,
                name='Volume',
                opacity=0.7
            ),
            row=row, col=col
        )
        
        # Highlight high volume bars
        fig.add_trace(
            go.Scatter(
                x=self.df.index[high_volume],
                y=self.df['Volume'][high_volume],
                mode='markers',
                marker=dict(
                    symbol='diamond',
                    size=8,
                    color='red',
                    line=dict(width=1)
                ),
                name='High Volume'
            ),
            row=row, col=col
        )
        
        # Add volume moving average
        fig.add_trace(
            go.Scatter(
                x=self.df.index,
                y=volume_sma,
                line=dict(color='rgba(0,0,0,0.5)', width=2),
                name='Volume MA (20)'
            ),
            row=row, col=col
        )
        
        return fig
    
    def add_price_channels(self, fig: go.Figure, period: int = 20) -> go.Figure:
        """
        Add price channels to the chart
        
        Args:
            fig (go.Figure): Plotly figure
            period (int): Channel period
            
        Returns:
            go.Figure: Updated figure with price channels
        """
        # Calculate price channels
        upper_channel = self.df['High'].rolling(window=period).max()
        lower_channel = self.df['Low'].rolling(window=period).min()
        
        # Add upper channel
        fig.add_trace(
            go.Scatter(
                x=self.df.index,
                y=upper_channel,
                mode='lines',
                line=dict(color='rgba(0,255,0,0.3)', width=1),
                name=f'Upper Channel ({period})',
                fill=None
            )
        )
        
        # Add lower channel
        fig.add_trace(
            go.Scatter(
                x=self.df.index,
                y=lower_channel,
                mode='lines',
                line=dict(color='rgba(255,0,0,0.3)', width=1),
                name=f'Lower Channel ({period})',
                fill='tonexty'  # Fill area between channels
            )
        )
        
        return fig

    def add_pivot_points(self, 
                        fig: go.Figure, 
                        method: str = 'standard') -> go.Figure:
        """
        Add pivot points to the chart
        
        Args:
            fig (go.Figure): Plotly figure
            method (str): Pivot point calculation method ('standard' or 'fibonacci')
            
        Returns:
            go.Figure: Updated figure with pivot points
        """
        pivot_levels = self._calculate_pivot_points(method)
        
        colors = {
            'P': 'black',
            'R1': 'green',
            'R2': 'darkgreen',
            'R3': 'forestgreen',
            'S1': 'red',
            'S2': 'darkred',
            'S3': 'maroon'
        }
        
        for level_name, level_value in pivot_levels.items():
            fig.add_trace(
                go.Scatter(
                    x=self.df.index,
                    y=[level_value] * len(self.df),
                    mode='lines',
                    line=dict(
                        color=colors[level_name],
                        width=1,
                        dash='dash'
                    ),
                    name=f'Pivot {level_name}'
                )
            )
        
        return fig

    def _calculate_pivot_points(self, method: str = 'standard') -> Dict[str, float]:
        """
        Calculate pivot points
        
        Args:
            method (str): Calculation method
            
        Returns:
            Dict[str, float]: Dictionary of pivot levels
        """
        high = self.df['High'].iloc[-1]
        low = self.df['Low'].iloc[-1]
        close = self.df['Close'].iloc[-1]
        
        if method == 'standard':
            pivot = (high + low + close) / 3
            r1 = (2 * pivot) - low
            r2 = pivot + (high - low)
            r3 = high + 2 * (pivot - low)
            s1 = (2 * pivot) - high
            s2 = pivot - (high - low)
            s3 = low - 2 * (high - pivot)
        
        elif method == 'fibonacci':
            pivot = (high + low + close) / 3
            r1 = pivot + 0.382 * (high - low)
            r2 = pivot + 0.618 * (high - low)
            r3 = pivot + 1.000 * (high - low)
            s1 = pivot - 0.382 * (high - low)
            s2 = pivot - 0.618 * (high - low)
            s3 = pivot - 1.000 * (high - low)
        
        else:
            raise ValueError(f"Unsupported pivot point method: {method}")
        
        return {
            'P': pivot,
            'R1': r1,
            'R2': r2,
            'R3': r3,
            'S1': s1,
            'S2': s2,
            'S3': s3
        }

    def add_trend_analysis(self, fig: go.Figure) -> go.Figure:
        """
        Add comprehensive trend analysis to the chart
        
        Args:
            fig (go.Figure): Plotly figure
            
        Returns:
            go.Figure: Updated figure with trend analysis
        """
        # Calculate trend indicators
        trends = self._calculate_trends()
        
        # Add trend lines
        for trend in trends:
            fig.add_trace(
                go.Scatter(
                    x=[trend['start_date'], trend['end_date']],
                    y=[trend['start_price'], trend['end_price']],
                    mode='lines',
                    line=dict(
                        color=self._get_trend_color(trend['strength']),
                        width=2,
                        dash='dot' if trend['type'] == 'minor' else 'solid'
                    ),
                    name=f"{trend['direction'].capitalize()} Trend"
                )
            )
        
        return fig

    def _calculate_trends(self, 
                        min_length: int = 5,
                        swing_threshold: float = 0.02) -> List[Dict]:
        """
        Calculate major and minor trend lines
        
        Args:
            min_length (int): Minimum number of bars for trend
            swing_threshold (float): Minimum price movement for swing point
            
        Returns:
            List[Dict]: List of trend information
        """
        trends = []
        highs = self._find_swing_highs(threshold=swing_threshold)
        lows = self._find_swing_lows(threshold=swing_threshold)
        
        # Combine swing points and sort by date
        swing_points = pd.concat([
            pd.Series(1, index=highs.index),
            pd.Series(-1, index=lows.index)
        ]).sort_index()
        
        # Identify trends between swing points
        for i in range(len(swing_points) - 1):
            start_idx = swing_points.index[i]
            end_idx = swing_points.index[i + 1]
            
            if (end_idx - start_idx).days >= min_length:
                start_price = self.df.loc[start_idx, 'Close']
                end_price = self.df.loc[end_idx, 'Close']
                price_change = (end_price - start_price) / start_price
                
                trend = {
                    'start_date': start_idx,
                    'end_date': end_idx,
                    'start_price': start_price,
                    'end_price': end_price,
                    'direction': 'upward' if price_change > 0 else 'downward',
                    'strength': abs(price_change),
                    'type': 'major' if abs(price_change) > swing_threshold * 2 else 'minor'
                }
                
                trends.append(trend)
        
        return trends

    def _find_swing_highs(self, 
                        window: int = 5, 
                        threshold: float = 0.02) -> pd.Series:
        """
        Find swing high points in price data
        
        Args:
            window (int): Window size for comparison
            threshold (float): Minimum price movement threshold
            
        Returns:
            pd.Series: Series of swing high prices
        """
        highs = pd.Series(index=self.df.index, dtype=float)
        
        for i in range(window, len(self.df) - window):
            current_high = self.df['High'].iloc[i]
            left_window = self.df['High'].iloc[i-window:i]
            right_window = self.df['High'].iloc[i+1:i+window+1]
            
            if (current_high > left_window.max() and 
                current_high > right_window.max() and
                (current_high - min(left_window.min(), right_window.min())) / current_high > threshold):
                highs.iloc[i] = current_high
        
        return highs.dropna()

    def _find_swing_lows(self, 
                        window: int = 5, 
                        threshold: float = 0.02) -> pd.Series:
        """
        Find swing low points in price data
        
        Args:
            window (int): Window size for comparison
            threshold (float): Minimum price movement threshold
            
        Returns:
            pd.Series: Series of swing low prices
        """
        lows = pd.Series(index=self.df.index, dtype=float)
        
        for i in range(window, len(self.df) - window):
            current_low = self.df['Low'].iloc[i]
            left_window = self.df['Low'].iloc[i-window:i]
            right_window = self.df['Low'].iloc[i+1:i+window+1]
            
            if (current_low < left_window.min() and 
                current_low < right_window.min() and
                (max(left_window.max(), right_window.max()) - current_low) / current_low > threshold):
                lows.iloc[i] = current_low
        
        return lows.dropna()

    def _get_trend_color(self, strength: float) -> str:
        """
        Get color based on trend strength
        
        Args:
            strength (float): Trend strength value
            
        Returns:
            str: Color in rgba format
        """
        if strength > 0.1:
            return f'rgba(0,255,0,{min(strength * 5, 1)})'
        else:
            return f'rgba(255,0,0,{min(strength * 5, 1)})'

    def add_pattern_annotations(self, 
                            fig: go.Figure, 
                            min_probability: float = 0.7) -> go.Figure:
        """
        Add pattern annotations to the chart
        
        Args:
            fig (go.Figure): Plotly figure
            min_probability (float): Minimum pattern probability threshold
            
        Returns:
            go.Figure: Updated figure with pattern annotations
        """
        patterns = self._identify_probable_patterns(min_probability)
        
        for pattern in patterns:
            # Add pattern shape
            fig.add_shape(
                type='rect',
                x0=pattern['start_date'],
                x1=pattern['end_date'],
                y0=pattern['low_price'] * 0.99,
                y1=pattern['high_price'] * 1.01,
                line=dict(
                    color=self._get_pattern_color(pattern),
                    width=1,
                    dash='dash'
                ),
                fillcolor=self._get_pattern_color(pattern, alpha=0.1)
            )
            
            # Add annotation
            fig.add_annotation(
                x=pattern['end_date'],
                y=pattern['high_price'] * 1.02,
                text=f"{pattern['name']} ({pattern['probability']:.0%})",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor=self._get_pattern_color(pattern)
            )
        
        return fig

    def _identify_probable_patterns(self, 
                                min_probability: float = 0.7) -> List[Dict]:
        """
        Identify patterns with high probability
        
        Args:
            min_probability (float): Minimum pattern probability threshold
            
        Returns:
            List[Dict]: List of identified patterns with metadata
        """
        patterns = []
        pattern_methods = self._get_safe_pattern_methods()
        
        for pattern_name, pattern_func in pattern_methods.items():
            try:
                signals = self._get_pattern_signals(pattern_name)
                if signals is not None:
                    if isinstance(signals, tuple):
                        bullish, bearish = signals
                        self._add_pattern_metadata(
                            patterns, pattern_name, 'bullish', bullish, min_probability)
                        self._add_pattern_metadata(
                            patterns, pattern_name, 'bearish', bearish, min_probability)
                    else:
                        self._add_pattern_metadata(
                            patterns, pattern_name, 'neutral', signals, min_probability)
            except Exception as e:
                print(f"Error identifying pattern {pattern_name}: {str(e)}")
        
        return patterns

    def _add_pattern_metadata(self,
                            patterns: List[Dict],
                            pattern_name: str,
                            direction: str,
                            signals: pd.Series,
                            min_probability: float) -> None:
        """
        Add pattern metadata to patterns list
        
        Args:
            patterns (List[Dict]): List to append pattern metadata
            pattern_name (str): Name of the pattern
            direction (str): Pattern direction
            signals (pd.Series): Pattern signals
            min_probability (float): Minimum probability threshold
        """
        if not signals.any():
            return
            
        for idx in signals[signals].index:
            window = slice(max(0, idx-5), min(len(self.df), idx+5))
            pattern = {
                'name': f"{pattern_name} ({direction})",
                'start_date': self.df.index[window.start],
                'end_date': self.df.index[window.stop - 1],
                'high_price': self.df['High'].iloc[window].max(),
                'low_price': self.df['Low'].iloc[window].min(),
                'probability': self._calculate_pattern_probability(
                    pattern_name, direction, idx),
                'direction': direction
            }
            
            if pattern['probability'] >= min_probability:
                patterns.append(pattern)

    def _add_volume_profile(self, fig: go.Figure, row: int, col: int) -> None:
        """Add volume profile with analysis"""
        if 'Volume' in self.df.columns:
            volume_colors = self._get_volume_colors()
            
            # Add volume bars
            fig.add_trace(
                go.Bar(
                    x=self.df.index,
                    y=self.df['Volume'],
                    marker_color=volume_colors,
                    name='Volume',
                    opacity=0.7
                ),
                row=row, col=col
            )
            
            # Add volume moving averages
            for period in [20, 50]:
                vol_ma = self.df['Volume'].rolling(window=period).mean()
                fig.add_trace(
                    go.Scatter(
                        x=self.df.index,
                        y=vol_ma,
                        name=f'Volume MA({period})',
                        line=dict(width=1)
                    ),
                    row=row, col=col
                )

    def _add_pattern_correlation(self, fig: go.Figure, row: int, col: int) -> None:
        """Add pattern correlation heatmap"""
        correlation_matrix = self._calculate_pattern_correlation()
        if correlation_matrix is not None:
            fig.add_trace(
                go.Heatmap(
                    z=correlation_matrix.values,
                    x=correlation_matrix.columns,
                    y=correlation_matrix.index,
                    colorscale='RdBu',
                    zmid=0,
                    showscale=True,
                    colorbar=dict(title='Correlation')
                ),
                row=row, col=col
            )

    def _add_technical_indicators(self, fig: go.Figure, row: int, col: int) -> None:
        """Add technical indicators panel"""
        # Calculate indicators
        rsi = self._calculate_rsi(self.df['Close'])
        macd, signal = self._calculate_macd_with_signal(self.df['Close'])
        
        # Add RSI
        fig.add_trace(
            go.Scatter(
                x=self.df.index,
                y=rsi,
                name='RSI',
                line=dict(color='purple', width=1)
            ),
            row=row, col=col
        )
        
        # Add MACD
        fig.add_trace(
            go.Scatter(
                x=self.df.index,
                y=macd,
                name='MACD',
                line=dict(color='blue', width=1)
            ),
            row=row, col=col
        )
        
        fig.add_trace(
            go.Scatter(
                x=self.df.index,
                y=signal,
                name='Signal',
                line=dict(color='orange', width=1)
            ),
            row=row, col=col
        )

    def _update_dashboard_layout(self, fig: go.Figure, buttons: List[Dict]) -> None:
        """Update dashboard layout and controls"""
        fig.update_layout(
            height=self.config.default_height * 1.8,
            width=self.config.default_width,
            template=self.config.theme,
            title=dict(
                text='Interactive Technical Analysis Dashboard',
                x=0.5,
                y=0.95
            ),
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            ),
            updatemenus=[{
                'buttons': buttons,
                'direction': 'down',
                'showactive': True,
                'x': 0.1,
                'y': 1.1,
                'xanchor': 'left',
                'yanchor': 'top'
            }]
        )
        
        # Update axes
        fig.update_xaxes(rangeslider_visible=False)
        fig.update_yaxes(autorange=True)
        
        # Update subplot titles
        for annotation in fig.layout.annotations:
            annotation.update(font=dict(size=12, color=self.config.color_scheme['text']))

    def _calculate_macd_with_signal(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD and signal line"""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd, signal

    def _get_pattern_colors(self, pattern_names: List[str]) -> List[str]:
        """Generate colors for pattern visualization"""
        colors = []
        for pattern in pattern_names:
            if 'bullish' in pattern.lower():
                colors.append(self.config.color_scheme['bullish'])
            elif 'bearish' in pattern.lower():
                colors.append(self.config.color_scheme['bearish'])
            else:
                colors.append(self.config.color_scheme['neutral'])
        return colors

    def _calculate_pattern_correlation(self) -> Optional[pd.DataFrame]:
        """Calculate correlation between different patterns"""
        try:
            pattern_signals = pd.DataFrame(index=self.df.index)
            
            for pattern_name, pattern_func in self._get_safe_pattern_methods().items():
                signals = self._get_pattern_signals(pattern_name)
                if signals is not None:
                    if isinstance(signals, tuple):
                        pattern_signals[f"{pattern_name}_bullish"] = signals[0]
                        pattern_signals[f"{pattern_name}_bearish"] = signals[1]
                    else:
                        pattern_signals[pattern_name] = signals
            
            if not pattern_signals.empty:
                return pattern_signals.corr()
            return None
            
        except Exception as e:
            print(f"Error calculating pattern correlation: {str(e)}")
            return None
    
    def _get_safe_pattern_methods(self) -> Dict[str, Callable]:
        """
        Get pattern detection methods with error handling
        
        Returns:
            Dict[str, Callable]: Dictionary of safe pattern methods
            
        Note:
            Filters out methods that require special handling
        """
        excluded_patterns = {
            'breakout_patterns', 'harmonic_patterns', 'multi_timeframe_patterns',
            'pattern_combinations', 'pattern_reliability', 'volatility_adjusted_patterns'
        }
        
        pattern_methods = {}
        for name, method in inspect.getmembers(self.patterns, predicate=inspect.isfunction):
            if name.startswith('detect_') and name[7:] not in excluded_patterns:
                pattern_methods[name[7:]] = method
        
        return pattern_methods


    
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

    def _calculate_atr(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """
        Calculate Average True Range
        
        Args:
            df (pd.DataFrame): OHLC data
            window (int): Calculation window
            
        Returns:
            pd.Series: ATR values
        """
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR
        atr = tr.rolling(window=window).mean()
        
        return atr
    
    def add_technical_overlay(self, 
                         indicator_name: str, 
                         params: Dict[str, Any] = None) -> pd.Series:
        """
        Add technical indicator overlay
        
        Args:
            indicator_name (str): Name of the indicator
            params (Dict[str, Any]): Indicator parameters
            
        Returns:
            pd.Series: Indicator values
            
        Raises:
            ValueError: If indicator is not supported
        """
        params = params or {}
        
        indicator_functions = {
            'sma': self._calculate_sma,
            'ema': self._calculate_ema,
            'bollinger_bands': self._calculate_bollinger_bands,
            'rsi': self._calculate_rsi,
            'macd': self._calculate_macd,
            'atr': self._calculate_atr
        }
        
        if indicator_name not in indicator_functions:
            raise ValueError(f"Unsupported indicator: {indicator_name}")
            
        return indicator_functions[indicator_name](self.df['Close'], **params)

    def _calculate_sma(self, prices: pd.Series, period: int = 20) -> pd.Series:
        """
        Calculate Simple Moving Average
        
        Args:
            prices (pd.Series): Price data
            period (int): Moving average period
            
        Returns:
            pd.Series: SMA values
        """
        return prices.rolling(window=period).mean()

    def _calculate_ema(self, prices: pd.Series, period: int = 20) -> pd.Series:
        """
        Calculate Exponential Moving Average
        
        Args:
            prices (pd.Series): Price data
            period (int): Moving average period
            
        Returns:
            pd.Series: EMA values
        """
        return prices.ewm(span=period, adjust=False).mean()

    def _calculate_bollinger_bands(self, 
                                prices: pd.Series, 
                                period: int = 20, 
                                std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands
        
        Args:
            prices (pd.Series): Price data
            period (int): Moving average period
            std_dev (float): Number of standard deviations
            
        Returns:
            Dict[str, pd.Series]: Dictionary containing upper, middle, and lower bands
        """
        middle_band = self._calculate_sma(prices, period)
        rolling_std = prices.rolling(window=period).std()
        
        upper_band = middle_band + (rolling_std * std_dev)
        lower_band = middle_band - (rolling_std * std_dev)
        
        return {
            'upper': upper_band,
            'middle': middle_band,
            'lower': lower_band
        }


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

class PatternQualityMetrics:
    """Helper class for pattern quality assessment"""
    def __init__(self, min_quality_score: float = 0.6):
        self.min_quality_score = min_quality_score
        self.quality_weights = {
            'volume_confirmation': 0.25,
            'price_momentum': 0.20,
            'pattern_symmetry': 0.15,
            'trend_alignment': 0.20,
            'support_resistance': 0.20
        }

    def analyze_pattern_quality(self, 
                            pattern: Dict[str, Any],
                            window: int = 20) -> Dict[str, float]:
        """
        Analyze the quality of detected patterns
        
        Args:
            pattern (Dict[str, Any]): Pattern information
            window (int): Analysis window size
            
        Returns:
            Dict[str, float]: Quality metrics
        """
        pattern_start = self.df.index.get_loc(pattern['start_date'])
        pattern_end = self.df.index.get_loc(pattern['end_date'])
        pattern_slice = slice(max(0, pattern_start - window), min(len(self.df), pattern_end + window))
        
        # Calculate quality metrics
        quality_metrics = {
            'volume_confirmation': self._check_volume_confirmation(pattern_slice),
            'price_momentum': self._check_price_momentum(pattern_slice),
            'pattern_symmetry': self._check_pattern_symmetry(pattern),
            'trend_alignment': self._check_trend_alignment(pattern),
            'support_resistance': self._check_support_resistance(pattern)
        }
        
        # Calculate overall quality score
        quality_score = sum(
            metric * self.quality_weights[name]
            for name, metric in quality_metrics.items()
        )
        
        quality_metrics['overall_score'] = quality_score
        return quality_metrics

    def _check_volume_confirmation(self, pattern_slice: slice) -> float:
        """
        Check volume confirmation for pattern
        
        Args:
            pattern_slice (slice): Time slice for pattern
            
        Returns:
            float: Volume confirmation score
        """
        if 'Volume' not in self.df.columns:
            return 0.5
            
        volume_data = self.df['Volume'].iloc[pattern_slice]
        avg_volume = volume_data.mean()
        recent_volume = volume_data.iloc[-5:].mean()
        
        volume_ratio = recent_volume / avg_volume
        return min(volume_ratio, 1.0) if volume_ratio > 1 else volume_ratio

    def _check_price_momentum(self, pattern_slice: slice) -> float:
        """
        Check price momentum during pattern formation
        
        Args:
            pattern_slice (slice): Time slice for pattern
            
        Returns:
            float: Momentum confirmation score
        """
        price_data = self.df['Close'].iloc[pattern_slice]
        
        # Calculate RSI and MACD
        rsi = self._calculate_rsi(price_data)
        macd, signal = self._calculate_macd_with_signal(price_data)
        
        # Check momentum alignment
        rsi_score = abs(rsi.iloc[-1] - 50) / 50
        macd_score = 1 if (macd.iloc[-1] > signal.iloc[-1]) else 0
        
        return (rsi_score + macd_score) / 2

    def _check_pattern_symmetry(self, pattern: Dict[str, Any]) -> float:
        """
        Check symmetry of pattern formation
        
        Args:
            pattern (Dict[str, Any]): Pattern information
            
        Returns:
            float: Symmetry score
        """
        pattern_prices = self.df['Close'].loc[pattern['start_date']:pattern['end_date']]
        
        if len(pattern_prices) < 4:
            return 0.5
            
        # Split pattern into two halves
        mid_point = len(pattern_prices) // 2
        first_half = pattern_prices.iloc[:mid_point]
        second_half = pattern_prices.iloc[mid_point:]
        
        # Calculate symmetry based on price movements
        first_move = abs(first_half.max() - first_half.min())
        second_move = abs(second_half.max() - second_half.min())
        
        symmetry_ratio = min(first_move, second_move) / max(first_move, second_move)
        return symmetry_ratio

    def analyze_pattern_confluence(self, 
                                timeframe: str = 'daily',
                                min_confidence: float = 0.7) -> List[Dict]:
        """
        Analyze pattern confluence across different indicators
        
        Args:
            timeframe (str): Analysis timeframe
            min_confidence (float): Minimum confidence threshold
            
        Returns:
            List[Dict]: Confluence points with metadata
        """
        confluence_points = []
        
        # Get all active patterns
        patterns = self._identify_probable_patterns(min_confidence)
        
        for pattern in patterns:
            # Check technical indicator confluence
            indicator_confluence = self._check_indicator_confluence(pattern)
            
            # Check support/resistance confluence
            sr_confluence = self._check_sr_confluence(pattern)
            
            # Check trend confluence
            trend_confluence = self._check_trend_confluence(pattern)
            
            # Calculate overall confluence score
            confluence_score = (
                indicator_confluence * 0.4 +
                sr_confluence * 0.3 +
                trend_confluence * 0.3
            )
            
            if confluence_score >= min_confidence:
                confluence_points.append({
                    'pattern': pattern['name'],
                    'date': pattern['end_date'],
                    'score': confluence_score,
                    'indicators': indicator_confluence,
                    'support_resistance': sr_confluence,
                    'trend': trend_confluence
                })
        
        return confluence_points

    def _check_indicator_confluence(self, pattern: Dict[str, Any]) -> float:
        """
        Check confluence with technical indicators
        
        Args:
            pattern (Dict[str, Any]): Pattern information
            
        Returns:
            float: Indicator confluence score
        """
        pattern_end = self.df.index.get_loc(pattern['end_date'])
        
        # Calculate various indicators
        sma_20 = self._calculate_sma(self.df['Close'], 20)
        sma_50 = self._calculate_sma(self.df['Close'], 50)
        rsi = self._calculate_rsi(self.df['Close'])
        macd, signal = self._calculate_macd_with_signal(self.df['Close'])
        
        # Check indicator alignments
        alignments = [
            sma_20.iloc[pattern_end] > sma_50.iloc[pattern_end],
            rsi.iloc[pattern_end] > 50 if pattern['direction'] == 'bullish' else rsi.iloc[pattern_end] < 50,
            macd.iloc[pattern_end] > signal.iloc[pattern_end] if pattern['direction'] == 'bullish' 
            else macd.iloc[pattern_end] < signal.iloc[pattern_end]
        ]
        
        return sum(alignments) / len(alignments)

    def _check_sr_confluence(self, pattern: Dict[str, Any]) -> float:
        """
        Check confluence with support and resistance levels
        
        Args:
            pattern (Dict[str, Any]): Pattern information
            
        Returns:
            float: Support/resistance confluence score
        """
        # Calculate key price levels
        price_levels = self._identify_key_levels(window=50)
        pattern_price = self.df['Close'].loc[pattern['end_date']]
        
        # Find closest levels
        distances = [abs(level - pattern_price) / pattern_price for level in price_levels]
        min_distance = min(distances) if distances else 1.0
        
        return 1 - min(min_distance, 1.0)

    def _check_trend_confluence(self, pattern: Dict[str, Any]) -> float:
        """
        Check confluence with existing trends
        
        Args:
            pattern (Dict[str, Any]): Pattern information
            
        Returns:
            float: Trend confluence score
        """
        trends = self._calculate_trends()
        
        # Find active trend at pattern end
        active_trends = [
            trend for trend in trends
            if trend['start_date'] <= pattern['end_date'] <= trend['end_date']
        ]
        
        if not active_trends:
            return 0.5
            
        # Check if pattern aligns with trend direction
        trend_alignment = sum(
            1 for trend in active_trends
            if trend['direction'] == pattern['direction']
        ) / len(active_trends)
        
        return trend_alignment
    
class MarketRegime:
    """
    Represents a market regime with its characteristics
    """
    def __init__(self, 
                 regime_type: str,
                 volatility: str,
                 trend: str,
                 volume: str,
                 start_date: pd.Timestamp,
                 end_date: pd.Timestamp,
                 confidence: float):
        self.regime_type = regime_type  # e.g., 'trending', 'ranging', 'transitioning'
        self.volatility = volatility    # e.g., 'high', 'medium', 'low'
        self.trend = trend              # e.g., 'bullish', 'bearish', 'neutral'
        self.volume = volume            # e.g., 'increasing', 'decreasing', 'stable'
        self.start_date = start_date
        self.end_date = end_date
        self.confidence = confidence

    def analyze_market_regime(self, 
                            window_size: int = 20,
                            volatility_window: int = 20,
                            trend_window: int = 50) -> List[MarketRegime]:
        """
        Analyze and identify market regimes
        
        Args:
            window_size (int): Base window for regime analysis
            volatility_window (int): Window for volatility calculations
            trend_window (int): Window for trend analysis
            
        Returns:
            List[MarketRegime]: List of identified market regimes
        """
        # Initialize result list
        regimes = []
        
        try:
            # Calculate regime components
            volatility = self._calculate_volatility_regime(volatility_window)
            trend = self._calculate_trend_regime(trend_window)
            volume = self._calculate_volume_regime(window_size)
            
            # Combine components to identify regime changes
            regime_changes = self._identify_regime_changes(
                volatility=volatility,
                trend=trend,
                volume=volume,
                window_size=window_size
            )
            
            # Create MarketRegime objects for each identified period
            current_regime = None
            for i, (date, regime_data) in enumerate(regime_changes.items()):
                if current_regime is None:
                    current_regime = {
                        'start_date': date,
                        'regime_data': regime_data
                    }
                else:
                    # Check if regime has changed significantly
                    if self._is_regime_change(current_regime['regime_data'], regime_data):
                        # Create regime object for the completed period
                        regime = MarketRegime(
                            regime_type=self._determine_regime_type(current_regime['regime_data']),
                            volatility=current_regime['regime_data']['volatility'],
                            trend=current_regime['regime_data']['trend'],
                            volume=current_regime['regime_data']['volume'],
                            start_date=current_regime['start_date'],
                            end_date=date,
                            confidence=self._calculate_regime_confidence(current_regime['regime_data'])
                        )
                        regimes.append(regime)
                        
                        # Start new regime
                        current_regime = {
                            'start_date': date,
                            'regime_data': regime_data
                        }
            
            # Add final regime if exists
            if current_regime is not None:
                regime = MarketRegime(
                    regime_type=self._determine_regime_type(current_regime['regime_data']),
                    volatility=current_regime['regime_data']['volatility'],
                    trend=current_regime['regime_data']['trend'],
                    volume=current_regime['regime_data']['volume'],
                    start_date=current_regime['start_date'],
                    end_date=self.df.index[-1],
                    confidence=self._calculate_regime_confidence(current_regime['regime_data'])
                )
                regimes.append(regime)
            
            return regimes
            
        except Exception as e:
            print(f"Error in market regime analysis: {str(e)}")
            return []

    def _calculate_volatility_regime(self, window: int) -> pd.Series:
        """
        Calculate volatility regime
        
        Args:
            window (int): Calculation window
            
        Returns:
            pd.Series: Volatility regime classifications
        """
        # Calculate daily returns
        returns = self.df['Close'].pct_change()
        
        # Calculate rolling volatility
        volatility = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
        
        # Classify volatility regimes
        vol_quantiles = volatility.quantile([0.33, 0.66])
        
        def classify_volatility(x):
            if pd.isna(x):
                return 'unknown'
            elif x <= vol_quantiles[0.33]:
                return 'low'
            elif x <= vol_quantiles[0.66]:
                return 'medium'
            else:
                return 'high'
        
        return volatility.apply(classify_volatility)

    def _calculate_trend_regime(self, window: int) -> pd.Series:
        """
        Calculate trend regime
        
        Args:
            window (int): Calculation window
            
        Returns:
            pd.Series: Trend regime classifications
        """
        # Calculate moving averages
        sma_short = self.df['Close'].rolling(window=window//2).mean()
        sma_long = self.df['Close'].rolling(window=window).mean()
        
        # Calculate trend direction
        trend = pd.Series(index=self.df.index, dtype=str)
        
        for i in range(len(self.df)):
            if pd.isna(sma_short.iloc[i]) or pd.isna(sma_long.iloc[i]):
                trend.iloc[i] = 'unknown'
            else:
                # Calculate trend strength
                price_to_sma = self.df['Close'].iloc[i] / sma_long.iloc[i] - 1
                
                if sma_short.iloc[i] > sma_long.iloc[i]:
                    trend.iloc[i] = 'bullish' if abs(price_to_sma) > 0.02 else 'weak_bullish'
                elif sma_short.iloc[i] < sma_long.iloc[i]:
                    trend.iloc[i] = 'bearish' if abs(price_to_sma) > 0.02 else 'weak_bearish'
                else:
                    trend.iloc[i] = 'neutral'
        
        return trend
    
    def _calculate_volume_regime(self, window: int) -> pd.Series:
        """
        Calculate volume regime
        
        Args:
            window (int): Calculation window
            
        Returns:
            pd.Series: Volume regime classifications
        """
        if 'Volume' not in self.df.columns:
            return pd.Series('unknown', index=self.df.index)
        
        # Calculate volume metrics
        volume = self.df['Volume']
        volume_ma = volume.rolling(window=window).mean()
        volume_std = volume.rolling(window=window).std()
        
        # Calculate relative volume
        relative_volume = volume / volume_ma
        
        # Classify volume regimes
        volume_regime = pd.Series(index=self.df.index, dtype=str)
        
        for i in range(len(self.df)):
            if pd.isna(relative_volume.iloc[i]):
                volume_regime.iloc[i] = 'unknown'
            else:
                if relative_volume.iloc[i] > 1.5:
                    volume_regime.iloc[i] = 'very_high'
                elif relative_volume.iloc[i] > 1.1:
                    volume_regime.iloc[i] = 'high'
                elif relative_volume.iloc[i] > 0.9:
                    volume_regime.iloc[i] = 'normal'
                elif relative_volume.iloc[i] > 0.5:
                    volume_regime.iloc[i] = 'low'
                else:
                    volume_regime.iloc[i] = 'very_low'
        
        return volume_regime

    def _identify_regime_changes(self,
                            volatility: pd.Series,
                            trend: pd.Series,
                            volume: pd.Series,
                            window_size: int) -> Dict[pd.Timestamp, Dict]:
        """
        Identify points where market regime changes
        
        Args:
            volatility (pd.Series): Volatility regime series
            trend (pd.Series): Trend regime series
            volume (pd.Series): Volume regime series
            window_size (int): Window for change detection
            
        Returns:
            Dict[pd.Timestamp, Dict]: Dictionary of regime changes with metadata
        """
        regime_changes = {}
        
        # Combine all regime components
        for i in range(len(self.df)):
            current_date = self.df.index[i]
            
            # Get current regime characteristics
            current_regime = {
                'volatility': volatility.iloc[i],
                'trend': trend.iloc[i],
                'volume': volume.iloc[i],
                'momentum': self._calculate_momentum_regime(i),
                'support_resistance': self._calculate_sr_regime(i)
            }
            
            # Check if this represents a regime change
            if i > 0:
                prev_regime = regime_changes.get(self.df.index[i-1], None)
                if prev_regime is None or self._is_significant_change(prev_regime, current_regime):
                    regime_changes[current_date] = current_regime
            else:
                regime_changes[current_date] = current_regime
        
        return regime_changes

    def _calculate_momentum_regime(self, index: int) -> str:
        """
        Calculate momentum regime for a specific index
        
        Args:
            index (int): Data index
            
        Returns:
            str: Momentum regime classification
        """
        try:
            # Calculate RSI
            rsi = self._calculate_rsi(self.df['Close'])
            current_rsi = rsi.iloc[index]
            
            # Calculate MACD
            macd, signal = self._calculate_macd_with_signal(self.df['Close'])
            current_macd = macd.iloc[index]
            current_signal = signal.iloc[index]
            
            # Determine momentum regime
            if pd.isna(current_rsi) or pd.isna(current_macd):
                return 'unknown'
                
            if current_rsi > 70 and current_macd > current_signal:
                return 'strong_bullish'
            elif current_rsi < 30 and current_macd < current_signal:
                return 'strong_bearish'
            elif current_rsi > 60:
                return 'bullish'
            elif current_rsi < 40:
                return 'bearish'
            else:
                return 'neutral'
                
        except Exception as e:
            print(f"Error calculating momentum regime: {str(e)}")
            return 'unknown'

    def _calculate_sr_regime(self, index: int, window: int = 20) -> str:
        """
        Calculate support/resistance regime for a specific index
        
        Args:
            index (int): Data index
            window (int): Lookback window
            
        Returns:
            str: Support/resistance regime classification
        """
        try:
            # Get relevant price data
            start_idx = max(0, index - window)
            price_window = self.df['Close'].iloc[start_idx:index+1]
            current_price = price_window.iloc[-1]
            
            # Calculate support and resistance levels
            support_levels = self._identify_support_levels(price_window)
            resistance_levels = self._identify_resistance_levels(price_window)
            
            # Find closest levels
            closest_support = min((abs(level - current_price), level) 
                                for level in support_levels)[1]
            closest_resistance = min((abs(level - current_price), level) 
                                for level in resistance_levels)[1]
            
            # Calculate distances as percentages
            support_distance = (current_price - closest_support) / current_price
            resistance_distance = (closest_resistance - current_price) / current_price
            
            # Determine regime
            if support_distance < 0.01 and resistance_distance < 0.01:
                return 'compression'
            elif support_distance < 0.01:
                return 'at_support'
            elif resistance_distance < 0.01:
                return 'at_resistance'
            elif support_distance < resistance_distance:
                return 'near_support'
            else:
                return 'near_resistance'
                
        except Exception as e:
            print(f"Error calculating S/R regime: {str(e)}")
            return 'unknown'

    def _is_significant_change(self, 
                            prev_regime: Dict[str, str], 
                            current_regime: Dict[str, str]) -> bool:
        """
        Determine if regime change is significant
        
        Args:
            prev_regime (Dict[str, str]): Previous regime characteristics
            current_regime (Dict[str, str]): Current regime characteristics
            
        Returns:
            bool: True if change is significant
        """
        # Define weights for different components
        weights = {
            'trend': 0.35,
            'volatility': 0.25,
            'momentum': 0.20,
            'volume': 0.10,
            'support_resistance': 0.10
        }
        
        # Calculate change score
        change_score = 0
        for component, weight in weights.items():
            if prev_regime[component] != current_regime[component]:
                change_score += weight
        
        return change_score >= 0.4  # Threshold for significant change
    
    def visualize_market_regimes(self, regimes: List[MarketRegime]) -> go.Figure:
        """
        Create visualization of market regimes
        
        Args:
            regimes (List[MarketRegime]): List of identified market regimes
            
        Returns:
            go.Figure: Plotly figure with regime visualization
        """
        # Create subplot structure
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                'Price with Regime Overlay',
                'Regime Characteristics',
                'Regime Confidence'
            ),
            row_heights=[0.5, 0.3, 0.2]
        )
        
        # Add price chart
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
        
        # Add regime overlays
        for regime in regimes:
            # Add regime background
            fig.add_vrect(
                x0=regime.start_date,
                x1=regime.end_date,
                fillcolor=self._get_regime_color(regime.regime_type),
                opacity=0.2,
                layer="below",
                line_width=0,
                row=1, col=1
            )
            
            # Add regime characteristics
            self._add_regime_characteristics(fig, regime, row=2, col=1)
            
            # Add confidence indicator
            fig.add_trace(
                go.Scatter(
                    x=[regime.start_date, regime.end_date],
                    y=[regime.confidence, regime.confidence],
                    mode='lines',
                    line=dict(
                        color=self._get_regime_color(regime.regime_type),
                        width=2
                    ),
                    name=f'Confidence ({regime.regime_type})'
                ),
                row=3, col=1
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Market Regime Analysis",
            xaxis_rangeslider_visible=False
        )
        
        return fig

    def _get_regime_color(self, regime_type: str) -> str:
        """
        Get color for regime visualization
        
        Args:
            regime_type (str): Type of market regime
            
        Returns:
            str: Color code for regime
        """
        color_map = {
            'trending': 'rgba(46, 204, 113, 0.8)',
            'ranging': 'rgba(52, 152, 219, 0.8)',
            'transitioning': 'rgba(155, 89, 182, 0.8)',
            'volatile': 'rgba(231, 76, 60, 0.8)',
            'consolidating': 'rgba(241, 196, 15, 0.8)'
        }
        
        return color_map.get(regime_type, 'rgba(149, 165, 166, 0.8)')

    def _add_regime_characteristics(self, 
                                fig: go.Figure, 
                                regime: MarketRegime,
                                row: int,
                                col: int) -> None:
        """
        Add regime characteristics visualization
        
        Args:
            fig (go.Figure): Plotly figure
            regime (MarketRegime): Market regime object
            row (int): Subplot row
            col (int): Subplot column
        """
        # Create characteristic indicators
        characteristics = {
            'Volatility': self._normalize_regime_value(regime.volatility),
            'Trend': self._normalize_regime_value(regime.trend),
            'Volume': self._normalize_regime_value(regime.volume)
        }
        
        # Add characteristics as stacked bars
        for i, (char_name, value) in enumerate(characteristics.items()):
            fig.add_trace(
                go.Bar(
                    x=[[regime.start_date, regime.end_date]],
                    y=[value],
                    name=f'{char_name} ({regime.regime_type})',
                    marker_color=self._get_characteristic_color(char_name, value),
                    showlegend=False
                ),
                row=row, col=col
            )

    def _normalize_regime_value(self, regime_value: str) -> float:
        """
        Normalize regime characteristic values for visualization
        
        Args:
            regime_value (str): Regime characteristic value
            
        Returns:
            float: Normalized value between 0 and 1
        """
        value_maps = {
            'volatility': {
                'low': 0.2,
                'medium': 0.5,
                'high': 0.8,
                'very_high': 1.0
            },
            'trend': {
                'strong_bearish': 0.0,
                'bearish': 0.2,
                'weak_bearish': 0.4,
                'neutral': 0.5,
                'weak_bullish': 0.6,
                'bullish': 0.8,
                'strong_bullish': 1.0
            },
            'volume': {
                'very_low': 0.0,
                'low': 0.25,
                'normal': 0.5,
                'high': 0.75,
                'very_high': 1.0
            }
        }
        
        # Try to match the regime value with each map
        for map_type, value_map in value_maps.items():
            if regime_value.lower() in value_map:
                return value_map[regime_value.lower()]
        
        return 0.5  # Default value if no match found

    def _get_characteristic_color(self, characteristic: str, value: float) -> str:
        """
        Get color for regime characteristic visualization
        
        Args:
            characteristic (str): Name of characteristic
            value (float): Normalized value
            
        Returns:
            str: Color code for characteristic
        """
        color_scales = {
            'Volatility': [
                [0, 'rgba(46, 204, 113, 0.8)'],  # Green for low volatility
                [1, 'rgba(231, 76, 60, 0.8)']    # Red for high volatility
            ],
            'Trend': [
                [0, 'rgba(231, 76, 60, 0.8)'],   # Red for bearish
                [0.5, 'rgba(149, 165, 166, 0.8)'], # Gray for neutral
                [1, 'rgba(46, 204, 113, 0.8)']    # Green for bullish
            ],
            'Volume': [
                [0, 'rgba(149, 165, 166, 0.8)'],  # Gray for low volume
                [1, 'rgba(52, 152, 219, 0.8)']    # Blue for high volume
            ]
        }
        
        scale = color_scales.get(characteristic, color_scales['Volume'])
        return self._interpolate_color(value, scale)