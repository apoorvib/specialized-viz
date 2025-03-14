# Common modules that might be needed
import os
import sys
import threading
import time
import json
import math
import warnings
from collections import defaultdict
from datetime import datetime, timedelta

# Data processing
import numpy as np
import pandas as pd
from scipy import stats

# Visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.figure import Figure

# Type hints
from typing import Dict, List, Tuple, Optional, Callable, Union, Any

# Data structures
from dataclasses import dataclass, field, asdict

# Machine learning
from sklearn.cluster import DBSCAN

# Local imports
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
    
    # New settings
    fonts: Dict[str, Any] = field(default_factory=lambda: {
        'family': 'Arial, sans-serif',
        'sizes': {
            'title': 16,
            'subtitle': 14,
            'axis': 12,
            'label': 10,
            'annotation': 10
        },
        'weights': {
            'title': 'bold',
            'subtitle': 'normal',
            'axis': 'normal',
            'label': 'normal'
        }
    })
    
    layout: Dict[str, Any] = field(default_factory=lambda: {
        'padding': {
            'top': 40,
            'right': 40,
            'bottom': 40,
            'left': 60
        },
        'spacing': {
            'vertical': 0.1,
            'horizontal': 0.1
        },
        'legend': {
            'position': 'top',
            'orientation': 'h'
        }
    })
    
    grid_settings: Dict[str, Any] = field(default_factory=lambda: {
        'color': '#ecf0f1',
        'opacity': 0.5,
        'width': 1,
        'style': 'dashed'
    })
    
    annotation_settings: Dict[str, Any] = field(default_factory=lambda: {
        'style': {
            'background_color': 'rgba(255, 255, 255, 0.8)',
            'border_color': '#95a5a6',
            'border_width': 1
        },
        'arrow': {
            'color': '#95a5a6',
            'width': 1,
            'style': 'solid'
        }
    })
    
    interactive_settings: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'animation': {
            'duration': 500,
            'easing': 'cubic-bezier(0.4, 0, 0.2, 1)',
            'on_load': True
        },
        'tooltip': {
            'enabled': True,
            'background_color': 'rgba(255, 255, 255, 0.95)',
            'border_color': '#95a5a6'
        }
    })
    
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
    
    def update_theme(self, theme_name: str) -> None:
        """
        Update visualization theme
        
        Args:
            theme_name (str): Name of the theme to apply
        """
        self.theme = theme_name
        self._apply_theme_settings(theme_name)
    
    def _apply_theme_settings(self, theme_name: str) -> None:
        """
        Apply settings for specific theme
        
        Args:
            theme_name (str): Theme name
        """
        theme_settings = THEME_PRESETS.get(theme_name, {})
        for key, value in theme_settings.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert config to dictionary for serialization
        
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'VisualizationConfig':
        """
        Create config from dictionary
        
        Args:
            config_dict (Dict[str, Any]): Configuration dictionary
            
        Returns:
            VisualizationConfig: Configuration instance
        """
        return cls(**config_dict)
    
class VisualizationCache:
    """
    Cache for visualization components and calculations
    
    Attributes:
        max_size (int): Maximum number of items in cache
        ttl (int): Time to live for cache items in seconds
        _cache (Dict): Main cache storage
        _metadata (Dict): Cache item metadata
        _lock (threading.Lock): Thread lock for cache operations
    """
    
    def __init__(self, max_size: int = 100, ttl: int = 3600):
        """
        Initialize visualization cache
        
        Args:
            max_size (int): Maximum cache size
            ttl (int): Cache item time to live in seconds
        """
        self.max_size = max_size
        self.ttl = ttl
        self._cache = {}
        self._metadata = {}
        self._lock = threading.Lock()
    
    def get_figure(self, key: str) -> Optional[go.Figure]:
        """
        Get cached figure
        
        Args:
            key (str): Cache key
            
        Returns:
            Optional[go.Figure]: Cached figure or None if not found/expired
        """
        with self._lock:
            if key not in self._cache:
                return None
            
            # Check if item has expired
            if self._is_expired(key):
                self._remove_item(key)
                return None
            
            # Update access metadata
            self._metadata[key]['last_access'] = time.time()
            self._metadata[key]['access_count'] += 1
            
            return self._cache[key]
    
    def cache_figure(self, key: str, figure: go.Figure) -> None:
        """
        Cache a figure
        
        Args:
            key (str): Cache key
            figure (go.Figure): Figure to cache
        """
        with self._lock:
            current_time = time.time()
            
            # Remove expired items
            self._remove_expired()
            
            # Check if we need to make room
            if len(self._cache) >= self.max_size:
                self._evict_lru_item()
            
            # Add new item
            self._cache[key] = figure
            self._metadata[key] = {
                'created_at': current_time,
                'last_access': current_time,
                'access_count': 0,
                'size': self._estimate_figure_size(figure)
            }
    
    def _is_expired(self, key: str) -> bool:
        """
        Check if cache item has expired
        
        Args:
            key (str): Cache key
            
        Returns:
            bool: True if item has expired
        """
        current_time = time.time()
        return current_time - self._metadata[key]['created_at'] > self.ttl
    
    def _remove_expired(self) -> None:
        """Remove all expired items from cache"""
        expired_keys = [key for key in self._cache if self._is_expired(key)]
        for key in expired_keys:
            self._remove_item(key)
    
    def _evict_lru_item(self) -> None:
        """Remove least recently used item from cache"""
        if not self._cache:
            return
            
        lru_key = min(self._metadata.items(), 
                     key=lambda x: x[1]['last_access'])[0]
        self._remove_item(lru_key)
    
    def _remove_item(self, key: str) -> None:
        """
        Remove item from cache
        
        Args:
            key (str): Cache key to remove
        """
        self._cache.pop(key, None)
        self._metadata.pop(key, None)
    
    def _estimate_figure_size(self, figure: go.Figure) -> int:
        """
        Estimate memory size of figure
        
        Args:
            figure (go.Figure): Figure to estimate
            
        Returns:
            int: Estimated size in bytes
        """
        # Basic size estimation based on number of traces and data points
        size = 0
        for trace in figure.data:
            if hasattr(trace, 'x'):
                size += len(trace.x) * 8  # Assume 8 bytes per number
            if hasattr(trace, 'y'):
                size += len(trace.y) * 8
        return size
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dict[str, Any]: Cache statistics
        """
        with self._lock:
            total_size = sum(meta['size'] for meta in self._metadata.values())
            return {
                'item_count': len(self._cache),
                'total_size_bytes': total_size,
                'hit_count': sum(meta['access_count'] for meta in self._metadata.values()),
                'max_size': self.max_size,
                'ttl': self.ttl
            }
    
    def clear(self) -> None:
        """Clear all items from cache"""
        with self._lock:
            self._cache.clear()
            self._metadata.clear()
    
# New class for theme management
class VisualizationTheme:
    """Theme manager for visualization customization"""
    
    def __init__(self, theme_name: str = "default"):
        self.theme_name = theme_name
        self.color_scheme = self._get_default_colors()
        self.font_settings = self._get_default_fonts()
        self.chart_settings = self._get_default_chart_settings()
    
    def _get_default_colors(self) -> Dict[str, str]:
        return {
            'background': '#ffffff',
            'text': '#2c3e50',
            'grid': '#ecf0f1',
            'bullish': '#2ecc71',
            'bearish': '#e74c3c',
            'neutral': '#3498db',
            'volume_up': '#2ecc71',
            'volume_down': '#e74c3c'
        }
    
    def _get_default_fonts(self) -> Dict[str, Any]:
        return {
            'family': 'Arial, sans-serif',
            'size': {
                'title': 16,
                'axis': 12,
                'label': 10,
                'annotation': 10
            }
        }
    
    def _get_default_chart_settings(self) -> Dict[str, Any]:
        return {
            'padding': {'top': 10, 'right': 50, 'bottom': 20, 'left': 50},
            'grid': True,
            'grid_opacity': 0.1,
            'show_legend': True,
            'legend_position': 'top'
        }

@dataclass    
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
        
class BaseVisualizationSettings:
    """
    Base settings and utilities for visualizations
    
    Attributes:
        config (VisualizationConfig): Visualization configuration
        cache (VisualizationCache): Visualization cache
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize base visualization settings
        
        Args:
            config (Optional[VisualizationConfig]): Visualization configuration
        """
        self.config = config or VisualizationConfig()
        self.cache = VisualizationCache()
        
    def apply_default_layout(self, fig: go.Figure) -> go.Figure:
        """
        Apply default layout settings to figure
        
        Args:
            fig (go.Figure): Plotly figure
            
        Returns:
            go.Figure: Figure with default layout applied
        """
        fig.update_layout(
            template=self.config.theme,
            height=self.config.default_height,
            width=self.config.default_width,
            font=dict(
                family=self.config.fonts['family'],
                size=self.config.fonts['sizes']['axis']
            ),
            showlegend=True,
            legend=dict(
                orientation=self.config.layout['legend']['orientation'],
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=self.config.fonts['sizes']['label'])
            ),
            margin=dict(
                t=self.config.layout['padding']['top'],
                r=self.config.layout['padding']['right'],
                b=self.config.layout['padding']['bottom'],
                l=self.config.layout['padding']['left']
            )
        )
        
        # Apply grid settings if enabled
        if self.config.show_grid:
            fig.update_layout(
                xaxis=dict(
                    showgrid=True,
                    gridwidth=self.config.grid_settings['width'],
                    gridcolor=self.config.grid_settings['color'],
                    zeroline=False
                ),
                yaxis=dict(
                    showgrid=True,
                    gridwidth=self.config.grid_settings['width'],
                    gridcolor=self.config.grid_settings['color'],
                    zeroline=False
                )
            )
        
        return fig
    
    def create_annotation(self, 
                         text: str,
                         x: Union[float, str],
                         y: float,
                         is_pattern: bool = False) -> Dict[str, Any]:
        """
        Create figure annotation with default settings
        
        Args:
            text (str): Annotation text
            x (Union[float, str]): X-coordinate
            y (float): Y-coordinate
            is_pattern (bool): Whether annotation is for a pattern
            
        Returns:
            Dict[str, Any]: Annotation settings
        """
        return dict(
            text=text,
            x=x,
            y=y,
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=self.config.annotation_settings['arrow']['width'],
            arrowcolor=self.config.annotation_settings['arrow']['color'],
            bgcolor=self.config.annotation_settings['style']['background_color'],
            bordercolor=self.config.annotation_settings['style']['border_color'],
            borderwidth=self.config.annotation_settings['style']['border_width'],
            font=dict(
                size=self.config.annotation_font_size,
                color=self.config.color_scheme['text']
            ),
            opacity=self.config.pattern_opacity if is_pattern else 1.0
        )
    
    def get_color_for_value(self, 
                           value: float, 
                           is_bullish: bool = True) -> str:
        """
        Get appropriate color for a value
        
        Args:
            value (float): Value to get color for
            is_bullish (bool): Whether context is bullish
            
        Returns:
            str: Color code
        """
        if value > 0:
            return self.config.color_scheme['bullish' if is_bullish else 'bearish']
        elif value < 0:
            return self.config.color_scheme['bearish' if is_bullish else 'bullish']
        return self.config.color_scheme['neutral']
    
    def create_hover_template(self,
                            fields: List[Tuple[str, str]],
                            include_date: bool = True) -> str:
        """
        Create hover template with specified fields
        
        Args:
            fields (List[Tuple[str, str]]): List of (label, value) pairs
            include_date (bool): Whether to include date
            
        Returns:
            str: Hover template string
        """
        template = []
        if include_date:
            template.append('<b>Date</b>: %{x}<br>')
            
        for label, value in fields:
            template.append(f'<b>{label}</b>: {value}<br>')
            
        return ''.join(template)
    
    def add_range_selector(self, fig: go.Figure) -> go.Figure:
        """
        Add default range selector to figure
        
        Args:
            fig (go.Figure): Plotly figure
            
        Returns:
            go.Figure: Figure with range selector
        """
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            )
        )
        return fig
    
    def format_number(self, 
                     value: float, 
                     precision: int = 2,
                     prefix: str = '',
                     suffix: str = '') -> str:
        """
        Format number with default settings
        
        Args:
            value (float): Number to format
            precision (int): Decimal precision
            prefix (str): Prefix to add
            suffix (str): Suffix to add
            
        Returns:
            str: Formatted number
        """
        formatted = f"{value:,.{precision}f}"
        return f"{prefix}{formatted}{suffix}"
    
    def create_subplot_layout(self,
                            num_rows: int,
                            row_heights: List[float] = None,
                            shared_xaxis: bool = True) -> Dict[str, Any]:
        """
        Create subplot layout configuration
        
        Args:
            num_rows (int): Number of subplot rows
            row_heights (List[float]): Relative heights for each row
            shared_xaxis (bool): Whether to share x-axis across subplots
            
        Returns:
            Dict[str, Any]: Subplot layout configuration
        """
        heights = row_heights or [1/num_rows] * num_rows
        
        return dict(
            rows=num_rows,
            cols=1,
            shared_xaxes=shared_xaxis,
            vertical_spacing=self.config.layout['spacing']['vertical'],
            row_heights=heights
        )
    
    def style_axis(self,
                   fig: go.Figure,
                   title: str = '',
                   row: int = 1,
                   axis: str = 'y',
                   log_scale: bool = False) -> go.Figure:
        """
        Apply consistent axis styling
        
        Args:
            fig (go.Figure): Plotly figure
            title (str): Axis title
            row (int): Subplot row number
            axis (str): Axis to style ('x' or 'y')
            log_scale (bool): Whether to use log scale
            
        Returns:
            go.Figure: Styled figure
        """
        axis_dict = dict(
            title=title,
            title_font=dict(
                size=self.config.fonts['sizes']['axis'],
                color=self.config.color_scheme['text']
            ),
            showgrid=self.config.show_grid,
            gridcolor=self.config.grid_settings['color'],
            gridwidth=self.config.grid_settings['width'],
            type='log' if log_scale else 'linear'
        )
        
        if axis == 'x':
            fig.update_xaxes(axis_dict, row=row)
        else:
            fig.update_yaxes(axis_dict, row=row)
            
        return fig
    
    def add_patterns_overlay(self,
                           fig: go.Figure,
                           patterns: Dict[str, pd.Series],
                           row: int = 1) -> go.Figure:
        """
        Add pattern markers overlay to figure
        
        Args:
            fig (go.Figure): Plotly figure
            patterns (Dict[str, pd.Series]): Pattern signals
            row (int): Subplot row number
            
        Returns:
            go.Figure: Figure with patterns overlay
        """
        for pattern_name, signal in patterns.items():
            if isinstance(signal, tuple):
                # Handle bullish/bearish patterns
                bullish, bearish = signal
                
                # Add bullish markers
                if bullish.any():
                    fig.add_trace(
                        go.Scatter(
                            x=bullish.index[bullish],
                            y=self.df['Low'][bullish] * 0.99,
                            mode='markers',
                            marker=dict(
                                symbol='triangle-up',
                                size=10,
                                color=self.config.color_scheme['bullish']
                            ),
                            name=f'{pattern_name} (Bullish)',
                            opacity=self.config.pattern_opacity
                        ),
                        row=row, col=1
                    )
                
                # Add bearish markers
                if bearish.any():
                    fig.add_trace(
                        go.Scatter(
                            x=bearish.index[bearish],
                            y=self.df['High'][bearish] * 1.01,
                            mode='markers',
                            marker=dict(
                                symbol='triangle-down',
                                size=10,
                                color=self.config.color_scheme['bearish']
                            ),
                            name=f'{pattern_name} (Bearish)',
                            opacity=self.config.pattern_opacity
                        ),
                        row=row, col=1
                    )
            else:
                # Handle neutral patterns
                if signal.any():
                    fig.add_trace(
                        go.Scatter(
                            x=signal.index[signal],
                            y=self.df['Close'][signal],
                            mode='markers',
                            marker=dict(
                                symbol='circle',
                                size=8,
                                color=self.config.color_scheme['neutral']
                            ),
                            name=pattern_name,
                            opacity=self.config.pattern_opacity
                        ),
                        row=row, col=1
                    )
        
        return fig
    
    def create_color_scale(self,
                          start_color: str,
                          end_color: str,
                          n_colors: int = 10) -> List[str]:
        """
        Create a continuous color scale
        
        Args:
            start_color (str): Starting color
            end_color (str): Ending color
            n_colors (int): Number of colors in scale
            
        Returns:
            List[str]: List of color codes
        """
        def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
            return '#{:02x}{:02x}{:02x}'.format(*rgb)
        
        start_rgb = hex_to_rgb(start_color)
        end_rgb = hex_to_rgb(end_color)
        
        colors = []
        for i in range(n_colors):
            t = i / (n_colors - 1)
            rgb = tuple(int(start_rgb[j] + t * (end_rgb[j] - start_rgb[j])) 
                       for j in range(3))
            colors.append(rgb_to_hex(rgb))
        
        return colors
    
    def apply_interactive_features(self, fig: go.Figure) -> go.Figure:
        """
        Apply interactive features to figure
        
        Args:
            fig (go.Figure): Plotly figure
            
        Returns:
            go.Figure: Figure with interactive features
        """
        if self.config.interactive_settings['enabled']:
            fig.update_layout(
                hovermode='x unified',
                hoverlabel=dict(
                    bgcolor=self.config.interactive_settings['tooltip']['background_color'],
                    bordercolor=self.config.interactive_settings['tooltip']['border_color'],
                    font=dict(
                        family=self.config.fonts['family'],
                        size=self.config.fonts['sizes']['label']
                    )
                )
            )
            
            if self.config.interactive_settings['animation']['enabled']:
                fig.update_layout(
                    transition=dict(
                        duration=self.config.interactive_settings['animation']['duration'],
                        easing=self.config.interactive_settings['animation']['easing']
                    )
                )
        
        return fig

    
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

    def apply_theme(self, theme: VisualizationTheme) -> None:
        """
        Apply theme to visualizer
        
        Args:
            theme (VisualizationTheme): Theme to apply
        """
        self.config.color_scheme = theme.color_scheme
        self._update_theme_settings(theme)

    def _update_theme_settings(self, theme: VisualizationTheme) -> None:
        """
        Update visualization settings based on theme
        
        Args:
            theme (VisualizationTheme): Theme to apply
        """
        # Update figure layout settings
        self.fig.update_layout(
            plot_bgcolor=theme.color_scheme['background'],
            paper_bgcolor=theme.color_scheme['background'],
            font=dict(
                family=theme.font_settings['family'],
                size=theme.font_settings['size']['axis'],
                color=theme.color_scheme['text']
            ),
            title=dict(
                font=dict(
                    family=theme.font_settings['family'],
                    size=theme.font_settings['size']['title'],
                    color=theme.color_scheme['text']
                )
            ),
            showlegend=theme.chart_settings['show_legend']
        )


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
    
    def detect_failed_patterns(self, 
                             lookback_period: int = 50,
                             failure_threshold: float = 0.02) -> pd.DataFrame:
        """
        Identify failed pattern setups and their characteristics
        
        Args:
            lookback_period (int): Period to analyze for failed patterns
            failure_threshold (float): Price threshold for pattern failure
            
        Returns:
            pd.DataFrame: Failed patterns with metadata
        """
        pattern_methods = self._get_safe_pattern_methods()
        failed_patterns = []
        
        for pattern_name, pattern_func in pattern_methods.items():
            try:
                signals = self._get_pattern_signals(pattern_name)
                if signals is None:
                    continue
                    
                if isinstance(signals, tuple):
                    # Handle bullish/bearish patterns
                    for direction, signal in zip(['bullish', 'bearish'], signals):
                        failed = self._analyze_pattern_failure(
                            signal, 
                            direction, 
                            pattern_name,
                            failure_threshold
                        )
                        failed_patterns.extend(failed)
                else:
                    # Handle neutral patterns
                    failed = self._analyze_pattern_failure(
                        signals,
                        'neutral',
                        pattern_name,
                        failure_threshold
                    )
                    failed_patterns.extend(failed)
                    
            except Exception as e:
                print(f"Error analyzing failures for {pattern_name}: {str(e)}")
                continue
        
        if not failed_patterns:
            return pd.DataFrame()
            
        return pd.DataFrame(failed_patterns)

    def _analyze_pattern_failure(self,
                               signal: pd.Series,
                               direction: str,
                               pattern_name: str,
                               failure_threshold: float) -> List[Dict]:
        """
        Analyze individual pattern failures
        
        Args:
            signal (pd.Series): Pattern signal series
            direction (str): Pattern direction
            pattern_name (str): Name of the pattern
            failure_threshold (float): Threshold for failure
            
        Returns:
            List[Dict]: List of failed pattern instances
        """
        failed_instances = []
        
        # Get signal occurrences
        pattern_occurrences = signal[signal > 0].index
        
        for date in pattern_occurrences:
            try:
                # Get price data after pattern
                idx = self.df.index.get_loc(date)
                if idx + 10 >= len(self.df):  # Skip if too close to end
                    continue
                    
                post_pattern_data = self.df.iloc[idx:idx+10]
                
                # Determine failure based on direction
                failure = self._check_pattern_failure(
                    post_pattern_data,
                    direction,
                    failure_threshold
                )
                
                if failure['failed']:
                    failed_instances.append({
                        'date': date,
                        'pattern': pattern_name,
                        'direction': direction,
                        'failure_type': failure['type'],
                        'failure_magnitude': failure['magnitude'],
                        'bars_to_failure': failure['bars']
                    })
                    
            except Exception as e:
                print(f"Error analyzing pattern at {date}: {str(e)}")
                continue
        
        return failed_instances

    def _check_pattern_failure(self,
                            data: pd.DataFrame,
                            direction: str,
                            threshold: float) -> Dict[str, Any]:
        """
        Check if a pattern has failed
        
        Args:
            data (pd.DataFrame): Post-pattern price data
            direction (str): Expected pattern direction
            threshold (float): Failure threshold
            
        Returns:
            Dict[str, Any]: Failure analysis results
        """
        initial_price = data['Close'].iloc[0]
        max_move = data['High'].max() / initial_price - 1
        min_move = 1 - data['Low'].min() / initial_price
        
        failure_result = {
            'failed': False,
            'type': None,
            'magnitude': 0.0,
            'bars': 0
        }
        
        if direction == 'bullish':
            # Check for bullish pattern failure
            if min_move > threshold:  # Price moved significantly lower
                failure_result.update({
                    'failed': True,
                    'type': 'bearish_reversal',
                    'magnitude': min_move,
                    'bars': data['Low'].idxmin() - data.index[0]
                })
            elif max_move < threshold/2:  # Price didn't move up enough
                failure_result.update({
                    'failed': True,
                    'type': 'no_follow_through',
                    'magnitude': max_move,
                    'bars': len(data)
                })
                
        elif direction == 'bearish':
            # Check for bearish pattern failure
            if max_move > threshold:  # Price moved significantly higher
                failure_result.update({
                    'failed': True,
                    'type': 'bullish_reversal',
                    'magnitude': max_move,
                    'bars': data['High'].idxmax() - data.index[0]
                })
            elif min_move < threshold/2:  # Price didn't move down enough
                failure_result.update({
                    'failed': True,
                    'type': 'no_follow_through',
                    'magnitude': min_move,
                    'bars': len(data)
                })
                
        else:  # neutral patterns
            # Check for any significant move
            if max(max_move, min_move) < threshold/2:
                failure_result.update({
                    'failed': True,
                    'type': 'no_breakout',
                    'magnitude': max(max_move, min_move),
                    'bars': len(data)
                })
        
        return failure_result

    def analyze_failure_patterns(self, failed_patterns: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze characteristics of failed patterns
        
        Args:
            failed_patterns (pd.DataFrame): DataFrame of failed patterns
            
        Returns:
            Dict[str, Any]: Analysis of failure patterns
        """
        if failed_patterns.empty:
            return {}
            
        analysis = {
            'total_failures': len(failed_patterns),
            'failure_by_pattern': failed_patterns['pattern'].value_counts().to_dict(),
            'failure_by_type': failed_patterns['failure_type'].value_counts().to_dict(),
            'average_magnitude': failed_patterns['failure_magnitude'].mean(),
            'average_bars_to_failure': failed_patterns['bars_to_failure'].mean(),
            'direction_distribution': failed_patterns['direction'].value_counts().to_dict(),
            'pattern_specific_analysis': {}
        }
        
        # Analyze each pattern type separately
        for pattern in failed_patterns['pattern'].unique():
            pattern_fails = failed_patterns[failed_patterns['pattern'] == pattern]
            
            analysis['pattern_specific_analysis'][pattern] = {
                'count': len(pattern_fails),
                'average_magnitude': pattern_fails['failure_magnitude'].mean(),
                'common_failure_type': pattern_fails['failure_type'].mode().iloc[0],
                'direction_distribution': pattern_fails['direction'].value_counts().to_dict()
            }
        
        return analysis

    def get_failure_zones(self, 
                        failed_patterns: pd.DataFrame,
                        min_failures: int = 3) -> pd.DataFrame:
        """
        Identify price zones with frequent pattern failures
        
        Args:
            failed_patterns (pd.DataFrame): DataFrame of failed patterns
            min_failures (int): Minimum failures to identify a zone
            
        Returns:
            pd.DataFrame: Price zones with frequent failures
        """
        if failed_patterns.empty:
            return pd.DataFrame()
            
        failure_zones = []
        
        # Group failures by price level
        price_levels = pd.IntervalIndex.from_arrays(
            self.df['Low'],
            self.df['High'],
            closed='both'
        )
        
        for date in failed_patterns['date']:
            idx = self.df.index.get_loc(date)
            price_range = price_levels[idx]
            
            # Find overlapping failures
            overlapping = [
                zone for zone in failure_zones
                if (zone['price_low'] <= price_range.right and 
                    zone['price_high'] >= price_range.left)
            ]
            
            if overlapping:
                # Update existing zone
                zone = overlapping[0]
                zone['count'] += 1
                zone['price_low'] = min(zone['price_low'], price_range.left)
                zone['price_high'] = max(zone['price_high'], price_range.right)
                zone['patterns'].append(failed_patterns.loc[
                    failed_patterns['date'] == date, 'pattern'].iloc[0])
            else:
                # Create new zone
                failure_zones.append({
                    'price_low': price_range.left,
                    'price_high': price_range.right,
                    'count': 1,
                    'patterns': [failed_patterns.loc[
                        failed_patterns['date'] == date, 'pattern'].iloc[0]]
                })
        
        # Filter zones by minimum failures
        significant_zones = [
            zone for zone in failure_zones
            if zone['count'] >= min_failures
        ]
        
        return pd.DataFrame(significant_zones)

    def analyze_pattern_completion_probability(self,
                                            lookback_period: int = 100,
                                            min_occurrences: int = 5) -> pd.DataFrame:
        """
        Analyze completion probability for detected patterns
        
        Args:
            lookback_period (int): Historical period to analyze
            min_occurrences (int): Minimum pattern occurrences for analysis
            
        Returns:
            pd.DataFrame: Pattern completion statistics and probabilities
        """
        pattern_methods = self._get_safe_pattern_methods()
        completion_stats = []
        
        for pattern_name, pattern_func in pattern_methods.items():
            try:
                signals = self._get_pattern_signals(pattern_name)
                if signals is None:
                    continue
                    
                if isinstance(signals, tuple):
                    # Handle bullish/bearish patterns separately
                    for direction, signal in zip(['bullish', 'bearish'], signals):
                        stats = self._calculate_completion_stats(
                            signal,
                            pattern_name,
                            direction,
                            lookback_period
                        )
                        if stats['total_occurrences'] >= min_occurrences:
                            completion_stats.append(stats)
                else:
                    # Handle neutral patterns
                    stats = self._calculate_completion_stats(
                        signals,
                        pattern_name,
                        'neutral',
                        lookback_period
                    )
                    if stats['total_occurrences'] >= min_occurrences:
                        completion_stats.append(stats)
                        
            except Exception as e:
                print(f"Error analyzing completion for {pattern_name}: {str(e)}")
                continue
        
        return pd.DataFrame(completion_stats)

    def _calculate_completion_stats(self,
                                signal: pd.Series,
                                pattern_name: str,
                                direction: str,
                                lookback_period: int) -> Dict[str, Any]:
        """
        Calculate completion statistics for a specific pattern
        
        Args:
            signal (pd.Series): Pattern signal series
            pattern_name (str): Name of the pattern
            direction (str): Pattern direction
            lookback_period (int): Analysis period
            
        Returns:
            Dict[str, Any]: Completion statistics
        """
        # Initialize statistics
        stats = {
            'pattern_name': pattern_name,
            'direction': direction,
            'total_occurrences': 0,
            'successful_completions': 0,
            'failed_completions': 0,
            'partial_completions': 0,
            'avg_completion_time': 0,
            'avg_price_target_reach': 0,
            'completion_probability': 0.0,
            'risk_reward_ratio': 0.0
        }
        
        # Get pattern occurrences
        pattern_dates = signal[signal > 0].index
        completion_times = []
        target_reaches = []
        
        for date in pattern_dates:
            try:
                completion_data = self._analyze_pattern_completion(date, direction)
                
                stats['total_occurrences'] += 1
                
                if completion_data['completed']:
                    stats['successful_completions'] += 1
                    completion_times.append(completion_data['bars_to_completion'])
                    target_reaches.append(completion_data['target_reach_percentage'])
                elif completion_data['partial']:
                    stats['partial_completions'] += 1
                else:
                    stats['failed_completions'] += 1
                    
            except Exception as e:
                print(f"Error analyzing completion at {date}: {str(e)}")
                continue
        
        # Calculate averages and probabilities
        if stats['total_occurrences'] > 0:
            stats['completion_probability'] = (
                stats['successful_completions'] / stats['total_occurrences']
            )
            
            if completion_times:
                stats['avg_completion_time'] = np.mean(completion_times)
                stats['avg_price_target_reach'] = np.mean(target_reaches)
                
            # Calculate risk/reward
            stats['risk_reward_ratio'] = self._calculate_risk_reward_ratio(
                pattern_name, direction)
        
        return stats

    def _analyze_pattern_completion(self,
                                pattern_date: pd.Timestamp,
                                direction: str,
                                max_bars: int = 20) -> Dict[str, Any]:
        """
        Analyze completion data for a specific pattern instance
        
        Args:
            pattern_date (pd.Timestamp): Date of pattern occurrence
            direction (str): Pattern direction
            max_bars (int): Maximum bars to check for completion
            
        Returns:
            Dict[str, Any]: Completion analysis results
        """
        try:
            idx = self.df.index.get_loc(pattern_date)
            if idx + max_bars >= len(self.df):
                return {'completed': False, 'partial': False}
                
            # Get price data after pattern
            post_pattern_data = self.df.iloc[idx:idx+max_bars+1]
            initial_price = post_pattern_data['Close'].iloc[0]
            
            # Calculate expected targets and stops
            targets, stops = self._calculate_pattern_targets(
                post_pattern_data.iloc[0],
                direction
            )
            
            completion_data = {
                'completed': False,
                'partial': False,
                'bars_to_completion': max_bars,
                'target_reach_percentage': 0.0
            }
            
            # Analyze price movement
            for i, (_, prices) in enumerate(post_pattern_data.iterrows()):
                current_price = prices['Close']
                price_change = (current_price - initial_price) / initial_price
                
                if direction == 'bullish':
                    if current_price >= targets['first']:
                        completion_data.update({
                            'completed': True,
                            'bars_to_completion': i,
                            'target_reach_percentage': 100
                        })
                        break
                    elif current_price <= stops['initial']:
                        break
                    elif current_price >= targets['partial']:
                        completion_data['partial'] = True
                        completion_data['target_reach_percentage'] = (
                            (current_price - initial_price) / 
                            (targets['first'] - initial_price) * 100
                        )
                        
                elif direction == 'bearish':
                    if current_price <= targets['first']:
                        completion_data.update({
                            'completed': True,
                            'bars_to_completion': i,
                            'target_reach_percentage': 100
                        })
                        break
                    elif current_price >= stops['initial']:
                        break
                    elif current_price <= targets['partial']:
                        completion_data['partial'] = True
                        completion_data['target_reach_percentage'] = (
                            (initial_price - current_price) / 
                            (initial_price - targets['first']) * 100
                        )
            
            return completion_data
            
        except Exception as e:
            print(f"Error in completion analysis: {str(e)}")
            return {'completed': False, 'partial': False}

    def _calculate_pattern_targets(self,
                                pattern_bar: pd.Series,
                                direction: str) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Calculate price targets and stop levels for pattern
        
        Args:
            pattern_bar (pd.Series): Bar data where pattern occurred
            direction (str): Pattern direction ('bullish', 'bearish', 'neutral')
            
        Returns:
            Tuple[Dict[str, float], Dict[str, float]]: Targets and stops
        """
        # Get price levels
        high = pattern_bar['High']
        low = pattern_bar['Low']
        close = pattern_bar['Close']
        pattern_range = high - low
        
        if direction == 'bullish':
            targets = {
                'partial': close + pattern_range * 0.5,
                'first': close + pattern_range,
                'second': close + pattern_range * 1.5,
                'third': close + pattern_range * 2.0
            }
            stops = {
                'initial': low - pattern_range * 0.2,
                'breakeven': close,
                'trailing': close + pattern_range * 0.5
            }
        
        elif direction == 'bearish':
            targets = {
                'partial': close - pattern_range * 0.5,
                'first': close - pattern_range,
                'second': close - pattern_range * 1.5,
                'third': close - pattern_range * 2.0
            }
            stops = {
                'initial': high + pattern_range * 0.2,
                'breakeven': close,
                'trailing': close - pattern_range * 0.5
            }
        
        else:  # neutral
            targets = {
                'partial': close + (pattern_range * 0.3 * (1 if close > open else -1)),
                'first': close + (pattern_range * 0.6 * (1 if close > open else -1)),
                'second': close + (pattern_range * 0.9 * (1 if close > open else -1)),
                'third': close + (pattern_range * 1.2 * (1 if close > open else -1))
            }
            stops = {
                'initial': close - (pattern_range * 0.3 * (1 if close > open else -1)),
                'breakeven': close,
                'trailing': close + (pattern_range * 0.3 * (1 if close > open else -1))
            }
        
        return targets, stops

    def _calculate_risk_reward_ratio(self,
                                pattern_name: str,
                                direction: str) -> float:
        """
        Calculate risk/reward ratio for pattern
        
        Args:
            pattern_name (str): Name of the pattern
            direction (str): Pattern direction
            
        Returns:
            float: Risk/reward ratio
        """
        try:
            # Get recent pattern signals
            signals = self._get_pattern_signals(pattern_name)
            if signals is None:
                return 0.0
                
            if isinstance(signals, tuple):
                signal = signals[0] if direction == 'bullish' else signals[1]
            else:
                signal = signals
                
            # Get most recent occurrence
            recent_dates = signal[signal > 0].index[-10:]  # Last 10 occurrences
            
            if len(recent_dates) == 0:
                return 0.0
                
            # Calculate average R/R from recent patterns
            risk_rewards = []
            for date in recent_dates:
                idx = self.df.index.get_loc(date)
                pattern_bar = self.df.iloc[idx]
                
                targets, stops = self._calculate_pattern_targets(pattern_bar, direction)
                
                # Calculate R/R using first target and initial stop
                reward = abs(targets['first'] - pattern_bar['Close'])
                risk = abs(stops['initial'] - pattern_bar['Close'])
                
                if risk > 0:
                    risk_rewards.append(reward / risk)
            
            return np.mean(risk_rewards) if risk_rewards else 0.0
            
        except Exception as e:
            print(f"Error calculating R/R ratio: {str(e)}")
            return 0.0

    def analyze_historical_performance(self,
                                    min_occurrences: int = 20,
                                    lookback_days: int = 252) -> pd.DataFrame:
        """
        Analyze historical performance of detected patterns
        
        Args:
            min_occurrences (int): Minimum pattern occurrences for analysis
            lookback_days (int): Number of days to look back
            
        Returns:
            pd.DataFrame: Historical performance metrics for each pattern
        """
        performance_data = []
        pattern_methods = self._get_safe_pattern_methods()
        
        # Calculate market environment metrics for context
        market_volatility = self.df['Close'].pct_change().std() * np.sqrt(252)
        market_trend = self._calculate_market_trend()
        
        for pattern_name, pattern_func in pattern_methods.items():
            try:
                signals = self._get_pattern_signals(pattern_name)
                if signals is None:
                    continue
                    
                if isinstance(signals, tuple):
                    # Analyze bullish and bearish patterns separately
                    for direction, signal in zip(['bullish', 'bearish'], signals):
                        metrics = self._calculate_pattern_metrics(
                            signal, pattern_name, direction, lookback_days)
                        if metrics['occurrence_count'] >= min_occurrences:
                            metrics.update({
                                'market_volatility': market_volatility,
                                'market_trend': market_trend
                            })
                            performance_data.append(metrics)
                else:
                    # Analyze neutral patterns
                    metrics = self._calculate_pattern_metrics(
                        signals, pattern_name, 'neutral', lookback_days)
                    if metrics['occurrence_count'] >= min_occurrences:
                        metrics.update({
                            'market_volatility': market_volatility,
                            'market_trend': market_trend
                        })
                        performance_data.append(metrics)
                        
            except Exception as e:
                print(f"Error analyzing {pattern_name}: {str(e)}")
                continue
        
        return pd.DataFrame(performance_data)

    def _calculate_trade_metrics(self,
                            entry_date: pd.Timestamp,
                            direction: str,
                            max_holding_period: int = 20) -> Dict[str, float]:
        """
        Calculate metrics for a single trade/pattern occurrence
        
        Args:
            entry_date (pd.Timestamp): Pattern occurrence date
            direction (str): Pattern direction
            max_holding_period (int): Maximum bars to hold position
            
        Returns:
            Dict[str, float]: Trade metrics
        """
        try:
            entry_idx = self.df.index.get_loc(entry_date)
            if entry_idx + max_holding_period >= len(self.df):
                max_holding_period = len(self.df) - entry_idx - 1
                
            if max_holding_period <= 0:
                raise ValueError("Insufficient data for trade analysis")
                
            # Get trade data
            trade_data = self.df.iloc[entry_idx:entry_idx + max_holding_period + 1]
            entry_price = trade_data['Close'].iloc[0]
            
            # Calculate targets and stops
            targets, stops = self._calculate_pattern_targets(trade_data.iloc[0], direction)
            
            # Initialize metrics
            metrics = {
                'return': 0.0,
                'holding_period': max_holding_period,
                'max_adverse': 0.0,
                'max_favorable': 0.0,
                'exit_reason': 'time_exit'
            }
            
            # Track price movement
            for i, (_, prices) in enumerate(trade_data.iterrows()):
                if i == 0:  # Skip entry bar
                    continue
                    
                current_price = prices['Close']
                price_change = (current_price - entry_price) / entry_price
                
                # Update max moves
                if direction == 'bullish':
                    adverse_move = min(0, price_change)
                    favorable_move = max(0, price_change)
                else:  # bearish or neutral
                    adverse_move = max(0, price_change)
                    favorable_move = min(0, price_change)
                    
                metrics['max_adverse'] = min(metrics['max_adverse'], adverse_move)
                metrics['max_favorable'] = max(metrics['max_favorable'], favorable_move)
                
                # Check for exit conditions
                if self._check_exit_conditions(current_price, targets, stops, direction):
                    metrics.update({
                        'return': price_change,
                        'holding_period': i,
                        'exit_reason': 'target_or_stop'
                    })
                    break
                    
                # Update final return if no exit triggered
                if i == len(trade_data) - 1:
                    metrics['return'] = price_change
            
            return metrics
            
        except Exception as e:
            raise ValueError(f"Error calculating trade metrics: {str(e)}")

    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """
        Calculate Sharpe ratio for pattern returns
        
        Args:
            returns (List[float]): List of pattern returns
            
        Returns:
            float: Sharpe ratio
        """
        if not returns or len(returns) < 2:
            return 0.0
            
        returns_array = np.array(returns)
        avg_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        
        if std_return == 0:
            return 0.0
            
        # Annualize assuming daily returns
        sharpe = avg_return / std_return * np.sqrt(252)
        return sharpe

    def _calculate_profit_factor(self, returns: List[float]) -> float:
        """
        Calculate profit factor (gross profits / gross losses)
        
        Args:
            returns (List[float]): List of pattern returns
            
        Returns:
            float: Profit factor
        """
        if not returns:
            return 0.0
            
        profits = sum(r for r in returns if r > 0)
        losses = sum(abs(r) for r in returns if r < 0)
        
        if losses == 0:
            return 999.0 if profits > 0 else 0.0
            
        return profits / losses

    def _calculate_market_trend(self, window: int = 50) -> str:
        """
        Calculate overall market trend
        
        Args:
            window (int): Window for trend calculation
            
        Returns:
            str: Market trend classification
        """
        try:
            # Calculate moving averages
            sma_short = self.df['Close'].rolling(window=window//2).mean()
            sma_long = self.df['Close'].rolling(window=window).mean()
            
            # Get current values
            current_price = self.df['Close'].iloc[-1]
            current_short_ma = sma_short.iloc[-1]
            current_long_ma = sma_long.iloc[-1]
            
            # Calculate trend strength
            trend_strength = (current_price / current_long_ma - 1) * 100
            
            # Classify trend
            if current_short_ma > current_long_ma:
                if trend_strength > 5:
                    return 'strong_uptrend'
                else:
                    return 'uptrend'
            elif current_short_ma < current_long_ma:
                if trend_strength < -5:
                    return 'strong_downtrend'
                else:
                    return 'downtrend'
            else:
                return 'sideways'
                
        except Exception as e:
            print(f"Error calculating market trend: {str(e)}")
            return 'unknown'

    def _check_exit_conditions(self,
                            current_price: float,
                            targets: Dict[str, float],
                            stops: Dict[str, float],
                            direction: str) -> bool:
        """
        Check if exit conditions are met
        
        Args:
            current_price (float): Current price
            targets (Dict[str, float]): Target prices
            stops (Dict[str, float]): Stop prices
            direction (str): Trade direction
            
        Returns:
            bool: True if exit conditions met
        """
        if direction == 'bullish':
            return (current_price >= targets['first'] or 
                    current_price <= stops['initial'])
        elif direction == 'bearish':
            return (current_price <= targets['first'] or 
                    current_price >= stops['initial'])
        else:  # neutral
            return (abs(current_price - targets['first']) <= 
                    abs(current_price - stops['initial']))

    def _compare_volatility(self,
                        current_vol: float,
                        historical_vol: float,
                        tolerance: float = 0.3) -> float:
        """
        Compare current and historical volatility levels
        
        Args:
            current_vol (float): Current volatility
            historical_vol (float): Historical volatility
            tolerance (float): Maximum acceptable difference
            
        Returns:
            float: Similarity score between 0 and 1
        """
        if current_vol == 0 or historical_vol == 0:
            return 0.0
            
        vol_ratio = min(current_vol, historical_vol) / max(current_vol, historical_vol)
        
        # Scale similarity score based on ratio
        if vol_ratio >= (1 - tolerance):
            return 1.0
        elif vol_ratio <= (1 - 2 * tolerance):
            return 0.0
        else:
            return (vol_ratio - (1 - 2 * tolerance)) / tolerance

    def _compare_trend(self,
                    current_trend: str,
                    historical_trend: str) -> float:
        """
        Compare current and historical market trends
        
        Args:
            current_trend (str): Current trend classification
            historical_trend (str): Historical trend classification
            
        Returns:
            float: Similarity score between 0 and 1
        """
        trend_weights = {
            'strong_uptrend': 2,
            'uptrend': 1,
            'sideways': 0,
            'downtrend': -1,
            'strong_downtrend': -2
        }
        
        try:
            current_weight = trend_weights[current_trend]
            historical_weight = trend_weights[historical_trend]
            
            # Calculate similarity based on weight difference
            diff = abs(current_weight - historical_weight)
            if diff == 0:
                return 1.0
            elif diff == 1:
                return 0.7
            elif diff == 2:
                return 0.3
            else:
                return 0.0
                
        except KeyError:
            return 0.0

    def _compare_volume_profile(self,
                            current_profile: Dict[str, float],
                            historical_profile: Dict[str, float]) -> float:
        """
        Compare current and historical volume profiles
        
        Args:
            current_profile (Dict[str, float]): Current volume metrics
            historical_profile (Dict[str, float]): Historical volume metrics
            
        Returns:
            float: Similarity score between 0 and 1
        """
        similarity_scores = []
        
        for metric in ['relative_volume', 'volume_trend', 'volume_distribution']:
            try:
                current_value = current_profile[metric]
                historical_value = historical_profile[metric]
                
                if isinstance(current_value, str):
                    # Compare categorical values
                    similarity_scores.append(
                        1.0 if current_value == historical_value else 0.0
                    )
                else:
                    # Compare numerical values
                    ratio = min(current_value, historical_value) / max(current_value, historical_value)
                    similarity_scores.append(ratio)
                    
            except (KeyError, ZeroDivisionError):
                similarity_scores.append(0.0)
        
        return np.mean(similarity_scores)

    def _compare_momentum(self,
                        current_momentum: Dict[str, float],
                        historical_momentum: Dict[str, float]) -> float:
        """
        Compare current and historical momentum indicators
        
        Args:
            current_momentum (Dict[str, float]): Current momentum metrics
            historical_momentum (Dict[str, float]): Historical momentum metrics
            
        Returns:
            float: Similarity score between 0 and 1
        """
        indicator_weights = {
            'rsi': 0.3,
            'macd': 0.3,
            'momentum': 0.2,
            'rate_of_change': 0.2
        }
        
        weighted_similarity = 0.0
        total_weight = 0.0
        
        for indicator, weight in indicator_weights.items():
            try:
                current_value = current_momentum[indicator]
                historical_value = historical_momentum[indicator]
                
                # Calculate normalized difference
                if indicator == 'rsi':
                    # RSI is already normalized
                    similarity = 1 - abs(current_value - historical_value) / 100
                else:
                    # Normalize other indicators
                    max_val = max(abs(current_value), abs(historical_value))
                    if max_val == 0:
                        similarity = 1.0
                    else:
                        similarity = 1 - abs(current_value - historical_value) / max_val
                
                weighted_similarity += weight * similarity
                total_weight += weight
                
            except (KeyError, ZeroDivisionError):
                continue
        
        if total_weight == 0:
            return 0.0
            
        return weighted_similarity / total_weight

    def _calculate_momentum_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate various momentum indicators
        
        Args:
            data (pd.DataFrame): Price data
            
        Returns:
            Dict[str, float]: Momentum indicators
        """
        close_prices = data['Close']
        
        try:
            indicators = {
                'rsi': self._calculate_rsi(close_prices).iloc[-1],
                'macd': self._calculate_macd(close_prices).iloc[-1],
                'momentum': (close_prices.iloc[-1] / close_prices.iloc[-5] - 1) * 100,
                'rate_of_change': (close_prices.pct_change(5) * 100).iloc[-1]
            }
            
            return indicators
            
        except Exception as e:
            print(f"Error calculating momentum indicators: {str(e)}")
            return {
                'rsi': 50.0,
                'macd': 0.0,
                'momentum': 0.0,
                'rate_of_change': 0.0
            }

    def predict_pattern_breakout(self,
                            lookback_period: int = 100,
                            confidence_threshold: float = 0.7) -> pd.DataFrame:
        """
        Predict potential pattern breakouts/breakdowns based on historical behavior
        
        Args:
            lookback_period (int): Historical period to analyze
            confidence_threshold (float): Minimum confidence for predictions
            
        Returns:
            pd.DataFrame: Breakout predictions with confidence levels
        """
        predictions = []
        pattern_methods = self._get_safe_pattern_methods()
        
        # Get current market context
        current_context = self._get_market_context()
        
        for pattern_name, pattern_func in pattern_methods.items():
            try:
                signals = self._get_pattern_signals(pattern_name)
                if signals is None:
                    continue
                
                # Handle current active patterns
                active_patterns = self._get_active_patterns(signals, pattern_name)
                
                for pattern in active_patterns:
                    # Calculate breakout probability
                    breakout_prob = self._calculate_breakout_probability(
                        pattern,
                        lookback_period,
                        current_context
                    )
                    
                    if breakout_prob['confidence'] >= confidence_threshold:
                        predictions.append({
                            'pattern_name': pattern_name,
                            'direction': pattern['direction'],
                            'start_date': pattern['start_date'],
                            'confidence': breakout_prob['confidence'],
                            'expected_movement': breakout_prob['expected_movement'],
                            'target_price': breakout_prob['target_price'],
                            'stop_price': breakout_prob['stop_price'],
                            'timeframe': breakout_prob['timeframe']
                        })
                        
            except Exception as e:
                print(f"Error analyzing breakout for {pattern_name}: {str(e)}")
                continue
        
        return pd.DataFrame(predictions)

    def _get_market_context(self) -> Dict[str, Any]:
        """
        Get current market context for breakout analysis
        
        Returns:
            Dict[str, Any]: Market context metrics
        """
        latest_data = self.df.iloc[-20:]  # Last 20 bars
        
        return {
            'volatility': latest_data['Close'].pct_change().std() * np.sqrt(252),
            'trend': self._calculate_market_trend(),
            'volume_profile': self._analyze_volume_profile(latest_data),
            'support_resistance': self._identify_key_levels(latest_data),
            'momentum': self._calculate_momentum_indicators(latest_data)
        }

    def _get_active_patterns(self,
                            signals: Union[pd.Series, Tuple[pd.Series, pd.Series]],
                            pattern_name: str) -> List[Dict[str, Any]]:
        """
        Identify currently active patterns
        
        Args:
            signals: Pattern signals
            pattern_name (str): Name of the pattern
            
        Returns:
            List[Dict[str, Any]]: Active pattern information
        """
        active_patterns = []
        
        if isinstance(signals, tuple):
            bullish, bearish = signals
            # Check last 5 bars for active patterns
            for i in range(-5, 0):
                if bullish.iloc[i]:
                    active_patterns.append({
                        'direction': 'bullish',
                        'start_date': bullish.index[i],
                        'pattern': pattern_name
                    })
                if bearish.iloc[i]:
                    active_patterns.append({
                        'direction': 'bearish',
                        'start_date': bearish.index[i],
                        'pattern': pattern_name
                    })
        else:
            for i in range(-5, 0):
                if signals.iloc[i]:
                    active_patterns.append({
                        'direction': 'neutral',
                        'start_date': signals.index[i],
                        'pattern': pattern_name
                    })
        
        return active_patterns

        def _calculate_breakout_probability(self,
                                        pattern: Dict[str, Any],
                                        lookback_period: int,
                                        current_context: Dict[str, Any]) -> Dict[str, Any]:
            """
            Calculate probability of pattern breakout
            
            Args:
                pattern (Dict[str, Any]): Pattern information
                lookback_period (int): Historical period to analyze
                current_context (Dict[str, Any]): Current market context
                
            Returns:
                Dict[str, Any]: Breakout probability and targets
            """
            # Find similar historical patterns
            historical_patterns = self._find_similar_historical_patterns(
                pattern,
                lookback_period,
                current_context
            )
            
            if not historical_patterns:
                return {
                    'confidence': 0.0,
                    'expected_movement': 0.0,
                    'target_price': None,
                    'stop_price': None,
                    'timeframe': None
                }
            
            # Analyze historical outcomes
            successful_breakouts = 0
            movement_sizes = []
            breakout_times = []
            
            for hist_pattern in historical_patterns:
                outcome = self._analyze_pattern_outcome(hist_pattern)
                if outcome['success']:
                    successful_breakouts += 1
                    movement_sizes.append(outcome['movement_size'])
                    breakout_times.append(outcome['breakout_time'])
            
            if not movement_sizes:
                return {
                    'confidence': 0.0,
                    'expected_movement': 0.0,
                    'target_price': None,
                    'stop_price': None,
                    'timeframe': None
                }
            
            # Calculate probabilities and targets
            confidence = successful_breakouts / len(historical_patterns)
            avg_movement = np.mean(movement_sizes)
            avg_timeframe = int(np.mean(breakout_times))
            
            current_price = self.df['Close'].iloc[-1]
            if pattern['direction'] == 'bullish':
                target_price = current_price * (1 + avg_movement)
                stop_price = current_price * (1 - avg_movement * 0.5)
            else:
                target_price = current_price * (1 - avg_movement)
                stop_price = current_price * (1 + avg_movement * 0.5)
            
            return {
                'confidence': confidence,
                'expected_movement': avg_movement,
                'target_price': target_price,
                'stop_price': stop_price,
                'timeframe': avg_timeframe
            }
            
    def _find_similar_historical_patterns(self,
                                        current_pattern: Dict[str, Any],
                                        lookback_period: int,
                                        current_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Find historical patterns with similar characteristics
        
        Args:
            current_pattern (Dict[str, Any]): Current pattern information
            lookback_period (int): Historical period to analyze
            current_context (Dict[str, Any]): Current market context
            
        Returns:
            List[Dict[str, Any]]: Similar historical patterns
        """
        similar_patterns = []
        pattern_start_idx = self.df.index.get_loc(current_pattern['start_date'])
        
        # Look back from current pattern
        start_idx = max(0, pattern_start_idx - lookback_period)
        historical_slice = slice(start_idx, pattern_start_idx)
        
        # Get historical signals
        signals = self._get_pattern_signals(current_pattern['pattern'])
        if signals is None:
            return []
        
        if isinstance(signals, tuple):
            signal = (signals[0] if current_pattern['direction'] == 'bullish' 
                    else signals[1])
        else:
            signal = signals
            
        # Find historical occurrences
        historical_dates = signal.iloc[historical_slice][signal.iloc[historical_slice] > 0].index
        
        for date in historical_dates:
            try:
                # Get historical context
                hist_context = self._get_historical_context(date)
                
                # Check context similarity
                if self._is_context_similar(current_context, hist_context):
                    pattern_data = {
                        'date': date,
                        'direction': current_pattern['direction'],
                        'pattern': current_pattern['pattern'],
                        'context': hist_context
                    }
                    similar_patterns.append(pattern_data)
                    
            except Exception as e:
                print(f"Error analyzing historical pattern at {date}: {str(e)}")
                continue
        
        return similar_patterns

    def _get_historical_context(self, date: pd.Timestamp) -> Dict[str, Any]:
        """
        Get market context for a historical date
        
        Args:
            date (pd.Timestamp): Historical date
            
        Returns:
            Dict[str, Any]: Historical market context
        """
        idx = self.df.index.get_loc(date)
        historical_data = self.df.iloc[max(0, idx-20):idx+1]
        
        return {
            'volatility': historical_data['Close'].pct_change().std() * np.sqrt(252),
            'trend': self._calculate_historical_trend(historical_data),
            'volume_profile': self._analyze_volume_profile(historical_data),
            'support_resistance': self._identify_key_levels(historical_data),
            'momentum': self._calculate_momentum_indicators(historical_data)
        }

    def _is_context_similar(self,
                        current_context: Dict[str, Any],
                        historical_context: Dict[str, Any],
                        threshold: float = 0.7) -> bool:
        """
        Compare current and historical market contexts
        
        Args:
            current_context (Dict[str, Any]): Current market context
            historical_context (Dict[str, Any]): Historical market context
            threshold (float): Similarity threshold
            
        Returns:
            bool: True if contexts are similar
        """
        similarity_scores = {
            'volatility': self._compare_volatility(
                current_context['volatility'],
                historical_context['volatility']
            ),
            'trend': self._compare_trend(
                current_context['trend'],
                historical_context['trend']
            ),
            'volume': self._compare_volume_profile(
                current_context['volume_profile'],
                historical_context['volume_profile']
            ),
            'momentum': self._compare_momentum(
                current_context['momentum'],
                historical_context['momentum']
            )
        }
        
        avg_similarity = np.mean(list(similarity_scores.values()))
        return avg_similarity >= threshold

    def _analyze_pattern_outcome(self,
                            pattern: Dict[str, Any],
                            max_bars: int = 20) -> Dict[str, Any]:
        """
        Analyze the outcome of a historical pattern
        
        Args:
            pattern (Dict[str, Any]): Pattern information
            max_bars (int): Maximum bars to analyze
            
        Returns:
            Dict[str, Any]: Pattern outcome analysis
        """
        try:
            pattern_idx = self.df.index.get_loc(pattern['date'])
            if pattern_idx + max_bars >= len(self.df):
                max_bars = len(self.df) - pattern_idx - 1
                
            if max_bars <= 0:
                return {
                    'success': False,
                    'movement_size': 0.0,
                    'breakout_time': 0
                }
                
            # Get post-pattern data
            post_pattern_data = self.df.iloc[pattern_idx:pattern_idx + max_bars + 1]
            initial_price = post_pattern_data['Close'].iloc[0]
            
            # Get targets and stops
            targets, stops = self._calculate_pattern_targets(
                post_pattern_data.iloc[0],
                pattern['direction']
            )
            
            # Track price movement
            for i, (_, prices) in enumerate(post_pattern_data.iterrows()):
                if i == 0:  # Skip pattern bar
                    continue
                    
                current_price = prices['Close']
                price_change = (current_price - initial_price) / initial_price
                
                # Check if pattern reached target
                if pattern['direction'] == 'bullish':
                    if current_price >= targets['first']:
                        return {
                            'success': True,
                            'movement_size': price_change,
                            'breakout_time': i
                        }
                    elif current_price <= stops['initial']:
                        return {
                            'success': False,
                            'movement_size': price_change,
                            'breakout_time': i
                        }
                else:  # bearish or neutral
                    if current_price <= targets['first']:
                        return {
                            'success': True,
                            'movement_size': abs(price_change),
                            'breakout_time': i
                        }
                    elif current_price >= stops['initial']:
                        return {
                            'success': False,
                            'movement_size': abs(price_change),
                            'breakout_time': i
                        }
            
            # If no clear outcome
            return {
                'success': False,
                'movement_size': 0.0,
                'breakout_time': max_bars
            }
            
        except Exception as e:
            print(f"Error analyzing pattern outcome: {str(e)}")
            return {
                'success': False,
                'movement_size': 0.0,
                'breakout_time': 0
            }

    def _calculate_pattern_metrics(self,
                                signal: pd.Series,
                                pattern_name: str,
                                direction: str,
                                lookback_days: int) -> Dict[str, Any]:
        """
        Calculate performance metrics for a specific pattern
        
        Args:
            signal (pd.Series): Pattern signal series
            pattern_name (str): Name of the pattern
            direction (str): Pattern direction
            lookback_days (int): Analysis lookback period
            
        Returns:
            Dict[str, Any]: Pattern performance metrics
        """
        lookback_start = self.df.index[-1] - pd.Timedelta(days=lookback_days)
        pattern_occurrences = signal[signal > 0].index
        pattern_occurrences = pattern_occurrences[pattern_occurrences >= lookback_start]
        
        if len(pattern_occurrences) == 0:
            return {'pattern_name': pattern_name, 'occurrence_count': 0}
            
        returns = []
        holding_periods = []
        max_adverse_moves = []
        max_favorable_moves = []
        
        for date in pattern_occurrences:
            try:
                metrics = self._calculate_trade_metrics(date, direction)
                returns.append(metrics['return'])
                holding_periods.append(metrics['holding_period'])
                max_adverse_moves.append(metrics['max_adverse'])
                max_favorable_moves.append(metrics['max_favorable'])
            except Exception as e:
                print(f"Error calculating metrics for {date}: {str(e)}")
                continue
        
        return {
            'pattern_name': pattern_name,
            'direction': direction,
            'occurrence_count': len(pattern_occurrences),
            'avg_return': np.mean(returns) if returns else 0,
            'win_rate': np.mean([r > 0 for r in returns]) if returns else 0,
            'avg_holding_period': np.mean(holding_periods) if holding_periods else 0,
            'avg_adverse_move': np.mean(max_adverse_moves) if max_adverse_moves else 0,
            'avg_favorable_move': np.mean(max_favorable_moves) if max_favorable_moves else 0,
            'sharpe_ratio': self._calculate_sharpe_ratio(returns) if returns else 0,
            'profit_factor': self._calculate_profit_factor(returns) if returns else 0
        }

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

    def _find_similar_patterns(self,
                            pattern: Dict[str, Any],
                            lookback_period: int) -> List[Dict[str, Any]]:
        """
        Find similar historical patterns
        
        Args:
            pattern (Dict[str, Any]): Current pattern to compare
            lookback_period (int): Historical lookback period
            
        Returns:
            List[Dict[str, Any]]: List of similar historical patterns
        """
        similar_patterns = []
        pattern_start_idx = self.df.index.get_loc(pattern['start_date'])
        
        # Look back from the current pattern
        start_idx = max(0, pattern_start_idx - lookback_period)
        historical_slice = slice(start_idx, pattern_start_idx)
        
        # Get pattern detection results for the historical period
        historical_signals = self._get_pattern_signals(pattern['name'])
        
        if historical_signals is None:
            return []
            
        if isinstance(historical_signals, tuple):
            if pattern['direction'] == 'bullish':
                signals = historical_signals[0].iloc[historical_slice]
            else:
                signals = historical_signals[1].iloc[historical_slice]
        else:
            signals = historical_signals.iloc[historical_slice]
        
        # Find occurrences
        for idx in signals[signals].index:
            similar_pattern = self._extract_pattern_data(idx, pattern['name'])
            if similar_pattern:
                similar_patterns.append(similar_pattern)
        
        return similar_patterns


    def _interpolate_color(self, value: float, color_scale: List[List]) -> str:
        """
        Interpolate between colors based on value
        
        Args:
            value (float): Value between 0 and 1
            color_scale (List[List]): List of [position, color] pairs
            
        Returns:
            str: Interpolated color in rgba format
        """
        # Find the color positions that bound our value
        for i in range(len(color_scale) - 1):
            pos1, color1 = color_scale[i]
            pos2, color2 = color_scale[i + 1]
            
            if pos1 <= value <= pos2:
                # Extract rgba components
                rgba1 = self._parse_rgba(color1)
                rgba2 = self._parse_rgba(color2)
                
                # Calculate interpolation factor
                factor = (value - pos1) / (pos2 - pos1)
                
                # Interpolate each component
                r = rgba1[0] + (rgba2[0] - rgba1[0]) * factor
                g = rgba1[1] + (rgba2[1] - rgba1[1]) * factor
                b = rgba1[2] + (rgba2[2] - rgba1[2]) * factor
                a = rgba1[3] + (rgba2[3] - rgba1[3]) * factor
                
                return f'rgba({int(r)}, {int(g)}, {int(b)}, {a})'
                
        # Return the last color if value is out of range
        return color_scale[-1][1]

    def _parse_rgba(self, color_str: str) -> Tuple[float, float, float, float]:
        """
        Parse rgba color string into components
        
        Args:
            color_str (str): Color in rgba format
            
        Returns:
            Tuple[float, float, float, float]: RGBA components
        """
        # Remove rgba() and split components
        components = color_str.replace('rgba(', '').replace(')', '').split(',')
        return tuple(float(c.strip()) for c in components)

    def analyze_pattern_completion(self, 
                                pattern: Dict[str, Any],
                                lookback_period: int = 100) -> Dict[str, float]:
        """
        Analyze pattern completion probability and targets
        
        Args:
            pattern (Dict[str, Any]): Pattern information
            lookback_period (int): Historical lookback period
            
        Returns:
            Dict[str, float]: Completion analysis metrics
        """
        try:
            # Get historical patterns of the same type
            historical_patterns = self._find_similar_patterns(
                pattern, lookback_period)
            
            if not historical_patterns:
                return {
                    'completion_probability': 0.0,
                    'average_target_reach': 0.0,
                    'average_stop_hit': 0.0,
                    'risk_reward_ratio': 0.0
                }
                
            # Calculate completion metrics
            completions = []
            target_reaches = []
            stop_hits = []
            
            for hist_pattern in historical_patterns:
                completion_data = self._calculate_pattern_outcome(hist_pattern)
                completions.append(completion_data['completed'])
                target_reaches.append(completion_data['target_reached'])
                stop_hits.append(completion_data['stop_hit'])
            
            # Calculate probabilities
            completion_prob = sum(completions) / len(completions)
            target_prob = sum(target_reaches) / len(target_reaches)
            stop_prob = sum(stop_hits) / len(stop_hits)
            
            # Calculate risk/reward
            risk_reward = self._calculate_risk_reward(pattern)
            
            return {
                'completion_probability': completion_prob,
                'target_reach_probability': target_prob,
                'stop_hit_probability': stop_prob,
                'risk_reward_ratio': risk_reward
            }
            
        except Exception as e:
            print(f"Error analyzing pattern completion: {str(e)}")
            return {}

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
        
        # Add RSI - this is the part that needs fixing
        fig.add_trace(
            go.Scatter(
                x=self.df.index,
                y=rsi,
                name='RSI',
                line=dict(color='purple', width=1),
                # Essential: specify the subplot explicitly
                xaxis=f'x{row}',
                yaxis=f'y{row}'
            ),
            row=row, col=col
        )

        # Force RSI y-axis to have appropriate range
        fig.update_yaxes(
            title_text="RSI", 
            range=[0, 100],  # Force RSI range to be 0-100
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
        
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = abs(loss)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Avoid division by zero
        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
        
        # Calculate RSI
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
    
class MarketRegimeAnalyzer:
    """
    Analyzes market regimes and their transitions
    
    This class is responsible for detecting and analyzing different market regimes,
    including volatility states, trend phases, and volume profiles.
    
    Attributes:
        df (pd.DataFrame): DataFrame containing OHLCV data
        window_size (int): Default window size for calculations
        min_regime_duration (int): Minimum number of periods for a regime
        regime_cache (Dict): Cache for computed regime data
    """
    
    def __init__(self, 
                 df: pd.DataFrame, 
                 window_size: int = 20,
                 min_regime_duration: int = 5):
        """
        Initialize MarketRegimeAnalyzer
        
        Args:
            df (pd.DataFrame): OHLCV data
            window_size (int): Default window for calculations
            min_regime_duration (int): Minimum regime duration
            
        Raises:
            ValueError: If required columns are missing in DataFrame
        """
        self._validate_dataframe(df)
        self.df = df.copy()
        self.window_size = window_size
        self.min_regime_duration = min_regime_duration
        self.regime_cache = {}
        
        # Initialize technical analysis indicators
        self._initialize_indicators()

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """
        Validate input DataFrame has required columns
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Raises:
            ValueError: If required columns are missing
        """
        required_columns = ['Open', 'High', 'Low', 'Close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be DatetimeIndex")

    def _initialize_indicators(self) -> None:
        """Initialize technical indicators used in regime analysis"""
        # Pre-calculate common indicators used across methods
        self.sma_short = self.df['Close'].rolling(window=self.window_size).mean()
        self.sma_long = self.df['Close'].rolling(window=self.window_size * 2).mean()
        self.volatility = self.df['Close'].pct_change().rolling(window=self.window_size).std()
        
        if 'Volume' in self.df.columns:
            self.volume_ma = self.df['Volume'].rolling(window=self.window_size).mean()
            
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
    
    def filter_patterns_by_regime(self,
                                patterns: pd.DataFrame,
                                min_reliability: float = 0.6) -> pd.DataFrame:
        """
        Filter patterns based on current market regime reliability
        
        Args:
            patterns (pd.DataFrame): Detected patterns
            min_reliability (float): Minimum reliability threshold
            
        Returns:
            pd.DataFrame: Filtered patterns with regime-specific reliability
        """
        current_regime = self.analyze_market_regime()[-1]  # Get most recent regime
        
        # Get regime-specific pattern performance
        regime_performance = self._calculate_regime_pattern_performance(
            patterns, current_regime)
        
        # Filter patterns based on regime performance
        reliable_patterns = regime_performance[
            regime_performance['regime_reliability'] >= min_reliability
        ]
        
        return reliable_patterns

    def _calculate_regime_pattern_performance(self,
                                           patterns: pd.DataFrame,
                                           current_regime: MarketRegime) -> pd.DataFrame:
        """
        Calculate pattern performance specific to current regime
        
        Args:
            patterns (pd.DataFrame): Detected patterns
            current_regime (MarketRegime): Current market regime
            
        Returns:
            pd.DataFrame: Patterns with regime-specific performance metrics
        """
        performance_data = []
        
        for _, pattern in patterns.iterrows():
            # Find similar historical regimes
            similar_regimes = self._find_similar_regimes(current_regime)
            
            # Calculate pattern performance in similar regimes
            regime_stats = self._analyze_pattern_in_regimes(
                pattern, similar_regimes)
            
            performance_data.append({
                'pattern_name': pattern['pattern_name'],
                'direction': pattern.get('direction', 'neutral'),
                'regime_reliability': regime_stats['success_rate'],
                'regime_avg_return': regime_stats['avg_return'],
                'regime_win_rate': regime_stats['win_rate'],
                'confidence': regime_stats['confidence'],
                'regime_risk_reward': regime_stats['risk_reward']
            })
        
        return pd.DataFrame(performance_data)

    def _find_similar_regimes(self, 
                            current_regime: MarketRegime,
                            lookback_days: int = 252) -> List[MarketRegime]:
        """
        Find historical regimes similar to current regime
        
        Args:
            current_regime (MarketRegime): Current market regime
            lookback_days (int): Days to look back for similar regimes
            
        Returns:
            List[MarketRegime]: List of similar historical regimes
        """
        lookback_start = self.df.index[-1] - pd.Timedelta(days=lookback_days)
        historical_regimes = self.analyze_market_regime(
            start_date=lookback_start,
            end_date=self.df.index[-1]
        )
        
        similar_regimes = []
        
        for regime in historical_regimes:
            if self._calculate_regime_similarity(current_regime, regime) >= 0.7:
                similar_regimes.append(regime)
        
        return similar_regimes

    def _calculate_regime_similarity(self,
                                regime1: MarketRegime,
                                regime2: MarketRegime) -> float:
        """
        Calculate similarity between two market regimes
        
        Args:
            regime1 (MarketRegime): First regime
            regime2 (MarketRegime): Second regime
            
        Returns:
            float: Similarity score between 0 and 1
        """
        # Component weights
        weights = {
            'regime_type': 0.3,
            'volatility': 0.25,
            'trend': 0.25,
            'volume': 0.2
        }
        
        # Calculate component similarities
        similarities = {
            'regime_type': 1.0 if regime1.regime_type == regime2.regime_type else 0.0,
            'volatility': self._compare_regime_volatility(regime1.volatility, regime2.volatility),
            'trend': self._compare_regime_trend(regime1.trend, regime2.trend),
            'volume': self._compare_regime_volume(regime1.volume, regime2.volume)
        }
        
        # Calculate weighted similarity
        total_similarity = sum(weights[k] * similarities[k] for k in weights)
        
        return total_similarity

    def _analyze_pattern_in_regimes(self,
                                pattern: pd.Series,
                                regimes: List[MarketRegime]) -> Dict[str, float]:
        """
        Analyze pattern performance in given regimes
        
        Args:
            pattern (pd.Series): Pattern information
            regimes (List[MarketRegime]): List of market regimes
            
        Returns:
            Dict[str, float]: Performance statistics in these regimes
        """
        pattern_results = []
        
        for regime in regimes:
            # Get pattern occurrences during this regime
            regime_patterns = self._get_patterns_in_regime(
                pattern['pattern_name'],
                regime.start_date,
                regime.end_date
            )
            
            if regime_patterns:
                # Calculate performance metrics
                results = self._calculate_pattern_regime_metrics(regime_patterns)
                pattern_results.append(results)
        
        if not pattern_results:
            return {
                'success_rate': 0.0,
                'avg_return': 0.0,
                'win_rate': 0.0,
                'confidence': 0.0,
                'risk_reward': 0.0
            }
        
        # Aggregate results
        return {
            'success_rate': np.mean([r['success_rate'] for r in pattern_results]),
            'avg_return': np.mean([r['avg_return'] for r in pattern_results]),
            'win_rate': np.mean([r['win_rate'] for r in pattern_results]),
            'confidence': np.mean([r['confidence'] for r in pattern_results]),
            'risk_reward': np.mean([r['risk_reward'] for r in pattern_results])
        }

    def _get_patterns_in_regime(self,
                            pattern_name: str,
                            start_date: pd.Timestamp,
                            end_date: pd.Timestamp) -> List[Dict[str, Any]]:
        """
        Get pattern occurrences within a specific regime period
        
        Args:
            pattern_name (str): Name of the pattern
            start_date (pd.Timestamp): Regime start date
            end_date (pd.Timestamp): Regime end date
            
        Returns:
            List[Dict[str, Any]]: Pattern occurrences with outcomes
        """
        regime_slice = self.df[start_date:end_date]
        pattern_signals = self._get_pattern_signals(pattern_name, regime_slice)
        
        if pattern_signals is None:
            return []
        
        pattern_instances = []
        
        if isinstance(pattern_signals, tuple):
            bullish_signals, bearish_signals = pattern_signals
            # Process bullish patterns
            for date in bullish_signals[bullish_signals > 0].index:
                outcome = self._analyze_pattern_outcome(date, 'bullish')
                if outcome:
                    pattern_instances.append(outcome)
            
            # Process bearish patterns
            for date in bearish_signals[bearish_signals > 0].index:
                outcome = self._analyze_pattern_outcome(date, 'bearish')
                if outcome:
                    pattern_instances.append(outcome)
        else:
            # Process neutral patterns
            for date in pattern_signals[pattern_signals > 0].index:
                outcome = self._analyze_pattern_outcome(date, 'neutral')
                if outcome:
                    pattern_instances.append(outcome)
        
        return pattern_instances

    def calculate_regime_transitions(self, 
                               lookback_period: int = 252) -> pd.DataFrame:
        """
        Calculate probability of transitions between different market regimes
        
        Args:
            lookback_period (int): Historical period to analyze in days
            
        Returns:
            pd.DataFrame: Transition probability matrix
        """
        # Get historical regimes
        lookback_start = self.df.index[-1] - pd.Timedelta(days=lookback_period)
        historical_regimes = self.analyze_market_regime(
            start_date=lookback_start,
            end_date=self.df.index[-1]
        )
        
        if len(historical_regimes) < 2:
            return pd.DataFrame()
        
        # Create transition matrix
        transitions = defaultdict(lambda: defaultdict(int))
        total_transitions = defaultdict(int)
        
        # Count regime transitions
        for i in range(len(historical_regimes) - 1):
            current_regime = historical_regimes[i].regime_type
            next_regime = historical_regimes[i + 1].regime_type
            
            transitions[current_regime][next_regime] += 1
            total_transitions[current_regime] += 1
        
        # Calculate probabilities
        unique_regimes = set(r.regime_type for r in historical_regimes)
        prob_matrix = pd.DataFrame(0.0, 
                                index=unique_regimes, 
                                columns=unique_regimes)
        
        for current_regime in unique_regimes:
            if total_transitions[current_regime] > 0:
                for next_regime in unique_regimes:
                    prob = (transitions[current_regime][next_regime] / 
                        total_transitions[current_regime])
                    prob_matrix.loc[current_regime, next_regime] = prob
        
        return prob_matrix

    def predict_next_regime(self, 
                        current_regime: MarketRegime,
                        confidence_threshold: float = 0.6) -> Dict[str, Any]:
        """
        Predict the most likely next regime
        
        Args:
            current_regime (MarketRegime): Current market regime
            confidence_threshold (float): Minimum confidence for prediction
            
        Returns:
            Dict[str, Any]: Prediction details
        """
        # Get transition probabilities
        transition_matrix = self.calculate_regime_transitions()
        if transition_matrix.empty:
            return {
                'predicted_regime': None,
                'confidence': 0.0,
                'probabilities': {}
            }
        
        # Get transition probabilities for current regime
        if current_regime.regime_type not in transition_matrix.index:
            return {
                'predicted_regime': None,
                'confidence': 0.0,
                'probabilities': {}
            }
        
        regime_transitions = transition_matrix.loc[current_regime.regime_type]
        
        # Sort probabilities
        sorted_probs = regime_transitions.sort_values(ascending=False)
        highest_prob = sorted_probs.iloc[0]
        
        # Check if prediction meets confidence threshold
        if highest_prob >= confidence_threshold:
            prediction = {
                'predicted_regime': sorted_probs.index[0],
                'confidence': highest_prob,
                'probabilities': sorted_probs.to_dict()
            }
        else:
            prediction = {
                'predicted_regime': None,
                'confidence': highest_prob,
                'probabilities': sorted_probs.to_dict()
            }
        
        # Add transition timing estimate
        prediction['estimated_duration'] = self._estimate_regime_duration(
            current_regime.regime_type)
        
        return prediction

    def _estimate_regime_duration(self, 
                                regime_type: str,
                                lookback_days: int = 252) -> Dict[str, float]:
        """
        Estimate the likely duration of a regime
        
        Args:
            regime_type (str): Type of regime
            lookback_days (int): Historical period to analyze
            
        Returns:
            Dict[str, float]: Duration statistics
        """
        # Get historical regimes
        lookback_start = self.df.index[-1] - pd.Timedelta(days=lookback_days)
        historical_regimes = self.analyze_market_regime(
            start_date=lookback_start,
            end_date=self.df.index[-1]
        )
        
        # Calculate durations for the specified regime type
        durations = []
        
        for regime in historical_regimes:
            if regime.regime_type == regime_type:
                duration = (regime.end_date - regime.start_date).days
                durations.append(duration)
        
        if not durations:
            return {
                'avg_duration': 0,
                'min_duration': 0,
                'max_duration': 0,
                'std_duration': 0
            }
        
        return {
            'avg_duration': np.mean(durations),
            'min_duration': min(durations),
            'max_duration': max(durations),
            'std_duration': np.std(durations)
        }
        
    def analyze_regime_stability(self, 
                            current_regime: MarketRegime,
                            window_size: int = 20) -> Dict[str, float]:
        """
        Analyze stability of current market regime
        
        Args:
            current_regime (MarketRegime): Current market regime
            window_size (int): Analysis window size
            
        Returns:
            Dict[str, float]: Stability metrics
        """
        latest_data = self.df.iloc[-window_size:]
        
        stability_metrics = {
            'volatility_stability': self._analyze_volatility_stability(latest_data),
            'trend_stability': self._analyze_trend_stability(latest_data),
            'volume_stability': self._analyze_volume_stability(latest_data),
            'momentum_stability': self._analyze_momentum_stability(latest_data),
            'support_resistance_stability': self._analyze_sr_stability(latest_data)
        }
        
        # Calculate overall stability score
        weights = {
            'volatility_stability': 0.25,
            'trend_stability': 0.25,
            'volume_stability': 0.15,
            'momentum_stability': 0.20,
            'support_resistance_stability': 0.15
        }
        
        stability_metrics['overall_stability'] = sum(
            stability_metrics[k] * weights[k] for k in weights
        )
        
        return stability_metrics

    def analyze_transition_drivers(self, 
                                current_regime: MarketRegime,
                                window_size: int = 20) -> Dict[str, Any]:
        """
        Analyze potential drivers of regime transitions
        
        Args:
            current_regime (MarketRegime): Current market regime
            window_size (int): Analysis window size
            
        Returns:
            Dict[str, Any]: Transition driver analysis
        """
        latest_data = self.df.iloc[-window_size:]
        
        # Analyze various potential transition drivers
        drivers = {
            'volatility_pressure': self._analyze_volatility_pressure(latest_data),
            'trend_exhaustion': self._analyze_trend_exhaustion(latest_data),
            'volume_anomalies': self._analyze_volume_anomalies(latest_data),
            'momentum_divergence': self._analyze_momentum_divergence(latest_data),
            'support_resistance_tests': self._analyze_sr_tests(latest_data)
        }
        
        # Calculate transition risk score
        transition_risk = self._calculate_transition_risk(drivers)
        drivers['transition_risk'] = transition_risk
        
        # Identify most likely transition scenarios
        drivers['likely_scenarios'] = self._identify_transition_scenarios(
            current_regime, drivers)
        
        return drivers

    def _analyze_volatility_pressure(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze building volatility pressure
        
        Args:
            data (pd.DataFrame): Price data
            
        Returns:
            Dict[str, float]: Volatility pressure metrics
        """
        returns = data['Close'].pct_change()
        
        # Calculate volatility metrics
        current_vol = returns.std()
        historical_vol = returns.rolling(window=50).std().mean()
        
        # Calculate volatility compression/expansion
        bb_width = self._calculate_bollinger_bandwidth(data['Close'])
        compression_ratio = bb_width.iloc[-1] / bb_width.mean()
        
        # Detect volatility clusters
        high_vol_clusters = self._detect_volatility_clusters(returns)
        
        return {
            'current_volatility': current_vol,
            'volatility_ratio': current_vol / historical_vol,
            'compression_ratio': compression_ratio,
            'volatility_clusters': high_vol_clusters,
            'pressure_score': self._calculate_volatility_pressure_score(
                current_vol, historical_vol, compression_ratio)
        }

    def _analyze_trend_exhaustion(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze signs of trend exhaustion
        
        Args:
            data (pd.DataFrame): Price data
            
        Returns:
            Dict[str, float]: Trend exhaustion metrics
        """
        # Calculate trend metrics
        price = data['Close']
        returns = price.pct_change()
        
        # Calculate momentum
        rsi = self._calculate_rsi(price)
        macd, signal = self._calculate_macd_with_signal(price)
        
        # Detect momentum divergence
        price_trend = price.iloc[-1] > price.iloc[-5]  # Simple 5-period trend
        momentum_trend = rsi.iloc[-1] > rsi.iloc[-5]
        
        # Calculate trend strength
        trend_strength = abs(price.iloc[-1] / price.mean() - 1)
        
        # Detect failed swings
        failed_swings = self._detect_failed_swings(data)
        
        return {
            'momentum_divergence': price_trend != momentum_trend,
            'trend_strength': trend_strength,
            'failed_swings': failed_swings,
            'exhaustion_score': self._calculate_exhaustion_score(
                price_trend, momentum_trend, trend_strength, failed_swings)
        }
        
    def _calculate_volatility_pressure_score(self,
                                        current_vol: float,
                                        historical_vol: float,
                                        compression_ratio: float) -> float:
        """
        Calculate volatility pressure score
        
        Args:
            current_vol (float): Current volatility
            historical_vol (float): Historical volatility
            compression_ratio (float): Bollinger Band compression ratio
            
        Returns:
            float: Pressure score between 0 and 1
        """
        # Score components
        vol_ratio_score = min(1.0, current_vol / historical_vol)
        compression_score = min(1.0, 1 / compression_ratio)
        
        # Combine scores with weights
        pressure_score = 0.6 * vol_ratio_score + 0.4 * compression_score
        
        return min(1.0, pressure_score)

    def _calculate_exhaustion_score(self,
                                price_trend: bool,
                                momentum_trend: bool,
                                trend_strength: float,
                                failed_swings: int) -> float:
        """
        Calculate trend exhaustion score
        
        Args:
            price_trend (bool): Current price trend direction
            momentum_trend (bool): Current momentum trend direction
            trend_strength (float): Strength of current trend
            failed_swings (int): Number of failed swing patterns
            
        Returns:
            float: Exhaustion score between 0 and 1
        """
        # Base score from trend strength
        base_score = min(1.0, trend_strength)
        
        # Adjust for divergence
        if price_trend != momentum_trend:
            base_score += 0.2
        
        # Adjust for failed swings
        swing_penalty = min(0.3, failed_swings * 0.1)
        base_score += swing_penalty
        
        return min(1.0, base_score)

    def _calculate_volume_anomaly_score(self,
                                    relative_volume: float,
                                    volume_spikes: int,
                                    price_volume_correlation: float) -> float:
        """
        Calculate volume anomaly score
        
        Args:
            relative_volume (float): Current relative volume
            volume_spikes (int): Number of volume spikes
            price_volume_correlation (float): Price-volume correlation
            
        Returns:
            float: Anomaly score between 0 and 1
        """
        # Score components
        volume_score = min(1.0, relative_volume / 3)
        spike_score = min(1.0, volume_spikes / 5)
        correlation_score = abs(price_volume_correlation)
        
        # Combine scores
        anomaly_score = (0.4 * volume_score + 
                        0.4 * spike_score + 
                        0.2 * correlation_score)
        
        return min(1.0, anomaly_score)

    def _calculate_divergence_score(self,
                                regular_divergence: Dict[str, bool],
                                hidden_divergence: Dict[str, bool]) -> float:
        """
        Calculate overall divergence score
        
        Args:
            regular_divergence (Dict[str, bool]): Regular divergence signals
            hidden_divergence (Dict[str, bool]): Hidden divergence signals
            
        Returns:
            float: Divergence score between 0 and 1
        """
        # Count divergence signals
        regular_count = sum(1 for v in regular_divergence.values() if v)
        hidden_count = sum(1 for v in hidden_divergence.values() if v)
        
        # Weight different types of divergences
        score = (0.6 * min(1.0, regular_count / 2) + 
                0.4 * min(1.0, hidden_count / 2))
        
        return min(1.0, score)

    def _detect_volatility_clusters(self, returns: pd.Series) -> int:
        """
        Detect clusters of high volatility
        
        Args:
            returns (pd.Series): Price returns
            
        Returns:
            int: Number of volatility clusters
        """
        # Calculate rolling volatility
        rolling_vol = returns.rolling(window=5).std()
        
        # Define high volatility threshold (e.g., 2 standard deviations)
        vol_threshold = rolling_vol.mean() + 2 * rolling_vol.std()
        
        # Identify high volatility periods
        high_vol_periods = rolling_vol > vol_threshold
        
        # Count clusters (consecutive high volatility periods)
        clusters = 0
        in_cluster = False
        
        for is_high_vol in high_vol_periods:
            if is_high_vol and not in_cluster:
                clusters += 1
                in_cluster = True
            elif not is_high_vol:
                in_cluster = False
        
        return clusters

    def _detect_failed_swings(self, data: pd.DataFrame) -> int:
        """
        Detect failed swing patterns
        
        Args:
            data (pd.DataFrame): Price data
            
        Returns:
            int: Number of failed swings
        """
        failed_swings = 0
        highs = self._find_swing_highs(data)
        lows = self._find_swing_lows(data)
        
        # Analyze recent swing points
        recent_highs = highs.iloc[-3:]
        recent_lows = lows.iloc[-3:]
        
        # Check for failed highs (lower highs)
        if len(recent_highs) >= 2:
            if recent_highs.iloc[-1] < recent_highs.iloc[-2]:
                failed_swings += 1
        
        # Check for failed lows (higher lows)
        if len(recent_lows) >= 2:
            if recent_lows.iloc[-1] > recent_lows.iloc[-2]:
                failed_swings += 1
        
        return failed_swings
    
    def _detect_regular_divergence(self,
                                price_highs: pd.Series,
                                price_lows: pd.Series,
                                indicator_highs: pd.Series,
                                indicator_lows: pd.Series) -> Dict[str, bool]:
        """
        Detect regular bullish and bearish divergences
        
        Args:
            price_highs (pd.Series): Series of price swing highs
            price_lows (pd.Series): Series of price swing lows
            indicator_highs (pd.Series): Series of indicator swing highs
            indicator_lows (pd.Series): Series of indicator swing lows
            
        Returns:
            Dict[str, bool]: Regular divergence signals
        """
        divergences = {
            'bullish': False,  # Price making lower lows, indicator making higher lows
            'bearish': False   # Price making higher highs, indicator making lower highs
        }
        
        # Check for bearish divergence (last 2 swing highs)
        if len(price_highs) >= 2 and len(indicator_highs) >= 2:
            price_higher = price_highs.iloc[-1] > price_highs.iloc[-2]
            indicator_lower = indicator_highs.iloc[-1] < indicator_highs.iloc[-2]
            
            if price_higher and indicator_lower:
                divergences['bearish'] = True
        
        # Check for bullish divergence (last 2 swing lows)
        if len(price_lows) >= 2 and len(indicator_lows) >= 2:
            price_lower = price_lows.iloc[-1] < price_lows.iloc[-2]
            indicator_higher = indicator_lows.iloc[-1] > indicator_lows.iloc[-2]
            
            if price_lower and indicator_higher:
                divergences['bullish'] = True
        
        return divergences

    def _detect_hidden_divergence(self,
                                price_highs: pd.Series,
                                price_lows: pd.Series,
                                indicator_highs: pd.Series,
                                indicator_lows: pd.Series) -> Dict[str, bool]:
        """
        Detect hidden bullish and bearish divergences
        
        Args:
            price_highs (pd.Series): Series of price swing highs
            price_lows (pd.Series): Series of price swing lows
            indicator_highs (pd.Series): Series of indicator swing highs
            indicator_lows (pd.Series): Series of indicator swing lows
            
        Returns:
            Dict[str, bool]: Hidden divergence signals
        """
        divergences = {
            'bullish': False,  # Price making higher lows, indicator making lower lows
            'bearish': False   # Price making lower highs, indicator making higher highs
        }
        
        # Check for hidden bearish divergence
        if len(price_highs) >= 2 and len(indicator_highs) >= 2:
            price_lower = price_highs.iloc[-1] < price_highs.iloc[-2]
            indicator_higher = indicator_highs.iloc[-1] > indicator_highs.iloc[-2]
            
            if price_lower and indicator_higher:
                divergences['bearish'] = True
        
        # Check for hidden bullish divergence
        if len(price_lows) >= 2 and len(indicator_lows) >= 2:
            price_higher = price_lows.iloc[-1] > price_lows.iloc[-2]
            indicator_lower = indicator_lows.iloc[-1] < indicator_lows.iloc[-2]
            
            if price_higher and indicator_lower:
                divergences['bullish'] = True
        
        return divergences

    def _detect_volume_climax(self, data: pd.DataFrame) -> Dict[str, bool]:
        """
        Detect volume climax patterns
        
        Args:
            data (pd.DataFrame): Price and volume data
            
        Returns:
            Dict[str, bool]: Volume climax signals
        """
        if 'Volume' not in data.columns:
            return {'buying_climax': False, 'selling_climax': False}
        
        # Calculate volume metrics
        volume = data['Volume']
        price = data['Close']
        volume_ma = volume.rolling(window=20).mean()
        
        # Define climax conditions
        high_volume = volume.iloc[-1] > 2 * volume_ma.iloc[-1]
        price_change = price.pct_change()
        
        signals = {
            'buying_climax': False,
            'selling_climax': False
        }
        
        # Check for buying climax
        if high_volume and price_change.iloc[-1] > 0:
            prior_trend = price_change.iloc[-20:-1].mean() > 0
            if prior_trend:
                signals['buying_climax'] = True
        
        # Check for selling climax
        if high_volume and price_change.iloc[-1] < 0:
            prior_trend = price_change.iloc[-20:-1].mean() < 0
            if prior_trend:
                signals['selling_climax'] = True
        
        return signals

    def _analyze_volume_distribution(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze volume distribution patterns
        
        Args:
            data (pd.DataFrame): Price and volume data
            
        Returns:
            Dict[str, float]: Volume distribution metrics
        """
        if 'Volume' not in data.columns:
            return {}
        
        volume = data['Volume']
        price = data['Close']
        
        # Calculate volume-weighted metrics
        vwap = (price * volume).cumsum() / volume.cumsum()
        volume_profile = pd.cut(price, bins=10).value_counts()
        
        # Calculate volume concentration
        volume_std = volume.std() / volume.mean()
        concentration = volume_profile.max() / volume_profile.sum()
        
        # Analyze up/down volume
        up_volume = volume[price > price.shift(1)].sum()
        down_volume = volume[price < price.shift(1)].sum()
        
        return {
            'vwap': vwap.iloc[-1],
            'volume_std': volume_std,
            'concentration': concentration,
            'up_down_ratio': up_volume / down_volume if down_volume > 0 else float('inf'),
            'above_vwap_ratio': len(price[price > vwap]) / len(price)
        }

    def _analyze_volume_anomalies(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze volume anomalies and patterns
        
        Args:
            data (pd.DataFrame): Price data
            
        Returns:
            Dict[str, float]: Volume anomaly metrics
        """
        if 'Volume' not in data.columns:
            return {'anomaly_score': 0.5}
        
        volume = data['Volume']
        price = data['Close']
        
        # Calculate volume metrics
        volume_ma = volume.rolling(window=10).mean()
        relative_volume = volume / volume_ma
        
        # Detect volume spikes
        volume_spikes = (relative_volume > 2).sum()
        
        # Analyze price-volume relationship
        price_volume_correlation = price.corr(volume)
        
        # Detect volume climax
        volume_climax = self._detect_volume_climax(data)
        
        return {
            'relative_volume': relative_volume.iloc[-1],
            'volume_spikes': volume_spikes,
            'price_volume_correlation': price_volume_correlation,
            'volume_climax': volume_climax,
            'anomaly_score': self._calculate_volume_anomaly_score(
                relative_volume.iloc[-1], volume_spikes, price_volume_correlation)
        }

    def _analyze_momentum_divergence(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze momentum divergences
        
        Args:
            data (pd.DataFrame): Price data
            
        Returns:
            Dict[str, float]: Momentum divergence metrics
        """
        price = data['Close']
        
        # Calculate various momentum indicators
        rsi = self._calculate_rsi(price)
        macd, signal = self._calculate_macd_with_signal(price)
        
        # Calculate price swings
        price_highs = self._find_swing_highs(data)
        price_lows = self._find_swing_lows(data)
        
        # Calculate momentum swings
        rsi_highs = self._find_swing_highs(pd.DataFrame({'Close': rsi}))
        rsi_lows = self._find_swing_lows(pd.DataFrame({'Close': rsi}))
        
        # Detect regular and hidden divergences
        regular_divergence = self._detect_regular_divergence(
            price_highs, price_lows, rsi_highs, rsi_lows)
        hidden_divergence = self._detect_hidden_divergence(
            price_highs, price_lows, rsi_highs, rsi_lows)
        
        return {
            'regular_divergence': regular_divergence,
            'hidden_divergence': hidden_divergence,
            'rsi_trend': rsi.iloc[-1] - rsi.iloc[-5],
            'macd_trend': macd.iloc[-1] - macd.iloc[-5],
            'divergence_score': self._calculate_divergence_score(
                regular_divergence, hidden_divergence)
        }

    def _analyze_volatility_stability(self, data: pd.DataFrame) -> float:
        """
        Analyze stability of volatility
        
        Args:
            data (pd.DataFrame): Price data
            
        Returns:
            float: Volatility stability score between 0 and 1
        """
        # Calculate rolling volatility
        returns = data['Close'].pct_change()
        rolling_vol = returns.rolling(window=5).std()
        
        # Calculate volatility of volatility
        vol_of_vol = rolling_vol.std()
        
        # Convert to stability score (lower vol of vol = higher stability)
        stability = 1 / (1 + vol_of_vol)
        
        return min(1.0, stability)

    def _analyze_trend_stability(self, data: pd.DataFrame) -> float:
        """
        Analyze stability of price trend
        
        Args:
            data (pd.DataFrame): Price data
            
        Returns:
            float: Trend stability score between 0 and 1
        """
        # Calculate various trend indicators
        sma20 = data['Close'].rolling(window=20).mean()
        sma50 = data['Close'].rolling(window=50).mean()
        
        # Calculate trend direction changes
        trend_changes = ((data['Close'] > sma20) != 
                        (data['Close'].shift(1) > sma20)).sum()
        
        # Calculate slope consistency
        price_changes = data['Close'].diff()
        slope_changes = (price_changes > 0) != (price_changes.shift(1) > 0)
        slope_consistency = 1 - slope_changes.sum() / len(data)
        
        # Combine metrics
        stability = (0.6 * slope_consistency + 
                    0.4 * (1 - trend_changes / len(data)))
        
        return min(1.0, stability)

    def _analyze_volume_stability(self, data: pd.DataFrame) -> float:
        """
        Analyze stability of trading volume
        
        Args:
            data (pd.DataFrame): Price data
            
        Returns:
            float: Volume stability score between 0 and 1
        """
        if 'Volume' not in data.columns:
            return 0.5
        
        # Calculate volume metrics
        volume = data['Volume']
        volume_ma = volume.rolling(window=10).mean()
        
        # Calculate volume volatility
        volume_volatility = volume.std() / volume.mean()
        
        # Calculate consistency of volume trend
        volume_trend_changes = ((volume > volume_ma) != 
                            (volume.shift(1) > volume_ma)).sum()
        
        # Combine metrics
        stability = (0.5 * (1 / (1 + volume_volatility)) + 
                    0.5 * (1 - volume_trend_changes / len(data)))
        
        return min(1.0, stability)

    def _analyze_momentum_stability(self, data: pd.DataFrame) -> float:
        """
        Analyze stability of momentum indicators
        
        Args:
            data (pd.DataFrame): Price data
            
        Returns:
            float: Momentum stability score between 0 and 1
        """
        # Calculate momentum indicators
        rsi = self._calculate_rsi(data['Close'])
        macd, signal = self._calculate_macd_with_signal(data['Close'])
        
        # Calculate RSI stability
        rsi_volatility = rsi.std()
        rsi_stability = 1 / (1 + rsi_volatility / 20)  # Normalize by typical RSI range
        
        # Calculate MACD stability
        macd_crossovers = ((macd > signal) != 
                        (macd.shift(1) > signal.shift(1))).sum()
        macd_stability = 1 - macd_crossovers / len(data)
        
        # Combine metrics
        stability = 0.5 * (rsi_stability + macd_stability)
        
        return min(1.0, stability)


    def _calculate_transition_risk(self, drivers: Dict[str, Any]) -> float:
        """
        Calculate overall regime transition risk
        
        Args:
            drivers (Dict[str, Any]): Analysis of transition drivers
            
        Returns:
            float: Transition risk score between 0 and 1
        """
        # Weight different risk components
        weights = {
            'volatility_pressure': 0.25,
            'trend_exhaustion': 0.25,
            'volume_anomalies': 0.20,
            'momentum_divergence': 0.20,
            'support_resistance_tests': 0.10
        }
        
        risk_score = 0.0
        for component, weight in weights.items():
            if component in drivers:
                if isinstance(drivers[component], dict) and 'pressure_score' in drivers[component]:
                    risk_score += weight * drivers[component]['pressure_score']
                elif isinstance(drivers[component], dict) and 'anomaly_score' in drivers[component]:
                    risk_score += weight * drivers[component]['anomaly_score']
                elif isinstance(drivers[component], float):
                    risk_score += weight * drivers[component]
        
        return min(1.0, risk_score)

    def _identify_transition_scenarios(self,
                                    current_regime: MarketRegime,
                                    drivers: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Identify likely regime transition scenarios
        
        Args:
            current_regime (MarketRegime): Current market regime
            drivers (Dict[str, Any]): Analysis of transition drivers
            
        Returns:
            List[Dict[str, Any]]: Potential transition scenarios with probabilities
        """
        scenarios = []
        transition_matrix = self.calculate_regime_transitions()
        
        if transition_matrix.empty:
            return scenarios
        
        # Get historical transitions from current regime
        if current_regime.regime_type in transition_matrix.index:
            possible_transitions = transition_matrix.loc[current_regime.regime_type]
            
            for next_regime, base_prob in possible_transitions.items():
                if base_prob > 0:
                    # Adjust probability based on current drivers
                    adjusted_prob = self._adjust_transition_probability(
                        base_prob,
                        current_regime.regime_type,
                        next_regime,
                        drivers
                    )
                    
                    if adjusted_prob >= 0.1:  # Only include significant possibilities
                        scenarios.append({
                            'from_regime': current_regime.regime_type,
                            'to_regime': next_regime,
                            'probability': adjusted_prob,
                            'estimated_timeframe': self._estimate_transition_timeframe(
                                current_regime.regime_type,
                                next_regime,
                                drivers
                            ),
                            'key_drivers': self._identify_key_drivers(
                                current_regime.regime_type,
                                next_regime,
                                drivers
                            )
                        })
        
        # Sort by probability
        scenarios.sort(key=lambda x: x['probability'], reverse=True)
        return scenarios

    def _adjust_transition_probability(self,
                                    base_prob: float,
                                    current_regime: str,
                                    target_regime: str,
                                    drivers: Dict[str, Any]) -> float:
        """
        Adjust transition probability based on current market drivers
        
        Args:
            base_prob (float): Base transition probability
            current_regime (str): Current regime type
            target_regime (str): Target regime type
            drivers (Dict[str, Any]): Current market drivers
            
        Returns:
            float: Adjusted probability
        """
        adjustment = 0.0
        
        # Adjust based on volatility pressure
        if 'volatility_pressure' in drivers:
            vol_pressure = drivers['volatility_pressure'].get('pressure_score', 0)
            if target_regime in ['high_volatility', 'crisis']:
                adjustment += 0.2 * vol_pressure
            elif current_regime in ['high_volatility', 'crisis']:
                adjustment -= 0.1 * vol_pressure
        
        # Adjust based on trend exhaustion
        if 'trend_exhaustion' in drivers:
            exhaustion = drivers['trend_exhaustion'].get('exhaustion_score', 0)
            if current_regime in ['trending', 'strong_trend']:
                adjustment += 0.15 * exhaustion
        
        # Adjust based on volume anomalies
        if 'volume_anomalies' in drivers:
            vol_anomaly = drivers['volume_anomalies'].get('anomaly_score', 0)
            adjustment += 0.1 * vol_anomaly
        
        # Calculate final probability
        adjusted_prob = base_prob * (1 + adjustment)
        return min(1.0, max(0.0, adjusted_prob))

    def _estimate_transition_timeframe(self,
                                    current_regime: str,
                                    target_regime: str,
                                    drivers: Dict[str, Any]) -> Dict[str, int]:
        """
        Estimate timeframe for regime transition
        
        Args:
            current_regime (str): Current regime type
            target_regime (str): Target regime type
            drivers (Dict[str, Any]): Current market drivers
            
        Returns:
            Dict[str, int]: Estimated timeframe ranges
        """
        # Get historical transition durations
        historical_durations = self._get_historical_transition_durations(
            current_regime, target_regime)
        
        if not historical_durations:
            return {
                'min_bars': 0,
                'max_bars': 0,
                'typical_bars': 0
            }
        
        # Calculate base estimates
        typical_duration = int(np.median(historical_durations))
        min_duration = int(min(historical_durations))
        max_duration = int(max(historical_durations))
        
        # Adjust based on current drivers
        transition_risk = drivers.get('transition_risk', 0.5)
        adjustment_factor = 1 - transition_risk  # Higher risk = faster transition
        
        return {
            'min_bars': max(1, int(min_duration * adjustment_factor)),
            'max_bars': max(2, int(max_duration * adjustment_factor)),
            'typical_bars': max(1, int(typical_duration * adjustment_factor))
        }

    def _identify_key_drivers(self,
                            current_regime: str,
                            target_regime: str,
                            drivers: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Identify key drivers for specific regime transition
        
        Args:
            current_regime (str): Current regime type
            target_regime (str): Target regime type
            drivers (Dict[str, Any]): Current market drivers
            
        Returns:
            List[Dict[str, Any]]: Key drivers for the transition
        """
        key_drivers = []
        
        for driver_name, driver_data in drivers.items():
            if isinstance(driver_data, dict) and any(
                score > 0.7 for score in driver_data.values() 
                if isinstance(score, (int, float))
            ):
                key_drivers.append({
                    'driver': driver_name,
                    'importance': self._calculate_driver_importance(
                        driver_name, current_regime, target_regime),
                    'current_reading': driver_data.get('pressure_score', 
                                                    driver_data.get('anomaly_score', 0)),
                    'signal': 'high'
                })
        
        # Sort by importance
        key_drivers.sort(key=lambda x: x['importance'], reverse=True)
        return key_drivers[:3]  # Return top 3 drivers

    def _analyze_sr_stability(self, data: pd.DataFrame) -> float:
        """
        Analyze stability of support and resistance levels
        
        Args:
            data (pd.DataFrame): Price data
            
        Returns:
            float: Support/resistance stability score between 0 and 1
        """
        # Calculate key levels
        levels = self._identify_key_levels(data)
        
        # Calculate price bounces off levels
        bounces = 0
        breaks = 0
        
        for level in levels:
            # Count bounces and breaks
            for i in range(1, len(data)):
                if (abs(data['Close'].iloc[i] - level) / level < 0.001):
                    if data['Close'].iloc[i-1] < level and data['Close'].iloc[i] > level:
                        breaks += 1
                    elif data['Close'].iloc[i-1] > level and data['Close'].iloc[i] < level:
                        breaks += 1
                    else:
                        bounces += 1
        
        if bounces + breaks == 0:
            return 0.5
        
        # Calculate stability based on bounce/break ratio
        stability = bounces / (bounces + breaks)
        
        return min(1.0, stability)
    
    def _calculate_pattern_regime_metrics(self,
                                        pattern_instances: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate performance metrics for pattern instances in a regime
        
        Args:
            pattern_instances (List[Dict[str, Any]]): Pattern instances with outcomes
            
        Returns:
            Dict[str, float]: Performance metrics
        """
        if not pattern_instances:
            return {
                'success_rate': 0.0,
                'avg_return': 0.0,
                'win_rate': 0.0,
                'confidence': 0.0,
                'risk_reward': 0.0
            }
        
        successful = sum(1 for p in pattern_instances if p['success'])
        total_return = sum(p['return'] for p in pattern_instances)
        winning_trades = sum(1 for p in pattern_instances if p['return'] > 0)
        
        metrics = {
            'success_rate': successful / len(pattern_instances),
            'avg_return': total_return / len(pattern_instances),
            'win_rate': winning_trades / len(pattern_instances),
            'confidence': self._calculate_confidence_score(pattern_instances),
            'risk_reward': self._calculate_regime_risk_reward(pattern_instances)
        }
        
        return metrics  
    
    def _compare_regime_volatility(self, vol1: str, vol2: str) -> float:
        """
        Compare volatility states between regimes
        
        Args:
            vol1 (str): First regime's volatility state
            vol2 (str): Second regime's volatility state
            
        Returns:
            float: Similarity score between 0 and 1
        """
        volatility_ranks = {
            'low': 0,
            'medium': 1,
            'high': 2,
            'very_high': 3
        }
        
        try:
            rank_diff = abs(volatility_ranks[vol1] - volatility_ranks[vol2])
            if rank_diff == 0:
                return 1.0
            elif rank_diff == 1:
                return 0.5
            else:
                return 0.0
        except KeyError:
            return 0.0

    def _compare_regime_trend(self, trend1: str, trend2: str) -> float:
        """
        Compare trend states between regimes
        
        Args:
            trend1 (str): First regime's trend state
            trend2 (str): Second regime's trend state
            
        Returns:
            float: Similarity score between 0 and 1
        """
        trend_ranks = {
            'strong_bearish': -2,
            'bearish': -1,
            'neutral': 0,
            'bullish': 1,
            'strong_bullish': 2
        }
        
        try:
            rank_diff = abs(trend_ranks[trend1] - trend_ranks[trend2])
            if rank_diff == 0:
                return 1.0
            elif rank_diff == 1:
                return 0.7
            elif rank_diff == 2:
                return 0.3
            else:
                return 0.0
        except KeyError:
            return 0.0

    def _compare_regime_volume(self, vol1: str, vol2: str) -> float:
        """
        Compare volume characteristics between regimes
        
        Args:
            vol1 (str): First regime's volume state
            vol2 (str): Second regime's volume state
            
        Returns:
            float: Similarity score between 0 and 1
        """
        volume_ranks = {
            'decreasing': -1,
            'stable': 0,
            'increasing': 1
        }
        
        try:
            rank_diff = abs(volume_ranks[vol1] - volume_ranks[vol2])
            if rank_diff == 0:
                return 1.0
            elif rank_diff == 1:
                return 0.5
            else:
                return 0.0
        except KeyError:
            return 0.0

    def _calculate_confidence_score(self, pattern_instances: List[Dict[str, Any]]) -> float:
        """
        Calculate confidence score for pattern instances
        
        Args:
            pattern_instances (List[Dict[str, Any]]): Pattern instances with outcomes
            
        Returns:
            float: Confidence score between 0 and 1
        """
        if not pattern_instances:
            return 0.0
        
        # Consider multiple factors for confidence
        factors = {
            'success_consistency': self._calculate_success_consistency(pattern_instances),
            'return_consistency': self._calculate_return_consistency(pattern_instances),
            'time_consistency': self._calculate_time_consistency(pattern_instances),
            'risk_consistency': self._calculate_risk_consistency(pattern_instances)
        }
        
        # Weighted average of factors
        weights = {
            'success_consistency': 0.35,
            'return_consistency': 0.25,
            'time_consistency': 0.20,
            'risk_consistency': 0.20
        }
        
        confidence = sum(factors[k] * weights[k] for k in factors)
        return confidence

    def _calculate_regime_risk_reward(self, pattern_instances: List[Dict[str, Any]]) -> float:
        """
        Calculate risk/reward ratio for pattern instances in regime
        
        Args:
            pattern_instances (List[Dict[str, Any]]): Pattern instances with outcomes
            
        Returns:
            float: Risk/reward ratio
        """
        if not pattern_instances:
            return 0.0
        
        # Calculate average positive and negative returns
        positive_returns = [p['return'] for p in pattern_instances if p['return'] > 0]
        negative_returns = [abs(p['return']) for p in pattern_instances if p['return'] < 0]
        
        avg_reward = np.mean(positive_returns) if positive_returns else 0
        avg_risk = np.mean(negative_returns) if negative_returns else float('inf')
        
        if avg_risk == 0:
            return float('inf') if avg_reward > 0 else 0.0
        
        return avg_reward / avg_risk

    def _calculate_success_consistency(self, pattern_instances: List[Dict[str, Any]]) -> float:
        """
        Calculate consistency of pattern success
        
        Args:
            pattern_instances (List[Dict[str, Any]]): Pattern instances with outcomes
            
        Returns:
            float: Consistency score between 0 and 1
        """
        if len(pattern_instances) < 2:
            return 0.0
        
        # Convert success/failure to binary series
        success_series = pd.Series([1 if p['success'] else 0 for p in pattern_instances])
        
        # Calculate runs test statistic
        runs = len(success_series.diff()[success_series.diff() != 0]) + 1
        expected_runs = (2 * success_series.mean() * (1 - success_series.mean()) * 
                        len(success_series) + 1)
        
        # Normalize consistency score
        consistency = 1 - abs(runs - expected_runs) / expected_runs
        return max(0, min(1, consistency))
    
    def _calculate_return_consistency(self, pattern_instances: List[Dict[str, Any]]) -> float:
        """
        Calculate consistency of pattern returns
        
        Args:
            pattern_instances (List[Dict[str, Any]]): Pattern instances with outcomes
            
        Returns:
            float: Return consistency score between 0 and 1
        """
        if len(pattern_instances) < 2:
            return 0.0
            
        returns = [p['return'] for p in pattern_instances]
        
        # Calculate coefficient of variation (CV)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if mean_return == 0:
            return 0.0
            
        cv = abs(std_return / mean_return)
        
        # Convert CV to consistency score (lower CV = higher consistency)
        consistency = 1 / (1 + cv)
        
        return consistency

    def _calculate_time_consistency(self, pattern_instances: List[Dict[str, Any]]) -> float:
        """
        Calculate consistency of pattern completion time
        
        Args:
            pattern_instances (List[Dict[str, Any]]): Pattern instances with outcomes
            
        Returns:
            float: Time consistency score between 0 and 1
        """
        if len(pattern_instances) < 2:
            return 0.0
            
        completion_times = [p.get('bars_to_completion', 0) for p in pattern_instances 
                        if p.get('bars_to_completion') is not None]
        
        if not completion_times:
            return 0.0
            
        # Calculate standard deviation of completion times
        mean_time = np.mean(completion_times)
        std_time = np.std(completion_times)
        
        if mean_time == 0:
            return 0.0
            
        # Calculate coefficient of variation and convert to consistency score
        cv = std_time / mean_time
        consistency = 1 / (1 + cv)
        
        # Adjust for outliers
        percentile_range = np.percentile(completion_times, [25, 75])
        iqr = percentile_range[1] - percentile_range[0]
        outliers = sum(1 for t in completion_times 
                    if t < percentile_range[0] - 1.5 * iqr 
                    or t > percentile_range[1] + 1.5 * iqr)
        
        outlier_penalty = outliers / len(completion_times)
        consistency *= (1 - outlier_penalty)
        
        return consistency

    def _calculate_risk_consistency(self, pattern_instances: List[Dict[str, Any]]) -> float:
        """
        Calculate consistency of pattern risk metrics
        
        Args:
            pattern_instances (List[Dict[str, Any]]): Pattern instances with outcomes
            
        Returns:
            float: Risk consistency score between 0 and 1
        """
        if len(pattern_instances) < 2:
            return 0.0
            
        # Calculate various risk metrics for each instance
        risk_metrics = []
        
        for instance in pattern_instances:
            if instance.get('max_adverse_move') is not None:
                risk_data = {
                    'max_adverse': instance['max_adverse_move'],
                    'risk_reward': (instance['return'] / abs(instance['max_adverse_move']) 
                                if instance['max_adverse_move'] != 0 else 0),
                    'stop_distance': instance.get('stop_distance', 0)
                }
                risk_metrics.append(risk_data)
        
        if not risk_metrics:
            return 0.0
        
        # Calculate consistency scores for each metric
        metric_scores = []
        
        for metric in ['max_adverse', 'risk_reward', 'stop_distance']:
            values = [r[metric] for r in risk_metrics]
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            if mean_val != 0:
                cv = std_val / abs(mean_val)
                metric_scores.append(1 / (1 + cv))
            else:
                metric_scores.append(0.0)
        
        # Weight and combine metric scores
        weights = [0.4, 0.4, 0.2]  # Weights for max_adverse, risk_reward, stop_distance
        consistency = sum(score * weight for score, weight in zip(metric_scores, weights))
        
        return consistency

    def _calculate_combined_consistency(self, pattern_instances: List[Dict[str, Any]]) -> float:
        """
        Calculate overall consistency score combining all metrics
        
        Args:
            pattern_instances (List[Dict[str, Any]]): Pattern instances with outcomes
            
        Returns:
            float: Combined consistency score between 0 and 1
        """
        consistency_scores = {
            'success': self._calculate_success_consistency(pattern_instances),
            'return': self._calculate_return_consistency(pattern_instances),
            'time': self._calculate_time_consistency(pattern_instances),
            'risk': self._calculate_risk_consistency(pattern_instances)
        }
        
        # Weight different consistency aspects
        weights = {
            'success': 0.35,
            'return': 0.25,
            'time': 0.20,
            'risk': 0.20
        }
        
        combined_score = sum(consistency_scores[k] * weights[k] for k in weights)
        
        return combined_score
    
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
        
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = abs(loss)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Avoid division by zero
        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    def _calculate_macd_with_signal(self, prices: pd.Series, 
                                    fast_period: int = 12, 
                                    slow_period: int = 26, 
                                    signal_period: int = 9) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate MACD and signal line
        
        Args:
            prices (pd.Series): Price data
            fast_period (int): Fast EMA period
            slow_period (int): Slow EMA period
            signal_period (int): Signal line period
            
        Returns:
            Tuple[pd.Series, pd.Series]: MACD and signal line
        """
        fast_ema = prices.ewm(span=fast_period, adjust=False).mean()
        slow_ema = prices.ewm(span=slow_period, adjust=False).mean()
        macd = fast_ema - slow_ema
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        return macd, signal

    def _calculate_bollinger_bandwidth(self, prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.Series:
        """
        Calculate Bollinger Band bandwidth
        
        Args:
            prices (pd.Series): Price data
            period (int): Bollinger Band period
            std_dev (float): Number of standard deviations
            
        Returns:
            pd.Series: Bollinger Band bandwidth
        """
        middle_band = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        # Calculate bandwidth
        bandwidth = (upper_band - lower_band) / middle_band
        
        return bandwidth

    def _find_swing_highs(self, data: pd.DataFrame, window: int = 5, threshold: float = 0.01) -> pd.Series:
        """
        Find swing high points in price data
        
        Args:
            data (pd.DataFrame): Price data with 'High' column
            window (int): Window size for comparison
            threshold (float): Minimum price movement threshold
            
        Returns:
            pd.Series: Series of swing high prices (non-NaN at swing points)
        """
        swing_highs = pd.Series(index=data.index, dtype=float)
        
        # We need at least 2*window+1 data points
        if len(data) < 2 * window + 1:
            return swing_highs
            
        for i in range(window, len(data) - window):
            current_high = data['High'].iloc[i]
            left_window = data['High'].iloc[i-window:i]
            right_window = data['High'].iloc[i+1:i+window+1]
            
            # Check if current point is higher than all points in both windows
            if (current_high > left_window.max() and 
                current_high > right_window.max() and
                (current_high - min(left_window.min(), right_window.min())) / current_high > threshold):
                
                swing_highs.iloc[i] = current_high
                
        return swing_highs

    def _find_swing_lows(self, data: pd.DataFrame, window: int = 5, threshold: float = 0.01) -> pd.Series:
        """
        Find swing low points in price data
        
        Args:
            data (pd.DataFrame): Price data with 'Low' column
            window (int): Window size for comparison
            threshold (float): Minimum price movement threshold
            
        Returns:
            pd.Series: Series of swing low prices (non-NaN at swing points)
        """
        swing_lows = pd.Series(index=data.index, dtype=float)
        
        # We need at least 2*window+1 data points
        if len(data) < 2 * window + 1:
            return swing_lows
            
        for i in range(window, len(data) - window):
            current_low = data['Low'].iloc[i]
            left_window = data['Low'].iloc[i-window:i]
            right_window = data['Low'].iloc[i+1:i+window+1]
            
            # Check if current point is lower than all points in both windows
            if (current_low < left_window.min() and 
                current_low < right_window.min() and
                (max(left_window.max(), right_window.max()) - current_low) / current_low > threshold):
                
                swing_lows.iloc[i] = current_low
                
        return swing_lows

    def _identify_support_levels(self, prices: pd.Series, window: int = 20, threshold: float = 0.01) -> List[float]:
        """
        Identify key support levels from price data
        
        Args:
            prices (pd.Series): Price data
            window (int): Lookback window
            threshold (float): Price proximity threshold
            
        Returns:
            List[float]: List of support levels
        """
        if len(prices) < window:
            return []
            
        # Find local minima
        lows = []
        for i in range(window, len(prices) - window):
            if all(prices.iloc[i] <= prices.iloc[i-j] for j in range(1, window+1)) and \
            all(prices.iloc[i] <= prices.iloc[i+j] for j in range(1, window+1)):
                lows.append(prices.iloc[i])
        
        # Cluster similar levels
        support_levels = []
        for low in sorted(lows):
            # Check if this level is close to an existing one
            if not any(abs(low - level) / level < threshold for level in support_levels):
                support_levels.append(low)
        
        return support_levels

    def _identify_resistance_levels(self, prices: pd.Series, window: int = 20, threshold: float = 0.01) -> List[float]:
        """
        Identify key resistance levels from price data
        
        Args:
            prices (pd.Series): Price data
            window (int): Lookback window
            threshold (float): Price proximity threshold
            
        Returns:
            List[float]: List of resistance levels
        """
        if len(prices) < window:
            return []
            
        # Find local maxima
        highs = []
        for i in range(window, len(prices) - window):
            if all(prices.iloc[i] >= prices.iloc[i-j] for j in range(1, window+1)) and \
            all(prices.iloc[i] >= prices.iloc[i+j] for j in range(1, window+1)):
                highs.append(prices.iloc[i])
        
        # Cluster similar levels
        resistance_levels = []
        for high in sorted(highs):
            # Check if this level is close to an existing one
            if not any(abs(high - level) / level < threshold for level in resistance_levels):
                resistance_levels.append(high)
        
        return resistance_levels

    def _is_regime_change(self, prev_regime: Dict[str, str], current_regime: Dict[str, str]) -> bool:
        """
        Determine if a regime change has occurred based on component changes
        
        Args:
            prev_regime (Dict[str, str]): Previous regime characteristics
            current_regime (Dict[str, str]): Current regime characteristics
            
        Returns:
            bool: True if regime has changed significantly
        """
        # Define weights for different components
        weights = {
            'trend': 0.45,       # Trend is most important
            'volatility': 0.30,  # Volatility is second
            'volume': 0.15,      # Volume less important
            'momentum': 0.05,    # Optional components
            'support_resistance': 0.05
        }
        
        # Calculate change score
        change_score = 0
        for component, weight in weights.items():
            if component in prev_regime and component in current_regime:
                if prev_regime[component] != current_regime[component]:
                    change_score += weight
        
        # Threshold for significant change
        return change_score >= 0.4

    def _determine_regime_type(self, regime_data: Dict[str, str]) -> str:
        """
        Determine overall regime type from component characteristics
        
        Args:
            regime_data (Dict[str, str]): Regime characteristics
            
        Returns:
            str: Overall regime type
        """
        trend = regime_data.get('trend', 'unknown')
        volatility = regime_data.get('volatility', 'unknown')
        
        if trend == 'unknown' or volatility == 'unknown':
            return 'unknown'
        
        # Determine regime type based on trend and volatility
        if trend in ['bullish', 'strong_bullish', 'weak_bullish']:
            if volatility == 'low':
                return 'trending'  # Stable uptrend
            elif volatility == 'high':
                return 'volatile'  # Volatile uptrend
            else:
                return 'trending'  # Normal uptrend
        elif trend in ['bearish', 'strong_bearish', 'weak_bearish']:
            if volatility == 'low':
                return 'trending'  # Stable downtrend
            elif volatility == 'high':
                return 'volatile'  # Volatile downtrend
            else:
                return 'trending'  # Normal downtrend
        else:  # neutral trend
            if volatility == 'low':
                return 'consolidating'  # Low volatility sideways
            elif volatility == 'high':
                return 'transitioning'  # High volatility sideways - likely transition
            else:
                return 'ranging'  # Normal volatility sideways

    def _calculate_regime_confidence(self, regime_data: Dict[str, str]) -> float:
        """
        Calculate confidence score for detected regime
        
        Args:
            regime_data (Dict[str, str]): Regime characteristics
            
        Returns:
            float: Confidence score between 0 and 1
        """
        # Base confidence
        confidence = 0.7
        
        # Conflicting signals reduce confidence
        trend = regime_data.get('trend', 'unknown')
        momentum = regime_data.get('momentum', 'unknown')
        
        if trend != 'unknown' and momentum != 'unknown':
            # Trend and momentum contradiction
            if (trend in ['bullish', 'strong_bullish'] and momentum in ['bearish', 'strong_bearish']) or \
            (trend in ['bearish', 'strong_bearish'] and momentum in ['bullish', 'strong_bullish']):
                confidence -= 0.2
        
        # Unknown components reduce confidence
        unknown_components = sum(1 for value in regime_data.values() if value == 'unknown')
        confidence -= 0.05 * unknown_components
        
        return max(0.3, min(0.95, confidence))

    def _analyze_sr_tests(self, data: pd.DataFrame) -> float:
        """
        Analyze support/resistance tests
        
        Args:
            data (pd.DataFrame): Price data
            
        Returns:
            float: Test intensity score (0-1)
        """
        # This is a simplified implementation
        if len(data) < 10:
            return 0.5
        
        # Calculate key levels
        support_levels = self._identify_support_levels(data['Low'])
        resistance_levels = self._identify_resistance_levels(data['High'])
        
        # Count tests of key levels
        test_count = 0
        for i in range(1, len(data)):
            close = data['Close'].iloc[i]
            prev_close = data['Close'].iloc[i-1]
            
            # Check if price is testing a level
            for level in support_levels + resistance_levels:
                test_distance = abs(close - level) / level
                if test_distance < 0.01:  # Within 1% of level
                    test_count += 1
                    break
        
        # Normalize test count
        max_possible_tests = len(data)
        test_intensity = min(1.0, test_count / max_possible_tests * 3)  # Scale up for meaningful scores
        
        return test_intensity

    def _get_historical_transition_durations(self, from_regime: str, to_regime: str) -> List[int]:
        """
        Get historical transition durations between regime types
        
        Args:
            from_regime (str): Starting regime type
            to_regime (str): Target regime type
            
        Returns:
            List[int]: List of historical durations in bars
        """
        # Simplified implementation - uses default values as we don't have historical data
        typical_durations = {
            ('trending', 'volatile'): [5, 7, 10],
            ('trending', 'ranging'): [8, 12, 15],
            ('trending', 'consolidating'): [10, 15, 20],
            ('volatile', 'trending'): [3, 5, 8],
            ('volatile', 'ranging'): [5, 8, 10],
            ('ranging', 'trending'): [8, 12, 18],
            ('ranging', 'volatile'): [3, 5, 8],
            ('consolidating', 'trending'): [7, 10, 15],
            ('consolidating', 'volatile'): [3, 5, 10],
        }
        
        key = (from_regime, to_regime)
        if key in typical_durations:
            return typical_durations[key]
        
        # Default durations if specific transition not found
        return [10, 15, 20]

    def _calculate_driver_importance(self, driver_name: str, current_regime: str, target_regime: str) -> float:
        """
        Calculate the importance of a driver for specific regime transition
        
        Args:
            driver_name (str): Name of the driver
            current_regime (str): Current regime type
            target_regime (str): Target regime type
            
        Returns:
            float: Importance score between 0 and 1
        """
        # Define driver importance for different transitions
        driver_importance = {
            'volatile': {
                'volatility_pressure': 0.8,
                'volume_anomalies': 0.6,
                'momentum_divergence': 0.5,
                'trend_exhaustion': 0.3,
                'support_resistance_tests': 0.4
            },
            'trending': {
                'trend_exhaustion': 0.7,
                'momentum_divergence': 0.6,
                'volatility_pressure': 0.4,
                'volume_anomalies': 0.5,
                'support_resistance_tests': 0.3
            },
            'ranging': {
                'support_resistance_tests': 0.7,
                'volatility_pressure': 0.5,
                'volume_anomalies': 0.4,
                'momentum_divergence': 0.3,
                'trend_exhaustion': 0.2
            },
            'consolidating': {
                'volatility_pressure': 0.6,
                'volume_anomalies': 0.7,
                'support_resistance_tests': 0.5,
                'trend_exhaustion': 0.3,
                'momentum_divergence': 0.2
            },
            'transitioning': {
                'momentum_divergence': 0.7,
                'volatility_pressure': 0.6,
                'trend_exhaustion': 0.5,
                'volume_anomalies': 0.4,
                'support_resistance_tests': 0.3
            }
        }
        
        # Get importance for target regime
        if target_regime in driver_importance and driver_name in driver_importance[target_regime]:
            return driver_importance[target_regime][driver_name]
        
        # Default importance
        return 0.5

    def _get_pattern_signals(self, pattern_name: str, slice_data: pd.DataFrame = None) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
        """
        Get pattern signals for a specific slice of data
        
        Args:
            pattern_name (str): Name of the pattern
            slice_data (pd.DataFrame): Data slice to analyze
            
        Returns:
            Union[pd.Series, Tuple[pd.Series, pd.Series]]: Pattern signals
        """
        # Use provided slice or full data
        data = slice_data if slice_data is not None else self.df
        
        # Create a pattern analyzer
        from .patterns import CandlestickPatterns
        patterns = CandlestickPatterns()
        
        # Get pattern detection method
        pattern_method = getattr(patterns, f'detect_{pattern_name}', None)
        
        if pattern_method is not None:
            return pattern_method(data)
        
        # Return empty signals if pattern not found
        empty_signal = pd.Series(False, index=data.index)
        return empty_signal
        
class VisualizationCache:
    """Cache manager for visualization calculations"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache = {}
        self._access_times = {}
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Any:
        """Get value from cache"""
        with self._lock:
            if key in self._cache:
                self._access_times[key] = time.time()
                return self._cache[key]
            return None
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache"""
        with self._lock:
            if len(self._cache) >= self.max_size:
                self._evict_oldest()
            self._cache[key] = value
            self._access_times[key] = time.time()
    
    def _evict_oldest(self) -> None:
        """Remove least recently used items"""
        oldest_key = min(self._access_times.items(), key=lambda x: x[1])[0]
        del self._cache[oldest_key]
        del self._access_times[oldest_key]

class CalculationCache:
    """
    Cache manager for computationally intensive calculations
    
    Attributes:
        max_size (int): Maximum number of items in cache
        ttl (int): Time to live for cache items in seconds
        cache (Dict): Main cache storage
        access_times (Dict): Last access times for items
        creation_times (Dict): Creation times for items
        lock (threading.Lock): Thread lock for cache operations
    """
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        """
        Initialize cache manager
        
        Args:
            max_size (int): Maximum cache size
            ttl (int): Cache item time to live in seconds
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache = {}
        self.access_times = {}
        self.creation_times = {}
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get item from cache
        
        Args:
            key (str): Cache key
            
        Returns:
            Optional[Any]: Cached value or None if not found/expired
        """
        with self.lock:
            if key not in self.cache:
                return None
                
            current_time = time.time()
            
            # Check if item has expired
            if current_time - self.creation_times[key] > self.ttl:
                self._remove_item(key)
                return None
            
            # Update access time
            self.access_times[key] = current_time
            return self.cache[key]
    
    def set(self, key: str, value: Any) -> None:
        """
        Set item in cache
        
        Args:
            key (str): Cache key
            value (Any): Value to cache
        """
        with self.lock:
            current_time = time.time()
            
            # Check if we need to make room
            if len(self.cache) >= self.max_size:
                self._evict_items()
            
            # Add new item
            self.cache[key] = value
            self.access_times[key] = current_time
            self.creation_times[key] = current_time
    
    def _evict_items(self) -> None:
        """Evict items based on access time and TTL"""
        current_time = time.time()
        
        # First remove expired items
        expired_keys = [
            k for k, creation_time in self.creation_times.items()
            if current_time - creation_time > self.ttl
        ]
        
        for key in expired_keys:
            self._remove_item(key)
            
        # If we still need to make room, remove least recently used
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_times.items(), key=lambda x: x[1])[0]
            self._remove_item(oldest_key)
    
    def _remove_item(self, key: str) -> None:
        """
        Remove item from cache
        
        Args:
            key (str): Cache key to remove
        """
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
        self.creation_times.pop(key, None)
    
    def clear(self) -> None:
        """Clear all items from cache"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.creation_times.clear()
            
class IndicatorManager:
    """
    Manages lazy loading and caching of technical indicators
    
    Attributes:
        df (pd.DataFrame): Price data
        cache (CalculationCache): Cache for computed indicators
        _computed_indicators (Dict): Tracks which indicators have been computed
    """
    
    def __init__(self, df: pd.DataFrame, cache_size: int = 1000):
        """
        Initialize indicator manager
        
        Args:
            df (pd.DataFrame): Price data
            cache_size (int): Maximum cache size
        """
        self.df = df
        self.cache = CalculationCache(max_size=cache_size)
        self._computed_indicators = {}
        
        # Register available indicators
        self._indicator_registry = {
            'sma': self._compute_sma,
            'ema': self._compute_ema,
            'rsi': self._compute_rsi,
            'macd': self._compute_macd,
            'bollinger_bands': self._compute_bollinger_bands,
            'atr': self._compute_atr,
            'volume_profile': self._compute_volume_profile
        }
    
    def get_indicator(self, 
                     name: str, 
                     params: Dict[str, Any] = None) -> Union[pd.Series, pd.DataFrame]:
        """
        Get indicator values, computing only if necessary
        
        Args:
            name (str): Indicator name
            params (Dict[str, Any]): Indicator parameters
            
        Returns:
            Union[pd.Series, pd.DataFrame]: Indicator values
            
        Raises:
            ValueError: If indicator is not supported
        """
        if name not in self._indicator_registry:
            raise ValueError(f"Unsupported indicator: {name}")
            
        # Create cache key
        cache_key = self._create_cache_key(name, params)
        
        # Check cache first
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Compute indicator
        params = params or {}
        result = self._indicator_registry[name](**params)
        
        # Cache result
        self.cache.set(cache_key, result)
        self._computed_indicators[cache_key] = True
        
        return result
    
    def _create_cache_key(self, name: str, params: Optional[Dict[str, Any]]) -> str:
        """
        Create unique cache key for indicator
        
        Args:
            name (str): Indicator name
            params (Optional[Dict[str, Any]]): Indicator parameters
            
        Returns:
            str: Cache key
        """
        if params:
            param_str = '_'.join(f"{k}_{v}" for k, v in sorted(params.items()))
            return f"{name}_{param_str}"
        return name
    
    def _compute_sma(self, period: int = 20) -> pd.Series:
        """Compute Simple Moving Average"""
        return self.df['Close'].rolling(window=period).mean()
    
    def _compute_ema(self, period: int = 20) -> pd.Series:
        """Compute Exponential Moving Average"""
        return self.df['Close'].ewm(span=period, adjust=False).mean()
    
    def _compute_rsi(self, period: int = 14) -> pd.Series:
        """Compute Relative Strength Index"""
        delta = self.df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _compute_macd(self, 
                     fast_period: int = 12, 
                     slow_period: int = 26, 
                     signal_period: int = 9) -> pd.DataFrame:
        """Compute MACD"""
        fast_ema = self._compute_ema(fast_period)
        slow_ema = self._compute_ema(slow_period)
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        
        return pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line,
            'histogram': macd_line - signal_line
        })
    
    def _compute_bollinger_bands(self, 
                               period: int = 20, 
                               std_dev: float = 2.0) -> pd.DataFrame:
        """Compute Bollinger Bands"""
        middle_band = self._compute_sma(period)
        std = self.df['Close'].rolling(window=period).std()
        
        return pd.DataFrame({
            'upper': middle_band + (std * std_dev),
            'middle': middle_band,
            'lower': middle_band - (std * std_dev)
        })
    
    def _compute_atr(self, period: int = 14) -> pd.Series:
        """Compute Average True Range"""
        high = self.df['High']
        low = self.df['Low']
        close = self.df['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    def _compute_volume_profile(self, bins: int = 50) -> pd.DataFrame:
        """Compute Volume Profile"""
        if 'Volume' not in self.df.columns:
            return pd.DataFrame()
            
        price_bins = pd.cut(self.df['Close'], bins=bins)
        volume_profile = self.df.groupby(price_bins)['Volume'].sum()
        
        return pd.DataFrame({
            'price_level': volume_profile.index.mid,
            'volume': volume_profile.values
        })
    
    def precompute_indicators(self, indicators: List[Dict[str, Any]]) -> None:
        """
        Precompute multiple indicators
        
        Args:
            indicators (List[Dict[str, Any]]): List of indicators to compute
        """
        for indicator in indicators:
            name = indicator['name']
            params = indicator.get('params', {})
            self.get_indicator(name, params)
    
    def clear_cache(self) -> None:
        """Clear indicator cache"""
        self.cache.clear()
        self._computed_indicators.clear()
        
class DataManager:
    """
    Manages data chunking and memory optimization for large datasets
    
    Attributes:
        df (pd.DataFrame): Full price data
        chunk_size (int): Size of data chunks
        max_chunks (int): Maximum number of chunks to keep in memory
        _chunks (Dict): Storage for data chunks
        _chunk_access_times (Dict): Last access times for chunks
    """
    
    def __init__(self, 
                 df: pd.DataFrame, 
                 chunk_size: int = 1000,
                 max_chunks: int = 10):
        """
        Initialize data manager
        
        Args:
            df (pd.DataFrame): Price data
            chunk_size (int): Number of rows per chunk
            max_chunks (int): Maximum chunks to keep in memory
        """
        self.df = df
        self.chunk_size = chunk_size
        self.max_chunks = max_chunks
        self._chunks = {}
        self._chunk_access_times = {}
        self.lock = threading.Lock()
        
        # Create chunk index
        self._create_chunk_index()
    
    def _create_chunk_index(self) -> None:
        """Create index of data chunks"""
        self.total_chunks = (len(self.df) + self.chunk_size - 1) // self.chunk_size
        self.chunk_boundaries = [
            (i * self.chunk_size, min((i + 1) * self.chunk_size, len(self.df)))
            for i in range(self.total_chunks)
        ]
    
    def get_data_chunk(self, chunk_id: int) -> pd.DataFrame:
        """
        Get a specific data chunk
        
        Args:
            chunk_id (int): Chunk identifier
            
        Returns:
            pd.DataFrame: Data chunk
            
        Raises:
            ValueError: If chunk_id is invalid
        """
        if not 0 <= chunk_id < self.total_chunks:
            raise ValueError(f"Invalid chunk_id: {chunk_id}")
            
        with self.lock:
            # Check if chunk is in memory
            if chunk_id in self._chunks:
                self._update_chunk_access(chunk_id)
                return self._chunks[chunk_id]
            
            # Load chunk into memory
            return self._load_chunk(chunk_id)
    
    def get_data_range(self, 
                      start_date: pd.Timestamp,
                      end_date: pd.Timestamp) -> pd.DataFrame:
        """
        Get data for specified date range
        
        Args:
            start_date (pd.Timestamp): Start date
            end_date (pd.Timestamp): End date
            
        Returns:
            pd.DataFrame: Data for specified range
        """
        # Find relevant chunks
        start_chunk = self._find_chunk_for_date(start_date)
        end_chunk = self._find_chunk_for_date(end_date)
        
        chunks_data = []
        for chunk_id in range(start_chunk, end_chunk + 1):
            chunk_data = self.get_data_chunk(chunk_id)
            chunks_data.append(chunk_data)
        
        # Combine chunks and filter date range
        combined_data = pd.concat(chunks_data)
        return combined_data[start_date:end_date]
    
    def _load_chunk(self, chunk_id: int) -> pd.DataFrame:
        """
        Load a data chunk into memory
        
        Args:
            chunk_id (int): Chunk identifier
            
        Returns:
            pd.DataFrame: Loaded data chunk
        """
        # Check if we need to free memory
        if len(self._chunks) >= self.max_chunks:
            self._evict_least_used_chunk()
        
        # Load chunk
        start_idx, end_idx = self.chunk_boundaries[chunk_id]
        chunk_data = self.df.iloc[start_idx:end_idx].copy()
        
        # Store in memory
        self._chunks[chunk_id] = chunk_data
        self._update_chunk_access(chunk_id)
        
        return chunk_data
    
    def _update_chunk_access(self, chunk_id: int) -> None:
        """
        Update last access time for chunk
        
        Args:
            chunk_id (int): Chunk identifier
        """
        self._chunk_access_times[chunk_id] = time.time()
    
    def _evict_least_used_chunk(self) -> None:
        """Remove least recently used chunk from memory"""
        if not self._chunk_access_times:
            return
            
        oldest_chunk = min(self._chunk_access_times.items(), 
                          key=lambda x: x[1])[0]
        
        self._chunks.pop(oldest_chunk, None)
        self._chunk_access_times.pop(oldest_chunk, None)
    
    def _find_chunk_for_date(self, date: pd.Timestamp) -> int:
        """
        Find chunk ID containing specified date
        
        Args:
            date (pd.Timestamp): Date to find
            
        Returns:
            int: Chunk ID
            
        Raises:
            ValueError: If date is out of range
        """
        if date < self.df.index[0] or date > self.df.index[-1]:
            raise ValueError(f"Date {date} out of range")
            
        for chunk_id, (start_idx, end_idx) in enumerate(self.chunk_boundaries):
            chunk_start_date = self.df.index[start_idx]
            chunk_end_date = self.df.index[end_idx - 1]
            
            if chunk_start_date <= date <= chunk_end_date:
                return chunk_id
        
        return self.total_chunks - 1  # Return last chunk if not found
    
    def optimize_memory(self) -> None:
        """Optimize memory usage by clearing unused chunks"""
        with self.lock:
            current_time = time.time()
            
            # Remove chunks not accessed in the last hour
            chunks_to_remove = [
                chunk_id for chunk_id, last_access 
                in self._chunk_access_times.items()
                if current_time - last_access > 3600
            ]
            
            for chunk_id in chunks_to_remove:
                self._chunks.pop(chunk_id, None)
                self._chunk_access_times.pop(chunk_id, None)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get memory usage statistics
        
        Returns:
            Dict[str, float]: Memory usage information
        """
        total_memory = sum(chunk.memory_usage(deep=True).sum() 
                          for chunk in self._chunks.values())
        
        return {
            'total_memory_mb': total_memory / 1024 / 1024,
            'num_chunks': len(self._chunks),
            'max_chunks': self.max_chunks,
            'chunk_size': self.chunk_size
        }
        
class ParallelProcessor:
    """
    Manages parallel processing for computationally intensive tasks
    
    Attributes:
        max_workers (int): Maximum number of worker processes
        chunk_size (int): Size of data chunks for parallel processing
        _pool (ProcessPoolExecutor): Process pool for parallel execution
    """
    
    def __init__(self, max_workers: Optional[int] = None, chunk_size: int = 1000):
        """
        Initialize parallel processor
        
        Args:
            max_workers (Optional[int]): Maximum number of worker processes
            chunk_size (int): Size of data chunks
        """
        self.max_workers = max_workers or mp.cpu_count()
        self.chunk_size = chunk_size
        self._pool = None
    
    def __enter__(self):
        """Context manager entry"""
        self._pool = ProcessPoolExecutor(max_workers=self.max_workers)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self._pool:
            self._pool.shutdown()
            self._pool = None
    
    def process_in_parallel(self,
                          data: pd.DataFrame,
                          operation: Callable,
                          **kwargs) -> pd.DataFrame:
        """
        Process data in parallel
        
        Args:
            data (pd.DataFrame): Input data
            operation (Callable): Operation to perform
            **kwargs: Additional arguments for operation
            
        Returns:
            pd.DataFrame: Processed data
        """
        if len(data) < self.chunk_size:
            return operation(data, **kwargs)
        
        # Split data into chunks
        chunks = self._split_into_chunks(data)
        
        # Process chunks in parallel
        with self as processor:
            futures = [
                processor._pool.submit(operation, chunk, **kwargs)
                for chunk in chunks
            ]
            
            results = [future.result() for future in as_completed(futures)]
        
        # Combine results
        return pd.concat(results)
    
    def parallel_indicator_calculation(self,
                                    data: pd.DataFrame,
                                    indicators: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
        """
        Calculate multiple indicators in parallel
        
        Args:
            data (pd.DataFrame): Price data
            indicators (List[Dict[str, Any]]): Indicators to calculate
            
        Returns:
            Dict[str, pd.DataFrame]: Calculated indicators
        """
        with self as processor:
            futures = []
            for indicator in indicators:
                future = processor._pool.submit(
                    self._calculate_indicator,
                    data,
                    indicator['name'],
                    indicator.get('params', {})
                )
                futures.append((indicator['name'], future))
            
            results = {
                name: future.result()
                for name, future in futures
            }
        
        return results
    
    def parallel_pattern_detection(self,
                                 data: pd.DataFrame,
                                 patterns: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Detect multiple patterns in parallel
        
        Args:
            data (pd.DataFrame): Price data
            patterns (List[str]): Patterns to detect
            
        Returns:
            Dict[str, pd.DataFrame]: Detected patterns
        """
        with self as processor:
            futures = []
            for pattern in patterns:
                future = processor._pool.submit(
                    self._detect_pattern,
                    data,
                    pattern
                )
                futures.append((pattern, future))
            
            results = {
                pattern: future.result()
                for pattern, future in futures
            }
        
        return results
    
    def _split_into_chunks(self, data: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Split data into chunks for parallel processing
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            List[pd.DataFrame]: List of data chunks
        """
        chunks = []
        for i in range(0, len(data), self.chunk_size):
            chunk = data.iloc[i:i + self.chunk_size].copy()
            chunks.append(chunk)
        return chunks
    
    @staticmethod
    def _calculate_indicator(data: pd.DataFrame,
                           indicator_name: str,
                           params: Dict[str, Any]) -> pd.DataFrame:
        """
        Calculate single indicator (for parallel processing)
        
        Args:
            data (pd.DataFrame): Price data
            indicator_name (str): Indicator name
            params (Dict[str, Any]): Indicator parameters
            
        Returns:
            pd.DataFrame: Calculated indicator
        """
        indicator_manager = IndicatorManager(data)
        return indicator_manager.get_indicator(indicator_name, params)
    
    @staticmethod
    def _detect_pattern(data: pd.DataFrame, pattern_name: str) -> pd.DataFrame:
        """
        Detect single pattern (for parallel processing)
        
        Args:
            data (pd.DataFrame): Price data
            pattern_name (str): Pattern name
            
        Returns:
            pd.DataFrame: Detected pattern
        """
        pattern_analyzer = MarketRegimeAnalyzer(data)
        return pattern_analyzer._get_pattern_signals(pattern_name)
    
    def parallel_regime_analysis(self,
                               data: pd.DataFrame,
                               analysis_types: List[str]) -> Dict[str, Any]:
        """
        Perform multiple regime analyses in parallel
        
        Args:
            data (pd.DataFrame): Price data
            analysis_types (List[str]): Types of analysis to perform
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        with self as processor:
            futures = []
            for analysis_type in analysis_types:
                future = processor._pool.submit(
                    self._perform_regime_analysis,
                    data,
                    analysis_type
                )
                futures.append((analysis_type, future))
            
            results = {
                analysis_type: future.result()
                for analysis_type, future in futures
            }
        
        return results
    
    @staticmethod
    def _perform_regime_analysis(data: pd.DataFrame,
                               analysis_type: str) -> Dict[str, Any]:
        """
        Perform single regime analysis (for parallel processing)
        
        Args:
            data (pd.DataFrame): Price data
            analysis_type (str): Type of analysis
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        analyzer = MarketRegimeAnalyzer(data)
        
        analysis_functions = {
            'volatility': analyzer._calculate_volatility_regime,
            'trend': analyzer._calculate_trend_regime,
            'volume': analyzer._calculate_volume_regime,
            'momentum': analyzer._calculate_momentum_regime,
            'support_resistance': analyzer._calculate_sr_regime
        }
        
        if analysis_type in analysis_functions:
            return analysis_functions[analysis_type](window=20)
        else:
            raise ValueError(f"Unsupported analysis type: {analysis_type}")
        
class ParallelTaskManager:
    """
    Manages parallel task execution with error handling and monitoring
    
    Attributes:
        processor (ParallelProcessor): Parallel processor instance
        max_queue_size (int): Maximum size of task queue
        _task_queue (Queue): Queue for pending tasks
        _results (Dict): Storage for task results
        _errors (Dict): Storage for task errors
    """
    
    def __init__(self, max_workers: Optional[int] = None, max_queue_size: int = 1000):
        """
        Initialize task manager
        
        Args:
            max_workers (Optional[int]): Maximum number of worker processes
            max_queue_size (int): Maximum size of task queue
        """
        self.processor = ParallelProcessor(max_workers=max_workers)
        self.max_queue_size = max_queue_size
        self._task_queue = Queue(maxsize=max_queue_size)
        self._results = {}
        self._errors = {}
        self._lock = threading.Lock()
        self._monitor = TaskMonitor()
    
    def submit_task(self, 
                   task_id: str,
                   operation: Callable,
                   data: pd.DataFrame,
                   **kwargs) -> None:
        """
        Submit task for parallel execution
        
        Args:
            task_id (str): Unique task identifier
            operation (Callable): Operation to perform
            data (pd.DataFrame): Input data
            **kwargs: Additional operation arguments
        """
        task = {
            'id': task_id,
            'operation': operation,
            'data': data,
            'kwargs': kwargs,
            'status': 'pending',
            'submit_time': time.time()
        }
        
        with self._lock:
            if self._task_queue.qsize() >= self.max_queue_size:
                raise QueueFullError("Task queue is full")
            
            self._task_queue.put(task)
            self._monitor.track_task(task_id)
    
    def execute_pending_tasks(self) -> None:
        """Execute all pending tasks in parallel"""
        pending_tasks = []
        
        # Collect pending tasks
        while not self._task_queue.empty():
            task = self._task_queue.get()
            pending_tasks.append(task)
            task['status'] = 'processing'
            self._monitor.update_task_status(task['id'], 'processing')
        
        # Execute tasks in parallel
        with self.processor as proc:
            futures = []
            for task in pending_tasks:
                future = proc._pool.submit(
                    self._execute_task_safely,
                    task
                )
                futures.append((task['id'], future))
            
            # Collect results
            for task_id, future in futures:
                try:
                    result = future.result()
                    self._handle_task_success(task_id, result)
                except Exception as e:
                    self._handle_task_error(task_id, e)
    
    def _execute_task_safely(self, task: Dict[str, Any]) -> Any:
        """
        Execute single task with error handling
        
        Args:
            task (Dict[str, Any]): Task information
            
        Returns:
            Any: Task result
            
        Raises:
            Exception: If task execution fails
        """
        try:
            start_time = time.time()
            result = task['operation'](task['data'], **task['kwargs'])
            execution_time = time.time() - start_time
            
            self._monitor.record_execution_time(task['id'], execution_time)
            return result
            
        except Exception as e:
            self._monitor.record_error(task['id'], str(e))
            raise
    
    def _handle_task_success(self, task_id: str, result: Any) -> None:
        """
        Handle successful task completion
        
        Args:
            task_id (str): Task identifier
            result (Any): Task result
        """
        with self._lock:
            self._results[task_id] = result
            self._monitor.update_task_status(task_id, 'completed')
    
    def _handle_task_error(self, task_id: str, error: Exception) -> None:
        """
        Handle task execution error
        
        Args:
            task_id (str): Task identifier
            error (Exception): Error that occurred
        """
        with self._lock:
            self._errors[task_id] = error
            self._monitor.update_task_status(task_id, 'failed')
    
    def get_task_result(self, task_id: str) -> Optional[Any]:
        """
        Get result of completed task
        
        Args:
            task_id (str): Task identifier
            
        Returns:
            Optional[Any]: Task result if available
        """
        return self._results.get(task_id)
    
    def get_task_error(self, task_id: str) -> Optional[Exception]:
        """
        Get error from failed task
        
        Args:
            task_id (str): Task identifier
            
        Returns:
            Optional[Exception]: Task error if failed
        """
        return self._errors.get(task_id)
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get current task status
        
        Args:
            task_id (str): Task identifier
            
        Returns:
            Dict[str, Any]: Task status information
        """
        return self._monitor.get_task_status(task_id)
    
class TaskMonitor:
    """
    Monitors task execution progress and system resources
    
    Attributes:
        _task_stats (Dict): Statistics for each task
        _system_stats (Dict): System resource statistics
        _lock (threading.Lock): Thread lock for stats updates
    """
    
    def __init__(self):
        """Initialize task monitor"""
        self._task_stats = {}
        self._system_stats = {
            'cpu_usage': [],
            'memory_usage': [],
            'task_queue_size': []
        }
        self._lock = threading.Lock()
        
        # Start resource monitoring
        self._start_resource_monitoring()
    
    def track_task(self, task_id: str) -> None:
        """
        Start tracking new task
        
        Args:
            task_id (str): Task identifier
        """
        with self._lock:
            self._task_stats[task_id] = {
                'status': 'pending',
                'start_time': None,
                'end_time': None,
                'execution_time': None,
                'error': None,
                'memory_peak': None,
                'cpu_usage': None
            }
    
    def update_task_status(self, task_id: str, status: str) -> None:
        """
        Update task status
        
        Args:
            task_id (str): Task identifier
            status (str): New status
        """
        with self._lock:
            if task_id in self._task_stats:
                self._task_stats[task_id]['status'] = status
                
                if status == 'processing' and not self._task_stats[task_id]['start_time']:
                    self._task_stats[task_id]['start_time'] = time.time()
                elif status in ['completed', 'failed']:
                    self._task_stats[task_id]['end_time'] = time.time()
    
    def record_execution_time(self, task_id: str, execution_time: float) -> None:
        """
        Record task execution time
        
        Args:
            task_id (str): Task identifier
            execution_time (float): Execution time in seconds
        """
        with self._lock:
            if task_id in self._task_stats:
                self._task_stats[task_id]['execution_time'] = execution_time
    
    def record_error(self, task_id: str, error: str) -> None:
        """
        Record task error
        
        Args:
            task_id (str): Task identifier
            error (str): Error message
        """
        with self._lock:
            if task_id in self._task_stats:
                self._task_stats[task_id]['error'] = error
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get current task status
        
        Args:
            task_id (str): Task identifier
            
        Returns:
            Dict[str, Any]: Task status information
        """
        with self._lock:
            return self._task_stats.get(task_id, {}).copy()
    
    def get_system_stats(self) -> Dict[str, List[float]]:
        """
        Get system resource statistics
        
        Returns:
            Dict[str, List[float]]: System statistics
        """
        with self._lock:
            return self._system_stats.copy()
    
    def _start_resource_monitoring(self) -> None:
        """Start monitoring system resources"""
        self._monitoring_thread = threading.Thread(
            target=self._monitor_resources,
            daemon=True
        )
        self._monitoring_thread.start()
    
    def _monitor_resources(self) -> None:
        """Monitor system resources periodically"""
        while True:
            try:
                # Get CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                
                # Get memory usage
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                
                with self._lock:
                    self._system_stats['cpu_usage'].append(cpu_percent)
                    self._system_stats['memory_usage'].append(memory_percent)
                    
                    # Keep only last hour of data (3600 seconds)
                    max_samples = 3600
                    for key in self._system_stats:
                        if len(self._system_stats[key]) > max_samples:
                            self._system_stats[key] = self._system_stats[key][-max_samples:]
                
                time.sleep(1)  # Update every second
                
            except Exception as e:
                print(f"Error monitoring resources: {str(e)}")
                time.sleep(5)  # Wait before retrying
    
    def get_resource_usage(self, task_id: str) -> Dict[str, float]:
        """
        Get resource usage for specific task
        
        Args:
            task_id (str): Task identifier
            
        Returns:
            Dict[str, float]: Resource usage statistics
        """
        with self._lock:
            stats = self._task_stats.get(task_id, {})
            if not stats:
                return {}
            
            return {
                'memory_peak': stats.get('memory_peak', 0),
                'cpu_usage': stats.get('cpu_usage', 0),
                'execution_time': stats.get('execution_time', 0)
            }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary for all tasks
        
        Returns:
            Dict[str, Any]: Performance summary
        """
        with self._lock:
            total_tasks = len(self._task_stats)
            completed_tasks = sum(1 for stats in self._task_stats.values() 
                                if stats['status'] == 'completed')
            failed_tasks = sum(1 for stats in self._task_stats.values() 
                             if stats['status'] == 'failed')
            
            execution_times = [stats['execution_time'] 
                             for stats in self._task_stats.values() 
                             if stats['execution_time'] is not None]
            
            return {
                'total_tasks': total_tasks,
                'completed_tasks': completed_tasks,
                'failed_tasks': failed_tasks,
                'average_execution_time': np.mean(execution_times) if execution_times else 0,
                'max_execution_time': max(execution_times) if execution_times else 0,
                'current_cpu_usage': self._system_stats['cpu_usage'][-1] 
                    if self._system_stats['cpu_usage'] else 0,
                'current_memory_usage': self._system_stats['memory_usage'][-1] 
                    if self._system_stats['memory_usage'] else 0
            }
            
class AdvancedAnnotationSystem:
    """
    Advanced system for managing and creating complex annotations
    
    Attributes:
        config (VisualizationConfig): Visualization configuration
        base_settings (BaseVisualizationSettings): Base visualization settings
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize annotation system
        
        Args:
            config (Optional[VisualizationConfig]): Visualization configuration
        """
        self.config = config or VisualizationConfig()
        self.base_settings = BaseVisualizationSettings(self.config)
        self._annotations = []
    
    def add_text_annotation(self,
                          text: str,
                          x: Union[float, str],
                          y: float,
                          annotation_type: str = 'default',
                          **kwargs) -> Dict[str, Any]:
        """
        Add text annotation with advanced styling
        
        Args:
            text (str): Annotation text
            x (Union[float, str]): X-coordinate
            y (float): Y-coordinate
            annotation_type (str): Type of annotation
            **kwargs: Additional annotation settings
            
        Returns:
            Dict[str, Any]: Annotation settings
        """
        style = self._get_annotation_style(annotation_type)
        style.update(kwargs)
        
        annotation = self.base_settings.create_annotation(text, x, y)
        annotation.update(style)
        
        self._annotations.append(annotation)
        return annotation
    
    def add_pattern_annotation(self,
                             pattern_name: str,
                             x: Union[float, str],
                             y: float,
                             direction: str = 'up',
                             confidence: float = 1.0) -> Dict[str, Any]:
        """
        Add pattern-specific annotation
        
        Args:
            pattern_name (str): Name of the pattern
            x (Union[float, str]): X-coordinate
            y (float): Y-coordinate
            direction (str): Pattern direction ('up' or 'down')
            confidence (float): Pattern confidence level
            
        Returns:
            Dict[str, Any]: Pattern annotation settings
        """
        base_annotation = self.base_settings.create_annotation(
            text=pattern_name,
            x=x,
            y=y,
            is_pattern=True
        )
        
        # Adjust style based on direction and confidence
        style = {
            'arrowhead': 2 if direction == 'up' else 3,
            'arrowcolor': (self.config.color_scheme['bullish'] if direction == 'up'
                          else self.config.color_scheme['bearish']),
            'opacity': min(1.0, max(0.3, confidence)),
            'font': {
                'color': (self.config.color_scheme['bullish'] if direction == 'up'
                         else self.config.color_scheme['bearish'])
            }
        }
        
        base_annotation.update(style)
        self._annotations.append(base_annotation)
        return base_annotation
    
    def add_technical_annotation(self,
                               indicator_name: str,
                               value: float,
                               x: Union[float, str],
                               y: float,
                               threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Add technical indicator annotation
        
        Args:
            indicator_name (str): Name of the indicator
            value (float): Indicator value
            x (Union[float, str]): X-coordinate
            y (float): Y-coordinate
            threshold (Optional[float]): Optional threshold value
            
        Returns:
            Dict[str, Any]: Technical annotation settings
        """
        text = f"{indicator_name}: {value:.2f}"
        if threshold is not None:
            text += f" (Threshold: {threshold:.2f})"
        
        annotation = self.base_settings.create_annotation(text, x, y)
        
        # Style based on threshold if provided
        if threshold is not None:
            color = (self.config.color_scheme['bullish'] if value > threshold
                    else self.config.color_scheme['bearish'])
            annotation.update({
                'font': {'color': color},
                'bordercolor': color
            })
        
        self._annotations.append(annotation)
        return annotation
    
    def add_zone_annotation(self,
                          zone_type: str,
                          x_range: Tuple[Union[float, str], Union[float, str]],
                          y_range: Tuple[float, float],
                          text: Optional[str] = None) -> Dict[str, Any]:
        """
        Add annotation for a zone or region
        
        Args:
            zone_type (str): Type of zone
            x_range (Tuple): Start and end x-coordinates
            y_range (Tuple): Lower and upper y-coordinates
            text (Optional[str]): Optional annotation text
            
        Returns:
            Dict[str, Any]: Zone annotation settings
        """
        style = self._get_zone_style(zone_type)
        
        annotation = {
            'type': 'rect',
            'x0': x_range[0],
            'x1': x_range[1],
            'y0': y_range[0],
            'y1': y_range[1],
            'fillcolor': style['fill'],
            'opacity': style['opacity'],
            'line': {
                'color': style['border'],
                'width': 1,
                'dash': 'dash'
            }
        }
        
        if text:
            text_annotation = self.add_text_annotation(
                text=text,
                x=x_range[0],
                y=y_range[1],
                showarrow=False,
                bordercolor=style['border']
            )
            return {'shape': annotation, 'text': text_annotation}
        
        return {'shape': annotation}
    
class MultipleTimeframeSynchronizer:
    """
    Synchronizes data and visualizations across multiple timeframes
    
    Attributes:
        timeframes (Dict[str, pd.DataFrame]): Data for different timeframes
        config (VisualizationConfig): Visualization configuration
        base_settings (BaseVisualizationSettings): Base visualization settings
    """
    
    def __init__(self, 
                 data: pd.DataFrame,
                 config: Optional[VisualizationConfig] = None):
        """
        Initialize timeframe synchronizer
        
        Args:
            data (pd.DataFrame): Base timeframe data
            config (Optional[VisualizationConfig]): Visualization configuration
        """
        self.config = config or VisualizationConfig()
        self.base_settings = BaseVisualizationSettings(self.config)
        self.timeframes = {}
        
        # Initialize with base data
        self.base_data = data
        self._generate_timeframes()
    
    def _generate_timeframes(self) -> None:
        """Generate standard timeframes from base data"""
        self.timeframes = {
            '1D': self.base_data,  # Base timeframe
            '1W': self._resample_data('W'),
            '1M': self._resample_data('M'),
            '1H': self._resample_data('H') if self._is_intraday() else None
        }
        
        # Remove None values
        self.timeframes = {k: v for k, v in self.timeframes.items() if v is not None}
    
    def _resample_data(self, freq: str) -> pd.DataFrame:
        """
        Resample data to different timeframe
        
        Args:
            freq (str): Frequency string for resampling
            
        Returns:
            pd.DataFrame: Resampled data
        """
        resampled = self.base_data.resample(freq).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum' if 'Volume' in self.base_data.columns else None
        }).dropna()
        
        return resampled
    
    def _is_intraday(self) -> bool:
        """
        Check if base data is intraday
        
        Returns:
            bool: True if data is intraday
        """
        if len(self.base_data) < 2:
            return False
            
        time_diff = self.base_data.index[1] - self.base_data.index[0]
        return time_diff.total_seconds() < 86400  # Less than one day
    
    def create_synchronized_view(self, 
                               timeframes: List[str] = None) -> go.Figure:
        """
        Create synchronized multi-timeframe view
        
        Args:
            timeframes (List[str]): List of timeframes to include
            
        Returns:
            go.Figure: Synchronized multi-timeframe figure
        """
        if timeframes is None:
            timeframes = list(self.timeframes.keys())
        
        # Create subplot figure
        fig = make_subplots(
            rows=len(timeframes),
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=timeframes
        )
        
        # Add candlesticks for each timeframe
        for i, tf in enumerate(timeframes, 1):
            if tf in self.timeframes:
                df = self.timeframes[tf]
                
                # Add candlestick chart
                fig.add_trace(
                    go.Candlestick(
                        x=df.index,
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'],
                        name=f'Price ({tf})'
                    ),
                    row=i, col=1
                )
                
                # Add volume if available
                if 'Volume' in df.columns:
                    fig.add_trace(
                        go.Bar(
                            x=df.index,
                            y=df['Volume'],
                            name=f'Volume ({tf})',
                            marker_color=self._get_volume_colors(df),
                            opacity=0.5
                        ),
                        row=i, col=1
                    )
        
        # Update layout
        fig = self.base_settings.apply_default_layout(fig)
        fig = self._add_sync_buttons(fig)
        
        return fig
    
    def _get_volume_colors(self, df: pd.DataFrame) -> List[str]:
        """
        Get volume bar colors based on price movement
        
        Args:
            df (pd.DataFrame): Price data
            
        Returns:
            List[str]: List of color codes
        """
        return [
            self.config.color_scheme['volume_up'] if close >= open_
            else self.config.color_scheme['volume_down']
            for close, open_ in zip(df['Close'], df['Open'])
        ]
    
    def _add_sync_buttons(self, fig: go.Figure) -> go.Figure:
        """
        Add timeframe synchronization buttons
        
        Args:
            fig (go.Figure): Plotly figure
            
        Returns:
            go.Figure: Figure with sync buttons
        """
        buttons = []
        ranges = {
            '1M': {'days': 30},
            '3M': {'days': 90},
            '6M': {'days': 180},
            '1Y': {'days': 365},
            'YTD': {'days': None},  # Special case
            'ALL': {'days': None}
        }
        
        for label, range_dict in ranges.items():
            if range_dict['days'] is None:
                if label == 'YTD':
                    # Calculate YTD range
                    start_date = pd.Timestamp.now().replace(
                        month=1, day=1, hour=0, minute=0, second=0)
                else:
                    # ALL - use full range
                    start_date = self.base_data.index[0]
                    
                end_date = self.base_data.index[-1]
            else:
                end_date = self.base_data.index[-1]
                start_date = end_date - pd.Timedelta(days=range_dict['days'])
            
            buttons.append(dict(
                label=label,
                method='relayout',
                args=[{'xaxis.range': [start_date, end_date]}]
            ))
        
        fig.update_layout(
            updatemenus=[{
                'buttons': buttons,
                'direction': 'left',
                'showactive': True,
                'x': 0.1,
                'y': 1.15,
                'xanchor': 'left',
                'yanchor': 'top'
            }]
        )
        
        return fig
        
    def add_synchronized_indicators(self,
                                  indicator_name: str,
                                  params: Dict[str, Any] = None,
                                  timeframes: List[str] = None) -> go.Figure:
        """
        Add technical indicator synchronized across timeframes
        
        Args:
            indicator_name (str): Name of indicator to add
            params (Dict[str, Any]): Indicator parameters
            timeframes (List[str]): Timeframes to include
            
        Returns:
            go.Figure: Updated figure with synchronized indicators
        """
        if timeframes is None:
            timeframes = list(self.timeframes.keys())
            
        fig = make_subplots(
            rows=len(timeframes),
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05
        )
        
        for i, tf in enumerate(timeframes, 1):
            if tf in self.timeframes:
                df = self.timeframes[tf]
                
                # Calculate indicator for this timeframe
                indicator_values = self._calculate_indicator(
                    df, indicator_name, params)
                
                # Add indicator trace
                if isinstance(indicator_values, pd.DataFrame):
                    # Handle multi-line indicators (e.g., Bollinger Bands)
                    for col in indicator_values.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=df.index,
                                y=indicator_values[col],
                                name=f'{col} ({tf})',
                                line=dict(
                                    width=1,
                                    dash='dash' if 'upper' in col.lower() or 
                                                  'lower' in col.lower() else 'solid'
                                )
                            ),
                            row=i, col=1
                        )
                else:
                    # Single-line indicator
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=indicator_values,
                            name=f'{indicator_name} ({tf})',
                            line=dict(width=1)
                        ),
                        row=i, col=1
                    )
        
        return self._apply_indicator_styling(fig, indicator_name)
    
    def add_synchronized_patterns(self,
                                pattern_types: List[str],
                                timeframes: List[str] = None) -> go.Figure:
        """
        Add pattern detection synchronized across timeframes
        
        Args:
            pattern_types (List[str]): Types of patterns to detect
            timeframes (List[str]): Timeframes to include
            
        Returns:
            go.Figure: Updated figure with synchronized patterns
        """
        if timeframes is None:
            timeframes = list(self.timeframes.keys())
            
        fig = self.create_synchronized_view(timeframes)
        
        for i, tf in enumerate(timeframes, 1):
            if tf in self.timeframes:
                df = self.timeframes[tf]
                
                for pattern_type in pattern_types:
                    # Detect patterns for this timeframe
                    patterns = self._detect_patterns(df, pattern_type)
                    
                    if isinstance(patterns, tuple):
                        # Handle bullish/bearish patterns
                        bullish, bearish = patterns
                        
                        # Add bullish markers
                        if bullish.any():
                            fig.add_trace(
                                go.Scatter(
                                    x=df.index[bullish],
                                    y=df['Low'][bullish] * 0.99,
                                    mode='markers',
                                    marker=dict(
                                        symbol='triangle-up',
                                        size=8,
                                        color=self.config.color_scheme['bullish']
                                    ),
                                    name=f'{pattern_type} Bullish ({tf})',
                                    opacity=self.config.pattern_opacity
                                ),
                                row=i, col=1
                            )
                        
                        # Add bearish markers
                        if bearish.any():
                            fig.add_trace(
                                go.Scatter(
                                    x=df.index[bearish],
                                    y=df['High'][bearish] * 1.01,
                                    mode='markers',
                                    marker=dict(
                                        symbol='triangle-down',
                                        size=8,
                                        color=self.config.color_scheme['bearish']
                                    ),
                                    name=f'{pattern_type} Bearish ({tf})',
                                    opacity=self.config.pattern_opacity
                                ),
                                row=i, col=1
                            )
        
        return self._apply_pattern_styling(fig)
    
    def _calculate_indicator(self,
                           df: pd.DataFrame,
                           indicator_name: str,
                           params: Dict[str, Any] = None) -> Union[pd.Series, pd.DataFrame]:
        """
        Calculate technical indicator
        
        Args:
            df (pd.DataFrame): Price data
            indicator_name (str): Indicator name
            params (Dict[str, Any]): Indicator parameters
            
        Returns:
            Union[pd.Series, pd.DataFrame]: Calculated indicator values
        """
        params = params or {}
        
        # Create indicator manager instance
        indicator_manager = IndicatorManager(df)
        
        # Calculate indicator
        try:
            return indicator_manager.get_indicator(indicator_name, params)
        except Exception as e:
            print(f"Error calculating {indicator_name}: {str(e)}")
            return pd.Series(index=df.index)
    
    def _detect_patterns(self,
                        df: pd.DataFrame,
                        pattern_type: str) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
        """
        Detect patterns in data
        
        Args:
            df (pd.DataFrame): Price data
            pattern_type (str): Type of pattern to detect
            
        Returns:
            Union[pd.Series, Tuple[pd.Series, pd.Series]]: Pattern signals
        """
        try:
            pattern_analyzer = MarketRegimeAnalyzer(df)
            return pattern_analyzer._get_pattern_signals(pattern_type)
        except Exception as e:
            print(f"Error detecting {pattern_type}: {str(e)}")
            return pd.Series(False, index=df.index)
        
    def _apply_indicator_styling(self, fig: go.Figure, indicator_name: str) -> go.Figure:
        """
        Apply consistent styling to indicator plots
        
        Args:
            fig (go.Figure): Plotly figure
            indicator_name (str): Name of indicator
            
        Returns:
            go.Figure: Styled figure
        """
        # Update layout based on indicator type
        indicator_styles = {
            'bollinger_bands': {
                'colors': ['purple', 'blue', 'purple'],
                'fills': [None, None, 'tonexty'],
                'opacity': 0.1
            },
            'rsi': {
                'range': [0, 100],
                'levels': [30, 70],
                'colors': ['red', 'green']
            },
            'macd': {
                'colors': ['blue', 'orange', 'gray'],
                'bar_colors': ['green', 'red']
            }
        }
        
        if indicator_name.lower() in indicator_styles:
            style = indicator_styles[indicator_name.lower()]
            fig = self._apply_specific_indicator_style(fig, style, indicator_name)
        
        # Apply general styling
        fig.update_layout(
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            )
        )
        
        return fig

    def _apply_pattern_styling(self, fig: go.Figure) -> go.Figure:
        """
        Apply consistent styling to pattern markers
        
        Args:
            fig (go.Figure): Plotly figure
            
        Returns:
            go.Figure: Styled figure
        """
        fig.update_layout(
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            )
        )
        
        # Add pattern highlight regions
        for trace in fig.data:
            if trace.name and ('Bullish' in trace.name or 'Bearish' in trace.name):
                fig.add_shape(
                    type='rect',
                    x0=trace.x[0] if len(trace.x) > 0 else None,
                    x1=trace.x[-1] if len(trace.x) > 0 else None,
                    y0=0,
                    y1=1,
                    xref='x',
                    yref='paper',
                    fillcolor=self.config.color_scheme['bullish' if 'Bullish' in trace.name 
                                                    else 'bearish'],
                    opacity=0.1,
                    layer='below',
                    line_width=0
                )
        
        return fig

    def analyze_cross_timeframe_signals(self,
                                    indicator_name: str,
                                    params: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Analyze signals across multiple timeframes
        
        Args:
            indicator_name (str): Name of indicator to analyze
            params (Dict[str, Any]): Indicator parameters
            
        Returns:
            pd.DataFrame: Cross-timeframe analysis results
        """
        signals = {}
        
        for tf, df in self.timeframes.items():
            # Calculate indicator for each timeframe
            indicator_values = self._calculate_indicator(df, indicator_name, params)
            
            # Generate signals based on indicator type
            signals[tf] = self._generate_signals(indicator_values, indicator_name)
        
        # Combine and analyze signals
        return self._combine_timeframe_signals(signals)

    def _generate_signals(self,
                        indicator_values: Union[pd.Series, pd.DataFrame],
                        indicator_name: str) -> pd.Series:
        """
        Generate trading signals from indicator values
        
        Args:
            indicator_values: Indicator calculations
            indicator_name (str): Name of indicator
            
        Returns:
            pd.Series: Generated signals
        """
        if isinstance(indicator_values, pd.DataFrame):
            # Handle multi-line indicators
            if indicator_name.lower() == 'bollinger_bands':
                middle = indicator_values['middle']
                upper = indicator_values['upper']
                lower = indicator_values['lower']
                
                return pd.Series(
                    np.where(self.timeframes['1D']['Close'] > upper, -1,
                            np.where(self.timeframes['1D']['Close'] < lower, 1, 0)),
                    index=indicator_values.index
                )
        else:
            # Handle single-line indicators
            if indicator_name.lower() == 'rsi':
                return pd.Series(
                    np.where(indicator_values > 70, -1,
                            np.where(indicator_values < 30, 1, 0)),
                    index=indicator_values.index
                )
        
        return pd.Series(0, index=indicator_values.index)

    def _combine_timeframe_signals(self, signals: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Combine signals from different timeframes
        
        Args:
            signals (Dict[str, pd.Series]): Signals for each timeframe
            
        Returns:
            pd.DataFrame: Combined signal analysis
        """
        # Resample all signals to the base timeframe
        resampled_signals = {}
        base_index = self.timeframes['1D'].index
        
        for tf, signal in signals.items():
            if tf == '1D':
                resampled_signals[tf] = signal
            else:
                # Forward fill signals to base timeframe
                resampled_signals[tf] = signal.reindex(base_index, method='ffill')
        
        # Combine signals
        combined = pd.DataFrame(resampled_signals)
        
        # Calculate signal strength
        combined['strength'] = combined.mean(axis=1)
        
        # Add confidence based on agreement between timeframes
        combined['confidence'] = (combined.abs().mean(axis=1) * 
                                (combined != 0).mean(axis=1))
        
        return combined

    def confirm_patterns_across_timeframes(self,
                                         pattern_type: str,
                                         min_confirmations: int = 2) -> pd.DataFrame:
        """
        Confirm pattern signals across multiple timeframes
        
        Args:
            pattern_type (str): Type of pattern to analyze
            min_confirmations (int): Minimum timeframes for confirmation
            
        Returns:
            pd.DataFrame: Confirmed patterns with metadata
        """
        # Get patterns for each timeframe
        pattern_signals = {}
        for tf, df in self.timeframes.items():
            signals = self._detect_patterns(df, pattern_type)
            if isinstance(signals, tuple):
                pattern_signals[tf] = {
                    'bullish': signals[0],
                    'bearish': signals[1]
                }
            else:
                pattern_signals[tf] = {'neutral': signals}
        
        # Find confirmed patterns
        confirmed_patterns = []
        base_df = self.timeframes['1D']
        
        for idx in base_df.index:
            confirmations = {
                'bullish': 0,
                'bearish': 0,
                'neutral': 0
            }
            
            # Count confirmations across timeframes
            for tf, signals in pattern_signals.items():
                for direction, signal in signals.items():
                    if idx in signal.index and signal[idx]:
                        confirmations[direction] += 1
            
            # Record confirmed patterns
            for direction, count in confirmations.items():
                if count >= min_confirmations:
                    confirmed_patterns.append({
                        'date': idx,
                        'pattern': pattern_type,
                        'direction': direction,
                        'confirmations': count,
                        'timeframes': [
                            tf for tf, signals in pattern_signals.items()
                            if direction in signals and 
                            idx in signals[direction].index and 
                            signals[direction][idx]
                        ],
                        'price': base_df['Close'][idx]
                    })
        
        return pd.DataFrame(confirmed_patterns)

    def filter_signals(self,
                      signals: pd.DataFrame,
                      filter_type: str = 'consensus',
                      threshold: float = 0.7) -> pd.DataFrame:
        """
        Filter signals based on specified criteria
        
        Args:
            signals (pd.DataFrame): Signal data
            filter_type (str): Type of filter to apply
            threshold (float): Filter threshold
            
        Returns:
            pd.DataFrame: Filtered signals
        """
        if filter_type == 'consensus':
            # Filter based on timeframe consensus
            mask = signals['confidence'] >= threshold
            return signals[mask]
            
        elif filter_type == 'strength':
            # Filter based on signal strength
            mask = abs(signals['strength']) >= threshold
            return signals[mask]
            
        elif filter_type == 'persistence':
            # Filter based on signal persistence
            persistence = signals.rolling(window=3).mean()
            mask = abs(persistence['strength']) >= threshold
            return signals[mask]
        
        return signals

    def get_timeframe_alignment(self) -> pd.DataFrame:
        """
        Calculate alignment of trends across timeframes
        
        Returns:
            pd.DataFrame: Trend alignment analysis
        """
        trends = {}
        
        for tf, df in self.timeframes.items():
            # Calculate trend indicators
            sma_short = df['Close'].rolling(window=20).mean()
            sma_long = df['Close'].rolling(window=50).mean()
            
            trends[tf] = pd.Series(
                np.where(sma_short > sma_long, 1,
                        np.where(sma_short < sma_long, -1, 0)),
                index=df.index
            )
        
        # Combine trends
        alignment = pd.DataFrame(trends)
        
        # Calculate alignment metrics
        alignment['consensus'] = alignment.mean(axis=1)
        alignment['agreement'] = (alignment != 0).mean(axis=1)
        
        return alignment

    def synchronize_regime_analysis(self) -> pd.DataFrame:
        """
        Synchronize market regime analysis across timeframes
        
        Returns:
            pd.DataFrame: Synchronized regime analysis
        """
        regimes = {}
        
        for tf, df in self.timeframes.items():
            analyzer = MarketRegimeAnalyzer(df)
            regime_data = analyzer.analyze_market_regime()
            regimes[tf] = pd.Series(
                [r.regime_type for r in regime_data],
                index=df.index
            )
        
        # Combine regime data
        combined_regimes = pd.DataFrame(regimes)
        
        # Add regime consistency
        combined_regimes['consistency'] = (
            combined_regimes.apply(lambda x: x.value_counts().iloc[0] / len(x), axis=1)
        )
        
        return combined_regimes
    
class DrawingTools:
    """
    Interactive drawing tools for technical analysis
    
    Attributes:
        fig (go.Figure): Plotly figure
        config (VisualizationConfig): Visualization configuration
        drawings (Dict): Storage for drawing objects
    """
    
    def __init__(self, fig: go.Figure, config: Optional[VisualizationConfig] = None):
        """
        Initialize drawing tools
        
        Args:
            fig (go.Figure): Plotly figure to draw on
            config (Optional[VisualizationConfig]): Visualization configuration
        """
        self.fig = fig
        self.config = config or VisualizationConfig()
        self.drawings = {
            'trendlines': [],
            'horizontal_lines': [],
            'fibonacci': [],
            'rectangles': [],
            'text': []
        }
    
    def add_trendline(self,
                     points: List[Tuple[Union[str, float], float]],
                     extend: bool = True,
                     style: Dict[str, Any] = None) -> None:
        """
        Add trendline to chart
        
        Args:
            points (List[Tuple]): List of (x, y) coordinates
            extend (bool): Whether to extend line
            style (Dict[str, Any]): Line style settings
        """
        default_style = {
            'color': self.config.color_scheme['neutral'],
            'width': 1,
            'dash': 'solid'
        }
        style = {**default_style, **(style or {})}
        
        x_values, y_values = zip(*points)
        
        if extend:
            # Extend line in both directions
            x_range = self.fig.layout.xaxis.range
            slope = (y_values[1] - y_values[0]) / (x_values[1] - x_values[0])
            
            # Calculate extended points
            x_ext = [x_range[0], x_range[1]]
            y_ext = [
                y_values[0] + slope * (x_range[0] - x_values[0]),
                y_values[0] + slope * (x_range[1] - x_values[0])
            ]
        else:
            x_ext, y_ext = x_values, y_values
        
        self.fig.add_shape(
            type='line',
            x0=x_ext[0],
            y0=y_ext[0],
            x1=x_ext[1],
            y1=y_ext[1],
            line=style
        )
        
        self.drawings['trendlines'].append({
            'points': points,
            'extended': extend,
            'style': style
        })
    
    def add_horizontal_line(self,
                          y_value: float,
                          label: str = '',
                          style: Dict[str, Any] = None) -> None:
        """
        Add horizontal line to chart
        
        Args:
            y_value (float): Y-coordinate for line
            label (str): Line label
            style (Dict[str, Any]): Line style settings
        """
        default_style = {
            'color': self.config.color_scheme['neutral'],
            'width': 1,
            'dash': 'dash'
        }
        style = {**default_style, **(style or {})}
        
        self.fig.add_hline(
            y=y_value,
            line=style,
            annotation_text=label if label else None,
            annotation=dict(
                font_size=self.config.annotation_font_size
            ) if label else None
        )
        
        self.drawings['horizontal_lines'].append({
            'y_value': y_value,
            'label': label,
            'style': style
        })
    
    def add_fibonacci_retracement(self,
                                high_point: Tuple[Union[str, float], float],
                                low_point: Tuple[Union[str, float], float]) -> None:
        """
        Add Fibonacci retracement levels
        
        Args:
            high_point (Tuple): High point coordinates (x, y)
            low_point (Tuple): Low point coordinates (x, y)
        """
        levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
        price_range = high_point[1] - low_point[1]
        
        for level in levels:
            price_level = high_point[1] - price_range * level
            self.add_horizontal_line(
                y_value=price_level,
                label=f'Fib {level:.3f}',
                style={
                    'color': self.config.color_scheme['complex'],
                    'width': 1,
                    'dash': 'dot'
                }
            )
        
        self.drawings['fibonacci'].append({
            'high_point': high_point,
            'low_point': low_point,
            'levels': levels
        })
    
    def add_rectangle(self,
                     x_range: Tuple[Union[str, float], Union[str, float]],
                     y_range: Tuple[float, float],
                     style: Dict[str, Any] = None) -> None:
        """
        Add rectangle to chart
        
        Args:
            x_range (Tuple): X-coordinates range
            y_range (Tuple): Y-coordinates range
            style (Dict[str, Any]): Rectangle style settings
        """
        default_style = {
            'fillcolor': self.config.color_scheme['neutral'],
            'opacity': 0.2,
            'line': {
                'color': self.config.color_scheme['neutral'],
                'width': 1
            }
        }
        style = {**default_style, **(style or {})}
        
        self.fig.add_shape(
            type='rect',
            x0=x_range[0],
            y0=y_range[0],
            x1=x_range[1],
            y1=y_range[1],
            **style
        )
        
        self.drawings['rectangles'].append({
            'x_range': x_range,
            'y_range': y_range,
            'style': style
        })

    def add_text_annotation(self,
                          text: str,
                          position: Tuple[Union[str, float], float],
                          style: Dict[str, Any] = None) -> None:
        """
        Add text annotation to chart
        
        Args:
            text (str): Annotation text
            position (Tuple): Position coordinates (x, y)
            style (Dict[str, Any]): Annotation style settings
        """
        default_style = {
            'font_size': self.config.annotation_font_size,
            'font_color': self.config.color_scheme['text'],
            'bgcolor': self.config.annotation_settings['style']['background_color'],
            'bordercolor': self.config.annotation_settings['style']['border_color'],
            'borderwidth': self.config.annotation_settings['style']['border_width']
        }
        style = {**default_style, **(style or {})}
        
        self.fig.add_annotation(
            text=text,
            x=position[0],
            y=position[1],
            showarrow=False,
            **style
        )
        
        self.drawings['text'].append({
            'text': text,
            'position': position,
            'style': style
        })

    def add_channel(self,
                   upper_points: List[Tuple[Union[str, float], float]],
                   lower_points: List[Tuple[Union[str, float], float]],
                   style: Dict[str, Any] = None) -> None:
        """
        Add price channel
        
        Args:
            upper_points (List[Tuple]): Points for upper line
            lower_points (List[Tuple]): Points for lower line
            style (Dict[str, Any]): Channel style settings
        """
        default_style = {
            'fillcolor': self.config.color_scheme['neutral'],
            'opacity': 0.1,
            'line_color': self.config.color_scheme['neutral']
        }
        style = {**default_style, **(style or {})}
        
        # Add upper and lower trendlines
        self.add_trendline(upper_points, style={'color': style['line_color']})
        self.add_trendline(lower_points, style={'color': style['line_color']})
        
        # Fill the channel area
        x_values = [p[0] for p in upper_points + lower_points[::-1]]
        y_values = [p[1] for p in upper_points + lower_points[::-1]]
        
        self.fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                fill='toself',
                fillcolor=style['fillcolor'],
                opacity=style['opacity'],
                line={'width': 0},
                showlegend=False
            )
        )

    def remove_drawing(self, drawing_id: str) -> None:
        """
        Remove specific drawing
        
        Args:
            drawing_id (str): ID of drawing to remove
        """
        for category in self.drawings:
            if drawing_id in self.drawings[category]:
                self.drawings[category].remove(drawing_id)
                self._redraw_figure()
                break

    def clear_all_drawings(self) -> None:
        """Clear all drawings from chart"""
        for category in self.drawings:
            self.drawings[category].clear()
        self._redraw_figure()

    def _redraw_figure(self) -> None:
        """Redraw figure with current drawings"""
        # Store base data
        base_data = [trace for trace in self.fig.data 
                    if trace.name not in ['drawing', 'annotation']]
        
        # Clear figure
        self.fig.data = []
        
        # Restore base data
        for trace in base_data:
            self.fig.add_trace(trace)
        
        # Redraw all drawings
        for category, drawings in self.drawings.items():
            for drawing in drawings:
                if category == 'trendlines':
                    self.add_trendline(
                        drawing['points'],
                        drawing['extended'],
                        drawing['style']
                    )
                elif category == 'horizontal_lines':
                    self.add_horizontal_line(
                        drawing['y_value'],
                        drawing['label'],
                        drawing['style']
                    )
                elif category == 'fibonacci':
                    self.add_fibonacci_retracement(
                        drawing['high_point'],
                        drawing['low_point']
                    )
                elif category == 'rectangles':
                    self.add_rectangle(
                        drawing['x_range'],
                        drawing['y_range'],
                        drawing['style']
                    )
                elif category == 'text':
                    self.add_text_annotation(
                        drawing['text'],
                        drawing['position'],
                        drawing['style']
                    )

    def save_drawings(self, filename: str) -> None:
        """
        Save current drawings to file
        
        Args:
            filename (str): File to save drawings to
        """
        with open(filename, 'w') as f:
            json.dump(self.drawings, f)

    def load_drawings(self, filename: str) -> None:
        """
        Load drawings from file
        
        Args:
            filename (str): File to load drawings from
        """
        with open(filename, 'r') as f:
            self.drawings = json.load(f)
        self._redraw_figure()
        
class ConfigurationManager:
    """
    Manages visualization settings and configuration persistence
    
    Attributes:
        config (VisualizationConfig): Current configuration
        base_settings (BaseVisualizationSettings): Base visualization settings
        _config_file (str): Configuration file path
    """
    
    def __init__(self, 
                 config: Optional[VisualizationConfig] = None,
                 config_file: str = 'viz_config.json'):
        """
        Initialize configuration manager
        
        Args:
            config (Optional[VisualizationConfig]): Initial configuration
            config_file (str): Configuration file path
        """
        self.config = config or VisualizationConfig()
        self.base_settings = BaseVisualizationSettings(self.config)
        self._config_file = config_file
        self._user_preferences = {}
    
    def export_configuration(self, filename: Optional[str] = None) -> None:
        """
        Export current configuration to file
        
        Args:
            filename (Optional[str]): Export filename
        """
        export_file = filename or self._config_file
        
        config_dict = {
            'color_scheme': self.config.color_scheme,
            'theme': self.config.theme,
            'dimensions': {
                'default_height': self.config.default_height,
                'default_width': self.config.default_width
            },
            'appearance': {
                'pattern_opacity': self.config.pattern_opacity,
                'show_grid': self.config.show_grid,
                'annotation_font_size': self.config.annotation_font_size
            },
            'fonts': self.config.fonts,
            'layout': self.config.layout,
            'grid_settings': self.config.grid_settings,
            'annotation_settings': self.config.annotation_settings,
            'interactive_settings': self.config.interactive_settings,
            'user_preferences': self._user_preferences
        }
        
        with open(export_file, 'w') as f:
            json.dump(config_dict, f, indent=4)
    
    def import_configuration(self, filename: Optional[str] = None) -> None:
        """
        Import configuration from file
        
        Args:
            filename (Optional[str]): Import filename
        """
        import_file = filename or self._config_file
        
        try:
            with open(import_file, 'r') as f:
                config_dict = json.load(f)
            
            # Update configuration
            self._update_config_from_dict(config_dict)
            
        except FileNotFoundError:
            print(f"Configuration file {import_file} not found")
        except json.JSONDecodeError:
            print(f"Invalid configuration file format in {import_file}")
    
    def set_user_preference(self, key: str, value: Any) -> None:
        """
        Set user preference
        
        Args:
            key (str): Preference key
            value (Any): Preference value
        """
        self._user_preferences[key] = value
        self._apply_user_preferences()
    
    def get_user_preference(self, key: str, default: Any = None) -> Any:
        """
        Get user preference
        
        Args:
            key (str): Preference key
            default (Any): Default value if not found
            
        Returns:
            Any: Preference value
        """
        return self._user_preferences.get(key, default)
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to defaults"""
        self.config = VisualizationConfig()
        self._user_preferences = {}
        self.base_settings = BaseVisualizationSettings(self.config)
    
    def _update_config_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """
        Update configuration from dictionary
        
        Args:
            config_dict (Dict[str, Any]): Configuration dictionary
        """
        # Update main configuration
        if 'color_scheme' in config_dict:
            self.config.color_scheme = config_dict['color_scheme']
        if 'theme' in config_dict:
            self.config.theme = config_dict['theme']
            
        # Update dimensions
        if 'dimensions' in config_dict:
            self.config.default_height = config_dict['dimensions']['default_height']
            self.config.default_width = config_dict['dimensions']['default_width']
            
        # Update appearance
        if 'appearance' in config_dict:
            self.config.pattern_opacity = config_dict['appearance']['pattern_opacity']
            self.config.show_grid = config_dict['appearance']['show_grid']
            self.config.annotation_font_size = config_dict['appearance']['annotation_font_size']
            
        # Update advanced settings
        if 'fonts' in config_dict:
            self.config.fonts = config_dict['fonts']
        if 'layout' in config_dict:
            self.config.layout = config_dict['layout']
        if 'grid_settings' in config_dict:
            self.config.grid_settings = config_dict['grid_settings']
        if 'annotation_settings' in config_dict:
            self.config.annotation_settings = config_dict['annotation_settings']
        if 'interactive_settings' in config_dict:
            self.config.interactive_settings = config_dict['interactive_settings']
            
        # Update user preferences
        if 'user_preferences' in config_dict:
            self._user_preferences = config_dict['user_preferences']
            self._apply_user_preferences()
            
    def _apply_user_preferences(self) -> None:
        """Apply stored user preferences to configuration"""
        for key, value in self._user_preferences.items():
            if key.startswith('chart_'):
                self._apply_chart_preference(key, value)
            elif key.startswith('indicator_'):
                self._apply_indicator_preference(key, value)
            elif key.startswith('pattern_'):
                self._apply_pattern_preference(key, value)

    def _apply_chart_preference(self, key: str, value: Any) -> None:
        """
        Apply chart-specific preference
        
        Args:
            key (str): Preference key
            value (Any): Preference value
        """
        chart_prefs = {
            'chart_style': lambda x: setattr(self.config, 'theme', x),
            'chart_height': lambda x: setattr(self.config, 'default_height', x),
            'chart_width': lambda x: setattr(self.config, 'default_width', x),
            'chart_grid': lambda x: setattr(self.config, 'show_grid', x)
        }
        
        if key in chart_prefs:
            chart_prefs[key](value)

    def _apply_indicator_preference(self, key: str, value: Any) -> None:
        """
        Apply indicator-specific preference
        
        Args:
            key (str): Preference key
            value (Any): Preference value
        """
        indicator_prefs = {
            'indicator_colors': lambda x: self._update_indicator_colors(x),
            'indicator_style': lambda x: self._update_indicator_style(x),
            'indicator_placement': lambda x: self._update_indicator_placement(x)
        }
        
        if key in indicator_prefs:
            indicator_prefs[key](value)

    def _apply_pattern_preference(self, key: str, value: Any) -> None:
        """
        Apply pattern-specific preference
        
        Args:
            key (str): Preference key
            value (Any): Preference value
        """
        pattern_prefs = {
            'pattern_opacity': lambda x: setattr(self.config, 'pattern_opacity', x),
            'pattern_colors': lambda x: self._update_pattern_colors(x),
            'pattern_annotations': lambda x: self._update_pattern_annotations(x)
        }
        
        if key in pattern_prefs:
            pattern_prefs[key](value)

    def create_config_backup(self) -> None:
        """Create backup of current configuration"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = f'config_backup_{timestamp}.json'
        self.export_configuration(backup_file)
    
    def restore_config_backup(self, backup_file: str) -> None:
        """
        Restore configuration from backup
        
        Args:
            backup_file (str): Backup file to restore from
        """
        self.import_configuration(backup_file)

    def validate_configuration(self) -> List[str]:
        """
        Validate current configuration
        
        Returns:
            List[str]: List of validation errors
        """
        errors = []
        
        # Validate color scheme
        required_colors = ['bullish', 'bearish', 'neutral', 'background', 'text']
        for color in required_colors:
            if color not in self.config.color_scheme:
                errors.append(f"Missing required color: {color}")
        
        # Validate dimensions
        if self.config.default_height <= 0:
            errors.append("Invalid default height")
        if self.config.default_width <= 0:
            errors.append("Invalid default width")
        
        # Validate opacity
        if not 0 <= self.config.pattern_opacity <= 1:
            errors.append("Pattern opacity must be between 0 and 1")
        
        return errors

    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get summary of current configuration
        
        Returns:
            Dict[str, Any]: Configuration summary
        """
        return {
            'theme': self.config.theme,
            'dimensions': f"{self.config.default_width}x{self.config.default_height}",
            'grid_enabled': self.config.show_grid,
            'active_preferences': list(self._user_preferences.keys()),
            'last_modified': datetime.now().isoformat()
        }
        
    def create_persistent_store(self) -> None:
        """Create persistent storage for configuration"""
        if not os.path.exists('config'):
            os.makedirs('config')
            os.makedirs('config/backups')
            os.makedirs('config/user_preferences')
        
        # Create default configuration if not exists
        if not os.path.exists(self._config_file):
            self.export_configuration()

    def auto_save_config(self) -> None:
        """Automatically save configuration periodically"""
        self.export_configuration()
        self.create_config_backup()

    def load_last_session(self) -> bool:
        """
        Load configuration from last session
        
        Returns:
            bool: True if successful, False otherwise
        """
        session_file = 'config/last_session.json'
        try:
            self.import_configuration(session_file)
            return True
        except Exception as e:
            print(f"Error loading last session: {str(e)}")
            return False

    def save_session(self, session_name: str) -> None:
        """
        Save current configuration as named session
        
        Args:
            session_name (str): Name of the session
        """
        session_file = f'config/user_preferences/{session_name}.json'
        self.export_configuration(session_file)
        
        # Update sessions list
        self._update_sessions_list(session_name)

    def _update_sessions_list(self, session_name: str) -> None:
        """
        Update list of saved sessions
        
        Args:
            session_name (str): Name of session to add
        """
        sessions_file = 'config/sessions.json'
        try:
            with open(sessions_file, 'r') as f:
                sessions = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            sessions = {'sessions': []}
            
        if session_name not in sessions['sessions']:
            sessions['sessions'].append(session_name)
            
        with open(sessions_file, 'w') as f:
            json.dump(sessions, f, indent=4)

    def get_saved_sessions(self) -> List[str]:
        """
        Get list of saved sessions
        
        Returns:
            List[str]: List of session names
        """
        sessions_file = 'config/sessions.json'
        try:
            with open(sessions_file, 'r') as f:
                sessions = json.load(f)
            return sessions['sessions']
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def cleanup_old_backups(self, days: int = 30) -> None:
        """
        Remove old configuration backups
        
        Args:
            days (int): Remove backups older than this many days
        """
        backup_dir = 'config/backups'
        cutoff = datetime.now() - timedelta(days=days)
        
        for backup in os.listdir(backup_dir):
            if backup.startswith('config_backup_'):
                backup_path = os.path.join(backup_dir, backup)
                backup_time = datetime.fromtimestamp(os.path.getctime(backup_path))
                
                if backup_time < cutoff:
                    os.remove(backup_path)

    def verify_config_integrity(self) -> bool:
        """
        Verify integrity of configuration files
        
        Returns:
            bool: True if all files are valid
        """
        required_files = [
            self._config_file,
            'config/sessions.json'
        ]
        
        for file in required_files:
            if not os.path.exists(file):
                return False
            
            try:
                with open(file, 'r') as f:
                    json.load(f)
            except json.JSONDecodeError:
                return False
        
        return True