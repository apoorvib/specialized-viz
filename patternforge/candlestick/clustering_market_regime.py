# Set up results container
        performance = []
        
        # Calculate returns for each regime
        for regime in self.regimes:
            # Get regime data
            regime_data = self.df[regime.start_date:regime.end_date]
            
            if len(regime_data) < 2:
                continue
                
            # Calculate metrics
            returns = regime_data['Close'].pct_change().dropna()
            
            # Calculate forward returns for each day in regime
            forward_returns = []
            for date in regime_data.index[:-1]:  # Skip last day
                try:
                    if market_days:
                        # Get return over next N market days
                        idx = regime_data.index.get_loc(date)
                        if idx + return_horizon < len(regime_data):
                            end_idx = idx + return_horizon
                        else:
                            end_idx = len(regime_data) - 1
                            
                        end_date = regime_data.index[end_idx]
                    else:
                        # Get return over next N calendar days
                        end_date = date + pd.Timedelta(days=return_horizon)
                        if end_date > regime.end_date:
                            end_date = regime.end_date
                    
                    start_price = regime_data.loc[date, 'Close']
                    end_price = self.df.loc[end_date, 'Close']
                    fwd_return = (end_price / start_price - 1) * 100  # percentage
                    forward_returns.append(fwd_return)
                except:
                    continue
            
            # Calculate performance metrics
            perf = {
                'regime': regime.regime_type,
                'start_date': regime.start_date,
                'end_date': regime.end_date,
                'duration_days': (regime.end_date - regime.start_date).days,
                'volatility': regime.volatility,
                'trend': regime.trend,
                'volume': regime.volume,
                'daily_return_mean': returns.mean() * 100,
                'daily_return_std': returns.std() * 100,
                'daily_sharpe': returns.mean() / returns.std() if returns.std() > 0 else 0,
                'win_rate': (returns > 0).mean() * 100,
                'total_return': (regime_data['Close'].iloc[-1] / regime_data['Close'].iloc[0] - 1) * 100
            }
            
            # Add forward-looking metrics
            if forward_returns:
                perf.update({
                    f'forward_{return_horizon}d_return_mean': np.mean(forward_returns),
                    f'forward_{return_horizon}d_return_std': np.std(forward_returns),
                    f'forward_{return_horizon}d_win_rate': (np.array(forward_returns) > 0).mean() * 100
                })
                
            performance.append(perf)
        
        # Convert to DataFrame
        perf_df = pd.DataFrame(performance)
        
        # Add aggregation by regime type
        if not perf_df.empty:
            regime_groups = perf_df.groupby('regime')
            
            agg_metrics = regime_groups.agg({
                'duration_days': 'mean',
                'daily_return_mean': 'mean',
                'daily_return_std': 'mean',
                'daily_sharpe': 'mean',
                'win_rate': 'mean',
                'total_return': 'mean'
            })
            
            if f'forward_{return_horizon}d_return_mean' in perf_df.columns:
                forward_agg = regime_groups.agg({
                    f'forward_{return_horizon}d_return_mean': 'mean',
                    f'forward_{return_horizon}d_return_std': 'mean',
                    f'forward_{return_horizon}d_win_rate': 'mean'
                })
                agg_metrics = pd.concat([agg_metrics, forward_agg], axis=1)
            
            # Add to performance dataframe
            perf_df = pd.concat([perf_df, agg_metrics.reset_index().assign(type='average')])
        
        return perf_df
    
    def create_trading_signals(self,
                             regime_rules: Dict[str, str] = None) -> pd.DataFrame:
        """
        Create trading signals based on regime classification
        
        Args:
            regime_rules (Dict[str, str]): Mapping of regime type to action
            
        Returns:
            pd.DataFrame: Trading signals
        """
        if not self.regimes:
            self.logger.warning("No regimes available for signal generation")
            return pd.DataFrame()
            
        # Default rules if none provided
        if regime_rules is None:
            regime_rules = {
                'bullish_trending': 'buy',
                'bullish_volatile': 'buy',
                'bullish_quiet': 'buy',
                'recovery': 'buy',
                'bearish_trending': 'sell',
                'bearish_volatile': 'sell',
                'bearish_quiet': 'sell',
                'crash': 'sell',
                'high_volatility': 'neutral',
                'low_volatility_range': 'neutral',
                'transitional': 'neutral',
                'undefined': 'neutral'
            }
        
        # Create signals dataframe
        signals = pd.DataFrame(index=self.df.index)
        signals['regime'] = pd.Series(dtype=str)
        signals['signal'] = pd.Series(dtype=str)
        signals['confidence'] = pd.Series(dtype=float)
        
        # Fill signals based on regimes
        for regime in self.regimes:
            date_range = self.df[regime.start_date:regime.end_date].index
            signals.loc[date_range, 'regime'] = regime.regime_type
            signals.loc[date_range, 'signal'] = regime_rules.get(regime.regime_type, 'neutral')
            signals.loc[date_range, 'confidence'] = regime.confidence
        
        # Convert signals to numeric (-1, 0, 1)
        signals['position'] = 0
        signals.loc[signals['signal'] == 'buy', 'position'] = 1
        signals.loc[signals['signal'] == 'sell', 'position'] = -1
        
        # Add close price for reference
        signals['close'] = self.df['Close']
        
        return signals
    
    def export_regime_data(self, filename: str = 'regime_data.csv') -> None:
        """
        Export regime data to CSV for further analysis
        
        Args:
            filename (str): Output filename
        """
        if not self.regimes:
            self.logger.warning("No regimes available for export")
            return
            
        # Create export data
        export_data = []
        
        for regime in self.regimes:
            # Create basic record
            record = {
                'regime_type': regime.regime_type,
                'start_date': regime.start_date,
                'end_date': regime.end_date,
                'duration_days': (regime.end_date - regime.start_date).days,
                'volatility': regime.volatility,
                'trend': regime.trend,
                'volume': regime.volume,
                'confidence': regime.confidence,
                'stability': regime.stability_score,
                'expected_duration': regime.expected_duration,
                'cluster_id': regime.cluster_id
            }
            
            # Add top feature importances
            if regime.feature            
        # Calculate metrics
        results = {
            'count': len(filtered_regimes),
            'avg_duration_days': np.mean([
                (r.end_date - r.start_date).days for r in filtered_regimes
            ]),
            'avg_stability': np.mean([r.stability_score for r in filtered_regimes]),
            'top_features': self._get_top_features_for_regimes(filtered_regimes),
            'common_transitions': self._get_common_transitions(filtered_regimes)
        }
        
        # Add price performance if we have data
        if filtered_regimes:
            price_performance = []
            for regime in filtered_regimes:
                try:
                    start_price = self.df.loc[regime.start_date, 'Close']
                    end_price = self.df.loc[regime.end_date, 'Close']
                    perf = (end_price / start_price - 1) * 100
                    price_performance.append(perf)
                except:
                    continue
                    
            if price_performance:
                results['avg_price_change_pct'] = np.mean(price_performance)
                results['price_change_std'] = np.std(price_performance)
                
        return results
    
    def _get_top_features_for_regimes(self, 
                                   regimes: List[ClusteringMarketRegime], 
                                   top_n: int = 5) -> Dict[str, float]:
        """
        Get top features for a set of regimes
        
        Args:
            regimes (List[ClusteringMarketRegime]): Regimes to analyze
            top_n (int): Number of top features to return
            
        Returns:
            Dict[str, float]:import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import pearsonr
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pickle
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

@dataclass
class ClusteringConfig:
    """Configuration for clustering-based regime detection"""
    # Feature extraction parameters
    return_periods: List[int] = field(default_factory=lambda: [1, 5, 10, 20, 60])
    volatility_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 60])
    volume_periods: List[int] = field(default_factory=lambda: [5, 10, 20])
    trend_periods: List[int] = field(default_factory=lambda: [10, 20, 50, 200])
    autocorr_lags: List[int] = field(default_factory=lambda: [1, 5, 10, 20])
    
    # Clustering parameters
    n_clusters: int = 5  # Default number of regimes for k-means and GMM
    eps: float = 0.5     # DBSCAN epsilon parameter
    min_samples: int = 5  # DBSCAN min_samples parameter
    cluster_method: str = 'gmm'  # 'gmm', 'kmeans', 'dbscan', 'hierarchical'
    
    # Training parameters
    lookback_window: int = 252  # Default 1 year for training
    feature_selection: bool = True
    use_pca: bool = True
    pca_components: int = 10
    use_robust_scaling: bool = True
    
    # Regime mapping
    regime_names: Dict[int, str] = field(default_factory=lambda: {
        0: 'bullish_trending',
        1: 'bearish_trending',
        2: 'high_volatility',
        3: 'low_volatility_range',
        4: 'transition'
    })
    
    # Validation
    cross_validate: bool = True
    validation_periods: int = 5
    
    # Runtime
    parallel_processing: bool = True
    max_workers: int = 4
    cache_results: bool = True
    cache_dir: str = '.regime_cache'
    
    def __post_init__(self):
        """Validate configuration parameters"""
        # Ensure cache directory exists
        if self.cache_results and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            
        # Validate clustering method
        valid_methods = ['gmm', 'kmeans', 'dbscan', 'hierarchical']
        if self.cluster_method not in valid_methods:
            raise ValueError(f"Invalid clustering method '{self.cluster_method}'. "
                            f"Must be one of {valid_methods}")
        
        # Ensure proper number of regime names if using fixed clusters
        if self.cluster_method in ['kmeans', 'gmm'] and len(self.regime_names) != self.n_clusters:
            warnings.warn(f"Number of regime names ({len(self.regime_names)}) doesn't match "
                         f"n_clusters ({self.n_clusters}). Regime mapping may be inconsistent.")

@dataclass
class ClusteringMarketRegime:
    """Extended market regime class with clustering-specific information"""
    regime_type: str
    volatility: str
    trend: str
    volume: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    confidence: float
    
    # Clustering-specific attributes
    cluster_id: int
    feature_vector: np.ndarray = None
    probability: Dict[int, float] = None
    distance_to_centroid: float = None
    silhouette_score: float = None
    
    # Transition probabilities
    transition_probs: Dict[str, float] = None
    
    # Stability metrics
    stability_score: float = None
    expected_duration: int = None
    
    # Feature importance for this regime
    feature_importance: Dict[str, float] = None
    
    def __post_init__(self):
        """Convert types and initialize empty containers"""
        if self.probability is None:
            self.probability = {}
        if self.transition_probs is None:
            self.transition_probs = {}
        if self.feature_importance is None:
            self.feature_importance = {}
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        result = {
            'regime_type': self.regime_type,
            'volatility': self.volatility,
            'trend': self.trend,
            'volume': self.volume,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'confidence': self.confidence,
            'cluster_id': self.cluster_id,
            'probability': self.probability,
            'distance_to_centroid': self.distance_to_centroid,
            'silhouette_score': self.silhouette_score,
            'stability_score': self.stability_score,
            'expected_duration': self.expected_duration,
            'transition_probs': self.transition_probs,
            'feature_importance': self.feature_importance
        }
        
        # Handle conversion of numpy arrays
        if self.feature_vector is not None:
            result['feature_vector'] = self.feature_vector.tolist()
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClusteringMarketRegime':
        """Create instance from dictionary"""
        # Convert feature_vector back to numpy array if present
        if 'feature_vector' in data and data['feature_vector'] is not None:
            data['feature_vector'] = np.array(data['feature_vector'])
            
        return cls(**data)

class ClusteringMarketRegimeAnalyzer:
    """
    Market regime analyzer using clustering techniques
    
    This class implements advanced market regime detection using 
    unsupervised learning methods to identify natural market states
    based on multiple features.
    """
    
    def __init__(self, 
                 df: pd.DataFrame, 
                 config: Optional[ClusteringConfig] = None):
        """
        Initialize clustering market regime analyzer
        
        Args:
            df (pd.DataFrame): OHLCV data
            config (Optional[ClusteringConfig]): Configuration parameters
        """
        # Setup logger
        self.logger = logging.getLogger('ClusteringMarketRegimeAnalyzer')
        
        self._validate_dataframe(df)
        self.df = df.copy()
        self.config = config or ClusteringConfig()
        
        # Initialize containers
        self.features = pd.DataFrame(index=df.index)
        self.feature_importances = {}
        self.cluster_labels = pd.Series(index=df.index, dtype=int)
        self.cluster_probs = pd.DataFrame(index=df.index)
        self.regimes = []
        self.transition_matrix = pd.DataFrame()
        
        # Initialize transformers
        self.scaler = None
        self.pca = None
        self.feature_scaler = None
        
        # Initialize clustering model
        self.cluster_model = None
        
        # Setup cache
        self._init_cache()
    
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
            try:
                # Try to convert index to datetime
                df.index = pd.to_datetime(df.index)
            except:
                raise ValueError("DataFrame index must be DatetimeIndex")
    
    def _init_cache(self) -> None:
        """Initialize caching system"""
        if self.config.cache_results:
            self.cache_dir = os.path.join(
                self.config.cache_dir, 
                f"cache_{self.config.cluster_method}"
            )
            
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)
    
    def detect_regimes(self, 
                      start_date: Optional[pd.Timestamp] = None,
                      end_date: Optional[pd.Timestamp] = None,
                      force_recompute: bool = False) -> List[ClusteringMarketRegime]:
        """
        Detect market regimes using clustering approach
        
        Args:
            start_date (Optional[pd.Timestamp]): Analysis start date
            end_date (Optional[pd.Timestamp]): Analysis end date
            force_recompute (bool): Recompute even if cached results exist
            
        Returns:
            List[ClusteringMarketRegime]: Detected market regimes
            
        This is the main entry point for regime detection.
        """
        # Set date range
        start_date = start_date or self.df.index[0]
        end_date = end_date or self.df.index[-1]
        
        # Check cache first
        cache_key = self._make_cache_key(start_date, end_date)
        if not force_recompute and self._load_from_cache(cache_key):
            self.logger.info(f"Loaded regimes from cache for {start_date} to {end_date}")
            return self.regimes
        
        self.logger.info(f"Detecting regimes from {start_date} to {end_date}")
        
        # Extract and preprocess features
        self._preprocess_features(start_date, end_date)
        
        # Perform clustering
        self._perform_clustering()
        
        # Map clusters to regimes
        self._map_clusters_to_regimes()
        
        # Analyze transitions 
        self._analyze_regime_transitions()
        
        # Calculate regime probabilities and stability
        self._calculate_regime_probabilities()
        
        # Cache results
        if self.config.cache_results:
            self._save_to_cache(cache_key)
        
        self.logger.info(f"Detected {len(self.regimes)} distinct regimes")
        return self.regimes
    
    def _preprocess_features(self, 
                            start_date: pd.Timestamp,
                            end_date: pd.Timestamp) -> None:
        """
        Extract and preprocess features for regime detection
        
        Args:
            start_date (pd.Timestamp): Start date for feature extraction
            end_date (pd.Timestamp): End date for feature extraction
        """
        self.logger.info("Extracting features from price data")
        
        # Get data slice allowing for lookback
        lookback_start = start_date - pd.Timedelta(days=self.config.lookback_window)
        data_slice = self.df.loc[lookback_start:end_date].copy()
        
        # Extract features
        features = pd.DataFrame(index=data_slice.index)
        
        # 1. Return-based features
        for period in self.config.return_periods:
            # Returns
            features[f'returns_{period}d'] = data_slice['Close'].pct_change(period)
            
            # Return volatility (standard deviation)
            features[f'return_vol_{period}d'] = features[f'returns_{period}d'].rolling(period).std()
            
            # Return skewness
            features[f'return_skew_{period}d'] = features[f'returns_{period}d'].rolling(period*2).skew()
            
            # Return kurtosis
            features[f'return_kurt_{period}d'] = features[f'returns_{period}d'].rolling(period*2).kurt()
        
        # 2. Volatility features
        for period in self.config.volatility_periods:
            # Volatility (standard deviation)
            returns = data_slice['Close'].pct_change()
            features[f'volatility_{period}d'] = returns.rolling(period).std() * np.sqrt(252)
            
            # Volatility of volatility
            features[f'vol_of_vol_{period}d'] = features[f'volatility_{period}d'].rolling(period).std()
            
            # Parkinson volatility estimator (uses High/Low)
            high_low_ratio = np.log(data_slice['High'] / data_slice['Low'])
            features[f'parkinsons_vol_{period}d'] = (high_low_ratio ** 2).rolling(period).mean() * 0.361 * np.sqrt(252)
            
            # True range and ATR
            tr1 = data_slice['High'] - data_slice['Low']
            tr2 = abs(data_slice['High'] - data_slice['Close'].shift())
            tr3 = abs(data_slice['Low'] - data_slice['Close'].shift())
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            features[f'atr_{period}d'] = true_range.rolling(period).mean() / data_slice['Close']
        
        # 3. Trend features
        for period in self.config.trend_periods:
            # Simple moving averages
            sma = data_slice['Close'].rolling(period).mean()
            features[f'close_sma_ratio_{period}d'] = data_slice['Close'] / sma - 1
            
            # Exponential moving averages
            ema = data_slice['Close'].ewm(span=period, adjust=False).mean()
            features[f'close_ema_ratio_{period}d'] = data_slice['Close'] / ema - 1
            
            # MA Slope
            features[f'sma_slope_{period}d'] = sma.diff(5) / sma.shift(5)
            
            # Price distance from SMA in volatility units
            price_dist = (data_slice['Close'] - sma) / (returns.rolling(period).std() * np.sqrt(period))
            features[f'price_distance_{period}d'] = price_dist
            
            # RSI
            delta = data_slice['Close'].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(period).mean()
            avg_loss = loss.rolling(period).mean()
            
            # Handle division by zero
            rs = avg_gain / avg_loss.replace(0, np.nan)
            rs = rs.fillna(100)  # RSI of 100 where avg_loss is zero
            
            features[f'rsi_{period}d'] = 100 - (100 / (1 + rs))
            
            # ROC (Rate of Change)
            features[f'roc_{period}d'] = data_slice['Close'].pct_change(period) * 100
        
        # 4. Volume features (if available)
        if 'Volume' in data_slice.columns:
            for period in self.config.volume_periods:
                # Volume changes
                features[f'volume_change_{period}d'] = data_slice['Volume'].pct_change(period)
                
                # Relative volume
                vol_sma = data_slice['Volume'].rolling(period).mean()
                features[f'relative_volume_{period}d'] = data_slice['Volume'] / vol_sma
                
                # Price-volume correlation
                def compute_pv_corr(x):
                    if len(x) < 3:  # Need at least 3 points for meaningful correlation
                        return np.nan
                    price_data = x['Close']
                    volume_data = x['Volume']
                    try:
                        return pearsonr(price_data, volume_data)[0]
                    except:
                        return np.nan
                
                # Rolling price-volume correlation
                pv_corr = data_slice[['Close', 'Volume']].rolling(period).apply(
                    compute_pv_corr, raw=False
                )
                features[f'price_volume_corr_{period}d'] = pv_corr
                
                # On-balance volume momentum
                obv = (data_slice['Close'].diff() > 0).astype(int) * data_slice['Volume']
                obv = obv.cumsum()
                features[f'obv_momentum_{period}d'] = obv.diff(period) / obv.shift(period)
        
        # 5. Autocorrelation features
        for lag in self.config.autocorr_lags:
            # Return autocorrelation
            returns = data_slice['Close'].pct_change()
            features[f'return_autocorr_{lag}d'] = returns.rolling(lag*3).apply(
                lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan
            )
        
        # Calculate derivative features
        features['volatility_trend_20d'] = features['volatility_20d'].diff(5) / features['volatility_20d'].shift(5)
        features['trend_strength'] = abs(features['sma_slope_50d']) / features['volatility_20d']
        features['mean_reversion'] = features['return_autocorr_5d'] < -0.2
        
        # Remove NaN values
        self.features = features.dropna(how='all')
        
        # Filter to analysis period
        self.features = self.features.loc[start_date:end_date]
        
        self.logger.info(f"Extracted {self.features.shape[1]} features for regime analysis")
        
        # Feature preprocessing
        self._preprocess_feature_matrix()
    
    def _preprocess_feature_matrix(self) -> None:
        """Preprocess feature matrix for clustering"""
        # Handle missing values
        self.features = self.features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Select robust feature scaling if configured
        if self.config.use_robust_scaling:
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
            
        # Scale features
        scaled_features = self.scaler.fit_transform(self.features)
        
        # Apply PCA if configured
        if self.config.use_pca:
            n_components = min(self.config.pca_components, scaled_features.shape[1])
            self.pca = PCA(n_components=n_components)
            scaled_features = self.pca.fit_transform(scaled_features)
            
            # Store feature importances
            if self.pca is not None:
                feature_names = self.features.columns
                for i, component in enumerate(self.pca.components_):
                    for j, feature in enumerate(feature_names):
                        importance = abs(component[j])
                        if feature not in self.feature_importances:
                            self.feature_importances[feature] = 0
                        # Weight by explained variance
                        self.feature_importances[feature] += importance * self.pca.explained_variance_ratio_[i]
        
            # Convert back to DataFrame
            self.features_processed = pd.DataFrame(
                scaled_features, 
                index=self.features.index,
                columns=[f'PC{i+1}' for i in range(scaled_features.shape[1])]
            )
        else:
            # Just use scaled features
            self.features_processed = pd.DataFrame(
                scaled_features,
                index=self.features.index,
                columns=self.features.columns
            )
            
            # Simple feature importance based on variance
            for feature in self.features.columns:
                self.feature_importances[feature] = self.features[feature].var()
        
        # Normalize feature importances
        total_importance = sum(self.feature_importances.values())
        for feature in self.feature_importances:
            self.feature_importances[feature] /= total_importance
    
    def _perform_clustering(self) -> None:
        """Perform clustering on preprocessed features"""
        self.logger.info(f"Performing clustering using {self.config.cluster_method}")
        
        # Get feature matrix
        X = self.features_processed.values
        
        # Select and apply clustering algorithm
        if self.config.cluster_method == 'kmeans':
            self.cluster_model = KMeans(
                n_clusters=self.config.n_clusters,
                random_state=42,
                n_init=10
            )
            self.cluster_labels = pd.Series(
                self.cluster_model.fit_predict(X),
                index=self.features_processed.index
            )
            
            # Get probabilities based on distance to centroids
            distances = self._calculate_kmeans_probabilities(X)
            self.cluster_probs = pd.DataFrame(
                distances,
                index=self.features_processed.index,
                columns=[f'Cluster_{i}' for i in range(self.config.n_clusters)]
            )
            
        elif self.config.cluster_method == 'gmm':
            self.cluster_model = GaussianMixture(
                n_components=self.config.n_clusters,
                random_state=42,
                n_init=10
            )
            self.cluster_model.fit(X)
            self.cluster_labels = pd.Series(
                self.cluster_model.predict(X),
                index=self.features_processed.index
            )
            
            # Get probabilities
            probs = self.cluster_model.predict_proba(X)
            self.cluster_probs = pd.DataFrame(
                probs,
                index=self.features_processed.index,
                columns=[f'Cluster_{i}' for i in range(self.config.n_clusters)]
            )
            
        elif self.config.cluster_method == 'dbscan':
            self.cluster_model = DBSCAN(
                eps=self.config.eps,
                min_samples=self.config.min_samples
            )
            self.cluster_labels = pd.Series(
                self.cluster_model.fit_predict(X),
                index=self.features_processed.index
            )
            
            # Create rough probability estimate for DBSCAN
            self._calculate_dbscan_probabilities(X)
            
        elif self.config.cluster_method == 'hierarchical':
            # Perform hierarchical clustering
            Z = linkage(X, method='ward')
            self.cluster_labels = pd.Series(
                fcluster(Z, t=self.config.n_clusters, criterion='maxclust') - 1,  # Zero-based indexing
                index=self.features_processed.index
            )
            
            # Calculate approximate probabilities based on distance
            self._calculate_hierarchical_probabilities(X, Z)
            
        # Calculate cluster quality metrics
        self._calculate_cluster_quality(X)
        
        self.logger.info(f"Identified {self.cluster_labels.nunique()} distinct clusters")
    
    def _calculate_kmeans_probabilities(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate probability-like scores for KMeans clusters
        
        Args:
            X (np.ndarray): Feature matrix
            
        Returns:
            np.ndarray: Pseudo-probability matrix
        """
        # Calculate distance to each centroid
        distances = np.zeros((X.shape[0], self.config.n_clusters))
        for i in range(self.config.n_clusters):
            centroid = self.cluster_model.cluster_centers_[i]
            distances[:, i] = np.linalg.norm(X - centroid, axis=1)
        
        # Convert distances to similarity scores
        similarities = 1 / (1 + distances)
        
        # Normalize to sum to 1 (like probabilities)
        probabilities = similarities / similarities.sum(axis=1, keepdims=True)
        return probabilities
    
    def _calculate_dbscan_probabilities(self, X: np.ndarray) -> None:
        """
        Calculate probability-like scores for DBSCAN clusters
        
        Args:
            X (np.ndarray): Feature matrix
        """
        # Count unique clusters
        unique_clusters = np.unique(self.cluster_labels)
        # Remove noise points from calculation (-1)
        valid_clusters = unique_clusters[unique_clusters >= 0]
        n_clusters = len(valid_clusters)
        
        # Initialize probability matrix
        probs = np.zeros((X.shape[0], n_clusters))
        
        # Calculate cluster centers
        cluster_centers = np.zeros((n_clusters, X.shape[1]))
        for i, cluster_id in enumerate(valid_clusters):
            mask = self.cluster_labels == cluster_id
            cluster_centers[i] = X[mask].mean(axis=0)
        
        # Calculate distances and convert to probabilities
        for i in range(X.shape[0]):
            point = X[i]
            distances = np.array([np.linalg.norm(point - center) for center in cluster_centers])
            if self.cluster_labels.iloc[i] == -1:  # Noise point
                probs[i] = 0  # No strong association with any cluster
            else:
                # Calculate similarity scores
                similarities = 1 / (1 + distances)
                # Normalize
                probs[i] = similarities / similarities.sum()
        
        # Convert to DataFrame
        self.cluster_probs = pd.DataFrame(
            probs,
            index=self.features_processed.index,
            columns=[f'Cluster_{int(i)}' for i in valid_clusters]
        )
    
    def _calculate_hierarchical_probabilities(self, X: np.ndarray, Z: np.ndarray) -> None:
        """
        Calculate probability-like scores for hierarchical clusters
        
        Args:
            X (np.ndarray): Feature matrix
            Z (np.ndarray): Linkage matrix
        """
        # Count unique clusters
        unique_clusters = np.unique(self.cluster_labels)
        n_clusters = len(unique_clusters)
        
        # Initialize probability matrix
        probs = np.zeros((X.shape[0], n_clusters))
        
        # Calculate cluster centers
        cluster_centers = np.zeros((n_clusters, X.shape[1]))
        for i, cluster_id in enumerate(unique_clusters):
            mask = self.cluster_labels == cluster_id
            cluster_centers[i] = X[mask].mean(axis=0)
        
        # Calculate distances and convert to probabilities
        for i in range(X.shape[0]):
            point = X[i]
            distances = np.array([np.linalg.norm(point - center) for center in cluster_centers])
            # Calculate similarity scores
            similarities = 1 / (1 + distances)
            # Normalize
            probs[i] = similarities / similarities.sum()
        
        # Convert to DataFrame
        self.cluster_probs = pd.DataFrame(
            probs,
            index=self.features_processed.index,
            columns=[f'Cluster_{i}' for i in unique_clusters]
        )
    
    def _calculate_cluster_quality(self, X: np.ndarray) -> None:
        """
        Calculate cluster quality metrics
        
        Args:
            X (np.ndarray): Feature matrix
        """
        try:
            # Silhouette score (only for more than one cluster)
            if len(np.unique(self.cluster_labels)) > 1 and -1 not in self.cluster_labels.values:
                self.silhouette_avg = silhouette_score(X, self.cluster_labels)
                self.logger.info(f"Silhouette Score: {self.silhouette_avg:.4f}")
            else:
                self.silhouette_avg = None
                
            # Calinski-Harabasz Index (if more than one cluster and no noise points)
            if len(np.unique(self.cluster_labels)) > 1 and -1 not in self.cluster_labels.values:
                self.ch_score = calinski_harabasz_score(X, self.cluster_labels)
                self.logger.info(f"Calinski-Harabasz Score: {self.ch_score:.4f}")
            else:
                self.ch_score = None
                
        except Exception as e:
            self.logger.warning(f"Error calculating cluster quality: {str(e)}")
            self.silhouette_avg = None
            self.ch_score = None
    
    def _map_clusters_to_regimes(self) -> None:
        """Map identified clusters to market regimes"""
        self.logger.info("Mapping clusters to market regimes")
        
        # Get OHLCV data corresponding to analysis period
        df_subset = self.df.loc[self.cluster_labels.index]
        
        # Calculate basic price statistics for each cluster
        cluster_stats = {}
        for cluster_id in sorted(self.cluster_labels.unique()):
            if cluster_id == -1:  # Skip noise points in DBSCAN
                continue
                
            # Get rows belonging to this cluster
            mask = self.cluster_labels == cluster_id
            cluster_df = df_subset[mask]
            
            if len(cluster_df) == 0:
                continue
                
            # Calculate statistics
            returns = cluster_df['Close'].pct_change().dropna()
            
            stats = {
                'count': len(cluster_df),
                'avg_return': returns.mean(),
                'return_std': returns.std(),
                'return_skew': returns.skew() if len(returns) > 2 else 0,
                'sharpe': returns.mean() / returns.std() if returns.std() > 0 else 0,
                'volatility': returns.std() * np.sqrt(252),
                'max_drawdown': self._calculate_max_drawdown(cluster_df['Close']),
                'win_rate': (returns > 0).mean(),
                'avg_volume': cluster_df['Volume'].mean() if 'Volume' in cluster_df else None,
                'volume_trend': (
                    cluster_df['Volume'].iloc[-1] / cluster_df['Volume'].iloc[0] - 1
                    if 'Volume' in cluster_df and len(cluster_df) > 1 else None
                ),
                'price_trend': cluster_df['Close'].iloc[-1] / cluster_df['Close'].iloc[0] - 1,
                'trend_strength': abs(pearsonr(
                    np.arange(len(cluster_df)), 
                    cluster_df['Close'].values
                )[0]) if len(cluster_df) > 2 else 0,
                'mean_features': self.features.loc[mask].mean()
            }