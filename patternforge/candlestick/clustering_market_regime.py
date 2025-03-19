import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans, HDBSCAN
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
    cluster_method: str = 'gmm'  # 'gmm', 'kmeans', 'dbscan', 'hierarchical', 'hdbscan', 'som'
    
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
        valid_methods = ['gmm', 'kmeans', 'dbscan', 'hierarchical', 'hdbscan', 'som']
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
        
        elif self.config.cluster_method == 'hdbscan':
            # Use HDBSCAN for improved density-based clustering
            try:
                import hdbscan
                self.cluster_model = hdbscan.HDBSCAN(
                    min_cluster_size=self.config.min_samples,
                    min_samples=self.config.min_samples // 2,  # Default to half min_samples
                    prediction_data=True
                )
                self.cluster_labels = pd.Series(
                    self.cluster_model.fit_predict(X),
                    index=self.features_processed.index
                )
                
                # Calculate probabilities using HDBSCAN's probabilities method
                self._calculate_hdbscan_probabilities(X)
            except ImportError:
                self.logger.warning("HDBSCAN package not found, falling back to DBSCAN")
                # Fall back to DBSCAN
                self.config.cluster_method = 'dbscan'
                self._perform_clustering()
                return
        
        elif self.config.cluster_method == 'som':
            # Use Self-Organizing Maps
            try:
                from minisom import MiniSom
                
                # Define SOM grid size based on data
                som_x = int(np.sqrt(self.config.n_clusters) * 2)
                som_y = int(np.sqrt(self.config.n_clusters) * 2)
                
                # Initialize and train SOM
                self.cluster_model = MiniSom(
                    som_x, som_y, X.shape[1], 
                    sigma=1.0, learning_rate=0.5,
                    random_seed=42
                )
                self.cluster_model.train_random(X, 5000)
                
                # Map data points to SOM grid
                som_clusters = np.zeros(X.shape[0], dtype=int)
                for i, x in enumerate(X):
                    som_clusters[i] = np.ravel_multi_index(
                        self.cluster_model.winner(x), 
                        (som_x, som_y)
                    )
                
                # Reduce number of clusters using hierarchical clustering if needed
                if som_x * som_y > self.config.n_clusters:
                    # Get unique winning nodes and their coordinates
                    unique_winners = np.unique(som_clusters)
                    winner_coords = np.array([
                        np.unravel_index(w, (som_x, som_y)) 
                        for w in unique_winners
                    ])
                    
                    # Perform hierarchical clustering on SOM grid
                    if len(winner_coords) > 1:
                        Z = linkage(winner_coords, method='ward')
                        cluster_map = fcluster(
                            Z, t=self.config.n_clusters, criterion='maxclust'
                        ) - 1  # Zero-based indexing
                        
                        # Map SOM nodes to final clusters
                        cluster_lookup = {
                            unique_winners[i]: cluster_map[i] 
                            for i in range(len(unique_winners))
                        }
                        final_clusters = np.array([
                            cluster_lookup.get(c, -1) for c in som_clusters
                        ])
                    else:
                        # Only one unique cluster found
                        final_clusters = np.zeros(X.shape[0], dtype=int)
                else:
                    final_clusters = som_clusters
                    
                self.cluster_labels = pd.Series(
                    final_clusters,
                    index=self.features_processed.index
                )
                
                # Calculate SOM-based probabilities
                self._calculate_som_probabilities(X)
            except ImportError:
                self.logger.warning("MiniSom package not found, falling back to KMeans")
                # Fall back to KMeans
                self.config.cluster_method = 'kmeans'
                self._perform_clustering()
                return
            
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

    def _calculate_hdbscan_probabilities(self, X: np.ndarray) -> None:
        """
        Calculate probability scores for HDBSCAN clusters
        
        Args:
            X (np.ndarray): Feature matrix
        """
        # Get unique valid clusters (excluding noise)
        unique_clusters = np.unique(self.cluster_labels)
        valid_clusters = unique_clusters[unique_clusters >= 0]
        n_clusters = len(valid_clusters)
        
        # Initialize probability matrix
        probs = np.zeros((X.shape[0], n_clusters))
        
        # Use HDBSCAN's probabilities if available
        if hasattr(self.cluster_model, 'probabilities_'):
            # For points assigned to a cluster, use the membership probability
            for i, (cluster, prob) in enumerate(zip(self.cluster_labels, self.cluster_model.probabilities_)):
                if cluster >= 0:  # Not noise
                    cluster_idx = np.where(valid_clusters == cluster)[0][0]
                    probs[i, cluster_idx] = prob
        else:
            # Fall back to distance-based approach
            # Calculate cluster centers
            cluster_centers = np.zeros((n_clusters, X.shape[1]))
            for i, cluster_id in enumerate(valid_clusters):
                mask = self.cluster_labels == cluster_id
                cluster_centers[i] = X[mask].mean(axis=0)
            
            # Calculate distances and convert to probabilities
            for i in range(X.shape[0]):
                if self.cluster_labels.iloc[i] >= 0:  # Not noise
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
            columns=[f'Cluster_{int(i)}' for i in valid_clusters]
        )

    def _calculate_som_probabilities(self, X: np.ndarray) -> None:
        """
        Calculate probability-like scores for SOM clusters
        
        Args:
            X (np.ndarray): Feature matrix
        """
        # Get unique valid clusters
        unique_clusters = np.unique(self.cluster_labels)
        n_clusters = len(unique_clusters)
        
        # Initialize probability matrix
        probs = np.zeros((X.shape[0], n_clusters))
        
        # Calculate cluster centers
        cluster_centers = np.zeros((n_clusters, X.shape[1]))
        for i, cluster_id in enumerate(unique_clusters):
            mask = self.cluster_labels == cluster_id
            cluster_centers[i] = X[mask].mean(axis=0)
        
        # Calculate quantization errors for each point to each cluster center
        for i in range(X.shape[0]):
            point = X[i]
            # Use negative quantization error (distance) as activation
            distances = np.array([np.linalg.norm(point - center) for center in cluster_centers])
            # Convert to soft assignments (probabilities)
            # Using softmax-like formula
            exp_neg_dist = np.exp(-distances)
            probs[i] = exp_neg_dist / exp_neg_dist.sum()
        
        # Convert to DataFrame
        self.cluster_probs = pd.DataFrame(
            probs,
            index=self.features_processed.index,
            columns=[f'Cluster_{i}' for i in unique_clusters]
        )

    def _get_common_transitions(self, regimes: List[ClusteringMarketRegime]) -> Dict[str, float]:
        """
        Get common regime transitions for a set of regimes
        
        Args:
            regimes (List[ClusteringMarketRegime]): Regimes to analyze
            
        Returns:
            Dict[str, float]: Common transitions and their probabilities
        """
        # Aggregate transition probabilities across all regimes
        transition_counts = {}
        
        for regime in regimes:
            if not regime.transition_probs:
                continue
                
            for to_regime, prob in regime.transition_probs.items():
                key = f"{regime.regime_type} -> {to_regime}"
                if key not in transition_counts:
                    transition_counts[key] = []
                    
                transition_counts[key].append(prob)
        
        # Calculate average probabilities
        avg_transitions = {}
        for transition, probs in transition_counts.items():
            avg_transitions[transition] = np.mean(probs)
        
        # Return sorted by probability
        return dict(sorted(
            avg_transitions.items(),
            key=lambda item: item[1],
            reverse=True
        ))              
        
    def _make_cache_key(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> str:
        """Create cache key based on data range and config
        
        Args:
            start_date (pd.Timestamp): Start date for analysis
            end_date (pd.Timestamp): End date for analysis
            
        Returns:
            str: Cache key string
        """
        # Format dates
        start_str = start_date.strftime('%Y%m%d')
        end_str = end_date.strftime('%Y%m%d')
        
        # Include key config parameters in cache key
        method = self.config.cluster_method
        n_clusters = self.config.n_clusters
        use_pca = 'pca' if self.config.use_pca else 'nopca'
        
        # Create hash of key parameters
        params_hash = hash((
            method, 
            n_clusters, 
            use_pca, 
            self.config.lookback_window,
            tuple(self.config.return_periods),
            tuple(self.config.volatility_periods),
            tuple(self.config.trend_periods)
        )) % 10000  # Keep hash short
        
        return f"regimes_{start_str}_{end_str}_{method}_{n_clusters}_{use_pca}_{params_hash}"  

    def _load_from_cache(self, cache_key: str) -> bool:
        """Load results from cache
        
        Args:
            cache_key (str): Cache key for results
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        if not self.config.cache_results:
            return False
            
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        if not os.path.exists(cache_path):
            return False
            
        try:
            # Load from pickle file
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
                
            # Restore regimes
            self.regimes = [
                ClusteringMarketRegime.from_dict(regime_dict)
                for regime_dict in cache_data['regimes']
            ]
            
            # Restore transition matrix
            if 'transition_matrix' in cache_data and cache_data['transition_matrix']:
                self.transition_matrix = pd.DataFrame.from_dict(cache_data['transition_matrix'])
            
            # Restore cluster labels
            if 'cluster_labels' in cache_data:
                self.cluster_labels = pd.Series(cache_data['cluster_labels'])
                
            self.logger.info(f"Loaded results from cache {cache_path}")
            return True
        except Exception as e:
            self.logger.warning(f"Error loading from cache: {str(e)}")
            return False
        
    def _save_to_cache(self, cache_key: str) -> None:
        """Save results to cache
        
        Args:
            cache_key (str): Cache key for results
        """
        if not self.config.cache_results:
            return
            
        try:
            # Create cache dir if it doesn't exist
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)
                
            # Convert regimes to dictionaries
            regimes_dict = [regime.to_dict() for regime in self.regimes]
            
            # Create data to save
            cache_data = {
                'regimes': regimes_dict,
                'transition_matrix': self.transition_matrix.to_dict() if not self.transition_matrix.empty else {},
                'cluster_labels': self.cluster_labels.to_dict(),
                'config': {
                    'method': self.config.cluster_method,
                    'n_clusters': self.config.n_clusters,
                    'use_pca': self.config.use_pca,
                    'feature_selection': self.config.feature_selection
                },
                'timestamp': datetime.now().isoformat()
            }
            
            # Save to pickle file
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
                
            self.logger.info(f"Cached results to {cache_path}")
        except Exception as e:
            self.logger.warning(f"Error saving to cache: {str(e)}")
        
    def find_similar_historical_regimes(self, 
                                    current_features: np.ndarray,
                                    n_matches: int = 3) -> List[ClusteringMarketRegime]:
        """
        Find historical regimes that are most similar to current market conditions
        
        Args:
            current_features (np.ndarray): Current market feature vector
            n_matches (int): Number of matches to return
            
        Returns:
            List[ClusteringMarketRegime]: Most similar historical regimes
        """
        if not self.regimes:
            return []
            
        # Scale current features using the same scaler
        if self.scaler is not None:
            current_features = self.scaler.transform(current_features.reshape(1, -1))[0]
        
        # Apply PCA if used
        if self.pca is not None:
            current_features = self.pca.transform(current_features.reshape(1, -1))[0]
        
        # Calculate distances to each regime's feature vector
        regime_distances = []
        
        for regime in self.regimes:
            if regime.feature_vector is None:
                continue
                
            # Calculate Euclidean distance
            distance = np.linalg.norm(current_features - regime.feature_vector)
            regime_distances.append((regime, distance))
        
        # Sort by distance (most similar first)
        regime_distances.sort(key=lambda x: x[1])
        
        # Return top N matches
        return [regime for regime, _ in regime_distances[:n_matches]]

    def predict_next_regime(self,
                        current_regime: str,
                        n_steps: int = 1) -> List[Tuple[str, float]]:
        """
        Predict the most likely next regime(s) using transition probabilities
        
        Args:
            current_regime (str): Current regime type
            n_steps (int): Number of steps to predict ahead
            
        Returns:
            List[Tuple[str, float]]: Predicted regimes and their probabilities
        """
        if not self.regimes or not self.transition_matrix.shape[0] > 0:
            return []
            
        # Get current regime row from transition matrix
        if current_regime not in self.transition_matrix.index:
            return []
            
        # For single step prediction
        if n_steps == 1:
            transitions = self.transition_matrix.loc[current_regime]
            # Sort by probability
            sorted_transitions = sorted(
                [(regime, prob) for regime, prob in transitions.items() if prob > 0],
                key=lambda x: x[1],
                reverse=True
            )
            return sorted_transitions
        
        # For multi-step prediction using Markov chain
        else:
            # Calculate n-step transition matrix (matrix power)
            transition_matrix_n = np.linalg.matrix_power(
                self.transition_matrix.values,
                n_steps
            )
            
            # Get row for current regime
            current_idx = list(self.transition_matrix.index).index(current_regime)
            probs = transition_matrix_n[current_idx]
            
            # Create predictions
            predictions = [
                (regime, probs[i])
                for i, regime in enumerate(self.transition_matrix.columns)
                if probs[i] > 0
            ]
            
            # Sort by probability
            predictions.sort(key=lambda x: x[1], reverse=True)
            return predictions

    def plot_regime_transitions(self, 
                            filename: Optional[str] = None,
                            min_prob: float = 0.05) -> None:
        """
        Plot regime transition network graph
        
        Args:
            filename (Optional[str]): Output filename, if None displays plot
            min_prob (float): Minimum transition probability to include
        """
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
            
            if not self.regimes or self.transition_matrix.empty:
                self.logger.warning("No regimes or transitions to plot")
                return
                
            # Create directed graph
            G = nx.DiGraph()
            
            # Add nodes (regimes)
            unique_regimes = set(r.regime_type for r in self.regimes)
            for regime in unique_regimes:
                G.add_node(regime)
            
            # Add edges (transitions)
            for i, from_regime in enumerate(self.transition_matrix.index):
                for j, to_regime in enumerate(self.transition_matrix.columns):
                    prob = self.transition_matrix.iloc[i, j]
                    if prob >= min_prob:
                        G.add_edge(from_regime, to_regime, weight=prob, label=f"{prob:.2f}")
            
            # Set up plot
            plt.figure(figsize=(12, 8))
            
            # Define node colors based on regime characteristics
            node_colors = []
            for regime in G.nodes():
                if 'bullish' in regime:
                    node_colors.append('green')
                elif 'bearish' in regime:
                    node_colors.append('red')
                elif 'volatile' in regime:
                    node_colors.append('orange')
                else:
                    node_colors.append('blue')
            
            # Layout
            pos = nx.spring_layout(G, k=0.5, iterations=50)
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_size=700, node_color=node_colors, alpha=0.8)
            
            # Draw edges with varying thickness based on probability
            edge_weights = [G[u][v]['weight'] * 5 for u, v in G.edges()]
            nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.7, 
                                edge_color='gray', arrowsize=20)
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, font_size=10)
            
            # Draw edge labels (probabilities)
            edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
            
            plt.title("Market Regime Transition Network")
            plt.axis('off')
            
            # Save or display
            if filename:
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                self.logger.info(f"Saved transition network plot to {filename}")
            else:
                plt.show()
                
        except ImportError:
            self.logger.warning("Could not plot transitions. NetworkX and matplotlib required.")

    def plot_regime_performance(self, 
                            metrics: List[str] = None,
                            filename: Optional[str] = None) -> None:
        """
        Plot performance metrics across different regime types
        
        Args:
            metrics (List[str]): List of metrics to plot
            filename (Optional[str]): Output filename, if None displays plot
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Default metrics if none provided
            if metrics is None:
                metrics = ['daily_return_mean', 'daily_return_std', 'win_rate', 'total_return']
                
            # Get performance stats for all regimes
            perf_df = self.analyze_regime_stats()
            
            if perf_df.empty:
                self.logger.warning("No regime performance data to plot")
                return
                
            # Filter to regime aggregates
            agg_df = perf_df[perf_df['type'] == 'average'].drop(columns=['type']).set_index('regime')
            
            if agg_df.empty:
                self.logger.warning("No aggregated regime data to plot")
                return
                
            # Check if requested metrics exist
            available_metrics = [m for m in metrics if m in agg_df.columns]
            if not available_metrics:
                self.logger.warning(f"None of the requested metrics {metrics} are available")
                return
                
            # Set up plot
            fig, axes = plt.subplots(len(available_metrics), 1, figsize=(10, 3*len(available_metrics)))
            
            # Handle single metric case
            if len(available_metrics) == 1:
                axes = [axes]
                
            # Plot each metric
            for i, metric in enumerate(available_metrics):
                ax = axes[i]
                metric_data = agg_df[metric].sort_values(ascending=False)
                
                # Create bar colors based on regime type
                colors = []
                for regime in metric_data.index:
                    if 'bullish' in regime:
                        colors.append('green')
                    elif 'bearish' in regime:
                        colors.append('red')
                    elif 'volatil' in regime:
                        colors.append('orange')
                    else:
                        colors.append('blue')
                        
                # Plot bars
                sns.barplot(x=metric_data.index, y=metric_data.values, palette=colors, ax=ax)
                
                # Format labels
                ax.set_title(f"{metric.replace('_', ' ').title()}")
                ax.set_xlabel('')
                
                # Rotate x-axis labels
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                
                # Add value labels on bars
                for j, v in enumerate(metric_data.values):
                    ax.text(j, v + 0.01 * abs(v), f"{v:.2f}", ha='center')
            
            plt.tight_layout()
            
            # Save or display
            if filename:
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                self.logger.info(f"Saved performance plot to {filename}")
            else:
                plt.show()
                
        except ImportError:
            self.logger.warning("Could not plot performance. Matplotlib and seaborn required.")

    def _calculate_cluster_quality(self, X: np.ndarray) -> None:
        """
        Calculate cluster quality metrics
        
        Args:
            X (np.ndarray): Feature matrix
        """
        try:
            # Get valid cluster labels (exclude noise points)
            valid_mask = self.cluster_labels >= 0
            valid_labels = self.cluster_labels[valid_mask]
            valid_X = X[valid_mask]
            
            # Silhouette score (only for more than one cluster)
            if len(np.unique(valid_labels)) > 1 and len(valid_labels) > 1:
                self.silhouette_avg = silhouette_score(valid_X, valid_labels)
                self.logger.info(f"Silhouette Score: {self.silhouette_avg:.4f}")
            else:
                self.silhouette_avg = None
                
            # Calinski-Harabasz Index (if more than one cluster and no noise points)
            if len(np.unique(valid_labels)) > 1 and len(valid_labels) > 1:
                self.ch_score = calinski_harabasz_score(valid_X, valid_labels)
                self.logger.info(f"Calinski-Harabasz Score: {self.ch_score:.4f}")
            else:
                self.ch_score = None
            
            # Calculate within-cluster sum of squares for each cluster
            self.within_cluster_ss = {}
            for cluster_id in np.unique(self.cluster_labels):
                if cluster_id == -1:  # Skip noise points
                    continue
                cluster_mask = self.cluster_labels == cluster_id
                cluster_points = X[cluster_mask]
                if len(cluster_points) > 0:
                    cluster_center = cluster_points.mean(axis=0)
                    ss = np.sum(np.linalg.norm(cluster_points - cluster_center, axis=1) ** 2)
                    self.within_cluster_ss[cluster_id] = ss
            
            # Calculate cluster stability using bootstrapping if enough data points
            if X.shape[0] > 100 and self.config.cluster_method in ['kmeans', 'gmm']:
                self._calculate_cluster_stability(X)
                
        except Exception as e:
            self.logger.warning(f"Error calculating cluster quality: {str(e)}")
            self.silhouette_avg = None
            self.ch_score = None

    def _calculate_cluster_stability(self, X: np.ndarray, n_bootstraps: int = 10) -> None:
        """
        Calculate cluster stability using bootstrapped resampling
        
        Args:
            X (np.ndarray): Feature matrix
            n_bootstraps (int): Number of bootstrap samples
        """
        self.logger.info("Calculating cluster stability...")
        
        # Initialize stability scores
        cluster_stability = {i: 0.0 for i in np.unique(self.cluster_labels) if i >= 0}
        
        # Create bootstrap samples and recluster
        np.random.seed(42)
        n_samples = X.shape[0]
        
        for _ in range(n_bootstraps):
            # Create bootstrap sample
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            bootstrap_X = X[bootstrap_indices]
            
            # Recluster using same model
            if self.config.cluster_method == 'kmeans':
                bootstrap_model = KMeans(
                    n_clusters=self.config.n_clusters,
                    random_state=42,
                    n_init=10
                )
                bootstrap_labels = bootstrap_model.fit_predict(bootstrap_X)
            elif self.config.cluster_method == 'gmm':
                bootstrap_model = GaussianMixture(
                    n_components=self.config.n_clusters,
                    random_state=42,
                    n_init=10
                )
                bootstrap_model.fit(bootstrap_X)
                bootstrap_labels = bootstrap_model.predict(bootstrap_X)
            else:
                continue  # Skip for other methods
            
            # Calculate cluster centers
            bootstrap_centers = {}
            for i in range(self.config.n_clusters):
                mask = bootstrap_labels == i
                if np.sum(mask) > 0:
                    bootstrap_centers[i] = bootstrap_X[mask].mean(axis=0)
            
            # Match bootstrap clusters to original clusters
            orig_centers = {}
            for i in np.unique(self.cluster_labels):
                if i >= 0:  # Skip noise
                    mask = self.cluster_labels == i
                    orig_centers[i] = X[mask].mean(axis=0)
            
            # Calculate similarity between original and bootstrap clusters
            for orig_id, orig_center in orig_centers.items():
                # Find closest bootstrap cluster
                min_dist = float('inf')
                closest_id = None
                
                for boot_id, boot_center in bootstrap_centers.items():
                    dist = np.linalg.norm(orig_center - boot_center)
                    if dist < min_dist:
                        min_dist = dist
                        closest_id = boot_id
                
                if closest_id is not None:
                    # Calculate Jaccard similarity between clusters
                    orig_points = set(np.where(self.cluster_labels == orig_id)[0])
                    boot_points = set(np.where(bootstrap_indices)[0][bootstrap_labels == closest_id])
                    
                    # Skip if either set is empty
                    if not orig_points or not boot_points:
                        continue
                        
                    # Calculate Jaccard similarity
                    intersection = len(orig_points.intersection(boot_points))
                    union = len(orig_points.union(boot_points))
                    
                    if union > 0:
                        jaccard = intersection / union
                        cluster_stability[orig_id] += jaccard / n_bootstraps
        
        # Store stability scores
        self.cluster_stability = cluster_stability
        self.logger.info(f"Cluster stability scores: {cluster_stability}")

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
            
            # Calculate max drawdown
            max_drawdown = self._calculate_max_drawdown(cluster_df['Close'])
            
            stats = {
                'count': len(cluster_df),
                'avg_return': returns.mean(),
                'return_std': returns.std(),
                'return_skew': returns.skew() if len(returns) > 2 else 0,
                'sharpe': returns.mean() / returns.std() if returns.std() > 0 else 0,
                'volatility': returns.std() * np.sqrt(252),
                'max_drawdown': max_drawdown,
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
            
            # Classify trend direction
            if stats['price_trend'] > 0.05:  # 5% threshold for bullish
                stats['trend_direction'] = 'bullish'
            elif stats['price_trend'] < -0.05:  # -5% threshold for bearish
                stats['trend_direction'] = 'bearish'
            else:
                stats['trend_direction'] = 'neutral'
                
            # Classify volatility level
            if stats['volatility'] < 0.10:  # Annualized volatility below 10%
                stats['volatility_level'] = 'low'
            elif stats['volatility'] > 0.25:  # Annualized volatility above 25%
                stats['volatility_level'] = 'high'
            else:
                stats['volatility_level'] = 'medium'
                
            # Classify volume level if available
            if 'Volume' in cluster_df:
                if stats['volume_trend'] > 0.20:  # 20% increase in volume
                    stats['volume_level'] = 'increasing'
                elif stats['volume_trend'] < -0.20:  # 20% decrease in volume
                    stats['volume_level'] = 'decreasing'
                else:
                    stats['volume_level'] = 'stable'
            else:
                stats['volume_level'] = 'unknown'
                
            # Store stats
            cluster_stats[cluster_id] = stats
            
        # Map clusters to market regime types based on their characteristics
        regime_mapping = {}
        
        for cluster_id, stats in cluster_stats.items():
            # Determine regime type based on trend, volatility, and returns
            if stats['trend_direction'] == 'bullish':
                if stats['volatility_level'] == 'high':
                    regime_type = 'bullish_volatile'
                elif stats['trend_strength'] > 0.7:  # Strong trend correlation
                    regime_type = 'bullish_trending'
                else:
                    regime_type = 'bullish_quiet'
            elif stats['trend_direction'] == 'bearish':
                if stats['volatility_level'] == 'high' and stats['max_drawdown'] < -0.15:
                    regime_type = 'crash'  # Severe drawdown with high volatility
                elif stats['volatility_level'] == 'high':
                    regime_type = 'bearish_volatile'
                elif stats['trend_strength'] > 0.7:  # Strong trend correlation
                    regime_type = 'bearish_trending'
                else:
                    regime_type = 'bearish_quiet'
            else:  # Neutral trend
                if stats['volatility_level'] == 'high':
                    regime_type = 'high_volatility'
                elif stats['volatility_level'] == 'low':
                    regime_type = 'low_volatility_range'
                else:
                    # Check for mean reversion characteristics
                    if 'return_autocorr_5d' in self.features.columns:
                        mean_reversion = self.features.loc[self.cluster_labels == cluster_id, 'return_autocorr_5d'].mean() < -0.2
                        if mean_reversion:
                            regime_type = 'mean_reverting'
                        else:
                            regime_type = 'transitional'
                    else:
                        regime_type = 'transitional'
                        
            # Special case for recovery
            if (stats['trend_direction'] == 'bullish' and 
                stats['price_trend'] > 0.10 and  # Significant upward move
                stats['max_drawdown'] > -0.05):  # Limited drawdowns during recovery
                regime_type = 'recovery'
                
            # Store mapping
            regime_mapping[cluster_id] = regime_type
            
        # Create regimes by finding continuous segments of the same cluster
        self.regimes = []
        
        # Helper function to detect continuous segments
        def detect_segments(series):
            segments = []
            current_val = None
            start_idx = None
            
            for idx, val in series.items():
                if val != current_val:
                    # End previous segment
                    if start_idx is not None:
                        segments.append((start_idx, idx, current_val))
                    
                    # Start new segment
                    current_val = val
                    start_idx = idx
            
            # Add final segment
            if start_idx is not None:
                segments.append((start_idx, series.index[-1], current_val))
                
            return segments
            
        # Detect segments
        segments = detect_segments(self.cluster_labels)
        
        # Convert segments to regime objects
        for start_date, end_date, cluster_id in segments:
            if cluster_id == -1:  # Skip noise points
                continue
                
            # Get regime characteristics
            stats = cluster_stats.get(cluster_id, {})
            regime_type = regime_mapping.get(cluster_id, 'undefined')
            
            # Get confidence from probabilities
            if f'Cluster_{cluster_id}' in self.cluster_probs.columns:
                cluster_probs = self.cluster_probs.loc[start_date:end_date, f'Cluster_{cluster_id}']
                confidence = cluster_probs.mean()
            else:
                confidence = 0.5  # Default confidence
                
            # Create regime object
            regime = ClusteringMarketRegime(
                regime_type=regime_type,
                volatility=stats.get('volatility_level', 'medium'),
                trend=stats.get('trend_direction', 'neutral'),
                volume=stats.get('volume_level', 'unknown'),
                start_date=start_date,
                end_date=end_date,
                confidence=confidence,
                cluster_id=cluster_id,
                stability_score=getattr(self, 'cluster_stability', {}).get(cluster_id, 0.5)
            )
            
            # Add feature vector (cluster centroid)
            if hasattr(self, 'cluster_model') and self.config.cluster_method == 'kmeans':
                regime.feature_vector = self.cluster_model.cluster_centers_[cluster_id]
            elif hasattr(self, 'cluster_model') and self.config.cluster_method == 'gmm':
                regime.feature_vector = self.cluster_model.means_[cluster_id]
            else:
                # Use average of points in cluster
                mask = self.cluster_labels == cluster_id
                regime.feature_vector = self.features_processed.loc[mask].mean().values
            
            # Store feature importance for this regime
            feature_importance = {}
            if self.config.use_pca:
                # For PCA, use overall feature importance
                feature_importance = self.feature_importances.copy()
            else:
                # Calculate feature importance based on distance from global mean
                global_mean = self.features.mean()
                regime_mean = self.features.loc[self.cluster_labels == cluster_id].mean()
                
                for feature in self.features.columns:
                    # Importance is deviation from global mean, normalized by std
                    feature_std = self.features[feature].std()
                    if feature_std > 0:
                        importance = abs(regime_mean[feature] - global_mean[feature]) / feature_std
                        feature_importance[feature] = importance
            
            # Store top features
            regime.feature_importance = dict(sorted(
                feature_importance.items(),
                key=lambda item: item[1],
                reverse=True
            )[:10])  # Keep top 10 features
            
            # Add to regimes list
            self.regimes.append(regime)
            
        self.logger.info(f"Mapped {len(self.regimes)} regime segments from {len(regime_mapping)} clusters")

    def _analyze_regime_transitions(self) -> None:
        """Analyze transitions between market regimes"""
        self.logger.info("Analyzing regime transitions")
        
        if not self.regimes or len(self.regimes) < 2:
            self.logger.warning("Not enough regimes to analyze transitions")
            return
        
        # Extract regime types and their sequences
        regime_types = [regime.regime_type for regime in self.regimes]
        unique_regimes = sorted(set(regime_types))
        
        # Initialize transition count matrix
        n_regimes = len(unique_regimes)
        transition_counts = np.zeros((n_regimes, n_regimes))
        
        # Count transitions
        for i in range(len(regime_types) - 1):
            from_idx = unique_regimes.index(regime_types[i])
            to_idx = unique_regimes.index(regime_types[i + 1])
            transition_counts[from_idx, to_idx] += 1
        
        # Convert to probabilities
        transition_probs = np.zeros_like(transition_counts, dtype=float)
        for i in range(n_regimes):
            row_sum = transition_counts[i].sum()
            if row_sum > 0:
                transition_probs[i] = transition_counts[i] / row_sum
        
        # Create transition matrix DataFrame
        self.transition_matrix = pd.DataFrame(
            transition_probs,
            index=unique_regimes,
            columns=unique_regimes
        )
        
        # Store transition probabilities in each regime object
        for regime in self.regimes:
            if regime.regime_type in unique_regimes:
                from_idx = unique_regimes.index(regime.regime_type)
                regime.transition_probs = {
                    to_regime: transition_probs[from_idx, to_idx]
                    for to_idx, to_regime in enumerate(unique_regimes)
                    if transition_probs[from_idx, to_idx] > 0
                }
        
        # Calculate expected duration for each regime type
        # Based on Markov chain properties: E[Duration] = 1 / (1 - p_ii)
        # where p_ii is the probability of staying in the same state
        for i, regime_type in enumerate(unique_regimes):
            # Probability of staying in the same regime
            p_stay = transition_probs[i, i]
            
            if p_stay < 1:
                expected_duration = 1 / (1 - p_stay)
            else:
                expected_duration = float('inf')  # Absorbing state
                
            # Update expected duration for all regimes of this type
            for regime in self.regimes:
                if regime.regime_type == regime_type:
                    regime.expected_duration = expected_duration
                    
        self.logger.info(f"Transition matrix calculated for {n_regimes} regime types")

    def _calculate_regime_probabilities(self) -> None:
        """Calculate regime probabilities and stability"""
        self.logger.info("Calculating regime probabilities and stability metrics")
        
        if not self.regimes:
            return
            
        # Calculate probability confidence for each regime
        for regime in self.regimes:
            # Date range for this regime
            date_range = pd.date_range(regime.start_date, regime.end_date)
            date_range = date_range[date_range.isin(self.cluster_probs.index)]
            
            if len(date_range) == 0:
                continue
                
            # Get probability values for this cluster
            if f'Cluster_{regime.cluster_id}' in self.cluster_probs.columns:
                probs = self.cluster_probs.loc[date_range, f'Cluster_{regime.cluster_id}']
                
                # Average probability gives confidence
                regime.confidence = probs.mean()
                
                # Probability variance gives stability
                stability = 1.0 - probs.std() * 2  # Lower variance = higher stability
                regime.stability_score = max(0.0, min(1.0, stability))  # Clamp to [0,1]
            else:
                # Default values if cluster not found
                regime.confidence = 0.5
                regime.stability_score = 0.5
                
        # Calculate silhouette scores for each regime if possible
        if hasattr(self, 'features_processed') and len(self.regimes) > 1:
            # Get processed features
            X = self.features_processed.values
            
            for regime in self.regimes:
                # Get points in this regime
                mask = (self.features_processed.index >= regime.start_date) & \
                    (self.features_processed.index <= regime.end_date)
                regime_X = X[mask]
                regime_labels = self.cluster_labels[mask]
                
                # Calculate silhouette score if enough points
                if len(regime_X) > 2 and len(np.unique(regime_labels)) > 1:
                    try:
                        regime.silhouette_score = silhouette_score(
                            regime_X, 
                            regime_labels
                        )
                    except:
                        regime.silhouette_score = None

    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown from price series
        
        Args:
            prices (pd.Series): Price series
            
        Returns:
            float: Maximum drawdown as a percentage (negative value)
        """
        if len(prices) < 2:
            return 0.0
            
        # Calculate running maximum
        running_max = prices.cummax()
        
        # Calculate drawdown
        drawdown = (prices / running_max - 1.0)
        
        # Return minimum (maximum drawdown)
        return drawdown.min()

    def analyze_regime_stats(self, 
                        regime_type: Optional[str] = None, 
                        return_horizon: int = 5,
                        market_days: bool = True) -> pd.DataFrame:
        """
        Analyze performance statistics for detected regimes
        
        Args:
            regime_type (Optional[str]): Filter by specific regime type
            return_horizon (int): Horizon for forward returns in days
            market_days (bool): Whether to use market days or calendar days
            
        Returns:
            pd.DataFrame: Performance statistics for regimes
        """
        if not self.regimes:
            self.logger.warning("No regimes available for analysis")
            return pd.DataFrame()
            
        # Filter regimes if specified
        filtered_regimes = self.regimes
        if regime_type is not None:
            filtered_regimes = [r for r in self.regimes if r.regime_type == regime_type]
            
        if not filtered_regimes:
            self.logger.warning(f"No regimes of type '{regime_type}' found")
            return pd.DataFrame()
        
        # Set up results container
        performance = []
        
        # Calculate returns for each regime
        for regime in filtered_regimes:
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
                    
                    # Fixed: Use proper indexing to avoid treating 'Close' as a date
                    start_price = regime_data.at[date, 'Close']  # Changed from loc[date, 'Close']
                    
                    # Make sure end_date is in df.index before accessing
                    if end_date in self.df.index:
                        end_price = self.df.at[end_date, 'Close']  # Changed from loc[end_date, 'Close']
                        fwd_return = (end_price / start_price - 1) * 100  # percentage
                        forward_returns.append(fwd_return)
                except Exception as e:
                    self.logger.warning(f"Error calculating forward return at {date}: {str(e)}")
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
    
    def _get_top_features_for_regimes(self, 
                                regimes: List[ClusteringMarketRegime], 
                                top_n: int = 5) -> Dict[str, float]:
        """
        Get top features for a set of regimes
        
        Args:
            regimes (List[ClusteringMarketRegime]): Regimes to analyze
            top_n (int): Number of top features to return
            
        Returns:
            Dict[str, float]: Top features and their importance scores
        """
        # Aggregate feature importance across all regimes
        feature_scores = {}
        
        for regime in regimes:
            if not regime.feature_importance:
                continue
                
            for feature, importance in regime.feature_importance.items():
                if feature not in feature_scores:
                    feature_scores[feature] = 0
                    
                # Weight by regime duration
                duration_days = (regime.end_date - regime.start_date).days
                feature_scores[feature] += importance * duration_days
        
        # Normalize scores
        if feature_scores:
            total_score = sum(feature_scores.values())
            if total_score > 0:
                for feature in feature_scores:
                    feature_scores[feature] /= total_score
        
        # Return top N features
        return dict(sorted(
            feature_scores.items(),
            key=lambda item: item[1],
            reverse=True
        )[:top_n])

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
            if regime.feature_importance:
                top_features = list(regime.feature_importance.items())[:5]
                for i, (feature, importance) in enumerate(top_features):
                    record[f'feature_{i+1}'] = feature
                    record[f'importance_{i+1}'] = importance
            
            # Add transition probabilities
            if regime.transition_probs:
                for to_regime, prob in regime.transition_probs.items():
                    record[f'transition_to_{to_regime}'] = prob
            
            # Add performance metrics
            try:
                regime_data = self.df[regime.start_date:regime.end_date]
                if len(regime_data) >= 2:
                    returns = regime_data['Close'].pct_change().dropna()
                    record.update({
                        'total_return': (regime_data['Close'].iloc[-1] / regime_data['Close'].iloc[0] - 1) * 100,
                        'annualized_return': returns.mean() * 252 * 100,
                        'annualized_volatility': returns.std() * np.sqrt(252) * 100,
                        'sharpe_ratio': (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0,
                        'win_rate': (returns > 0).mean() * 100,
                        'max_drawdown': self._calculate_max_drawdown(regime_data['Close']) * 100
                    })
            except Exception as e:
                self.logger.warning(f"Error calculating performance metrics: {str(e)}")
            
            # Add to export data
            export_data.append(record)
        
        # Convert to DataFrame and export
        try:
            export_df = pd.DataFrame(export_data)
            export_df.to_csv(filename, index=False)
            self.logger.info(f"Exported regime data to {filename}")
        except Exception as e:
            self.logger.error(f"Error exporting regime data: {str(e)}")

    def visualize_regimes(self, 
                        include_clusters: bool = True, 
                        filename: Optional[str] = None) -> None:
        """
        Visualize regimes with cluster assignments
        
        Args:
            include_clusters (bool): Whether to include cluster IDs
            filename (Optional[str]): Output filename, if None displays plot
        """
        if not self.regimes:
            self.logger.warning("No regimes to visualize")
            return
            
        try:
            import matplotlib.pyplot as plt
            from matplotlib.dates import date2num
            
            # Create figure
            fig, ax = plt.subplots(figsize=(15, 8))
            
            # Plot price data
            ax.plot(self.df.index, self.df['Close'], color='black', alpha=0.7)
            
            # Add regime boundaries
            y_min, y_max = ax.get_ylim()
            height = y_max - y_min
            
            # Define colors for different regime types
            regime_colors = {
                'bullish_trending': 'green',
                'bearish_trending': 'red',
                'high_volatility': 'orange',
                'low_volatility_range': 'blue',
                'transitional': 'gray',
                'bullish_volatile': 'lightgreen',
                'bearish_volatile': 'salmon',
                'bullish_quiet': 'palegreen',
                'bearish_quiet': 'mistyrose',
                'mean_reverting': 'lightblue',
                'recovery': 'yellowgreen',
                'crash': 'darkred'
            }
            
            # Plot regimes as colored backgrounds
            for i, regime in enumerate(self.regimes):
                # Get color for this regime
                color = regime_colors.get(regime.regime_type, 'gray')
                
                # Add colored background for regime period
                ax.axvspan(
                    regime.start_date, 
                    regime.end_date, 
                    alpha=0.2, 
                    color=color, 
                    label=f"{regime.regime_type}" if i == 0 else ""
                )
                
                # Optionally add cluster ID
                if include_clusters:
                    # Get midpoint of regime period
                    mid_date = regime.start_date + (regime.end_date - regime.start_date) / 2
                    
                    # Add cluster ID text
                    ax.text(
                        mid_date, 
                        y_min + height * 0.05, 
                        f"C{regime.cluster_id}", 
                        ha='center', 
                        va='bottom', 
                        fontsize=8
                    )
            
            # Add legend for regime types (deduplicate by colors)
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper left')
            
            # Set labels and title
            ax.set_title("Market Regimes Visualization")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            
            # Format x-axis dates
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save or display
            if filename:
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                self.logger.info(f"Saved regime visualization to {filename}")
            else:
                plt.show()
        
        except ImportError as e:
            self.logger.warning(f"Could not visualize regimes: {str(e)}. Matplotlib required.")

    def generate_regime_report(self, output_dir: str = '.') -> None:
        """
        Generate comprehensive HTML report of regime analysis
        
        Args:
            output_dir (str): Directory to save report files
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.figure import Figure
            from io import BytesIO
            import base64
            import os
            
            # Create output directory if it doesn't exist
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Get regime statistics
            perf_df = self.analyze_regime_stats()
            
            # Create HTML content
            html_content = [
                "<!DOCTYPE html>",
                "<html>",
                "<head>",
                "    <title>Market Regime Analysis Report</title>",
                "    <style>",
                "        body { font-family: Arial, sans-serif; margin: 20px; }",
                "        h1, h2, h3 { color: #333366; }",
                "        .container { max-width: 1200px; margin: auto; }",
                "        .section { margin-bottom: 30px; }",
                "        table { border-collapse: collapse; width: 100%; }",
                "        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
                "        th { background-color: #f2f2f2; }",
                "        tr:nth-child(even) { background-color: #f9f9f9; }",
                "        .figure { margin: 20px 0; text-align: center; }",
                "        .bullish { color: green; }",
                "        .bearish { color: red; }",
                "        .volatile { color: orange; }",
                "        .neutral { color: blue; }",
                "    </style>",
                "</head>",
                "<body>",
                "    <div class='container'>",
                f"        <h1>Market Regime Analysis Report</h1>",
                f"        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>",
                "        <div class='section'>",
                "            <h2>Summary</h2>",
                f"            <p>Analyzed {len(self.df)} data points from {self.df.index[0].date()} to {self.df.index[-1].date()}.</p>",
                f"            <p>Detected {len(self.regimes)} distinct market regimes using {self.config.cluster_method} clustering.</p>",
                "        </div>"
            ]
            
            # Create regime timeline visualization
            timeline_img = os.path.join(output_dir, "regime_timeline.png")
            self.plot_regime_timeline(filename=timeline_img)
            
            html_content.extend([
                "        <div class='section'>",
                "            <h2>Regime Timeline</h2>",
                f"            <div class='figure'><img src='regime_timeline.png' width='100%' alt='Regime Timeline'></div>",
                "        </div>"
            ])
            
            # Regime statistics table
            if not perf_df.empty:
                html_content.extend([
                    "        <div class='section'>",
                    "            <h2>Regime Performance Statistics</h2>",
                    "            <table>",
                    "                <tr>"
                ])
                
                # Add table headers
                columns = ['regime', 'duration_days', 'daily_return_mean', 'daily_return_std', 
                        'daily_sharpe', 'win_rate', 'total_return']
                column_names = ['Regime Type', 'Avg Duration (Days)', 'Daily Return (%)', 
                            'Daily Std (%)', 'Sharpe Ratio', 'Win Rate (%)', 'Total Return (%)']
                
                for col_name in column_names:
                    html_content.append(f"                    <th>{col_name}</th>")
                
                html_content.append("                </tr>")
                
                # Add regime rows (aggregated statistics)
                agg_df = perf_df[perf_df['type'] == 'average']
                
                for _, row in agg_df.iterrows():
                    regime_type = row['regime']
                    css_class = ''
                    
                    if 'bullish' in regime_type:
                        css_class = 'bullish'
                    elif 'bearish' in regime_type:
                        css_class = 'bearish'
                    elif 'volatile' in regime_type:
                        css_class = 'volatile'
                    else:
                        css_class = 'neutral'
                    
                    html_content.append(f"                <tr>")
                    html_content.append(f"                    <td class='{css_class}'>{regime_type.replace('_', ' ').title()}</td>")
                    
                    for col in columns[1:]:
                        value = row[col]
                        if col in ['daily_return_mean', 'daily_return_std', 'win_rate', 'total_return']:
                            html_content.append(f"                    <td>{value:.2f}%</td>")
                        elif col == 'daily_sharpe':
                            html_content.append(f"                    <td>{value:.2f}</td>")
                        else:
                            html_content.append(f"                    <td>{value:.1f}</td>")
                    
                    html_content.append(f"                </tr>")
                
                html_content.extend([
                    "            </table>",
                    "        </div>"
                ])
            
            # Regime transitions
            transitions_img = os.path.join(output_dir, "regime_transitions.png")
            try:
                self.plot_regime_transitions(filename=transitions_img)
                html_content.extend([
                    "        <div class='section'>",
                    "            <h2>Regime Transitions</h2>",
                    f"            <div class='figure'><img src='regime_transitions.png' width='80%' alt='Regime Transitions'></div>",
                    "        </div>"
                ])
            except:
                self.logger.warning("Could not generate regime transitions visualization")
            
            # Cluster visualization
            clusters_img = os.path.join(output_dir, "clusters.png")
            try:
                self.visualize_clusters(filename=clusters_img)
                html_content.extend([
                    "        <div class='section'>",
                    "            <h2>Cluster Visualization</h2>",
                    f"            <div class='figure'><img src='clusters.png' width='80%' alt='Cluster Visualization'></div>",
                    "        </div>"
                ])
            except:
                self.logger.warning("Could not generate cluster visualization")
            
            # Individual regime details
            html_content.extend([
                "        <div class='section'>",
                "            <h2>Individual Regime Details</h2>"
            ])
            
            for i, regime in enumerate(self.regimes):
                regime_class = ''
                if 'bullish' in regime.regime_type:
                    regime_class = 'bullish'
                elif 'bearish' in regime.regime_type:
                    regime_class = 'bearish'
                elif 'volatile' in regime.regime_type:
                    regime_class = 'volatile'
                else:
                    regime_class = 'neutral'
                    
                html_content.extend([
                    f"            <h3 class='{regime_class}'>{i+1}. {regime.regime_type.replace('_', ' ').title()}</h3>",
                    "            <table>",
                    "                <tr><th>Start Date</th><td>" + regime.start_date.strftime('%Y-%m-%d') + "</td></tr>",
                    "                <tr><th>End Date</th><td>" + regime.end_date.strftime('%Y-%m-%d') + "</td></tr>",
                    f"                <tr><th>Duration</th><td>{(regime.end_date - regime.start_date).days} days</td></tr>",
                    f"                <tr><th>Trend</th><td>{regime.trend}</td></tr>",
                    f"                <tr><th>Volatility</th><td>{regime.volatility}</td></tr>",
                    f"                <tr><th>Volume</th><td>{regime.volume}</td></tr>",
                    f"                <tr><th>Confidence</th><td>{regime.confidence:.2f}</td></tr>",
                    f"                <tr><th>Stability</th><td>{regime.stability_score:.2f}</td></tr>"
                ])
                
                # Add top features
                if regime.feature_importance:
                    html_content.append("                <tr><th>Top Features</th><td>")
                    for feature, importance in list(regime.feature_importance.items())[:5]:
                        html_content.append(f"{feature}: {importance:.3f}<br>")
                    html_content.append("</td></tr>")
                    
                # Add transition probabilities
                if regime.transition_probs:
                    html_content.append("                <tr><th>Likely Transitions</th><td>")
                    for to_regime, prob in sorted(regime.transition_probs.items(), key=lambda x: x[1], reverse=True)[:3]:
                        html_content.append(f"To {to_regime.replace('_', ' ').title()}: {prob:.2f}<br>")
                    html_content.append("</td></tr>")
                    
                html_content.append("            </table>")
            
            html_content.extend([
                "        </div>",
                "    </div>",
                "</body>",
                "</html>"
            ])
            
            # Write HTML file
            report_path = os.path.join(output_dir, "regime_report.html")
            with open(report_path, 'w') as f:
                f.write('\n'.join(html_content))
                
            self.logger.info(f"Generated regime report at {report_path}")
            
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")

    def get_regime_stats(self, regime_type: str) -> Dict[str, Any]:
        """
        Get detailed statistics for a specific regime type
        
        Args:
            regime_type (str): Regime type to analyze
            
        Returns:
            Dict[str, Any]: Regime statistics
        """
        # Filter regimes by type
        filtered_regimes = [r for r in self.regimes if r.regime_type == regime_type]
        
        if not filtered_regimes:
            self.logger.warning(f"No regimes of type '{regime_type}' found")
            return {}
            
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
                    # Fixed: Use proper indexing
                    if regime.start_date in self.df.index and regime.end_date in self.df.index:
                        start_price = self.df.at[regime.start_date, 'Close']  # Changed from loc
                        end_price = self.df.at[regime.end_date, 'Close']      # Changed from loc
                        perf = (end_price / start_price - 1) * 100
                        price_performance.append(perf)
                except Exception as e:
                    self.logger.warning(f"Error calculating performance: {str(e)}")
                    continue
                    
            if price_performance:
                results['avg_price_change_pct'] = np.mean(price_performance)
                results['price_change_std'] = np.std(price_performance)
                
        return results    
    def visualize_clusters(self, 
                        dim_reduction: str = 'pca',
                        filename: Optional[str] = None) -> None:
        """
        Visualize clusters in 2D space
        
        Args:
            dim_reduction (str): Dimensionality reduction technique ('pca' or 'tsne')
            filename (Optional[str]): Output filename, if None displays plot
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            if not hasattr(self, 'features_processed') or len(self.features_processed) == 0:
                self.logger.warning("No processed features available for visualization")
                return
                
            # Get feature matrix
            X = self.features_processed.values
            
            # Reduce to 2D for visualization
            if dim_reduction.lower() == 'tsne':
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=2, random_state=42)
            else:  # Default to PCA
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=2, random_state=42)
                
            # Apply dimensionality reduction
            X_2d = reducer.fit_transform(X)
            
            # Create DataFrame for plotting
            viz_df = pd.DataFrame({
                'x': X_2d[:, 0],
                'y': X_2d[:, 1],
                'cluster': self.cluster_labels,
                'date': self.features_processed.index
            })
            
            # Set up plot
            plt.figure(figsize=(12, 8))
            
            # Define cluster colors and markers
            unique_clusters = viz_df['cluster'].unique()
            
            # Skip noise points (cluster -1) in color mapping
            valid_clusters = [c for c in unique_clusters if c >= 0]
            
            # Create colormap for clusters
            cluster_colors = plt.cm.tab10(np.linspace(0, 1, len(valid_clusters)))
            
            # Plot each cluster
            for i, cluster in enumerate(valid_clusters):
                cluster_data = viz_df[viz_df['cluster'] == cluster]
                plt.scatter(
                    cluster_data['x'], 
                    cluster_data['y'],
                    s=50, 
                    c=[cluster_colors[i]], 
                    label=f"Cluster {cluster}",
                    alpha=0.7
                )
                
            # Plot noise points with distinct style if present
            if -1 in unique_clusters:
                noise_data = viz_df[viz_df['cluster'] == -1]
                plt.scatter(
                    noise_data['x'], 
                    noise_data['y'],
                    s=30, 
                    c='gray', 
                    marker='x',
                    label="Noise",
                    alpha=0.5
                )
                
            # Add timeline effects (connecting points in chronological order)
            dates = viz_df['date'].sort_values().unique()
            if len(dates) > 1:
                # Get points in chronological order
                timeline_df = viz_df.sort_values('date')
                
                # Plot lines connecting points
                plt.plot(timeline_df['x'], timeline_df['y'], 
                    c='gray', alpha=0.2, linewidth=0.5)
                
                # Highlight start and end points
                plt.scatter(
                    timeline_df['x'].iloc[0], 
                    timeline_df['y'].iloc[0],
                    s=100, 
                    c='none', 
                    edgecolor='green',
                    linewidth=2,
                    label="Start"
                )
                
                plt.scatter(
                    timeline_df['x'].iloc[-1], 
                    timeline_df['y'].iloc[-1],
                    s=100, 
                    c='none', 
                    edgecolor='red',
                    linewidth=2,
                    label="End"
                )
                
            plt.title(f"Market Regimes Visualization ({dim_reduction.upper()})")
            plt.xlabel(f"{dim_reduction.upper()} Component 1")
            plt.ylabel(f"{dim_reduction.upper()} Component 2")
            plt.legend()
            
            # Add grid
            plt.grid(True, alpha=0.3)
            
            # Save or display
            if filename:
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                self.logger.info(f"Saved cluster visualization to {filename}")
            else:
                plt.show()
                
        except ImportError:
            self.logger.warning("Could not visualize clusters. Required libraries missing.")

    def plot_regime_timeline(self, 
                        with_prices: bool = True,
                        filename: Optional[str] = None) -> None:
        """
        Plot timeline of detected regimes with price overlay
        
        Args:
            with_prices (bool): Whether to include price chart
            filename (Optional[str]): Output filename, if None displays plot
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            import matplotlib.patches as patches
            
            if not self.regimes:
                self.logger.warning("No regimes to plot")
                return
                
            # Set up plot
            fig, ax1 = plt.subplots(figsize=(14, 8))
            
            # Plot price if requested
            if with_prices:
                ax1.plot(self.df.index, self.df['Close'], color='black', alpha=0.7, linewidth=1.5)
                ax1.set_ylabel('Price', fontsize=12)
                ax1.set_xlabel('Date', fontsize=12)
                
                # Format y-axis
                ax1.tick_params(axis='y', labelsize=10)
                
                # Add price grid
                ax1.grid(True, alpha=0.3)
                
            # Add regime background colors
            ylim = ax1.get_ylim()
            height = ylim[1] - ylim[0]
            
            # Define colors for different regime types
            regime_colors = {
                'bullish_trending': 'lightgreen',
                'bullish_volatile': 'palegreen',
                'bullish_quiet': 'honeydew',
                'recovery': 'limegreen',
                'bearish_trending': 'lightcoral',
                'bearish_volatile': 'mistyrose',
                'bearish_quiet': 'lavenderblush',
                'crash': 'tomato',
                'high_volatility': 'wheat',
                'low_volatility_range': 'lightsteelblue',
                'transitional': 'lightgray',
                'mean_reverting': 'lightskyblue',
                'undefined': 'whitesmoke'
            }
            
            # Add default color for any regime types not in the dictionary
            default_color = 'lightgray'
            
            # Plot each regime as a colored background
            for i, regime in enumerate(self.regimes):
                # Get color for this regime
                color = regime_colors.get(regime.regime_type, default_color)
                
                # Create rectangle for this regime
                rect = patches.Rectangle(
                    (mdates.date2num(regime.start_date), ylim[0]),
                    mdates.date2num(regime.end_date) - mdates.date2num(regime.start_date),
                    height,
                    facecolor=color,
                    alpha=0.4,
                    edgecolor='none'
                )
                ax1.add_patch(rect)
                
                # Add regime labels
                mid_date = regime.start_date + (regime.end_date - regime.start_date) / 2
                
                # Skip labels that would be too crowded
                if i > 0:
                    prev_mid = self.regimes[i-1].start_date + (self.regimes[i-1].end_date - self.regimes[i-1].start_date) / 2
                    days_diff = (mid_date - prev_mid).days
                    
                    # Skip if less than 30 days from previous label
                    if days_diff < 30:
                        continue
                
                ax1.text(
                    mid_date, 
                    ylim[0] + height * 0.05,  # Position at bottom of plot
                    regime.regime_type.replace('_', ' ').title(),
                    fontsize=9,
                    ha='center',
                    rotation=90,
                    color='black'
                )
                
            # Format x-axis dates
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax1.xaxis.set_major_locator(mdates.YearLocator())
            plt.xticks(rotation=45, ha='right')
            
            # Create legend for regime types
            legend_handles = []
            for regime_type, color in regime_colors.items():
                # Only add to legend if this regime type appears in data
                if any(r.regime_type == regime_type for r in self.regimes):
                    handle = patches.Patch(
                        facecolor=color,
                        alpha=0.4,
                        label=regime_type.replace('_', ' ').title()
                    )
                    legend_handles.append(handle)
                    
            plt.legend(handles=legend_handles, loc='upper left', fontsize=9)
            
            plt.title('Market Regimes Timeline', fontsize=14)
            plt.tight_layout()
            
            # Save or display
            if filename:
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                self.logger.info(f"Saved regime timeline to {filename}")
            else:
                plt.show()
                
        except ImportError:
            self.logger.warning("Could not plot regime timeline. matplotlib required.")