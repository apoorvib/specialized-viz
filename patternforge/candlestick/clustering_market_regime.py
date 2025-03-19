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