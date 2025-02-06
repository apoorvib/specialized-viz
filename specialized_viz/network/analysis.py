import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from ..candlestick.patterns import CandlestickPatterns
from ..candlestick.visualization import VisualizationConfig
from sklearn.cluster import SpectralClustering
from scipy import stats

@dataclass
class NetworkConfig:
    """Configuration class for network analysis settings"""
    correlation_threshold: float = 0.5
    pattern_threshold: float = 0.3
    min_edge_weight: float = 0.1
    community_resolution: float = 1.0
    temporal_window: int = 20
    pattern_lookback: int = 100
    regime_window: int = 50
    
    # Color schemes aligned with VisualizationConfig
    color_scheme: Dict[str, str] = None
    
    def __post_init__(self):
        if self.color_scheme is None:
            self.color_scheme = {
                'node_default': '#3498db',
                'edge_default': '#95a5a6',
                'community_colors': [
                    '#2ecc71', '#e74c3c', '#f1c40f', '#9b59b6',
                    '#34495e', '#1abc9c', '#e67e22', '#7f8c8d'
                ],
                'pattern_colors': {
                    'bullish': '#2ecc71',
                    'bearish': '#e74c3c',
                    'neutral': '#3498db'
                }
            }

class NetworkBuilder:
    """Core class for building and analyzing financial networks"""
    
    def __init__(self, 
                 data: Dict[str, pd.DataFrame], 
                 config: Optional[NetworkConfig] = None):
        """
        Initialize NetworkBuilder with multiple asset data
        
        Args:
            data: Dictionary mapping asset names to their OHLCV DataFrames
            config: Network configuration settings
        """
        self.data = data
        self.config = config or NetworkConfig()
        self.patterns = CandlestickPatterns()
        self.graph = None
        self._pattern_cache = {}
        
    def create_correlation_network(self, 
                                 method: str = 'pearson',
                                 threshold: Optional[float] = None) -> nx.Graph:
        """
        Create correlation network from asset returns
        
        Args:
            method: Correlation method ('pearson', 'spearman', or 'kendall')
            threshold: Optional correlation threshold (defaults to config value)
            
        Returns:
            nx.Graph: Correlation network
        """
        # Calculate returns for all assets
        returns = pd.DataFrame({
            asset: df['Close'].pct_change()
            for asset, df in self.data.items()
        })
        
        # Calculate correlation matrix
        corr_matrix = returns.corr(method=method)
        
        # Create network
        threshold = threshold or self.config.correlation_threshold
        self.graph = nx.Graph()
        
        # Add nodes
        self.graph.add_nodes_from(corr_matrix.index)
        
        # Add edges above threshold
        for i in range(len(corr_matrix)):
            for j in range(i + 1, len(corr_matrix)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) >= threshold:
                    self.graph.add_edge(
                        corr_matrix.index[i],
                        corr_matrix.index[j],
                        weight=abs(corr)
                    )
        
        return self.graph
    
    def create_pattern_network(self, pattern_types: Optional[List[str]] = None) -> nx.Graph:
        """
        Create network based on candlestick pattern co-occurrence
        
        Args:
            pattern_types: List of pattern types to consider (None for all)
            
        Returns:
            nx.Graph: Pattern co-occurrence network
        """
        # Get all pattern detection methods if not specified
        if pattern_types is None:
            pattern_methods = {
                name.replace('detect_', ''): method 
                for name, method in CandlestickPatterns.__dict__.items()
                if name.startswith('detect_') and callable(method)
            }
        else:
            pattern_methods = {
                pattern: getattr(CandlestickPatterns, f'detect_{pattern}')
                for pattern in pattern_types
            }
        
        # Calculate pattern occurrences for each asset
        pattern_occurrences = {}
        for asset, df in self.data.items():
            asset_patterns = {}
            for pattern_name, pattern_func in pattern_methods.items():
                try:
                    result = pattern_func(df)
                    if isinstance(result, tuple):
                        # Handle patterns that return bullish/bearish signals
                        for idx, signal_type in enumerate(['bullish', 'bearish']):
                            pattern_key = f"{pattern_name}_{signal_type}"
                            asset_patterns[pattern_key] = result[idx]
                    else:
                        asset_patterns[pattern_name] = result
                except Exception as e:
                    print(f"Error detecting {pattern_name} for {asset}: {str(e)}")
            pattern_occurrences[asset] = asset_patterns
        
        # Create network based on pattern similarity
        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.data.keys())
        
        # Calculate pattern similarity between assets
        for i, asset1 in enumerate(self.data.keys()):
            patterns1 = pattern_occurrences[asset1]
            for j, asset2 in enumerate(list(self.data.keys())[i+1:], i+1):
                patterns2 = pattern_occurrences[asset2]
                
                # Calculate Jaccard similarity for each pattern type
                pattern_similarities = []
                for pattern in patterns1.keys():
                    if pattern in patterns2:
                        intersection = sum(patterns1[pattern] & patterns2[pattern])
                        union = sum(patterns1[pattern] | patterns2[pattern])
                        if union > 0:
                            similarity = intersection / union
                            pattern_similarities.append(similarity)
                
                if pattern_similarities:
                    avg_similarity = np.mean(pattern_similarities)
                    if avg_similarity >= self.config.pattern_threshold:
                        self.graph.add_edge(
                            asset1, asset2,
                            weight=avg_similarity
                        )
        
        return self.graph
    
    def create_regime_network(self) -> nx.Graph:
        """
        Create network based on market regime synchronization
        
        Returns:
            nx.Graph: Market regime network
        """
        # Calculate regime states for each asset
        regime_states = {}
        for asset, df in self.data.items():
            # Calculate basic regime indicators
            returns = df['Close'].pct_change()
            volatility = returns.rolling(window=self.config.regime_window).std()
            trend = df['Close'].rolling(window=self.config.regime_window).mean().pct_change()
            
            # Classify regimes
            regime_states[asset] = pd.DataFrame({
                'volatility': pd.qcut(volatility, q=3, labels=['low', 'medium', 'high']),
                'trend': pd.qcut(trend, q=3, labels=['down', 'neutral', 'up'])
            })
        
        # Create network based on regime synchronization
        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.data.keys())
        
        # Calculate regime similarity between assets
        for i, asset1 in enumerate(self.data.keys()):
            regimes1 = regime_states[asset1]
            for j, asset2 in enumerate(list(self.data.keys())[i+1:], i+1):
                regimes2 = regime_states[asset2]
                
                # Calculate regime synchronization
                vol_sync = (regimes1['volatility'] == regimes2['volatility']).mean()
                trend_sync = (regimes1['trend'] == regimes2['trend']).mean()
                
                # Combine metrics
                sync_score = (vol_sync + trend_sync) / 2
                if sync_score >= self.config.correlation_threshold:
                    self.graph.add_edge(
                        asset1, asset2,
                        weight=sync_score
                    )
        
        return self.graph
    
    def detect_communities(self, 
                         method: str = 'louvain',
                         **kwargs) -> Dict[str, int]:
        """
        Detect communities in the network
        
        Args:
            method: Community detection method
            **kwargs: Additional arguments for community detection
            
        Returns:
            Dict[str, int]: Mapping of nodes to community indices
        """
        if self.graph is None:
            raise ValueError("Network must be created before detecting communities")
        
        if method == 'louvain':
            import community
            return community.best_partition(
                self.graph,
                resolution=self.config.community_resolution,
                **kwargs
            )
        elif method == 'spectral':
            # Prepare adjacency matrix
            adj_matrix = nx.adjacency_matrix(self.graph).todense()
            
            # Apply spectral clustering
            n_clusters = kwargs.get('n_clusters', 
                                 min(8, len(self.graph.nodes)))
            clustering = SpectralClustering(
                n_clusters=n_clusters,
                affinity='precomputed',
                **kwargs
            )
            labels = clustering.fit_predict(adj_matrix)
            
            return dict(zip(self.graph.nodes(), labels))
        else:
            raise ValueError(f"Unsupported community detection method: {method}")
    
    def analyze_pattern_propagation(self, 
                                  pattern_type: str,
                                  window: Optional[int] = None) -> pd.DataFrame:
        """
        Analyze how patterns propagate through the network
        
        Args:
            pattern_type: Type of pattern to analyze
            window: Rolling window for propagation analysis
            
        Returns:
            pd.DataFrame: Pattern propagation metrics
        """
        window = window or self.config.temporal_window
        pattern_func = getattr(CandlestickPatterns, f'detect_{pattern_type}')
        
        # Detect patterns for each asset
        pattern_signals = {}
        for asset, df in self.data.items():
            try:
                result = pattern_func(df)
                if isinstance(result, tuple):
                    pattern_signals[asset] = {
                        'bullish': result[0],
                        'bearish': result[1]
                    }
                else:
                    pattern_signals[asset] = {'signal': result}
            except Exception as e:
                print(f"Error detecting {pattern_type} for {asset}: {str(e)}")
                continue
        
        # Calculate propagation metrics
        propagation_metrics = pd.DataFrame(index=self.data[list(self.data.keys())[0]].index)
        
        # For each point in time, calculate:
        # 1. Pattern concentration in network
        # 2. Pattern spread rate
        # 3. Node influence in propagation
        for timestamp in propagation_metrics.index:
            metrics = {}
            
            # Calculate pattern concentration
            for signal_type in next(iter(pattern_signals.values())).keys():
                concentration = np.mean([
                    signals[signal_type][timestamp] 
                    for asset, signals in pattern_signals.items()
                    if timestamp in signals[signal_type].index
                ])
                metrics[f'{signal_type}_concentration'] = concentration
            
            # Calculate spread rate (change in concentration)
            if timestamp != propagation_metrics.index[0]:
                for signal_type in next(iter(pattern_signals.values())).keys():
                    prev_concentration = propagation_metrics.loc[
                        propagation_metrics.index[
                            propagation_metrics.index.get_loc(timestamp) - 1
                        ],
                        f'{signal_type}_concentration'
                    ]
                    metrics[f'{signal_type}_spread_rate'] = (
                        metrics[f'{signal_type}_concentration'] - prev_concentration
                    )
            
            propagation_metrics.loc[timestamp] = metrics
        
        return propagation_metrics

    def calculate_node_influence(self) -> Dict[str, float]:
        """
        Calculate influence metrics for each node
        
        Returns:
            Dict[str, float]: Node influence scores
        """
        if self.graph is None:
            raise ValueError("Network must be created before calculating influence")
        
        # Calculate various centrality metrics
        degree_cent = nx.degree_centrality(self.graph)
        between_cent = nx.betweenness_centrality(self.graph)
        eigen_cent = nx.eigenvector_centrality_numpy(self.graph)
        
        # Combine metrics into overall influence score
        influence_scores = {}
        for node in self.graph.nodes():
            influence_scores[node] = (
                0.3 * degree_cent[node] +
                0.3 * between_cent[node] +
                0.4 * eigen_cent[node]
            )
        
        return influence_scores
    
    def calculate_pattern_reliability(self, 
                                   pattern_type: str) -> Dict[str, Dict[str, float]]:
        """
        Calculate pattern reliability metrics across the network
        
        Args:
            pattern_type: Type of pattern to analyze
            
        Returns:
            Dict[str, Dict[str, float]]: Reliability metrics for each asset
        """
        pattern_func = getattr(CandlestickPatterns, f'detect_{pattern_type}')
        reliability_metrics = {}
        
        for asset, df in self.data.items():
            try:
                # Detect patterns
                result = pattern_func(df)
                
                # Calculate forward returns
                forward_returns = df['Close'].pct_change().shift(-1)
                
                if isinstance(result, tuple):
                    # Handle bullish/bearish patterns
                    for idx, signal_type in enumerate(['bullish', 'bearish']):
                        pattern_signals = result[idx]
                        if pattern_signals.any():
                            # Calculate success rate
                            if signal_type == 'bullish':
                                success = (forward_returns[pattern_signals] > 0).mean()
                            else:
                                success = (forward_returns[pattern_signals] < 0).mean()
                            
                            # Calculate average return
                            avg_return = forward_returns[pattern_signals].mean()
                            
                            # Calculate risk-adjusted return
                            sharpe = (
                                avg_return / forward_returns[pattern_signals].std()
                                if forward_returns[pattern_signals].std() != 0
                                else 0
                            )
                            
                            reliability_metrics[f"{asset}_{signal_type}"] = {
                                'success_rate': success,
                                'avg_return': avg_return,
                                'sharpe_ratio': sharpe,
                                'occurrence_count': pattern_signals.sum()
                            }
                else:
                    # Handle single signal patterns
                    if result.any():
                        success = (forward_returns[result] > 0).mean()
                        avg_return = forward_returns[result].mean()
                        sharpe = (
                            avg_return / forward_returns[result].std()
                            if forward_returns[result].std() != 0
                            else 0
                        )
                        
                        reliability_metrics[asset] = {
                            'success_rate': success,
                            'avg_return': avg_return,
                            'sharpe_ratio': sharpe,
                            'occurrence_count': result.sum()
                        }
            except Exception as e:
                print(f"Error calculating reliability for {asset}: {str(e)}")
                
        return reliability_metrics

    def create_leading_indicators_network(self, 
                                       min_lag: int = 1,
                                       max_lag: int = 5) -> nx.DiGraph:
        """
        Create directed network based on leading/lagging relationships
        
        Args:
            min_lag: Minimum lag to consider
            max_lag: Maximum lag to consider
            
        Returns:
            nx.DiGraph: Directed network of lead-lag relationships
        """
        # Create directed graph for lead-lag relationships
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.data.keys())
        
        # Calculate returns for all assets
        returns = pd.DataFrame({
            asset: df['Close'].pct_change()
            for asset, df in self.data.items()
        })
        
        # Calculate lead-lag relationships
        for asset1 in self.data.keys():
            for asset2 in self.data.keys():
                if asset1 != asset2:
                    max_corr = 0
                    optimal_lag = 0
                    
                    # Test different lags
                    for lag in range(min_lag, max_lag + 1):
                        corr = returns[asset1].corr(returns[asset2].shift(-lag))
                        if abs(corr) > abs(max_corr):
                            max_corr = corr
                            optimal_lag = lag
                    
                    # Add edge if correlation is significant
                    if abs(max_corr) >= self.config.correlation_threshold:
                        self.graph.add_edge(
                            asset1, asset2,
                            weight=abs(max_corr),
                            lag=optimal_lag,
                            correlation=max_corr
                        )
        
        return self.graph

    def analyze_network_stability(self, 
                                window: Optional[int] = None) -> pd.DataFrame:
        """
        Analyze network stability over time
        
        Args:
            window: Rolling window for stability analysis
            
        Returns:
            pd.DataFrame: Network stability metrics
        """
        window = window or self.config.temporal_window
        stability_metrics = pd.DataFrame(index=self.data[list(self.data.keys())[0]].index)
        
        # Calculate rolling network metrics
        for i in range(window, len(stability_metrics)):
            # Create sub-networks for rolling windows
            current_data = {
                asset: df.iloc[i-window:i]
                for asset, df in self.data.items()
            }
            prev_data = {
                asset: df.iloc[i-window-1:i-1]
                for asset, df in self.data.items()
            }
            
            # Build networks for both windows
            current_network = NetworkBuilder(current_data, self.config)
            current_network.create_correlation_network()
            
            prev_network = NetworkBuilder(prev_data, self.config)
            prev_network.create_correlation_network()
            
            # Calculate stability metrics
            metrics = {}
            
            # Network density change
            current_density = nx.density(current_network.graph)
            prev_density = nx.density(prev_network.graph)
            metrics['density_change'] = current_density - prev_density
            
            # Community structure stability
            current_communities = current_network.detect_communities()
            prev_communities = prev_network.detect_communities()
            
            # Calculate Normalized Mutual Information between community structures
            metrics['community_stability'] = self._calculate_community_similarity(
                current_communities, prev_communities
            )
            
            # Edge weight stability
            metrics['edge_stability'] = self._calculate_edge_stability(
                current_network.graph, prev_network.graph
            )
            
            stability_metrics.loc[stability_metrics.index[i]] = metrics
        
        return stability_metrics

    def _calculate_community_similarity(self,
                                     communities1: Dict[str, int],
                                     communities2: Dict[str, int]) -> float:
        """Calculate similarity between two community structures"""
        nodes = list(set(communities1.keys()) & set(communities2.keys()))
        if not nodes:
            return 0.0
            
        # Create label arrays
        labels1 = [communities1[node] for node in nodes]
        labels2 = [communities2[node] for node in nodes]
        
        # Calculate normalized mutual information
        return float(
            self._normalized_mutual_info_score(labels1, labels2)
        )

    def _normalized_mutual_info_score(self, labels1: List[int], 
                                    labels2: List[int]) -> float:
        """Calculate normalized mutual information between two labelings"""
        labels1 = np.array(labels1)
        labels2 = np.array(labels2)
        
        if len(labels1) != len(labels2):
            return 0.0
            
        # Calculate contingency matrix
        contingency = pd.crosstab(labels1, labels2)
        
        # Calculate marginal probabilities
        pi = contingency.sum(axis=1) / len(labels1)
        pj = contingency.sum(axis=0) / len(labels1)
        
        # Calculate mutual information
        contingency = np.array(contingency)
        pij = contingency / len(labels1)
        mutual_info = np.sum(pij * np.log2(pij / (pi.reshape(-1, 1) @ pj.reshape(1, -1))), where=(pij > 0))
        
        # Calculate entropies
        hi = -np.sum(pi * np.log2(pi, where=(pi > 0)))
        hj = -np.sum(pj * np.log2(pj, where=(pj > 0)))
        
        # Return normalized mutual information
        if hi == 0 or hj == 0:
            return 0.0
        return 2 * mutual_info / (hi + hj)

    def _calculate_edge_stability(self, 
                                graph1: nx.Graph,
                                graph2: nx.Graph) -> float:
        """Calculate stability of edge weights between two networks"""
        common_edges = set(graph1.edges()) & set(graph2.edges())
        if not common_edges:
            return 0.0
            
        weight_diffs = []
        for edge in common_edges:
            weight1 = graph1.edges[edge]['weight']
            weight2 = graph2.edges[edge]['weight']
            weight_diffs.append(abs(weight1 - weight2))
            
        return 1 - (np.mean(weight_diffs) if weight_diffs else 0)

    def identify_systemic_assets(self) -> Dict[str, float]:
        """
        Identify systemically important assets in the network
        
        Returns:
            Dict[str, float]: Systemic importance scores
        """
        if self.graph is None:
            raise ValueError("Network must be created before identifying systemic assets")
            
        # Calculate multiple centrality metrics
        centrality_metrics = {
            'degree': nx.degree_centrality(self.graph),
            'betweenness': nx.betweenness_centrality(self.graph),
            'eigenvector': nx.eigenvector_centrality_numpy(self.graph),
            'closeness': nx.closeness_centrality(self.graph)
        }
        
        # Calculate correlation impact
        correlation_impact = {}
        for node in self.graph.nodes():
            # Calculate average correlation with other nodes
            correlations = []
            for neighbor in self.graph.neighbors(node):
                if 'weight' in self.graph[node][neighbor]:
                    correlations.append(self.graph[node][neighbor]['weight'])
            correlation_impact[node] = np.mean(correlations) if correlations else 0
            
        # Combine metrics into systemic importance score
        systemic_scores = {}
        for node in self.graph.nodes():
            systemic_scores[node] = (
                0.25 * centrality_metrics['degree'][node] +
                0.25 * centrality_metrics['betweenness'][node] +
                0.25 * centrality_metrics['eigenvector'][node] +
                0.25 * correlation_impact[node]
            )
            
        return systemic_scores