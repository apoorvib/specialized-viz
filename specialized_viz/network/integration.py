import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from ..candlestick.patterns import CandlestickPatterns
from .analysis import NetworkBuilder, NetworkConfig

class PatternNetworkIntegration:
    """Class for integrating candlestick pattern analysis with network analysis"""
    
    def __init__(self,
                network_builder: NetworkBuilder,
                lookback_window: int = 100):
        """
        Initialize pattern-network integration
        
        Args:
            network_builder: NetworkBuilder instance
            lookback_window: Window for historical pattern analysis
        """
        self.network_builder = network_builder
        self.lookback_window = lookback_window
        self.patterns = CandlestickPatterns()
        self._pattern_cache = {}
        
    def analyze_pattern_propagation(self,
                                  pattern_type: str,
                                  min_lag: int = 1,
                                  max_lag: int = 5) -> pd.DataFrame:
        """
        Analyze how patterns propagate through the network
        
        Args:
            pattern_type: Type of pattern to analyze
            min_lag: Minimum lag to consider
            max_lag: Maximum lag to consider
            
        Returns:
            pd.DataFrame: Pattern propagation metrics
        """
        # Get pattern detection method
        pattern_func = getattr(self.patterns, f'detect_{pattern_type}')
        
        # Detect patterns for each asset
        pattern_signals = {}
        for asset, df in self.network_builder.data.items():
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
        
        # Calculate lead-lag relationships
        propagation_metrics = pd.DataFrame()
        
        for asset1 in pattern_signals.keys():
            asset_metrics = {}
            
            for asset2 in pattern_signals.keys():
                if asset1 != asset2:
                    # Calculate cross-correlation for different lags
                    max_corr = 0
                    optimal_lag = 0
                    
                    for lag in range(min_lag, max_lag + 1):
                        for signal_type in pattern_signals[asset1].keys():
                            # Align signals with lag
                            sig1 = pattern_signals[asset1][signal_type]
                            sig2 = pattern_signals[asset2][signal_type].shift(-lag)
                            
                            # Calculate correlation
                            corr = pd.Series(sig1).corr(pd.Series(sig2))
                            if abs(corr) > abs(max_corr):
                                max_corr = corr
                                optimal_lag = lag
                    
                    asset_metrics[f'{asset2}_corr'] = max_corr
                    asset_metrics[f'{asset2}_lag'] = optimal_lag
            
            propagation_metrics = pd.concat([
                propagation_metrics,
                pd.DataFrame([asset_metrics], index=[asset1])
            ])
        
        return propagation_metrics
    
    def create_pattern_influence_network(self,
                                       pattern_type: str) -> nx.DiGraph:
        """
        Create directed network based on pattern influence
        
        Args:
            pattern_type: Type of pattern to analyze
            
        Returns:
            nx.DiGraph: Directed network of pattern influences
        """
        # Get propagation metrics
        prop_metrics = self.analyze_pattern_propagation(pattern_type)
        
        # Create directed graph
        influence_network = nx.DiGraph()
        
        # Add nodes
        influence_network.add_nodes_from(prop_metrics.index)
        
        # Add edges based on correlations and lags
        for asset1 in prop_metrics.index:
            for asset2 in prop_metrics.index:
                if asset1 != asset2:
                    corr = prop_metrics.loc[asset1, f'{asset2}_corr']
                    lag = prop_metrics.loc[asset1, f'{asset2}_lag']
                    
                    if abs(corr) >= self.network_builder.config.correlation_threshold:
                        influence_network.add_edge(
                            asset1,
                            asset2,
                            weight=abs(corr),
                            lag=lag,
                            correlation=corr
                        )
        
        return influence_network
    
    def analyze_pattern_communities(self,
                                  pattern_types: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Analyze pattern-based communities
        
        Args:
            pattern_types: List of pattern types to analyze
            
        Returns:
            Dict[str, Dict]: Community analysis for each pattern type
        """
        if pattern_types is None:
            pattern_types = [
                name.replace('detect_', '')
                for name in dir(self.patterns)
                if name.startswith('detect_') and callable(getattr(self.patterns, name))
            ]
        
        community_analysis = {}
        
        for pattern_type in pattern_types:
            try:
                # Create pattern influence network
                influence_network = self.create_pattern_influence_network(pattern_type)
                
                # Detect communities
                communities = self.network_builder.detect_communities(
                    method='louvain'
                )
                
                # Calculate community metrics
                metrics = {
                    'num_communities': len(set(communities.values())),
                    'modularity': self._calculate_modularity(influence_network, communities),
                    'community_sizes': pd.Series(communities).value_counts().to_dict(),
                    'community_members': {
                        com: [node for node, c in communities.items() if c == com]
                        for com in set(communities.values())
                    }
                }
                
                community_analysis[pattern_type] = metrics
                
            except Exception as e:
                print(f"Error analyzing communities for {pattern_type}: {str(e)}")
                continue
        
        return community_analysis
    
    def identify_pattern_leaders(self,
                               pattern_type: str,
                               top_n: int = 5) -> Dict[str, float]:
        """
        Identify assets that lead pattern formation
        
        Args:
            pattern_type: Type of pattern to analyze
            top_n: Number of leading assets to return
            
        Returns:
            Dict[str, float]: Top pattern leaders with scores
        """
        # Create pattern influence network
        influence_network = self.create_pattern_influence_network(pattern_type)
        
        # Calculate leadership scores
        leadership_scores = {}
        
        for node in influence_network.nodes():
            # Factors contributing to leadership:
            # 1. Number of influenced assets (out-degree)
            # 2. Strength of influence (edge weights)
            # 3. Speed of influence (inverse of lags)
            # 4. PageRank centrality
            
            out_degree = influence_network.out_degree(node, weight='weight')
            
            # Calculate average influence strength
            influence_strengths = [
                influence_network[node][target]['weight']
                for target in influence_network.successors(node)
            ]
            avg_strength = np.mean(influence_strengths) if influence_strengths else 0
            
            # Calculate average lag (inverse)
            lags = [
                influence_network[node][target]['lag']
                for target in influence_network.successors(node)
            ]
            avg_lag_inverse = 1 / (np.mean(lags) if lags else float('inf'))
            
            # Calculate PageRank
            pagerank = nx.pagerank(influence_network, weight='weight')
            
            # Combine metrics
            leadership_scores[node] = (
                0.3 * out_degree +
                0.3 * avg_strength +
                0.2 * avg_lag_inverse +
                0.2 * pagerank[node]
            )
        
        # Return top N leaders
        return dict(
            sorted(
                leadership_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_n]
        )
    
    def calculate_pattern_synchronization(self,
                                       pattern_type: str,
                                       window: Optional[int] = None) -> pd.DataFrame:
        """
        Calculate pattern synchronization metrics over time
        
        Args:
            pattern_type: Type of pattern to analyze
            window: Rolling window size
            
        Returns:
            pd.DataFrame: Synchronization metrics
        """
        window = window or self.lookback_window
        pattern_func = getattr(self.patterns, f'detect_{pattern_type}')
        
        # Detect patterns for each asset
        pattern_signals = {}
        for asset, df in self.network_builder.data.items():
            try:
                result = pattern_func(df)
                if isinstance(result, tuple):
                    for idx, signal_type in enumerate(['bullish', 'bearish']):
                        pattern_signals[f'{asset}_{signal_type}'] = result[idx]
                else:
                    pattern_signals[asset] = result
            except Exception as e:
                print(f"Error detecting {pattern_type} for {asset}: {str(e)}")
                continue
        
        # Convert to DataFrame
        signals_df = pd.DataFrame(pattern_signals)
        
        # Calculate synchronization metrics
        sync_metrics = pd.DataFrame(index=signals_df.index)
        
        # Global synchronization
        sync_metrics['global_sync'] = signals_df.mean(axis=1)
        
        # Rolling correlation
        sync_metrics['avg_correlation'] = signals_df.rolling(window).corr().mean()
        
        # Pattern concentration
        sync_metrics['pattern_concentration'] = (
            signals_df.rolling(window).sum() / window
        ).mean(axis=1)
        
        return sync_metrics
    
    def _calculate_modularity(self,
                            graph: nx.Graph,
                            communities: Dict[str, int]) -> float:
        """Calculate modularity of community structure"""
        if not graph.is_directed():
            return nx.algorithms.community.modularity(
                graph,
                [
                    [node for node, com in communities.items() if com == i]
                    for i in range(max(communities.values()) + 1)
                ]
            )
        else:
            # For directed graphs, calculate modified modularity
            m = graph.size(weight='weight')
            if m == 0:
                return 0.0
                
            modularity = 0.0
            for com in set(communities.values()):
                community_nodes = [
                    node for node, c in communities.items() if c == com
                ]
                
                for u in community_nodes:
                    for v in community_nodes:
                        if graph.has_edge(u, v):
                            w_uv = graph[u][v].get('weight', 1.0)
                            k_out_u = graph.out_degree(u, weight='weight')
                            k_in_v = graph.in_degree(v, weight='weight')
                            modularity += w_uv - (k_out_u * k_in_v) / m
                            
            return modularity / (2 * m)