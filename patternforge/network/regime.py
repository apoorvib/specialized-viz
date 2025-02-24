import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from ..candlestick.visualization import VisualizationConfig
from .analysis import NetworkBuilder, NetworkConfig
from dataclasses import dataclass

@dataclass
class MarketRegime:
    """Class for storing market regime information"""
    name: str
    volatility: str  # 'low', 'medium', 'high'
    trend: str      # 'up', 'down', 'sideways'
    volume: str     # 'low', 'medium', 'high'
    start_date: pd.Timestamp
    end_date: Optional[pd.Timestamp] = None

class RegimeNetworkAnalyzer:
    """Class for analyzing network structure across different market regimes"""
    
    def __init__(self,
                network_builder: NetworkBuilder,
                config: Optional[NetworkConfig] = None):
        """
        Initialize regime network analyzer
        
        Args:
            network_builder: NetworkBuilder instance
            config: Network configuration
        """
        self.network_builder = network_builder
        self.config = config or NetworkConfig()
        
    def detect_regime_transitions(self,
                                window: int = 20) -> List[MarketRegime]:
        """
        Detect market regime transitions
        
        Args:
            window: Rolling window for regime detection
            
        Returns:
            List[MarketRegime]: Detected market regimes
        """
        regimes = []
        
        # Calculate market-wide metrics
        metrics = pd.DataFrame()
        
        # Calculate average returns and volatility
        for asset, df in self.network_builder.data.items():
            returns = df['Close'].pct_change()
            metrics[f'{asset}_returns'] = returns
            metrics[f'{asset}_volatility'] = returns.rolling(window).std()
            metrics[f'{asset}_volume'] = df['Volume'].pct_change()
        
        # Calculate market-wide metrics
        market_metrics = pd.DataFrame()
        market_metrics['avg_returns'] = metrics.filter(like='_returns').mean(axis=1)
        market_metrics['avg_volatility'] = metrics.filter(like='_volatility').mean(axis=1)
        market_metrics['avg_volume'] = metrics.filter(like='_volume').mean(axis=1)
        
        # Classify regimes
        market_metrics['volatility_regime'] = pd.qcut(
            market_metrics['avg_volatility'],
            q=3,
            labels=['low', 'medium', 'high']
        )
        
        # Classify trend using rolling returns
        trend = market_metrics['avg_returns'].rolling(window).mean()
        market_metrics['trend_regime'] = pd.qcut(
            trend,
            q=3,
            labels=['down', 'sideways', 'up']
        )
        
        market_metrics['volume_regime'] = pd.qcut(
            market_metrics['avg_volume'],
            q=3,
            labels=['low', 'medium', 'high']
        )
        
        # Detect regime changes
        regime_changes = (
            (market_metrics['volatility_regime'].shift() != market_metrics['volatility_regime']) |
            (market_metrics['trend_regime'].shift() != market_metrics['trend_regime']) |
            (market_metrics['volume_regime'].shift() != market_metrics['volume_regime'])
        )
        
        change_points = market_metrics.index[regime_changes]
        
        # Create regime objects
        for i in range(len(change_points)):
            start_date = change_points[i]
            end_date = change_points[i + 1] if i + 1 < len(change_points) else None
            
            regimes.append(MarketRegime(
                name=f"Regime_{i}",
                volatility=market_metrics.loc[start_date, 'volatility_regime'],
                trend=market_metrics.loc[start_date, 'trend_regime'],
                volume=market_metrics.loc[start_date, 'volume_regime'],
                start_date=start_date,
                end_date=end_date
            ))
        
        return regimes
    
    def analyze_regime_networks(self,
                              regimes: List[MarketRegime]) -> Dict[str, nx.Graph]:
        """
        Create and analyze networks for different market regimes
        
        Args:
            regimes: List of detected market regimes
            
        Returns:
            Dict[str, nx.Graph]: Networks for each regime
        """
        regime_networks = {}
        
        for regime in regimes:
            # Filter data for regime period
            regime_data = {}
            for asset, df in self.network_builder.data.items():
                if regime.end_date:
                    mask = (df.index >= regime.start_date) & (df.index < regime.end_date)
                else:
                    mask = df.index >= regime.start_date
                regime_data[asset] = df[mask]
            
            # Create network for this regime
            regime_network = NetworkBuilder(regime_data, self.config)
            regime_network.create_correlation_network()
            
            regime_networks[regime.name] = regime_network.graph
            
        return regime_networks
    
    def compare_regime_networks(self,
                              regime_networks: Dict[str, nx.Graph]) -> pd.DataFrame:
        """
        Compare network properties across different regimes
        
        Args:
            regime_networks: Dictionary of regime networks
            
        Returns:
            pd.DataFrame: Comparison metrics for each regime
        """
        comparison = pd.DataFrame()
        
        for regime_name, network in regime_networks.items():
            metrics = {}
            
            # Basic network metrics
            metrics['density'] = nx.density(network)
            metrics['avg_clustering'] = nx.average_clustering(network)
            metrics['avg_shortest_path'] = nx.average_shortest_path_length(network)
            
            # Centralization metrics
            degree_cent = nx.degree_centrality(network)
            betweenness_cent = nx.betweenness_centrality(network)
            eigenvector_cent = nx.eigenvector_centrality_numpy(network)
            
            metrics['degree_centralization'] = self._calculate_centralization(degree_cent)
            metrics['betweenness_centralization'] = self._calculate_centralization(betweenness_cent)
            metrics['eigenvector_centralization'] = self._calculate_centralization(eigenvector_cent)
            
            # Community structure
            communities = self.network_builder.detect_communities(method='louvain')
            metrics['num_communities'] = len(set(communities.values()))
            metrics['modularity'] = self._calculate_modularity(network, communities)
            
            comparison = pd.concat([
                comparison,
                pd.DataFrame([metrics], index=[regime_name])
            ])
        
        return comparison
    
    def analyze_regime_transitions(self,
                                 regimes: List[MarketRegime],
                                 regime_networks: Dict[str, nx.Graph]) -> pd.DataFrame:
        """
        Analyze network changes during regime transitions
        
        Args:
            regimes: List of market regimes
            regime_networks: Dictionary of regime networks
            
        Returns:
            pd.DataFrame: Transition analysis metrics
        """
        transitions = pd.DataFrame()
        
        for i in range(len(regimes) - 1):
            from_regime = regimes[i]
            to_regime = regimes[i + 1]
            
            metrics = {}
            
            # Calculate network similarity
            from_network = regime_networks[from_regime.name]
            to_network = regime_networks[to_regime.name]
            
            metrics['edge_similarity'] = self._calculate_edge_similarity(
                from_network, to_network
            )
            
            # Calculate community stability
            from_communities = self.network_builder.detect_communities(
                method='louvain', G=from_network
            )
            to_communities = self.network_builder.detect_communities(
                method='louvain', G=to_network
            )
            
            metrics['community_stability'] = self._calculate_community_similarity(
                from_communities, to_communities
            )
            
            # Calculate centrality changes
            metrics.update(
                self._calculate_centrality_changes(from_network, to_network)
            )
            
            transitions = pd.concat([
                transitions,
                pd.DataFrame(
                    [metrics],
                    index=[f"{from_regime.name}_to_{to_regime.name}"]
                )
            ])
        
        return transitions
    
    def identify_regime_indicators(self,
                                 regimes: List[MarketRegime],
                                 window: int = 20) -> Dict[str, List[str]]:
        """
        Identify network-based indicators of regime changes
        
        Args:
            regimes: List of market regimes
            window: Window for indicator calculation
            
        Returns:
            Dict[str, List[str]]: Leading indicators for each regime
        """
        indicators = {}
        
        for regime in regimes:
            # Calculate pre-regime network metrics
            start_idx = self.network_builder.data[
                list(self.network_builder.data.keys())[0]
            ].index.get_loc(regime.start_date)
            
            if start_idx >= window:
                pre_regime_data = {}
                for asset, df in self.network_builder.data.items():
                    pre_regime_data[asset] = df.iloc[start_idx - window:start_idx]
                
                # Create pre-regime network
                pre_network = NetworkBuilder(pre_regime_data, self.config)
                pre_network.create_correlation_network()
                
                # Analyze network properties
                leading_indicators = []
                
                # Check for density changes
                if nx.density(pre_network.graph) > 0.7:
                    leading_indicators.append("high_network_density")
                
                # Check for clustering changes
                if nx.average_clustering(pre_network.graph) > 0.6:
                    leading_indicators.append("high_clustering")
                
                # Check for centralization
                degree_cent = nx.degree_centrality(pre_network.graph)
                if self._calculate_centralization(degree_cent) > 0.5:
                    leading_indicators.append("high_centralization")
                
                # Check for community structure
                communities = self.network_builder.detect_communities(
                    method='louvain', G=pre_network.graph
                )
                if len(set(communities.values())) <= 2:
                    leading_indicators.append("consolidated_communities")
                
                indicators[regime.name] = leading_indicators
            
        return indicators
    
    def create_regime_stability_network(self,
                                      regimes: List[MarketRegime]) -> nx.Graph:
        """
        Create network representing regime stability relationships
        
        Args:
            regimes: List of market regimes
            
        Returns:
            nx.Graph: Regime stability network
        """
        stability_network = nx.Graph()
        
        # Add nodes for each regime
        for regime in regimes:
            stability_network.add_node(
                regime.name,
                volatility=regime.volatility,
                trend=regime.trend,
                volume=regime.volume
            )
        
        # Add edges based on regime similarity
        for i, regime1 in enumerate(regimes):
            for regime2 in regimes[i+1:]:
                similarity = self._calculate_regime_similarity(regime1, regime2)
                if similarity > 0.5:  # Threshold for similarity
                    stability_network.add_edge(
                        regime1.name,
                        regime2.name,
                        weight=similarity
                    )
        
        return stability_network
    
    def _calculate_centralization(self,
                                centrality_dict: Dict[str, float]) -> float:
        """Calculate network centralization from centrality scores"""
        values = list(centrality_dict.values())
        max_cent = max(values)
        
        if max_cent == 0:
            return 0.0
            
        n = len(values)
        theoretical_max = (n - 1) * (n - 2)
        
        centralization = sum(max_cent - v for v in values) / theoretical_max
        return centralization
    
    def _calculate_modularity(self,
                            graph: nx.Graph,
                            communities: Dict[str, int]) -> float:
        """Calculate modularity of community structure"""
        return nx.algorithms.community.modularity(
            graph,
            [
                [node for node, com in communities.items() if com == i]
                for i in range(max(communities.values()) + 1)
            ]
        )
    
    def _calculate_edge_similarity(self,
                                 graph1: nx.Graph,
                                 graph2: nx.Graph) -> float:
        """Calculate similarity between two networks' edge sets"""
        edges1 = set(graph1.edges())
        edges2 = set(graph2.edges())
        
        intersection = len(edges1 & edges2)
        union = len(edges1 | edges2)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_community_similarity(self,
                                     communities1: Dict[str, int],
                                     communities2: Dict[str, int]) -> float:
        """Calculate similarity between community structures"""
        nodes = list(set(communities1.keys()) & set(communities2.keys()))
        if not nodes:
            return 0.0
            
        matches = sum(
            communities1[node] == communities2[node]
            for node in nodes
        )
        
        return matches / len(nodes)
    
    def _calculate_centrality_changes(self,
                                    graph1: nx.Graph,
                                    graph2: nx.Graph) -> Dict[str, float]:
        """Calculate changes in centrality metrics between networks"""
        changes = {}
        
        # Calculate centrality metrics for both networks
        metrics = {
            'degree': (nx.degree_centrality, 'degree_change'),
            'betweenness': (nx.betweenness_centrality, 'betweenness_change'),
            'eigenvector': (nx.eigenvector_centrality_numpy, 'eigenvector_change')
        }
        
        for metric_name, (metric_func, change_name) in metrics.items():
            cent1 = metric_func(graph1)
            cent2 = metric_func(graph2)
            
            # Calculate average absolute change
            common_nodes = set(cent1.keys()) & set(cent2.keys())
            if common_nodes:
                avg_change = np.mean([
                    abs(cent2[node] - cent1[node])
                    for node in common_nodes
                ])
                changes[change_name] = avg_change
            else:
                changes[change_name] = 0.0
        
        return changes
    
    def _calculate_regime_similarity(self,
                                   regime1: MarketRegime,
                                   regime2: MarketRegime) -> float:
        """Calculate similarity between two market regimes"""
        similarity = 0.0
        
        # Compare volatility
        if regime1.volatility == regime2.volatility:
            similarity += 0.4
        elif abs(
            ['low', 'medium', 'high'].index(regime1.volatility) -
            ['low', 'medium', 'high'].index(regime2.volatility)
        ) == 1:
            similarity += 0.2
        
        # Compare trend
        if regime1.trend == regime2.trend:
            similarity += 0.4
        elif abs(
            ['down', 'sideways', 'up'].index(regime1.trend) -
            ['down', 'sideways', 'up'].index(regime2.trend)
        ) == 1:
            similarity += 0.2
        
        # Compare volume
        if regime1.volume == regime2.volume:
            similarity += 0.2
        elif abs(
            ['low', 'medium', 'high'].index(regime1.volume) -
            ['low', 'medium', 'high'].index(regime2.volume)
        ) == 1:
            similarity += 0.1
        
        return similarity