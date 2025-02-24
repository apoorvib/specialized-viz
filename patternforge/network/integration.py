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

    def analyze_pattern_influence_network(self, pattern_type: str) -> Dict:
        """Analyze how patterns influence network structure.
        
        Args:
            pattern_type: Type of pattern to analyze
            
        Returns:
            Dict: Pattern influence analysis results
        """
        results = {
            'node_influence': {},
            'edge_weights': {},
            'communities': {},
            'temporal': pd.DataFrame()
        }
        
        # Calculate pattern occurrences
        pattern_data = self.analyze_pattern_propagation(pattern_type)
        
        # Calculate node influence
        for node in self.network_builder.graph.nodes():
            # Calculate pattern success rate
            if node in pattern_data:
                success_rate = pattern_data[node]['success_rate'].mean()
                propagation_speed = pattern_data[node]['propagation_speed'].mean()
                
                results['node_influence'][node] = {
                    'success_rate': success_rate,
                    'propagation_speed': propagation_speed,
                    'impact_score': success_rate * propagation_speed
                }
        
        # Calculate edge weights based on pattern co-occurrence
        for edge in self.network_builder.graph.edges():
            node1, node2 = edge
            if node1 in pattern_data and node2 in pattern_data:
                # Calculate pattern similarity between nodes
                similarity = pattern_data[node1]['signal'].corr(
                    pattern_data[node2]['signal']
                )
                results['edge_weights'][edge] = similarity
        
        # Analyze community structure
        communities = self.network_builder.detect_communities()
        for community_id in set(communities.values()):
            community_nodes = [
                node for node, comm in communities.items()
                if comm == community_id
            ]
            
            # Calculate community-level pattern metrics
            results['communities'][community_id] = {
                'avg_success_rate': np.mean([
                    results['node_influence'][node]['success_rate']
                    for node in community_nodes
                    if node in results['node_influence']
                ]),
                'avg_propagation_speed': np.mean([
                    results['node_influence'][node]['propagation_speed']
                    for node in community_nodes
                    if node in results['node_influence']
                ])
            }
        
        # Analyze temporal aspects
        timestamps = pattern_data[next(iter(pattern_data))].index
        results['temporal'] = pd.DataFrame(index=timestamps)
        
        results['temporal']['global_success_rate'] = np.mean([
            data['success_rate'] for data in pattern_data.values()
        ], axis=0)
        
        results['temporal']['community_coherence'] = self._calculate_community_coherence(
            pattern_data, communities, timestamps
        )
        
        return results

    def _calculate_community_coherence(self,
                                    pattern_data: Dict,
                                    communities: Dict,
                                    timestamps: pd.DatetimeIndex) -> pd.Series:
        """Calculate how coherently patterns appear within communities."""
        coherence = pd.Series(index=timestamps, dtype=float)
        
        for timestamp in timestamps:
            community_signals = defaultdict(list)
            
            # Group pattern signals by community
            for node, data in pattern_data.items():
                if node in communities:
                    community_id = communities[node]
                    signal = data['signal'][timestamp]
                    community_signals[community_id].append(signal)
            
            # Calculate average intra-community correlation
            correlations = []
            for signals in community_signals.values():
                if len(signals) > 1:
                    correlation = np.corrcoef(signals)[0, 1]
                    correlations.append(correlation)
            
            coherence[timestamp] = np.mean(correlations) if correlations else 0
        
        return coherence

    def create_pattern_impact_network(self, 
                                    pattern_type: str,
                                    time_window: int = 30) -> nx.DiGraph:
        """Create directed network showing pattern impact propagation.
        
        Args:
            pattern_type: Type of pattern to analyze
            time_window: Window for temporal analysis
            
        Returns:
            nx.DiGraph: Directed network of pattern impacts
        """
        # Create directed graph
        impact_network = nx.DiGraph()
        
        # Get pattern data
        pattern_data = self.analyze_pattern_propagation(pattern_type)
        
        # Calculate temporal relationships
        temporal_effects = self._calculate_temporal_effects(
            pattern_data, time_window
        )
        
        # Add nodes and edges based on impact
        for source, impacts in temporal_effects.items():
            impact_network.add_node(
                source,
                pattern_strength=pattern_data[source]['strength'].mean()
            )
            
            for target, impact in impacts.items():
                if impact['significance'] > 0.05:  # Significance threshold
                    impact_network.add_edge(
                        source,
                        target,
                        weight=impact['effect_size'],
                        lag=impact['lag']
                    )
        
        return impact_network

    def _calculate_temporal_effects(self,
                                pattern_data: Dict,
                                time_window: int) -> Dict:
        """Calculate temporal effects between nodes."""
        effects = {}
        
        for source_node, source_data in pattern_data.items():
            effects[source_node] = {}
            
            for target_node, target_data in pattern_data.items():
                if source_node != target_node:
                    # Calculate lagged correlations
                    max_effect = 0
                    optimal_lag = 0
                    min_pvalue = 1.0
                    
                    for lag in range(1, time_window + 1):
                        lagged_corr = source_data['signal'].corr(
                            target_data['signal'].shift(-lag)
                        )
                        
                        if abs(lagged_corr) > abs(max_effect):
                            max_effect = lagged_corr
                            optimal_lag = lag
                            
                            # Calculate significance
                            min_pvalue = self._calculate_correlation_significance(
                                source_data['signal'],
                                target_data['signal'].shift(-lag)
                            )
                    
                    effects[source_node][target_node] = {
                        'effect_size': max_effect,
                        'lag': optimal_lag,
                        'significance': min_pvalue
                    }
        
        return effects

    def _calculate_correlation_significance(self,
                                        series1: pd.Series,
                                        series2: pd.Series) -> float:
        """Calculate significance of correlation between two series."""
        # Remove NaN values
        valid_data = pd.concat([series1, series2], axis=1).dropna()
        
        if len(valid_data) < 2:
            return 1.0
            
        # Calculate correlation coefficient and p-value
        correlation, p_value = stats.pearsonr(
            valid_data.iloc[:, 0],
            valid_data.iloc[:, 1]
        )
        
        return p_value

    def _create_edge_trace_with_patterns(self, pos: Dict, pattern_metrics: Dict) -> go.Scatter:
        """Create edge trace with pattern-based properties."""
        edge_x = []
        edge_y = []
        edge_colors = []
        edge_widths = []
        edge_text = []
        
        for edge in self.network_builder.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # Calculate pattern similarity between nodes
            similarity = self._calculate_pattern_similarity(
                pattern_metrics[edge[0]],
                pattern_metrics[edge[1]]
            )
            edge_colors.extend([similarity, similarity, None])
            
            # Calculate edge width based on combined pattern strength
            strength = self._calculate_combined_pattern_strength(
                pattern_metrics[edge[0]],
                pattern_metrics[edge[1]]
            )
            edge_widths.extend([strength, strength, None])
            
            edge_text.append(
                f"Pattern Similarity: {similarity:.2f}<br>" +
                f"Combined Strength: {strength:.2f}"
            )
        
        return go.Scatter(
            x=edge_x,
            y=edge_y,
            mode='lines',
            line=dict(
                width=edge_widths,
                color=edge_colors,
                colorscale='RdYlBu'
            ),
            hovertext=edge_text,
            hoverinfo='text'
        )

    def _calculate_pattern_similarity(self, 
                                    patterns1: Dict,
                                    patterns2: Dict) -> float:
        """Calculate similarity between pattern occurrences."""
        common_patterns = set(patterns1.keys()) & set(patterns2.keys())
        if not common_patterns:
            return 0
            
        similarities = []
        for pattern in common_patterns:
            corr = np.corrcoef(
                [patterns1[pattern], patterns2[pattern]]
            )[0, 1]
            similarities.append(corr)
        
        return np.mean(similarities)

    def _calculate_combined_pattern_strength(self,
                                        patterns1: Dict,
                                        patterns2: Dict) -> float:
        """Calculate combined pattern strength between nodes."""
        strength1 = sum(patterns1.values())
        strength2 = sum(patterns2.values())
        return (strength1 + strength2) / 2

    def _get_network_relationships(self) -> Dict[str, List[str]]:
        """Get related nodes based on network structure."""
        relationships = {}
        
        # Calculate network metrics
        centrality = nx.eigenvector_centrality_numpy(self.network_builder.graph)
        communities = self.network_builder.detect_communities()
        
        for node in self.network_builder.graph.nodes():
            related = []
            
            # Add neighbors
            related.extend(self.network_builder.graph.neighbors(node))
            
            # Add nodes in same community
            node_community = communities[node]
            community_nodes = [
                n for n, c in communities.items()
                if c == node_community and n != node
            ]
            related.extend(community_nodes)
            
            # Add nodes with similar centrality
            node_centrality = centrality[node]
            similar_centrality = [
                n for n, c in centrality.items()
                if abs(c - node_centrality) < 0.1 and n != node
            ]
            related.extend(similar_centrality)
            
            relationships[node] = list(set(related))
        
        return relationships

    def _create_node_forecast(self,
                            node: str,
                            related_nodes: List[str],
                            horizon: int) -> pd.DataFrame:
        """Create forecast for a node using network and time series data."""
        # Get node data
        node_data = self.network_builder.data[node]
        
        # Get related nodes' data
        related_data = {
            n: self.network_builder.data[n]
            for n in related_nodes
        }
        
        # Create features from related nodes
        features = self._create_forecast_features(node_data, related_data)
        
        # Get time series decomposition
        decomposition = self.timeseries_analyzer.decompose(node_data['Close'])
        
        # Combine forecasts from different methods
        forecasts = pd.DataFrame()
        
        # Network-based forecast
        network_forecast = self._create_network_based_forecast(
            features, horizon
        )
        forecasts['network'] = network_forecast
        
        # Time series forecast
        ts_forecast = self._create_timeseries_forecast(
            decomposition, horizon
        )
        forecasts['timeseries'] = ts_forecast
        
        # Combined forecast (weighted average)
        forecasts['combined'] = (
            0.6 * forecasts['network'] +
            0.4 * forecasts['timeseries']
        )
        
        return forecasts

    def _create_forecast_features(self,
                                node_data: pd.DataFrame,
                                related_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create features for forecasting from network data."""
        features = pd.DataFrame(index=node_data.index)
        
        # Add lagged features
        for lag in [1, 5, 10, 20]:
            features[f'lag_{lag}'] = node_data['Close'].shift(lag)
        
        # Add related nodes' features
        for name, data in related_data.items():
            # Add concurrent and lagged values
            features[f'{name}_concurrent'] = data['Close']
            features[f'{name}_lag_1'] = data['Close'].shift(1)
            
            # Add correlation
            rolling_corr = node_data['Close'].rolling(20).corr(data['Close'])
            features[f'{name}_correlation'] = rolling_corr
        
        return features

    def _create_network_based_forecast(self,
                                    features: pd.DataFrame,
                                    horizon: int) -> pd.Series:
        """Create forecast based on network relationships."""
        from sklearn.ensemble import GradientBoostingRegressor
        
        # Prepare training data
        X = features.dropna()
        y = self.network_builder.data[node]['Close'].loc[X.index]
        
        # Train model
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3
        )
        model.fit(X, y)
        
        # Generate forecast
        forecast = pd.Series(index=pd.date_range(
            start=X.index[-1],
            periods=horizon + 1,
            freq='D'
        )[1:])
        
        # Iterative forecasting
        last_features = X.iloc[-1:]
        for i in range(horizon):
            # Make prediction
            pred = model.predict(last_features)
            forecast.iloc[i] = pred[0]
            
            # Update features for next prediction
            last_features = self._update_forecast_features(
                last_features,
                pred[0],
                features.columns
            )
        
        return forecast

    def _analyze_pattern_components(self,
                                patterns: Dict,
                                decomposition: Dict) -> Dict:
        """Analyze relationship between patterns and time series components."""
        metrics = {}
        
        # Calculate correlation with trend
        trend_corr = pd.Series(patterns).corr(decomposition['trend'])
        metrics['trend_correlation'] = trend_corr
        
        # Calculate correlation with seasonal component
        seasonal_corr = pd.Series(patterns).corr(decomposition['seasonal'])
        metrics['seasonal_correlation'] = seasonal_corr
        
        # Calculate correlation with residuals
        residual_corr = pd.Series(patterns).corr(decomposition['residual'])
        metrics['residual_correlation'] = residual_corr
        
        # Calculate pattern timing relative to components
        metrics['trend_alignment'] = self._calculate_component_alignment(
            patterns, decomposition['trend']
        )
        metrics['seasonal_alignment'] = self._calculate_component_alignment(
            patterns, decomposition['seasonal']
        )
        
        return metrics

    def _calculate_component_alignment(self,
                                    patterns: Dict,
                                    component: pd.Series) -> float:
        """Calculate how well patterns align with component peaks/troughs."""
        # Find peaks and troughs in component
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(component)
        troughs, _ = find_peaks(-component)
        
        # Calculate alignment score
        pattern_times = pd.Series(patterns).index[patterns > 0]
        
        alignment_scores = []
        for pattern_time in pattern_times:
            # Find nearest peak/trough
            peak_distance = min(abs(peaks - pattern_time.value))
            trough_distance = min(abs(troughs - pattern_time.value))
            
            # Take minimum distance to either peak or trough
            min_distance = min(peak_distance, trough_distance)
            alignment_scores.append(1 / (1 + min_distance))
        
        return np.mean(alignment_scores) if alignment_scores else 0