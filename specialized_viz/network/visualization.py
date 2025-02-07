"""Enhanced network visualization module for specialized-viz library - Part 1."""

import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from ..candlestick.visualization import VisualizationConfig
from .analysis import NetworkBuilder, NetworkConfig
from .integration import PatternNetworkIntegration

@dataclass
class NetworkVisualizationConfig(VisualizationConfig):
    """Extended configuration for network visualization settings."""
    node_size_range: Tuple[int, int] = (10, 50)
    edge_width_range: Tuple[float, float] = (0.5, 5.0)
    show_labels: bool = True
    label_size: int = 10
    animation_duration: int = 500
    layout_iterations: int = 100
    layout_seed: int = 42
    
    def __post_init__(self):
        super().__post_init__()
        if not hasattr(self, 'color_scheme'):
            self.color_scheme.update({
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
            })

class NetworkVisualizer:
    """Enhanced network visualization class with advanced capabilities."""
    
    def __init__(self, 
                 network_builder: NetworkBuilder,
                 pattern_integration: Optional[PatternNetworkIntegration] = None,
                 config: Optional[NetworkVisualizationConfig] = None):
        """
        Initialize network visualizer
        
        Args:
            network_builder: NetworkBuilder instance
            pattern_integration: Optional PatternNetworkIntegration instance
            config: Optional visualization configuration
        """
        self.network_builder = network_builder
        self.pattern_integration = pattern_integration
        self.config = config or NetworkVisualizationConfig()
        
    def create_comprehensive_view(self) -> go.Figure:
        """Create comprehensive network visualization dashboard."""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Network Structure',
                'Community Detection',
                'Pattern Distribution',
                'Node Centrality',
                'Edge Weight Distribution',
                'Time Evolution'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'scatter'}],
                [{'type': 'bar'}, {'type': 'scatter3d'}],
                [{'type': 'histogram'}, {'type': 'scatter'}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # 1. Network Structure
        self._add_network_structure(fig, row=1, col=1)
        
        # 2. Community Detection
        self._add_community_structure(fig, row=1, col=2)
        
        # 3. Pattern Distribution
        if self.pattern_integration:
            self._add_pattern_distribution(fig, row=2, col=1)
            
        # 4. Node Centrality Analysis
        self._add_centrality_analysis(fig, row=2, col=2)
        
        # 5. Edge Weight Distribution
        self._add_edge_analysis(fig, row=3, col=1)
        
        # 6. Time Evolution
        self._add_time_evolution(fig, row=3, col=2)
        
        # Update layout
        fig.update_layout(
            height=1200,
            width=1600,
            title_text="Comprehensive Network Analysis",
            template="plotly_white",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
        
    def create_pattern_network_view(self, pattern_type: str) -> go.Figure:
        """Create visualization focused on pattern propagation in network."""
        if not self.pattern_integration:
            raise ValueError("Pattern integration required for pattern network view")
            
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Pattern Influence Network',
                'Pattern Synchronization',
                'Leading Indicators',
                'Pattern Propagation'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'heatmap'}],
                [{'type': 'bar'}, {'type': 'scatter'}]
            ]
        )
        
        # 1. Pattern Influence Network
        influence_network = self.pattern_integration.create_pattern_influence_network(
            pattern_type
        )
        self._add_directed_network(
            fig, influence_network,
            row=1, col=1
        )
        
        # 2. Pattern Synchronization
        sync_metrics = self.pattern_integration.calculate_pattern_synchronization(
            pattern_type
        )
        self._add_synchronization_heatmap(
            fig, sync_metrics,
            row=1, col=2
        )
        
        # 3. Leading Indicators
        leaders = self.pattern_integration.identify_pattern_leaders(
            pattern_type
        )
        fig.add_trace(
            go.Bar(
                x=list(leaders.keys()),
                y=list(leaders.values()),
                name='Pattern Leaders'
            ),
            row=2, col=1
        )
        
        # 4. Pattern Propagation
        prop_metrics = self.pattern_integration.analyze_pattern_propagation(
            pattern_type
        )
        self._add_propagation_analysis(
            fig, prop_metrics,
            row=2, col=2
        )
        
        fig.update_layout(
            height=1000,
            width=1400,
            title_text=f"Pattern Network Analysis - {pattern_type}",
            template="plotly_white"
        )
        
        return fig
        
    def create_regime_network_view(self, regimes: List[Dict]) -> go.Figure:
        """Create visualization of network behavior across market regimes."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Regime Transition Network',
                'Network Metrics by Regime',
                'Community Stability',
                'Node Role Evolution'
            )
        )
        
        # Implementation of regime network visualization
        # (Adding specific regime visualization logic)
        
        return fig
    
    def create_interactive_dashboard(self) -> go.Figure:
        """Create interactive dashboard with filtering and exploration capabilities."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Network Explorer',
                'Metric Analysis',
                'Time Series View',
                'Pattern Analysis'
            )
        )
        
        # Implementation of interactive dashboard
        # (Adding interactive elements and controls)
        
        return fig

    def _add_network_structure(self, 
                             fig: go.Figure,
                             row: int,
                             col: int) -> None:
        """Add basic network structure visualization."""
        if self.network_builder.graph is None:
            return
            
        pos = nx.spring_layout(
            self.network_builder.graph,
            k=1/np.sqrt(len(self.network_builder.graph.nodes())),
            iterations=self.config.layout_iterations,
            seed=self.config.layout_seed
        )
        
        # Add edges
        edge_trace = self._create_edge_trace(pos)
        fig.add_trace(edge_trace, row=row, col=col)
        
        # Add nodes
        node_trace = self._create_node_trace(pos)
        fig.add_trace(node_trace, row=row, col=col)
        
        # Update layout for this subplot
        fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False,
                        row=row, col=col)
        fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False,
                        row=row, col=col)
                        
    def _create_edge_trace(self, pos: Dict) -> go.Scatter:
        """Create edge trace for network visualization."""
        edge_x = []
        edge_y = []
        edge_weights = []
        
        for edge in self.network_builder.graph.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(edge[2].get('weight', 1.0))
            
        return go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(
                width=1,
                color=self.config.color_scheme['edge_default']
            ),
            hoverinfo='none',
            mode='lines',
            name='Edges'
        )
        
    def _create_node_trace(self, pos: Dict) -> go.Scatter:
        """Create node trace for network visualization."""
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        
        for node in self.network_builder.graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Node text with metrics
            degree = self.network_builder.graph.degree(node)
            centrality = nx.eigenvector_centrality_numpy(self.network_builder.graph)[node]
            node_text.append(
                f"Node: {node}<br>" +
                f"Degree: {degree}<br>" +
                f"Centrality: {centrality:.3f}"
            )
            
            # Node size based on degree
            size = self._scale_node_size(degree)
            node_size.append(size)
            
        return go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text' if self.config.show_labels else 'markers',
            hoverinfo='text',
            text=[str(node) for node in self.network_builder.graph.nodes()]
                if self.config.show_labels else None,
            textposition='top center',
            hovertext=node_text,
            marker=dict(
                size=node_size,
                color=self.config.color_scheme['node_default'],
                line_width=2
            ),
            name='Nodes'
        )

# Importing all needed dependencies from part 1
from .network_visualizer_part1 import NetworkVisualizer, NetworkVisualizationConfig

class NetworkVisualizerExtension(NetworkVisualizer):
    """Extension of NetworkVisualizer with additional visualization methods."""
        
    def _add_community_structure(self,
                               fig: go.Figure,
                               row: int,
                               col: int) -> None:
        """Add community structure visualization."""
        if self.network_builder.graph is None:
            return
            
        # Detect communities
        communities = self.network_builder.detect_communities()
        
        # Get position layout
        pos = nx.spring_layout(
            self.network_builder.graph,
            k=1/np.sqrt(len(self.network_builder.graph.nodes())),
            iterations=self.config.layout_iterations,
            seed=self.config.layout_seed
        )
        
        # Add edges
        edge_trace = self._create_edge_trace(pos)
        fig.add_trace(edge_trace, row=row, col=col)
        
        # Add nodes colored by community
        for community_id in set(communities.values()):
            community_nodes = [
                node for node, com in communities.items()
                if com == community_id
            ]
            
            node_x = []
            node_y = []
            node_text = []
            node_size = []
            
            for node in community_nodes:
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                degree = self.network_builder.graph.degree(node)
                centrality = nx.eigenvector_centrality_numpy(
                    self.network_builder.graph
                )[node]
                
                node_text.append(
                    f"Node: {node}<br>" +
                    f"Community: {community_id}<br>" +
                    f"Degree: {degree}<br>" +
                    f"Centrality: {centrality:.3f}"
                )
                
                size = self._scale_node_size(degree)
                node_size.append(size)
                
            fig.add_trace(
                go.Scatter(
                    x=node_x,
                    y=node_y,
                    mode='markers+text' if self.config.show_labels else 'markers',
                    text=[str(node) for node in community_nodes]
                        if self.config.show_labels else None,
                    textposition='top center',
                    hovertext=node_text,
                    marker=dict(
                        size=node_size,
                        color=self.config.color_scheme['community_colors'][
                            community_id % len(self.config.color_scheme['community_colors'])
                        ],
                        line_width=2
                    ),
                    name=f'Community {community_id}'
                ),
                row=row, col=col
            )
            
        fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False,
                        row=row, col=col)
        fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False,
                        row=row, col=col)

    def _add_time_evolution(self,
                           fig: go.Figure,
                           row: int,
                           col: int) -> None:
        """Add network evolution over time visualization."""
        if not hasattr(self.network_builder, 'temporal_networks'):
            return
            
        # Calculate network metrics over time
        metrics = pd.DataFrame(index=self.network_builder.temporal_networks.keys())
            
# """Enhanced network visualization module for specialized-viz library - Part 2."""

# # Importing all needed dependencies from part 1
# from .network_visualizer_part1 import NetworkVisualizer, NetworkVisualizationConfig

# class NetworkVisualizerExtension(NetworkVisualizer):
#     """Extension of NetworkVisualizer with additional visualization methods."""
        
    def _add_community_structure(self,
                               fig: go.Figure,
                               row: int,
                               col: int) -> None:
        """Add community structure visualization."""
        if self.network_builder.graph is None:
            return
            
        # Detect communities
        communities = self.network_builder.detect_communities()
        
        # Get position layout
        pos = nx.spring_layout(
            self.network_builder.graph,
            k=1/np.sqrt(len(self.network_builder.graph.nodes())),
            iterations=self.config.layout_iterations,
            seed=self.config.layout_seed
        )
        
        # Add edges
        edge_trace = self._create_edge_trace(pos)
        fig.add_trace(edge_trace, row=row, col=col)
        
        # Add nodes colored by community
        for community_id in set(communities.values()):
            community_nodes = [
                node for node, com in communities.items()
                if com == community_id
            ]
            
            node_x = []
            node_y = []
            node_text = []
            node_size = []
            
            for node in community_nodes:
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                degree = self.network_builder.graph.degree(node)
                centrality = nx.eigenvector_centrality_numpy(
                    self.network_builder.graph
                )[node]
                
                node_text.append(
                    f"Node: {node}<br>" +
                    f"Community: {community_id}<br>" +
                    f"Degree: {degree}<br>" +
                    f"Centrality: {centrality:.3f}"
                )
                
                size = self._scale_node_size(degree)
                node_size.append(size)
                
            fig.add_trace(
                go.Scatter(
                    x=node_x,
                    y=node_y,
                    mode='markers+text' if self.config.show_labels else 'markers',
                    text=[str(node) for node in community_nodes]
                        if self.config.show_labels else None,
                    textposition='top center',
                    hovertext=node_text,
                    marker=dict(
                        size=node_size,
                        color=self.config.color_scheme['community_colors'][
                            community_id % len(self.config.color_scheme['community_colors'])
                        ],
                        line_width=2
                    ),
                    name=f'Community {community_id}'
                ),
                row=row, col=col
            )
            
        fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False,
                        row=row, col=col)
        fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False,
                        row=row, col=col)

    def _add_time_evolution(self, fig: go.Figure, row: int, col: int) -> None:
        """Add network evolution over time visualization."""
        if not hasattr(self.network_builder, 'temporal_networks'):
            return
            
        # Calculate network metrics over time
        metrics = pd.DataFrame(index=self.network_builder.temporal_networks.keys())
        
        for timestamp, network in self.network_builder.temporal_networks.items():
            metrics.loc[timestamp, 'density'] = nx.density(network)
            metrics.loc[timestamp, 'avg_clustering'] = nx.average_clustering(network)
            metrics.loc[timestamp, 'avg_path_length'] = nx.average_shortest_path_length(network)
            metrics.loc[timestamp, 'num_components'] = nx.number_connected_components(network)
            
            # Calculate node centrality distributions
            centrality = nx.eigenvector_centrality_numpy(network)
            metrics.loc[timestamp, 'centralization'] = np.std(list(centrality.values()))
            
        # Plot each metric as a line
        for column in metrics.columns:
            fig.add_trace(
                go.Scatter(
                    x=metrics.index,
                    y=metrics[column],
                    name=column.replace('_', ' ').title(),
                    line=dict(shape='spline', smoothing=0.3)
                ),
                row=row, col=col
            )
            
        # Update axes labels
        fig.update_xaxes(title_text="Time", row=row, col=col)
        fig.update_yaxes(title_text="Metric Value", row=row, col=col)
        
    def _add_synchronization_heatmap(self, fig: go.Figure, sync_metrics: pd.DataFrame,
                                row: int, col: int) -> None:
        """Add synchronization analysis heatmap."""
        # Create correlation matrix of synchronization metrics
        corr_matrix = sync_metrics.corr()
        
        fig.add_trace(
            go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu',
                zmid=0,
                colorbar=dict(
                    title='Synchronization',
                    titleside='right'
                )
            ),
            row=row, col=col
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Asset", row=row, col=col)
        fig.update_yaxes(title_text="Asset", row=row, col=col)
        
    def _calculate_arrow_angle(self, start_pos: Tuple[float, float],
                        end_pos: Tuple[float, float]) -> float:
        """Calculate angle for arrow marker."""
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        return np.degrees(np.arctan2(dy, dx))
        
    def _scale_edge_width(self, weight: float) -> float:
        """Scale edge width based on weight."""
        min_width, max_width = self.config.edge_width_range
        if not hasattr(self, '_max_weight'):
            self._max_weight = max(
                d.get('weight', 1.0)
                for _, _, d in self.network_builder.graph.edges(data=True)
            )
            
        if self._max_weight == 0:
            return min_width
            
        normalized = weight / self._max_weight
        return min_width + normalized * (max_width - min_width)
        
    def _add_pattern_propagation_analysis(self, fig: go.Figure, 
                                    propagation_metrics: pd.DataFrame,
                                    row: int, col: int) -> None:
        """Add visualization of pattern propagation analysis."""
        # Calculate propagation speed
        prop_speed = np.gradient(propagation_metrics['global_concentration'])
        
        # Add pattern concentration trace
        fig.add_trace(
            go.Scatter(
                x=propagation_metrics.index,
                y=propagation_metrics['global_concentration'],
                name='Pattern Concentration',
                line=dict(color=self.config.color_scheme['primary'])
            ),
            row=row, col=col
        )
        
        # Add propagation speed trace
        fig.add_trace(
            go.Scatter(
                x=propagation_metrics.index,
                y=prop_speed,
                name='Propagation Speed',
                line=dict(
                    color=self.config.color_scheme['secondary'],
                    dash='dash'
                )
            ),
            row=row, col=col
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Time", row=row, col=col)
        fig.update_yaxes(title_text="Metric Value", row=row, col=col)
        
    def _add_directed_network(self, fig: go.Figure, network: nx.DiGraph,
                        row: int, col: int) -> None:
        """Add directed network visualization with arrows."""
        pos = nx.spring_layout(
            network,
            k=1/np.sqrt(len(network.nodes())),
            iterations=self.config.layout_iterations,
            seed=self.config.layout_seed
        )
        
        # Add edges with arrows
        for edge in network.edges(data=True):
            start_pos = pos[edge[0]]
            end_pos = pos[edge[1]]
            
            # Calculate arrow position (80% along edge)
            arrow_pos = (
                0.8 * start_pos[0] + 0.2 * end_pos[0],
                0.8 * start_pos[1] + 0.2 * end_pos[1]
            )
            
            # Add edge line
            fig.add_trace(
                go.Scatter(
                    x=[start_pos[0], end_pos[0]],
                    y=[start_pos[1], end_pos[1]],
                    mode='lines',
                    line=dict(
                        width=self._scale_edge_width(edge[2].get('weight', 1.0)),
                        color=self.config.color_scheme['edge_default']
                    ),
                    hoverinfo='none'
                ),
                row=row, col=col
            )
            
            # Add arrow marker
            fig.add_trace(
                go.Scatter(
                    x=[arrow_pos[0]],
                    y=[arrow_pos[1]],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-right',
                        size=8,
                        color=self.config.color_scheme['edge_default'],
                        angle=self._calculate_arrow_angle(start_pos, end_pos)
                    ),
                    hoverinfo='none'
                ),
                row=row, col=col
            )
            
        # Add nodes with proper formatting
        self._add_node_trace(fig, network, pos, row=row, col=col)

    def create_pattern_regime_analysis(self, pattern_type: str, regimes: List[Dict]) -> go.Figure:
        """Analyze pattern behavior across different market regimes."""
        if not self.pattern_integration:
            raise ValueError("Pattern integration required")
            
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Pattern Success by Regime',
                'Regime Transition Impact',
                'Pattern Strength Distribution',
                'Regime-Pattern Correlation'
            )
        )
        
        # Get pattern metrics for each regime
        regime_metrics = {}
        pattern_strengths = {}
        transition_impacts = {}
        
        for regime in regimes:
            # Get pattern metrics for current regime
            pattern_metrics = self.pattern_integration.analyze_pattern_propagation(
                pattern_type, 
                start_date=regime['start_date'],
                end_date=regime['end_date']
            )
            regime_metrics[regime['name']] = pattern_metrics
            
            # Calculate pattern strengths
            strengths = pattern_metrics['signal_strength'].mean()
            pattern_strengths[regime['name']] = strengths
            
            # Calculate transition impacts (if not first regime)
            if len(transition_impacts) > 0:
                prev_regime = list(regime_metrics.keys())[-2]
                impact = abs(strengths - pattern_strengths[prev_regime])
                transition_impacts[f"{prev_regime}_to_{regime['name']}"] = impact
        
        # 1. Plot success rates by regime
        success_rates = {
            regime: metrics['success_rate'].mean()
            for regime, metrics in regime_metrics.items()
        }
        
        fig.add_trace(
            go.Bar(
                x=list(success_rates.keys()),
                y=list(success_rates.values()),
                name='Pattern Success Rate',
                marker_color=[
                    self.config.color_scheme['bullish'] if v > 0.5 
                    else self.config.color_scheme['bearish']
                    for v in success_rates.values()
                ]
            ),
            row=1, col=1
        )
        
        # 2. Plot regime transition impact
        fig.add_trace(
            go.Scatter(
                x=list(transition_impacts.keys()),
                y=list(transition_impacts.values()),
                mode='lines+markers',
                name='Transition Impact',
                line=dict(color=self.config.color_scheme['primary'])
            ),
            row=1, col=2
        )
        
        # 3. Plot pattern strength distribution
        for regime, metrics in regime_metrics.items():
            fig.add_trace(
                go.Violin(
                    x=[regime] * len(metrics['signal_strength']),
                    y=metrics['signal_strength'],
                    name=regime,
                    box_visible=True,
                    meanline_visible=True
                ),
                row=2, col=1
            )
        
        # 4. Plot regime-pattern correlation matrix
        correlation_data = pd.DataFrame({
            regime: metrics['signal_strength']
            for regime, metrics in regime_metrics.items()
        }).corr()
        
        fig.add_trace(
            go.Heatmap(
                z=correlation_data.values,
                x=correlation_data.columns,
                y=correlation_data.index,
                colorscale='RdBu',
                zmid=0
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            width=1200,
            showlegend=True,
            title_text=f"Pattern-Regime Analysis for {pattern_type}"
        )
        
        # Update axes
        fig.update_xaxes(title_text="Regime", row=1, col=1)
        fig.update_xaxes(title_text="Regime Transition", row=1, col=2)
        fig.update_xaxes(title_text="Regime", row=2, col=1)
        fig.update_xaxes(title_text="Regime", row=2, col=2)
        
        fig.update_yaxes(title_text="Success Rate", row=1, col=1)
        fig.update_yaxes(title_text="Impact Magnitude", row=1, col=2)
        fig.update_yaxes(title_text="Pattern Strength", row=2, col=1)
        fig.update_yaxes(title_text="Regime", row=2, col=2)
        
        return fig

    def create_multi_pattern_comparison(self, pattern_types: List[str]) -> go.Figure:
        """Compare multiple pattern behaviors in the network."""
        if not self.pattern_integration:
            raise ValueError("Pattern integration required")
            
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Pattern Co-occurrence Network',
                'Pattern Correlation Matrix',
                'Pattern Success Comparison',
                'Pattern Sequence Analysis'
            )
        )
        
        # Create pattern co-occurrence network
        co_occurrence = np.zeros((len(pattern_types), len(pattern_types)))
        pattern_metrics = {}
        
        for i, pattern1 in enumerate(pattern_types):
            metrics1 = self.pattern_integration.analyze_pattern_propagation(pattern1)
            pattern_metrics[pattern1] = metrics1
            
            for j, pattern2 in enumerate(pattern_types):
                if i != j:
                    metrics2 = self.pattern_integration.analyze_pattern_propagation(pattern2)
                    co_occurrence[i, j] = np.corrcoef(
                        metrics1['global_concentration'],
                        metrics2['global_concentration']
                    )[0, 1]
        
        # 1. Pattern Co-occurrence Network
        G = nx.Graph()
        for i, pattern1 in enumerate(pattern_types):
            for j, pattern2 in enumerate(pattern_types):
                if i < j and abs(co_occurrence[i, j]) > 0.3:  # Threshold
                    G.add_edge(pattern1, pattern2, weight=abs(co_occurrence[i, j]))
        
        pos = nx.spring_layout(G)
        
        # Add edges
        edge_trace = go.Scatter(
            x=[],
            y=[],
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += (x0, x1, None)
            edge_trace['y'] += (y0, y1, None)
            
        fig.add_trace(edge_trace, row=1, col=1)
        
        # Add nodes
        node_trace = go.Scatter(
            x=[pos[node][0] for node in G.nodes()],
            y=[pos[node][1] for node in G.nodes()],
            mode='markers+text',
            text=list(G.nodes()),
            textposition='top center',
            marker=dict(
                size=20,
                color=list(range(len(G.nodes()))),
                colorscale='Viridis',
                line_width=2
            )
        )
        
        fig.add_trace(node_trace, row=1, col=1)
        
        # 2. Pattern Correlation Matrix
        fig.add_trace(
            go.Heatmap(
                z=co_occurrence,
                x=pattern_types,
                y=pattern_types,
                colorscale='RdBu',
                zmid=0
            ),
            row=1, col=2
        )
        
        # 3. Pattern Success Comparison
        success_rates = {
            pattern: metrics['success_rate'].mean()
            for pattern, metrics in pattern_metrics.items()
        }
        
        fig.add_trace(
            go.Bar(
                x=list(success_rates.keys()),
                y=list(success_rates.values()),
                name='Success Rate',
                marker_color=self.config.color_scheme['primary']
            ),
            row=2, col=1
        )
        
        # 4. Pattern Sequence Analysis
        sequence_probs = self._calculate_pattern_sequences(pattern_metrics)
        
        fig.add_trace(
            go.Heatmap(
                z=sequence_probs.values,
                x=sequence_probs.columns,
                y=sequence_probs.index,
                colorscale='Viridis'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1000,
            width=1400,
            showlegend=True,
            title_text="Multi-Pattern Analysis"
        )
        
        return fig

    def create_network_evolution_analysis(self, 
                                        window_size: int = 30,
                                        step_size: int = 5) -> go.Figure:
        """Analyze how network structure evolves over time."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Network Density Evolution',
                'Community Structure Changes',
                'Node Role Dynamics',
                'Edge Weight Distribution'
            )
        )
        
        # Calculate rolling network metrics
        dates = list(next(iter(self.network_builder.data.values())).index)
        metrics = []
        community_evolution = []
        node_roles = defaultdict(list)
        edge_weights = []
        
        for i in range(window_size, len(dates), step_size):
            window_data = {
                asset: df.iloc[i-window_size:i]
                for asset, df in self.network_builder.data.items()
            }
            
            window_network = NetworkBuilder(window_data, self.network_builder.config)
            window_network.create_correlation_network()
            
            # Basic metrics
            metrics.append({
                'date': dates[i],
                'density': nx.density(window_network.graph),
                'clustering': nx.average_clustering(window_network.graph),
                'components': nx.number_connected_components(window_network.graph),
                'avg_degree': np.mean([d for _, d in window_network.graph.degree()])
            })
            
            # Community detection
            communities = self.network_builder.detect_communities(
                method='louvain',
                G=window_network.graph
            )
            community_evolution.append({
                'date': dates[i],
                'num_communities': len(set(communities.values())),
                'modularity': self._calculate_modularity(window_network.graph, communities)
            })
            
            # Node roles
            centrality = nx.eigenvector_centrality_numpy(window_network.graph)
            clustering = nx.clustering(window_network.graph)
            
            for node in window_network.graph.nodes():
                role = self._determine_node_role(
                    centrality[node],
                    clustering[node],
                    window_network.graph.degree(node)
                )
                node_roles[role].append(dates[i])
                
            # Edge weights
            weights = [d['weight'] for _, _, d in window_network.graph.edges(data=True)]
            edge_weights.append({
                'date': dates[i],
                'mean': np.mean(weights),
                'std': np.std(weights),
                'max': np.max(weights)
            })
        
        # Convert to DataFrames
        metrics_df = pd.DataFrame(metrics)
        community_df = pd.DataFrame(community_evolution)
        edge_weights_df = pd.DataFrame(edge_weights)
        
        # 1. Network Density Evolution
        for col in ['density', 'clustering', 'avg_degree']:
            fig.add_trace(
                go.Scatter(
                    x=metrics_df['date'],
                    y=metrics_df[col],
                    name=col.replace('_', ' ').title(),
                    line=dict(shape='spline', smoothing=0.3)
                ),
                row=1, col=1
            )
        
        # 2. Community Structure Changes
        fig.add_trace(
            go.Scatter(
                x=community_df['date'],
                y=community_df['num_communities'],
                name='Number of Communities',
                line=dict(color=self.config.color_scheme['primary'])
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=community_df['date'],
                y=community_df['modularity'],
                name='Modularity',
                line=dict(
                    color=self.config.color_scheme['secondary'],
                    dash='dash'
                )
            ),
            row=1, col=2
        )
        
        # 3. Node Role Dynamics
        for role, dates in node_roles.items():
            hist_data = pd.Series(dates).value_counts().sort_index()
            fig.add_trace(
                go.Scatter(
                    x=hist_data.index,
                    y=hist_data.values,
                    name=role,
                    fill='tonexty'
                ),
                row=2, col=1
            )
        
        # 4. Edge Weight Distribution Evolution
        fig.add_trace(
            go.Scatter(
                x=edge_weights_df['date'],
                y=edge_weights_df['mean'],
                name='Mean Weight',
                line=dict(color=self.config.color_scheme['primary'])
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=edge_weights_df['date'],
                y=edge_weights_df['mean'] + edge_weights_df['std'],
                y0=edge_weights_df['mean'] - edge_weights_df['std'],
                name='Weight Std Dev',
                fill='tonexty',
                line=dict(color='rgba(0,0,0,0)')
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1000,
            width=1400,
            showlegend=True,
            title_text="Network Evolution Analysis"
        )
        
        # Update axes
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_xaxes(title_text="Time", row=1, col=2)
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_xaxes(title_text="Time", row=2, col=2)
        
        fig.update_yaxes(title_text="Metric Value", row=1, col=1)
        fig.update_yaxes(title_text="Community Metrics", row=1, col=2)
        fig.update_yaxes(title_text="Node Count", row=2, col=1)
        fig.update_yaxes(title_text="Edge Weight", row=2, col=2)
        
        return fig

    def _determine_node_role(self, centrality: float, clustering: float, degree: int) -> str:
        """Determine node role based on network metrics."""
        if centrality > 0.8:
            return 'Hub'
        elif centrality > 0.5 and clustering > 0.7:
            return 'Broker'
        elif degree > np.mean(list(dict(self.network_builder.graph.degree()).values())):
            return 'Connector'
        elif clustering > 0.8:
            return 'Core'
        else:
            return 'Peripheral'
    
    def _calculate_pattern_sequences(self, pattern_metrics: Dict) -> pd.DataFrame:
        """Calculate pattern sequence probabilities.
        
        Args:
            pattern_metrics: Dictionary of pattern metrics over time
            
        Returns:
            DataFrame: Transition probabilities between patterns
        """
        # Initialize transition matrix
        patterns = list(pattern_metrics.keys())
        transitions = pd.DataFrame(0.0, index=patterns, columns=patterns)
        
        # Calculate transitions
        for i in range(len(patterns)):
            pattern1 = patterns[i]
            signal1 = pattern_metrics[pattern1]['signal'] > 0
            
            for j in range(len(patterns)):
                pattern2 = patterns[j]
                signal2 = pattern_metrics[pattern2]['signal'] > 0
                
                # Count transitions from pattern1 to pattern2
                transitions.loc[pattern1, pattern2] = np.sum(
                    signal1[:-1] & signal2[1:]
                ) / max(1, np.sum(signal1[:-1]))
                
        return transitions

    def create_pattern_forecast(self, pattern_type: str, horizon: int = 30) -> go.Figure:
        """Create pattern occurrence forecast visualization.
        
        Args:
            pattern_type: Type of pattern to forecast
            horizon: Number of periods to forecast
            
        Returns:
            go.Figure: Pattern forecast visualization
        """
        if not self.pattern_integration:
            raise ValueError("Pattern integration required")
            
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Pattern Occurrence Forecast',
                'Forecast Confidence',
                'Contributing Factors',
                'Historical Performance'
            )
        )
        
        # Get historical pattern data
        pattern_metrics = self.pattern_integration.analyze_pattern_propagation(pattern_type)
        
        # Create forecast using pattern metrics
        forecast, confidence = self._forecast_pattern_occurrence(
            pattern_metrics, horizon
        )
        
        # 1. Pattern Occurrence Forecast
        fig.add_trace(
            go.Scatter(
                x=pattern_metrics.index,
                y=pattern_metrics['global_concentration'],
                name='Historical',
                line=dict(color=self.config.color_scheme['primary'])
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=forecast.index,
                y=forecast,
                name='Forecast',
                line=dict(color=self.config.color_scheme['secondary'])
            ),
            row=1, col=1
        )
        
        # Add confidence intervals
        fig.add_trace(
            go.Scatter(
                x=forecast.index,
                y=forecast + confidence,
                line=dict(width=0),
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=forecast.index,
                y=forecast - confidence,
                fill='tonexty',
                line=dict(width=0),
                name='Confidence Interval'
            ),
            row=1, col=1
        )
        
        # 2. Forecast Confidence
        fig.add_trace(
            go.Scatter(
                x=forecast.index,
                y=confidence,
                name='Uncertainty',
                fill='tozeroy',
                line=dict(color=self.config.color_scheme['tertiary'])
            ),
            row=1, col=2
        )
        
        # 3. Contributing Factors
        factors = self._analyze_pattern_factors(pattern_metrics)
        fig.add_trace(
            go.Bar(
                x=list(factors.keys()),
                y=list(factors.values()),
                name='Factor Importance'
            ),
            row=2, col=1
        )
        
        # 4. Historical Performance
        performance = self._evaluate_pattern_performance(pattern_metrics)
        fig.add_trace(
            go.Scatter(
                x=performance.index,
                y=performance['accuracy'],
                name='Forecast Accuracy',
                line=dict(color=self.config.color_scheme['primary'])
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            width=1200,
            showlegend=True,
            title_text=f"Pattern Forecast Analysis - {pattern_type}"
        )
        
        return fig

    def _forecast_pattern_occurrence(self, 
                                pattern_metrics: pd.DataFrame,
                                horizon: int) -> Tuple[pd.Series, pd.Series]:
        """Forecast pattern occurrences.
        
        Args:
            pattern_metrics: Historical pattern metrics
            horizon: Forecast horizon
            
        Returns:
            Tuple of (forecast, confidence) Series
        """
        # Prepare features for forecasting
        features = pd.DataFrame({
            'concentration': pattern_metrics['global_concentration'],
            'momentum': pattern_metrics['global_concentration'].diff(),
            'volatility': pattern_metrics['global_concentration'].rolling(5).std(),
            'trend': pattern_metrics['global_concentration'].rolling(20).mean()
        }).dropna()
        
        # Split into train/test
        train_size = int(len(features) * 0.8)
        train_features = features[:train_size]
        train_target = pattern_metrics['global_concentration'][train_features.index]
        
        # Train model
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(train_features, train_target)
        
        # Generate forecast
        last_features = features.iloc[-1:]
        forecast = []
        confidence = []
        
        for _ in range(horizon):
            # Make prediction
            pred = model.predict(last_features)
            forecast.append(pred[0])
            
            # Estimate uncertainty
            predictions = []
            for estimator in model.estimators_:
                predictions.append(estimator[0].predict(last_features)[0])
            confidence.append(np.std(predictions))
            
            # Update features for next prediction
            new_features = pd.DataFrame({
                'concentration': [pred[0]],
                'momentum': [pred[0] - last_features['concentration'].iloc[-1]],
                'volatility': [last_features['volatility'].iloc[-1]],  # Use last known volatility
                'trend': [last_features['trend'].iloc[-1]]  # Use last known trend
            }, index=[last_features.index[-1] + pd.Timedelta(days=1)])
            
            last_features = new_features
            
        # Create forecast series
        dates = pd.date_range(
            start=pattern_metrics.index[-1] + pd.Timedelta(days=1),
            periods=horizon,
            freq='D'
        )
        
        return (
            pd.Series(forecast, index=dates),
            pd.Series(confidence, index=dates)
        )

    def _analyze_pattern_factors(self, pattern_metrics: pd.DataFrame) -> Dict[str, float]:
        """Analyze factors contributing to pattern occurrence."""
        # Calculate factor importance
        factors = {
            'Market_Volatility': abs(np.corrcoef(
                pattern_metrics['global_concentration'],
                pattern_metrics['global_concentration'].rolling(5).std()
            )[0, 1]),
            'Trend_Strength': abs(np.corrcoef(
                pattern_metrics['global_concentration'],
                pattern_metrics['global_concentration'].rolling(20).mean()
            )[0, 1]),
            'Volume': abs(np.corrcoef(
                pattern_metrics['global_concentration'],
                pattern_metrics.get('volume_ratio', np.zeros_like(pattern_metrics.index))
            )[0, 1]),
            'Previous_Success': abs(np.corrcoef(
                pattern_metrics['global_concentration'],
                pattern_metrics['success_rate'].shift(1)
            )[0, 1])
        }
        
        # Normalize importance scores
        total = sum(factors.values())
        return {k: v/total for k, v in factors.items()}

    def _evaluate_pattern_performance(self, pattern_metrics: pd.DataFrame) -> pd.DataFrame:
        """Evaluate historical pattern forecasting performance."""
        # Calculate rolling accuracy
        window = 20
        accuracy = pd.DataFrame(index=pattern_metrics.index)
        
        accuracy['accuracy'] = pattern_metrics['success_rate'].rolling(window).mean()
        accuracy['volatility'] = pattern_metrics['success_rate'].rolling(window).std()
        accuracy['trend'] = accuracy['accuracy'].rolling(window).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0]
        )
        
        return accuracy
    
    def create_resilience_analysis(self, 
                             n_simulations: int = 100,
                             removal_fraction: float = 0.3) -> go.Figure:
        """Analyze network resilience through node/edge removal simulations.
        
        Args:
            n_simulations: Number of simulation runs
            removal_fraction: Fraction of nodes/edges to remove
            
        Returns:
            go.Figure: Network resilience visualization
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Random Node Removal Impact',
                'Targeted Node Removal Impact',
                'Edge Weight Impact',
                'Recovery Analysis'
            )
        )
        
        # Run node removal simulations
        random_results = self._simulate_node_removal(
            'random', n_simulations, removal_fraction
        )
        targeted_results = self._simulate_node_removal(
            'targeted', n_simulations, removal_fraction
        )
        
        # 1. Random Node Removal Impact
        for metric, values in random_results.items():
            fig.add_trace(
                go.Scatter(
                    x=np.linspace(0, removal_fraction, len(values)),
                    y=values,
                    name=f'Random - {metric}',
                    line=dict(dash='solid')
                ),
                row=1, col=1
            )
        
        # 2. Targeted Node Removal Impact
        for metric, values in targeted_results.items():
            fig.add_trace(
                go.Scatter(
                    x=np.linspace(0, removal_fraction, len(values)),
                    y=values,
                    name=f'Targeted - {metric}',
                    line=dict(dash='dash')
                ),
                row=1, col=2
            )
        
        # 3. Edge Weight Impact Analysis
        edge_impact = self._analyze_edge_weight_impact()
        fig.add_trace(
            go.Heatmap(
                z=edge_impact.values,
                x=edge_impact.columns,
                y=edge_impact.index,
                colorscale='RdBu',
                colorbar=dict(title='Impact')
            ),
            row=2, col=1
        )
        
        # 4. Recovery Analysis
        recovery_metrics = self._analyze_network_recovery()
        fig.add_trace(
            go.Scatter(
                x=recovery_metrics.index,
                y=recovery_metrics['recovery_rate'],
                name='Recovery Rate',
                line=dict(color=self.config.color_scheme['primary'])
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=900,
            width=1200,
            showlegend=True,
            title_text="Network Resilience Analysis"
        )
        
        return fig

    def _simulate_node_removal(self, 
                            mode: str, 
                            n_simulations: int,
                            removal_fraction: float) -> Dict[str, List[float]]:
        """Simulate network behavior under node removal.
        
        Args:
            mode: 'random' or 'targeted'
            n_simulations: Number of simulations
            removal_fraction: Fraction of nodes to remove
            
        Returns:
            Dict of metrics over removal process
        """
        metrics = defaultdict(list)
        G = self.network_builder.graph.copy()
        n_nodes = len(G.nodes())
        nodes_to_remove = int(n_nodes * removal_fraction)
        
        for _ in range(n_simulations):
            temp_G = G.copy()
            
            if mode == 'random':
                remove_sequence = random.sample(list(temp_G.nodes()), nodes_to_remove)
            else:  # targeted
                centrality = nx.eigenvector_centrality_numpy(temp_G)
                remove_sequence = sorted(
                    centrality.keys(),
                    key=lambda x: centrality[x],
                    reverse=True
                )[:nodes_to_remove]
            
            metrics_sequence = []
            for node in remove_sequence:
                temp_G.remove_node(node)
                
                # Calculate network metrics
                metrics_sequence.append({
                    'connectivity': nx.node_connectivity(temp_G),
                    'efficiency': nx.global_efficiency(temp_G),
                    'clustering': nx.average_clustering(temp_G),
                    'components': nx.number_connected_components(temp_G)
                })
            
            # Average metrics across simulations
            for metric in metrics_sequence[0].keys():
                metrics[metric].append([m[metric] for m in metrics_sequence])
        
        # Average across simulations
        return {
            metric: np.mean(values, axis=0)
            for metric, values in metrics.items()
        }

    def _analyze_edge_weight_impact(self) -> pd.DataFrame:
        """Analyze impact of edge weight perturbations."""
        G = self.network_builder.graph
        n_edges = len(G.edges())
        perturbation_levels = np.linspace(-0.5, 0.5, 11)  # -50% to +50%
        
        impact_matrix = pd.DataFrame(
            index=perturbation_levels,
            columns=['connectivity', 'efficiency', 'modularity']
        )
        
        base_metrics = {
            'connectivity': nx.node_connectivity(G),
            'efficiency': nx.global_efficiency(G),
            'modularity': self._calculate_modularity(
                G, self.network_builder.detect_communities()
            )
        }
        
        for perturbation in perturbation_levels:
            # Create perturbed graph
            G_perturbed = G.copy()
            for u, v, data in G_perturbed.edges(data=True):
                data['weight'] *= (1 + perturbation)
            
            # Calculate metrics
            impact_matrix.loc[perturbation] = [
                nx.node_connectivity(G_perturbed) / base_metrics['connectivity'],
                nx.global_efficiency(G_perturbed) / base_metrics['efficiency'],
                self._calculate_modularity(
                    G_perturbed,
                    self.network_builder.detect_communities(G=G_perturbed)
                ) / base_metrics['modularity']
            ]
        
        return impact_matrix

    def _analyze_network_recovery(self) -> pd.DataFrame:
        """Analyze network recovery after perturbations."""
        recovery_metrics = pd.DataFrame()
        G = self.network_builder.graph
        
        # Simulate recovery process
        n_steps = 20
        for step in range(n_steps):
            # Remove random edges
            G_damaged = G.copy()
            edges_to_remove = random.sample(
                list(G_damaged.edges()),
                int(len(G_damaged.edges()) * 0.3)
            )
            G_damaged.remove_edges_from(edges_to_remove)
            
            # Simulate recovery by adding edges back
            edges_recovered = edges_to_remove[:int(len(edges_to_remove) * step/n_steps)]
            G_damaged.add_edges_from(edges_recovered)
            
            # Calculate recovery metrics
            recovery_metrics.loc[step, 'recovery_rate'] = (
                nx.node_connectivity(G_damaged) / nx.node_connectivity(G)
            )
            recovery_metrics.loc[step, 'efficiency_recovery'] = (
                nx.global_efficiency(G_damaged) / nx.global_efficiency(G)
            )
            
        return recovery_metrics

    def create_pattern_cascade_analysis(self, pattern_type: str) -> go.Figure:
        """Analyze pattern propagation cascades through the network.
        
        Args:
            pattern_type: Type of pattern to analyze
            
        Returns:
            go.Figure: Pattern cascade visualization
        """
        if not self.pattern_integration:
            raise ValueError("Pattern integration required")
            
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Cascade Progression',
                'Node Influence in Cascades',
                'Cascade Size Distribution',
                'Temporal Cascade Patterns'
            )
        )
        
        # Get pattern cascade data
        cascade_data = self._analyze_pattern_cascades(pattern_type)
        
        # 1. Cascade Progression
        for cascade_id, cascade in enumerate(cascade_data['progressions']):
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(cascade))),
                    y=cascade,
                    name=f'Cascade {cascade_id}',
                    line=dict(width=1, opacity=0.6)
                ),
                row=1, col=1
            )
        
        # 2. Node Influence
        node_influence = cascade_data['node_influence']
        fig.add_trace(
            go.Bar(
                x=list(node_influence.keys()),
                y=list(node_influence.values()),
                name='Node Influence'
            ),
            row=1, col=2
        )
        
        # 3. Cascade Size Distribution
        cascade_sizes = cascade_data['cascade_sizes']
        fig.add_trace(
            go.Histogram(
                x=cascade_sizes,
                nbinsx=20,
                name='Cascade Sizes'
            ),
            row=2, col=1
        )
        
        # 4. Temporal Patterns
        temporal_patterns = cascade_data['temporal_patterns']
        fig.add_trace(
            go.Scatter(
                x=temporal_patterns.index,
                y=temporal_patterns['cascade_frequency'],
                name='Cascade Frequency',
                line=dict(color=self.config.color_scheme['primary'])
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=900,
            width=1200,
            showlegend=True,
            title_text=f"Pattern Cascade Analysis - {pattern_type}"
        )
        
        return fig

    def _analyze_pattern_cascades(self, pattern_type: str) -> Dict:
        """Analyze pattern propagation cascades."""
        pattern_metrics = self.pattern_integration.analyze_pattern_propagation(pattern_type)
        cascades = []
        node_influence = defaultdict(int)
        cascade_sizes = []
        temporal_patterns = pd.DataFrame(index=pattern_metrics.index)
        
        # Identify cascades
        for timestamp in pattern_metrics.index:
            if pattern_metrics.loc[timestamp, 'global_concentration'] > 0.5:
                # Track cascade progression
                progression = []
                affected_nodes = set()
                
                for t in range(5):  # Look ahead 5 periods
                    if timestamp + t in pattern_metrics.index:
                        newly_affected = set(
                            node for node in self.network_builder.graph.nodes()
                            if pattern_metrics.loc[timestamp + t, f'{node}_concentration'] > 0.5
                        )
                        progression.append(len(newly_affected))
                        affected_nodes.update(newly_affected)
                        
                        # Update node influence
                        for node in newly_affected:
                            node_influence[node] += 1
                
                cascades.append(progression)
                cascade_sizes.append(len(affected_nodes))
                
        # Calculate temporal patterns
        temporal_patterns['cascade_frequency'] = pattern_metrics['global_concentration'].rolling(
            window=20
        ).apply(lambda x: sum(x > 0.5))
        
        return {
            'progressions': cascades,
            'node_influence': dict(node_influence),
            'cascade_sizes': cascade_sizes,
            'temporal_patterns': temporal_patterns
        }

    def create_regime_transition_network(self, regimes: List[Dict]) -> go.Figure:
        """Create visualization of regime transitions and their characteristics.
        
        Args:
            regimes: List of regime dictionaries with name, start_date, end_date, and characteristics
            
        Returns:
            go.Figure: Interactive regime transition network visualization
        """
        # Create directed graph for regime transitions
        G = nx.DiGraph()
        
        # Add nodes for each regime
        for regime in regimes:
            G.add_node(regime['name'], 
                    start_date=regime['start_date'],
                    end_date=regime['end_date'],
                    characteristics=regime.get('characteristics', {}))
        
        # Add edges for transitions
        for i in range(len(regimes)-1):
            current = regimes[i]
            next_regime = regimes[i+1]
            
            # Calculate transition metrics
            transition_metrics = self._calculate_transition_metrics(
                current, next_regime
            )
            
            G.add_edge(
                current['name'],
                next_regime['name'],
                **transition_metrics
            )
        
        # Create visualization
        fig = go.Figure()
        
        # Add edges with arrows
        pos = nx.spring_layout(G)
        edge_x = []
        edge_y = []
        edge_text = []
        
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            # Create curved path for edge
            path = self._create_curved_path(
                (x0, y0), (x1, y1), curvature=0.2
            )
            
            edge_x.extend(path[0] + [None])
            edge_y.extend(path[1] + [None])
            
            # Create edge label
            edge_text.append(
                f"Transition strength: {edge[2]['strength']:.2f}<br>" +
                f"Duration: {edge[2]['duration']:.1f} days"
            )
            
            # Add arrow marker
            arrow_pos = self._calculate_arrow_position(path[0], path[1], 0.8)
            fig.add_trace(
                go.Scatter(
                    x=[arrow_pos[0]],
                    y=[arrow_pos[1]],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-right',
                        size=15,
                        angle=self._calculate_arrow_angle(
                            path[0][-2:], path[1][-2:]
                        ),
                        color=self.config.color_scheme['edge_default']
                    ),
                    showlegend=False,
                    hoverinfo='none'
                )
            )
        
        # Add edges
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode='lines',
                line=dict(
                    width=2,
                    color=self.config.color_scheme['edge_default']
                ),
                hovertext=edge_text,
                hoverinfo='text',
                name='Transitions'
            )
        )
        
        # Add nodes
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        
        for node in G.nodes(data=True):
            x, y = pos[node[0]]
            node_x.append(x)
            node_y.append(y)
            
            # Create node label
            characteristics = node[1]['characteristics']
            node_text.append(
                f"Regime: {node[0]}<br>" +
                f"Start: {node[1]['start_date']:%Y-%m-%d}<br>" +
                f"End: {node[1]['end_date']:%Y-%m-%d}<br>" +
                f"Volatility: {characteristics.get('volatility', 'N/A')}<br>" +
                f"Trend: {characteristics.get('trend', 'N/A')}"
            )
            
            # Set node color based on regime characteristics
            node_colors.append(
                self._get_regime_color(characteristics)
            )
        
        fig.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode='markers+text',
                marker=dict(
                    size=30,
                    color=node_colors,
                    line=dict(width=2, color='white')
                ),
                text=[node[0] for node in G.nodes(data=True)],
                textposition='middle center',
                hovertext=node_text,
                hoverinfo='text',
                name='Regimes'
            )
        )
        
        # Update layout
        fig.update_layout(
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=1000,
            height=800,
            title_text="Market Regime Transition Network"
        )
        
        return fig

    def _calculate_transition_metrics(self, 
                                    current_regime: Dict,
                                    next_regime: Dict) -> Dict:
        """Calculate metrics for regime transition.
        
        Args:
            current_regime: Current regime information
            next_regime: Next regime information
            
        Returns:
            Dict: Transition metrics
        """
        # Calculate duration between regimes
        duration = (next_regime['start_date'] - 
                current_regime['end_date']).total_seconds() / (24 * 3600)
        
        # Calculate characteristic changes
        curr_chars = current_regime.get('characteristics', {})
        next_chars = next_regime.get('characteristics', {})
        
        # Calculate transition strength based on characteristic changes
        strength = 0
        metrics = ['volatility', 'trend', 'volume']
        
        for metric in metrics:
            if metric in curr_chars and metric in next_chars:
                strength += abs(
                    self._normalize_characteristic(curr_chars[metric]) -
                    self._normalize_characteristic(next_chars[metric])
                )
        
        strength = strength / len(metrics)  # Normalize to [0, 1]
        
        return {
            'duration': duration,
            'strength': strength
        }

    def _normalize_characteristic(self, value: str) -> float:
        """Normalize regime characteristic to numerical value."""
        if isinstance(value, (int, float)):
            return float(value)
            
        mappings = {
            'low': 0.0,
            'medium': 0.5,
            'high': 1.0,
            'up': 1.0,
            'down': 0.0,
            'sideways': 0.5
        }
        
        return mappings.get(value.lower(), 0.5)

    def _get_regime_color(self, characteristics: Dict) -> str:
        """Get color for regime based on its characteristics."""
        # Calculate color based on volatility and trend
        volatility = self._normalize_characteristic(
            characteristics.get('volatility', 'medium')
        )
        trend = self._normalize_characteristic(
            characteristics.get('trend', 'sideways')
        )
        
        # Use different colors for different regime types
        if trend > 0.7 and volatility < 0.3:
            return self.config.color_scheme['bullish']  # Strong uptrend, low vol
        elif trend < 0.3 and volatility > 0.7:
            return self.config.color_scheme['bearish']  # Strong downtrend, high vol
        elif volatility < 0.3:
            return self.config.color_scheme['neutral']  # Low volatility
        else:
            return self.config.color_scheme['complex']  # Complex regime

    def _create_curved_path(self, 
                        start: Tuple[float, float],
                        end: Tuple[float, float],
                        curvature: float = 0.2) -> Tuple[List[float], List[float]]:
        """Create curved path between two points.
        
        Args:
            start: Start point coordinates
            end: End point coordinates
            curvature: Amount of curvature (0 = straight line)
            
        Returns:
            Tuple of x and y coordinates for path
        """
        # Calculate midpoint
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        
        # Calculate perpendicular vector for control point
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        
        control_x = mid_x - dy * curvature
        control_y = mid_y + dx * curvature
        
        # Create path points
        t = np.linspace(0, 1, 100)
        x = ((1-t)**2 * start[0] + 
            2*(1-t)*t * control_x + 
            t**2 * end[0])
        y = ((1-t)**2 * start[1] + 
            2*(1-t)*t * control_y + 
            t**2 * end[1])
        
        return x.tolist(), y.tolist()