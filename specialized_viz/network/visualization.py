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
            
"""Enhanced network visualization module for specialized-viz library - Part 2."""

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