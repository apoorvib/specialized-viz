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
        """Calculate pattern sequence probabilities."""
        sequences