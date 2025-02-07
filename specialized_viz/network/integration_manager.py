class NetworkIntegrationManager:
    """Manages integration between network analysis and other modules."""
    
    def __init__(self,
                 network_builder: NetworkBuilder,
                 candlestick_visualizer: Optional['CandlestickVisualizer'] = None,
                 timeseries_analyzer: Optional['TimeseriesAnalysis'] = None):
        """
        Initialize integration manager.
        
        Args:
            network_builder: Network analysis component
            candlestick_visualizer: Optional candlestick visualization component
            timeseries_analyzer: Optional time series analysis component
        """
        self.network_builder = network_builder
        self.candlestick_visualizer = candlestick_visualizer
        self.timeseries_analyzer = timeseries_analyzer
        self.config = self._merge_configurations()
        
    def create_integrated_view(self) -> go.Figure:
        """Create comprehensive integrated visualization."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Network-Pattern Analysis',
                'Time Series Integration',
                'Cross-Module Metrics',
                'Combined Alerts'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'scatter'}],
                [{'type': 'heatmap'}, {'type': 'scatter'}]
            ]
        )
        
        # 1. Network-Pattern Integration
        if self.candlestick_visualizer:
            self._add_pattern_network_view(fig, row=1, col=1)
        
        # 2. Time Series Integration
        if self.timeseries_analyzer:
            self._add_timeseries_network_view(fig, row=1, col=2)
        
        # 3. Cross-Module Metrics
        cross_metrics = self._calculate_cross_module_metrics()
        fig.add_trace(
            go.Heatmap(
                z=cross_metrics.values,
                x=cross_metrics.columns,
                y=cross_metrics.index,
                colorscale='RdBu',
                colorbar=dict(title='Correlation')
            ),
            row=2, col=1
        )
        
        # 4. Combined Alerts
        alerts = self._gather_cross_module_alerts()
        self._add_combined_alerts_view(fig, alerts, row=2, col=2)
        
        # Update layout
        fig.update_layout(
            height=1000,
            width=1400,
            showlegend=True,
            title_text="Integrated Analysis Dashboard"
        )
        
        return fig
    
    def analyze_cross_module_patterns(self) -> pd.DataFrame:
        """Analyze patterns across different modules."""
        results = pd.DataFrame()
        
        if self.candlestick_visualizer and self.timeseries_analyzer:
            # Get pattern data
            pattern_data = self._get_pattern_data()
            
            # Get time series decomposition
            ts_components = self._get_timeseries_components()
            
            # Analyze relationships
            for pattern, data in pattern_data.items():
                for component, series in ts_components.items():
                    correlation = data.corr(series)
                    results.loc[pattern, component] = correlation
        
        return results
    
    def _merge_configurations(self) -> Dict:
        """Merge configurations from different modules."""
        config = {
            'network': self.network_builder.config.__dict__,
            'visualization': {}
        }
        
        if self.candlestick_visualizer:
            config['candlestick'] = self.candlestick_visualizer.config.__dict__
            
        if self.timeseries_analyzer:
            config['timeseries'] = self.timeseries_analyzer.config.__dict__
            
        return config
    
    def _add_pattern_network_view(self, fig: go.Figure, row: int, col: int):
        """Add network view colored by pattern analysis."""
        G = self.network_builder.graph
        pos = nx.spring_layout(G)
        
        # Get pattern information
        pattern_info = self._get_pattern_influence()
        
        # Create edge trace with pattern-based weights
        edge_x = []
        edge_y = []
        edge_colors = []
        
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # Color based on pattern similarity
            similarity = pattern_info['edge_weights'].get(edge, 0)
            edge_colors.extend([similarity, similarity, None])
        
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode='lines',
                line=dict(
                    width=1,
                    color=edge_colors,
                    colorscale='Viridis'
                ),
                hoverinfo='none'
            ),
            row=row, col=col
        )
        
        # Create node trace with pattern-based colors
        node_x = []
        node_y = []
        node_colors = []
        node_sizes = []
        node_text = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Node properties based on patterns
            node_colors.append(pattern_info['node_scores'][node])
            node_sizes.append(20 + 30 * pattern_info['node_importance'][node])
            
            # Node information
            node_text.append(
                f"Node: {node}<br>" +
                f"Pattern Score: {pattern_info['node_scores'][node]:.3f}<br>" +
                f"Top Pattern: {pattern_info['node_patterns'][node]}"
            )
        
        fig.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode='markers+text',
                text=[str(node) for node in G.nodes()],
                textposition='top center',
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    colorscale='RdYlBu',
                    showscale=True,
                    colorbar=dict(title='Pattern Score')
                ),
                hovertext=node_text,
                hoverinfo='text'
            ),
            row=row, col=col
        )