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
        
    def _add_timeseries_network_view(self, fig: go.Figure, row: int, col: int):
        """Add network view integrated with time series analysis."""
        if not self.timeseries_analyzer:
            return
            
        G = self.network_builder.graph
        pos = nx.spring_layout(G)
        
        # Get time series metrics
        ts_metrics = self._calculate_timeseries_metrics()
        
        # Create edge trace with time-based weights
        edge_x = []
        edge_y = []
        edge_weights = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # Calculate temporal correlation between nodes
            corr = ts_metrics['correlations'].get((edge[0], edge[1]), 0)
            edge_weights.extend([corr, corr, None])
        
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode='lines',
                line=dict(
                    width=1,
                    color=edge_weights,
                    colorscale='RdBu'
                ),
                hoverinfo='none'
            ),
            row=row, col=col
        )
        
        # Create node trace with time series properties
        node_x = []
        node_y = []
        node_colors = []
        node_sizes = []
        node_text = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Node properties based on time series
            volatility = ts_metrics['volatility'].get(node, 0)
            trend = ts_metrics['trend'].get(node, 0)
            
            node_colors.append(trend)  # Color by trend
            node_sizes.append(20 + 30 * volatility)  # Size by volatility
            
            node_text.append(
                f"Node: {node}<br>" +
                f"Trend: {trend:.3f}<br>" +
                f"Volatility: {volatility:.3f}"
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
                    colorbar=dict(title='Trend Strength')
                ),
                hovertext=node_text,
                hoverinfo='text'
            ),
            row=row, col=col
        )

    def _calculate_cross_module_metrics(self) -> pd.DataFrame:
        """Calculate metrics that combine data from all modules."""
        metrics = pd.DataFrame()
        
        if self.candlestick_visualizer and self.timeseries_analyzer:
            for node in self.network_builder.graph.nodes():
                # Get pattern metrics
                pattern_metrics = self._get_node_pattern_metrics(node)
                
                # Get time series metrics
                ts_metrics = self._get_node_timeseries_metrics(node)
                
                # Combine metrics
                combined = {**pattern_metrics, **ts_metrics}
                metrics = pd.concat([
                    metrics,
                    pd.DataFrame(combined, index=[node])
                ])
        
        return metrics

    def _add_combined_alerts_view(self, fig: go.Figure, alerts: Dict, row: int, col: int):
        """Add visualization of combined alerts from all modules."""
        # Create timeline of alerts
        timestamps = sorted(set(
            timestamp
            for alert_type in alerts.values()
            for timestamp in alert_type['timestamps']
        ))
        
        for alert_type, data in alerts.items():
            fig.add_trace(
                go.Scatter(
                    x=data['timestamps'],
                    y=data['severity'],
                    mode='markers+lines',
                    name=alert_type,
                    marker=dict(
                        size=10,
                        symbol='circle',
                        color=self._get_alert_color(alert_type)
                    )
                ),
                row=row, col=col
            )
            
    def create_multi_module_analysis(self) -> Dict[str, go.Figure]:
        """Create comprehensive multi-module analysis visualizations."""
        figures = {}
        
        # 1. Network-Pattern-Time Series Integration
        figures['integrated'] = self._create_integrated_analysis()
        
        # 2. Cross-Module Event Analysis
        figures['events'] = self._create_event_analysis()
        
        # 3. Predictive Analysis
        figures['predictions'] = self._create_predictive_analysis()
        
        # 4. Risk Analysis
        figures['risk'] = self._create_risk_analysis()
        
        return figures

    def _create_integrated_analysis(self) -> go.Figure:
        """Create visualization combining all module analyses."""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Network Evolution',
                'Pattern Impact',
                'Time Series Clustering',
                'Cross-Module Correlations',
                'Event Detection',
                'Risk Indicators'
            )
        )
        
        # Get integrated metrics
        integrated_metrics = self._calculate_integrated_metrics()
        
        # 1. Network Evolution with Pattern Overlay
        self._add_network_evolution_view(
            fig, integrated_metrics['network_evolution'],
            row=1, col=1
        )
        
        # 2. Pattern Impact Analysis
        self._add_pattern_impact_view(
            fig, integrated_metrics['pattern_impact'],
            row=1, col=2
        )
        
        # 3. Time Series Clustering
        self._add_timeseries_clustering_view(
            fig, integrated_metrics['ts_clusters'],
            row=2, col=1
        )
        
        # 4. Cross-Module Correlations
        correlation_matrix = self._calculate_cross_module_correlations()
        fig.add_trace(
            go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.index,
                colorscale='RdBu',
                zmid=0
            ),
            row=2, col=2
        )
        
        # 5. Event Detection
        events = self._detect_cross_module_events()
        self._add_event_detection_view(fig, events, row=3, col=1)
        
        # 6. Risk Indicators
        risk_metrics = self._calculate_risk_metrics()
        self._add_risk_indicator_view(fig, risk_metrics, row=3, col=2)
        
        return fig

    def _calculate_integrated_metrics(self) -> Dict:
        """Calculate metrics combining data from all modules."""
        metrics = {
            'network_evolution': self._calculate_network_evolution(),
            'pattern_impact': self._calculate_pattern_impact(),
            'ts_clusters': self._calculate_timeseries_clusters(),
            'events': self._detect_significant_events(),
            'risk': self._calculate_risk_indicators()
        }
        
        # Add cross-module interactions
        if self.candlestick_visualizer and self.timeseries_analyzer:
            metrics['interactions'] = self._calculate_module_interactions()
        
        return metrics

    def _detect_cross_module_events(self) -> pd.DataFrame:
        """Detect events by combining signals from all modules."""
        events = pd.DataFrame()
        
        # Get timestamps from network data
        if hasattr(self.network_builder, 'data'):
            timestamps = next(iter(self.network_builder.data.values())).index
            events = pd.DataFrame(index=timestamps)
            
            # Network structure changes
            network_changes = self._detect_network_changes()
            events['network_change'] = network_changes
            
            # Pattern events
            if self.candlestick_visualizer:
                pattern_events = self._detect_pattern_events()
                events['pattern_event'] = pattern_events
            
            # Time series events
            if self.timeseries_analyzer:
                ts_events = self._detect_timeseries_events()
                events['ts_event'] = ts_events
            
            # Calculate combined event score
            events['combined_score'] = events.mean(axis=1)
            
            # Detect significant events
            events['is_significant'] = (
                events['combined_score'] > events['combined_score'].mean() +
                2 * events['combined_score'].std()
            )
        
        return events

    def create_risk_dashboard(self) -> go.Figure:
        """Create dashboard for integrated risk analysis."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Network Risk Indicators',
                'Pattern-Based Risk',
                'Time Series Risk Metrics',
                'Combined Risk Assessment'
            )
        )
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics()
        
        # 1. Network Risk Indicators
        self._add_network_risk_view(
            fig, risk_metrics['network_risk'],
            row=1, col=1
        )
        
        # 2. Pattern-Based Risk
        if self.candlestick_visualizer:
            self._add_pattern_risk_view(
                fig, risk_metrics['pattern_risk'],
                row=1, col=2
            )
        
        # 3. Time Series Risk
        if self.timeseries_analyzer:
            self._add_timeseries_risk_view(
                fig, risk_metrics['ts_risk'],
                row=2, col=1
            )
        
        # 4. Combined Risk Assessment
        self._add_combined_risk_view(
            fig, risk_metrics['combined_risk'],
            row=2, col=2
        )
        
        return fig

    def _calculate_risk_metrics(self) -> Dict:
        """Calculate comprehensive risk metrics across all modules."""
        risk_metrics = {
            'network_risk': self._calculate_network_risk(),
            'combined_risk': pd.DataFrame()
        }
        
        # Add pattern-based risk if available
        if self.candlestick_visualizer:
            risk_metrics['pattern_risk'] = self._calculate_pattern_risk()
        
        # Add time series risk if available
        if self.timeseries_analyzer:
            risk_metrics['ts_risk'] = self._calculate_timeseries_risk()
        
        # Calculate combined risk metrics
        available_metrics = [
            metrics for metrics in risk_metrics.values()
            if isinstance(metrics, pd.DataFrame) and not metrics.empty
        ]
        
        if available_metrics:
            risk_metrics['combined_risk'] = pd.concat(
                available_metrics, axis=1
            ).mean(axis=1)
        
        return risk_metrics
    
    def create_pattern_colored_network(self) -> go.Figure:
        """Create network visualization with candlestick pattern-based coloring."""
        if not self.candlestick_visualizer:
            raise ValueError("Candlestick visualizer required for pattern coloring")
            
        fig = go.Figure()
        
        # Get pattern information from candlestick visualizer
        pattern_metrics = {}
        for asset, df in self.network_builder.data.items():
            patterns = self.candlestick_visualizer.patterns._get_all_pattern_methods()
            pattern_results = {}
            for pattern_name, pattern_func in patterns.items():
                try:
                    result = pattern_func(df)
                    if isinstance(result, tuple):
                        pattern_results[f"{pattern_name}_bullish"] = result[0].sum()
                        pattern_results[f"{pattern_name}_bearish"] = result[1].sum()
                    else:
                        pattern_results[pattern_name] = result.sum()
                except Exception as e:
                    continue
            pattern_metrics[asset] = pattern_results
        
        # Create network layout
        pos = nx.spring_layout(self.network_builder.graph)
        
        # Add edges
        edge_trace = self._create_edge_trace_with_patterns(
            pos, pattern_metrics
        )
        fig.add_trace(edge_trace)
        
        # Add nodes
        node_trace = self._create_node_trace_with_patterns(
            pos, pattern_metrics
        )
        fig.add_trace(node_trace)
        
        # Update layout
        fig.update_layout(
            title="Pattern-Based Network Analysis",
            showlegend=True,
            height=800,
            width=1000
        )
        
        return fig

    def combine_pattern_network_indicators(self) -> pd.DataFrame:
        """Create combined indicators from patterns and network metrics."""
        if not self.candlestick_visualizer:
            raise ValueError("Candlestick visualizer required")
            
        combined_indicators = pd.DataFrame()
        
        # Get network centrality metrics
        centrality = nx.eigenvector_centrality_numpy(self.network_builder.graph)
        clustering = nx.clustering(self.network_builder.graph)
        
        # Get pattern metrics for each node
        for node in self.network_builder.graph.nodes():
            node_data = self.network_builder.data[node]
            
            # Calculate pattern metrics
            pattern_metrics = self._calculate_node_pattern_metrics(node)
            
            # Combine with network metrics
            metrics = {
                'centrality': centrality[node],
                'clustering': clustering[node],
                **pattern_metrics
            }
            
            combined_indicators = pd.concat([
                combined_indicators,
                pd.DataFrame([metrics], index=[node])
            ])
        
        return combined_indicators

    def detect_cross_module_events(self) -> pd.DataFrame:
        """Detect events using both pattern and network analysis."""
        events = pd.DataFrame()
        
        if self.candlestick_visualizer:
            for node in self.network_builder.graph.nodes():
                # Get node data
                node_data = self.network_builder.data[node]
                
                # Detect pattern-based events
                pattern_events = self._detect_pattern_events(node_data)
                
                # Detect network structure changes
                network_events = self._detect_network_events(node)
                
                # Combine events
                combined_events = pd.concat([pattern_events, network_events], axis=1)
                events = pd.concat([events, combined_events])
        
        return events
    
    def calculate_time_varying_network_metrics(self) -> pd.DataFrame:
        """Calculate network metrics that vary over time."""
        if not self.timeseries_analyzer:
            raise ValueError("Time series analyzer required")
            
        metrics = pd.DataFrame()
        
        # Get time windows from data
        timestamps = list(next(iter(self.network_builder.data.values())).index)
        window_size = 30  # Configurable window size
        
        for i in range(window_size, len(timestamps)):
            window_metrics = {}
            
            # Create network for current window
            window_data = {
                asset: df.iloc[i-window_size:i]
                for asset, df in self.network_builder.data.items()
            }
            window_network = NetworkBuilder(window_data, self.network_builder.config)
            window_network.create_correlation_network()
            
            # Calculate network metrics
            window_metrics['density'] = nx.density(window_network.graph)
            window_metrics['clustering'] = nx.average_clustering(window_network.graph)
            window_metrics['modularity'] = self._calculate_modularity(
                window_network.graph,
                window_network.detect_communities()
            )
            
            metrics = pd.concat([
                metrics,
                pd.DataFrame([window_metrics], index=[timestamps[i]])
            ])
        
        return metrics

    def create_combined_forecast(self, horizon: int = 30) -> Dict[str, pd.DataFrame]:
        """Create forecasts combining network and time series analysis."""
        if not self.timeseries_analyzer:
            raise ValueError("Time series analyzer required")
            
        forecasts = {}
        
        # Get network-based relationships
        network_relationships = self._get_network_relationships()
        
        for node in self.network_builder.graph.nodes():
            # Get node data
            node_data = self.network_builder.data[node]
            
            # Get related nodes from network
            related_nodes = network_relationships[node]
            
            # Create combined forecast
            forecast = self._create_node_forecast(
                node, related_nodes, horizon
            )
            forecasts[node] = forecast
        
        return forecasts

    def analyze_temporal_patterns(self) -> pd.DataFrame:
        """Analyze how patterns evolve over time in the network."""
        analysis = pd.DataFrame()
        
        if self.candlestick_visualizer and self.timeseries_analyzer:
            # Get temporal decomposition
            for node in self.network_builder.graph.nodes():
                node_data = self.network_builder.data[node]
                
                # Decompose time series
                decomposition = self.timeseries_analyzer.decompose(node_data)
                
                # Get pattern occurrences
                patterns = self._get_node_patterns(node)
                
                # Analyze relationship between patterns and components
                temporal_metrics = self._analyze_pattern_components(
                    patterns, decomposition
                )
                
                analysis = pd.concat([
                    analysis,
                    pd.DataFrame([temporal_metrics], index=[node])
                ])
        
        return analysis