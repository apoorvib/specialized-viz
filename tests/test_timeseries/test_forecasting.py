import unittest
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from specialized_viz.timeseries.visualization import TimeseriesVisualizer
from specialized_viz.timeseries.analysis import TimeseriesAnalysis

class TestTimeseriesVisualizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data used across test methods"""
        # Create sample timeseries data with trend and seasonality
        dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
        trend = np.linspace(0, 10, 200)
        seasonal = np.sin(np.linspace(0, 8*np.pi, 200))
        noise = np.random.normal(0, 0.1, 200)
        
        cls.test_data = pd.DataFrame({
            'value': trend + seasonal + noise,
            'exog1': np.random.randn(200),
            'exog2': np.random.randn(200)
        }, index=dates)
        
        # Create analyzer and visualizer instances
        cls.analyzer = TimeseriesAnalysis(cls.test_data)
        cls.visualizer = TimeseriesVisualizer(cls.analyzer)

    def test_initialization(self):
        """Test visualizer initialization"""
        self.assertIsInstance(self.visualizer, TimeseriesVisualizer)
        self.assertIsInstance(self.visualizer.analyzer, TimeseriesAnalysis)
        self.assertTrue(hasattr(self.visualizer, 'color_scheme'))

    def test_plot_correlogram(self):
        """Test correlogram plotting"""
        fig = self.visualizer.plot_correlogram('value')
        self.assertIsInstance(fig, go.Figure)
        self.assertTrue(len(dashboard.layout.annotations) >= 9)

    def test_correlogram_components(self):
        """Test specific components of correlogram"""
        fig = self.visualizer.plot_correlogram('value')
        # Test ACF trace
        acf_trace = fig.data[0]
        self.assertEqual(acf_trace.type, 'bar')
        # Test PACF trace
        pacf_trace = fig.data[1]
        self.assertEqual(pacf_trace.type, 'bar')
        # Test rolling correlation
        rolling_trace = fig.data[2]
        self.assertEqual(rolling_trace.type, 'scatter')

    def test_distribution_evolution_components(self):
        """Test components of distribution evolution plot"""
        fig = self.visualizer.plot_distribution_evolution('value')
        # Check for rolling probability density
        density_traces = [trace for trace in fig.data if trace.name.startswith('Period')]
        self.assertTrue(len(density_traces) >= 4)
        # Check for quantile plot
        qq_trace = [trace for trace in fig.data if trace.name == 'Q-Q Plot']
        self.assertTrue(len(qq_trace) > 0)

    def test_feature_importance_plot_components(self):
        """Test components of feature importance plot"""
        features = pd.DataFrame({
            'feature1': np.random.randn(200),
            'feature2': np.random.randn(200),
            'feature3': np.random.randn(200)
        }, index=self.test_data.index)
        
        fig = self.visualizer.plot_feature_importance(features, self.test_data['value'])
        # Test SHAP values trace
        shap_trace = [trace for trace in fig.data if trace.name == 'SHAP Values']
        self.assertTrue(len(shap_trace) > 0)
        # Test correlation heatmap
        heatmap_trace = [trace for trace in fig.data if trace.type == 'heatmap']
        self.assertTrue(len(heatmap_trace) > 0)

    def test_comprehensive_dashboard_components(self):
        """Test all components of comprehensive dashboard"""
        fig = self.visualizer.create_comprehensive_dashboard('value')
        
        # Test time series plot
        self.assertTrue(any(trace.name == 'Original' for trace in fig.data))
        # Test decomposition
        self.assertTrue(any(trace.name == 'Trend' for trace in fig.data))
        # Test seasonality
        self.assertTrue(any('Seasonal' in trace.name for trace in fig.data))
        # Test volume profile if exists
        if 'volume' in self.test_data.columns:
            self.assertTrue(any('Volume' in trace.name for trace in fig.data))

    def test_trend_analysis_components(self):
        """Test components of trend analysis plot"""
        fig = self.visualizer.plot_trend_analysis('value')
        
        # Test original data trace
        self.assertTrue(any(trace.name == 'Original' for trace in fig.data))
        # Test trend line
        self.assertTrue(any(trace.name == 'Trend Line' for trace in fig.data))
        # Test residuals
        self.assertTrue(any(trace.name == 'Residuals' for trace in fig.data))
        # Test annotations (statistics)
        self.assertTrue(any('Slope:' in ann.text for ann in fig.layout.annotations))

    def test_change_points_components(self):
        """Test components of change points plot"""
        fig = self.visualizer.plot_change_points('value', methods=['cusum', 'pettitt'])
        
        # Test original data trace
        self.assertTrue(any(trace.name == 'Original' for trace in fig.data))
        # Test change point markers
        self.assertTrue(any('CUSUM Change Point' in str(shape) for shape in fig.layout.shapes if hasattr(shape, 'name')))
        self.assertTrue(any('Pettitt Change Point' in str(shape) for shape in fig.layout.shapes if hasattr(shape, 'name')))

    def test_interactive_dashboard_components(self):
        """Test components of interactive dashboard"""
        fig = self.visualizer.create_interactive_dashboard('value')
        
        # Test presence of key components
        self.assertTrue(any('Time Series with Trend' in ann.text for ann in fig.layout.annotations))
        self.assertTrue(any('Decomposition' in ann.text for ann in fig.layout.annotations))
        self.assertTrue(any('Change Points' in ann.text for ann in fig.layout.annotations))
        self.assertTrue(any('Seasonal Pattern' in ann.text for ann in fig.layout.annotations))
        
        # Test interactivity elements
        self.assertTrue(hasattr(fig.layout, 'updatemenus'))
        self.assertTrue(hasattr(fig.layout, 'sliders') or len(fig.layout.sliders or []) > 0)

    def test_plot_animations(self):
        """Test animation creation with different parameters"""
        # Test with different window sizes
        fig_small = self.visualizer.create_animation('value', window_size=20)
        fig_large = self.visualizer.create_animation('value', window_size=50)
        
        self.assertIsInstance(fig_small, go.Figure)
        self.assertIsInstance(fig_large, go.Figure)
        self.assertTrue(len(fig_small.frames) > len(fig_large.frames))

    def test_layout_configurations(self):
        """Test various layout configurations"""
        fig = self.visualizer.create_comprehensive_dashboard('value')
        
        # Test layout properties
        self.assertEqual(fig.layout.template, 'plotly_white')
        self.assertTrue(fig.layout.showlegend)
        self.assertTrue(fig.layout.hovermode in ['x', 'x unified'])
        
        # Test axes configurations
        self.assertTrue(all(hasattr(fig.layout, f'xaxis{i}') for i in range(1, 4)))
        self.assertTrue(all(hasattr(fig.layout, f'yaxis{i}') for i in range(1, 4)))

    def test_error_handling(self):
        """Test error handling in visualization methods"""
        # Test with invalid column name
        with self.assertRaises(KeyError):
            self.visualizer.plot_correlogram('nonexistent_column')
        
        # Test with invalid method parameter
        with self.assertRaises(ValueError):
            self.visualizer.plot_change_points('value', methods=['invalid_method'])
        
        # Test with insufficient data
        small_data = pd.DataFrame({'value': [1, 2, 3]})
        small_analyzer = TimeseriesAnalysis(small_data)
        small_visualizer = TimeseriesVisualizer(small_analyzer)
        with self.assertWarns(Warning):
            small_visualizer.plot_distribution_evolution('value')

if __name__ == '__main__':
    unittest.main()
    fig.data) >= 4)  # Should have ACF, PACF, Rolling Correlation, and Heatmap

    def test_plot_distribution_evolution(self):
        """Test distribution evolution plotting"""
        fig = self.visualizer.plot_distribution_evolution('value')
        self.assertIsInstance(fig, go.Figure)
        self.assertTrue(len(fig.data) >= 4)  # Should have distributions for different periods

    def test_create_animation(self):
        """Test animation creation"""
        fig = self.visualizer.create_animation('value')
        self.assertIsInstance(fig, go.Figure)
        self.assertTrue(hasattr(fig.layout, 'updatemenus'))

    def test_plot_feature_importance(self):
        """Test feature importance plotting"""
        features = pd.DataFrame({
            'feature1': np.random.randn(200),
            'feature2': np.random.randn(200)
        }, index=self.test_data.index)
        
        fig = self.visualizer.plot_feature_importance(features, self.test_data['value'])
        self.assertIsInstance(fig, go.Figure)
        self.assertTrue(len(fig.data) >= 4)  # Should have multiple feature importance plots

    def test_create_comprehensive_dashboard(self):
        """Test comprehensive dashboard creation"""
        fig = self.visualizer.create_comprehensive_dashboard('value')
        self.assertIsInstance(fig, go.Figure)
        self.assertTrue(len(fig.data) >= 9)  # Should have multiple analysis plots

    def test_plot_decomposition(self):
        """Test decomposition plotting"""
        fig = self.visualizer.plot_decomposition('value')
        self.assertIsInstance(fig, go.Figure)
        self.assertTrue(len(fig.data) >= 4)  # Should have original, trend, seasonal, and residual

    def test_plot_trend_analysis(self):
        """Test trend analysis plotting"""
        fig = self.visualizer.plot_trend_analysis('value')
        self.assertIsInstance(fig, go.Figure)
        self.assertTrue(len(fig.data) >= 3)  # Should have original, trend line, and residuals

    def test_plot_change_points(self):
        """Test change point plotting"""
        fig = self.visualizer.plot_change_points('value')
        self.assertIsInstance(fig, go.Figure)
        self.assertTrue(len(fig.data) >= 1)  # Should have at least original data

    def test_create_interactive_dashboard(self):
        """Test interactive dashboard creation"""
        fig = self.visualizer.create_interactive_dashboard('value')
        self.assertIsInstance(fig, go.Figure)
        self.assertTrue(len(fig.data) >= 6)  # Should have multiple interactive components

    def test_color_scheme(self):
        """Test color scheme functionality"""
        self.assertIn('primary', self.visualizer.color_scheme)
        self.assertIn('secondary', self.visualizer.color_scheme)
        self.assertIn('tertiary', self.visualizer.color_scheme)
        self.assertIn('background', self.visualizer.color_scheme)

    def test_subplot_creation(self):
        """Test subplot creation in various plots"""
        # Test comprehensive dashboard subplots
        dashboard = self.visualizer.create_comprehensive_dashboard('value')
        self.assertTrue(hasattr(dashboard.layout, 'annotations'))
        self.assertTrue(len(dashboard.layout.annotations) >= 9)
        
        # Test subplot grid structure
        self.assertTrue(hasattr(dashboard.layout, 'grid'))
        self.assertEqual(dashboard.layout.grid.rows, 3)
        self.assertEqual(dashboard.layout.grid.cols, 3)
        
        # Test subplot spacing
        self.assertGreater(dashboard.layout.grid.rowgap, 0)
        self.assertGreater(dashboard.layout.grid.colgap, 0)
        
        # Test subplot titles
        titles = [ann.text for ann in dashboard.layout.annotations]
        expected_titles = [
            'Time Series Plot',
            'Decomposition',
            'Seasonality Analysis',
            'Anomaly Detection',
            'Cycle Analysis',
            'Distribution',
            'Correlation Analysis',
            'Feature Importance',
            'Forecasting'
        ]
        for title in expected_titles:
            self.assertTrue(any(title in t for t in titles))
        
        # Test subplot axes
        for i in range(1, 10):
            self.assertTrue(hasattr(dashboard.layout, f'xaxis{i}'))
            self.assertTrue(hasattr(dashboard.layout, f'yaxis{i}'))
            
        # Test subplot data presence
        traces_per_subplot = {}
        for trace in dashboard.data:
            subplot = f"subplot{trace.xaxis.replace('x', '')}" if hasattr(trace, 'xaxis') else "subplot1"
            traces_per_subplot[subplot] = traces_per_subplot.get(subplot, 0) + 1
        
        # Verify each subplot has at least one trace
        for subplot in traces_per_subplot:
            self.assertGreater(traces_per_subplot[subplot], 0)

        # Test correlogram subplots
        correlogram = self.visualizer.plot_correlogram('value')
        self.assertEqual(len(correlogram.layout.annotations), 4)  # ACF, PACF, Rolling Correlation, Heatmap
        
        # Test distribution evolution subplots
        dist_evolution = self.visualizer.plot_distribution_evolution('value')
        self.assertEqual(len(dist_evolution.layout.annotations), 4)  # Rolling PDF, Q-Q Plot, Box Plot, Violin Plot
        
        # Test interactive dashboard subplots
        interactive = self.visualizer.create_interactive_dashboard('value')
        self.assertTrue(len(interactive.layout.annotations) >= 6)  # All main components
        
        # Test update menu positioning
        if hasattr(interactive.layout, 'updatemenus'):
            menu = interactive.layout.updatemenus[0]
            self.assertIn('x', menu)
            self.assertIn('y', menu)
            self.assertEqual(menu.xanchor, 'left')
            self.assertEqual(menu.yanchor, 'top')
                
    def test_decomposition_periods(self):
        """Test decomposition with different periods"""
        # Test with default period
        fig1 = self.visualizer.plot_decomposition('value')
        self.assertIsInstance(fig1, go.Figure)
        
        # Test with custom period
        fig2 = self.visualizer.plot_decomposition('value', period=7)
        self.assertIsInstance(fig2, go.Figure)
        
        # Test subplot components
        components = ['Original', 'Trend', 'Seasonal', 'Residual']
        for fig in [fig1, fig2]:
            subplot_titles = [ann.text for ann in fig.layout.annotations]
            for component in components:
                self.assertTrue(any(component in title for title in subplot_titles))

    def test_decomposition_customization(self):
        """Test decomposition plot customization"""
        fig = self.visualizer.plot_decomposition('value')
        
        # Test layout properties
        self.assertEqual(fig.layout.height, 800)
        self.assertEqual(fig.layout.width, 1200)
        self.assertTrue(fig.layout.showlegend)
        
        # Test y-axes titles
        y_titles = ['Value', 'Trend', 'Seasonal', 'Residual']
        for i, title in enumerate(y_titles, 1):
            yaxis = getattr(fig.layout, f'yaxis{i}')
            self.assertEqual(yaxis.title.text, title)

    def test_interactive_dashboard_full(self):
        """Test all components of interactive dashboard"""
        fig = self.visualizer.create_interactive_dashboard('value')
        
        # Test all required subplots
        required_titles = [
            'Price Action with Patterns',
            'Pattern Distribution',
            'Pattern Occurrences',
            'Volume Profile'
        ]
        subplot_titles = [ann.text for ann in fig.layout.annotations]
        for title in required_titles:
            self.assertTrue(any(title in t for t in subplot_titles))
        
        # Test interactive components
        self.assertTrue(hasattr(fig.layout, 'updatemenus'))
        buttons = fig.layout.updatemenus[0].buttons
        self.assertTrue(len(buttons) > 0)
        self.assertEqual(buttons[0].label, 'All Patterns')
        
        # Test trace configurations
        trace_types = [trace.type for trace in fig.data]
        required_types = ['candlestick', 'pie', 'bar', 'scatter']
        for req_type in required_types:
            self.assertIn(req_type, trace_types)

    def test_pattern_distribution_calculation(self):
        """Test pattern distribution calculation"""
        # Create visualizer with known patterns
        pattern_dist = self.visualizer._get_pattern_distribution()
        
        # Test distribution properties
        self.assertIsInstance(pattern_dist, pd.Series)
        self.assertTrue(len(pattern_dist) > 0)
        self.assertTrue(all(isinstance(v, (int, np.integer)) for v in pattern_dist.values))
        
        # Test with empty data
        empty_data = pd.DataFrame(index=pd.date_range('2023-01-01', periods=10))
        empty_analyzer = TimeseriesAnalysis(empty_data)
        empty_viz = TimeseriesVisualizer(empty_analyzer)
        empty_dist = empty_viz._get_pattern_distribution()
        self.assertTrue(empty_dist.empty)

    def test_volume_colors(self):
        """Test volume color generation"""
        colors = self.visualizer._get_volume_colors()
        
        # Test color list properties
        self.assertEqual(len(colors), len(self.test_data))
        self.assertTrue(all(c in [
            self.visualizer.color_scheme['volume_up'],
            self.visualizer.color_scheme['volume_down']
        ] for c in colors))
        
        # Test color assignment logic
        for i in range(len(self.test_data)):
            is_up = self.test_data['value'].iloc[i] >= self.test_data['value'].iloc[i-1]
            expected_color = (self.visualizer.color_scheme['volume_up'] if is_up 
                            else self.visualizer.color_scheme['volume_down'])
            self.assertEqual(colors[i], expected_color)

    def test_color_scheme_customization(self):
        """Test color scheme customization"""
        # Test with custom color scheme
        custom_colors = {
            'primary': '#000000',
            'secondary': '#ffffff',
            'tertiary': '#ff0000',
            'quaternary': '#00ff00',
            'background': '#0000ff',
            'grid': '#cccccc'
        }
        
        analyzer = TimeseriesAnalysis(self.test_data)
        viz = TimeseriesVisualizer(analyzer)
        viz.color_scheme = custom_colors
        
        fig = viz.create_comprehensive_dashboard('value')
        
        # Test color application
        traces = fig.data
        self.assertTrue(any(trace.line.color == custom_colors['primary'] 
                          for trace in traces if hasattr(trace, 'line')))
        
        # Test invalid color scheme
        with self.assertRaises(ValueError):
            viz.color_scheme = {'invalid': 'scheme'}

    def test_interactive_components(self):
        """Test all interactive components"""
        fig = self.visualizer.create_interactive_dashboard('value')
        
        # Test pattern selection menu
        menu = fig.layout.updatemenus[0]
        self.assertEqual(menu.direction, 'down')
        self.assertTrue(menu.showactive)
        
        # Test button functionality
        buttons = menu.buttons
        for button in buttons:
            self.assertIn('visible', button.args[0])
            self.assertIsInstance(button.args[0]['visible'], list)
            
        # Test hover templates
        for trace in fig.data:
            if hasattr(trace, 'hovertemplate'):
                self.assertIsNotNone(trace.hovertemplate)

    def test_subplot_configurations(self):
        """Test subplot configurations"""
        fig = self.visualizer.create_comprehensive_dashboard('value')
        
        # Test subplot grid structure
        self.assertTrue(hasattr(fig.layout, 'grid'))
        
        # Test subplot spacing
        self.assertGreater(fig.layout.grid.rowgap, 0)
        self.assertGreater(fig.layout.grid.colgap, 0)
        
        # Test subplot titles alignment
        annotations = fig.layout.annotations
        for ann in annotations:
            self.assertIn('y', ann)
            self.assertIn('x', ann)
            self.assertEqual(ann.font.size, 14)
            
    def test_error_handling_comprehensive(self):
        """Test comprehensive error handling"""
        # Test with invalid period
        with self.assertRaises(ValueError):
            self.visualizer.plot_decomposition('value', period=0)
            
        # Test with non-numeric data
        invalid_data = self.test_data.copy()
        invalid_data['value'] = 'invalid'
        invalid_analyzer = TimeseriesAnalysis(invalid_data)
        invalid_viz = TimeseriesVisualizer(invalid_analyzer)
        with self.assertRaises(TypeError):
            invalid_viz.create_comprehensive_dashboard('value')
            
        # Test with missing required columns
        missing_data = pd.DataFrame(index=self.test_data.index)
        missing_analyzer = TimeseriesAnalysis(missing_data)
        missing_viz = TimeseriesVisualizer(missing_analyzer)
        with self.assertRaises(KeyError):
            missing_viz.plot_correlogram('nonexistent')

    def test_secondary_y_axis(self):
        """Test plots with secondary y-axis"""
        # Add volume data to test secondary axis
        self.test_data['volume'] = np.random.randint(1000, 5000, len(self.test_data))
        viz = TimeseriesVisualizer(TimeseriesAnalysis(self.test_data))
        
        fig = viz.create_comprehensive_dashboard('value')
        
        # Check for secondary y-axis presence
        self.assertTrue(any(trace.yaxis == 'y2' for trace in fig.data))
        self.assertTrue(hasattr(fig.layout, 'yaxis2'))
        
        # Verify secondary axis properties
        secondary_axis = fig.layout.yaxis2
        self.assertEqual(secondary_axis.overlaying, 'y')
        self.assertEqual(secondary_axis.side, 'right')

    def test_animation_frames_details(self):
        """Test detailed animation frame properties"""
        fig = self.visualizer.create_animation('value')
        
        # Test frame properties
        self.assertTrue(hasattr(fig, 'frames'))
        self.assertGreater(len(fig.frames), 0)
        
        for frame in fig.frames:
            # Test frame data structure
            self.assertTrue(hasattr(frame, 'data'))
            self.assertGreaterEqual(len(frame.data), 1)
            
            # Test frame naming
            self.assertTrue(hasattr(frame, 'name'))
            self.assertTrue(frame.name.startswith('frame'))
            
            # Test data consistency
            for trace in frame.data:
                self.assertEqual(len(trace.x), len(trace.y))

        # Test animation controls
        self.assertTrue(hasattr(fig.layout, 'updatemenus'))
        menu = fig.layout.updatemenus[0]
        self.assertEqual(menu.buttons[0].label, 'Play')
        self.assertIn('animate', menu.buttons[0].method)

    def test_hover_template_customization(self):
        """Test hover data customization"""
        fig = self.visualizer.create_comprehensive_dashboard('value')
        
        # Test hover templates for different trace types
        for trace in fig.data:
            if trace.type == 'candlestick':
                self.assertIn('Open', trace.hovertext if hasattr(trace, 'hovertext') else '')
                self.assertIn('Close', trace.hovertext if hasattr(trace, 'hovertext') else '')
            elif trace.type == 'scatter':
                self.assertTrue(hasattr(trace, 'hovertemplate') or hasattr(trace, 'hovertext'))

    def test_pattern_sequence_visualization(self):
        """Test pattern visualization sequences"""
        fig = self.visualizer.create_interactive_dashboard('value')
        
        # Get all pattern traces
        pattern_traces = [trace for trace in fig.data 
                         if trace.mode == 'markers' and 'pattern' in trace.name.lower()]
        
        # Test pattern sequence
        for i in range(len(pattern_traces) - 1):
            current_pattern = pattern_traces[i]
            next_pattern = pattern_traces[i + 1]
            
            # Verify pattern markers
            self.assertIn('marker', dir(current_pattern))
            self.assertIn('marker', dir(next_pattern))
            
            # Verify pattern visibility
            self.assertTrue(hasattr(current_pattern, 'visible'))
            self.assertTrue(hasattr(next_pattern, 'visible'))

    def test_font_and_text_styles(self):
        """Test text style customizations"""
        fig = self.visualizer.create_comprehensive_dashboard('value')
        
        # Test title font
        self.assertTrue(hasattr(fig.layout.title, 'font'))
        self.assertEqual(fig.layout.title.font.size, 16)
        
        # Test annotation fonts
        for annotation in fig.layout.annotations:
            self.assertTrue(hasattr(annotation, 'font'))
            self.assertEqual(annotation.font.size, 14)
            self.assertEqual(annotation.font.color, self.visualizer.color_scheme['text'])
        
        # Test axis label fonts
        for i in range(1, 10):
            if hasattr(fig.layout, f'xaxis{i}'):
                self.assertTrue(hasattr(getattr(fig.layout, f'xaxis{i}').title, 'font'))
            if hasattr(fig.layout, f'yaxis{i}'):
                self.assertTrue(hasattr(getattr(fig.layout, f'yaxis{i}').title, 'font'))

    def test_legend_customization(self):
        """Test legend positioning and styling"""
        fig = self.visualizer.create_comprehensive_dashboard('value')
        
        # Test legend properties
        self.assertTrue(hasattr(fig.layout, 'legend'))
        self.assertEqual(fig.layout.legend.orientation, 'h')
        self.assertEqual(fig.layout.legend.yanchor, 'bottom')
        self.assertEqual(fig.layout.legend.y, 1.02)
        self.assertEqual(fig.layout.legend.xanchor, 'right')
        self.assertEqual(fig.layout.legend.x, 1)
        
        # Test legend font
        self.assertTrue(hasattr(fig.layout.legend, 'font'))
        self.assertEqual(fig.layout.legend.font.size, 10)