"""Enhanced time series analysis module for specialized-viz library."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
from scipy import stats, signal
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from scipy.fft import fft, fftfreq
import pywt

@dataclass
class TimeseriesConfig:
    """Configuration for time series analysis settings.
    
    Attributes:
        decomposition_method: Method for time series decomposition ('additive' or 'multiplicative')
        seasonal_periods: List of periods for seasonality analysis
        trend_window: Window size for trend calculations
        forecast_horizon: Number of periods to forecast
        model_type: Type of model to use ('ensemble', 'sarima', 'ets', 'lstm', 'nbeats', 'var')
        cv_folds: Number of cross-validation folds
        optimization_trials: Number of hyperparameter optimization trials
        confidence_level: Confidence level for prediction intervals
        use_exogenous: Whether to use exogenous variables in models
        lstm_units: Number of LSTM units
        lstm_epochs: Number of training epochs for LSTM
        cycle_max_period: Maximum period for cycle analysis
    """
    decomposition_method: str = 'additive'
    seasonal_periods: List[int] = field(default_factory=lambda: [5, 21, 63, 252])  # Daily, Weekly, Monthly, Yearly
    trend_window: int = 20
    forecast_horizon: int = 30
    model_type: str = 'ensemble'
    cv_folds: int = 5
    optimization_trials: int = 100
    confidence_level: float = 0.95
    use_exogenous: bool = False
    lstm_units: int = 50
    lstm_epochs: int = 100
    cycle_max_period: int = 252  # Default to one trading year
    anomaly_threshold: float = 0.1
        
class TimeseriesAnalysis:
    """Enhanced class for time series analysis operations."""
    
    def __init__(self, data: pd.DataFrame, config: Optional[TimeseriesConfig] = None):
        """Initialize the time series analyzer.
        
        Args:
            data: DataFrame with datetime index and value columns
            config: Optional configuration settings
        """
        self.data = data
        self.config = config or TimeseriesConfig()
        self._validate_data()
        
    def _validate_data(self) -> None:
        """Validate and prepare input data."""
        if not isinstance(self.data.index, pd.DatetimeIndex):
            try:
                self.data.index = pd.to_datetime(self.data.index)
            except Exception as e:
                raise ValueError(f"Index must be convertible to datetime. Error: {str(e)}")
                
        # Check for missing values
        if self.data.isnull().any().any():
            print("Warning: Data contains missing values. Consider using TimeseriesPreprocessor.")

    def analyze_seasonality(self, column: str) -> Dict:
        """Perform comprehensive seasonality analysis.
        
        Args:
            column: Name of the column to analyze
            
        Returns:
            Dictionary containing seasonality metrics and patterns
        """
        series = self.data[column]
        results = {}
        
        # Multiple seasonality detection
        for period in self.config.seasonal_periods:
            if len(series) >= period * 2:  # Need at least 2 full cycles
                seasonal_strength = self._calculate_seasonal_strength(series, period)
                results[f'seasonal_strength_{period}'] = seasonal_strength
        
        # Season-on-season comparison
        results['seasonal_changes'] = self._analyze_seasonal_changes(series)
        
        # Holiday effects
        results['holiday_effects'] = self._detect_holiday_effects(series)
        
        return results
        
    def _calculate_seasonal_strength(self, series: pd.Series, period: int) -> float:
        """Calculate strength of seasonality for given period."""
        # Decompose series
        decomposition = self.decompose(series.name)
        seasonal = decomposition['seasonal']
        residual = decomposition['residual']
        
        # Calculate variance ratio
        var_seasonal = np.var(seasonal)
        var_residual = np.var(residual)
        
        if var_seasonal + var_residual == 0:
            return 0
            
        return var_seasonal / (var_seasonal + var_residual)
        
    def _analyze_seasonal_changes(self, series: pd.Series) -> Dict:
        """Analyze how seasonality changes over time."""
        results = {}
        
        # Split series into years
        yearly_data = [g for _, g in series.groupby(pd.Grouper(freq='Y'))]
        
        if len(yearly_data) > 1:
            # Compare seasonal patterns between consecutive years
            for i in range(len(yearly_data) - 1):
                correlation = yearly_data[i].corr(yearly_data[i + 1])
                results[f'year_{i}_to_{i+1}_correlation'] = correlation
                
        return results
        
    def _detect_holiday_effects(self, series: pd.Series) -> Dict:
        """Detect and quantify holiday effects in the series."""
        # Create DataFrame with same index as series
        holidays = pd.DataFrame(index=series.index)
        
        # Add holiday indicators using proper date attributes
        holidays['is_weekend'] = series.index.dayofweek.isin([5, 6])
        holidays['month_end'] = series.index.is_month_end
        
        # Handle year end correctly
        holidays['fiscal_year_end'] = (series.index.month == 12) & (series.index.day == 31)
        
        effects = {}
        for holiday in holidays.columns:
            # Calculate mean difference for holiday vs non-holiday
            holiday_values = series[holidays[holiday]]
            non_holiday_values = series[~holidays[holiday]]
            
            if len(holiday_values) > 0 and len(non_holiday_values) > 0:
                holiday_mean = holiday_values.mean()
                non_holiday_mean = non_holiday_values.mean()
                effects[holiday] = holiday_mean - non_holiday_mean
            else:
                effects[holiday] = 0.0
            
        return effects
    
    def detect_anomalies(self, column: str, methods: List[str] = None) -> Dict[str, pd.Series]:
        """Detect anomalies using multiple methods.
        
        Args:
            column: Name of the column to analyze
            methods: List of methods to use ('iqr', 'zscore', 'isolation_forest', 'dbscan', 'lof')
            
        Returns:
            Dictionary of boolean Series indicating anomalies
        """
        if methods is None:
            methods = ['iqr', 'zscore', 'isolation_forest', 'dbscan', 'lof']
            
        series = self.data[column]
        results = {}
        
        for method in methods:
            if method == 'iqr':
                results['iqr'] = self._detect_iqr_anomalies(series)
            elif method == 'zscore':
                results['zscore'] = self._detect_zscore_anomalies(series)
            elif method == 'isolation_forest':
                results['isolation_forest'] = self._detect_isolation_forest_anomalies(series)
            elif method == 'dbscan':
                results['dbscan'] = self._detect_dbscan_anomalies(series)
            elif method == 'lof':
                results['lof'] = self._detect_lof_anomalies(series)
                
        return results
        
    def _detect_iqr_anomalies(self, series: pd.Series) -> pd.Series:
        """Detect anomalies using IQR method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (series < lower_bound) | (series > upper_bound)
        
    def _detect_zscore_anomalies(self, series: pd.Series) -> pd.Series:
        """Detect anomalies using Z-score method."""
        z_scores = np.abs(stats.zscore(series))
        return pd.Series(z_scores > 3, index=series.index)
        
    def _detect_isolation_forest_anomalies(self, series: pd.Series) -> pd.Series:
        """Detect anomalies using Isolation Forest."""
        clf = IsolationForest(contamination=self.config.anomaly_threshold, random_state=42)
        values = series.values.reshape(-1, 1)
        predictions = clf.fit_predict(values)
        return pd.Series(predictions == -1, index=series.index)
        
    def _detect_dbscan_anomalies(self, series: pd.Series) -> pd.Series:
        """Detect anomalies using DBSCAN."""
        clustering = DBSCAN(eps=0.5, min_samples=5)
        values = series.values.reshape(-1, 1)
        predictions = clustering.fit_predict(values)
        return pd.Series(predictions == -1, index=series.index)
        
    def _detect_lof_anomalies(self, series: pd.Series) -> pd.Series:
        """Detect anomalies using Local Outlier Factor."""
        lof = LocalOutlierFactor(contamination=self.config.anomaly_threshold)
        values = series.values.reshape(-1, 1)
        predictions = lof.fit_predict(values)
        return pd.Series(predictions == -1, index=series.index)

    def analyze_cycles(self, column: str) -> Dict:
        """Perform comprehensive cycle analysis.
        
        Args:
            column: Name of the column to analyze
            
        Returns:
            Dictionary containing cycle analysis results
        """
        series = self.data[column]
        results = {}
        
        # Fourier analysis
        results['fourier'] = self._fourier_analysis(series)
        
        # Wavelet analysis
        results['wavelet'] = self._wavelet_analysis(series)
        
        # Cycle periodicity
        results['periodicity'] = self._detect_periodicity(series)
        
        # Phase analysis
        results['phase'] = self._analyze_phase(series)
        
        return results
        
    def _fourier_analysis(self, series: pd.Series) -> Dict:
        """Perform Fourier analysis."""
        # Compute FFT
        fft_values = fft(series.values)
        freqs = fftfreq(len(series))
        
        # Get dominant frequencies
        dominant_freq_idx = np.argsort(np.abs(fft_values))[-5:]  # Top 5 frequencies
        
        return {
            'dominant_frequencies': freqs[dominant_freq_idx],
            'amplitudes': np.abs(fft_values)[dominant_freq_idx]
        }
        
    def _wavelet_analysis(self, series: pd.Series) -> Dict:
        """Perform wavelet analysis."""
        # Use continuous wavelet transform
        wavelet = 'cmor1.0-0.5'  # Explicit parameters instead of just 'cmor'
        scales = np.arange(1, min(len(series) // 2, self.config.cycle_max_period))
        coefficients, frequencies = pywt.cwt(series.values, scales, wavelet)

        # Find dominant scales
        power = np.sum(np.abs(coefficients), axis=1)
        dominant_scales = scales[np.argsort(power)[-5:]]  # Top 5 scales
        
        return {
            'dominant_scales': dominant_scales,
            'power_spectrum': power
        }
        
    def _detect_periodicity(self, series: pd.Series) -> Dict:
        """Detect periodic components."""
        # Calculate autocorrelation with explicit lags
        max_lag = min(len(series) // 2, self.config.cycle_max_period)
        lags = range(1, max_lag)
        
        # Calculate autocorrelation for each lag
        autocorr = [series.autocorr(lag=lag) for lag in lags]
        autocorr = pd.Series(autocorr, index=lags)
        
        # Find peaks in autocorrelation
        peaks, properties = signal.find_peaks(autocorr, height=0.1)
        
        return {
            'periodic_lengths': peaks,
            'correlation_strength': autocorr[peaks],
            'autocorrelation': autocorr  # Return full autocorrelation for visualization
        }
        
    def _analyze_phase(self, series: pd.Series) -> Dict:
        """Analyze phase relationships."""
        hilbert = signal.hilbert(series.values)
        analytic_signal = pd.Series(hilbert, index=series.index)
        
        phase = np.angle(analytic_signal)
        amplitude_envelope = np.abs(analytic_signal)
        
        return {
            'phase': phase,
            'amplitude_envelope': amplitude_envelope
        }

    def causality_tests(self, column1: str, column2: str) -> Dict:
        """Perform causality analysis between two series.
        
        Args:
            column1: First column name
            column2: Second column name
            
        Returns:
            Dictionary containing causality test results
        """
        series1 = self.data[column1]
        series2 = self.data[column2]
        results = {}
        
        # Granger causality
        results['granger'] = self._granger_causality(series1, series2)
        
        # Cross-correlation
        results['cross_correlation'] = self._cross_correlation(series1, series2)
        
        # Transfer entropy
        results['transfer_entropy'] = self._transfer_entropy(series1, series2)
        
        # Mutual information
        results['mutual_information'] = self._mutual_information(series1, series2)
        
        return results
        
    def _granger_causality(self, series1: pd.Series, series2: pd.Series) -> Dict:
        """Perform Granger causality test."""
        from statsmodels.tsa.stattools import grangercausalitytests
        
        # Prepare data
        data = pd.concat([series1, series2], axis=1)
        
        # Test both directions
        results_1_to_2 = grangercausalitytests(data, maxlag=self.config.causality_max_lag)
        results_2_to_1 = grangercausalitytests(data.iloc[:, ::-1], maxlag=self.config.causality_max_lag)
        
        return {
            '1_causes_2': results_1_to_2,
            '2_causes_1': results_2_to_1
        }
        
    def _cross_correlation(self, series1: pd.Series, series2: pd.Series) -> Dict:
        """Calculate cross-correlation."""
        correlations = []
        lags = range(-self.config.causality_max_lag, self.config.causality_max_lag + 1)
        
        for lag in lags:
            if lag < 0:
                corr = series1.corr(series2.shift(-lag))
            else:
                corr = series1.corr(series2.shift(lag))
            correlations.append(corr)
            
        return {
            'lags': lags,
            'correlations': correlations,
            'max_correlation_lag': lags[np.argmax(np.abs(correlations))]
        }
        
    def _transfer_entropy(self, series1: pd.Series, series2: pd.Series) -> float:
        """Calculate transfer entropy (simplified version)."""
        # Discretize the data
        bins = 5
        x = pd.qcut(series1, bins, labels=False)
        y = pd.qcut(series2, bins, labels=False)
        
        # Calculate joint and marginal probabilities
        p_joint = np.histogram2d(x, y, bins=bins)[0]
        p_joint = p_joint / np.sum(p_joint)
        
        p_x = np.sum(p_joint, axis=1)
        p_y = np.sum(p_joint, axis=0)
        
        # Calculate transfer entropy
        entropy = 0
        for i in range(bins):
            for j in range(bins):
                if p_joint[i,j] > 0:
                    entropy += p_joint[i,j] * np.log(p_joint[i,j] / (p_x[i] * p_y[j]))
                    
        return entropy
        
    def _mutual_information(self, series1: pd.Series, series2: pd.Series) -> float:
        """Calculate mutual information."""
        # Discretize the data
        bins = 5
        x = pd.qcut(series1, bins, labels=False)
        y = pd.qcut(series2, bins, labels=False)
        
        # Calculate mutual information using scikit-learn
        from sklearn.metrics import mutual_info_score
        mi = mutual_info_score(x, y)
        
        return mi
        
    def analyze_stationarity(self, column: str) -> Dict:
        """Analyze time series stationarity.
        
        Args:
            column: Name of column to analyze
            
        Returns:
            Dictionary with stationarity test results
        """
        from statsmodels.tsa.stattools import adfuller, kpss
        
        series = self.data[column]
        results = {}
        
        # Augmented Dickey-Fuller test
        adf_test = adfuller(series.dropna())
        results['adf_test'] = {
            'statistic': adf_test[0],
            'p_value': adf_test[1],
            'critical_values': adf_test[4],
            'is_stationary': adf_test[1] < 0.05
        }
        
        # KPSS test
        kpss_test = kpss(series.dropna())
        results['kpss_test'] = {
            'statistic': kpss_test[0],
            'p_value': kpss_test[1],
            'critical_values': kpss_test[3],
            'is_trend_stationary': kpss_test[1] > 0.05
        }
        
        # Rolling statistics
        results['rolling_stats'] = {
            'mean': series.rolling(window=30).mean(),
            'std': series.rolling(window=30).std(),
            'is_stable_mean': self._check_stable_statistics(series)
        }
        
        return results
    
    def _check_stable_statistics(self, series: pd.Series) -> bool:
        """Check if series has stable statistical properties."""
        # Split series into segments and compare their means
        segments = np.array_split(series, 4)
        means = [segment.mean() for segment in segments]
        return stats.variation(means) < 0.1  # Threshold for stability
    
    def detect_structural_breaks(self, column: str) -> Dict:
        """Detect structural breaks in time series.
        
        Args:
            column: Name of column to analyze
            
        Returns:
            Dictionary with structural break analysis
        """
        from statsmodels.stats.diagnostic import breaks_cusumolsresid
        
        series = self.data[column]
        results = {}
        
        # CUSUM test
        cusum_test = breaks_cusumolsresid(series.values)
        results['cusum_test'] = {
            'statistic': cusum_test[0],
            'p_value': cusum_test[1],
            'has_break': cusum_test[1] < 0.05
        }
        
        # Chow test for known break points
        results['break_points'] = self._detect_chow_breaks(series)
        
        # Bai-Perron test for multiple breaks
        results['multiple_breaks'] = self._detect_multiple_breaks(series)
        
        return results
    
    def _detect_chow_breaks(self, series: pd.Series) -> List[Dict]:
        """Detect break points using Chow test."""
        from statsmodels.stats.diagnostic import het_breuschpagan
        breaks = []
        
        # Test potential break points
        for i in range(len(series) // 4, 3 * len(series) // 4):
            subseries1 = series[:i]
            subseries2 = series[i:]
            
            # Perform Chow test
            bp_test = het_breuschpagan(subseries1, subseries2.values.reshape(-1, 1))
            
            if bp_test[1] < 0.05:  # Significant break point
                breaks.append({
                    'index': series.index[i],
                    'statistic': bp_test[0],
                    'p_value': bp_test[1]
                })
        
        return breaks
    
    def _detect_multiple_breaks(self, series: pd.Series) -> List[Dict]:
        """Detect multiple structural breaks."""
        from statsmodels.tsa.stattools import adfuller
        breaks = []
        min_segment = len(series) // 10  # Minimum segment length
        
        # Recursive detection
        def find_breaks(start_idx: int, end_idx: int):
            if end_idx - start_idx < min_segment:
                return
                
            segment = series.iloc[start_idx:end_idx]
            
            # Test for break in middle of segment
            mid = (start_idx + end_idx) // 2
            subseries1 = segment[:mid-start_idx]
            subseries2 = segment[mid-start_idx:]
            
            # Compare statistical properties
            if (abs(subseries1.mean() - subseries2.mean()) > subseries1.std() or
                abs(subseries1.std() - subseries2.std()) / subseries1.std() > 0.2):
                breaks.append({
                    'index': series.index[mid],
                    'type': 'mean_variance_break'
                })
                
                # Recurse on both segments
                find_breaks(start_idx, mid)
                find_breaks(mid, end_idx)
        
        find_breaks(0, len(series))
        return sorted(breaks, key=lambda x: x['index'])
    
    def analyze_nonlinearity(self, column: str) -> Dict:
        """Analyze nonlinear patterns in time series.
        
        Args:
            column: Name of column to analyze
            
        Returns:
            Dictionary with nonlinearity analysis
        """
        series = self.data[column]
        results = {}
        
        # Terasvirta Neural Network Test
        results['terasvirta_test'] = self._terasvirta_test(series)
        
        # BDS Test
        results['bds_test'] = self._bds_test(series)
        
        # Correlation dimension
        results['correlation_dimension'] = self._correlation_dimension(series)
        
        # Largest Lyapunov exponent
        results['lyapunov_exponent'] = self._largest_lyapunov(series)
        
        return results
    
    def _terasvirta_test(self, series: pd.Series) -> Dict:
        """Perform Terasvirta Neural Network test."""
        from statsmodels.stats.diagnostic import acorr_ljungbox
        
        # Prepare data
        X = series.shift(1).dropna()
        y = series[1:]
        
        # Fit neural network terms
        X_quad = X**2
        X_cubic = X**3
        
        # Perform LM test
        residuals = y - X
        lm_test = acorr_ljungbox(residuals, lags=[5])
        
        return {
            'statistic': lm_test.iloc[0, 0],
            'p_value': lm_test.iloc[0, 1],
            'is_nonlinear': lm_test.iloc[0, 1] < 0.05
        }
    
    def _bds_test(self, series: pd.Series) -> Dict:
        """Perform BDS test for independence."""
        # Calculate correlation integral
        def correlation_integral(data, epsilon):
            n = len(data)
            count = 0
            for i in range(n):
                for j in range(i+1, n):
                    if abs(data[i] - data[j]) < epsilon:
                        count += 1
            return 2 * count / (n * (n-1))
        
        # Standardize series
        std_series = (series - series.mean()) / series.std()
        
        # Calculate BDS statistic for different dimensions
        dimensions = range(2, 6)
        bds_stats = []
        
        for m in dimensions:
            c_m = correlation_integral(std_series, 1.5)  # epsilon = 1.5 std
            c_1 = correlation_integral(std_series, 1.5)**(m)
            
            # Calculate standard error
            n = len(std_series)
            variance = (c_1 * (1 - c_1)) / n
            bds_stat = (c_m - c_1) / np.sqrt(variance)
            bds_stats.append(bds_stat)
        
        return {
            'statistics': bds_stats,
            'dimensions': list(dimensions),
            'is_independent': all(abs(stat) < 1.96 for stat in bds_stats)  # 5% significance
        }
    
    def _correlation_dimension(self, series: pd.Series, max_dim: int = 5) -> Dict:
        """Calculate correlation dimension."""
        def grassberger_procaccia(data, dim, r):
            n = len(data)
            count = 0
            for i in range(n-dim+1):
                for j in range(i+1, n-dim+1):
                    if max(abs(data[i+k] - data[j+k]) for k in range(dim)) < r:
                        count += 1
            return 2 * count / ((n-dim+1) * (n-dim))
        
        # Normalize series
        normalized = (series - series.mean()) / series.std()
        
        # Calculate correlation dimension for different embedding dimensions
        dims = range(1, max_dim + 1)
        correlations = []
        
        for dim in dims:
            r = 0.5  # Fixed radius
            correlation = grassberger_procaccia(normalized.values, dim, r)
            correlations.append(correlation)
        
        # Estimate correlation dimension from slope
        slope = np.polyfit(np.log(dims), np.log(correlations), 1)[0]
        
        return {
            'correlation_dimension': slope,
            'dimensions': list(dims),
            'correlations': correlations
        }
    
    def _largest_lyapunov(self, series: pd.Series, 
                         embedding_dim: int = 3, 
                         delay: int = 1) -> Dict:
        """Calculate largest Lyapunov exponent."""
        # Create time-delay embedding
        embedded = np.array([
            series.values[i:i-embedding_dim*delay:-delay] 
            for i in range(len(series)-embedding_dim*delay)
        ])
        
        # Find nearest neighbors
        def find_nearest_neighbor(point, excluded_range):
            distances = np.linalg.norm(embedded - point, axis=1)
            distances[excluded_range] = np.inf
            return np.argmin(distances)
        
        # Calculate divergence
        lyap = []
        for i in range(len(embedded)):
            j = find_nearest_neighbor(embedded[i], 
                                   range(max(0, i-5), min(len(embedded), i+5)))
            
            # Track divergence
            div = np.linalg.norm(embedded[i] - embedded[j])
            if div > 0:
                lyap.append(np.log(div))
        
        # Estimate Lyapunov exponent from slope
        times = np.arange(len(lyap))
        slope = np.polyfit(times, lyap, 1)[0]
        
        return {
            'lyapunov_exponent': slope,
            'is_chaotic': slope > 0,
            'divergence_rates': lyap
        }

    def decompose(self, column: str, period: int = None) -> Dict[str, pd.Series]:
        """Decompose time series into trend, seasonal, and residual components.
        
        Args:
            column: Name of the column to decompose
            period: Seasonal period to use (defaults to first period in config)
            
        Returns:
            Dictionary containing trend, seasonal, and residual components
        """
        series = self.data[column]
        
        # Use first seasonal period if none specified
        if period is None:
            period = self.config.seasonal_periods[0]
        
        # Calculate trend using rolling mean
        trend = series.rolling(window=self.config.trend_window, 
                             center=True).mean()
        
        # Calculate seasonal component
        seasonal_means = pd.DataFrame()
        for i in range(period):
            seasonal_means[i] = series[i::period].reset_index(drop=True)
        seasonal_pattern = seasonal_means.mean()
        
        # Normalize seasonal pattern
        seasonal_pattern = seasonal_pattern - seasonal_pattern.mean()
        
        # Create full seasonal component
        seasonal = pd.Series(index=series.index, dtype=float)
        for i in range(len(series)):
            seasonal.iloc[i] = seasonal_pattern[i % period]
        
        if self.config.decomposition_method == 'multiplicative':
            residual = series / (trend * seasonal)
        else:  # additive
            residual = series - trend - seasonal
            
        return {
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual,
            'original': series
        }
    def analyze_trend(self, column: str) -> Dict[str, Union[float, pd.Series]]:
        """Analyze trend characteristics of the time series.
        
        Args:
            column: Name of the column to analyze
            
        Returns:
            Dictionary containing trend metrics and indicators
        """
        series = self.data[column]
        
        # Calculate linear trend
        X = np.arange(len(series)).reshape(-1, 1)
        y = series.values.reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, y)
        
        trend_line = pd.Series(model.predict(X).flatten(), index=series.index)
        
        # Calculate trend strength
        r_squared = model.score(X, y)
        
        # Calculate trend direction and significance
        slope = model.coef_[0][0]
        mann_kendall = self._mann_kendall_test(series)
        
        return {
            'trend_line': trend_line,
            'slope': slope,
            'r_squared': r_squared,
            'mann_kendall_stats': mann_kendall,
            'is_significant': mann_kendall['p_value'] < 0.05
        }
    
    def _mann_kendall_test(self, series: pd.Series) -> Dict[str, float]:
        """Perform Mann-Kendall trend test.
        
        Args:
            series: Time series to test
            
        Returns:
            Dictionary containing test statistics
        """
        n = len(series)
        s = 0
        for i in range(n-1):
            for j in range(i+1, n):
                s += np.sign(series.iloc[j] - series.iloc[i])
        
        # Calculate variance
        unique_values = np.unique(series)
        g = len(unique_values)
        if n == g:  # No ties
            var_s = (n * (n - 1) * (2 * n + 5)) / 18
        else:  # Handle ties
            tp = np.zeros(g)
            for i in range(g):
                tp[i] = sum(series == unique_values[i])
            var_s = (n * (n - 1) * (2 * n + 5) - np.sum(tp * (tp - 1) * (2 * tp + 5))) / 18
        
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0
        
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        return {
            'statistic': s,
            'variance': var_s,
            'z_score': z,
            'p_value': p_value,
            'trend': 'increasing' if s > 0 else 'decreasing' if s < 0 else 'no trend'
        }
    
    def detect_change_points(self, column: str, 
                           method: str = 'cusum') -> pd.Series:
        """Detect structural changes in the time series.
        
        Args:
            column: Name of the column to analyze
            method: Detection method ('cusum' or 'pettitt')
            
        Returns:
            Series indicating change points
        """
        series = self.data[column]
        
        if method == 'cusum':
            return self._cusum_detection(series)
        elif method == 'pettitt':
            return self._pettitt_detection(series)
        else:
            raise ValueError("Method must be either 'cusum' or 'pettitt'")
    
    def _cusum_detection(self, series: pd.Series) -> pd.Series:
        """CUSUM change point detection."""
        mean = series.mean()
        std = series.std()
        cusums = np.zeros(len(series))
        change_points = pd.Series(False, index=series.index)
        
        # Calculate cumulative sums
        for i in range(1, len(series)):
            cusums[i] = cusums[i-1] + (series.iloc[i] - mean) / std
            
        # Detect significant changes
        threshold = 1.96 * np.sqrt(len(series))  # 95% confidence
        for i in range(1, len(series)-1):
            if abs(cusums[i]) > threshold:
                change_points.iloc[i] = True
                
        return change_points
    
    def _pettitt_detection(self, series: pd.Series) -> pd.Series:
        """Pettitt change point detection."""
        n = len(series)
        change_points = pd.Series(False, index=series.index)
        
        # Calculate U statistics
        u_stats = np.zeros(n)
        for t in range(1, n):
            u_stats[t] = 2 * sum(series.iloc[:t] > series.iloc[t:]) - t * (n - t)
            
        # Find most significant change point
        k = np.argmax(np.abs(u_stats))
        u_k = u_stats[k]
        
        # Calculate significance
        p_value = 2 * np.exp(-6 * u_k**2 / (n**3 + n**2))
        
        if p_value < 0.05:  # 95% confidence
            change_points.iloc[k] = True
            
        return change_points