"""Enhanced time series forecasting module for specialized-viz library."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import norm
import optuna
from datetime import datetime, timedelta

@dataclass
class ForecastConfig:
    """Configuration for forecasting settings.
    
    Attributes:
        forecast_horizon: Number of periods to forecast
        train_test_split: Proportion of data for training
        cv_folds: Number of cross-validation folds
        features_lag: List of lag periods for feature creation
        features_window: List of rolling window sizes
        model_type: Type of model to use ('ensemble', 'linear', 'gbm')
        optimization_trials: Number of hyperparameter optimization trials
        confidence_level: Confidence level for prediction intervals
    """
    forecast_horizon: int = 30
    train_test_split: float = 0.8
    cv_folds: int = 5
    features_lag: List[int] = field(default_factory=lambda: [1, 7, 14, 30])
    features_window: List[int] = field(default_factory=lambda: [7, 14, 30, 90])
    model_type: str = 'ensemble'
    optimization_trials: int = 100
    confidence_level: float = 0.95
    model_type: str = 'ensemble'  # Added options: 'prophet', 'var', 'lstm', 'nbeats'
    use_exogenous: bool = False
    lstm_units: int = 50
    lstm_epochs: int = 100
    prophet_params: Dict = field(default_factory=lambda: {
        'changepoint_prior_scale': 0.05,
        'seasonality_prior_scale': 10,
        'holidays_prior_scale': 10
    })
        forecast_horizon: int = 30
    train_test_split: float = 0.8
    cv_folds: int = 5
    features_lag: List[int] = field(default_factory=lambda: [1, 7, 14, 30])
    features_window: List[int] = field(default_factory=lambda: [7, 14, 30, 90])
    model_type: str = 'ensemble'
    optimization_trials: int = 100
    confidence_level: float = 0.95
    seasonal_periods: List[int] = field(default_factory=lambda: [5, 21, 63, 252])  # Daily, Weekly, Monthly, Yearly


class TimeseriesForecasting:
    """Enhanced forecasting class with advanced capabilities."""
    
    def __init__(self, data: pd.DataFrame, config: Optional[ForecastConfig] = None):
        """Initialize forecasting module.
        
        Args:
            data: DataFrame with datetime index
            config: Optional forecasting configuration
        """
        self.data = data
        self.config = config or ForecastConfig()
        self.models = {}
        self.feature_importance = {}
        self.metrics = {}
        
    def create_features(self, column: str) -> pd.DataFrame:
        """Create comprehensive feature set for forecasting.
        
        Args:
            column: Target column name
            
        Returns:
            DataFrame with engineered features
        """
        series = self.data[column]
        features = pd.DataFrame(index=series.index)
        
        # Lag features
        for lag in self.config.features_lag:
            features[f'lag_{lag}'] = series.shift(lag)
            
        # Rolling window features
        for window in self.config.features_window:
            features[f'rolling_mean_{window}'] = series.rolling(window=window).mean()
            features[f'rolling_std_{window}'] = series.rolling(window=window).std()
            features[f'rolling_min_{window}'] = series.rolling(window=window).min()
            features[f'rolling_max_{window}'] = series.rolling(window=window).max()
            
        # Date features
        features['month'] = series.index.month
        features['day_of_week'] = series.index.dayofweek
        features['day_of_year'] = series.index.dayofyear
        features['week_of_year'] = series.index.isocalendar().week
        features['is_month_end'] = series.index.is_month_end
        features['is_month_start'] = series.index.is_month_start
        features['is_quarter_end'] = series.index.is_quarter_end
        
        # Cyclical encoding of time features
        for col in ['month', 'day_of_week', 'day_of_year']:
            max_val = features[col].max()
            features[f'{col}_sin'] = np.sin(2 * np.pi * features[col]/max_val)
            features[f'{col}_cos'] = np.cos(2 * np.pi * features[col]/max_val)
            
        # Interaction features
        for lag1, lag2 in zip(self.config.features_lag[:-1], self.config.features_lag[1:]):
            features[f'lag_diff_{lag1}_{lag2}'] = (
                features[f'lag_{lag1}'] - features[f'lag_{lag2}']
            )
            
        return features

    def ensemble_forecast(self, features: pd.DataFrame, target: pd.Series) -> Dict:
        # Initialize base models
        models = {
            'rf': RandomForestRegressor(random_state=42),
            'gbm': GradientBoostingRegressor(random_state=42),
            'linear': LinearRegression()
        }
        
        # Train models and get predictions
        predictions = {}
        for name, model in models.items():
            model.fit(features, target)
            predictions[name] = model.predict(features)
            self.models[name] = model
            
        # Add this block here, after base model predictions but before weights optimization
        if self.config.model_type in ['prophet', 'var', 'lstm', 'nbeats']:
            specialized_pred = getattr(self, f"{self.config.model_type}_forecast")(features, target)
            predictions[self.config.model_type] = specialized_pred['forecast']
                
        # Combine predictions with weighted average
        weights = self._optimize_ensemble_weights(predictions, target)
        ensemble_pred = sum(pred * weights[name] 
                        for name, pred in predictions.items())
        
        return {
            'predictions': ensemble_pred,
            'weights': weights,
            'model_predictions': predictions
        }

    def _optimize_ensemble_weights(self, predictions: Dict[str, np.ndarray], 
                                 actual: pd.Series) -> Dict[str, float]:
        """Optimize ensemble weights using Optuna."""
        def objective(trial):
            weights = {
                name: trial.suggest_float(f'weight_{name}', 0, 1)
                for name in predictions.keys()
            }
            # Normalize weights
            total = sum(weights.values())
            weights = {k: v/total for k, v in weights.items()}
            
            # Calculate weighted prediction
            weighted_pred = sum(pred * weights[name] 
                              for name, pred in predictions.items())
            return mean_squared_error(actual, weighted_pred)
            
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.config.optimization_trials)
        
        # Get best weights
        best_weights = {
            name: study.best_params[f'weight_{name}']
            for name in predictions.keys()
        }
        # Normalize
        total = sum(best_weights.values())
        return {k: v/total for k, v in best_weights.items()}

    def hierarchical_forecast(self, column: str, groups: List[str]) -> Dict:
        """Perform hierarchical forecasting.
        
        Args:
            column: Target column name
            groups: List of grouping columns
            
        Returns:
            Dictionary with hierarchical forecasts
        """
        forecasts = {}
        reconciled = {}
        
        # Bottom-up forecasts
        for group in groups:
            group_data = self.data.groupby(group)[column]
            group_forecasts = {}
            
            for name, series in group_data:
                features = self.create_features(series.name)
                forecast = self.ensemble_forecast(features.dropna(), 
                                               series[features.dropna().index])
                group_forecasts[name] = forecast['predictions']
                
            forecasts[group] = group_forecasts
            
        # Reconcile forecasts
        reconciled = self._reconcile_hierarchical(forecasts, column)
        
        return {
            'base_forecasts': forecasts,
            'reconciled_forecasts': reconciled
        }
        
    def _reconcile_hierarchical(self, forecasts: Dict, column: str) -> Dict:
        """Reconcile hierarchical forecasts using MinT (Minimum Trace) method."""
        # Implementation of MinT reconciliation
        reconciled = forecasts.copy()
        
        # Calculate variance-covariance matrix of forecast errors
        errors = {}
        for level, level_forecasts in forecasts.items():
            level_errors = {}
            for name, forecast in level_forecasts.items():
                actual = self.data[self.data.index.isin(forecast.index)][column]
                error = actual - forecast
                level_errors[name] = error
            errors[level] = pd.DataFrame(level_errors)
            
        # Compute reconciliation matrix
        S = self._compute_summing_matrix(forecasts)
        W = np.linalg.inv(sum(error.cov().values 
                             for error in errors.values()))
        R = S @ np.linalg.inv(S.T @ W @ S) @ S.T @ W
        
        # Reconcile forecasts
        for level in reconciled:
            for name in reconciled[level]:
                reconciled[level][name] = R @ forecasts[level][name]
                
        return reconciled
        
    def _compute_summing_matrix(self, forecasts: Dict) -> np.ndarray:
        """Compute summing matrix for hierarchical reconciliation."""
        n_bottom = len(forecasts[list(forecasts.keys())[-1]])
        n_total = sum(len(level) for level in forecasts.values())
        
        S = np.zeros((n_total, n_bottom))
        row = 0
        
        for level, level_forecasts in forecasts.items():
            for name in level_forecasts:
                if level == list(forecasts.keys())[-1]:  # Bottom level
                    S[row, list(level_forecasts.keys()).index(name)] = 1
                else:
                    # Add aggregation constraints
                    children = self._get_children(name, forecasts)
                    for child in children:
                        S[row, list(forecasts[list(forecasts.keys())[-1]].keys()).index(child)] = 1
                row += 1
                
        return S
        
    def _get_children(self, node: str, forecasts: Dict) -> List[str]:
        """Get children nodes in hierarchical structure."""
        children = []
        for level in forecasts:
            if node in forecasts[level]:
                next_level = list(forecasts.keys())[list(forecasts.keys()).index(level) + 1]
                children.extend([k for k in forecasts[next_level].keys() 
                              if k.startswith(node)])
        return children

    def probabilistic_forecast(self, features: pd.DataFrame, 
                             target: pd.Series) -> Dict:
        """Generate probabilistic forecasts.
        
        Args:
            features: Feature DataFrame
            target: Target series
            
        Returns:
            Dictionary with probabilistic forecasts
        """
        # Train quantile regression models
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        quantile_predictions = {}
        
        for q in quantiles:
            gbm = GradientBoostingRegressor(
                loss='quantile', alpha=q,
                random_state=42
            )
            gbm.fit(features, target)
            quantile_predictions[q] = gbm.predict(features)
            
        # Calculate prediction intervals
        z_score = norm.ppf((1 + self.config.confidence_level) / 2)
        std_err = np.std(target - quantile_predictions[0.5])
        
        return {
            'median': quantile_predictions[0.5],
            'quantiles': quantile_predictions,
            'lower_bound': quantile_predictions[0.5] - z_score * std_err,
            'upper_bound': quantile_predictions[0.5] + z_score * std_err
        }

    def online_learning(self, features: pd.DataFrame, target: pd.Series,
                       window_size: int = 100) -> None:
        """Implement online learning with concept drift detection.
        
        Args:
            features: Feature DataFrame
            target: Target series
            window_size: Size of rolling window for updating
        """
        from river import drift
        
        # Initialize drift detector
        drift_detector = drift.ADWIN()
        
        # Initialize metrics tracker
        performance_log = []
        
        # Sliding window update
        for i in range(window_size, len(features), window_size):
            # Get window data
            X_window = features.iloc[i-window_size:i]
            y_window = target.iloc[i-window_size:i]
            
            # Train model on window
            model = GradientBoostingRegressor(random_state=42)
            model.fit(X_window, y_window)
            
            # Make prediction and update drift detector
            pred = model.predict(features.iloc[i:i+1])
            actual = target.iloc[i]
            error = abs(actual - pred[0])
            
            drift_detector.update(error)
            
            # Check for concept drift
            if drift_detector.change_detected:
                print(f"Concept drift detected at index {i}")
                # Retrain model on recent data
                X_recent = features.iloc[i-window_size//2:i]
                y_recent = target.iloc[i-window_size//2:i]
                model.fit(X_recent, y_recent)
            
            # Log performance
            performance_log.append({
                'index': i,
                'error': error,
                'drift_detected': drift_detector.change_detected
            })
        
        self.online_performance = pd.DataFrame(performance_log)

    def cross_validate(self, features: pd.DataFrame, target: pd.Series,
                      n_splits: int = None) -> Dict[str, float]:
        """Perform time series cross-validation.
        
        Args:
            features: Feature DataFrame
            target: Target series
            n_splits: Number of CV splits
            
        Returns:
            Dictionary with cross-validation metrics
        """
        if n_splits is None:
            n_splits = self.config.cv_folds
            
        cv_metrics = []
        
        # Time series split
        for i in range(n_splits):
            # Calculate split indices
            train_size = int(len(features) * (0.6 + i * 0.1))
            val_size = int(len(features) * 0.1)
            
            if train_size + val_size > len(features):
                break
                
            # Split data
            X_train = features.iloc[:train_size]
            y_train = target.iloc[:train_size]
            X_val = features.iloc[train_size:train_size + val_size]
            y_val = target.iloc[train_size:train_size + val_size]
            
            # Train and evaluate
            model = self._get_model()
            model.fit(X_train, y_train)
            predictions = model.predict(X_val)
            
            # Calculate metrics
            metrics = {
                'mse': mean_squared_error(y_val, predictions),
                'mae': mean_absolute_error(y_val, predictions),
                'r2': r2_score(y_val, predictions)
            }
            cv_metrics.append(metrics)
            
        # Average metrics
        avg_metrics = {}
        for metric in cv_metrics[0].keys():
            avg_metrics[metric] = np.mean([m[metric] for m in cv_metrics])
            avg_metrics[f'{metric}_std'] = np.std([m[metric] for m in cv_metrics])
            
        return avg_metrics
        
    def _get_model(self) -> Union[RandomForestRegressor, 
                                 GradientBoostingRegressor,
                                 LinearRegression]:
        """Get the appropriate model based on configuration."""
        if self.config.model_type == 'prophet':
            from fbprophet import Prophet
            return Prophet(**self.config.prophet_params)
        elif self.config.model_type == 'var':
            from statsmodels.tsa.api import VAR
            return VAR()
        elif self.config.model_type == 'lstm':
            return self.lstm_forecast
        elif self.config.model_type == 'nbeats':
            return self.nbeats_forecast

        
        if self.config.model_type == 'ensemble':
            return RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )
        elif self.config.model_type == 'gbm':
            return GradientBoostingRegressor(
                n_estimators=100,
                random_state=42
            )
        else:
            return LinearRegression()

    def evaluate_forecast(self, predictions: np.ndarray, 
                         actual: pd.Series) -> Dict[str, float]:
        """Evaluate forecast performance with multiple metrics.
        
        Args:
            predictions: Predicted values
            actual: Actual values
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {
            'mse': mean_squared_error(actual, predictions),
            'mae': mean_absolute_error(actual, predictions),
            'rmse': np.sqrt(mean_squared_error(actual, predictions)),
            'r2': r2_score(actual, predictions),
            'mape': self._calculate_mape(actual, predictions),
            'smape': self._calculate_smape(actual, predictions),
            'directional_accuracy': self._calculate_directional_accuracy(actual, predictions)
        }
        
        # Add scale-independent metrics
        metrics.update(self._calculate_scale_independent_metrics(actual, predictions))
        
        return metrics
    
    def _calculate_mape(self, actual: pd.Series, 
                       predictions: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error."""
        return np.mean(np.abs((actual - predictions) / actual)) * 100
    
    def _calculate_smape(self, actual: pd.Series, 
                        predictions: np.ndarray) -> float:
        """Calculate Symmetric Mean Absolute Percentage Error."""
        return np.mean(2 * np.abs(predictions - actual) / 
                      (np.abs(actual) + np.abs(predictions))) * 100
    
    def _calculate_directional_accuracy(self, actual: pd.Series,
                                      predictions: np.ndarray) -> float:
        """Calculate directional accuracy of forecasts."""
        actual_dir = np.sign(np.diff(actual))
        pred_dir = np.sign(np.diff(predictions))
        return np.mean(actual_dir == pred_dir) * 100
    
    def _calculate_scale_independent_metrics(self, actual: pd.Series,
                                          predictions: np.ndarray) -> Dict[str, float]:
        """Calculate scale-independent forecast accuracy metrics."""
        # Mean Scaled Error
        mae_baseline = mean_absolute_error(actual[1:], actual[:-1])
        mase = mean_absolute_error(actual, predictions) / mae_baseline
        
        # Normalized RMSE
        nrmse = np.sqrt(mean_squared_error(actual, predictions)) / (actual.max() - actual.min())
        
        return {
            'mase': mase,
            'nrmse': nrmse
        }

    def generate_scenarios(self, features: pd.DataFrame, 
                         n_scenarios: int = 100) -> Dict[str, pd.DataFrame]:
        """Generate forecast scenarios using bootstrap and simulation.
        
        Args:
            features: Feature DataFrame
            n_scenarios: Number of scenarios to generate
            
        Returns:
            Dictionary containing scenario forecasts
        """
        scenarios = []
        model = self._get_model()
        
        # Bootstrap residuals approach
        for _ in range(n_scenarios):
            # Train model on bootstrapped sample
            bootstrap_idx = np.random.choice(
                len(features), 
                size=len(features),
                replace=True
            )
            X_boot = features.iloc[bootstrap_idx]
            y_boot = features.iloc[bootstrap_idx]
            
            model.fit(X_boot, y_boot)
            
            # Generate scenario
            base_prediction = model.predict(features)
            residuals = np.random.choice(
                y_boot - model.predict(X_boot),
                size=len(features)
            )
            scenarios.append(base_prediction + residuals)
        
        # Convert to DataFrame
        scenario_df = pd.DataFrame(
            scenarios,
            columns=features.index,
            index=[f'scenario_{i+1}' for i in range(n_scenarios)]
        )
        
        # Calculate scenario statistics
        stats = {
            'median': scenario_df.median(),
            'mean': scenario_df.mean(),
            'std': scenario_df.std(),
            'q10': scenario_df.quantile(0.1),
            'q90': scenario_df.quantile(0.9)
        }
        
        return {
            'scenarios': scenario_df,
            'statistics': pd.DataFrame(stats)
        }

    def adapt_to_regime(self, features: pd.DataFrame, 
                       target: pd.Series,
                       regime_indicators: pd.DataFrame) -> Dict:
        """Adapt forecasting to different market regimes.
        
        Args:
            features: Feature DataFrame
            target: Target series
            regime_indicators: DataFrame with regime indicators
            
        Returns:
            Dictionary with regime-specific models and predictions
        """
        regime_models = {}
        regime_predictions = {}
        
        # Train regime-specific models
        for regime in regime_indicators['regime'].unique():
            regime_mask = regime_indicators['regime'] == regime
            
            if regime_mask.sum() > len(features) * 0.1:  # Minimum data requirement
                X_regime = features[regime_mask]
                y_regime = target[regime_mask]
                
                model = self._get_model()
                model.fit(X_regime, y_regime)
                
                regime_models[regime] = model
                regime_predictions[regime] = model.predict(features)
        
        # Combine predictions based on regime probabilities
        combined_predictions = np.zeros(len(features))
        for regime in regime_models:
            regime_prob = (regime_indicators['regime'] == regime).astype(float)
            combined_predictions += regime_predictions[regime] * regime_prob
        
        return {
            'regime_models': regime_models,
            'regime_predictions': regime_predictions,
            'combined_predictions': combined_predictions
        }
    
    def feature_selection(self, features: pd.DataFrame, 
                         target: pd.Series,
                         method: str = 'importance') -> pd.DataFrame:
        """Select most relevant features for forecasting.
        
        Args:
            features: Feature DataFrame
            target: Target series
            method: Feature selection method ('importance' or 'correlation')
            
        Returns:
            DataFrame with selected features
        """
        if method == 'importance':
            # Use Random Forest importance
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(features, target)
            
            importance = pd.Series(
                model.feature_importances_,
                index=features.columns
            )
            selected_features = features[importance[importance > importance.mean()].index]
            
        else:  # correlation method
            # Calculate correlation with target
            correlations = features.corrwith(target).abs()
            selected_features = features[correlations[correlations > 0.1].index]
        
        return selected_features
    
    def train_forecast_model(self, 
                           features: pd.DataFrame,
                           target: pd.Series,
                           test_size: float = 0.2
                           ) -> Tuple[LinearRegression, Dict[str, float]]:
        """Train forecasting model and evaluate performance.
        
        Args:
            features: Feature DataFrame
            target: Target series
            test_size: Proportion of data for testing
            
        Returns:
            Tuple of (trained model, performance metrics)
        """
        # Split data
        split_idx = int(len(features) * (1 - test_size))
        train_features = features.iloc[:split_idx].dropna()
        test_features = features.iloc[split_idx:].dropna()
        train_target = target.iloc[:split_idx].loc[train_features.index]
        test_target = target.iloc[split_idx:].loc[test_features.index]
        
        # Train model
        model = LinearRegression()
        model.fit(train_features, train_target)
        
        # Make predictions
        train_pred = model.predict(train_features)
        test_pred = model.predict(test_features)
        
        # Calculate metrics
        metrics = {
            'train_mse': mean_squared_error(train_target, train_pred),
            'test_mse': mean_squared_error(test_target, test_pred),
            'train_mae': mean_absolute_error(train_target, train_pred),
            'test_mae': mean_absolute_error(test_target, test_pred),
            'r2_score': model.score(test_features, test_target)
        }
        
        return model, metrics
    
    def make_forecast(self, 
                     model: LinearRegression,
                     features: pd.DataFrame,
                     horizon: int,
                     uncertainty: bool = True,
                     confidence: float = 0.95
                     ) -> Dict[str, pd.Series]:
        """Generate forecasts with optional uncertainty intervals.
        
        Args:
            model: Trained forecasting model
            features: Feature DataFrame
            horizon: Number of periods to forecast
            uncertainty: Whether to include uncertainty intervals
            confidence: Confidence level for intervals
            
        Returns:
            Dictionary with forecast and intervals
        """
        forecast = pd.Series(
            model.predict(features),
            index=features.index,
            name='forecast'
        )
        
        result = {'forecast': forecast}
        
        if uncertainty:
            # Calculate prediction intervals
            mse = mean_squared_error(
                self.data.iloc[-len(forecast):],
                forecast
            )
            std_err = np.sqrt(mse)
            z_score = norm.ppf((1 + confidence) / 2)
            
            margin = z_score * std_err
            result.update({
                'lower': forecast - margin,
                'upper': forecast + margin
            })
            
        return result
    
    def evaluate_residuals(self, 
                          actual: pd.Series,
                          predicted: pd.Series
                          ) -> Dict[str, float]:
        """Analyze forecast residuals.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            
        Returns:
            Dictionary of residual statistics
        """
        residuals = actual - predicted
        
        stats = {
            'mean': residuals.mean(),
            'std': residuals.std(),
            'skew': residuals.skew(),
            'kurtosis': residuals.kurtosis(),
            'ljung_box_p': self._ljung_box_test(residuals)
        }
        
        return stats
    
    def _ljung_box_test(self, residuals: pd.Series, lags: int = 10) -> float:
        """Perform Ljung-Box test for residual autocorrelation.
        
        Args:
            residuals: Residual series
            lags: Number of lags to test
            
        Returns:
            p-value from test
        """
        n = len(residuals)
        acf = [1] + [residuals.autocorr(lag) for lag in range(1, lags + 1)]
        q_stat = n * (n + 2) * sum([(acf[k]**2) / (n - k) for k in range(1, lags + 1)])
        p_value = 1 - norm.cdf(q_stat)
        return p_value

    def prophet_forecast(self, features: pd.DataFrame, target: pd.Series) -> Dict:
        """Forecast using Facebook Prophet.
        
        Args:
            features: Feature DataFrame
            target: Target series
            
        Returns:
            Dictionary with forecast results
        """
        from fbprophet import Prophet
        
        # Prepare data for Prophet
        df_prophet = pd.DataFrame({
            'ds': target.index,
            'y': target.values
        })
        
        # Add exogenous features if specified
        if self.config.use_exogenous:
            for col in features.columns:
                df_prophet[col] = features[col]
        
        # Initialize and fit Prophet model
        model = Prophet(**self.config.prophet_params)
        
        # Add exogenous regressors
        if self.config.use_exogenous:
            for col in features.columns:
                model.add_regressor(col)
        
        model.fit(df_prophet)
        
        # Create future dataframe
        future = model.make_future_dataframe(
            periods=self.config.forecast_horizon,
            freq=pd.infer_freq(target.index)
        )
        
        # Add features to future if using exogenous
        if self.config.use_exogenous:
            for col in features.columns:
                future[col] = features[col]
        
        # Make forecast
        forecast = model.predict(future)
        
        return {
            'forecast': forecast['yhat'],
            'lower_bound': forecast['yhat_lower'],
            'upper_bound': forecast['yhat_upper'],
            'components': model.component_predictions(forecast)
        }

    def var_forecast(self, features: pd.DataFrame, target: pd.Series) -> Dict:
        """Forecast using Vector Autoregression.
        
        Args:
            features: Feature DataFrame
            target: Target series
            
        Returns:
            Dictionary with VAR forecast results
        """
        from statsmodels.tsa.api import VAR
        
        # Combine target and features
        data = pd.concat([target, features], axis=1)
        
        # Fit VAR model
        model = VAR(data)
        results = model.fit()
        
        # Make forecast
        forecast = results.forecast(data.values, steps=self.config.forecast_horizon)
        
        return {
            'forecast': pd.Series(forecast[:, 0], 
                                index=pd.date_range(target.index[-1], 
                                periods=self.config.forecast_horizon+1, 
                                freq=pd.infer_freq(target.index))[1:]),
            'model': results,
            'feature_forecasts': pd.DataFrame(forecast[:, 1:], 
                                            columns=features.columns)
        }

    def lstm_forecast(self, features: pd.DataFrame, target: pd.Series) -> Dict:
        """Forecast using LSTM neural network.
        
        Args:
            features: Feature DataFrame
            target: Target series
            
        Returns:
            Dictionary with LSTM forecast results
        """
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        
        # Prepare data for LSTM
        def create_sequences(data, seq_length):
            xs, ys = [], []
            for i in range(len(data) - seq_length):
                x = data.iloc[i:(i + seq_length)]
                y = data.iloc[i + seq_length]
                xs.append(x)
                ys.append(y)
            return np.array(xs), np.array(ys)
        
        # Normalize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(
            pd.concat([target.to_frame(), features], axis=1)
        )
        
        # Create sequences
        X, y = create_sequences(
            pd.DataFrame(scaled_data), 
            seq_length=max(self.config.features_lag)
        )
        
        # Build LSTM model
        model = Sequential([
            LSTM(self.config.lstm_units, return_sequences=True,
                input_shape=(X.shape[1], X.shape[2])),
            Dropout(0.2),
            LSTM(self.config.lstm_units // 2),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        
        # Train model
        model.fit(X, y, epochs=self.config.lstm_epochs, 
                batch_size=32, verbose=0)
        
        # Generate forecast
        last_sequence = scaled_data[-max(self.config.features_lag):]
        forecast = []
        
        for _ in range(self.config.forecast_horizon):
            next_pred = model.predict(
                last_sequence.reshape(1, last_sequence.shape[0], 
                                    last_sequence.shape[1])
            )
            forecast.append(next_pred[0, 0])
            # Update sequence
            last_sequence = np.roll(last_sequence, -1, axis=0)
            last_sequence[-1] = next_pred
        
        # Inverse transform forecast
        forecast = scaler.inverse_transform(
            np.column_stack([forecast, np.zeros((len(forecast), 
                                            features.shape[1]))])
        )[:, 0]
        
        return {
            'forecast': pd.Series(forecast, 
                                index=pd.date_range(target.index[-1], 
                                periods=self.config.forecast_horizon+1, 
                                freq=pd.infer_freq(target.index))[1:]),
            'model': model
        }

    def nbeats_forecast(self, features: pd.DataFrame, target: pd.Series) -> Dict:
        """Forecast using N-BEATS neural network.
        
        Args:
            features: Feature DataFrame
            target: Target series
            
        Returns:
            Dictionary with N-BEATS forecast results
        """
        from nbeats_pytorch import NBeatsNet
        
        # Prepare data
        scaler = StandardScaler()
        scaled_target = scaler.fit_transform(target.values.reshape(-1, 1))
        
        # Configure N-BEATS model
        nbeats = NBeatsNet(
            stack_types=['trend', 'seasonality'],
            forecast_length=self.config.forecast_horizon,
            backcast_length=3 * self.config.forecast_horizon,
            hidden_layer_units=256
        )
        
        # Train model
        nbeats.fit(scaled_target, 
                epochs=self.config.lstm_epochs, 
                verbose=False)
        
        # Generate forecast
        forecast = nbeats.predict(scaled_target)
        forecast = scaler.inverse_transform(forecast.reshape(-1, 1))
        
        return {
            'forecast': pd.Series(forecast.flatten(), 
                                index=pd.date_range(target.index[-1], 
                                periods=self.config.forecast_horizon+1, 
                                freq=pd.infer_freq(target.index))[1:]),
            'model': nbeats
        }
    
    def seasonal_forecast(self, features: pd.DataFrame, target: pd.Series) -> Dict:
        """Forecast using seasonal decomposition and trend modeling (alternative to Prophet).
        
        Args:
            features: Feature DataFrame
            target: Target series
            
        Returns:
            Dictionary with forecast results
        """
        from statsmodels.tsa.seasonal import seasonal_decompose
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        
        # Perform seasonal decomposition
        decomposition = seasonal_decompose(target, period=self.config.seasonal_periods[0])
        
        # Forecast trend using Holt-Winters
        hw_model = ExponentialSmoothing(
            target,
            seasonal_periods=self.config.seasonal_periods[0],
            trend='add',
            seasonal='add'
        ).fit()
        
        # Generate forecasts
        forecast = hw_model.forecast(self.config.forecast_horizon)
        
        # Add prediction intervals
        conf_int = hw_model.get_prediction(
            start=len(target),
            end=len(target) + self.config.forecast_horizon - 1
        ).conf_int(alpha=0.05)
        
        return {
            'forecast': forecast,
            'lower_bound': conf_int.iloc[:, 0],
            'upper_bound': conf_int.iloc[:, 1],
            'components': {
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'resid': decomposition.resid
            }
        }

    def sarima_forecast(self, features: pd.DataFrame, target: pd.Series) -> Dict:
        """Forecast using SARIMA model.
        
        Args:
            features: Feature DataFrame
            target: Target series
            
        Returns:
            Dictionary with SARIMA forecast results
        """
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        from pmdarima import auto_arima
        
        # Automatically find best SARIMA parameters
        auto_model = auto_arima(
            target,
            seasonal=True,
            m=self.config.seasonal_periods[0],
            suppress_warnings=True,
            error_action="ignore",
            max_p=5, max_q=5,
            max_P=2, max_Q=2,
            max_order=5,
            max_d=2, max_D=1
        )
        
        # Get best parameters
        order = auto_model.order
        seasonal_order = auto_model.seasonal_order
        
        # Fit SARIMA model
        model = SARIMAX(
            target,
            order=order,
            seasonal_order=seasonal_order
        ).fit(disp=False)
        
        # Generate forecasts with confidence intervals
        forecast = model.get_forecast(self.config.forecast_horizon)
        
        return {
            'forecast': forecast.predicted_mean,
            'lower_bound': forecast.conf_int().iloc[:, 0],
            'upper_bound': forecast.conf_int().iloc[:, 1],
            'model': model,
            'order': order,
            'seasonal_order': seasonal_order
        }

    def ets_forecast(self, features: pd.DataFrame, target: pd.Series) -> Dict:
        """Forecast using ETS (Error, Trend, Seasonal) model.
        
        Args:
            features: Feature DataFrame
            target: Target series
            
        Returns:
            Dictionary with ETS forecast results
        """
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        
        # Determine best ETS model type using cross-validation
        ets_types = [
            ('add', 'add'),
            ('add', 'mul'),
            ('mul', 'add'),
            ('mul', 'mul')
        ]
        
        best_aic = float('inf')
        best_model = None
        best_type = None
        
        for trend, seasonal in ets_types:
            try:
                model = ExponentialSmoothing(
                    target,
                    trend=trend,
                    seasonal=seasonal,
                    seasonal_periods=self.config.seasonal_periods[0]
                ).fit()
                
                if model.aic < best_aic:
                    best_aic = model.aic
                    best_model = model
                    best_type = (trend, seasonal)
            except:
                continue
        
        # Generate forecasts with the best model
        forecast = best_model.forecast(self.config.forecast_horizon)
        
        # Calculate prediction intervals using simulation
        simulations = []
        for _ in range(100):
            sim = best_model.simulate(
                nsimulations=self.config.forecast_horizon,
                anchor=len(target)
            )
            simulations.append(sim)
        
        # Calculate confidence intervals from simulations
        simulations = np.array(simulations)
        lower = np.percentile(simulations, 2.5, axis=0)
        upper = np.percentile(simulations, 97.5, axis=0)
        
        return {
            'forecast': forecast,
            'lower_bound': pd.Series(lower, index=forecast.index),
            'upper_bound': pd.Series(upper, index=forecast.index),
            'model': best_model,
            'model_type': best_type
        }

    def combined_seasonal_forecast(self, features: pd.DataFrame, target: pd.Series) -> Dict:
        """Combine multiple seasonal forecasting methods.
        
        Args:
            features: Feature DataFrame
            target: Target series
            
        Returns:
            Dictionary with combined forecast results
        """
        # Get forecasts from different models
        hw_result = self.seasonal_forecast(features, target)
        sarima_result = self.sarima_forecast(features, target)
        ets_result = self.ets_forecast(features, target)
        
        # Combine forecasts with equal weights
        combined_forecast = (
            hw_result['forecast'] +
            sarima_result['forecast'] +
            ets_result['forecast']
        ) / 3
        
        # Calculate combined confidence intervals
        lower_bound = (
            hw_result['lower_bound'] +
            sarima_result['lower_bound'] +
            ets_result['lower_bound']
        ) / 3
        
        upper_bound = (
            hw_result['upper_bound'] +
            sarima_result['upper_bound'] +
            ets_result['upper_bound']
        ) / 3
        
        return {
            'forecast': combined_forecast,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'component_forecasts': {
                'hw': hw_result['forecast'],
                'sarima': sarima_result['forecast'],
                'ets': ets_result['forecast']
            },
            'decomposition': hw_result['components']
        }
