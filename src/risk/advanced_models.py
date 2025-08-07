import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats
from typing import Dict, List

class AdvancedRiskModels:
    """Advanced risk modeling including GARCH and regime switching."""
    
    def __init__(self, config):
        self.config = config
        
    def garch_volatility_forecast(self, returns: pd.Series, 
                                 horizon: int = 1) -> Dict:
        """GARCH volatility forecasting."""
        
        # Fit GARCH(1,1) model
        model = arch_model(returns * 100, vol='Garch', p=1, q=1)
        fitted_model = model.fit(disp='off')
        
        # Forecast volatility
        forecast = fitted_model.forecast(horizon=horizon)
        
        return {
            'forecast_variance': forecast.variance.iloc[-1, :] / 10000,  # Convert back to decimal
            'forecast_volatility': np.sqrt(forecast.variance.iloc[-1, :] / 10000),
            'model_summary': fitted_model.summary(),
            'fitted_model': fitted_model
        }
        
    def value_at_risk(self, returns: pd.Series, 
                     confidence_levels: List[float] = [0.95, 0.99],
                     methods: List[str] = ['historical', 'parametric', 'monte_carlo']) -> Dict:
        """Calculate VaR using multiple methods."""
        
        var_results = {}
        
        for confidence in confidence_levels:
            var_results[confidence] = {}
            
            # Historical VaR
            if 'historical' in methods:
                var_results[confidence]['historical'] = np.percentile(returns, (1-confidence)*100)
            
            # Parametric VaR (normal distribution)
            if 'parametric' in methods:
                mean_return = returns.mean()
                std_return = returns.std()
                var_results[confidence]['parametric'] = stats.norm.ppf(1-confidence, mean_return, std_return)
            
            # Monte Carlo VaR
            if 'monte_carlo' in methods:
                n_simulations = 10000
                simulated_returns = np.random.normal(returns.mean(), returns.std(), n_simulations)
                var_results[confidence]['monte_carlo'] = np.percentile(simulated_returns, (1-confidence)*100)
                
        return var_results
        
    def expected_shortfall(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Expected Shortfall (Conditional VaR)."""
        var_threshold = np.percentile(returns, (1-confidence)*100)
        return returns[returns <= var_threshold].mean()
        
    def maximum_drawdown_analysis(self, portfolio_values: pd.Series) -> Dict:
        """Detailed maximum drawdown analysis."""
        
        # Calculate rolling maximum
        rolling_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - rolling_max) / rolling_max
        
        # Find maximum drawdown period
        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()
        
        # Recovery analysis
        recovery_date = None
        if max_dd_date < portfolio_values.index[-1]:
            recovery_series = portfolio_values[max_dd_date:]
            recovery_threshold = rolling_max[max_dd_date]
            recovery_points = recovery_series[recovery_series >= recovery_threshold]
            if len(recovery_points) > 0:
                recovery_date = recovery_points.index[0]
                
        return {
            'maximum_drawdown': max_dd,
            'max_drawdown_date': max_dd_date,
            'recovery_date': recovery_date,
            'underwater_period': (recovery_date - max_dd_date).days if recovery_date else None,
            'drawdown_series': drawdown
        }