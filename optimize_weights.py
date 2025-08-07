import numpy as np
from scipy.optimize import minimize
from main import TradingStrategyOrchestrator

def objective(weights, orchestrator, symbols):
    """Objective function to maximize Sharpe ratio."""
    # Update factor weights in config
    weight_dict = {
        'rsi_factor': weights[0],
        'vwap_factor': weights[1],
        'gap_factor': weights[2],
        'vol_regime_factor': weights[3],
        'time_factor': weights[4],
        'volume_factor': weights[5],
        'support_resistance_factor': weights[6],
        'market_regime_factor': weights[7]
    }
    
    orchestrator.config['strategy']['factor_weights'] = weight_dict
    orchestrator.setup_components()  # Reinitialize with new weights
    
    results = orchestrator.run_full_backtest(symbols, days=20)
    sharpe = results['portfolio_metrics']['sharpe_ratio']
    
    return -sharpe  # Minimize negative Sharpe (maximize Sharpe)

# Initialize
orchestrator = TradingStrategyOrchestrator()
symbols = ['AAPL', 'MSFT', 'GOOGL']

# Initial weights (equal)
initial_weights = np.array([0.125] * 8)

# Constraints (weights sum to 1)
constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
bounds = [(0.05, 0.3)] * 8  # Each weight between 5% and 30%

# Optimize
result = minimize(objective, initial_weights, 
                 args=(orchestrator, symbols),
                 method='SLSQP',
                 bounds=bounds,
                 constraints=constraints)

print("Optimized weights:", result.x)