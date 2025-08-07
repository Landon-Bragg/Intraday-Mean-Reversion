import numpy as np
import pandas as pd
from scipy.optimize import minimize
import cvxpy as cp
from typing import Dict

class PortfolioOptimizer:
    """Advanced portfolio optimization with constraints."""
    
    def __init__(self, config):
        self.config = config
        
    def optimize_portfolio(self, expected_returns: np.ndarray,
                          covariance_matrix: np.ndarray,
                          constraints: Dict = None) -> Dict:
        """Optimize portfolio using mean-variance optimization."""
        
        n_assets = len(expected_returns)
        
        # Decision variables
        weights = cp.Variable(n_assets)
        
        # Objective: maximize utility (return - risk penalty)
        risk_aversion = self.config.get('risk_aversion', 1.0)
        utility = expected_returns.T @ weights - 0.5 * risk_aversion * cp.quad_form(weights, covariance_matrix)
        
        # Constraints
        constraints_list = [cp.sum(weights) == 1]  # Weights sum to 1
        
        # Long-only constraint (optional)
        if self.config.get('long_only', True):
            constraints_list.append(weights >= 0)
            
        # Maximum position size
        max_weight = self.config.get('max_position_weight', 0.1)
        constraints_list.append(weights <= max_weight)
        
        # Minimum position size (to avoid tiny positions)
        min_weight = self.config.get('min_position_weight', 0.01)
        constraints_list.append(cp.logical_or(weights == 0, weights >= min_weight))
        
        # Solve optimization
        problem = cp.Problem(cp.Maximize(utility), constraints_list)
        problem.solve()
        
        if problem.status not in ["infeasible", "unbounded"]:
            optimal_weights = weights.value
            expected_return = expected_returns.T @ optimal_weights
            portfolio_risk = np.sqrt(optimal_weights.T @ covariance_matrix @ optimal_weights)
            
            return {
                'weights': optimal_weights,
                'expected_return': expected_return,
                'risk': portfolio_risk,
                'sharpe_ratio': expected_return / portfolio_risk,
                'status': problem.status
            }
        else:
            return {'status': 'failed', 'message': problem.status}
            
    def black_litterman_optimization(self,
                                    market_caps: np.ndarray,
                                    covariance_matrix: np.ndarray,
                                    views_matrix: np.ndarray = None,
                                    view_returns: np.ndarray = None,
                                    tau: float = 0.05) -> Dict:
        """
        Black-Litterman portfolio optimization.
        
        Args:
            market_caps: Market capitalizations of assets (used to get prior weights).
            covariance_matrix: Asset covariance matrix (Σ).
            views_matrix: P matrix (k × n), optional.
            view_returns: Q vector (k × 1), optional.
            tau: Scalar reflecting uncertainty in the prior (typically 0.025–0.1).
        
        Returns:
            Dict with weights and other results.
        """
        n = len(market_caps)
        market_weights = market_caps / np.sum(market_caps)

        # 1. Compute implied equilibrium returns (π)
        risk_aversion = self.config.get('risk_aversion', 1.0)
        pi = risk_aversion * covariance_matrix @ market_weights

        # 2. Incorporate views
        if views_matrix is not None and view_returns is not None:
            # Omega: Uncertainty of views (diagonal matrix, can be user-defined)
            omega = np.diag(np.diag(views_matrix @ (tau * covariance_matrix) @ views_matrix.T))

            # Black-Litterman posterior expected returns
            middle_term = np.linalg.inv(views_matrix @ (tau * covariance_matrix) @ views_matrix.T + omega)
            adjusted_returns = pi + tau * covariance_matrix @ views_matrix.T @ middle_term @ (view_returns - views_matrix @ pi)
        else:
            adjusted_returns = pi

        # 3. Optimize portfolio using adjusted returns
        return self.optimize_portfolio(adjusted_returns, covariance_matrix)
