import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from typing import Dict, List

class MLFactorSelector:
    """Machine learning-based factor selection and weighting."""
    
    def __init__(self, config):
        self.config = config
        self.models = {
            'ridge': Ridge(alpha=0.1),
            'lasso': Lasso(alpha=0.01),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
    def optimize_factor_weights(self, factors: pd.DataFrame, 
                              returns: pd.Series) -> Dict:
        """Optimize factor weights using ML."""
        
        # Prepare data
        X = factors.fillna(0).values
        y = returns.shift(-1).fillna(0).values  # Next period returns
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        results = {}
        for name, model in self.models.items():
            scores = cross_val_score(model, X, y, cv=tscv, 
                                   scoring='neg_mean_squared_error')
            
            # Fit full model
            model.fit(X, y)
            
            # Get feature importances/coefficients
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            else:
                importances = np.abs(model.coef_)
                
            # Normalize to sum to 1
            weights = importances / importances.sum()
            
            results[name] = {
                'weights': dict(zip(factors.columns, weights)),
                'cv_score': scores.mean(),
                'model': model
            }
            
        return results

    def feature_selection(self, factors: pd.DataFrame, 
                         returns: pd.Series, k: int = 5) -> List[str]:
        """Select top k factors using statistical tests."""
        
        X = factors.fillna(0).values
        y = returns.shift(-1).fillna(0).values
        
        selector = SelectKBest(score_func=f_regression, k=k)
        selector.fit(X, y)
        
        selected_features = factors.columns[selector.get_support()].tolist()
        feature_scores = dict(zip(factors.columns, selector.scores_))
        
        return selected_features, feature_scores