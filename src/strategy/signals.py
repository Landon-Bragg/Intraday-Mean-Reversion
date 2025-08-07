# src/strategy/signals.py
"""
Signal Generation and Portfolio Construction
Combines factors into actionable trading signals
"""
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignalGenerator:
    """
    Professional signal generation engine that combines multiple factors
    into actionable trading signals with proper risk controls.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.factor_weights = config.get('factor_weights', {
            'rsi_factor': 0.15,
            'vwap_factor': 0.20,
            'gap_factor': 0.10,
            'vol_regime_factor': 0.15,
            'time_factor': 0.10,
            'volume_factor': 0.15,
            'support_resistance_factor': 0.10,
            'market_regime_factor': 0.05
        })
        
    def generate_signals(self, factors: pd.DataFrame, 
                        prices: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from factor scores.
        
        Args:
            factors: DataFrame with factor scores
            prices: DataFrame with price data
            
        Returns:
            DataFrame with signals and metadata
        """
        # Combine factors using weights
        composite_signal = self._combine_factors(factors)
        
        # Apply signal filtering
        filtered_signals = self._filter_signals(composite_signal, prices)
        
        # Calculate position sizes
        position_sizes = self._calculate_position_sizes(filtered_signals, prices)
        
        # Create signal DataFrame
        signals = pd.DataFrame({
            'raw_signal': composite_signal,
            'filtered_signal': filtered_signals,
            'position_size': position_sizes,
            'entry_price': prices['close'],
            'timestamp': prices.index
        })
        
        return signals
    
    def _combine_factors(self, factors: pd.DataFrame) -> pd.Series:
        """Combine individual factors using configured weights."""
        composite = pd.Series(0, index=factors.index)
        
        for factor_name, weight in self.factor_weights.items():
            if factor_name in factors.columns:
                composite += factors[factor_name] * weight
                
        # Normalize to [-1, 1] range
        composite = np.tanh(composite)
        
        return composite
    
    def _filter_signals(self, signals: pd.Series, 
                       prices: pd.DataFrame) -> pd.Series:
        """Apply signal filtering rules."""
        filtered = signals.copy()
        
        # Minimum signal strength threshold
        min_threshold = self.config.get('min_signal_strength', 0.3)
        filtered = np.where(np.abs(filtered) < min_threshold, 0, filtered)
        
        # Volume filter
        min_volume = self.config.get('min_volume', 100000)
        volume_filter = prices['volume'] > min_volume
        filtered = np.where(volume_filter, filtered, 0)
        
        # Price filter
        min_price = self.config.get('min_price', 5.0)
        max_price = self.config.get('max_price', 500.0)
        price_filter = (prices['close'] > min_price) & (prices['close'] < max_price)
        filtered = np.where(price_filter, filtered, 0)
        
        return pd.Series(filtered, index=signals.index)
    
    def _calculate_position_sizes(self, signals: pd.Series, 
                                 prices: pd.DataFrame) -> pd.Series:
        """Calculate position sizes based on signal strength and risk."""
        # Base position size from signal strength
        base_size = np.abs(signals) * self.config.get('max_position_size', 0.05)
        
        # Adjust for volatility
        returns = prices['close'].pct_change()
        volatility = returns.rolling(window=20).std()
        vol_adjustment = 1 / (1 + volatility * 10)  # Reduce size in high vol
        
        position_sizes = base_size * vol_adjustment
        
        # Apply signal direction
        position_sizes = position_sizes * np.sign(signals)
        
        return position_sizes


