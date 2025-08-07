# src/strategy/factors.py
"""
Eight Custom Intraday Mean Reversion Factors
Professional implementation with proper documentation and testing
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class IntradayFactors:
    """
    Professional-grade intraday mean reversion factor engine.
    Implements 8 proprietary factors for short-term price prediction.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.lookback_window = config.get('lookback_window', 20)
        self.z_score_threshold = config.get('z_score_threshold', 2.0)
        
    def calculate_all_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all 8 intraday factors for given price data.
        
        Args:
            data: DataFrame with OHLCV minute-level data
            
        Returns:
            DataFrame with factor scores (-1 to 1)
        """
        factors = pd.DataFrame(index=data.index)
        
        # Factor 1: Intraday RSI
        factors['rsi_factor'] = self.intraday_rsi_factor(data)
        
        # Factor 2: VWAP Deviation  
        factors['vwap_factor'] = self.vwap_deviation_factor(data)
        
        # Factor 3: Gap Reversion Probability
        factors['gap_factor'] = self.gap_reversion_factor(data)
        
        # Factor 4: Volatility Regime Detection
        factors['vol_regime_factor'] = self.volatility_regime_factor(data)
        
        # Factor 5: Time-of-Day Effects
        factors['time_factor'] = self.time_of_day_factor(data)
        
        # Factor 6: Volume-Adjusted Price Changes
        factors['volume_factor'] = self.volume_adjusted_factor(data)
        
        # Factor 7: Support/Resistance Proximity
        factors['support_resistance_factor'] = self.support_resistance_factor(data)
        
        # Factor 8: Market Regime Classification
        factors['market_regime_factor'] = self.market_regime_factor(data)
        
        return factors.fillna(0)
    
    def intraday_rsi_factor(self, data: pd.DataFrame) -> pd.Series:
        """
        Factor 1: Intraday RSI for mean reversion signals.
        RSI > 70 suggests oversold (negative signal)
        RSI < 30 suggests overbought (positive signal)
        """
        close = data['close']
        delta = close.diff()
        
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Convert to mean reversion signal (-1 to 1)
        factor = np.where(rsi > 70, -1, np.where(rsi < 30, 1, 0))
        factor = pd.Series(factor, index=data.index)
        
        # Smooth the signal
        return factor.rolling(window=5).mean()
    
    def vwap_deviation_factor(self, data: pd.DataFrame) -> pd.Series:
        """
        Factor 2: VWAP Deviation - measures distance from volume-weighted price.
        Large deviations suggest mean reversion opportunities.
        """
        # Calculate VWAP
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        vwap = (typical_price * data['volume']).cumsum() / data['volume'].cumsum()
        
        # Calculate percentage deviation
        deviation = (data['close'] - vwap) / vwap
        
        # Rolling z-score for standardization
        rolling_mean = deviation.rolling(window=20).mean()
        rolling_std = deviation.rolling(window=20).std()
        z_score = (deviation - rolling_mean) / rolling_std
        
        # Mean reversion signal (opposite of deviation)
        factor = -np.tanh(z_score / 2)  # Bounded between -1 and 1
        
        return factor
    
    def gap_reversion_factor(self, data: pd.DataFrame) -> pd.Series:
        """
        Factor 3: Gap Reversion Probability based on historical patterns.
        Gaps tend to fill, creating mean reversion opportunities.
        """
        # Identify gaps (using daily opens vs previous close)
        daily_data = data.resample('1D').agg({
            'open': 'first',
            'high': 'max', 
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        prev_close = daily_data['close'].shift(1)
        gap = (daily_data['open'] - prev_close) / prev_close
        
        # Reindex to minute data
        gap_expanded = gap.reindex(data.index, method='ffill')
        
        # Gap reversion signal (stronger for larger gaps)
        factor = -np.tanh(gap_expanded * 10)  # Invert for mean reversion
        
        return factor.fillna(0)
    
    def volatility_regime_factor(self, data: pd.DataFrame) -> pd.Series:
        """
        Factor 4: Volatility Regime Detection.
        Mean reversion works better in normal volatility regimes.
        """
        # Calculate rolling volatility
        returns = data['close'].pct_change()
        vol_short = returns.rolling(window=20).std() * np.sqrt(252 * 390)  # Annualized
        vol_long = returns.rolling(window=60).std() * np.sqrt(252 * 390)
        
        # Volatility ratio
        vol_ratio = vol_short / vol_long
        
        # Optimal range for mean reversion (0.8 to 1.2)
        factor = np.where(
            (vol_ratio > 0.8) & (vol_ratio < 1.2), 
            1,  # Favorable regime
            np.where(vol_ratio > 1.5, -1, 0)  # Unfavorable regime
        )
        
        return pd.Series(factor, index=data.index)
    
    def time_of_day_factor(self, data: pd.DataFrame) -> pd.Series:
        """
        Factor 5: Time-of-Day Effects.
        Mean reversion is stronger during market open/close.
        """
        hour = data.index.hour
        minute = data.index.minute
        time_of_day = hour + minute / 60
        
        # Stronger mean reversion: 9:30-10:30 and 15:30-16:00
        morning_boost = np.where((time_of_day >= 9.5) & (time_of_day <= 10.5), 0.5, 0)
        afternoon_boost = np.where((time_of_day >= 15.5) & (time_of_day <= 16.0), 0.5, 0)
        
        # Lunch hour penalty (12:00-14:00)
        lunch_penalty = np.where((time_of_day >= 12.0) & (time_of_day <= 14.0), -0.3, 0)
        
        factor = morning_boost + afternoon_boost + lunch_penalty
        
        return pd.Series(factor, index=data.index)
    
    def volume_adjusted_factor(self, data: pd.DataFrame) -> pd.Series:
        """
        Factor 6: Volume-Adjusted Price Changes.
        Large price moves on low volume are more likely to revert.
        """
        returns = data['close'].pct_change()
        volume_ratio = data['volume'] / data['volume'].rolling(window=20).mean()
        
        # Volume-adjusted returns
        vol_adj_returns = returns / np.sqrt(volume_ratio)
        
        # Z-score for standardization
        rolling_mean = vol_adj_returns.rolling(window=20).mean()
        rolling_std = vol_adj_returns.rolling(window=20).std()
        z_score = (vol_adj_returns - rolling_mean) / rolling_std
        
        # Mean reversion signal
        factor = -np.tanh(z_score)  # Opposite direction for reversion
        
        return factor
    
    def support_resistance_factor(self, data: pd.DataFrame) -> pd.Series:
        """
        Factor 7: Support/Resistance Proximity.
        Increase position sizes near key technical levels.
        """
        # Calculate support and resistance levels (using pivot points)
        high_20 = data['high'].rolling(window=20).max()
        low_20 = data['low'].rolling(window=20).min()
        
        # Distance to support/resistance
        dist_to_resistance = (high_20 - data['close']) / data['close']
        dist_to_support = (data['close'] - low_20) / data['close']
        
        # Factor increases near these levels
        resistance_factor = np.exp(-dist_to_resistance * 20)  # Closer = higher
        support_factor = np.exp(-dist_to_support * 20)
        
        # Combine both effects
        factor = (resistance_factor + support_factor) / 2
        
        return factor
    
    def market_regime_factor(self, data: pd.DataFrame) -> pd.Series:
        """
        Factor 8: Market Regime Classification.
        Adapt strategy based on current market conditions.
        """
        # Calculate market indicators
        returns = data['close'].pct_change()
        
        # Trend strength (using moving average slopes)
        ma_short = data['close'].rolling(window=10).mean()
        ma_long = data['close'].rolling(window=30).mean()
        trend = (ma_short - ma_long) / ma_long
        
        # Market volatility
        volatility = returns.rolling(window=20).std()
        vol_percentile = volatility.rolling(window=100).rank(pct=True)
        
        # Mean reversion works better in:
        # - Low to medium volatility (20th-80th percentile)
        # - Sideways markets (low trend)
        
        vol_factor = np.where(
            (vol_percentile > 0.2) & (vol_percentile < 0.8),
            1,  # Good regime
            -0.5  # Poor regime
        )
        
        trend_factor = np.where(
            np.abs(trend) < 0.02,  # Low trend
            0.5,
            -0.5  # Strong trend (momentum regime)
        )
        
        factor = vol_factor + trend_factor
        
        return pd.Series(factor, index=data.index)


