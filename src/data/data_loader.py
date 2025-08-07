# src/data/data_loader.py
"""
Professional Data Loading and Preprocessing Module
Handles multiple data sources with fallbacks and quality checks
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time
from typing import Dict
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Professional data loader with multiple sources and quality controls.
    Supports Alpha Vantage, Yahoo Finance, and other providers.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.alpha_vantage_key = config.get('data', {}).get('api_keys', {}).get('alpha_vantage')
        self.rate_limits = {
            'alpha_vantage': {'calls': 0, 'last_reset': datetime.now()},
            'yfinance': {'calls': 0, 'last_reset': datetime.now()}
        }
        print(f"[DEBUG] Alpha Vantage key loaded: {self.alpha_vantage_key}")

        
    def load_intraday_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """
        Load intraday minute data with fallback sources.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            days: Number of days of data to fetch
            
        Returns:
            DataFrame with OHLCV minute data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Try primary source first
        try:
            if self.config.get('primary_source') == 'alpha_vantage':
                data = self._fetch_alpha_vantage_intraday(symbol)
            else:
                data = self._fetch_yahoo_intraday(symbol, start_date, end_date)
                
            if self._validate_data(data):
                logger.info(f"Successfully loaded {len(data)} rows for {symbol}")
                return self._preprocess_data(data)
                
        except Exception as e:
            logger.warning(f"Primary source failed for {symbol}: {e}")
            
        # Try backup source
        try:
            if self.config.get('backup_source') == 'yfinance':
                data = self._fetch_yahoo_intraday(symbol, start_date, end_date)
            else:
                data = self._fetch_alpha_vantage_intraday(symbol)
                
            if self._validate_data(data):
                logger.info(f"Backup source succeeded for {symbol}")
                return self._preprocess_data(data)
                
        except Exception as e:
            logger.error(f"All sources failed for {symbol}: {e}")
            raise
            
    def _fetch_alpha_vantage_intraday(self, symbol: str) -> pd.DataFrame:
        """Fetch intraday data from Alpha Vantage API."""
        if not self._check_rate_limit('alpha_vantage'):
            time.sleep(12)  # Wait for rate limit reset
            
        url = 'https://www.alphavantage.co/query'
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': symbol,
            'interval': '5min',
            'apikey': self.alpha_vantage_key,
            'outputsize': 'full',
            'datatype': 'json'
        }
        
        response = requests.get(url, params=params)
        self.rate_limits['alpha_vantage']['calls'] += 1
        
        if response.status_code != 200:
            raise Exception(f"API request failed: {response.status_code}")
            
        data = response.json()
        
        if 'Error Message' in data:
            raise Exception(f"API Error: {data['Error Message']}")
            
        if 'Note' in data:
            raise Exception("Rate limit exceeded")
            
        # Parse time series data
        time_series = data.get('Time Series (5min)', {})
        
        df_data = []
        for timestamp, values in time_series.items():
            df_data.append({
                'timestamp': pd.to_datetime(timestamp),
                'open': float(values['1. open']),
                'high': float(values['2. high']),
                'low': float(values['3. low']),
                'close': float(values['4. close']),
                'volume': int(values['5. volume'])
            })
            
        df = pd.DataFrame(df_data)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        return df
        
    def _fetch_yahoo_intraday(self, symbol: str, start_date: datetime, 
                             end_date: datetime) -> pd.DataFrame:
        """Fetch intraday data from Yahoo Finance."""
        ticker = yf.Ticker(symbol)
        
        # Yahoo Finance requires specific period format for intraday
        data = ticker.history(
            start=start_date,
            end=end_date,
            interval='5m',
            prepost=False,
            auto_adjust=True,
            back_adjust=False
        )
        
        if data.empty:
            raise Exception("No data returned from Yahoo Finance")
            
        # Standardize column names
        data.columns = [col.lower() for col in data.columns]
        
        return data
        
    def _check_rate_limit(self, source: str) -> bool:
        """Check if we're within rate limits for the source."""
        limits = self.rate_limits[source]
        now = datetime.now()
        
        # Reset counter if it's been more than an hour
        if (now - limits['last_reset']).seconds > 3600:
            limits['calls'] = 0
            limits['last_reset'] = now
            
        # Alpha Vantage: 5 calls per minute, 500 per day
        if source == 'alpha_vantage':
            return limits['calls'] < 5
            
        return True
        
    def _validate_data(self, data: pd.DataFrame) -> bool:
        """Validate data quality and completeness."""
        if data is None or data.empty:
            return False
            
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            return False
            
        # Check for reasonable data ranges
        if data['close'].max() > 10000 or data['close'].min() < 0:
            return False
            
        # Check for minimum data points
        if len(data) < 100:
            return False
            
        return True
        
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess raw data."""
        # Remove any NaN values
        data = data.dropna()
        
        # Ensure proper data types
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        data[numeric_cols] = data[numeric_cols].astype(float)
        
        # Remove outliers (using IQR method)
        for col in ['open', 'high', 'low', 'close']:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
            
        # Ensure high >= low, close between high and low
        data = data[data['high'] >= data['low']]
        data = data[(data['close'] >= data['low']) & (data['close'] <= data['high'])]
        
        # Add derived features
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
        
        return data



