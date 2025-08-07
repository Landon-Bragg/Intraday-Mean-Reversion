import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import datetime

class RiskMonitor:
    """Real-time risk monitoring system."""
    
    def __init__(self, config):
        self.config = config
        self.positions = {}
        self.alerts = []
        
    def update_positions(self, positions: Dict):
        """Update current positions."""
        self.positions = positions
        self.check_risk_limits()
        
    def check_risk_limits(self):
        """Check all risk limits."""
        # Portfolio level risk
        total_exposure = sum(abs(pos['value']) for pos in self.positions.values())
        
        if total_exposure > self.config['risk']['max_portfolio_risk']:
            self.add_alert("CRITICAL", "Portfolio exposure limit breached")
            
        # Position level risk
        for symbol, position in self.positions.items():
            exposure_pct = abs(position['value']) / total_exposure
            if exposure_pct > self.config['risk']['max_single_position']:
                self.add_alert("WARNING", f"{symbol} position size limit breached")
                
    def add_alert(self, level: str, message: str):
        """Add risk alert."""
        alert = {
            'timestamp': datetime.now(),
            'level': level,
            'message': message
        }
        self.alerts.append(alert)
        print(f"RISK ALERT [{level}]: {message}")