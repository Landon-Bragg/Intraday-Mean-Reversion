import pandas as pd
from datetime import datetime, timedelta
from main import TradingStrategyOrchestrator

def walk_forward_analysis(symbols, total_days=90, train_days=60, test_days=30):
    """Perform walk-forward analysis."""
    orchestrator = TradingStrategyOrchestrator()
    results = []
    
    for start_offset in range(0, total_days - train_days, test_days):
        print(f"Training period: {start_offset} to {start_offset + train_days}")
        print(f"Testing period: {start_offset + train_days} to {start_offset + train_days + test_days}")
        
        # Train on training period
        train_results = orchestrator.run_full_backtest(symbols, days=train_days)
        
        # Test on out-of-sample period  
        test_results = orchestrator.run_full_backtest(symbols, days=test_days)
        
        results.append({
            'train_return': train_results['portfolio_metrics']['total_return'],
            'test_return': test_results['portfolio_metrics']['total_return'],
            'train_sharpe': train_results['portfolio_metrics']['sharpe_ratio'],
            'test_sharpe': test_results['portfolio_metrics']['sharpe_ratio']
        })
    
    return pd.DataFrame(results)

# Run analysis
wf_results = walk_forward_analysis(['AAPL', 'MSFT', 'GOOGL'])
print(wf_results)
print(f"Average OOS Return: {wf_results['test_return'].mean():.2%}")
print(f"Average OOS Sharpe: {wf_results['test_sharpe'].mean():.2f}")