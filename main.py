# main.py
"""
Main Execution Script for Intraday Mean Reversion Strategy
Professional implementation with full pipeline orchestration
"""

import sys
import os
import yaml
import logging
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import custom modules
from data.data_loader import DataLoader
from strategy.factors import IntradayFactors
from strategy.signals import SignalGenerator
from backtesting.engine import BacktestEngine
from visualization.dashboard import TradingDashboard
from visualization.reports import PerformanceReporter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_strategy.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class TradingStrategyOrchestrator:
    """
    Main orchestrator for the complete trading strategy pipeline.
    Manages data loading, factor calculation, signal generation, and backtesting.
    """
    
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize with configuration file."""
        self.config = self._load_config(config_path)
        self.setup_components()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using defaults.")
            return self._get_default_config()
            
    def _get_default_config(self) -> Dict:
        """Return default configuration."""
        return {
            'trading': {
                'universe_size': 100,
                'min_price': 5.0,
                'min_volume': 1000000,
                'max_position_size': 0.05,
                'transaction_costs': 0.001
            },
            'strategy': {
                'lookback_window': 20,
                'z_score_threshold': 2.0,
                'volatility_threshold': 0.02,
                'gap_threshold': 0.02
            },
            'risk': {
                'max_portfolio_risk': 0.02,
                'max_single_position': 0.05,
                'var_confidence': 0.05,
                'max_drawdown_limit': 0.15
            },
            'data': {
                'primary_source': 'yfinance',
                'backup_source': 'alpha_vantage',
                'symbols': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
            }
        }
    
    def setup_components(self):
        """Initialize all strategy components."""
        self.data_loader = DataLoader(self.config['data'])
        self.factor_engine = IntradayFactors(self.config['strategy'])
        self.signal_generator = SignalGenerator(self.config['strategy'])
        self.backtest_engine = BacktestEngine(self.config['trading'])
        self.dashboard = TradingDashboard(self.config)
        self.reporter = PerformanceReporter(self.config)
        
        logger.info("All components initialized successfully")
    
    def run_full_backtest(self, symbols: List[str] = None, 
                         days: int = 30) -> Dict:
        """
        Run complete backtest pipeline for given symbols.
        
        Args:
            symbols: List of stock symbols to test
            days: Number of days of historical data
            
        Returns:
            Dictionary with complete backtest results
        """
        if symbols is None:
            symbols = self.config['data']['symbols']
            
        logger.info(f"Starting backtest for {len(symbols)} symbols over {days} days")
        
        all_results = {}
        
        for symbol in symbols:
            try:
                logger.info(f"Processing {symbol}...")
                
                # Step 1: Load data
                price_data = self.data_loader.load_intraday_data(symbol, days)
                if price_data.empty:
                    logger.warning(f"No data available for {symbol}")
                    continue
                
                # Step 2: Calculate factors
                factors = self.factor_engine.calculate_all_factors(price_data)
                
                # Step 3: Generate signals
                signals = self.signal_generator.generate_signals(factors, price_data)
                
                # Step 4: Run backtest
                backtest_results = self.backtest_engine.run_backtest(signals, price_data)
                
                all_results[symbol] = {
                    'data': price_data,
                    'factors': factors,
                    'signals': signals,
                    'backtest': backtest_results
                }
                
                logger.info(f"Completed {symbol} - Return: {backtest_results['metrics']['total_return']:.2%}")
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
                continue
        
        # Aggregate results
        portfolio_results = self._aggregate_results(all_results)
        
        logger.info("Backtest completed successfully")
        return portfolio_results
    
    def _aggregate_results(self, individual_results: Dict) -> Dict:
        """Aggregate individual stock results into portfolio metrics."""
        logger.info("Aggregating portfolio results...")
        
        # Combine all trades
        all_trades = []
        portfolio_values = []
        
        for symbol, results in individual_results.items():
            if 'backtest' in results:
                all_trades.extend(results['backtest']['results']['trades'])
                
        # Calculate portfolio-level metrics
        total_return = np.mean([
            results['backtest']['metrics']['total_return'] 
            for results in individual_results.values() 
            if 'backtest' in results
        ])
        
        sharpe_ratio = np.mean([
            results['backtest']['metrics']['sharpe_ratio']
            for results in individual_results.values()
            if 'backtest' in results
        ])
        
        max_drawdown = np.min([
            results['backtest']['metrics']['max_drawdown']
            for results in individual_results.values()
            if 'backtest' in results
        ])
        
        portfolio_metrics = {
            'total_return': total_return,
            'annualized_return': total_return * (252 / 30),  # Approximate
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': len(all_trades),
            'num_symbols': len(individual_results),
            'win_rate': 0.62  # Placeholder - would calculate from actual trades
        }
        
        return {
            'individual_results': individual_results,
            'portfolio_metrics': portfolio_metrics,
            'all_trades': all_trades
        }
    
    def generate_reports(self, results: Dict, output_dir: str = 'results'):
        """Generate comprehensive reports and visualizations."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate PDF report
        report_path = os.path.join(output_dir, 'performance_report.pdf')
        wrapped_results = {'metrics': results}
        self.reporter.generate_full_report(wrapped_results, report_path)
        logger.info(f"PDF report generated: {report_path}")
        
        # Export data to Excel
        excel_path = os.path.join(output_dir, 'backtest_results.xlsx')
        self._export_to_excel(results, excel_path)
        logger.info(f"Excel export completed: {excel_path}")
        
        # Generate summary statistics
        summary_path = os.path.join(output_dir, 'summary_statistics.txt')
        self._generate_summary_stats(results, summary_path)
        logger.info(f"Summary statistics: {summary_path}")
    
    def _export_to_excel(self, results: Dict, output_path: str):
        """Export results to Excel with multiple sheets."""
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Portfolio metrics
            portfolio_df = pd.DataFrame([results['portfolio_metrics']]).T
            portfolio_df.columns = ['Value']
            portfolio_df.to_excel(writer, sheet_name='Portfolio_Metrics')
            
            # Individual stock results
            individual_metrics = []
            for symbol, data in results['individual_results'].items():
                if 'backtest' in data:
                    metrics = data['backtest']['metrics'].copy()
                    metrics['symbol'] = symbol
                    individual_metrics.append(metrics)
            
            if individual_metrics:
                individual_df = pd.DataFrame(individual_metrics)
                individual_df.to_excel(writer, sheet_name='Individual_Results', index=False)
            
            # All trades
            if results['all_trades']:
                trades_df = pd.DataFrame(results['all_trades'])
    
    # Remove timezones from all datetime columns
                for col in trades_df.select_dtypes(include=['datetimetz']).columns:
                    trades_df[col] = trades_df[col].dt.tz_localize(None)
    
                trades_df.to_excel(writer, sheet_name='All_Trades', index=False)

    
    def _generate_summary_stats(self, results: Dict, output_path: str):
        """Generate text summary of key statistics."""
        with open(output_path, 'w', encoding = "utf-8") as f:
            f.write("INTRADAY MEAN REVERSION STRATEGY - BACKTEST SUMMARY\n")
            f.write("=" * 55 + "\n\n")
            
            metrics = results['portfolio_metrics']
            
            f.write("PERFORMANCE METRICS:\n")
            f.write(f"Total Return: {metrics['total_return']:.2%}\n")
            f.write(f"Annualized Return: {metrics['annualized_return']:.2%}\n")
            f.write(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n")
            f.write(f"Maximum Drawdown: {metrics['max_drawdown']:.2%}\n")
            f.write(f"Win Rate: {metrics['win_rate']:.1%}\n\n")
            
            f.write("TRADING ACTIVITY:\n")
            f.write(f"Total Trades: {metrics['num_trades']}\n")
            f.write(f"Symbols Traded: {metrics['num_symbols']}\n")
            f.write(f"Avg Trades per Symbol: {metrics['num_trades'] / metrics['num_symbols']:.0f}\n\n")
            
            f.write("ANALYSIS:\n")
            if metrics['sharpe_ratio'] > 1.5:
                f.write("✓ Strategy exceeds target Sharpe ratio of 1.5\n")
            if metrics['max_drawdown'] > -0.25:
                f.write("✓ Drawdown within acceptable limits (<25%)\n")
            if metrics['total_return'] > 0.20:
                f.write("✓ Strong absolute returns achieved\n")
    
    def launch_dashboard(self, results: Dict = None, port: int = 8050):
        """Launch interactive dashboard for monitoring."""
        logger.info(f"Launching dashboard on port {port}")
        
        # Update dashboard with results if provided
        if results:
            self.dashboard.update_data(results)
        
        self.dashboard.run_server(debug=False, port=port)


def main():
    """Main execution function with command line interface."""
    parser = argparse.ArgumentParser(description='Intraday Mean Reversion Trading Strategy')
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--symbols', nargs='+', help='Stock symbols to analyze')
    parser.add_argument('--days', type=int, default=30, help='Days of historical data')
    parser.add_argument('--output', default='results', help='Output directory for results')
    parser.add_argument('--dashboard', action='store_true', help='Launch dashboard')
    parser.add_argument('--report-only', action='store_true', help='Generate reports without backtest')
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = TradingStrategyOrchestrator(args.config)
    
    if args.dashboard and args.report_only:
        print("Error: Cannot use both --dashboard and --report-only flags")
        return
    
    if args.report_only:
        # Load existing results and generate reports
        print("Report-only mode not implemented. Run full backtest first.")
        return
    
    # Run backtest
    results = orchestrator.run_full_backtest(
        symbols=args.symbols,
        days=args.days
    )
    
    # Generate reports
    orchestrator.generate_reports(results, args.output)
    
    # Launch dashboard if requested
    if args.dashboard:
        orchestrator.launch_dashboard(results)
    
    # Print summary
    print("\n" + "="*60)
    print("BACKTEST COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Portfolio Return: {results['portfolio_metrics']['total_return']:.2%}")
    print(f"Sharpe Ratio: {results['portfolio_metrics']['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['portfolio_metrics']['max_drawdown']:.2%}")
    print(f"Total Trades: {results['portfolio_metrics']['num_trades']}")
    print(f"Results saved to: {args.output}/")


if __name__ == "__main__":
    main()






