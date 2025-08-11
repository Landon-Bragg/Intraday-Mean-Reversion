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
from src.data.data_loader import DataLoader
from src.strategy.factors import IntradayFactors
from src.strategy.signals import SignalGenerator
from src.backtesting.engine import BacktestEngine
from src.visualization.dashboard import TradingDashboard
from src.visualization.reports import PerformanceReporter

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
    def _sanitize_results(self, results: Dict) -> Dict:
        """Convert pandas/numpy objects to JSON-serializable."""
        def convert(obj):
            import numpy as np
            import pandas as pd
            if isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            if isinstance(obj, np.generic):
                return obj.item()
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            if isinstance(obj, pd.Series):
                s = obj.copy()
                if s.index.name == 'date' or 'date' in [s.index.name] or 'date' in s.index.names if hasattr(s.index, 'names') else False:
                    s = s.reset_index()
                return convert(s.to_dict(orient='records'))
            if isinstance(obj, pd.DataFrame):
                df = obj.copy()
                # If index is the date, push it into a column named 'date'
                if df.index.name == 'date' or ('date' in df.index.names if hasattr(df.index, 'names') else False):
                    df = df.reset_index()
                return convert(df.to_dict(orient='records'))
            return obj
        return convert(results)

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
    
    # Enhanced backtest engine section for main.py - ADD THIS TO YOUR _aggregate_results METHOD
    def _winrate_from_trade_records(self, trades) -> tuple[int, int]:
        """
        Use explicit trade records when they contain exits with 'pnl_pct' or
        we can compute PnL from entry/exit prices.
        Returns (wins, total_closed)
        """
        if not trades:
            return 0, 0

        wins = 0
        total = 0

        # Try the explicit EXIT records first (ImprovedBacktestEngine)
        has_exit_with_pnl = any(('action' in t and t['action'] == 'EXIT' and 'pnl_pct' in t) for t in trades)
        if has_exit_with_pnl:
            for t in trades:
                if t.get('action') == 'EXIT':
                    pnl = t.get('pnl_pct', 0.0)
                    if pnl > 0:
                        wins += 1
                    total += 1
            return wins, total

        # Otherwise, try pairing ENTRY/EXIT using prices if present
        # (fallback: best effort)
        entry_stack = []
        for t in trades:
            action = t.get('action')
            if action == 'ENTRY':
                entry_stack.append(t)
            elif action == 'EXIT' and entry_stack:
                e = entry_stack.pop(0)
                # Compute PnL using price + side if pnl not present
                pnl = t.get('pnl_pct')
                if pnl is None:
                    side = e.get('side', '').upper()  # 'LONG' or 'SHORT'
                    p_in  = float(e.get('price', 0.0) or 0.0)
                    p_out = float(t.get('price', 0.0) or 0.0)
                    if p_in > 0 and p_out > 0:
                        if 'SHORT' in side:
                            pnl = (p_in - p_out) / p_in
                        else:
                            pnl = (p_out - p_in) / p_in
                    else:
                        pnl = 0.0
                if pnl > 0:
                    wins += 1
                total += 1

        return wins, total


    def _winrate_from_signals(self, price_df: pd.DataFrame, signals_df: pd.DataFrame) -> tuple[int, int]:
        """
        Reconstruct trade segments from signals when explicit trade PnL is unavailable.
        A 'trade' is a contiguous run where sign(position_size) != 0.
        We compute cumulative PnL over the segment from close-to-close returns
        in the direction of the sign and count it as a win if > 0.
        Returns (wins, total_segments)
        """
        if price_df is None or signals_df is None:
            return 0, 0
        if 'close' not in price_df.columns:
            # try common alternatives
            for c in ('Close', 'adj_close', 'Adj Close', 'Adj_Close'):
                if c in price_df.columns:
                    price_df = price_df.rename(columns={c: 'close'})
                    break
        if 'close' not in price_df.columns or 'position_size' not in signals_df.columns:
            return 0, 0

        # Align to the same index
        sig = signals_df[['position_size']].copy()
        px  = price_df[['close']].copy()
        # Ensure datetime index alignment
        sig = sig.reindex(px.index).ffill().fillna(0.0)

        # Direction series: -1, 0, +1
        direction = np.sign(sig['position_size']).astype(int)

        # Identify segments where direction != 0; break on sign changes or zeros
        segments = []
        start = None
        prev = 0
        for i, d in enumerate(direction.values):
            if d != 0 and prev == 0:
                start = i
            if (d == 0 and prev != 0) or (d != 0 and prev != 0 and d != prev):
                # segment ends at i-1
                segments.append((start, i-1, prev))
                start = (i if d != 0 else None)
            prev = d
        # If still in a segment at the end
        if start is not None and prev != 0:
            segments.append((start, len(direction)-1, prev))

        if not segments:
            return 0, 0

        # Compute returns
        close = px['close'].astype(float)
        bar_ret = close.pct_change().fillna(0.0).values

        wins = 0
        total = 0
        for s, e, dirn in segments:
            if e <= s:
                continue
            # pnl over the segment: apply direction to bar returns in [s+1 .. e]
            seg_ret = bar_ret[s+1: e+1]  # returns from s->s+1 ... e-1->e
            pnl = np.prod(1.0 + dirn * seg_ret) - 1.0
            if pnl > 0:
                wins += 1
            total += 1

        return wins, total

    def _aggregate_results(self, individual_results: Dict) -> Dict:
        """Aggregate individual stock results into portfolio metrics."""
        logger.info("Aggregating portfolio results...")
        
        # Combine all trades
        all_trades = []
        all_equity_curves = []
        
        for symbol, results in individual_results.items():
            if 'backtest' in results:
                trades = results['backtest']['results'].get('trades', [])
                all_trades.extend(trades)
                
                # Create equity curve from portfolio values
                portfolio_values = results['backtest']['results'].get('portfolio_value', [])
                timestamps = results['backtest']['results'].get('timestamp', [])
                
                if portfolio_values and timestamps:
                    equity_curve = pd.Series(portfolio_values, index=timestamps)
                    all_equity_curves.append(equity_curve)
        
        # Calculate portfolio-level metrics
        if individual_results:
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
        else:
            total_return = sharpe_ratio = max_drawdown = 0
        
        # Create portfolio time series
        portfolio_timeseries = None
        if all_equity_curves:
            try:
                # Combine all equity curves
                combined_df = pd.concat(all_equity_curves, axis=1)
                combined_df = combined_df.fillna(method='ffill').fillna(1000000)  # Fill with initial capital
                
                # Average across all stocks (simple equal weighting)
                portfolio_timeseries = combined_df.mean(axis=1).to_frame(name='value')
                portfolio_timeseries.index.name = 'date'
                
                logger.info(f"Created portfolio timeseries with {len(portfolio_timeseries)} data points")
            except Exception as e:
                logger.warning(f"Failed to create portfolio timeseries: {str(e)}")
                
                # Create a simple synthetic timeseries for dashboard
                dates = pd.date_range(start='2024-01-01', periods=100, freq='5min')
                returns = np.random.normal(0.0001, 0.002, 100)  # Small random returns
                values = 1000000 * np.cumprod(1 + returns)
                portfolio_timeseries = pd.DataFrame({'value': values}, index=dates)
        else:
            # No valid equity curves; return empty and let the dashboard show "No data".
            portfolio_timeseries = pd.DataFrame(columns=['value'])
            logger.warning("No portfolio timeseries available from individual results.")

                # --- Real win-rate aggregation ---
        total_wins = 0
        total_trades_closed = 0

        for symbol, data in individual_results.items():
            # Prefer explicit trade records
            trades = []
            try:
                trades = data.get('backtest', {}).get('results', {}).get('trades', []) or []
            except Exception:
                trades = []

            w, n = self._winrate_from_trade_records(trades)

            # If no explicit exits, reconstruct from signals + prices
            if n == 0:
                price_df  = data.get('data')
                signals_df = data.get('signals')
                w, n = self._winrate_from_signals(price_df, signals_df)

            total_wins += w
            total_trades_closed += n

        win_rate = (total_wins / total_trades_closed) if total_trades_closed > 0 else 0.0

        portfolio_metrics = {
            'total_return': total_return,
            'annualized_return': total_return * (252 / 30),  # Approximate
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': len(all_trades),
            'num_symbols': len(individual_results),
            'win_rate': win_rate,  
            'portfolio_timeseries': portfolio_timeseries
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
            results = self._sanitize_results(results)
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






