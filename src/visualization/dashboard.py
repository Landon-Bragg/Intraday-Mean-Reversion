# src/visualization/dashboard.py
"""
Interactive Dashboard for Real-time Strategy Monitoring
Professional Plotly/Dash implementation - FIXED VERSION
"""

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List
import datetime as dt


class TradingDashboard:
    """
    Professional trading dashboard with real-time monitoring capabilities.
    Features performance tracking, risk monitoring, and factor analysis.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.app = dash.Dash(__name__)
        self.results = None  # Initialize results storage
        self.setup_layout()
        self.setup_callbacks()
        
    def setup_layout(self):
        """Create the dashboard layout."""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("Intraday Mean Reversion Strategy Dashboard",
                       style={'textAlign': 'center', 'color': '#2c3e50',
                             'fontFamily': 'Arial, sans-serif'}),
                html.P("Professional Quantitative Trading System",
                      style={'textAlign': 'center', 'color': '#7f8c8d'})
            ], className='header', style={'padding': '20px', 'backgroundColor': '#ecf0f1'}),
            
            # Performance metrics row
            html.Div(id='metrics-row', style={'display': 'flex', 'justifyContent': 'space-around', 'margin': '20px 0'}),
            
            # Main content
            html.Div([
                # Charts row 1
                html.Div([
                    html.Div([
                        dcc.Graph(id='portfolio-performance-chart')
                    ], style={'width': '48%', 'display': 'inline-block', 'margin': '1%'}),
                    
                    html.Div([
                        dcc.Graph(id='drawdown-chart')
                    ], style={'width': '48%', 'display': 'inline-block', 'margin': '1%'})
                ]),
                
                # Charts row 2
                html.Div([
                    html.Div([
                        dcc.Graph(id='factor-performance-chart')
                    ], style={'width': '48%', 'display': 'inline-block', 'margin': '1%'}),
                    
                    html.Div([
                        dcc.Graph(id='risk-metrics-chart')
                    ], style={'width': '48%', 'display': 'inline-block', 'margin': '1%'})
                ]),
                
                # Controls
                html.Div([
                    html.Label('Time Range:', style={'fontWeight': 'bold', 'marginRight': '10px'}),
                    dcc.Dropdown(
                        id='time-range-dropdown',
                        options=[
                            {'label': 'Last 7 Days', 'value': '7D'},
                            {'label': 'Last 30 Days', 'value': '30D'},
                            {'label': 'Last 3 Months', 'value': '3M'},
                            {'label': 'Year to Date', 'value': 'YTD'}
                        ],
                        value='30D',
                        style={'width': '200px', 'display': 'inline-block'}
                    )
                ], style={'margin': '20px', 'textAlign': 'center'}),
                
                # Update interval
                dcc.Interval(
                    id='interval-component',
                    interval=30*1000,  # Update every 30 seconds
                    n_intervals=0
                ),
                
                # Store for data
                dcc.Store(id='data-store')
            ])
        ])
        
    def _create_metric_card(self, title: str, value: str, color: str):
        """Create a metric display card."""
        return html.Div([
            html.H4(title, style={'margin': '0', 'color': '#2c3e50', 'fontSize': '14px'}),
            html.H2(value, style={'margin': '10px 0', 'color': color,
                                 'fontWeight': 'bold', 'fontSize': '24px'})
        ], style={
            'backgroundColor': 'white',
            'padding': '15px',
            'borderRadius': '8px',
            'boxShadow': '0 2px 10px rgba(0,0,0,0.1)',
            'textAlign': 'center',
            'width': '180px',
            'margin': '5px'
        })
    def _parse_portfolio_df(self, stored_data):
        """Return a clean DataFrame with ['date','value'] from the stored_data structure."""
        pm = stored_data.get('portfolio_metrics', {})
        pt = pm.get('portfolio_timeseries')

        if pt is None:
            return pd.DataFrame(columns=['date', 'value'])

        # Handle sanitize formats: list[dict] or dict of arrays
        if isinstance(pt, list):
            df = pd.DataFrame(pt)
        elif isinstance(pt, dict):
            df = pd.DataFrame(pt)
        else:
            # Last resort: try to convert directly
            df = pd.DataFrame(pt)

        # Normalize column names
        cols = {c.lower(): c for c in df.columns}
        date_col = cols.get('date') or cols.get('timestamp') or cols.get('time') or cols.get('index')
        value_col = cols.get('value') or cols.get('portfolio_value')

        if date_col is None or value_col is None or df.empty:
            return pd.DataFrame(columns=['date', 'value'])

        df = df[[date_col, value_col]].rename(columns={date_col: 'date', value_col: 'value'})
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df.dropna(subset=['date', 'value']).sort_values('date').reset_index(drop=True)
        return df

    def _apply_time_range(self, df, trange):
        """Filter df by time-range dropdown value."""
        if df.empty:
            return df
        end = df['date'].max()
        if trange == '7D':
            start = end - pd.Timedelta(days=7)
        elif trange == '30D':
            start = end - pd.Timedelta(days=30)
        elif trange == '3M':
            start = end - pd.DateOffset(months=3)
        elif trange == 'YTD':
            start = pd.Timestamp(year=end.year, month=1, day=1, tz=end.tz) if end.tz is not None else pd.Timestamp(year=end.year, month=1, day=1)
        else:
            return df
        return df[df['date'] >= start]
    
    def setup_callbacks(self):
        """Setup dashboard callbacks for interactivity."""
        
        @self.app.callback(
            Output('data-store', 'data'),
            Input('interval-component', 'n_intervals')
        )
        def update_data_store(n_intervals):
            """Store current results data."""
            if self.results is None:
                return {}
            return self.results
        
        @self.app.callback(
            Output('metrics-row', 'children'),
            Input('data-store', 'data')
        )
        def update_metrics(stored_data):
            """Update performance metrics cards."""
            if not stored_data or 'portfolio_metrics' not in stored_data:
                return [
                    self._create_metric_card("Total Return", "N/A", "#95a5a6"),
                    self._create_metric_card("Sharpe Ratio", "N/A", "#95a5a6"),
                    self._create_metric_card("Max Drawdown", "N/A", "#95a5a6"),
                    self._create_metric_card("Win Rate", "N/A", "#95a5a6"),
                ]

            metrics = stored_data['portfolio_metrics']
            
            return [
                self._create_metric_card("Total Return", f"{metrics.get('total_return', 0):.2%}", "#27ae60"),
                self._create_metric_card("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}", "#3498db"),
                self._create_metric_card("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2%}", "#e74c3c"),
                self._create_metric_card("Win Rate", f"{metrics.get('win_rate', 0):.1%}", "#f39c12"),
            ]
        
        @self.app.callback(
            [Output('portfolio-performance-chart', 'figure'),
             Output('drawdown-chart', 'figure'),
             Output('factor-performance-chart', 'figure'),
             Output('risk-metrics-chart', 'figure')],
            [Input('time-range-dropdown', 'value'),
             Input('data-store', 'data')]
        )
        def update_charts(time_range, stored_data):
            """Update all charts based on real data."""
            # Default figure
            def make_empty():
                fig = go.Figure()
                fig.add_annotation(text="No data available", xref="paper", yref="paper",
                                x=0.5, y=0.5, showarrow=False, font=dict(size=16, color="gray"))
                fig.update_layout(template='plotly_white')
                return fig

            if not stored_data or 'portfolio_metrics' not in stored_data:
                empty = make_empty()
                return empty, empty, empty, empty

            # ---- Portfolio DF (real) ----
            df = self._parse_portfolio_df(stored_data)
            if df.empty:
                empty = make_empty()
                return empty, empty, empty, empty

            df = self._apply_time_range(df, time_range)
            if df.empty:
                empty = make_empty()
                return empty, empty, empty, empty

            dates = df['date']
            pv = df['value'].astype(float)

            # ---- 1) Portfolio performance (real) ----
            portfolio_fig = go.Figure()
            portfolio_fig.add_trace(go.Scatter(
                x=dates, y=pv, mode='lines', name='Portfolio Value',
                line=dict(color='#3498db', width=2)
            ))
            portfolio_fig.update_layout(
                title='Portfolio Performance',
                xaxis_title='Date',
                yaxis_title='Portfolio Value ($)',
                template='plotly_white'
            )

            # ---- 2) Drawdown (real) ----
            peak = np.maximum.accumulate(pv.values)
            drawdown = (pv.values - peak) / peak * 100.0  # %
            drawdown_fig = go.Figure()
            drawdown_fig.add_trace(go.Scatter(
                x=dates, y=drawdown, mode='lines', name='Drawdown (%)',
                line=dict(width=2), fill='tozeroy'
            ))
            drawdown_fig.update_layout(
                title='Portfolio Drawdown',
                xaxis_title='Date',
                yaxis_title='Drawdown (%)',
                template='plotly_white'
            )

            # ---- 3) Replace "Factor Performance" with Rolling Sharpe (real) ----
            # simple returns from equity curve (assumes equally spaced bars)
            rets = pd.Series(pv).pct_change().dropna()
            # use window ~ 1 trading day (for 5-min bars ~ 78 bars). If your bars differ, adjust.
            win = min(78, max(5, len(rets)))  # guard window
            if len(rets) >= 5:
                roll_mean = rets.rolling(win, min_periods=5).mean()
                roll_std  = rets.rolling(win, min_periods=5).std()
                # annualize with intraday bars ~ 78/day * 252
                ann_factor = 78 * 252
                rolling_sharpe = (roll_mean / (roll_std + 1e-12)) * np.sqrt(ann_factor)
                rs_dates = dates.iloc[1:]  # align after pct_change
                rs_fig = go.Figure()
                rs_fig.add_trace(go.Scatter(
                    x=rs_dates, y=rolling_sharpe, mode='lines', name='Rolling Sharpe',
                ))
                rs_fig.update_layout(
                    title=f'Rolling Sharpe (window={win})',
                    xaxis_title='Date',
                    yaxis_title='Sharpe',
                    template='plotly_white'
                )
            else:
                rs_fig = make_empty()

            # ---- 4) Risk metrics (real VaR/ES/Vol/MaxDD) ----
            def var_es(x, alpha=0.95):
                if len(x) == 0:
                    return 0.0, 0.0
                q = np.quantile(x, 1 - alpha)  # e.g., 5% left tail is negative
                es = x[x <= q].mean() if (x <= q).any() else q
                return float(q), float(es)

            daily_factor = 78  # ~ five-minute bars per day
            ann_vol = rets.std(ddof=0) * np.sqrt(daily_factor * 252) if len(rets) > 1 else 0.0
            mdd = float((pd.Series(pv) / pd.Series(pv).cummax() - 1.0).min())  # fractional (negative)
            var95, es95 = var_es(rets.values, alpha=0.95)  # per-bar returns

            # Convert to human friendly:
            risk_labels = ['VaR 95% (per bar)', 'ES 95% (per bar)', 'Ann. Vol', 'Max DD']
            risk_vals = [var95 * 100.0, es95 * 100.0, ann_vol * 100.0, mdd * 100.0]

            risk_fig = go.Figure()
            risk_fig.add_trace(go.Bar(x=risk_labels, y=risk_vals))
            risk_fig.update_layout(
                title='Risk Metrics (from realized returns)',
                xaxis_title='Metric',
                yaxis_title='Value (%)',
                template='plotly_white'
            )

            return portfolio_fig, drawdown_fig, rs_fig, risk_fig

    
    def run_server(self, debug=True, port=8050, host='127.0.0.1'):
        """Run the dashboard server."""
        print(f"Dashboard starting at http://{host}:{port}")
        self.app.run(debug=debug, port=port, host=host)

    def update_data(self, results: Dict):
        """Update dashboard with new results."""
        self.results = results
        print(f"Dashboard data updated with {len(results)} result keys")