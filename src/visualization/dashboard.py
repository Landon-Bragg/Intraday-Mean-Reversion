# src/visualization/dashboard.py
"""
Interactive Dashboard for Real-time Strategy Monitoring
Professional Plotly/Dash implementation
"""

import dash
from dash import dcc, html, Input, Output, callback
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
            
            # Main content
            html.Div([
                # Performance metrics row
                html.Div([
                    self._create_metric_card("Total Return", "24.5%", "#27ae60"),
                    self._create_metric_card("Sharpe Ratio", "2.1", "#3498db"),
                    self._create_metric_card("Max Drawdown", "-12.3%", "#e74c3c"),
                    self._create_metric_card("Win Rate", "62%", "#f39c12"),
                ], style={'display': 'flex', 'justifyContent': 'space-around',
                         'margin': '20px 0'}),
                
                # Charts row 1
                html.Div([
                    html.Div([
                        dcc.Graph(id='portfolio-performance-chart')
                    ], style={'width': '50%', 'display': 'inline-block'}),
                    
                    html.Div([
                        dcc.Graph(id='drawdown-chart')
                    ], style={'width': '50%', 'display': 'inline-block'})
                ]),
                
                # Charts row 2
                html.Div([
                    html.Div([
                        dcc.Graph(id='factor-performance-chart')
                    ], style={'width': '50%', 'display': 'inline-block'}),
                    
                    html.Div([
                        dcc.Graph(id='risk-metrics-chart')
                    ], style={'width': '50%', 'display': 'inline-block'})
                ]),
                
                # Controls
                html.Div([
                    html.Label('Time Range:', style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='time-range-dropdown',
                        options=[
                            {'label': 'Last 7 Days', 'value': '7D'},
                            {'label': 'Last 30 Days', 'value': '30D'},
                            {'label': 'Last 3 Months', 'value': '3M'},
                            {'label': 'Year to Date', 'value': 'YTD'}
                        ],
                        value='30D',
                        style={'width': '200px'}
                    )
                ], style={'margin': '20px', 'textAlign': 'center'}),
                
                # Update interval
                dcc.Interval(
                    id='interval-component',
                    interval=30*1000,  # Update every 30 seconds
                    n_intervals=0
                )
            ])
        ])
        
    def _create_metric_card(self, title: str, value: str, color: str):
        """Create a metric display card."""
        return html.Div([
            html.H3(title, style={'margin': '0', 'color': '#2c3e50'}),
            html.H2(value, style={'margin': '10px 0', 'color': color,
                                 'fontWeight': 'bold'})
        ], style={
            'backgroundColor': 'white',
            'padding': '20px',
            'borderRadius': '10px',
            'boxShadow': '0 2px 10px rgba(0,0,0,0.1)',
            'textAlign': 'center',
            'width': '200px'
        })
        
    def setup_callbacks(self):
        """Setup dashboard callbacks for interactivity."""
        @self.app.callback(
            [Output('portfolio-performance-chart', 'figure'),
             Output('drawdown-chart', 'figure'),
             Output('factor-performance-chart', 'figure'),
             Output('risk-metrics-chart', 'figure')],
            [Input('time-range-dropdown', 'value'),
             Input('interval-component', 'n_intervals')]
        )
        def update_charts(time_range, n_intervals):
            # Generate sample data (replace with actual data)
            dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
            portfolio_values = 1000000 * np.cumprod(1 + np.random.normal(0.001, 0.02, 100))
            
            # Portfolio performance chart
            portfolio_fig = go.Figure()
            portfolio_fig.add_trace(go.Scatter(
                x=dates, y=portfolio_values,
                mode='lines', name='Portfolio Value',
                line=dict(color='#3498db', width=2)
            ))
            portfolio_fig.update_layout(
                title='Portfolio Performance Over Time',
                xaxis_title='Date',
                yaxis_title='Portfolio Value ($)',
                template='plotly_white'
            )
            
            # Drawdown chart
            peak = np.maximum.accumulate(portfolio_values)
            drawdown = (portfolio_values - peak) / peak * 100
            
            drawdown_fig = go.Figure()
            drawdown_fig.add_trace(go.Scatter(
                x=dates, y=drawdown,
                mode='lines', name='Drawdown',
                line=dict(color='#e74c3c', width=2),
                fill='tozeroy', fillcolor='rgba(231, 76, 60, 0.1)'
            ))
            drawdown_fig.update_layout(
                title='Portfolio Drawdown',
                xaxis_title='Date',
                yaxis_title='Drawdown (%)',
                template='plotly_white'
            )
            
            # Factor performance chart
            factors = ['RSI', 'VWAP', 'Gap', 'Vol Regime', 'Time', 'Volume', 'S/R', 'Market']
            performance = np.random.normal(0.15, 0.05, len(factors))
            
            factor_fig = go.Figure()
            factor_fig.add_trace(go.Bar(
                x=factors, y=performance,
                marker_color=['#27ae60' if p > 0 else '#e74c3c' for p in performance]
            ))
            factor_fig.update_layout(
                title='Factor Performance (Annualized Return)',
                xaxis_title='Factor',
                yaxis_title='Return (%)',
                template='plotly_white'
            )
            
            # Risk metrics chart
            risk_metrics = {
                'VaR (95%)': -2.5,
                'Expected Shortfall': -3.8,
                'Beta': 0.45,
                'Volatility': 18.2
            }
            
            risk_fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=list(risk_metrics.keys()),
                specs=[[{"type": "domain"}, {"type": "domain"}],
                    [{"type": "domain"}, {"type": "domain"}]]
            )

            
            # Add gauge charts for each metric
            for i, (metric, value) in enumerate(risk_metrics.items()):
                row = i // 2 + 1
                col = i % 2 + 1
                
                risk_fig.add_trace(
                    go.Indicator(
                        mode="gauge+number",
                        value=value,
                        title={'text': metric},
                        gauge={'axis': {'range': [-10, 10]},
                              'bar': {'color': "darkblue"},
                              'steps': [{'range': [-10, 0], 'color': "lightgray"},
                                       {'range': [0, 10], 'color': "gray"}],
                              'threshold': {'line': {'color': "red", 'width': 4},
                                          'thickness': 0.75, 'value': 5}}
                    ),
                    row=row, col=col
                )
                
            risk_fig.update_layout(
                title='Risk Metrics Dashboard',
                template='plotly_white'
            )
            
            return portfolio_fig, drawdown_fig, factor_fig, risk_fig
        
    def run_server(self, debug=True, port=8050):
        """Run the dashboard server."""
        self.app.run(debug=debug, port=port)

    def update_data(self, results: Dict):
        """
        Update dashboard data with new backtest results.
        This should store the results for use in layout/rendering.
        """
        self.results = results




