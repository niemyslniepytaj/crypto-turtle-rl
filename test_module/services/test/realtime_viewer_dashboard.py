"""Dashboard Dash: podgląd sesji realtime z shared/realtime_simulation."""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
REALTIME_DIR = os.path.join(BASE_DIR, "shared", "realtime_simulation")

THEME = {
    'bg': '#111111',
    'sidebar': '#0a0a0a',
    'card': '#1e1e1e',
    'inner_card': '#161616',
    'text': '#ffffff',
    'subtext': '#a0a0a0',
    'border': '#333333',
    'accent': '#00e676',
    'danger': '#f78b8b',
    'blue': '#448aff',
    'orange': '#ffab40',
    'dropdown_bg': '#2a2a2a'
}

app = dash.Dash(__name__, title="Crypto-Turtle Bot")

app.index_string = f'''
<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{{%title%}}</title>
        {{%favicon%}}
        {{%css%}}
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
            body {{
                font-family: 'Inter', sans-serif !important;
                margin: 0;
                background-color: {THEME['bg']};
                color: {THEME['text']};
                overflow-x: hidden;
            }}

            /* --- DARK MODE DLA DROPDOWNA --- */
            .dash-dropdown .Select-control {{
                background-color: {THEME['dropdown_bg']} !important;
                border: 1px solid {THEME['border']} !important;
                color: #ffffff !important;
            }}
            .dash-dropdown .Select-value-label {{ color: #ffffff !important; }}
            .dash-dropdown .Select-menu-outer {{
                background-color: {THEME['dropdown_bg']} !important;
                border: 1px solid {THEME['border']} !important;
            }}
            .dash-dropdown .VirtualizedSelectOption {{
                background-color: {THEME['dropdown_bg']} !important;
                color: #ffffff !important;
            }}
            .dash-dropdown .VirtualizedSelectFocusedOption {{
                background-color: #3d3d3d !important;
                color: {THEME['accent']} !important;
            }}
            .dash-dropdown .Select-arrow-zone, .dash-dropdown .Select-clear-zone {{ color: #aaaaaa !important; }}
            .dash-dropdown .Select-placeholder {{ color: #aaaaaa !important; }}

            /* --- TABELA ZLECEŃ --- */
            .orders-table {{ width: 100%; border-collapse: collapse; text-align: left; font-size: 13px; }}
            .orders-table th {{ 
                position: sticky; top: 0; background-color: {THEME['inner_card']}; 
                padding: 12px 15px; border-bottom: 2px solid {THEME['border']}; 
                color: {THEME['subtext']}; font-weight: 600; z-index: 10;
            }}
            .orders-table td {{ 
                padding: 12px 15px; border-bottom: 1px solid #222; color: {THEME['text']}; 
            }}
            .orders-table tbody tr:hover {{ background-color: #252525; }}

            /* Scrollbar */
            ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
            ::-webkit-scrollbar-track {{ background: {THEME['sidebar']}; }}
            ::-webkit-scrollbar-thumb {{ background: {THEME['border']}; border-radius: 4px; }}
            ::-webkit-scrollbar-thumb:hover {{ background: #555; }}
        </style>
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>
'''

# --- LAYOUT ---
app.layout = html.Div(style={'backgroundColor': THEME['bg'], 'minHeight': '100vh', 'fontFamily': 'Inter'}, children=[

    # Sidebar (Lewy panel)
    html.Div(style={
        'width': '320px', 'position': 'fixed', 'top': 0, 'left': 0, 'bottom': 0,
        'backgroundColor': THEME['sidebar'], 'padding': '25px', 'borderRight': f'1px solid {THEME["border"]}',
        'zIndex': 1000, 'display': 'flex', 'flexDirection': 'column'
    }, children=[
        html.Div([
            html.H2("Crypto-Turtle Bot",
                    style={'color': THEME['accent'], 'fontSize': '22px', 'fontWeight': '700', 'marginBottom': '30px',
                           'textAlign': 'center'}),
            html.Label("WYBIERZ SESJĘ:",
                       style={'fontSize': '12px', 'color': THEME['subtext'], 'fontWeight': '600', 'marginBottom': '8px',
                              'display': 'block'}),
            dcc.Dropdown(id='session-selector', options=[], placeholder="Wybierz folder...", className='dash-dropdown',
                         style={'marginBottom': '20px'}),
        ]),
        html.Div(id='session-info', style={
            'marginTop': 'auto', 'backgroundColor': THEME['card'], 'padding': '15px',
            'borderRadius': '12px', 'border': f'1px solid {THEME["border"]}', 'boxShadow': '0 4px 15px rgba(0,0,0,0.2)'
        })
    ]),

    # Main Panel Wrapper - Zwiększony marginLeft wymusza przerwę od paska bocznego
    html.Div(style={'marginLeft': '350px', 'display': 'flex', 'justifyContent': 'center'}, children=[

        # Wewnętrzny kontener
        html.Div(style={'width': '100%', 'maxWidth': '1600px', 'padding': '40px 40px'}, children=[

            dcc.Interval(id='interval-component', interval=20 * 1000, n_intervals=0),

            # Portfolio Value Chart
            html.Div(style={
                'backgroundColor': THEME['card'], 'padding': '20px', 'borderRadius': '12px',
                'border': f'1px solid {THEME["border"]}', 'marginBottom': '30px',
            }, children=[
                html.H3("WARTOŚĆ CAŁEGO PORTFELA (USD)",
                        style={'marginTop': '0', 'fontSize': '15px', 'fontWeight': '600', 'color': THEME['text'],
                               'textAlign': 'center'}),
                dcc.Graph(id='portfolio-value-graph', config={'displayModeBar': False})
            ]),

            # Symbol Grid
            html.Div(id='symbol-grid', style={
                'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fill, minmax(500px, 1fr))',
                'gap': '25px', 'marginBottom': '30px'
            }),

            # --- SEKCJA: Ostatnie Zlecenia ---
            html.Div(style={
                'backgroundColor': THEME['card'], 'padding': '20px', 'borderRadius': '12px',
                'border': f'1px solid {THEME["border"]}'
            }, children=[
                html.H3("OSTATNIE ZLECENIA",
                        style={'marginTop': '0', 'fontSize': '15px', 'fontWeight': '600', 'color': THEME['text'],
                               'marginBottom': '15px', 'textAlign': 'center'}),
                html.Div(id='recent-orders-container')
            ])

        ])
    ])
])


# --- LOGIKA ---

@app.callback(
    [Output('session-selector', 'options'),
     Output('session-selector', 'value')],
    [Input('interval-component', 'n_intervals')],
    [State('session-selector', 'value')]
)
def update_folder_list(n, current_session):
    if not os.path.exists(REALTIME_DIR):
        return [], dash.no_update

    folders = [d for d in os.listdir(REALTIME_DIR) if os.path.isdir(os.path.join(REALTIME_DIR, d))]
    folders.sort(reverse=True)
    options = [{'label': f, 'value': f} for f in folders]

    if not folders:
        return [], dash.no_update

    if current_session is None or current_session not in folders:
        return options, folders[0]

    return options, dash.no_update


@app.callback(
    [Output('portfolio-value-graph', 'figure'),
     Output('symbol-grid', 'children'),
     Output('session-info', 'children'),
     Output('recent-orders-container', 'children')],
    [Input('session-selector', 'value'),
     Input('interval-component', 'n_intervals')]
)
def update_dashboard(selected_session, n):
    if not selected_session:
        return go.Figure(), [], html.Div(html.P("Oczekuję na wybór sesji...",
                                                style={'textAlign': 'center', 'color': THEME['subtext'],
                                                       'marginTop': '20px'})), []

    csv_path = os.path.join(REALTIME_DIR, selected_session, "realtime_orders_logs.csv")
    if not os.path.exists(csv_path):
        return go.Figure(), [], html.P(f"Brak pliku CSV w sesji: {selected_session}"), []

    try:
        df = pd.read_csv(csv_path)
        df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp_dt')

        accepted_df = df[df['status'].str.lower() == 'accepted']
        total_accepted = len(accepted_df)
        total_buys = len(accepted_df[accepted_df['action'].str.lower() == 'buy'])
        total_sells = len(accepted_df[accepted_df['action'].str.lower() == 'sell'])
        unique_timestamps = df['timestamp_dt'].nunique()

        # 1. Wykres Portfela
        pdf = df.groupby('timestamp_dt').agg({
            'total_portfolio_value': 'last',
            'actual_usdt': 'last'
        }).reset_index()

        val_min = pdf['total_portfolio_value'].min()
        val_max = pdf['total_portfolio_value'].max()
        margin = (val_max - val_min) * 0.05 if val_max != val_min else 1.0

        start_value = pdf['total_portfolio_value'].iloc[0]
        current_value = pdf['total_portfolio_value'].iloc[-1]
        portfolio_color = THEME['accent'] if current_value >= start_value else THEME['danger']

        max_idx = pdf['total_portfolio_value'].idxmax()
        min_idx = pdf['total_portfolio_value'].idxmin()
        max_portfolio_val = pdf['total_portfolio_value'].loc[max_idx]
        max_portfolio_time = pdf['timestamp_dt'].loc[max_idx].strftime('%d/%m/%Y_%H:%M')
        min_portfolio_val = pdf['total_portfolio_value'].loc[min_idx]
        min_portfolio_time = pdf['timestamp_dt'].loc[min_idx].strftime('%d/%m/%Y_%H:%M')

        fig_portfolio = go.Figure()
        fig_portfolio.add_trace(go.Scatter(
            x=pdf['timestamp_dt'], y=pdf['total_portfolio_value'],
            mode='lines+markers', line=dict(color=portfolio_color, width=2.5), marker=dict(size=4),
            fill='tozeroy',
            fillcolor=f"rgba({0 if current_value >= start_value else 255}, {230 if current_value >= start_value else 82}, {118 if current_value >= start_value else 82}, 0.05)",
            hovertemplate='%{y:.2f} USD<extra></extra>'
        ))
        fig_portfolio.update_layout(
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=40, r=10, t=10, b=40), height=350, font=dict(family="Inter", color=THEME['text']),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(gridcolor=THEME['border'], zeroline=False, range=[val_min - margin, val_max + margin])
        )

        # 2. Grid symboli
        symbols = df['symbol'].unique()
        grid_items = []
        for sym in symbols:
            sdf = df[df['symbol'] == sym]
            s_min, s_max = sdf['price'].min(), sdf['price'].max()
            s_margin = (s_max - s_min) * 0.05 if s_max != s_min else 1.0

            marker_colors = []
            marker_line_colors = []

            for action, status in zip(sdf['action'], sdf['status']):
                act = str(action).lower()
                stat = str(status).lower()

                if stat == 'accepted' and act == 'buy':
                    marker_colors.append(THEME['accent'])
                    marker_line_colors.append(THEME['bg'])
                elif stat == 'accepted' and act == 'sell':
                    marker_colors.append(THEME['danger'])
                    marker_line_colors.append(THEME['bg'])
                else:
                    marker_colors.append('rgba(0,0,0,0)')
                    marker_line_colors.append('rgba(0,0,0,0)')

            fig_sym = go.Figure()

            fig_sym.add_trace(go.Scatter(
                x=sdf['timestamp_dt'], y=sdf['price'],
                name="Cena",
                mode='lines',
                line=dict(color=THEME['orange'], width=2, dash='dot')
            ))

            fig_sym.add_trace(go.Scatter(
                x=sdf['timestamp_dt'], y=sdf['actual_qty'],
                name="Ilość",
                mode='lines+markers',
                line=dict(color=THEME['blue'], width=2),
                marker=dict(color=marker_colors, size=7, line=dict(width=1, color=marker_line_colors)),
                yaxis="y2"
            ))

            fig_sym.update_layout(
                title=dict(text=f"<b>{sym.upper()}</b>", font=dict(size=15, color=THEME['text']), x=0.5,
                           xanchor='center'),
                template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=40, r=40, t=40, b=10),
                height=320,
                font=dict(family="Inter"),
                showlegend=True,
                legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5),
                xaxis=dict(showgrid=False),
                yaxis=dict(title="Cena", gridcolor=THEME['border'], range=[s_min - s_margin, s_max + s_margin]),
                yaxis2=dict(title="Posiadana Ilość", overlaying='y', side='right', showgrid=False, rangemode='tozero')
            )
            grid_items.append(html.Div(
                style={'backgroundColor': THEME['card'], 'padding': '15px', 'borderRadius': '12px',
                       'border': f'1px solid {THEME["border"]}'},
                children=[dcc.Graph(figure=fig_sym, config={'displayModeBar': False})]))

        # Box USDT & PORTFOLIO na końcu
        fig_usdt = go.Figure()

        usdt_min = min(pdf['actual_usdt'].min(), pdf['total_portfolio_value'].min())
        usdt_max = max(pdf['actual_usdt'].max(), pdf['total_portfolio_value'].max())
        usdt_margin = (usdt_max - usdt_min) * 0.05 if usdt_max != usdt_min else 1.0

        fig_usdt.add_trace(go.Scatter(
            x=pdf['timestamp_dt'], y=pdf['total_portfolio_value'],
            name="Portfolio", mode='lines', line=dict(color='#888888', width=2)
        ))

        fig_usdt.add_trace(go.Scatter(
            x=pdf['timestamp_dt'], y=pdf['actual_usdt'],
            name="Wolne USDT", mode='lines', line=dict(color='#2e7d32', width=2)
        ))

        fig_usdt.update_layout(
            title=dict(text=f"<b>USDT & PORTFOLIO</b>", font=dict(size=15, color=THEME['text']), x=0.5,
                       xanchor='center'),
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=40, r=40, t=40, b=10),
            height=320,
            font=dict(family="Inter"),
            showlegend=True,
            legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5),
            xaxis=dict(showgrid=False),
            yaxis=dict(title="USD", gridcolor=THEME['border'], range=[usdt_min - usdt_margin, usdt_max + usdt_margin])
        )

        grid_items.append(html.Div(style={
            'backgroundColor': THEME['card'], 'padding': '15px', 'borderRadius': '12px',
            'border': f'1px solid {THEME["border"]}'
        }, children=[dcc.Graph(figure=fig_usdt, config={'displayModeBar': False})]))

        # 3. Info Panel
        last_row = df.iloc[-1]
        holdings_list = []
        for i, sym in enumerate(symbols):
            sdf = df[df['symbol'] == sym]
            sym_accepted_df = sdf[sdf['status'].str.lower() == 'accepted']
            sym_buys = len(sym_accepted_df[sym_accepted_df['action'].str.lower() == 'buy'])
            sym_sells = len(sym_accepted_df[sym_accepted_df['action'].str.lower() == 'sell'])
            total_sym_tx = sym_buys + sym_sells

            last_sym_row = sdf.iloc[-1]
            h_qty = last_sym_row['actual_qty']
            h_price = last_sym_row['price']
            h_value_usd = h_qty * h_price

            item_border = f'1px solid {THEME["border"]}' if i < len(symbols) - 1 else 'none'

            holdings_list.append(
                html.Div(style={'borderBottom': item_border, 'paddingBottom': '10px', 'marginBottom': '10px'},
                         children=[
                             html.Div([
                                 html.Span(f"{sym.upper()}",
                                           style={'color': THEME['text'], 'fontWeight': '600', 'fontSize': '14px'}),
                                 html.Div([
                                     html.Span(f"{h_qty:.4f}", style={'fontWeight': '600', 'color': THEME['text']}),
                                     html.Span(f" (${h_value_usd:.2f})",
                                               style={'fontSize': '12px', 'color': THEME['subtext'],
                                                      'marginLeft': '6px'})
                                 ])
                             ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center',
                                       'marginBottom': '6px'}),

                             html.Div([
                                 html.Span(f"Zlecenia: {total_sym_tx}"),
                                 html.Div([
                                     html.Span(f"B: {sym_buys}", style={'color': THEME['accent'], 'marginRight': '8px',
                                                                        'fontWeight': '500'}),
                                     html.Span(f"S: {sym_sells}", style={'color': THEME['danger'], 'fontWeight': '500'})
                                 ])
                             ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center',
                                       'fontSize': '11px', 'color': THEME['subtext']})
                         ])
            )

        info_content = [
            html.Div("● NA ŻYWO",
                     style={'color': THEME['accent'], 'fontWeight': '700', 'fontSize': '13px', 'textAlign': 'center',
                            'marginBottom': '15px', 'letterSpacing': '1px'}),

            html.Div(style={'backgroundColor': THEME['inner_card'], 'padding': '15px', 'borderRadius': '8px',
                            'marginBottom': '15px'}, children=[
                html.Div([html.Div("Wartość Portfela:",
                                   style={'color': THEME['subtext'], 'fontSize': '12px', 'marginBottom': '2px'}),
                          html.Div(f"${current_value:.2f}",
                                   style={'fontSize': '22px', 'fontWeight': '700', 'color': portfolio_color})],
                         style={'marginBottom': '15px'}),
                html.Div([html.Div("Wolne USDT:",
                                   style={'color': THEME['subtext'], 'fontSize': '12px', 'marginBottom': '2px'}),
                          html.Div(f"${last_row['actual_usdt']:.2f}",
                                   style={'fontSize': '18px', 'fontWeight': '600', 'color': THEME['text']})],
                         style={'marginBottom': '15px'}),

                html.Div(style={'borderTop': f'1px dashed {THEME["border"]}', 'paddingTop': '12px', 'display': 'grid',
                                'gridTemplateColumns': '1fr 1fr', 'gap': '8px', 'fontSize': '12px',
                                'color': THEME['subtext']}, children=[
                    html.Div([html.Span("Zlecenia (Udane): "),
                              html.Span(f"{total_accepted}", style={'color': THEME['text'], 'fontWeight': '600'})]),
                    html.Div([html.Span("Cykle (Czas): "),
                              html.Span(f"{unique_timestamps}", style={'color': THEME['text'], 'fontWeight': '600'})]),
                    html.Div([html.Span("Suma BUY: "),
                              html.Span(f"{total_buys}", style={'color': THEME['accent'], 'fontWeight': '600'})]),
                    html.Div([html.Span("Suma SELL: "),
                              html.Span(f"{total_sells}", style={'color': THEME['danger'], 'fontWeight': '600'})])
                ]),

                html.Div(style={'borderTop': f'1px dashed {THEME["border"]}', 'paddingTop': '12px', 'marginTop': '12px',
                                'display': 'flex', 'flexDirection': 'column', 'gap': '6px', 'fontSize': '12px'},
                         children=[
                             html.Div([
                                 html.Span("Max Portfolio: ", style={'color': THEME['subtext']}),
                                 html.Span(f"${max_portfolio_val:.2f} ({max_portfolio_time})",
                                           style={'color': THEME['accent'], 'fontWeight': '600'})
                             ], style={'display': 'flex', 'justifyContent': 'space-between'}),
                             html.Div([
                                 html.Span("Min Portfolio: ", style={'color': THEME['subtext']}),
                                 html.Span(f"${min_portfolio_val:.2f} ({min_portfolio_time})",
                                           style={'color': THEME['danger'], 'fontWeight': '600'})
                             ], style={'display': 'flex', 'justifyContent': 'space-between'})
                         ])
            ]),

            html.Div(style={'backgroundColor': THEME['inner_card'], 'padding': '15px', 'borderRadius': '8px',
                            'marginBottom': '15px'}, children=[
                html.Div("POSIADANE KRYPTOWALUTY",
                         style={'fontSize': '11px', 'color': THEME['text'], 'fontWeight': '700',
                                'letterSpacing': '0.5px', 'marginBottom': '15px'}),
                html.Div(holdings_list)
            ]),

            html.Div([html.Span("Ostatnia aktualizacja: ", style={'color': THEME['subtext'], 'fontSize': '12px'}),
                      html.Span(last_row['timestamp_dt'].strftime('%H:%M:%S'),
                                style={'fontSize': '12px', 'fontWeight': '600', 'color': THEME['text']})],
                     style={'textAlign': 'center', 'marginTop': '10px'})
        ]

        # 4. TABELA OSTATNICH ZLECEŃ
        df_orders = df.sort_values('timestamp_dt', ascending=False)
        table_rows = []

        for _, row in df_orders.iterrows():
            action_raw = str(row.get('action', '')).lower()
            status_raw = str(row.get('status', '')).lower()
            reason_val = row.get('reason', '')
            reason_raw = '' if pd.isna(reason_val) else str(reason_val).lower()

            if status_raw != 'accepted':
                action_color = THEME['orange']
            elif action_raw == 'buy':
                action_color = THEME['accent']
            elif action_raw == 'sell':
                action_color = THEME['danger']
            else:
                action_color = THEME['orange']

            status_color = THEME['subtext'] if status_raw == 'accepted' else THEME['orange']

            buy_p = f"{row.get('buy_p', 0.0):.2f}"
            hold_p = f"{row.get('hold_p', 0.0):.2f}"
            sell_p = f"{row.get('sell_p', 0.0):.2f}"

            table_rows.append(html.Tr([
                html.Td(row['timestamp_dt'].strftime('%Y-%m-%d %H:%M:%S')),
                html.Td(str(row.get('symbol', '')).upper(), style={'fontWeight': '600'}),
                html.Td(html.Span([
                    html.Span(f"B: {buy_p} ", style={'color': THEME['accent']}),
                    html.Span(f"| H: {hold_p} ", style={'color': THEME['orange']}),
                    html.Span(f"| S: {sell_p}", style={'color': THEME['danger']})
                ])),
                html.Td(f"{row.get('qty', 0.0):.4f}"),
                html.Td(action_raw.upper(), style={'color': action_color, 'fontWeight': '700'}),
                html.Td(status_raw.upper(), style={'color': status_color, 'fontWeight': '600'}),
                html.Td(reason_raw.upper(), style={'color': status_color, 'fontWeight': '600'})
            ]))

        recent_orders_table = html.Div(style={
            'maxHeight': '450px', 'overflowY': 'auto',
            'borderRadius': '8px', 'border': f'1px solid {THEME["border"]}'
        }, children=[
            html.Table(className='orders-table', children=[
                html.Thead(html.Tr([
                    html.Th("Czas"), html.Th("Symbol"), html.Th("Predykcje modelu"),
                    html.Th("Qty"), html.Th("Akcja"), html.Th("Status"), html.Th("Reason")
                ])),
                html.Tbody(table_rows)
            ])
        ])

        return fig_portfolio, grid_items, info_content, recent_orders_table

    except Exception as e:
        return go.Figure(), [], html.Div(f"Błąd przetwarzania: {str(e)}",
                                         style={'color': THEME['danger'], 'padding': '10px'}), []


if __name__ == '__main__':
    app.run(debug=False, port=8051)