"""Dashboard Dash: podgląd na żywo treningu RL (actual_training_logs)."""

import glob
import dash
import time
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import json
import os

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
LOG_DIR = os.path.join(base_dir, "shared", "agents", "actual_training_logs")
os.makedirs(LOG_DIR, exist_ok=True)
UPDATE_INTERVAL_MS = 4000
TARGET_EPOCHS = 100

app = dash.Dash(__name__)
app.title = "RL Training Viewer"

# Oryginalne kolory (używane do linii wykresów i trybu ciemnego)
COLORS = {
    'bg': '#0B0F19', 'card': '#151A23', 'text': '#E2E8F0', 'accent': '#00E5FF',
    'success': '#00FFA3', 'warning': '#FFD166', 'danger': '#FF5252'
}

# Definicje motywów (Dark / Light)
THEMES = {
    'dark': {
        'bg': '#0B0F19',
        'card': '#1C2331',
        'text': '#E2E8F0',
        'subtext': '#94A3B8',
        'template': 'plotly_dark',
        'grid_color': '#2D3748',
        'btn_bg': '#2A3441',
        'btn_border': '#3E4C5E',
        'shadow': '0 8px 32px 0 rgba(0, 0, 0, 0.37)'
    },
    'light': {
        'bg': '#F8FAFC',
        'card': '#FFFFFF',
        'text': '#0F172A',
        'subtext': '#64748B',
        'template': 'plotly_white',
        'grid_color': '#E2E8F0',
        'btn_bg': '#F1F5F9',
        'btn_border': '#CBD5E1',
        'shadow': '0 8px 32px 0 rgba(31, 38, 135, 0.07)'
    }
}

# Style bazowe
base_btn_style = {
    'padding': '8px 12px', 'margin': '5px', 'borderRadius': '6px',
    'cursor': 'pointer', 'fontSize': '11px', 'fontFamily': '"Inter", "Roboto", sans-serif',
    'fontWeight': '600', 'textTransform': 'uppercase', 'letterSpacing': '0.5px',
    'transition': 'all 0.2s ease',
    'flex': '1 1 auto', 'textAlign': 'center'
}

base_card_style = {
    'borderRadius': '12px', 'padding': '15px',
    'margin': '0', 'width': '30%',
    'border': '1px solid rgba(255, 255, 255, 0.05)',
    'backdropFilter': 'blur(10px)',
    'boxSizing': 'border-box'
}


# --- FUNKCJE DANYCH ---
def get_full_training_data():
    path = os.path.join(LOG_DIR, "full_training.json")
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return None


def get_process_data():
    files = glob.glob(os.path.join(LOG_DIR, "proc_*.json"))
    try:
        files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    except:
        files.sort()
    data = []
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                data.append(json.load(fp))
        except:
            pass
    return data


# --- LAYOUT SEKCJI ---

global_section = html.Div([
    html.Div([
        # LEWY PANEL (Global 1)
        html.Div(id='card-g1', children=[
            html.Div([
                html.Button("📈 Reward", id='btn-g1-rew', n_clicks=0),
                html.Button("💰 Port.", id='btn-g1-port', n_clicks=0),
                html.Button("Δ Excess", id='btn-g1-excess', n_clicks=0),
                html.Button("🌊 Volat.", id='btn-g1-heat', n_clicks=0),
                html.Button("📊 Trans.", id='btn-g1-hist', n_clicks=0),
                html.Button("📉 Range", id='btn-g1-none', n_clicks=0),
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center', 'marginBottom': '10px', 'gap': '5px'}),
            dcc.Graph(id='global-graph-1', style={'height': '400px'}, config={'displayModeBar': False}),
            dcc.Store(id='store-g1', data='reward')
        ], style={**base_card_style, 'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

        # PRAWY PANEL (Global 2)
        html.Div(id='card-g2', children=[
            html.Div([
                html.Button("💰 Port.", id='btn-g2-port', n_clicks=0),
                html.Button("📈 Reward", id='btn-g2-rew', n_clicks=0),
                html.Button("Δ Excess", id='btn-g2-excess', n_clicks=0),
                html.Button("🌊 Volat.", id='btn-g2-heat', n_clicks=0),
                html.Button("📊 Trans.", id='btn-g2-hist', n_clicks=0),
                html.Button("📉 Range", id='btn-g2-none', n_clicks=0),
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center', 'marginBottom': '10px', 'gap': '5px'}),
            dcc.Graph(id='global-graph-2', style={'height': '400px'}, config={'displayModeBar': False}),
            dcc.Store(id='store-g2', data='portfolio')
        ], style={**base_card_style, 'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
    ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-between', 'gap': '20px', 'marginBottom': '30px'}),

    # STATYSTYKI NA ŻYWO (Summary Row) + panel INFO pod kafelkami
    html.Div([
        html.Div(id='live-stats-row', style={'display': 'flex', 'justifyContent': 'space-between', 'gap': '15px', 'flexWrap': 'wrap'}),
        html.Div(id='live-info-panel', style={'marginTop': '16px'}),
    ], style={'margin': '0 15px 30px 15px'}),

    # PASEK POSTĘPU
    html.Div([
        html.Div([
            html.Span("POSTĘP TRENINGU", id='progress-label', style={'color': '#888', 'fontSize': '12px', 'letterSpacing': '1px', 'fontWeight': 'bold'}),
            html.Span(id='progress-text', style={'color': COLORS['accent'], 'fontWeight': '900', 'marginLeft': '10px', 'fontSize': '14px'})
        ], style={'marginBottom': '10px', 'display': 'flex', 'alignItems': 'center'}),
        html.Div([
            html.Div(id='progress-bar', style={'width': '0%', 'height': '100%', 'backgroundColor': COLORS['success'],
                                               'transition': 'width 0.8s ease-in-out',
                                               'boxShadow': f"0 0 10px {COLORS['success']}"})
        ], id='progress-bg', style={'width': '100%', 'height': '8px', 'backgroundColor': '#2D3748', 'borderRadius': '4px',
                                    'overflow': 'hidden'})
    ], style={'margin': '30px 0', 'padding': '20px', 'backgroundColor': 'rgba(255,255,255,0.02)', 'borderRadius': '12px', 'border': '1px solid rgba(255,255,255,0.05)'}),

    html.Hr(id='main-hr', style={'display': 'none'})
])

# --- GŁÓWNY LAYOUT APLIKACJI ---
app.layout = html.Div(
    id='main-layout',
    style={'backgroundColor': COLORS['bg'], 'color': COLORS['text'], 'fontFamily': '"Inter", "Roboto", sans-serif',
           'minHeight': '100vh', 'padding': '20px', 'margin': '-8px', 'transition': 'background-color 0.5s ease'},
    children=[
        # Store dla Motywu
        dcc.Store(id='theme-store', data='dark'),

        # HEADER
        html.Div([
            html.Div([
                html.H1(id='main-title', children="Monitor Treningu Agenta RL",
                        style={'margin': 0, 'fontSize': '32px', 'fontWeight': '900', 'letterSpacing': '-0.5px'}),
                html.P("Live Performance & Activity Dashboard", style={'margin': '5px 0 0 0', 'color': COLORS['accent'], 'fontSize': '14px', 'fontWeight': '500'})
            ]),
            html.Button("☀ / 🌙", id='btn-theme', n_clicks=0, style={
                'padding': '10px 15px', 'cursor': 'pointer', 'borderRadius': '8px', 'border': '1px solid rgba(255,255,255,0.1)',
                'backgroundColor': 'rgba(255,255,255,0.05)', 'color': 'inherit', 'fontWeight': 'bold',
                'transition': 'all 0.3s'
            }),
        ], id='header-container', style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'marginBottom': '30px', 'paddingBottom': '20px', 'borderBottom': '1px solid rgba(255,255,255,0.1)'}),

        dcc.Store(id='activity-store', data={'last_epoch': -1, 'last_timestamp': time.time()}),

        global_section,

        # GRID PROCESÓW
        html.Div([
            html.Div([
                html.Button("🔄 AUTO", id='btn-auto', n_clicks=0),
                html.Button("🥧 Decyzje", id='btn-pie', n_clicks=0),
                html.Button("💰 Portfel", id='btn-port', n_clicks=0),
                html.Button("📈 Nagrody", id='btn-rew', n_clicks=0),
            ], style={'textAlign': 'center', 'marginBottom': '20px'}),

            dcc.Store(id='view-mode-store', data='auto'),

            html.Div([
                html.Span(id='global-title',
                          style={'fontSize': '20px', 'color': '#00d4ff', 'fontWeight': 'bold', 'marginRight': '20px'}),
                html.Span(id='process-count', style={'color': '#888', 'fontSize': '16px'})
            ], style={'textAlign': 'center', 'marginBottom': '20px'}),

            html.Div(id='graphs-container', style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center', 'gap': '20px'}),
        ]),

        dcc.Interval(id='interval-component', interval=UPDATE_INTERVAL_MS, n_intervals=0)
    ]
)


# --- GENEROWANIE WYKRESÓW GLOBALNYCH ---
def generate_global_fig(mode, full_data, theme_name='dark'):
    """Generuje wykresy globalne z obsługą motywu."""
    fig = go.Figure()

    thm = THEMES[theme_name]

    # Style layoutu (zależne od motywu)
    layout_args = dict(
        template=thm['template'],  # plotly_dark lub plotly_white
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=20, t=50, b=30),
        font=dict(color=thm['text'], family='"Inter", sans-serif'),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=thm['card'], font_size=13, font_family='"Inter", sans-serif')
    )
    title_defaults = dict(x=0.5, xanchor='center', font=dict(size=14, color=thm['text'], family='"Inter", sans-serif'))

    if not full_data or not full_data.get('history'):
        fig.update_layout(title=dict(text="Oczekiwanie na dane...", **title_defaults), **layout_args)
        return fig

    hist = full_data['history']
    config = full_data.get('config', {})
    epochs = list(range(1, len(hist) + 1))
    max_x = config.get('global_epochs', 100)

    # --- PONIŻEJ LOGIKA WYKRESÓW JEST IDENTYCZNA JAK W ORYGINALE ---

    if mode == 'reward':
        avg = [h['reward']['avg_end_r'] for h in hist]
        mn = [h['reward']['min_end_r'] for h in hist]
        mx = [h['reward']['max_end_r'] for h in hist]

        fig.add_trace(go.Scatter(
            x=epochs + epochs[::-1], y=mx + mn[::-1],
            fill='toself', fillcolor='rgba(255, 145, 0, 0.15)',
            line=dict(color='rgba(0,0,0,0)'), name='Range'
        ))
        fig.add_trace(go.Scatter(
            x=epochs, y=avg, mode='lines+markers',
            # Marker dostosowany lekko do kontrastu w trybie jasnym, ale logika ta sama
            marker=dict(size=[0] * (len(avg) - 1) + [6], color=thm['text']),
            line=dict(color=COLORS['warning'], width=2), name='Avg End Reward'
        ))
        fig.update_layout(
            title=dict(text="Wykres średniej końcowej wartości nagrody", **title_defaults),
            xaxis_title="Global Epoch", yaxis_title="Reward", **layout_args,
            xaxis=dict(range=[1, max_x], gridcolor=thm['grid_color']),
            yaxis=dict(gridcolor=thm['grid_color']),
            legend=dict(orientation="h", yanchor="top", y=-0.3, xanchor="center", x=0.5, bgcolor="rgba(0,0,0,0)",
                        font=dict(size=15))
        )

    elif mode == 'portfolio':
        avg = [h['portfolio']['avg_end_p'] for h in hist]
        mn = [h['portfolio']['min_end_p'] for h in hist]
        mx = [h['portfolio']['max_end_p'] for h in hist]
        # końcowy HODL z % (start 1000$ jak w treningu) — już w dumpach, bez restartu
        start_bal = float(config.get('balance_start') or 1000.0)
        hodl_end = []
        for h in hist:
            hp = h.get('avg_hodl_pct')
            if hp is None:
                hodl_end.append(None)
            else:
                hodl_end.append(start_bal * (1.0 + float(hp) / 100.0))

        fig.add_trace(go.Scatter(
            x=epochs + epochs[::-1], y=mx + mn[::-1],
            fill='toself', fillcolor='rgba(0, 212, 255, 0.15)',
            line=dict(color='rgba(0,0,0,0)'), name='Range'
        ))
        fig.add_trace(go.Scatter(
            x=epochs, y=avg, mode='lines+markers',
            marker=dict(size=[0] * (len(avg) - 1) + [6], color=thm['text']),
            line=dict(color=COLORS['accent'], width=2), name='Avg End Portfolio'
        ))
        if any(v is not None for v in hodl_end):
            fig.add_trace(go.Scatter(
                x=epochs, y=hodl_end, mode='lines+markers',
                marker=dict(size=[0] * (len(hodl_end) - 1) + [6], color=COLORS['warning']),
                line=dict(color=COLORS['warning'], width=2, dash='dash'),
                name='Avg End HODL',
                connectgaps=False,
            ))
        fig.update_layout(
            title=dict(text="Wykres średniej końcowej wartości portfela", **title_defaults),
            xaxis_title="Global Epoch", yaxis_title="Value ($)", **layout_args,
            xaxis=dict(range=[1, max_x], gridcolor=thm['grid_color']),
            yaxis=dict(gridcolor=thm['grid_color']),
            legend=dict(orientation="h", yanchor="top", y=-0.3, xanchor="center", x=0.5, bgcolor="rgba(0,0,0,0)",
                        font=dict(size=15))
        )

    elif mode == 'excess':
        excess = [h.get('avg_excess_pct') for h in hist]
        strat = [h.get('avg_strat_pct') for h in hist]
        hodl = [h.get('avg_hodl_pct') for h in hist]
        has_excess = any(v is not None for v in excess)
        has_strat = any(v is not None for v in strat)
        has_hodl = any(v is not None for v in hodl)

        if has_excess:
            fig.add_trace(go.Scatter(
                x=epochs + epochs[::-1],
                y=[max(float(v or 0), 0) for v in excess] + [0] * len(epochs),
                fill='toself', fillcolor='rgba(0, 230, 118, 0.12)',
                line=dict(color='rgba(0,0,0,0)'), name='Above HODL', showlegend=False,
                hoverinfo='skip',
            ))
            fig.add_trace(go.Scatter(
                x=epochs + epochs[::-1],
                y=[min(float(v or 0), 0) for v in excess] + [0] * len(epochs),
                fill='toself', fillcolor='rgba(255, 23, 68, 0.12)',
                line=dict(color='rgba(0,0,0,0)'), name='Below HODL', showlegend=False,
                hoverinfo='skip',
            ))
            fig.add_trace(go.Scatter(
                x=epochs, y=excess, mode='lines+markers',
                marker=dict(size=[0] * (len(excess) - 1) + [6], color=thm['text']),
                line=dict(color=COLORS['success'], width=2.5),
                name='Avg Excess %', connectgaps=False,
            ))
        if has_strat:
            fig.add_trace(go.Scatter(
                x=epochs, y=strat, mode='lines',
                line=dict(color=COLORS['accent'], width=1.5),
                name='Avg Strat %', connectgaps=False,
            ))
        if has_hodl:
            fig.add_trace(go.Scatter(
                x=epochs, y=hodl, mode='lines',
                line=dict(color=COLORS['warning'], width=1.5, dash='dash'),
                name='Avg HODL %', connectgaps=False,
            ))
        fig.add_hline(y=0, line_width=1, line_dash='dot', line_color=thm['subtext'], opacity=0.6)
        fig.update_layout(
            title=dict(text="Excess vs HODL (średnie % na dump)", **title_defaults),
            xaxis_title="Global Epoch", yaxis_title="Return / Excess (%)", **layout_args,
            xaxis=dict(range=[1, max_x], gridcolor=thm['grid_color']),
            yaxis=dict(gridcolor=thm['grid_color'], zeroline=True, zerolinecolor=thm['subtext']),
            legend=dict(orientation="h", yanchor="top", y=-0.3, xanchor="center", x=0.5, bgcolor="rgba(0,0,0,0)",
                        font=dict(size=15))
        )

    elif mode == 'heatmap':
        max_max = [h['portfolio']['max_max_p'] for h in hist]
        min_min = [h['portfolio']['min_min_p'] for h in hist]
        max_mean = [h['portfolio']['max_mean_p'] for h in hist]
        min_mean = [h['portfolio']['min_mean_p'] for h in hist]

        fig.add_trace(go.Scatter(
            x=epochs + epochs[::-1], y=max_max + min_min[::-1],
            fill='toself', fillcolor='rgba(0, 89, 255, 0.2)',
            line=dict(color='rgba(0,0,0,0)'), name='Global Extr.'
        ))
        fig.add_trace(go.Scatter(
            x=epochs + epochs[::-1], y=max_mean + min_mean[::-1],
            fill='toself', fillcolor='rgba(0, 212, 255, 0.4)',
            line=dict(color='rgba(0,0,0,0)'), name='Mean Range'
        ))
        fig.update_layout(
            title=dict(text="Wykres zmienności portfela", **title_defaults),
            xaxis_title="Global Epoch", yaxis_title="Value ($)", **layout_args,
            xaxis=dict(range=[1, max_x], gridcolor=thm['grid_color']),
            yaxis=dict(gridcolor=thm['grid_color']),
            legend=dict(orientation="h", yanchor="top", y=-0.3, xanchor="center", x=0.5, bgcolor="rgba(0,0,0,0)",
                        font=dict(size=15))
        )

    elif mode == 'histogram':
        buys, sells = [], []
        for h in hist:
            tot = h['avg_transactions']
            b, s = h['avg_buys'], h['avg_sells']
            if (b + s) > tot and (b + s) > 0:
                diff = (b + s) - tot
                b -= diff * (b / (b + s))
                s -= diff * (s / (b + s))
            buys.append(b)
            sells.append(s)

        fig.add_trace(go.Bar(x=epochs, y=buys, name='Buys', marker_color=COLORS['success']))
        fig.add_trace(go.Bar(x=epochs, y=sells, name='Sells', marker_color=COLORS['danger']))

        tx_peak = max((b + s for b, s in zip(buys, sells)), default=0)
        y_max = max(tx_peak * 1.15, 10)
        fig.update_layout(
            title=dict(text="Wykres wykonanych transakcji", **title_defaults),
            barmode='stack', yaxis_range=[0, y_max], yaxis_title="Transactions", xaxis_title="Global Epoch",
            xaxis=dict(range=[1, max_x], gridcolor=thm['grid_color']),
            yaxis=dict(gridcolor=thm['grid_color']),
            **layout_args,
            legend=dict(orientation="h", yanchor="top", y=-0.3, xanchor="center", x=0.5, bgcolor="rgba(0,0,0,0)",
                        font=dict(size=15))
        )

    elif mode == 'empty':
        max_max = [h['reward']['max_max_r'] for h in hist]
        min_min = [h['reward']['min_min_r'] for h in hist]
        max_mean = [h['reward']['max_mean_r'] for h in hist]
        min_mean = [h['reward']['min_mean_r'] for h in hist]

        fig.add_trace(go.Scatter(
            x=epochs + epochs[::-1], y=max_max + min_min[::-1],
            fill='toself', fillcolor='rgba(200, 50, 0, 0.2)',
            line=dict(color='rgba(0,0,0,0)'), name='Global Extr.'
        ))
        fig.add_trace(go.Scatter(
            x=epochs + epochs[::-1], y=max_mean + min_mean[::-1],
            fill='toself', fillcolor='rgba(255, 145, 0, 0.4)',
            line=dict(color='rgba(0,0,0,0)'), name='Mean Range'
        ))
        fig.update_layout(
            title=dict(text="Wykres zmienności nagrody", **title_defaults),
            xaxis_title="Global Epoch", yaxis_title="Reward", **layout_args,
            xaxis=dict(range=[1, max_x], gridcolor=thm['grid_color']),
            yaxis=dict(gridcolor=thm['grid_color']),
            legend=dict(orientation="h", yanchor="top", y=-0.3, xanchor="center", x=0.5, bgcolor="rgba(0,0,0,0)",
                        font=dict(size=15))
        )

    return fig


# --- CALLBACKS ---

# 1. Przełączanie Motywu (Zmienia TYLKO dane w Store)
@app.callback(
    Output('theme-store', 'data'),
    [Input('btn-theme', 'n_clicks')],
    [State('theme-store', 'data')]
)
def toggle_theme_data(n_clicks, current_theme):
    if n_clicks > 0:
        return 'light' if current_theme == 'dark' else 'dark'
    return 'dark'


# 2. Aktualizacja Stylów (Wygląd całej strony + przyciski)
@app.callback(
    [Output('main-layout', 'style'),
     Output('main-title', 'style'),
     Output('progress-label', 'style'),
     Output('progress-bg', 'style'),
     Output('main-hr', 'style'),
     Output('card-g1', 'style'),
     Output('card-g2', 'style'),
     Output('process-count', 'style'),
     # Przyciski Global G1
     Output('btn-g1-rew', 'style'), Output('btn-g1-port', 'style'), Output('btn-g1-excess', 'style'),
     Output('btn-g1-heat', 'style'), Output('btn-g1-hist', 'style'), Output('btn-g1-none', 'style'),
     # Przyciski Global G2
     Output('btn-g2-port', 'style'), Output('btn-g2-rew', 'style'), Output('btn-g2-excess', 'style'),
     Output('btn-g2-heat', 'style'), Output('btn-g2-hist', 'style'), Output('btn-g2-none', 'style'),
     # Przyciski Grid
     Output('btn-auto', 'style'), Output('btn-pie', 'style'), Output('btn-port', 'style'), Output('btn-rew', 'style')
     ],
    [Input('theme-store', 'data'),
     Input('store-g1', 'data'), Input('store-g2', 'data'), Input('view-mode-store', 'data')]
)
def apply_theme_visuals(theme_name, g1_mode, g2_mode, grid_mode):
    theme_name = theme_name or 'dark'
    thm = THEMES[theme_name]

    # Style kontenerów
    layout_style = {
        'backgroundColor': thm['bg'], 'color': thm['text'],
        'fontFamily': '"Inter", "Roboto", sans-serif', 'minHeight': '100vh',
        'padding': '20px', 'margin': '-8px', 'transition': 'background-color 0.5s, color 0.5s'
    }
    title_style = {'textAlign': 'center', 'color': thm['text'], 'marginBottom': '15px', 'fontSize': '42px',
                   'fontWeight': '800', 'letterSpacing': '-1px'}
    prog_lbl_style = {'color': thm['subtext'], 'fontSize': '12px', 'letterSpacing': '1px', 'fontWeight': 'bold'}
    prog_bg_style = {'width': '100%', 'height': '8px', 'backgroundColor': thm['grid_color'], 'borderRadius': '4px',
                     'overflow': 'hidden', 'boxShadow': 'inset 0 1px 3px rgba(0,0,0,0.5)'}
    hr_style = {'borderColor': thm['grid_color'], 'marginBottom': '40px', 'opacity': '0.5'}
    proc_cnt_style = {'color': thm['subtext'], 'fontSize': '18px', 'fontWeight': '500'}

    # Style Kart
    card_style = {
        **base_card_style,
        'backgroundColor': thm['card'],
        'color': thm['text'],
        'boxShadow': thm['shadow'],
        'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top',
        'boxSizing': 'border-box', 'minWidth': '400px', 'flex': '1 1 auto'
    }

    # Funkcja pomocnicza do styli przycisków
    def get_btn_style(is_active):
        style = {
            **base_btn_style,
            'backgroundColor': thm['btn_bg'],
            'color': thm['text'],
            'border': f'1px solid {thm["btn_border"]}',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
        }
        
        if is_active:
            style.update({
                'backgroundColor': COLORS['accent'],
                'color': '#0B0F19',
                'fontWeight': 'bold',
                'border': f'1px solid {COLORS["accent"]}',
                'boxShadow': f'0 0 15px {COLORS["accent"]}80, 0 4px 6px rgba(0,0,0,0.2)'
            })
        return style

    # Generowanie styli przycisków
    g1_modes = ['reward', 'portfolio', 'excess', 'heatmap', 'histogram', 'empty']
    g1_styles = [get_btn_style(m == g1_mode) for m in g1_modes]

    g2_modes = ['portfolio', 'reward', 'excess', 'heatmap', 'histogram', 'empty']
    g2_styles = [get_btn_style(m == g2_mode) for m in g2_modes]

    # Grid (auto, 0, 1, 2)
    grid_styles = []
    grid_styles.append(get_btn_style(grid_mode == 'auto'))
    for idx in [0, 1, 2]:
        grid_styles.append(get_btn_style(grid_mode == idx))

    return (
        layout_style, title_style, prog_lbl_style, prog_bg_style, hr_style,
        card_style, card_style, proc_cnt_style,
        *g1_styles, *g2_styles, *grid_styles
    )


# 3. Status aktywności
@app.callback(
    [Output('main-title', 'children'), Output('activity-store', 'data')],
    [Input('interval-component', 'n_intervals')],
    [State('activity-store', 'data')]
)
def update_activity_status(n, activity_data):
    full_data = get_full_training_data()
    hist = full_data.get('history', []) if full_data else []

    last_ep = activity_data.get('last_epoch', -1)
    last_time = activity_data.get('last_timestamp', time.time())
    current_time = time.time()
    current_ep_sum = hist[-1]['epoch_sum'] if hist else 0

    if current_ep_sum > last_ep:
        new_last_time = current_time
        new_last_ep = current_ep_sum
    else:
        new_last_time = last_time
        new_last_ep = last_ep

    is_active = (current_time - new_last_time) < 300
    status_text = "🟢 (Aktywny)" if is_active else "🔴 (Nieaktywny)"
    if not hist: status_text = "🟡 (Oczekiwanie)"

    title = ["Monitor Treningu Agenta RL ", html.Span(status_text, style={'fontSize': '20px', 'color': '#ccc'})]
    return title, {'last_epoch': new_last_ep, 'last_timestamp': new_last_time}


# 4. Wykres Globalny 1
@app.callback(
    [Output('global-graph-1', 'figure'), Output('store-g1', 'data')],
    [Input('interval-component', 'n_intervals'),
     Input('theme-store', 'data'),  # Dodano
     Input('btn-g1-rew', 'n_clicks'), Input('btn-g1-port', 'n_clicks'), Input('btn-g1-excess', 'n_clicks'),
     Input('btn-g1-heat', 'n_clicks'), Input('btn-g1-hist', 'n_clicks'), Input('btn-g1-none', 'n_clicks')],
    [State('store-g1', 'data')]
)
def update_g1(n, theme_name, b1, b2, b2e, b3, b4, b5, current):
    ctx = dash.callback_context
    if ctx.triggered and 'btn' in ctx.triggered[0]['prop_id']:
        bid = ctx.triggered[0]['prop_id']
        if 'rew' in bid:
            current = 'reward'
        elif 'port' in bid:
            current = 'portfolio'
        elif 'excess' in bid:
            current = 'excess'
        elif 'heat' in bid:
            current = 'heatmap'
        elif 'hist' in bid:
            current = 'histogram'
        elif 'none' in bid:
            current = 'empty'

    return generate_global_fig(current, get_full_training_data(), theme_name), current


# 5. Wykres Globalny 2
@app.callback(
    [Output('global-graph-2', 'figure'), Output('store-g2', 'data')],
    [Input('interval-component', 'n_intervals'),
     Input('theme-store', 'data'),  # Dodano
     Input('btn-g2-port', 'n_clicks'), Input('btn-g2-rew', 'n_clicks'), Input('btn-g2-excess', 'n_clicks'),
     Input('btn-g2-heat', 'n_clicks'), Input('btn-g2-hist', 'n_clicks'), Input('btn-g2-none', 'n_clicks')],
    [State('store-g2', 'data')]
)
def update_g2(n, theme_name, b1, b2, b2e, b3, b4, b5, current):
    ctx = dash.callback_context
    if ctx.triggered and 'btn' in ctx.triggered[0]['prop_id']:
        bid = ctx.triggered[0]['prop_id']
        if 'port' in bid:
            current = 'portfolio'
        elif 'rew' in bid:
            current = 'reward'
        elif 'excess' in bid:
            current = 'excess'
        elif 'heat' in bid:
            current = 'heatmap'
        elif 'hist' in bid:
            current = 'histogram'
        elif 'none' in bid:
            current = 'empty'

    return generate_global_fig(current, get_full_training_data(), theme_name), current


# 6. Pasek postępu i Statystyki + panel INFO
@app.callback(
    [Output('progress-bar', 'style'), Output('progress-text', 'children'),
     Output('live-stats-row', 'children'), Output('live-info-panel', 'children')],
    [Input('interval-component', 'n_intervals'), Input('theme-store', 'data')]
)
def update_prog_and_stats(n, theme_name):
    theme_name = theme_name or 'dark'
    thm = THEMES[theme_name]
    full_data = get_full_training_data()
    
    max_eps = full_data.get('config', {}).get('max_episodes', TARGET_EPOCHS) if full_data else TARGET_EPOCHS
    num_cpu = full_data.get('config', {}).get('num_processes', TARGET_EPOCHS) if full_data else TARGET_EPOCHS
    hist = full_data.get('history', []) if full_data else []
    cfg = full_data.get('config', {}) if full_data else {}
    
    current = len(hist) * num_cpu if hist else 0
    pct = min(100, (current / max_eps) * 100) if max_eps > 0 else 0
    
    prog_style = {'width': f'{pct}%', 'height': '100%', 'backgroundColor': COLORS['success'],
                  'transition': 'width 0.8s ease-in-out', 'boxShadow': f"0 0 10px {COLORS['success']}"}
    prog_text = f"{current}/{max_eps} ({pct:.1f}%)"
    
    # --- Live Stats Builder ---
    def stat_card(title, val, color, delta=None, reverse_delta_color=False):
        delta_html = None
        if delta is not None:
            if delta == 0:
                d_color = thm['subtext']
                sign = ""
            else:
                is_positive = delta > 0
                if reverse_delta_color:
                    d_color = COLORS['danger'] if is_positive else COLORS['success']
                else:
                    d_color = COLORS['success'] if is_positive else COLORS['danger']
                sign = "▲ +" if is_positive else "▼ "
                
            delta_html = html.Div(f"{sign}{delta:.2f}", style={'color': d_color, 'fontSize': '13px', 'fontWeight': 'bold', 'marginTop': '8px'})

        return html.Div([
            html.Div(title, style={'color': thm['subtext'], 'fontSize': '12px', 'fontWeight': 'bold', 'textTransform': 'uppercase', 'letterSpacing': '1px', 'marginBottom': '5px'}),
            html.Div(val, style={'color': color, 'fontSize': '28px', 'fontWeight': '900'}),
            delta_html if delta_html else html.Div(style={'height': '21px', 'marginTop': '8px'})
        ], style={'flex': '1', 'minWidth': '140px', 'backgroundColor': thm['card'], 'padding': '20px', 'borderRadius': '12px', 'border': '1px solid rgba(255,255,255,0.05)', 'textAlign': 'center', 'boxShadow': thm['shadow']})

    def chip(label, value, color=None):
        return html.Div([
            html.Span(label, style={'color': thm['subtext'], 'fontSize': '10px', 'fontWeight': '700', 'letterSpacing': '0.6px', 'textTransform': 'uppercase', 'display': 'block', 'marginBottom': '4px'}),
            html.Span(str(value), style={'color': color or thm['text'], 'fontSize': '13px', 'fontWeight': '700'}),
        ], style={
            'backgroundColor': thm['btn_bg'], 'border': f"1px solid {thm['btn_border']}",
            'borderRadius': '8px', 'padding': '10px 12px', 'minWidth': '110px', 'flex': '0 1 auto',
        })

    plateau_alert = html.Div(style={'display': 'none'})
    info_panel = html.Div()

    if not hist:
        stats_html = [stat_card("Oczekuję na dane...", "-", COLORS['text'])]
        if cfg:
            info_panel = html.Div([
                html.Div("INFO / KONFIG", style={'color': thm['subtext'], 'fontSize': '11px', 'fontWeight': '800', 'letterSpacing': '1px', 'marginBottom': '10px'}),
                html.Div([
                    chip("exp", cfg.get('experiment_version', '—'), COLORS['accent']),
                    chip("window", cfg.get('window_id') or 'full'),
                    chip("ep bars", cfg.get('episode_length', '—')),
                    chip("gamma", cfg.get('gamma', '—')),
                    chip("models", ", ".join(f"{k}:{v}" for k, v in (cfg.get('model_files') or {}).items()) or '—'),
                ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '8px'}),
            ], style={'backgroundColor': thm['card'], 'padding': '16px 18px', 'borderRadius': '12px', 'border': '1px solid rgba(255,255,255,0.05)', 'boxShadow': thm['shadow']})
    else:
        last_ep = hist[-1]
        prev_ep = hist[-2] if len(hist) > 1 else last_ep
        
        avg_p = last_ep.get('portfolio', {}).get('avg_end_p', 0)
        prev_p = prev_ep.get('portfolio', {}).get('avg_end_p', 0)
        p_delta = avg_p - prev_p
        
        avg_r = last_ep.get('reward', {}).get('avg_end_r', 0)
        prev_r = prev_ep.get('reward', {}).get('avg_end_r', 0)
        r_delta = avg_r - prev_r
        
        avg_t = last_ep.get('avg_transactions', 0)
        prev_t = prev_ep.get('avg_transactions', 0)
        t_delta = avg_t - prev_t

        excess = float(last_ep.get('avg_excess_pct', 0.0) or 0.0)
        prev_excess = float(prev_ep.get('avg_excess_pct', excess) or excess)
        tx_day = float(last_ep.get('avg_tx_per_day', 0.0) or 0.0)
        prev_tx_day = float(prev_ep.get('avg_tx_per_day', tx_day) or tx_day)
        
        p_color = COLORS['success'] if avg_p >= 1000 else COLORS['danger']
        r_color = COLORS['success'] if avg_r > 0 else COLORS['warning']
        x_color = COLORS['success'] if excess >= 0 else COLORS['danger']
        txd_color = COLORS['success'] if tx_day <= 6 else (COLORS['warning'] if tx_day <= 20 else COLORS['danger'])
        
        import datetime
        start_time_str = cfg.get('start_time')
        time_str = "-"
        if start_time_str:
            try:
                start_time = datetime.datetime.fromisoformat(start_time_str)
                elapsed = datetime.datetime.now() - start_time
                hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
                minutes, seconds = divmod(remainder, 60)
                time_str = f"{hours}h {minutes}m {seconds}s"
            except Exception:
                pass

        stats_html = [
            stat_card("Czas Treningu", time_str, COLORS['text']),
            stat_card("Średni Kapitał", f"${avg_p:.2f}", p_color, delta=p_delta),
            stat_card("Średnia Nagroda", f"{avg_r:.3f}", r_color, delta=r_delta),
            stat_card("Transakcje / Epoka", f"{avg_t:.0f}", COLORS['accent'], delta=t_delta, reverse_delta_color=True),
            stat_card("Excess vs HODL", f"{excess:+.2f}%", x_color, delta=excess - prev_excess),
            stat_card("Tx / Dzień", f"{tx_day:.2f}", txd_color, delta=tx_day - prev_tx_day, reverse_delta_color=True),
        ]

        knobs = last_ep.get('knobs') or {}
        models = cfg.get('model_files') or {}
        models_txt = ", ".join(f"{k}:{v}" for k, v in models.items()) or "—"

        info_rows = [
            chip("exp", cfg.get('experiment_version', '—'), COLORS['accent']),
            chip("window", cfg.get('window_id') or 'full'),
            chip("symbols", ",".join(cfg.get('symbols') or []) or "—"),
            chip("ticks", cfg.get('num_ticks', '—')),
            chip("ep bars", cfg.get('episode_length', '—')),
            chip("gamma", cfg.get('gamma', '—')),
            chip("n_steps", cfg.get('n_steps', '—')),
            chip("batch", cfg.get('batch_size', '—')),
            chip("ent", cfg.get('ent_coef', '—')),
            chip("seed", cfg.get('seed', '—')),
            chip("lr", cfg.get('learning_rate', '—')),
            chip("total steps", cfg.get('total_timesteps', '—')),
            chip("models", models_txt),
            chip("strat %", f"{float(last_ep.get('avg_strat_pct', 0) or 0):+.2f}"),
            chip("hodl %", f"{float(last_ep.get('avg_hodl_pct', 0) or 0):+.4f}"),
            chip("max DD %", f"{float(last_ep.get('avg_max_dd_pct', 0) or 0):.2f}", COLORS['danger']),
            chip("closed", f"{float(last_ep.get('avg_closed', 0) or 0):.1f}"),
        ]
        if knobs:
            info_rows.extend([
                chip("bias", f"{knobs.get('mp_alloc_bias', 0):.3f}"),
                chip("scale", f"{knobs.get('mp_alloc_scale', 0):.3f}"),
                chip("conf", f"{knobs.get('mp_confidence', 0):.3f}"),
                chip("urg", f"{knobs.get('mp_urgency', 0):.3f}"),
                chip("sig_dz", f"{knobs.get('mp_signal_deadzone', 0):.3f}"),
                chip("reb_dz", f"{knobs.get('mp_rebalance_deadzone', 0):.3f}"),
            ])

        info_panel = html.Div([
            html.Div("INFO — konfiguracja + ostatni dump", style={
                'color': thm['subtext'], 'fontSize': '11px', 'fontWeight': '800',
                'letterSpacing': '1px', 'marginBottom': '12px', 'textTransform': 'uppercase',
            }),
            html.Div(info_rows, style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '8px'}),
        ], style={
            'backgroundColor': thm['card'], 'padding': '16px 18px', 'borderRadius': '12px',
            'border': '1px solid rgba(255,255,255,0.05)', 'boxShadow': thm['shadow'],
        })

        # Early Stopping / Plateau Detection Mechanic
        if len(hist) > 30:
            recent_rewards = [h['reward']['avg_end_r'] for h in hist[-20:]]
            past_rewards = [h['reward']['avg_end_r'] for h in hist[-40:-20]]
            if sum(recent_rewards) <= sum(past_rewards) + 0.001:
                plateau_alert = html.Div([
                    html.Strong("⚠️ ZASTÓJ TRENINGU (PLATEAU DETECTED): "),
                    html.Span("Nagroda nie rośnie od ponad 20 epok. Dalszy trening może prowadzić do overfittingu. Rozważ zatrzymanie procesu (Early Stopping).")
                ], style={'backgroundColor': 'rgba(255, 23, 68, 0.15)', 'color': COLORS['danger'], 'padding': '15px', 'borderRadius': '8px', 'border': f'1px solid {COLORS["danger"]}', 'marginBottom': '12px', 'textAlign': 'center', 'width': '100%', 'boxSizing': 'border-box'})

    return prog_style, prog_text, [plateau_alert] + stats_html, info_panel


# 7. Logika przycisków Gridu
@app.callback(
    Output('view-mode-store', 'data'),
    [Input('btn-auto', 'n_clicks'), Input('btn-pie', 'n_clicks'),
     Input('btn-port', 'n_clicks'), Input('btn-rew', 'n_clicks')],
    [State('view-mode-store', 'data')]
)
def change_mode(b1, b2, b3, b4, current_mode):
    ctx = dash.callback_context
    if not ctx.triggered: return dash.no_update
    
    # Ochrona przed fałszywym odpalaniem na starcie (kiedy n_clicks jest puste)
    if not any([b1, b2, b3, b4]): return dash.no_update
    
    bid = ctx.triggered[0]['prop_id'].split('.')[0]
    if bid == 'btn-pie':
        return 0
    elif bid == 'btn-port':
        return 1
    elif bid == 'btn-rew':
        return 2
    elif bid == 'btn-auto':
        return 'auto'
        
    return dash.no_update


# 8. Generowanie Gridu (Procesy)
@app.callback(
    [Output('graphs-container', 'children'), Output('global-title', 'children'), Output('process-count', 'children')],
    [Input('interval-component', 'n_intervals'),
     Input('view-mode-store', 'data'),
     Input('theme-store', 'data')]  # Dodano input motywu
)
def update_grid(n_intervals, view_mode, theme_name):
    data_list = get_process_data()
    theme_name = theme_name or 'dark'
    thm = THEMES[theme_name]

    if not data_list: return [html.Div("Oczekiwanie...", style={'color': thm['subtext']})], "Brak Danych", ""

    current_view = n_intervals % 3 if view_mode == 'auto' else view_mode

    global_title_text = "Widok: Decyzje"
    if current_view == 1:
        global_title_text = "Widok: Portfel"
    elif current_view == 2:
        global_title_text = "Widok: Nagrody (skumulowane)"

    graphs_html = []

    # Dynamiczny styl karty w gridzie
    current_card_style = {
        **base_card_style,
        'backgroundColor': thm['card'],
        'color': thm['text'],
        'boxShadow': thm['shadow'],
        'width': '30%',
        'minWidth': '300px', # Zapobiega zgniataniu na małych ekranach
        'marginBottom': '20px'
    }

    for p in data_list:
        fig = go.Figure()

        metrics = p.get('metrics', {'buys': 0, 'sells': 0, 'holds': 0})
        port_data = p.get('portfolio', {})
        rew_data = p.get('reward', {})

        p_id = p.get('process_id', '?')
        epoch = p.get('epoch', '?')

        # --- LOGIKA WYKRESÓW IDENTYCZNA JAK W ORYGINALE ---
        if current_view == 0:
            v = [metrics['buys'], metrics['sells'], metrics['holds']]
            if sum(v) == 0: v = [0, 0, 1]
            fig.add_trace(go.Pie(labels=['Buy', 'Sell', 'Hold'], values=v, hole=0.65,
                                 marker=dict(colors=[COLORS['success'], COLORS['danger'], '#475569']),
                                 textinfo='percent', hoverinfo='label+value'))

        elif current_view == 1:
            y = port_data.get('portfolio_values', [])
            x = list(range(len(y)))

            fig.add_trace(go.Scatter(
                x=x, y=y, mode='lines',
                fill='tozeroy', fillcolor='rgba(0, 212, 255, 0.1)',
                line=dict(color=COLORS['accent'], width=2),
                name='Agent'
            ))

            # HODL: pełna krzywa z logów (::50 jak portfel); fallback = linia trendu
            hodl_y = port_data.get('hodl_values') or []
            hodl_ret = metrics.get('hodl_return_pct')
            if hodl_y and y:
                # wyrównaj długość do portfela gdy lekki mismatch
                if len(hodl_y) != len(y):
                    if len(hodl_y) > 1 and len(y) > 1:
                        idxs = [
                            int(round(i * (len(hodl_y) - 1) / (len(y) - 1)))
                            for i in range(len(y))
                        ]
                        hodl_y = [hodl_y[j] for j in idxs]
                    else:
                        hodl_y = list(hodl_y[:len(y)]) + [hodl_y[-1]] * max(0, len(y) - len(hodl_y))
                fig.add_trace(go.Scatter(
                    x=x, y=hodl_y, mode='lines',
                    line=dict(color=COLORS['warning'], dash='dash', width=2),
                    name=f"HODL ({float(hodl_ret or 0):+.4f}%)"
                ))
            elif y and hodl_ret is not None:
                try:
                    start_p = float(y[0]) if y[0] else 1000.0
                    end_hodl = start_p * (1.0 + float(hodl_ret) / 100.0)
                    if len(y) == 1:
                        hodl_y = [end_hodl]
                    else:
                        hodl_y = [
                            start_p + (end_hodl - start_p) * (i / (len(y) - 1))
                            for i in range(len(y))
                        ]
                    fig.add_trace(go.Scatter(
                        x=x, y=hodl_y, mode='lines',
                        line=dict(color=COLORS['warning'], dash='dash', width=2),
                        name=f"HODL trend ({float(hodl_ret):+.4f}%)"
                    ))
                except (TypeError, ValueError):
                    hodl_y = None
            else:
                hodl_y = None

            mn = port_data.get('min_p')
            mx = port_data.get('max_p')
            avg = port_data.get('mean_p')
            x_range = [x[0], x[-1]] if x else [0, 0]

            if mx:
                fig.add_trace(go.Scatter(
                    x=x_range, y=[mx, mx], mode='lines',
                    line=dict(color=COLORS['success'], dash='dot', width=1),
                    name='Max'
                ))
            if mn:
                fig.add_trace(go.Scatter(
                    x=x_range, y=[mn, mn], mode='lines',
                    line=dict(color=COLORS['danger'], dash='dot', width=1),
                    name='Min'
                ))
            if avg:
                fig.add_trace(go.Scatter(
                    x=x_range, y=[avg, avg], mode='lines',
                    line=dict(color='#bbbbbb', dash='dash', width=1),  # Szary
                    name='Mean'
                ))

            if y:
                y_all = list(y) + (hodl_y or [])
                y_min = min(y_all)
                y_max = max(y_all)
                # stały padding ±50$ — małe wykresy procesów czytelniejsze
                fig.update_layout(yaxis=dict(range=[y_min - 50, y_max + 50]))

        elif current_view == 2:
            y = rew_data.get('reward_values', [])
            x = list(range(len(y)))

            fig.add_trace(go.Scatter(
                x=x, y=y, mode='lines',
                line=dict(color=COLORS['warning'], width=2),
                name='Reward'
            ))

            fig.add_hline(y=0, line_color="#444", line_width=1)

            mn = rew_data.get('min_r')
            mx = rew_data.get('max_r')
            avg = rew_data.get('mean_r')
            x_range = [x[0], x[-1]] if x else [0, 0]

            if mx:
                fig.add_trace(go.Scatter(
                    x=x_range, y=[mx, mx], mode='lines',
                    line=dict(color=COLORS['success'], dash='dot', width=1),
                    name='Max'
                ))
            if mn:
                fig.add_trace(go.Scatter(
                    x=x_range, y=[mn, mn], mode='lines',
                    line=dict(color=COLORS['danger'], dash='dot', width=1),
                    name='Min'
                ))
            if avg:
                fig.add_trace(go.Scatter(
                    x=x_range, y=[avg, avg], mode='lines',
                    line=dict(color='#bbbbbb', dash='dash', width=1),
                    name='Mean'
                ))

        fig.update_layout(
            template=thm['template'],
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(size=11, color=thm['text'], family='"Inter", sans-serif'),
            margin=dict(l=10, r=10, t=40, b=30),
            xaxis=dict(visible=False),
            yaxis=dict(showgrid=True, gridcolor=thm['grid_color']),
            title=dict(text=f"Proces: {p_id} | Epoka: {epoch}", font=dict(size=12, color=thm['subtext']), x=0.5,
                       xanchor='center'),
            showlegend=True,
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.1,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(0,0,0,0)",
                font=dict(size=11)
            ),
            height=300
        )
        card = html.Div([dcc.Graph(figure=fig, config={'displayModeBar': False})], style=current_card_style)
        graphs_html.append(card)

    return graphs_html, global_title_text, f"Aktywne procesy: {len(data_list)}"


if __name__ == '__main__':
    app.run(debug=False, port=8050)