"""Dashboard Dash: przegląd wyników backtestów z shared/simulation/outputs."""

import dash
from dash import dcc, html, ctx
from dash.dependencies import Input, Output, State, ALL, MATCH
import plotly.graph_objs as go
import plotly.io as pio
import json
import gzip
import os
import glob
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUTPUT_DIR = os.path.join(BASE_DIR, "shared", "simulation", "outputs")

THEMES = {
    'dark': {
        'bg': '#0B0F19',
        'sidebar': 'rgba(8, 12, 20, 0.85)',
        'card': 'rgba(28, 35, 49, 0.7)',
        'text': '#E2E8F0',
        'subtext': '#94A3B8',
        'border': 'rgba(255, 255, 255, 0.1)',
        'chart_template': 'plotly_dark',
        'success': '#00E5FF',
        'shadow': '0 8px 32px 0 rgba(0, 0, 0, 0.37)'
    },
    'light': {
        'bg': '#F1F5F9',
        'sidebar': 'rgba(255, 255, 255, 0.9)',
        'card': 'rgba(255, 255, 255, 0.8)',
        'text': '#1E293B',
        'subtext': '#64748B',
        'border': '#CBD5E1',
        'chart_template': 'plotly_white',
        'success': '#0EA5E9',
        'shadow': '0 8px 32px 0 rgba(0, 0, 0, 0.1)'
    }
}

ACCENTS = {
    'blue': '#00E5FF',
    'cyan': '#0EA5E9',
    'red': '#F43F5E',
    'orange': '#F59E0B',
    'success': '#22C55E',
}

# ModeBar bez kamery — eksport „pod pracę” osobnym przyciskiem (inny styl niż ekran)
GRAPH_CONFIG = {
    'responsive': True,
    'displayModeBar': True,
    'displaylogo': False,
    'modeBarButtonsToRemove': [
        'lasso2d', 'select2d', 'autoScale2d', 'toImage',
    ],
}


def wrap_exportable_chart(fig, chart_key, download_name, thm):
    """Wykres na ekranie bez zmian; PNG do pracy przez przycisk (osobny styl)."""
    return html.Div([
        html.Div([
            html.Button(
                "Pobierz PNG (praca)",
                id={'type': 'dl-paper-btn', 'index': chart_key},
                n_clicks=0,
                title="Eksport pod pracę: czarne napisy, większa czcionka, linie bez przezroczystości",
                style={
                    'padding': '8px 14px',
                    'borderRadius': '8px',
                    'cursor': 'pointer',
                    'border': f"1px solid {thm['border']}",
                    'backgroundColor': 'rgba(0, 229, 255, 0.12)',
                    'color': thm['text'],
                    'fontFamily': "'Inter', sans-serif",
                    'fontSize': '12px',
                    'fontWeight': '600',
                },
            ),
            html.Span(id={'type': 'dl-paper-status', 'index': chart_key}, style={'display': 'none'}),
            dcc.Store(id={'type': 'dl-paper-name', 'index': chart_key}, data=download_name),
        ], style={'display': 'flex', 'justifyContent': 'flex-end', 'marginBottom': '8px', 'gap': '8px'}),
        dcc.Graph(
            id={'type': 'sim-chart', 'index': chart_key},
            figure=fig,
            style={'height': '550px', 'width': '100%'},
            config=GRAPH_CONFIG,
        ),
    ], style={
        'backgroundColor': thm['card'],
        'padding': '15px',
        'borderRadius': '16px',
        'border': f"1px solid {thm['border']}",
        'marginBottom': '40px',
        'boxShadow': thm['shadow'],
        'backdropFilter': 'blur(10px)',
    })

pio.templates.default = "plotly_dark"

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Simulation Viewer"

# Pamięć podręczna: raw JSON (nie cache'uj obiektów dcc.Tab — Dash/React się myli)
DATA_CACHE = {}
MAX_CHART_POINTS = 2500


def downsample_series(values, max_points=MAX_CHART_POINTS):
    """Zmniejsza serię do wykresu — pełne 40k punktów zabija 2./3. zakładkę."""
    if not values:
        return []
    n = len(values)
    if n <= max_points:
        return list(values)
    step = max(1, n // max_points)
    out = list(values[::step])
    if out[-1] != values[-1]:
        out.append(values[-1])
    return out


def downsample_with_day_axis(values, max_points=MAX_CHART_POINTS, minutes_per_tick=1):
    """
    Downsample Y + oś X w dniach od startu okna.
    Symulacja zapisuje co 1 min (pełna seria) — NIE ::50 (to tylko logi treningu RL).
    Viewer ucina do ~max_points dla płynności (np. 40k → step≈16 ≈ co 16 min).
    """
    if not values:
        return [], []
    n = len(values)
    if n <= max_points:
        idxs = list(range(n))
    else:
        step = max(1, n // max_points)
        idxs = list(range(0, n, step))
        if idxs[-1] != n - 1:
            idxs.append(n - 1)
    ys = [values[i] for i in idxs]
    xs = [i * float(minutes_per_tick) / (60.0 * 24.0) for i in idxs]
    return xs, ys


def scatter_y(y, **kwargs):
    """Scatter z osią X = dni od startu (1 tick = 1 min w danych tezy)."""
    xs, ys = downsample_with_day_axis(y)
    trace_cls = go.Scattergl if len(ys) > 800 else go.Scatter
    return trace_cls(x=xs, y=ys, **kwargs)


AXIS_TITLE_X = "Dni od startu okna"
AXIS_TITLE_Y = "Wartość portfela (USDT)"

# --- CUSTOM CSS ---
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
        <style>
            body {
                font-family: 'Inter', 'Roboto', sans-serif !important;
                margin: 0;
                overflow-x: hidden;
                box-sizing: border-box;
                transition: background-color 0.3s ease;
            }
            *, *:before, *:after { box-sizing: inherit; }

            /* Custom Scrollbar */
            ::-webkit-scrollbar { width: 8px; height: 8px; }
            ::-webkit-scrollbar-track { background: rgba(0,0,0,0.1); }
            ::-webkit-scrollbar-thumb { background: rgba(128,128,128,0.5); border-radius: 4px; }
            ::-webkit-scrollbar-thumb:hover { background: rgba(128,128,128,0.8); }

            /* Grid layout for Metrics */
            .responsive-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 15px;
                width: 100%;
            }
            
            /* Tabs */
            .custom-tabs-container { margin-top: 20px; width: 100%; }

            /* Modern Glass Table */
            .table-wrapper {
                overflow-x: auto;
                border-radius: 12px;
                box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
                backdrop-filter: blur(10px);
            }
            .comparison-table {
                width: 100%;
                border-collapse: collapse;
                font-size: 14px;
                background-color: transparent;
            }
            .comparison-table th, .comparison-table td {
                padding: 16px 20px;
                text-align: left;
                border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            }
            .comparison-table tr {
                transition: background-color 0.2s;
            }
            .comparison-table tbody tr:hover {
                background-color: rgba(255, 255, 255, 0.03);
            }
            .comparison-table th {
                font-weight: 600;
                text-transform: uppercase;
                font-size: 12px;
                letter-spacing: 1px;
                color: var(--subtext-color);
                background-color: rgba(0, 0, 0, 0.2);
            }
            .badge {
                padding: 4px 8px;
                border-radius: 12px;
                font-weight: bold;
                font-size: 13px;
            }
            
            /* HTML Details marker */
            details > summary { list-style: none; }
            details > summary::-webkit-details-marker { display: none; }
            details > summary::before {
                content: '▶';
                font-size: 10px;
                margin-right: 8px;
                display: inline-block;
                transition: transform 0.2s;
            }
            details[open] > summary::before { transform: rotate(90deg); }

            .filter-tiles {
                display: flex;
                gap: 8px;
                margin-bottom: 4px;
                width: 100%;
            }
            .filter-tile {
                flex: 1;
                padding: 10px 6px;
                border-radius: 10px;
                cursor: pointer;
                font-family: 'Inter', sans-serif;
                font-size: 11px;
                font-weight: 600;
                letter-spacing: 0.3px;
                text-align: center;
                line-height: 1.25;
                transition: background-color 0.2s, border-color 0.2s, color 0.2s, box-shadow 0.2s;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }
            .filter-tile:hover {
                filter: brightness(1.12);
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# --- BAZOWE STYLE ---
sidebar_base_style = {
    'position': 'fixed', 'top': 0, 'left': 0, 'bottom': 0, 'width': '22rem',
    'padding': '2rem 1.5rem', 'overflowY': 'auto',
    'zIndex': 100,
    'transition': 'all 0.3s',
    'backdropFilter': 'blur(15px)'
}

content_base_style = {
    'marginLeft': '22rem',
    'padding': '40px',
    'minHeight': '100vh',
    'overflowX': 'hidden',
    'transition': 'background-color 0.3s, color 0.3s'
}

file_btn_base_style = {
    'width': '100%', 'textAlign': 'left', 'padding': '12px', 'marginBottom': '8px',
    'backgroundColor': 'transparent',
    'cursor': 'pointer', 'borderRadius': '6px', 'fontSize': '13px', 'transition': '0.2s',
    'fontFamily': 'Roboto, sans-serif'
}

# Filtry sidebar: 3 kafelki obok siebie
FILE_FILTER_TILES = (
    ('all', 'Wszystkie'),
    ('thesis', 'Thesis'),
    ('rltrain', 'RL train'),
)


def filter_tile_style(value, selected, theme_name):
    thm = THEMES[theme_name or 'dark']
    active = value == (selected or 'all')
    if active:
        return {
            'backgroundColor': 'rgba(0, 229, 255, 0.14)' if theme_name != 'light' else 'rgba(14, 165, 233, 0.14)',
            'border': f"1px solid {ACCENTS['blue'] if theme_name != 'light' else ACCENTS['cyan']}",
            'color': ACCENTS['blue'] if theme_name != 'light' else ACCENTS['cyan'],
            'boxShadow': '0 0 0 1px rgba(0, 229, 255, 0.15)',
        }
    return {
        'backgroundColor': thm['card'],
        'border': f"1px solid {thm['border']}",
        'color': thm['subtext'],
        'boxShadow': 'none',
    }

metric_card_base_style = {
    'borderRadius': '12px', 'padding': '20px',
    'textAlign': 'left',
    'width': '100%',
    'boxSizing': 'border-box',
    'transition': 'all 0.3s ease'
}

section_header_base_style = {
    'paddingBottom': '5px',
    'marginBottom': '15px',
    'marginTop': '30px',
    'fontWeight': 'bold',
    'fontSize': '18px',
    'letterSpacing': '1px',
    'width': '100%',
    'fontFamily': "'Inter', 'Roboto', sans-serif"
}

close_btn_style = {
    'position': 'absolute',
    'top': '0px',
    'right': '0px',
    'backgroundColor': 'rgba(255, 82, 82, 0.1)',
    'border': '1px solid #ff5252',
    'color': '#ff5252',
    'padding': '6px 12px',
    'borderRadius': '4px',
    'cursor': 'pointer',
    'fontWeight': 'bold',
    'fontSize': '12px',
    'zIndex': 10,
    'transition': '0.2s'
}

# --- FUNKCJE POMOCNICZE ---

def clean_filename(filename):
    return filename.replace(".json.gz", "").replace(".json", "")


def sim_symbols(data):
    """Kolejność: config.symbols → keys modeli → keys data → HODL refs."""
    if not data:
        return []
    cfg = data.get("config") or {}
    syms = list(cfg.get("symbols") or [])
    if not syms:
        files = data.get("files") or {}
        for key in ("models", "agents", "data"):
            block = files.get(key) or {}
            if isinstance(block, dict) and block:
                syms = list(block.keys())
                break
    if not syms:
        refs = (data.get("hodl") or {}).get("crypto_ref_values") or {}
        syms = list(refs.keys())
    return [str(s) for s in syms if s]


def legend_mp_label(data):
    syms = sim_symbols(data)
    if len(syms) >= 2:
        return f"Model MP ({' | '.join(syms)})"
    if len(syms) == 1:
        return f"Model MP ({syms[0]})"
    return "Model MP"


def legend_rl_label(data, prefix="Agent RL"):
    syms = sim_symbols(data)
    if len(syms) >= 2:
        return f"{prefix} ({' | '.join(syms)})"
    if len(syms) == 1:
        return f"{prefix} ({syms[0]})"
    return prefix

def parse_simulation_version(filename):
    base = clean_filename(filename)
    ver_match = re.search(r'_v(\d+)', base)
    if ver_match:
        return f"Model v{ver_match.group(1)}"
    return "Inne Modele"

def parse_simulation_name(filename):
    try:
        return os.path.getmtime(os.path.join(OUTPUT_DIR, filename))
    except:
        return 0

def is_rltrain_sim(filename: str) -> bool:
    """Symulacje z watchera treningu RL — tag `rltrain` w nazwie."""
    return "rltrain" in (filename or "").lower()


def is_thesis_seq_sim(filename: str) -> bool:
    """Sekwencja tezy — prefix `seq_` (R0–M3, okna + sequence_B)."""
    name = (filename or "").lower()
    return name.startswith("seq_")


def get_simulation_files_grouped(file_filter="all"):
    if not os.path.exists(OUTPUT_DIR): return {}
    files = glob.glob(os.path.join(OUTPUT_DIR, "*.json.gz"))
    files = [os.path.basename(f) for f in files]
    if file_filter == "thesis":
        files = [f for f in files if is_thesis_seq_sim(f)]
    elif file_filter == "rltrain":
        files = [f for f in files if is_rltrain_sim(f)]
    elif file_filter == "other":
        files = [f for f in files if not is_rltrain_sim(f) and not is_thesis_seq_sim(f)]
    files.sort(key=parse_simulation_name, reverse=True)
    
    grouped = {}
    for f in files:
        import datetime
        mtime = os.path.getmtime(os.path.join(OUTPUT_DIR, f))
        date_str = datetime.datetime.fromtimestamp(mtime).strftime("%d.%m.%Y")
        if date_str not in grouped:
            grouped[date_str] = []
        grouped[date_str].append(f)
    return grouped

def load_sim_data_cached(filename):
    if filename in DATA_CACHE:
        return DATA_CACHE[filename]
    filepath = os.path.join(OUTPUT_DIR, filename)
    try:
        with gzip.open(filepath, "rt", encoding="utf-8") as f:
            data = json.load(f)
        if len(DATA_CACHE) >= 40:
            oldest = next(iter(DATA_CACHE))
            del DATA_CACHE[oldest]
        DATA_CACHE[filename] = data
        return data
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

def format_value_for_card(value):
    if isinstance(value, float): return f"{value:.4f}"
    if isinstance(value, list):
        filenames = [os.path.basename(str(v)) for v in value]
        if not filenames: return "-"
        if len(filenames) > 3: return f"{len(filenames)} files"
        return ", ".join(filenames)
    if isinstance(value, dict):
        items = [f"{k}: {os.path.basename(str(v))}" for k, v in value.items()]
        return html.Div([html.Div(i, style={'fontSize': '12px', 'marginBottom': '2px'}) for i in items])
    return str(value)

def create_info_cards(info_dict, theme_name, text_color=None):
    cards = []
    thm = THEMES[theme_name]
    if text_color is None: text_color = thm['text']

    if not info_dict: return [html.Div("Brak danych", style={'color': thm['subtext'], 'padding': '10px'})]

    style = {
        **metric_card_base_style,
        'backgroundColor': thm['card'],
        'border': f'1px solid {thm["border"]}',
        'boxShadow': thm['shadow'],
        'backdropFilter': 'blur(10px)',
        'borderLeft': f'3px solid {thm["border"]}'
    }

    for key, value in info_dict.items():
        display_value = format_value_for_card(value)
        display_key = key.replace("_", " ").upper()
        
        card_style = style.copy()
        if "[RL]" in display_key:
            card_style['borderLeft'] = f"3px solid {ACCENTS['red']}"
        elif "[MP]" in display_key:
            card_style['borderLeft'] = f"3px solid {ACCENTS['cyan']}"
        elif "[HODL]" in display_key:
            card_style['borderLeft'] = f"3px solid {ACCENTS['orange']}"
        else:
            card_style['borderLeft'] = f"3px solid {text_color}"

        card = html.Div([
            html.Div(display_key, style={'color': thm['subtext'], 'fontSize': '11px', 'fontWeight': '600', 'letterSpacing': '1px', 'marginBottom': '8px'}),
            html.Div(display_value, style={'color': text_color, 'fontSize': '24px', 'fontWeight': '700', 'margin': '0', 'wordBreak': 'break-word'})
        ], style=card_style)
        cards.append(card)
    return cards


def get_tab_styles(theme_name):
    thm = THEMES[theme_name]
    tab_style = {
        'backgroundColor': thm['card'],
        'color': thm['subtext'],
        'border': f'1px solid {thm["border"]}',
        'padding': '12px 20px',
        'fontFamily': 'Roboto, sans-serif',
        'width': 'auto',
        'flex': 'none',
        'marginRight': '5px',
        'cursor': 'pointer',
        'borderRadius': '8px 8px 0 0'
    }
    selected_tab_style = {
        'backgroundColor': thm['bg'],
        'color': thm['text'],
        'fontWeight': 'bold',
        'borderTop': f'2px solid {ACCENTS["blue"]}',
        'borderBottom': 'none',
        'borderLeft': f'1px solid {thm["border"]}',
        'borderRight': f'1px solid {thm["border"]}',
        'padding': '12px 20px',
        'fontFamily': 'Roboto, sans-serif',
        'width': 'auto',
        'flex': 'none',
        'marginRight': '5px',
        'cursor': 'pointer',
        'borderRadius': '8px 8px 0 0'
    }
    return tab_style, selected_tab_style

# --- GENERATORY ZAKŁADEK ---

def _fmt_price(value):
    try:
        return f"{float(value):,.2f} USDT"
    except (TypeError, ValueError):
        return "-"


def build_price_windows_figure(data, theme_name):
    """Cena close + pionowe linie na granicach okien. Zwykłe symulacje tu nie wchodzą."""
    thm = THEMES[theme_name or "dark"]
    dark = (theme_name or "dark") != "light"
    price = data.get("price") or {}
    xs = list(price.get("timestamps") or [])
    ys = list(price.get("close") or [])
    cfg = data.get("config") or {}
    symbol = str(cfg.get("symbol") or "cena")
    line_color = "#60A5FA" if dark else "#1D4ED8"
    # Żółte, grube kreski — cienkie shape z add_vline giną w Plotly.toImage (PNG).
    rule_color = "#FBBF24" if dark else "#B45309"
    label_color = "#111827"
    label_bg = "#FDE68A"

    finite = []
    for value in ys:
        try:
            finite.append(float(value))
        except (TypeError, ValueError):
            continue
    if finite:
        ymin, ymax = min(finite), max(finite)
        span = ymax - ymin
        if span <= 0:
            span = abs(ymax) * 0.05 or 1.0
        y0 = ymin - span * 0.05
        y1 = ymax + span * 0.28
    else:
        y0, y1 = 0.0, 1.0

    fig = go.Figure()
    fig.update_xaxes(type="date")

    windows = list(data.get("windows") or [])
    boundaries = []
    for w in windows:
        start = w.get("start")
        if start and start not in boundaries:
            boundaries.append(start)
    if windows and windows[-1].get("end") and windows[-1]["end"] not in boundaries:
        boundaries.append(windows[-1]["end"])
    for i, bound in enumerate(boundaries):
        fig.add_trace(go.Scatter(
            x=[bound, bound],
            y=[y0, y1],
            mode="lines",
            name="Granice okien",
            legendgroup="window-boundary",
            showlegend=(i == 0),
            hoverinfo="skip",
            line=dict(color=rule_color, width=2.6, dash="dash"),
        ))
    if xs and ys:
        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            name=symbol,
            line=dict(color=line_color, width=2.2),
            hovertemplate="%{x|%d.%m.%Y %H:%M}<br>Cena: %{y:,.2f} USDT<extra></extra>",
        ))
    for w in windows:
        label_x = w.get("label_x") or w.get("start")
        wid = str(w.get("id") or "")
        if not label_x or not wid:
            continue
        fig.add_annotation(
            x=label_x,
            y=1.0,
            yref="paper",
            text=f"<b>{wid}</b>",
            showarrow=False,
            yanchor="top",
            yshift=-8,
            font=dict(size=16, color=label_color, family="Inter, Arial, sans-serif"),
            bgcolor=label_bg,
            bordercolor=label_color,
            borderwidth=1,
            borderpad=3,
        )

    x_range = data.get("x_range") or None
    fig.update_layout(
        template=thm["chart_template"],
        title=dict(
            text=f"Cena {symbol}",
            font=dict(family='"Inter", sans-serif', size=18, color=thm["text"], weight="bold"),
            x=0.5,
            xanchor="center",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=thm["subtext"], family='"Inter", sans-serif'),
        hovermode="x unified",
        xaxis=dict(
            title=dict(text="Data", font=dict(size=12, color=thm["subtext"])),
            type="date",
            range=x_range,
            tickformat="%m.%Y",
            dtick="M1",
            showgrid=True,
            gridcolor=thm["border"],
        ),
        yaxis=dict(
            title=dict(text="Cena (USDT)", font=dict(size=12, color=thm["subtext"])),
            showgrid=True,
            gridcolor=thm["border"],
            separatethousands=True,
            range=[y0, y1] if finite else None,
        ),
        legend=dict(
            orientation="v",
            yanchor="bottom",
            y=0.02,
            xanchor="left",
            x=0.01,
            bgcolor=thm["card"],
            bordercolor=thm["border"],
            borderwidth=1,
            font=dict(size=12, color=thm["text"]),
        ),
        margin=dict(l=80, r=28, t=64, b=55),
        height=580,
    )
    fig.update_annotations(
        font=dict(size=16, color=label_color, family="Inter, Arial, sans-serif"),
        bgcolor=label_bg,
        bordercolor=label_color,
        borderwidth=1,
        borderpad=3,
    )
    return fig


def build_price_windows_tab(filename, data, theme_name):
    thm = THEMES[theme_name]
    fig = build_price_windows_figure(data, theme_name)
    chart_section = wrap_exportable_chart(
        fig,
        chart_key=filename,
        download_name=clean_filename(filename or "cena"),
        thm=thm,
    )

    closes = list((data.get("price") or {}).get("close") or [])
    metrics_dict = {}
    if closes:
        first, last = float(closes[0]), float(closes[-1])
        metrics_dict["Cena na początku"] = _fmt_price(first)
        metrics_dict["Cena na końcu"] = _fmt_price(last)
        metrics_dict["Minimum"] = _fmt_price(min(closes))
        metrics_dict["Maksimum"] = _fmt_price(max(closes))
        if first:
            metrics_dict["Zmiana w okresie"] = f"{(last - first) / first * 100:+.2f}%"
        metrics_dict["Punkty na wykresie"] = len(closes)

    h_style = section_header_base_style.copy()
    h_style["color"] = thm["text"]
    h_style["borderBottom"] = f"1px solid {thm['border']}"

    metrics_section = html.Div([
        html.Div("Cena", style=h_style),
        html.Div(create_info_cards(metrics_dict, theme_name, ACCENTS["blue"]), className="responsive-grid"),
    ], style={"marginBottom": "40px"})

    config_section = html.Div([
        html.Div("Zakres i okna", style=h_style),
        html.Div(create_info_cards(data.get("config") or {}, theme_name, thm["text"]), className="responsive-grid"),
    ], style={"marginBottom": "50px"})

    close_btn = html.Button(
        "Zamknij",
        id={"type": "close-tab-btn", "index": filename},
        style=close_btn_style,
        title="Zamknij zakładkę",
    )
    tab_content = html.Div(
        [close_btn, html.Div(style={"clear": "both"}), chart_section, metrics_section, config_section],
        style={"position": "relative", "marginTop": "10px", "width": "100%"},
    )
    tab_style, selected_tab_style = get_tab_styles(theme_name)
    return dcc.Tab(
        label=clean_filename(filename),
        value=f"tab-{filename}",
        children=tab_content,
        style=tab_style,
        selected_style=selected_tab_style,
    )


def build_single_tab(filename, theme_name, hodl_mode="avg"):
    data = load_sim_data_cached(filename)
    if not data: return None
    if data.get("chart_kind") == "price_windows":
        return build_price_windows_tab(filename, data, theme_name)
    thm = THEMES[theme_name]
    show_per_symbol_hodl = (hodl_mode or "avg") == "all"

    # Wykres
    fig = go.Figure()
    all_values = []
    initial_balance = data.get('config', {}).get('initial_balance',
                         data.get('config', {}).get('balance_start', 1000))

    hodl_block = data.get('hodl') or {}
    hodl_data = hodl_block.get('crypto_ref_values', {}) or {}
    hodl_portfolio = list(hodl_block.get('portfolio_values') or [])
    if not hodl_portfolio and hodl_data:
        series = [v for v in hodl_data.values() if v]
        if series:
            n = min(len(s) for s in series)
            hodl_portfolio = [sum(s[i] for s in series) / len(series) for i in range(n)]

    if len(hodl_data) >= 2 and hodl_portfolio and hodl_portfolio[0] > 0:
        all_values.extend(downsample_series(hodl_portfolio))
        fig.add_trace(scatter_y(
            hodl_portfolio, mode='lines', name='HODL portfolio',
            line=dict(color=ACCENTS['orange'], width=2, dash='dash'), opacity=0.85
        ))
        if show_per_symbol_hodl:
            for symbol, values in hodl_data.items():
                if values and values[0] > 0:
                    norm_values = [v / values[0] * initial_balance for v in values]
                    fig.add_trace(scatter_y(
                        norm_values, mode='lines', name=f'HODL {symbol}',
                        line=dict(color=ACCENTS['orange'], width=1, dash='dot'), opacity=0.35
                    ))
    elif len(hodl_data) == 1:
        symbol, values = next(iter(hodl_data.items()))
        curve = hodl_portfolio if hodl_portfolio and hodl_portfolio[0] > 0 else (
            [v / values[0] * initial_balance for v in values] if values and values[0] > 0 else []
        )
        if curve:
            all_values.extend(downsample_series(curve))
            fig.add_trace(scatter_y(
                curve, mode='lines', name=f'HODL {symbol}',
                line=dict(color=ACCENTS['orange'], width=2, dash='dash'), opacity=0.85
            ))

    if 'agent' in data:
        agent_vals = data['agent'].get('portfolio_values', [])
        all_values.extend(downsample_series(agent_vals))
        fig.add_trace(scatter_y(
            agent_vals, mode='lines', name=legend_rl_label(data),
            line=dict(color=ACCENTS['red'], width=2, dash='solid'),
            fill='tozeroy', fillcolor='rgba(239, 68, 68, 0.05)'
        ))

    if 'model' in data:
        model_vals = data['model'].get('portfolio_values', [])
        all_values.extend(downsample_series(model_vals))
        fig.add_trace(scatter_y(
            model_vals, mode='lines', name=legend_mp_label(data),
            line=dict(color=ACCENTS['cyan'], width=2, dash='solid')
        ))

    y_range = [min(all_values) - 100, max(all_values) + 100] if all_values else None

    fig.update_layout(
        template=thm['chart_template'],
        title=dict(
            text="Wartość Portfela",
            font=dict(family='"Inter", sans-serif', size=18, color=thm['text'], weight='bold'),
            x=0.5,
            xanchor='center',
        ),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=thm['subtext'], family='"Inter", sans-serif'),
        hovermode="x unified",
        xaxis=dict(
            title=dict(text=AXIS_TITLE_X, font=dict(size=12, color=thm['subtext'])),
            showgrid=True, gridcolor=thm['border'],
        ),
        yaxis=dict(
            title=dict(text=AXIS_TITLE_Y, font=dict(size=12, color=thm['subtext'])),
            showgrid=True, gridcolor=thm['border'], range=y_range,
        ),
        legend=dict(
            orientation="v",
            yanchor="bottom",
            y=0.02,
            xanchor="left",
            x=0.01,
            bgcolor=thm['card'],
            bordercolor=thm['border'],
            borderwidth=1,
            font=dict(size=12, color=thm['text']),
        ),
        margin=dict(l=70, r=20, t=60, b=55),
        height=550,
    )

    chart_section = wrap_exportable_chart(
        fig,
        chart_key=filename,
        download_name=clean_filename(filename or 'simulation'),
        thm=thm,
    )

    # Dane do siatek
    sim_files = data.get('files', {})
    sim_config = data.get('config', {})
    metrics_dict = {}
    if 'agent' in data and 'metrics' in data['agent']:
        for k, v in data['agent']['metrics'].items(): metrics_dict[f"[RL] {k}"] = v
    if 'model' in data and 'metrics' in data['model']:
        for k, v in data['model']['metrics'].items(): metrics_dict[f"[MP] {k}"] = v
    if hodl_portfolio and hodl_portfolio[0] > 0 and len(hodl_data) >= 2:
        ew_ret = ((hodl_portfolio[-1] - hodl_portfolio[0]) / hodl_portfolio[0]) * 100
        metrics_dict["[HODL] Portfolio Return"] = f"{ew_ret:+.2f}%"
    if show_per_symbol_hodl and len(hodl_data) > 1:
        for symbol, values in hodl_data.items():
            if values and values[0] > 0:
                hodl_return = ((values[-1] - values[0]) / values[0]) * 100
                metrics_dict[f"[HODL] {symbol} Return"] = f"{hodl_return:+.2f}%"
    elif len(hodl_data) == 1:
        for symbol, values in hodl_data.items():
            if values and values[0] > 0:
                hodl_return = ((values[-1] - values[0]) / values[0]) * 100
                metrics_dict[f"[HODL] {symbol} Return"] = f"{hodl_return:+.2f}%"

    h_style = section_header_base_style.copy()
    h_style['color'] = thm['text']
    h_style['borderBottom'] = f'1px solid {thm["border"]}'

    files_section = html.Div([
        html.Div("Simulation Files", style=h_style),
        html.Div(create_info_cards(sim_files, theme_name, thm['success']), className='responsive-grid')
    ], style={'marginBottom': '40px'})

    metrics_section = html.Div([
        html.Div("Simulation Metrics", style=h_style),
        html.Div(create_info_cards(metrics_dict, theme_name, ACCENTS['blue']), className='responsive-grid')
    ], style={'marginBottom': '40px'})

    config_section = html.Div([
        html.Div("Simulation Config", style=h_style),
        html.Div(create_info_cards(sim_config, theme_name, thm['text']), className='responsive-grid')
    ], style={'marginBottom': '50px'})

    close_btn = html.Button("Zamknij", id={'type': 'close-tab-btn', 'index': filename}, style=close_btn_style, title="Zamknij zakładkę")

    tab_content = html.Div(
        [close_btn, html.Div(style={'clear': 'both'}), chart_section, metrics_section, files_section, config_section],
        style={'position': 'relative', 'marginTop': '10px', 'width': '100%'}
    )

    tab_style, selected_tab_style = get_tab_styles(theme_name)
    return dcc.Tab(label=clean_filename(filename), value=f'tab-{filename}', children=tab_content, style=tab_style, selected_style=selected_tab_style)


def get_cached_single_tab(filename, theme_name):
    # zawsze świeży dcc.Tab (dane i tak z DATA_CACHE)
    return build_single_tab(filename, theme_name)


def build_comparison_tab(open_files, theme_name):
    thm = THEMES[theme_name]
    fig = go.Figure()
    colors = [ACCENTS['red'], ACCENTS['cyan'], ACCENTS['blue'], '#A855F7', '#EC4899', '#22C55E', '#EAB308']
    color_idx = 0
    table_rows = []
    hodl_plotted = False

    for f in open_files:
        data = load_sim_data_cached(f)
        if not data: continue
        if data.get("chart_kind") == "price_windows":
            continue

        fname = clean_filename(f)
        c_color = colors[color_idx % len(colors)]
        color_idx += 1
        
        initial_balance = data.get('config', {}).get('initial_balance', 1000)
        
        hodl_block = data.get('hodl') or {}
        hodl_data = hodl_block.get('crypto_ref_values', {}) or {}
        hodl_portfolio = hodl_block.get('portfolio_values') or []
        if not hodl_portfolio and hodl_data:
            series = [v for v in hodl_data.values() if v]
            if series:
                n = min(len(s) for s in series)
                hodl_portfolio = [sum(s[i] for s in series) / len(series) for i in range(n)]
        hodl_return_str = "-"
        if hodl_portfolio and hodl_portfolio[0] > 0:
            if not hodl_plotted:
                fig.add_trace(scatter_y(
                    hodl_portfolio, mode='lines', name='HODL portfolio',
                    line=dict(color=ACCENTS['orange'], width=2, dash='dash'), opacity=0.7
                ))
                hodl_plotted = True
            ret = ((hodl_portfolio[-1] - hodl_portfolio[0]) / hodl_portfolio[0]) * 100
            hodl_return_str = f"{ret:+.2f}%"
        elif hodl_data:
            if not hodl_plotted:
                for symbol, values in hodl_data.items():
                    if values and values[0] > 0:
                        norm_values = [v / values[0] * initial_balance for v in values]
                        fig.add_trace(scatter_y(
                            norm_values, mode='lines', name=f'HODL {symbol}',
                            line=dict(color=ACCENTS['orange'], width=2, dash='dash'), opacity=0.7
                        ))
                hodl_plotted = True
            for symbol, values in hodl_data.items():
                if values and values[0] > 0:
                    ret = ((values[-1] - values[0]) / values[0]) * 100
                    hodl_return_str = f"{ret:+.2f}%"
                    break

        sym_tag = " | ".join(sim_symbols(data))
        rl_return_str = "-"
        if 'agent' in data:
            agent_vals = data['agent'].get('portfolio_values', [])
            if agent_vals:
                rl_name = f"RL: {fname}" + (f" ({sym_tag})" if sym_tag else "")
                fig.add_trace(scatter_y(
                    agent_vals, mode='lines', name=rl_name,
                    line=dict(color=c_color, width=2, dash='solid')
                ))
                ret = ((agent_vals[-1] - initial_balance) / initial_balance) * 100
                rl_return_str = f"{ret:+.2f}%"

        mp_return_str = "-"
        if 'model' in data:
            model_vals = data['model'].get('portfolio_values', [])
            if model_vals:
                mp_name = f"MP: {fname}" + (f" ({sym_tag})" if sym_tag else "")
                fig.add_trace(scatter_y(
                    model_vals, mode='lines', name=mp_name,
                    line=dict(color=c_color, width=2, dash='solid'), opacity=0.9
                ))
                ret = ((model_vals[-1] - initial_balance) / initial_balance) * 100
                mp_return_str = f"{ret:+.2f}%"

        agent_metrics = data.get('agent', {}).get('metrics', {})
        model_metrics = data.get('model', {}).get('metrics', {})
        trades = agent_metrics.get('Total Trades', model_metrics.get('Total Trades', '-'))
        win_rate = agent_metrics.get('Win Rate', model_metrics.get('Win Rate', '-'))

        rl_badge_bg = 'rgba(0, 229, 255, 0.1)' if '+' in rl_return_str else 'rgba(244, 63, 94, 0.1)'
        rl_color = ACCENTS['success'] if '+' in rl_return_str else ACCENTS['red']
        rl_element = html.Span(rl_return_str, className='badge', style={'backgroundColor': rl_badge_bg, 'color': rl_color})

        mp_badge_bg = 'rgba(14, 165, 233, 0.1)' if '+' in mp_return_str else 'rgba(244, 63, 94, 0.1)'
        mp_color = ACCENTS['cyan'] if '+' in mp_return_str else ACCENTS['red']
        mp_element = html.Span(mp_return_str, className='badge', style={'backgroundColor': mp_badge_bg, 'color': mp_color}) if mp_return_str != "-" else "-"

        table_rows.append(html.Tr([
            html.Td(fname, style={'fontWeight': 'bold', 'color': c_color, 'fontSize': '13px'}),
            html.Td(rl_element), html.Td(mp_element),
            html.Td(hodl_return_str, style={'color': ACCENTS['orange'], 'fontWeight': '500'}),
            html.Td(str(trades), style={'color': thm['text']}), html.Td(str(win_rate), style={'color': thm['text']})
        ]))

    fig.update_layout(
        template=thm['chart_template'],
        title=dict(
            text="Zestawienie Zbiorcze Portfeli",
            font=dict(family='"Inter", sans-serif', size=18, color=thm['text'], weight='bold'),
            x=0.5,
            xanchor='center',
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=thm['subtext'], family='"Inter", sans-serif'),
        hovermode="x unified",
        xaxis=dict(
            title=dict(text=AXIS_TITLE_X, font=dict(size=12, color=thm['subtext'])),
            showgrid=True, gridcolor=thm['border'],
        ),
        yaxis=dict(
            title=dict(text=AXIS_TITLE_Y, font=dict(size=12, color=thm['subtext'])),
            showgrid=True, gridcolor=thm['border'],
        ),
        legend=dict(
            orientation="v",
            yanchor="bottom",
            y=0.02,
            xanchor="left",
            x=0.01,
            bgcolor=thm['card'],
            bordercolor=thm['border'],
            borderwidth=1,
            font=dict(size=12, color=thm['text']),
        ),
        margin=dict(l=70, r=20, t=60, b=55),
        height=550,
    )

    chart_container = wrap_exportable_chart(
        fig,
        chart_key='__comparison__',
        download_name='comparison_portfolios',
        thm=thm,
    )

    table_header = html.Tr([html.Th("Symulacja"), html.Th("Agent RL ROI"), html.Th("Model MP ROI"), html.Th("HODL portfel"), html.Th("Transakcje"), html.Th("Win Rate")])
    table_container = html.Div([
        html.Div("Zestawienie Zbiorcze Wyników", style={'color': thm['text'], 'fontWeight': 'bold', 'fontSize': '20px', 'marginBottom': '20px', 'letterSpacing': '1px'}),
        html.Div(html.Table([html.Thead(table_header), html.Tbody(table_rows)], className='comparison-table', style={'--border-color': thm['border'], '--subtext-color': thm['subtext'], 'color': thm['text']}), className='table-wrapper')
    ], style={'backgroundColor': thm['card'], 'padding': '25px', 'borderRadius': '16px', 'border': f'1px solid {thm["border"]}', 'boxShadow': thm['shadow'], 'backdropFilter': 'blur(10px)'})

    tab_content = html.Div([chart_container, table_container], style={'marginTop': '20px'})
    tab_style, selected_tab_style = get_tab_styles(theme_name)
    
    # Podświetlamy zakładkę porównania specjalnym kolorem
    comp_selected_style = selected_tab_style.copy()
    comp_selected_style['borderTop'] = f'3px solid {ACCENTS["orange"]}'
    
    return dcc.Tab(label="📊 Analiza Porównawcza", value='tab-comparison', children=tab_content, style=tab_style, selected_style=comp_selected_style)


# --- LAYOUT ---
app.layout = html.Div([
    dcc.Store(id='open-files-store', data=[]),
    dcc.Store(id='theme-store', data='dark'),
    dcc.Store(id='file-filter-store', data='all'),
    dcc.Interval(id='auto-refresh', interval=60 * 1000, n_intervals=0),

    html.Button("☀/🌙 Theme", id='theme-toggle', n_clicks=0, style={
        'position': 'fixed', 'top': '20px', 'right': '20px', 'zIndex': 1000,
        'padding': '10px 15px', 'borderRadius': '8px', 'cursor': 'pointer',
        'border': '1px solid rgba(255,255,255,0.1)', 'backgroundColor': 'rgba(255,255,255,0.05)', 'color': '#FFF',
        'backdropFilter': 'blur(10px)', 'fontFamily': "'Inter', sans-serif", 'fontSize': '12px', 'fontWeight': 'bold',
        'boxShadow': '0 4px 15px rgba(0,0,0,0.2)', 'transition': 'all 0.3s'
    }),

    # PANEL BOCZNY
    html.Div(id='sidebar', children=[
        html.H2("Symulacje", id='sidebar-title',
                style={'marginBottom': '12px', 'textAlign': 'left', 'letterSpacing': '1px', 'fontSize': '22px', 'fontFamily': "'Inter', sans-serif", 'fontWeight': 'bold'}),
        html.Div(
            className='filter-tiles',
            children=[
                html.Button(
                    label,
                    id={'type': 'file-filter-btn', 'index': value},
                    n_clicks=0,
                    className='filter-tile',
                    style=filter_tile_style(value, 'all', 'dark'),
                )
                for value, label in FILE_FILTER_TILES
            ],
        ),
        html.Hr(id='sidebar-hr', style={'margin': '16px 0', 'borderTop': '1px solid #333', 'borderBottom': 'none'}),
        html.Div(id='file-list-container')
    ], style=sidebar_base_style),

    # GŁÓWNY PANEL
    html.Div(id='content', children=[
        html.H1("Wyniki Symulacji", id='main-title', style={
            'textAlign': 'left', 'paddingBottom': '15px', 'marginBottom': '25px', 'letterSpacing': '1px',
            'fontFamily': "'Inter', sans-serif", 'fontSize': '28px', 'fontWeight': 'bold'
        }),
        html.Div([
            dcc.Tabs(id='simulation-tabs', value='', className='custom-tabs', children=[])
        ], className='custom-tabs-container')
    ], style=content_base_style)
])

# --- CALLBACKI ---

@app.callback(
    Output('theme-store', 'data'),
    [Input('theme-toggle', 'n_clicks')],
    [State('theme-store', 'data')]
)
def toggle_theme(n, current):
    if n > 0:
        return 'light' if current == 'dark' else 'dark'
    return 'dark'


app.clientside_callback(
    """
    function(theme) {
        document.body.setAttribute('data-theme', theme || 'dark');
        return theme === 'light' ? 'Przełącz na ciemny motyw' : 'Przełącz na jasny motyw';
    }
    """,
    Output('theme-toggle', 'title'),
    Input('theme-store', 'data'),
)


# Eksport PNG pod pracę — NIE zmienia figury na ekranie
app.clientside_callback(
    """
    function(n_clicks, fig, fname) {
        if (!n_clicks || !fig) {
            return window.dash_clientside.no_update;
        }
        const paper = JSON.parse(JSON.stringify(fig));
        const isPriceWindows = (paper.data || []).some(function(t) {
            return t.legendgroup === 'window-boundary';
        });
        // Na szerokości strony A4 (~16 cm) font 16 px przy 1600 px szerokości
        // schodzi do ok. 4 pt. Wykres ceny do pracy ma napisy bliżej 10–11 pt.
        const tickSize = isPriceWindows ? 34 : 16;
        const axisTitleSize = isPriceWindows ? 36 : 20;
        const titleSize = isPriceWindows ? 42 : 32;
        const legendSize = isPriceWindows ? 26 : 20;
        const annSize = isPriceWindows ? 30 : 22;
        // Okno wsp?lne dla wszystkich symulacji (z nazw plik?w) trafia do tytu?u zestawienia.
        const winSet = {};
        (paper.data || []).forEach(function(t) {
            const wm = String(t.name || '').match(/seq_[A-Za-z]+\\d+_[^_\\s]+_(?:seed\\d+|mean)_(sequence_(B\\d+)_(B\\d+)|[AB]\\d+)/);
            if (wm) { winSet[wm[1]] = true; }
        });
        const winKeys = Object.keys(winSet);
        let windowSuffix = '';
        if (winKeys.length === 1) {
            const w = winKeys[0];
            const sq = w.match(/^sequence_(B\\d+)_(B\\d+)$/);
            windowSuffix = sq ? (' w oknach ' + sq[1] + '\u2013' + sq[2]) : (' w oknie ' + w);
        }
        (paper.data || []).forEach(function(t) {
            t.opacity = 1;
            if (t.marker) { t.marker.opacity = 1; }
            if (t.fill && t.fill !== 'none') {
                t.fill = 'none';
                delete t.fillcolor;
            }
            // HODL i granice okien zostają przerywane. Reszta ciągła.
            // Nazwy linii w eksporcie do pracy (ekran bez zmian):
            // "RL: seq_M2_thesis_seed396379_B3 (BTC | ETH)" -> "M2_396379: Agent RL (BTC | ETH)",
            // uśrednione po ziarnach (..._mean_...) -> "M2_mean: ...", R3/M3 -> "Agent RL z UL",
            // "MP: seq_M1_..." -> "M1_...: Model MP (BTC | ETH)".
            const seqMatch = String(t.name || '').match(
                /^(RL|MP):\\s*seq_([A-Za-z]+\\d+)_[^_\\s]+_(?:seed(\\d+)|(mean))_[^\\s(]*\\s*(\\(.*\\))?\\s*$/
            );
            if (seqMatch) {
                const arm = seqMatch[2];
                const tag = seqMatch[3] || seqMatch[4];
                const withUl = (arm === 'R3' || arm === 'M3');
                const kind = seqMatch[1] === 'MP' ? 'Model MP' : (withUl ? 'Agent RL z UL' : 'Agent RL');
                t.name = arm + '_' + tag + ': ' + kind + (seqMatch[5] ? ' ' + seqMatch[5] : '');
            }
            const nm = String(t.name || '');
            const isHodl = nm.toUpperCase().indexOf('HODL') >= 0;
            const isWindow = t.legendgroup === 'window-boundary' || nm.indexOf('Granice okien') === 0;
            t.line = Object.assign({}, t.line || {});
            t.line.dash = (isHodl || isWindow) ? 'dash' : 'solid';
            if (isWindow) {
                t.line.color = '#000000';
                t.line.width = isPriceWindows ? 3.5 : 3;
            } else if (isPriceWindows && !isHodl) {
                t.line.color = '#1E3A8A';
                t.line.width = 3.4;
            }
        });
        paper.layout = paper.layout || {};
        paper.layout.paper_bgcolor = '#ffffff';
        paper.layout.plot_bgcolor = '#ffffff';
        paper.layout.font = Object.assign({}, paper.layout.font || {}, {
            color: '#000000', size: tickSize, family: 'Inter, Arial, sans-serif'
        });
        let title = paper.layout.title;
        if (typeof title === 'string') { title = {text: title}; }
        title = title || {};
        if (windowSuffix && String(title.text || '').toLowerCase() === 'zestawienie zbiorcze portfeli') {
            title.text = 'Zestawienie zbiorcze portfeli' + windowSuffix;
        }
        title.font = Object.assign({}, title.font || {}, {
            color: '#000000', size: titleSize, family: 'Inter, Arial, sans-serif'
        });
        paper.layout.title = title;
        paper.layout.legend = paper.layout.legend || {};
        paper.layout.legend.font = Object.assign({}, paper.layout.legend.font || {}, {
            color: '#000000', size: legendSize
        });
        paper.layout.legend.bgcolor = '#ffffff';
        paper.layout.legend.bordercolor = '#333333';
        paper.layout.legend.borderwidth = 1;
        function axisTitle(existing, fallback) {
            if (typeof existing === 'string' && existing) { return existing; }
            if (existing && typeof existing === 'object' && existing.text) { return existing.text; }
            return fallback;
        }
        paper.layout.xaxis = paper.layout.xaxis || {};
        paper.layout.yaxis = paper.layout.yaxis || {};
        paper.layout.xaxis.title = {
            text: axisTitle(paper.layout.xaxis.title, 'Dni od startu okna'),
            font: {color: '#000000', size: axisTitleSize, family: 'Inter, Arial, sans-serif'}
        };
        paper.layout.yaxis.title = {
            text: axisTitle(paper.layout.yaxis.title, 'Wartość portfela (USDT)'),
            font: {color: '#000000', size: axisTitleSize, family: 'Inter, Arial, sans-serif'}
        };
        if (isPriceWindows) {
            paper.layout.xaxis.dtick = 'M2';
            paper.layout.xaxis.tickformat = '%m.%Y';
            paper.layout.xaxis.tickangle = 0;
            var yr = paper.layout.yaxis.range;
            if (yr && yr.length === 2 && yr[1] > yr[0]) {
                var span = yr[1] - yr[0];
                var raw = span / 6;
                var mag = Math.pow(10, Math.floor(Math.log10(raw)));
                var err = raw / mag;
                var nice = err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1;
                var step = nice * mag;
                var ticks = [];
                var start = Math.ceil(yr[0] / step) * step;
                for (var v = start; v <= yr[1] + step * 0.01; v += step) {
                    ticks.push(v);
                }
                paper.layout.yaxis.tickmode = 'array';
                paper.layout.yaxis.tickvals = ticks;
                paper.layout.yaxis.ticktext = ticks.map(function(val) {
                    return Math.round(val).toString().replace(/\\B(?=(\\d{3})+(?!\\d))/g, ' ');
                });
            }
        }
        ['xaxis', 'yaxis'].forEach(function(ax) {
            if (!paper.layout[ax]) { return; }
            paper.layout[ax].color = '#000000';
            paper.layout[ax].tickfont = Object.assign({}, paper.layout[ax].tickfont || {}, {
                color: '#000000', size: tickSize
            });
            paper.layout[ax].gridcolor = '#dddddd';
            paper.layout[ax].linecolor = '#000000';
            paper.layout[ax].zerolinecolor = '#888888';
            paper.layout[ax].linewidth = isPriceWindows ? 2 : 1;
        });
        var hasWindowLabels = (paper.layout.annotations || []).length > 0;
        paper.layout.margin = Object.assign({}, paper.layout.margin || {}, isPriceWindows
            ? {l: 165, r: 36, t: 96, b: 105}
            : {l: 90, r: 36, t: hasWindowLabels ? 78 : 70, b: 70}
        );
        (paper.layout.shapes || []).forEach(function(s) {
            s.line = Object.assign({}, s.line || {}, {color: '#000000', width: 2.5});
        });
        (paper.layout.annotations || []).forEach(function(a) {
            a.font = Object.assign({}, a.font || {}, {
                color: '#000000',
                size: annSize,
                family: 'Inter, Arial, sans-serif'
            });
            a.bgcolor = '#FDE68A';
            a.bordercolor = '#000000';
            a.borderwidth = isPriceWindows ? 2 : 1.5;
            a.borderpad = isPriceWindows ? 3 : 4;
        });
        const name = String(fname || 'simulation_chart')
            .replace(/\\.json(\\.gz)?$/i, '')
            .replace(/[^\\w\\-]+/g, '_');
        const Plotly = window.Plotly;
        if (!Plotly || !Plotly.toImage) {
            return 'brak Plotly.toImage';
        }
        var exportW = isPriceWindows ? 1700 : 1600;
        var exportH = isPriceWindows ? 980 : 900;
        paper.layout.width = exportW;
        paper.layout.height = exportH;
        return Plotly.toImage(paper, {
            format: 'png', width: exportW, height: exportH, scale: 2
        }).then(function(url) {
            const a = document.createElement('a');
            a.href = url;
            a.download = name + '_praca.png';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            return '';
        });
    }
    """,
    Output({'type': 'dl-paper-status', 'index': MATCH}, 'children'),
    Input({'type': 'dl-paper-btn', 'index': MATCH}, 'n_clicks'),
    State({'type': 'sim-chart', 'index': MATCH}, 'figure'),
    State({'type': 'dl-paper-name', 'index': MATCH}, 'data'),
    prevent_initial_call=True,
)


@app.callback(
    [Output('sidebar', 'style'), Output('content', 'style'),
     Output('sidebar-title', 'style'), Output('main-title', 'style'), Output('sidebar-hr', 'style')],
    [Input('theme-store', 'data')],
    [State('sidebar', 'style'), State('content', 'style')]
)
def update_global_styles(theme_name, sidebar_s, content_s):
    thm = THEMES[theme_name]
    new_sidebar = sidebar_s.copy()
    new_sidebar['backgroundColor'] = thm['sidebar']
    new_sidebar['borderRight'] = f"1px solid {thm['border']}"
    new_content = content_s.copy()
    new_content['backgroundColor'] = thm['bg']
    new_content['color'] = thm['text']

    title_style = {'color': thm['text'], 'marginBottom': '12px', 'textAlign': 'left', 'letterSpacing': '1px', 'fontSize': '22px', 'fontFamily': "'Inter', sans-serif", 'fontWeight': 'bold'}
    main_title_style = {'color': thm['text'], 'textAlign': 'left', 'borderBottom': f"1px solid {thm['border']}", 'paddingBottom': '15px', 'marginBottom': '25px', 'letterSpacing': '1px', 'fontFamily': "'Inter', sans-serif", 'fontSize': '28px', 'fontWeight': 'bold'}
    hr_style = {'borderTop': f"1px solid {thm['border']}", 'borderBottom': 'none', 'margin': '16px 0'}

    return new_sidebar, new_content, title_style, main_title_style, hr_style

@app.callback(
    Output('open-files-store', 'data'),
    [Input({'type': 'file-btn', 'index': ALL}, 'n_clicks'),
     Input({'type': 'close-tab-btn', 'index': ALL}, 'n_clicks')],
    [State('open-files-store', 'data')],
    prevent_initial_call=True,
)
def update_open_files(file_clicks, close_clicks, open_files):
    """Szybki callback — tylko lista otwartych plików (bez budowania wykresów)."""
    trigger = ctx.triggered_id
    if not trigger:
        return dash.no_update

    open_files = list(open_files or [])

    if isinstance(trigger, dict) and trigger.get('type') == 'file-btn':
        if not any((c or 0) > 0 for c in (file_clicks or [])):
            return dash.no_update
        filename = trigger['index']
        if filename not in open_files:
            open_files.append(filename)
        return open_files

    if isinstance(trigger, dict) and trigger.get('type') == 'close-tab-btn':
        filename = trigger['index']
        if filename in open_files:
            open_files.remove(filename)
        return open_files

    return dash.no_update


@app.callback(
    [Output('simulation-tabs', 'children'),
     Output('simulation-tabs', 'value')],
    [Input('open-files-store', 'data'),
     Input('theme-store', 'data')],
    [State('simulation-tabs', 'value')],
)
def render_tabs(open_files, theme_name, current_value):
    """Buduje zakładki; dane JSON z DATA_CACHE (szybkie), komponenty zawsze świeże."""
    open_files = list(open_files or [])
    theme_name = theme_name or 'dark'

    if not open_files:
        return [], ''

    try:
        new_tabs = []
        portfolio_files = []
        for f in open_files:
            cached = load_sim_data_cached(f) or {}
            if cached.get("chart_kind") != "price_windows":
                portfolio_files.append(f)
        if len(portfolio_files) > 1:
            new_tabs.append(build_comparison_tab(portfolio_files, theme_name))

        for f in open_files:
            tab = build_single_tab(f, theme_name)
            if tab:
                new_tabs.append(tab)

        trigger = ctx.triggered_id
        if trigger == 'open-files-store' and open_files:
            next_val = f'tab-{open_files[-1]}'
        elif current_value:
            next_val = current_value
        else:
            next_val = f'tab-{open_files[-1]}' if open_files else ''

        valid_vals = [t.value for t in new_tabs]
        if next_val not in valid_vals:
            next_val = valid_vals[-1] if valid_vals else ''

        return new_tabs, next_val
    except Exception as e:
        import traceback
        print(f"[simulation_viewer] render_tabs ERROR: {e}")
        traceback.print_exc()
        # nie kasuj istniejących zakładek przy błędzie porównania
        return dash.no_update, dash.no_update


@app.callback(
    Output('file-filter-store', 'data'),
    Input({'type': 'file-filter-btn', 'index': ALL}, 'n_clicks'),
    prevent_initial_call=True,
)
def store_file_filter(_n_clicks):
    trigger = ctx.triggered_id
    if isinstance(trigger, dict) and trigger.get('type') == 'file-filter-btn':
        return trigger.get('index') or 'all'
    return dash.no_update


@app.callback(
    Output({'type': 'file-filter-btn', 'index': ALL}, 'style'),
    [Input('file-filter-store', 'data'), Input('theme-store', 'data')],
)
def style_file_filter_tiles(selected, theme_name):
    return [filter_tile_style(value, selected, theme_name) for value, _ in FILE_FILTER_TILES]


@app.callback(
    Output('file-list-container', 'children'),
    [Input('auto-refresh', 'n_intervals'),
     Input('theme-store', 'data'),
     Input('file-filter-store', 'data'),
     Input('open-files-store', 'data')],
)
def update_sidebar_files(n, theme_name, file_filter, open_files):
    grouped_files = get_simulation_files_grouped(file_filter or 'all')
    thm = THEMES[theme_name or 'dark']
    open_files = open_files or []

    if not grouped_files:
        return html.Div("Brak plików", style={'color': thm['subtext'], 'textAlign': 'center'})

    components = []
    for version, files in grouped_files.items():
        btns = []
        for f in files:
            style = file_btn_base_style.copy()
            style['color'] = thm['text']
            style['border'] = f"1px solid {thm['border']}"
            rl = is_rltrain_sim(f)
            if str(f).startswith("price_"):
                prefix = "📈 "
            elif rl:
                prefix = "🤖 "
            else:
                prefix = "📄 "
            if rl:
                style['borderLeft'] = f"4px solid {ACCENTS['orange']}"
            if f in open_files:
                style['border'] = f"1px solid {ACCENTS['blue']}"
                style['borderLeft'] = f"4px solid {ACCENTS['blue']}"
                style['backgroundColor'] = 'rgba(0, 229, 255, 0.1)'
                style['fontWeight'] = 'bold'

            btns.append(html.Button(
                f"{prefix}{clean_filename(f)}",
                id={'type': 'file-btn', 'index': f},
                n_clicks=0,
                style=style,
            ))

        summary = html.Summary(version, style={
            'fontWeight': 'bold', 'color': thm['text'], 'cursor': 'pointer',
            'padding': '12px 0', 'borderBottom': f'1px solid {thm["border"]}',
            'fontSize': '14px', 'letterSpacing': '1px', 'textTransform': 'uppercase'
        })
        details = html.Details([summary, html.Div(btns, style={'paddingTop': '12px'})], open=True)
        components.append(details)

    return components

if __name__ == '__main__':
    app.run(debug=False, port=8052)
