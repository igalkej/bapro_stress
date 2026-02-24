"""
BAPRO Financial Stress Dashboard.

Tab 1 — Entrenamiento:
  - Training chart: FSI actual + out-of-sample val/test predictions
  - Shaded train/val/test regions
  - Per-day article viewer (all headlines for selected date)
  - Model metrics panel (MAE/RMSE primary, MAPE when available)
  - EDA panels: FSI distribution, articles/day, correlation,
    FSI components, Optuna trials, loss curves

Tab 2 — Predicciones:
  - FSI history chart with 1D/1W/1M/1Y/5Y/MAX range selector
  - Daily stress score gauge (from daily_predictions table)

Run:
    python dashboard/app.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import os

import numpy as np
import pandas as pd
from dash import Dash, Input, Output, State, callback_context, dcc, html
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sentence_transformers import SentenceTransformer
from sqlalchemy import text

from config import ARTIFACTS_DIR, EMBEDDING_MODEL, TIDE_MODEL_PATH
from db.connection import get_engine

# ---------------------------------------------------------------------------
# Globals / lazy loading
# ---------------------------------------------------------------------------
_engine = None
_tide_model = None
_st_model = None
_metadata = None


def get_db_engine():
    global _engine
    if _engine is None:
        _engine = get_engine()
    return _engine


def load_artifacts():
    global _tide_model, _metadata
    meta_path = os.path.join(ARTIFACTS_DIR, "metadata.json")

    if _metadata is None and os.path.exists(meta_path):
        with open(meta_path) as f:
            _metadata = json.load(f)

    if _tide_model is None and os.path.exists(TIDE_MODEL_PATH):
        try:
            import torch
            _orig_load = torch.load
            torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": False})
            from darts.models import TiDEModel
            _tide_model = TiDEModel.load(TIDE_MODEL_PATH)
            torch.load = _orig_load
        except Exception:
            pass

    return _tide_model, _metadata


def get_st_model():
    global _st_model
    if _st_model is None:
        _st_model = SentenceTransformer(EMBEDDING_MODEL)
    return _st_model


# ---------------------------------------------------------------------------
# Data helpers — Entrenamiento tab
# ---------------------------------------------------------------------------

def fetch_training_data():
    """Load FSI actual series and out-of-sample training predictions."""
    engine = get_db_engine()
    with engine.connect() as conn:
        fsi_rows = conn.execute(
            text("SELECT date, fsi_value FROM fsi_target ORDER BY date")
        ).fetchall()
        pred_rows = conn.execute(
            text(
                "SELECT date, fsi_actual, fsi_pred, split "
                "FROM training_predictions ORDER BY date"
            )
        ).fetchall()

    fsi_list = [{"date": str(r[0])[:10], "fsi_value": float(r[1])} for r in fsi_rows]
    pred_list = [
        {
            "date": str(r[0])[:10],
            "fsi_actual": float(r[1]) if r[1] is not None else None,
            "fsi_pred": float(r[2]),
            "split": r[3],
        }
        for r in pred_rows
    ]
    pred_dates = sorted({p["date"] for p in pred_list})
    return {"fsi": fsi_list, "predictions": pred_list, "pred_dates": pred_dates}


def fetch_articles_for_date(date_str):
    """Return all articles for a given date, ordered by source."""
    engine = get_db_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT headline, source, url FROM articles "
                "WHERE date = :d ORDER BY source, headline"
            ),
            {"d": date_str},
        ).fetchall()
    return [{"headline": r[0] or "", "source": r[1] or "", "url": r[2] or ""} for r in rows]


def fetch_article_counts():
    """Return article count and mean FSI per day (for EDA)."""
    engine = get_db_engine()
    with engine.connect() as conn:
        cnt_rows = conn.execute(
            text("SELECT date, COUNT(*) FROM articles GROUP BY date ORDER BY date")
        ).fetchall()
        fsi_rows = conn.execute(
            text("SELECT date, fsi_value FROM fsi_target ORDER BY date")
        ).fetchall()
    cnt_df = pd.DataFrame(cnt_rows, columns=["date", "count"])
    cnt_df["date"] = pd.to_datetime(cnt_df["date"].astype(str).str[:10])
    fsi_df = pd.DataFrame(fsi_rows, columns=["date", "fsi_value"])
    fsi_df["date"] = pd.to_datetime(fsi_df["date"].astype(str).str[:10])
    return cnt_df, fsi_df


def fetch_daily_tones():
    """Return mean GDELT tone and FSI per day for correlation EDA."""
    engine = get_db_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT a.date, AVG(a.gdelt_tone) AS mean_tone, f.fsi_value "
                "FROM articles a "
                "JOIN fsi_target f ON f.date = a.date "
                "WHERE a.gdelt_tone IS NOT NULL "
                "GROUP BY a.date, f.fsi_value ORDER BY a.date"
            )
        ).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["date", "mean_tone", "fsi_value"])
    df["date"] = pd.to_datetime(df["date"].astype(str).str[:10])
    return df


def fetch_fsi_components():
    """Return normalised FSI components from fsi_components table."""
    engine = get_db_engine()
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT date, merv_vol, argt_spread, usd_ars, emb_spread "
                    "FROM fsi_components ORDER BY date"
                )
            ).fetchall()
    except Exception:
        return pd.DataFrame()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["date", "merv_vol", "argt_spread", "usd_ars", "emb_spread"])
    df["date"] = pd.to_datetime(df["date"].astype(str).str[:10])
    return df


def fetch_daily_predictions():
    """Load daily predictions and full FSI series for the Predicciones tab."""
    engine = get_db_engine()
    with engine.connect() as conn:
        fsi_rows = conn.execute(
            text("SELECT date, fsi_value FROM fsi_target ORDER BY date")
        ).fetchall()
        pred_rows = conn.execute(
            text("SELECT date, fsi_pred, model_version FROM daily_predictions ORDER BY date")
        ).fetchall()
    fsi_list = [{"date": str(r[0])[:10], "fsi_value": float(r[1])} for r in fsi_rows]
    pred_list = [
        {"date": str(r[0])[:10], "fsi_pred": float(r[1]), "model_version": str(r[2] or "")}
        for r in pred_rows
    ]
    return {"fsi": fsi_list, "daily_preds": pred_list}


def fetch_optuna_results():
    """Return Optuna trial results from optuna_trials table. None if unavailable."""
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT trial_number, rank_val, mape_val, mape_test, "
                    "is_production, hyperparams "
                    "FROM optuna_trials ORDER BY rank_val"
                )
            ).fetchall()
        if not rows:
            return None
        df = pd.DataFrame(
            rows,
            columns=["trial_number", "rank_val", "mape_val", "mape_test",
                     "is_production", "hyperparams"],
        )
        return df
    except Exception:
        return None


def load_loss_curve(rank):
    """Load loss curve CSV for a trial rank. None if unavailable."""
    try:
        csv_path = Path(ARTIFACTS_DIR) / f"trial_{rank}" / "metrics.csv"
        if not csv_path.exists():
            return None
        return pd.read_csv(csv_path)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

app = Dash(__name__, title="BAPRO Financial Stress")

app.layout = html.Div(
    style={"fontFamily": "Arial, sans-serif", "backgroundColor": "#0d1117", "minHeight": "100vh"},
    children=[
        html.Div(
            style={"padding": "16px 32px", "borderBottom": "1px solid #30363d"},
            children=[
                html.H2(
                    "BAPRO Financial Stress System",
                    style={"color": "#e6edf3", "margin": 0, "fontSize": "22px"},
                ),
                html.P(
                    "Argentina sovereign credit stress — real-time FSI",
                    style={"color": "#8b949e", "margin": "4px 0 0", "fontSize": "13px"},
                ),
            ],
        ),
        dcc.Tabs(
            id="tabs",
            value="tab-history",
            style={"backgroundColor": "#161b22"},
            children=[
                dcc.Tab(
                    label="Entrenamiento",
                    value="tab-history",
                    style={"color": "#8b949e", "backgroundColor": "#161b22"},
                    selected_style={"color": "#e6edf3", "backgroundColor": "#0d1117", "borderTop": "2px solid #388bfd"},
                ),
                dcc.Tab(
                    label="Predicciones",
                    value="tab-predict",
                    style={"color": "#8b949e", "backgroundColor": "#161b22"},
                    selected_style={"color": "#e6edf3", "backgroundColor": "#0d1117", "borderTop": "2px solid #388bfd"},
                ),
            ],
        ),
        html.Div(id="tab-content", style={"padding": "24px 32px"}),
        dcc.Store(id="selected-date"),
        dcc.Store(id="history-store"),
    ],
)


# ---------------------------------------------------------------------------
# Callbacks — load history data on page load
# ---------------------------------------------------------------------------

@app.callback(Output("history-store", "data"), Input("tabs", "value"))
def load_history_store(_):
    return fetch_training_data()


# ---------------------------------------------------------------------------
# Callbacks — render tab content
# ---------------------------------------------------------------------------

@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    if tab == "tab-history":
        return _history_layout()
    return _predict_layout()


_DARK_PANEL = {
    "backgroundColor": "#161b22",
    "borderRadius": "8px",
    "padding": "16px",
    "border": "1px solid #21262d",
}
_SECTION_TITLE = {"color": "#8b949e", "fontSize": "11px",
                  "textTransform": "uppercase", "marginBottom": "8px", "marginTop": 0}
_TAB_STYLE = {"color": "#8b949e", "backgroundColor": "#161b22", "fontSize": "12px"}
_TAB_SELECTED = {"color": "#e6edf3", "backgroundColor": "#0d1117",
                 "borderTop": "2px solid #388bfd", "fontSize": "12px"}


def _history_layout():
    return html.Div(children=[
        # --- 1. Training time-series chart ---
        dcc.Graph(id="ts-chart", style={"height": "420px"}),

        # --- 2. Date selector (for article viewer) ---
        html.Div(
            style={"display": "flex", "gap": "12px", "alignItems": "center",
                   "marginTop": "8px", "marginBottom": "20px"},
            children=[
                html.Label("Fecha:", style={"color": "#8b949e", "fontSize": "13px"}),
                dcc.Dropdown(
                    id="date-dropdown",
                    options=[],
                    placeholder="Seleccionar fecha…",
                    style={"width": "200px", "backgroundColor": "#161b22",
                           "color": "#e6edf3", "border": "1px solid #30363d"},
                    className="dark-dropdown",
                ),
            ],
        ),

        # --- 3. Articles viewer | Metrics ---
        html.Div(
            style={"display": "flex", "gap": "24px", "marginBottom": "32px"},
            children=[
                html.Div(
                    style={**_DARK_PANEL, "flex": "1.5"},
                    children=[
                        html.P("Noticias del dia", style=_SECTION_TITLE),
                        html.Div(id="doc-viewer",
                                 style={"maxHeight": "380px", "overflowY": "auto"}),
                    ],
                ),
                html.Div(
                    style={**_DARK_PANEL, "flex": "1"},
                    children=[
                        html.P("Metricas del modelo", style=_SECTION_TITLE),
                        html.Div(id="model-quality-content"),
                    ],
                ),
            ],
        ),

        # --- 4. EDA section ---
        html.P("Analisis Exploratorio del Entrenamiento",
               style={**_SECTION_TITLE, "fontSize": "13px", "marginBottom": "12px"}),
        dcc.Tabs(
            id="eda-tabs",
            value="eda-dist",
            style={"backgroundColor": "#161b22"},
            children=[
                dcc.Tab(label="Distribucion FSI",    value="eda-dist",
                        style=_TAB_STYLE, selected_style=_TAB_SELECTED),
                dcc.Tab(label="Articulos / Dia",     value="eda-articles",
                        style=_TAB_STYLE, selected_style=_TAB_SELECTED),
                dcc.Tab(label="Correlacion",         value="eda-corr",
                        style=_TAB_STYLE, selected_style=_TAB_SELECTED),
                dcc.Tab(label="Componentes FSI",     value="eda-components",
                        style=_TAB_STYLE, selected_style=_TAB_SELECTED),
                dcc.Tab(label="Optuna Trials",       value="eda-optuna",
                        style=_TAB_STYLE, selected_style=_TAB_SELECTED),
                dcc.Tab(label="Curvas de Loss",      value="eda-loss",
                        style=_TAB_STYLE, selected_style=_TAB_SELECTED),
            ],
        ),
        html.Div(id="eda-content", style={"marginTop": "16px"}),
    ])


def _predict_layout():
    return html.Div(children=[
        # --- FSI history chart with range selector ---
        dcc.Graph(id="fsi-history-chart", style={"height": "380px"}),

        # --- Daily score (auto-loaded from daily_predictions) ---
        html.Div(id="predict-output", style={"marginTop": "24px"}),
    ])


# ---------------------------------------------------------------------------
# Callbacks — history tab
# ---------------------------------------------------------------------------

def _empty_fig(msg="Sin datos — ejecutar el pipeline primero"):
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font_color="#8b949e",
        annotations=[{"text": msg, "xref": "paper", "yref": "paper",
                       "x": 0.5, "y": 0.5, "showarrow": False,
                       "font": {"size": 14, "color": "#8b949e"}}],
    )
    return fig


@app.callback(
    Output("ts-chart", "figure"),
    Output("date-dropdown", "options"),
    Output("date-dropdown", "value"),
    Input("history-store", "data"),
)
def render_ts_chart(data):
    if not data or not data.get("fsi"):
        return _empty_fig(), [], None

    fsi_df = pd.DataFrame(data["fsi"])
    fsi_df["date"] = pd.to_datetime(fsi_df["date"])

    pred_df = pd.DataFrame(data.get("predictions", []))
    has_preds = not pred_df.empty

    _, metadata = load_artifacts()

    fig = go.Figure()

    # --- Shaded regions (train / val / test) ---
    if metadata and len(fsi_df) > 0:
        n = len(fsi_df)
        train_size = metadata.get("train_samples", int(n * 0.70))
        val_size   = metadata.get("val_samples",   int(n * 0.15))

        t_end = fsi_df["date"].iloc[min(train_size - 1, n - 1)]
        v_end = fsi_df["date"].iloc[min(train_size + val_size - 1, n - 1)]
        s_start = fsi_df["date"].iloc[0]
        s_end   = fsi_df["date"].iloc[-1]

        for x0, x1, color, label in [
            (s_start, t_end, "rgba(56,139,253,0.07)",  "TRAIN (in-sample)"),
            (t_end,   v_end, "rgba(240,180,41,0.10)",  "VAL"),
            (v_end,   s_end, "rgba(46,160,67,0.10)",   "TEST"),
        ]:
            fig.add_vrect(
                x0=x0, x1=x1, fillcolor=color,
                layer="below", line_width=0,
                annotation_text=label,
                annotation_position="top left",
                annotation_font={"size": 10, "color": "#8b949e"},
            )

    # --- FSI actual ---
    fig.add_trace(go.Scatter(
        x=fsi_df["date"], y=fsi_df["fsi_value"],
        name="FSI Real",
        line={"color": "#388bfd", "width": 2},
        mode="lines",
    ))

    # --- VAL predictions ---
    if has_preds:
        pred_df["date"] = pd.to_datetime(pred_df["date"])
        val_df = pred_df[pred_df["split"] == "val"]
        if not val_df.empty:
            fig.add_trace(go.Scatter(
                x=val_df["date"], y=val_df["fsi_pred"],
                name="Pred. VAL (out-of-sample)",
                line={"color": "#f0b429", "width": 2, "dash": "dot"},
                mode="lines+markers", marker={"size": 5, "symbol": "diamond"},
            ))

        # --- TEST predictions ---
        test_df = pred_df[pred_df["split"] == "test"]
        if not test_df.empty:
            fig.add_trace(go.Scatter(
                x=test_df["date"], y=test_df["fsi_pred"],
                name="Pred. TEST (out-of-sample)",
                line={"color": "#2ea043", "width": 2, "dash": "dot"},
                mode="lines+markers", marker={"size": 5, "symbol": "diamond"},
            ))

    fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        font_color="#e6edf3",
        legend={"font": {"color": "#e6edf3"}, "bgcolor": "#0d1117",
                "bordercolor": "#30363d", "borderwidth": 1},
        xaxis={"gridcolor": "#21262d", "title": "Fecha", "tickformat": "%b %Y"},
        yaxis={"gridcolor": "#21262d", "title": "FSI"},
        margin={"t": 16, "b": 40, "l": 60, "r": 20},
        hovermode="x unified",
    )

    pred_dates = data.get("pred_dates", [])
    options = [{"label": d, "value": d} for d in pred_dates]
    default_date = pred_dates[-1] if pred_dates else None
    return fig, options, default_date


@app.callback(
    Output("selected-date", "data"),
    Input("ts-chart", "clickData"),
    Input("date-dropdown", "value"),
)
def update_selected_date(click_data, dropdown_value):
    ctx = callback_context
    if not ctx.triggered:
        return dropdown_value

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if trigger_id == "ts-chart" and click_data:
        point = click_data["points"][0]
        return point["x"][:10]  # YYYY-MM-DD
    return dropdown_value


@app.callback(
    Output("doc-viewer", "children"),
    Input("selected-date", "data"),
    State("history-store", "data"),
)
def update_doc_viewer(selected_date, data):
    if not selected_date:
        return html.P("Seleccionar una fecha para ver las noticias.",
                      style={"color": "#8b949e", "fontSize": "13px"})

    articles = fetch_articles_for_date(selected_date)

    # Show prediction info for this date if available
    header_parts = [
        html.Span(selected_date, style={"color": "#388bfd", "fontWeight": "600"}),
        html.Span(f"  — {len(articles)} articulo(s)",
                  style={"color": "#8b949e", "fontSize": "12px"}),
    ]
    if data and data.get("predictions"):
        pred_df = pd.DataFrame(data["predictions"])
        row = pred_df[pred_df["date"] == selected_date]
        if not row.empty:
            r = row.iloc[0]
            pred_val = r.get("fsi_pred")
            actual_val = r.get("fsi_actual")
            split_label = r.get("split", "").upper()
            if pred_val is not None:
                header_parts.append(
                    html.Span(
                        f"  |  FSI pred: {pred_val:.3f}"
                        + (f"  actual: {actual_val:.3f}" if actual_val is not None else "")
                        + f"  [{split_label}]",
                        style={"color": "#8b949e", "fontSize": "12px"},
                    )
                )

    if not articles:
        return html.Div([
            html.Div(header_parts, style={"marginBottom": "8px"}),
            html.P("Sin noticias en la DB para esta fecha.",
                   style={"color": "#8b949e", "fontSize": "12px"}),
        ])

    cards = []
    for art in articles:
        cards.append(html.Div(
            style={
                "borderBottom": "1px solid #21262d",
                "paddingBottom": "10px",
                "marginBottom": "10px",
            },
            children=[
                html.Div([
                    html.Span(art["source"], style={
                        "backgroundColor": "#21262d",
                        "color": "#8b949e",
                        "fontSize": "10px",
                        "padding": "2px 6px",
                        "borderRadius": "3px",
                        "marginRight": "8px",
                    }),
                    html.Span(art["headline"], style={
                        "color": "#e6edf3",
                        "fontSize": "12px",
                    }),
                ]),
                html.A(
                    art["url"],
                    href=art["url"],
                    target="_blank",
                    style={"color": "#388bfd", "fontSize": "10px",
                           "textDecoration": "none", "display": "block",
                           "marginTop": "4px", "wordBreak": "break-all"},
                ) if art["url"] else None,
            ],
        ))

    return html.Div([
        html.Div(header_parts, style={"marginBottom": "12px"}),
        html.Div(cards),
    ])


@app.callback(
    Output("model-quality-content", "children"),
    Input("history-store", "data"),
)
def render_model_quality(_):
    _, metadata = load_artifacts()

    if metadata is None:
        return html.P(
            "Sin modelo. Ejecutar training/train.py primero.",
            style={"color": "#8b949e", "fontSize": "13px"},
        )

    hs = {"padding": "5px 10px", "color": "#388bfd", "fontSize": "11px",
          "fontWeight": "600", "borderBottom": "2px solid #30363d",
          "textTransform": "uppercase"}

    def _td(txt, bold=False, color=None):
        return html.Td(txt, style={
            "padding": "5px 10px",
            "color": color or ("#e6edf3" if bold else "#8b949e"),
            "fontWeight": "600" if bold else "normal",
            "fontSize": "12px",
            "borderBottom": "1px solid #21262d",
        })

    def _fmt(val, decimals=4):
        return f"{val:.{decimals}f}" if isinstance(val, float) else "—"

    # Primary: MAPE (available after ML-02); Secondary: MAE / RMSE
    has_mape = "mape_val_best" in metadata or "mape_test_prod" in metadata
    rows = []
    if has_mape:
        rows.append(html.Tr([
            _td("VAL (best trial)", bold=True),
            _td(_fmt(metadata.get("mape_val_best")), color="#f0b429"),
            _td("—"), _td("—"),
            _td(str(metadata.get("val_samples", "—"))),
        ]))
        rows.append(html.Tr([
            _td("TEST (prod model)", bold=True),
            _td(_fmt(metadata.get("mape_test_prod")), color="#2ea043"),
            _td(_fmt(metadata.get("mae_test_prod"))),
            _td(_fmt(metadata.get("rmse_test_prod"))),
            _td(str(metadata.get("test_samples", "—"))),
        ]))
    else:
        # Legacy format (pre-Optuna)
        for split, mk, rk in [
            ("Train", "train_mae", "train_rmse"),
            ("Val",   "val_mae",   "val_rmse"),
            ("Test",  "test_mae",  "test_rmse"),
        ]:
            rows.append(html.Tr([
                _td(split, bold=True),
                _td("—"),
                _td(_fmt(metadata.get(mk))),
                _td(_fmt(metadata.get(rk))),
                _td(str(metadata.get(f"{split.lower()}_samples", "—"))),
            ]))

    metrics_table = html.Table(
        style={"width": "100%", "borderCollapse": "collapse", "marginBottom": "16px"},
        children=[
            html.Thead(html.Tr([
                html.Th("Split",   style=hs),
                html.Th("MAPE",    style={**hs, "color": "#f0b429"}),
                html.Th("MAE",     style=hs),
                html.Th("RMSE",    style=hs),
                html.Th("N",       style=hs),
            ])),
            html.Tbody(rows),
        ],
    )

    # Config table (use best_params if available, else tide_params)
    bp = metadata.get("best_params", metadata.get("tide_params", {}))
    config_items = [
        ("Modelo",          "Darts TiDE"),
        ("Input chunk",     str(bp.get("input_chunk_length", "—"))),
        ("Hidden size",     str(bp.get("hidden_size", "—"))),
        ("Encoder layers",  str(bp.get("num_encoder_layers", "—"))),
        ("Dropout",         str(bp.get("dropout", "—"))),
        ("LR",              str(bp.get("lr", "—"))),
        ("Epochs",          str(bp.get("n_epochs", "—"))),
        ("Covariables",     "384-dim embeddings"),
        ("Version",         str(metadata.get("model_version", "—"))[:15]),
    ]
    config_table = html.Table(
        style={"width": "100%", "borderCollapse": "collapse"},
        children=[html.Tbody([
            html.Tr([
                html.Td(k, style={"color": "#8b949e", "fontSize": "11px",
                                  "padding": "3px 10px 3px 0"}),
                html.Td(v, style={"color": "#e6edf3", "fontSize": "11px",
                                  "padding": "3px 0"}),
            ])
            for k, v in config_items
        ])],
    )

    return html.Div([
        html.P("Metricas", style=_SECTION_TITLE),
        metrics_table,
        html.P("Configuracion", style=_SECTION_TITLE),
        config_table,
    ])


# ---------------------------------------------------------------------------
# Callbacks — EDA panels
# ---------------------------------------------------------------------------

def _no_data_fig(msg):
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor="#161b22", plot_bgcolor="#161b22",
        font_color="#8b949e", margin={"t": 20, "b": 20, "l": 20, "r": 20},
        annotations=[{"text": msg, "xref": "paper", "yref": "paper",
                       "x": 0.5, "y": 0.5, "showarrow": False,
                       "font": {"size": 13, "color": "#8b949e"}}],
    )
    return fig


def _fig_layout(fig, title=None, height=320):
    upd = dict(
        paper_bgcolor="#161b22", plot_bgcolor="#21262d",
        font_color="#e6edf3",
        xaxis={"gridcolor": "#30363d"},
        yaxis={"gridcolor": "#30363d"},
        margin={"t": 36 if title else 20, "b": 40, "l": 60, "r": 20},
        hovermode="x unified",
        height=height,
    )
    if title:
        upd["title"] = {"text": title, "font": {"size": 13, "color": "#8b949e"}}
    fig.update_layout(**upd)
    return fig


@app.callback(Output("eda-content", "children"), Input("eda-tabs", "value"))
def render_eda_content(tab):
    # ── 4a. FSI Distribution ─────────────────────────────────────────────────
    if tab == "eda-dist":
        data = fetch_training_data()
        if not data["fsi"]:
            return dcc.Graph(figure=_no_data_fig("Sin datos FSI"))
        fsi_df = pd.DataFrame(data["fsi"])
        vals = fsi_df["fsi_value"].dropna().values

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=vals, nbinsx=30, name="Distribucion",
            marker_color="#388bfd", opacity=0.75,
        ))
        # KDE overlay (manual Gaussian kernel)
        x_range = np.linspace(vals.min(), vals.max(), 200)
        bw = 1.06 * vals.std() * len(vals) ** (-0.2)
        kde = np.mean(
            np.exp(-0.5 * ((x_range[:, None] - vals[None, :]) / bw) ** 2), axis=1
        ) / (bw * np.sqrt(2 * np.pi))
        kde_scaled = kde * len(vals) * (vals.max() - vals.min()) / 30
        fig.add_trace(go.Scatter(
            x=x_range, y=kde_scaled, name="KDE",
            line={"color": "#f0b429", "width": 2}, mode="lines",
        ))
        # Percentile annotations
        for pct, color in [(10, "#8b949e"), (50, "#e6edf3"), (90, "#f78166")]:
            v = np.percentile(vals, pct)
            fig.add_vline(x=v, line_dash="dot", line_color=color,
                          annotation_text=f"p{pct}={v:.2f}",
                          annotation_font_size=10, annotation_font_color=color)
        _fig_layout(fig, "Distribucion del FSI")
        return dcc.Graph(figure=fig)

    # ── 4b. Articles per Day ─────────────────────────────────────────────────
    if tab == "eda-articles":
        cnt_df, fsi_df = fetch_article_counts()
        if cnt_df.empty:
            return dcc.Graph(figure=_no_data_fig("Sin articulos en la DB"))
        from plotly.subplots import make_subplots as _msp
        fig = _msp(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=cnt_df["date"], y=cnt_df["count"],
            name="Articulos/dia", marker_color="#388bfd", opacity=0.7,
        ), secondary_y=False)
        if not fsi_df.empty:
            fig.add_trace(go.Scatter(
                x=fsi_df["date"], y=fsi_df["fsi_value"],
                name="FSI Real", line={"color": "#f0b429", "width": 2},
                mode="lines",
            ), secondary_y=True)
        fig.update_layout(
            paper_bgcolor="#161b22", plot_bgcolor="#21262d",
            font_color="#e6edf3", height=320,
            margin={"t": 36, "b": 40, "l": 60, "r": 60},
            title={"text": "Articulos por dia vs FSI",
                   "font": {"size": 13, "color": "#8b949e"}},
        )
        fig.update_xaxes(gridcolor="#30363d")
        fig.update_yaxes(title_text="Articulos", gridcolor="#30363d", secondary_y=False)
        fig.update_yaxes(title_text="FSI", gridcolor="#30363d", secondary_y=True)
        return dcc.Graph(figure=fig)

    # ── 4c. Correlation FSI vs tone ──────────────────────────────────────────
    if tab == "eda-corr":
        tone_df = fetch_daily_tones()
        if tone_df.empty:
            return dcc.Graph(figure=_no_data_fig("Sin datos de tono GDELT"))
        x = tone_df["mean_tone"].values
        y = tone_df["fsi_value"].values
        # Linear regression
        m, b = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        corr = np.corrcoef(x, y)[0, 1]
        r2 = corr ** 2

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers",
            name="Dias",
            marker={"color": "#388bfd", "size": 6, "opacity": 0.7},
            text=tone_df["date"].dt.strftime("%Y-%m-%d"),
            hovertemplate="Fecha: %{text}<br>Tone: %{x:.2f}<br>FSI: %{y:.3f}",
        ))
        fig.add_trace(go.Scatter(
            x=x_line, y=m * x_line + b,
            name=f"Regresion (R²={r2:.3f})",
            line={"color": "#f78166", "width": 2, "dash": "dot"},
        ))
        _fig_layout(fig, "Correlacion: Tono GDELT vs FSI")
        fig.update_layout(
            xaxis_title="Tono GDELT medio diario",
            yaxis_title="FSI",
        )
        return dcc.Graph(figure=fig)

    # ── 4d. FSI Components ───────────────────────────────────────────────────
    if tab == "eda-components":
        comp_df = fetch_fsi_components()
        fsi_data = fetch_training_data()
        fsi_df = pd.DataFrame(fsi_data["fsi"])
        if not fsi_df.empty:
            fsi_df["date"] = pd.to_datetime(fsi_df["date"])
        if comp_df.empty:
            return dcc.Graph(figure=_no_data_fig(
                "Sin componentes FSI — ejecutar build_fsi_target.py"
            ))
        component_map = [
            ("merv_vol",    "Vol. MERV (30d)",     "#388bfd"),
            ("argt_spread", "Spread ARGT (inv.)",  "#f0b429"),
            ("usd_ars",     "USD/ARS",             "#f78166"),
            ("emb_spread",  "Spread EMB (inv.)",   "#a371f7"),
        ]
        fig = go.Figure()
        for col, label, color in component_map:
            fig.add_trace(go.Scatter(
                x=comp_df["date"], y=comp_df[col],
                name=label, line={"color": color, "width": 1.5},
                mode="lines", opacity=0.85,
            ))
        if not fsi_df.empty:
            fig.add_trace(go.Scatter(
                x=fsi_df["date"], y=fsi_df["fsi_value"],
                name="FSI (PCA)", line={"color": "#e6edf3", "width": 2.5},
                mode="lines",
            ))
        _fig_layout(fig, "Componentes individuales del FSI (z-score)", height=380)
        fig.update_layout(yaxis_title="Z-score")
        return dcc.Graph(figure=fig)

    # ── 4e. Optuna Trials ────────────────────────────────────────────────────
    if tab == "eda-optuna":
        optuna_df = fetch_optuna_results()
        if optuna_df is None:
            return html.Div(
                html.P(
                    "Datos Optuna no disponibles — ejecutar training/train.py con Optuna (ML-02).",
                    style={"color": "#8b949e", "fontSize": "13px"},
                ),
                style=_DARK_PANEL,
            )
        # Bar chart MAPE val + test
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f"Trial {r}" for r in optuna_df["trial_number"]],
            y=optuna_df["mape_val"],
            name="MAPE VAL",
            marker_color="#f0b429", opacity=0.85,
        ))
        if optuna_df["mape_test"].notna().any():
            fig.add_trace(go.Bar(
                x=[f"Trial {r}" for r in optuna_df["trial_number"]],
                y=optuna_df["mape_test"],
                name="MAPE TEST",
                marker_color="#2ea043", opacity=0.85,
            ))
        # Highlight production trial
        prod_idx = optuna_df[optuna_df["is_production"] == True].index
        if len(prod_idx):
            prod_trial = optuna_df.loc[prod_idx[0], "trial_number"]
            fig.add_annotation(
                x=f"Trial {prod_trial}", y=0, yref="paper",
                text="PROD", showarrow=False,
                font={"color": "#2ea043", "size": 10},
                yanchor="bottom",
            )
        _fig_layout(fig, "MAPE por trial Optuna")
        fig.update_layout(barmode="group", yaxis_title="MAPE")

        # Hyperparams table
        rows = []
        for _, r in optuna_df.iterrows():
            try:
                hp = json.loads(r["hyperparams"]) if isinstance(r["hyperparams"], str) else r["hyperparams"]
            except Exception:
                hp = {}
            is_prod = bool(r.get("is_production"))
            row_style = {"backgroundColor": "#0d2b0d"} if is_prod else {}
            rows.append(html.Tr(
                style=row_style,
                children=[
                    html.Td(str(int(r["trial_number"])),
                            style={"padding": "5px 10px", "color": "#388bfd",
                                   "fontSize": "12px"}),
                    html.Td(f"{r['mape_val']:.4f}" if r["mape_val"] is not None else "—",
                            style={"padding": "5px 10px", "color": "#f0b429",
                                   "fontSize": "12px"}),
                    html.Td(f"{r['mape_test']:.4f}" if r["mape_test"] is not None else "—",
                            style={"padding": "5px 10px", "color": "#2ea043",
                                   "fontSize": "12px"}),
                    html.Td("PROD" if is_prod else "",
                            style={"padding": "5px 10px", "color": "#2ea043",
                                   "fontSize": "11px", "fontWeight": "600"}),
                    html.Td(
                        ", ".join(f"{k}={v}" for k, v in hp.items()),
                        style={"padding": "5px 10px", "color": "#8b949e",
                               "fontSize": "11px"},
                    ),
                ],
            ))
        th_style = {"padding": "5px 10px", "color": "#388bfd", "fontSize": "11px",
                    "fontWeight": "600", "borderBottom": "2px solid #30363d",
                    "textTransform": "uppercase"}
        table = html.Table(
            style={"width": "100%", "borderCollapse": "collapse",
                   "marginTop": "16px"},
            children=[
                html.Thead(html.Tr([
                    html.Th("Trial", style=th_style),
                    html.Th("MAPE VAL", style=th_style),
                    html.Th("MAPE TEST", style=th_style),
                    html.Th("", style=th_style),
                    html.Th("Hiperparametros", style=th_style),
                ])),
                html.Tbody(rows),
            ],
        )
        return html.Div([dcc.Graph(figure=fig), table])

    # ── 4f. Loss Curves ──────────────────────────────────────────────────────
    if tab == "eda-loss":
        colors = ["#388bfd", "#f0b429", "#f78166", "#2ea043"]
        fig = go.Figure()
        found_any = False
        for rank in [1, 2]:
            df_loss = load_loss_curve(rank)
            if df_loss is None:
                continue
            found_any = True
            color = colors[(rank - 1) % len(colors)]
            # Try common column names from PyTorch Lightning CSV logger
            epoch_col = next(
                (c for c in df_loss.columns if "epoch" in c.lower()), None
            )
            train_col = next(
                (c for c in df_loss.columns
                 if "train" in c.lower() and "loss" in c.lower()), None
            )
            val_col = next(
                (c for c in df_loss.columns
                 if "val" in c.lower() and "loss" in c.lower()), None
            )
            if epoch_col is None:
                continue
            epoch_vals = df_loss[epoch_col].dropna()
            if train_col:
                fig.add_trace(go.Scatter(
                    x=epoch_vals, y=df_loss[train_col].dropna(),
                    name=f"Trial {rank} train",
                    line={"color": color, "width": 2},
                    mode="lines",
                ))
            if val_col:
                fig.add_trace(go.Scatter(
                    x=epoch_vals, y=df_loss[val_col].dropna(),
                    name=f"Trial {rank} val",
                    line={"color": color, "width": 2, "dash": "dot"},
                    mode="lines",
                ))
        if not found_any:
            return dcc.Graph(figure=_no_data_fig(
                "Curvas de loss no disponibles — ejecutar train.py con Optuna (ML-02)"
            ))
        _fig_layout(fig, "Curvas de loss por trial", height=360)
        fig.update_layout(xaxis_title="Epoch", yaxis_title="Loss")
        return dcc.Graph(figure=fig)

    return html.P("Tab no reconocido.", style={"color": "#8b949e"})


# ---------------------------------------------------------------------------
# Callbacks — prediction tab
# ---------------------------------------------------------------------------

@app.callback(
    Output("fsi-history-chart", "figure"),
    Input("tabs", "value"),
)
def render_fsi_history_chart(tab):
    if tab != "tab-predict":
        return go.Figure()

    data = fetch_daily_predictions()
    fsi_df = pd.DataFrame(data["fsi"])
    daily_df = pd.DataFrame(data["daily_preds"])

    if fsi_df.empty:
        return _empty_fig("Sin datos FSI — ejecutar build_fsi_target.py y seed_fsi.py")

    fsi_df["date"] = pd.to_datetime(fsi_df["date"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fsi_df["date"], y=fsi_df["fsi_value"],
        name="FSI Real",
        line={"color": "#388bfd", "width": 2},
        mode="lines",
        fill="tozeroy",
        fillcolor="rgba(56,139,253,0.07)",
    ))

    if not daily_df.empty:
        daily_df["date"] = pd.to_datetime(daily_df["date"])
        fig.add_trace(go.Scatter(
            x=daily_df["date"], y=daily_df["fsi_pred"],
            name="Pred. diaria",
            mode="markers",
            marker={"color": "#f0b429", "size": 7, "symbol": "circle",
                    "line": {"color": "#e6edf3", "width": 1}},
            hovertemplate="Fecha: %{x|%Y-%m-%d}<br>FSI pred: %{y:.3f}",
        ))

    fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        font_color="#e6edf3",
        legend={"font": {"color": "#e6edf3"}, "bgcolor": "#0d1117",
                "bordercolor": "#30363d", "borderwidth": 1},
        margin={"t": 20, "b": 40, "l": 60, "r": 20},
        hovermode="x unified",
        xaxis=dict(
            gridcolor="#21262d",
            title="Fecha",
            rangeselector=dict(
                bgcolor="#161b22",
                activecolor="#388bfd",
                font={"color": "#e6edf3", "size": 11},
                buttons=[
                    dict(count=1,  label="1D", step="day",  stepmode="backward"),
                    dict(count=7,  label="1S", step="day",  stepmode="backward"),
                    dict(count=1,  label="1M", step="month", stepmode="backward"),
                    dict(count=1,  label="1A", step="year",  stepmode="backward"),
                    dict(count=5,  label="5A", step="year",  stepmode="backward"),
                    dict(step="all", label="MAX"),
                ],
            ),
            rangeslider=dict(visible=False),
            type="date",
        ),
        yaxis={"gridcolor": "#21262d", "title": "FSI"},
    )
    return fig


def _build_gauge(score, lo, hi, pred_date):
    """Build the stress score gauge with 95% CI markers."""
    eps = 0.05
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": f"Score al {pred_date}", "font": {"color": "#e6edf3", "size": 13}},
        number={"font": {"color": "#e6edf3"}},
        gauge={
            "axis": {"range": [-3, 4], "tickcolor": "#8b949e"},
            "bar": {"color": "#388bfd"},
            "bgcolor": "#161b22",
            "bordercolor": "#30363d",
            "steps": [
                {"range": [-3, -1],  "color": "#1a7f37"},
                {"range": [-1,  1],  "color": "#9e6a03"},
                {"range": [ 1,  2.5],"color": "#bc4c00"},
                {"range": [ 2.5, 4], "color": "#a40e26"},
                {"range": [lo - eps, lo + eps], "color": "#e6edf3"},
                {"range": [hi - eps, hi + eps], "color": "#e6edf3"},
            ],
            "threshold": {"line": {"color": "#f78166", "width": 3},
                          "thickness": 0.75, "value": score},
        },
    ))
    gauge.update_layout(
        paper_bgcolor="#161b22", font_color="#e6edf3",
        height=300, margin={"t": 48, "b": 20, "l": 40, "r": 40},
    )
    gauge.add_annotation(
        text=f"95% CI: [{lo:.2f} \u2014 {hi:.2f}]",
        x=0.5, y=0.05, showarrow=False,
        font={"size": 12, "color": "#8b949e"},
        xref="paper", yref="paper",
    )
    if score >= 2.5:
        label, bg = "ALTO ESTRES",     "#a40e26"
    elif score >= 1.0:
        label, bg = "ESTRES ELEVADO",  "#bc4c00"
    elif score >= -1.0:
        label, bg = "NEUTRAL",         "#9e6a03"
    else:
        label, bg = "CALMA",           "#1a7f37"

    return html.Div([
        html.Div(
            f"{label}  ({score:.3f})  IC 95% [{lo:.2f} \u2014 {hi:.2f}]",
            style={"backgroundColor": bg, "color": "white",
                   "padding": "8px 20px", "borderRadius": "6px",
                   "display": "inline-block", "fontWeight": "bold",
                   "marginBottom": "16px", "fontSize": "14px"},
        ),
        dcc.Graph(figure=gauge, style={"height": "300px"}),
    ])


@app.callback(
    Output("predict-output", "children"),
    Input("tabs", "value"),
)
def load_daily_score(tab):
    if tab != "tab-predict":
        return ""

    engine = get_db_engine()
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT date, fsi_pred FROM daily_predictions "
                    "ORDER BY date DESC LIMIT 1"
                )
            ).fetchone()
    except Exception as exc:
        return html.P(f"Error al leer daily_predictions: {exc}",
                      style={"color": "#f78166", "fontSize": "13px"})

    if row is None:
        return html.Div(
            style=_DARK_PANEL,
            children=html.P(
                "Sin predicciones disponibles — ejecutar el pipeline diario.",
                style={"color": "#8b949e", "fontSize": "13px"},
            ),
        )

    pred_date = str(row[0])[:10]
    score     = float(row[1])

    _, metadata = load_artifacts()
    test_rmse   = (metadata or {}).get("test_rmse", 0.5)
    lo = score - 1.96 * test_rmse
    hi = score + 1.96 * test_rmse

    return _build_gauge(score, lo, hi, pred_date)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
