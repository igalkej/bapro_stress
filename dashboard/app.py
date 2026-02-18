"""
BAPRO Financial Stress Dashboard.

Tab 1 — Historical Results:
  - Dual-line time series: predicted vs actual stress index
  - Day selector (click on chart or use dropdown)
  - Document viewer for the selected date
  - XGBoost feature importances bar chart

Tab 2 — New Prediction:
  - Textarea for a new document
  - Score button → sentence-transformers + XGBoost → stress gauge

Run:
    python dashboard/app.py
"""
import sys
sys.path.insert(0, "/workspace")

import json
import os
import pickle

import numpy as np
import pandas as pd
from dash import Dash, Input, Output, State, callback_context, dcc, html
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sentence_transformers import SentenceTransformer
from sqlalchemy import text

from config import ARTIFACTS_DIR, EMBEDDING_MODEL
from db.connection import get_engine

# ---------------------------------------------------------------------------
# Globals / lazy loading
# ---------------------------------------------------------------------------
_engine = None
_pca = None
_xgb = None
_st_model = None
_metadata = None


def get_db_engine():
    global _engine
    if _engine is None:
        _engine = get_engine()
    return _engine


def load_artifacts():
    global _pca, _xgb, _metadata
    pca_path = os.path.join(ARTIFACTS_DIR, "pca.pkl")
    model_path = os.path.join(ARTIFACTS_DIR, "xgb_model.pkl")
    meta_path = os.path.join(ARTIFACTS_DIR, "metadata.json")

    if _pca is None and os.path.exists(pca_path):
        with open(pca_path, "rb") as f:
            _pca = pickle.load(f)
    if _xgb is None and os.path.exists(model_path):
        with open(model_path, "rb") as f:
            _xgb = pickle.load(f)
    if _metadata is None and os.path.exists(meta_path):
        with open(meta_path) as f:
            _metadata = json.load(f)

    return _pca, _xgb, _metadata


def get_st_model():
    global _st_model
    if _st_model is None:
        _st_model = SentenceTransformer(EMBEDDING_MODEL)
    return _st_model


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def fetch_history():
    engine = get_db_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT
                    d.doc_date,
                    d.content,
                    d.doc_type,
                    p.stress_score_pred,
                    s.stress_value
                FROM predictions p
                JOIN documents d   ON d.id = p.doc_id
                JOIN stress_index s ON s.index_date = d.doc_date
                ORDER BY d.doc_date
                """
            )
        ).fetchall()

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(
        [
            {
                "date": str(r.doc_date),
                "content": r.content,
                "doc_type": r.doc_type,
                "stress_pred": r.stress_score_pred,
                "stress_actual": r.stress_value,
            }
            for r in rows
        ]
    )


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
                    "Argentina sovereign credit stress — January 2024",
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
                    label="Historical Results",
                    value="tab-history",
                    style={"color": "#8b949e", "backgroundColor": "#161b22"},
                    selected_style={"color": "#e6edf3", "backgroundColor": "#0d1117", "borderTop": "2px solid #388bfd"},
                ),
                dcc.Tab(
                    label="New Prediction",
                    value="tab-predict",
                    style={"color": "#8b949e", "backgroundColor": "#161b22"},
                    selected_style={"color": "#e6edf3", "backgroundColor": "#0d1117", "borderTop": "2px solid #388bfd"},
                ),
            ],
        ),
        html.Div(id="tab-content", style={"padding": "24px 32px"}),
        # Store for selected date
        dcc.Store(id="selected-date"),
        # Store for historical data (loaded once)
        dcc.Store(id="history-store"),
    ],
)


# ---------------------------------------------------------------------------
# Callbacks — load history data on page load
# ---------------------------------------------------------------------------

@app.callback(Output("history-store", "data"), Input("tabs", "value"))
def load_history_store(_):
    df = fetch_history()
    if df.empty:
        return {"rows": [], "dates": []}
    return {"rows": df.to_dict("records"), "dates": df["date"].tolist()}


# ---------------------------------------------------------------------------
# Callbacks — render tab content
# ---------------------------------------------------------------------------

@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    if tab == "tab-history":
        return _history_layout()
    return _predict_layout()


def _history_layout():
    return html.Div(
        children=[
            # Time series chart
            dcc.Graph(id="ts-chart", style={"height": "360px"}),
            # Date selector
            html.Div(
                style={"display": "flex", "gap": "12px", "alignItems": "center", "marginTop": "8px"},
                children=[
                    html.Label("Select date:", style={"color": "#8b949e", "fontSize": "13px"}),
                    dcc.Dropdown(
                        id="date-dropdown",
                        options=[],
                        style={
                            "width": "200px",
                            "backgroundColor": "#161b22",
                            "color": "#e6edf3",
                            "border": "1px solid #30363d",
                        },
                        className="dark-dropdown",
                    ),
                ],
            ),
            # Bottom row: doc viewer | feature importances
            html.Div(
                style={"display": "flex", "gap": "24px", "marginTop": "24px"},
                children=[
                    # Document viewer
                    html.Div(
                        style={"flex": "1", "backgroundColor": "#161b22", "borderRadius": "8px", "padding": "16px"},
                        children=[
                            html.H4("Document", style={"color": "#e6edf3", "marginTop": 0, "fontSize": "14px"}),
                            html.Pre(
                                id="doc-viewer",
                                style={
                                    "color": "#8b949e",
                                    "fontSize": "11px",
                                    "whiteSpace": "pre-wrap",
                                    "maxHeight": "380px",
                                    "overflowY": "auto",
                                    "margin": 0,
                                },
                            ),
                        ],
                    ),
                    # Feature importances
                    html.Div(
                        style={"flex": "1", "backgroundColor": "#161b22", "borderRadius": "8px", "padding": "16px"},
                        children=[
                            html.H4("XGBoost Feature Importances", style={"color": "#e6edf3", "marginTop": 0, "fontSize": "14px"}),
                            dcc.Graph(id="importance-chart", style={"height": "380px"}),
                        ],
                    ),
                ],
            ),
        ]
    )


def _predict_layout():
    return html.Div(
        children=[
            html.H4("Paste a new financial document to score", style={"color": "#e6edf3", "marginTop": 0}),
            dcc.Textarea(
                id="new-doc-text",
                placeholder="Paste the document text here…",
                style={
                    "width": "100%",
                    "height": "280px",
                    "backgroundColor": "#161b22",
                    "color": "#e6edf3",
                    "border": "1px solid #30363d",
                    "borderRadius": "6px",
                    "padding": "12px",
                    "fontSize": "13px",
                    "resize": "vertical",
                    "boxSizing": "border-box",
                },
            ),
            html.Button(
                "Score Document",
                id="score-btn",
                n_clicks=0,
                style={
                    "marginTop": "12px",
                    "padding": "10px 24px",
                    "backgroundColor": "#388bfd",
                    "color": "white",
                    "border": "none",
                    "borderRadius": "6px",
                    "cursor": "pointer",
                    "fontSize": "14px",
                },
            ),
            html.Div(id="predict-output", style={"marginTop": "24px"}),
        ]
    )


# ---------------------------------------------------------------------------
# Callbacks — history tab
# ---------------------------------------------------------------------------

@app.callback(
    Output("ts-chart", "figure"),
    Output("date-dropdown", "options"),
    Output("date-dropdown", "value"),
    Input("history-store", "data"),
)
def render_ts_chart(data):
    if not data or not data["rows"]:
        empty = go.Figure()
        empty.update_layout(
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            font_color="#8b949e",
            title="No data — run the pipeline first",
        )
        return empty, [], None

    df = pd.DataFrame(data["rows"])
    df["date"] = pd.to_datetime(df["date"])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["stress_actual"],
            name="Actual Stress Index",
            line={"color": "#388bfd", "width": 2},
            mode="lines+markers",
            marker={"size": 7},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["stress_pred"],
            name="XGBoost Prediction",
            line={"color": "#f78166", "width": 2, "dash": "dot"},
            mode="lines+markers",
            marker={"size": 7, "symbol": "diamond"},
        )
    )
    fig.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font_color="#e6edf3",
        legend={"font": {"color": "#e6edf3"}},
        xaxis={
            "gridcolor": "#21262d",
            "title": "Date",
            "tickformat": "%b %d",
        },
        yaxis={"gridcolor": "#21262d", "title": "Stress Score"},
        margin={"t": 20, "b": 40, "l": 60, "r": 20},
        hovermode="x unified",
    )

    options = [{"label": d, "value": d} for d in data["dates"]]
    default_date = data["dates"][-1] if data["dates"] else None
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
    if not selected_date or not data or not data["rows"]:
        return "Select a date to view the document."
    df = pd.DataFrame(data["rows"])
    row = df[df["date"] == selected_date]
    if row.empty:
        return f"No document found for {selected_date}."
    record = row.iloc[0]
    header = f"[{record['doc_type'].upper()} — {selected_date}]\n"
    header += f"Predicted: {record['stress_pred']:.3f} | Actual: {record['stress_actual']:.3f}\n"
    header += "─" * 60 + "\n\n"
    return header + record["content"]


@app.callback(
    Output("importance-chart", "figure"),
    Input("history-store", "data"),
)
def render_importance_chart(_):
    _, _, metadata = load_artifacts()

    if metadata is None:
        empty = go.Figure()
        empty.update_layout(
            paper_bgcolor="#161b22",
            plot_bgcolor="#161b22",
            font_color="#8b949e",
            title="No model loaded — run training/train.py first",
        )
        return empty

    importances = metadata.get("feature_importances", {})
    if not importances:
        return go.Figure()

    items = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:15]
    labels = [k for k, _ in items]
    values = [v for _, v in items]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color="#388bfd",
        )
    )
    fig.update_layout(
        paper_bgcolor="#161b22",
        plot_bgcolor="#161b22",
        font_color="#e6edf3",
        xaxis={"gridcolor": "#21262d", "title": "Importance"},
        yaxis={"autorange": "reversed"},
        margin={"t": 10, "b": 40, "l": 90, "r": 20},
    )
    return fig


# ---------------------------------------------------------------------------
# Callbacks — prediction tab
# ---------------------------------------------------------------------------

@app.callback(
    Output("predict-output", "children"),
    Input("score-btn", "n_clicks"),
    State("new-doc-text", "value"),
    prevent_initial_call=True,
)
def score_new_document(n_clicks, text):
    if not text or not text.strip():
        return html.P("Please paste a document first.", style={"color": "#f78166"})

    pca, xgb_model, _ = load_artifacts()
    if pca is None or xgb_model is None:
        return html.P(
            "Model artifacts not found. Run the training pipeline first.",
            style={"color": "#f78166"},
        )

    try:
        st_model = get_st_model()
        vec = st_model.encode([text], convert_to_numpy=True)
        vec_pca = pca.transform(vec)
        score = float(xgb_model.predict(vec_pca)[0])
    except Exception as exc:
        return html.P(f"Error: {exc}", style={"color": "#f78166"})

    # Gauge chart
    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": "Stress Score", "font": {"color": "#e6edf3"}},
            number={"font": {"color": "#e6edf3"}},
            gauge={
                "axis": {"range": [-3, 4], "tickcolor": "#8b949e"},
                "bar": {"color": "#388bfd"},
                "bgcolor": "#161b22",
                "bordercolor": "#30363d",
                "steps": [
                    {"range": [-3, -1], "color": "#1a7f37"},
                    {"range": [-1, 1], "color": "#9e6a03"},
                    {"range": [1, 2.5], "color": "#bc4c00"},
                    {"range": [2.5, 4], "color": "#a40e26"},
                ],
                "threshold": {
                    "line": {"color": "#f78166", "width": 3},
                    "thickness": 0.75,
                    "value": score,
                },
            },
        )
    )
    gauge.update_layout(
        paper_bgcolor="#161b22",
        font_color="#e6edf3",
        height=300,
        margin={"t": 40, "b": 20, "l": 40, "r": 40},
    )

    # Interpretation
    if score >= 2.5:
        label, color = "HIGH STRESS", "#a40e26"
    elif score >= 1.0:
        label, color = "ELEVATED STRESS", "#bc4c00"
    elif score >= -1.0:
        label, color = "NEUTRAL", "#9e6a03"
    else:
        label, color = "CALM — CARRY CONDITIONS", "#1a7f37"

    return html.Div(
        children=[
            html.Div(
                style={
                    "backgroundColor": color,
                    "color": "white",
                    "padding": "8px 20px",
                    "borderRadius": "6px",
                    "display": "inline-block",
                    "fontWeight": "bold",
                    "marginBottom": "16px",
                },
                children=f"{label} ({score:.3f})",
            ),
            dcc.Graph(figure=gauge, style={"height": "300px"}),
        ]
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
