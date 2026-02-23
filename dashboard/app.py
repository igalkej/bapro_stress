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
            from darts.models import TiDEModel
            _tide_model = TiDEModel.load(TIDE_MODEL_PATH)
        except Exception:
            pass

    return _tide_model, _metadata


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
                    p.date,
                    MIN(a.headline)  AS content,
                    MIN(a.source)    AS doc_type,
                    p.stress_score_pred,
                    f.fsi_value      AS stress_actual
                FROM predictions p
                LEFT JOIN articles a  ON a.date = p.date
                JOIN fsi_target f     ON f.date = p.date
                GROUP BY p.date, p.stress_score_pred, f.fsi_value
                ORDER BY p.date
                """
            )
        ).fetchall()

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(
        [
            {
                "date": str(r.date)[:10],
                "content": r.content or "",
                "doc_type": r.doc_type or "",
                "stress_pred": r.stress_score_pred,
                "stress_actual": r.stress_actual,
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
                    # Model quality panel
                    html.Div(
                        id="model-quality-panel",
                        style={"flex": "1", "backgroundColor": "#161b22", "borderRadius": "8px", "padding": "16px"},
                        children=[
                            html.H4("TiDE Model Quality", style={"color": "#e6edf3", "marginTop": 0, "fontSize": "14px"}),
                            html.Div(id="model-quality-content"),
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
            name="TiDE Prediction",
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
    Output("model-quality-content", "children"),
    Input("history-store", "data"),
)
def render_model_quality(_):
    _, metadata = load_artifacts()

    if metadata is None:
        return html.P(
            "No model loaded. Run training/train.py first.",
            style={"color": "#8b949e", "fontSize": "13px"},
        )

    _cell = lambda txt, bold=False: html.Td(
        txt,
        style={
            "padding": "6px 12px",
            "color": "#e6edf3" if bold else "#8b949e",
            "fontWeight": "600" if bold else "normal",
            "fontSize": "13px",
            "borderBottom": "1px solid #21262d",
        },
    )

    def metric_row(split, mae_key, rmse_key):
        return html.Tr([
            _cell(split, bold=True),
            _cell(f"{metadata.get(mae_key, 'N/A'):.4f}" if isinstance(metadata.get(mae_key), float) else "N/A"),
            _cell(f"{metadata.get(rmse_key, 'N/A'):.4f}" if isinstance(metadata.get(rmse_key), float) else "N/A"),
            _cell(str(metadata.get(f"{split.lower()}_samples", "N/A"))),
        ])

    header_style = {
        "padding": "6px 12px",
        "color": "#388bfd",
        "fontSize": "12px",
        "fontWeight": "600",
        "borderBottom": "2px solid #30363d",
        "textTransform": "uppercase",
    }

    metrics_table = html.Table(
        style={"width": "100%", "borderCollapse": "collapse", "marginBottom": "20px"},
        children=[
            html.Thead(html.Tr([
                html.Th("Split", style=header_style),
                html.Th("MAE", style=header_style),
                html.Th("RMSE", style=header_style),
                html.Th("Samples", style=header_style),
            ])),
            html.Tbody([
                metric_row("Train", "train_mae", "train_rmse"),
                metric_row("Val", "val_mae", "val_rmse"),
                metric_row("Test", "test_mae", "test_rmse"),
            ]),
        ],
    )

    # TiDE config
    tp = metadata.get("tide_params", {})
    config_items = [
        ("Model", "Darts TiDE"),
        ("Input window", str(tp.get("input_chunk_length", ""))),
        ("Output window", str(tp.get("output_chunk_length", ""))),
        ("Hidden size", str(tp.get("hidden_size", ""))),
        ("Encoder layers", str(tp.get("num_encoder_layers", ""))),
        ("Decoder out dim", str(tp.get("decoder_output_dim", ""))),
        ("Dropout", str(tp.get("dropout", ""))),
        ("Epochs", str(tp.get("n_epochs", ""))),
        ("Batch size", str(tp.get("batch_size", ""))),
        ("LR", str(tp.get("lr", ""))),
        ("Past covariates", "384-dim embeddings"),
        ("Version", metadata.get("model_version", "")[:15]),
    ]

    config_rows = [
        html.Tr([
            html.Td(k, style={"color": "#8b949e", "fontSize": "12px", "padding": "3px 12px 3px 0"}),
            html.Td(v, style={"color": "#e6edf3", "fontSize": "12px", "padding": "3px 0"}),
        ])
        for k, v in config_items
    ]

    config_table = html.Table(
        style={"width": "100%", "borderCollapse": "collapse"},
        children=[html.Tbody(config_rows)],
    )

    return html.Div([
        html.P("Evaluation Metrics", style={"color": "#8b949e", "fontSize": "11px",
                                             "textTransform": "uppercase", "marginBottom": "8px"}),
        metrics_table,
        html.P("Model Configuration", style={"color": "#8b949e", "fontSize": "11px",
                                              "textTransform": "uppercase", "marginBottom": "8px"}),
        config_table,
    ])


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

    tide_model, _ = load_artifacts()
    if tide_model is None:
        return html.P(
            "TiDE model not found. Run training/train.py first.",
            style={"color": "#f78166"},
        )

    try:
        import json as _json
        import numpy as _np
        import pandas as _pd
        from darts import TimeSeries as _TS
        from sqlalchemy import text as _text

        # Encode new document
        st_model = get_st_model()
        new_emb = st_model.encode([text], convert_to_numpy=True)[0].tolist()

        # Load last input_chunk_length data points from DB for context
        n_ctx = tide_model.input_chunk_length
        engine = get_db_engine()
        with engine.connect() as conn:
            rows = conn.execute(
                _text(
                    """
                    SELECT f.date, ae.embedding, f.fsi_value
                    FROM fsi_target f
                    JOIN articles a ON a.date = f.date
                    JOIN article_embeddings ae ON ae.id = a.id
                    ORDER BY f.date DESC
                    LIMIT :n
                    """
                ),
                {"n": n_ctx},
            ).fetchall()
        rows = list(reversed(rows))

        dates = [_pd.Timestamp(str(r[0])[:10]) for r in rows]
        stress_vals = [float(r[2]) for r in rows]
        embeddings = [
            _json.loads(r[1]) if isinstance(r[1], str) else list(r[1])
            for r in rows
        ]

        # Append new doc embedding as covariate for prediction step
        delta = dates[-1] - dates[-2] if len(dates) >= 2 else _pd.Timedelta(days=1)
        next_date = dates[-1] + delta
        all_dates = dates + [next_date]
        all_embs = embeddings + [new_emb]

        target_df = _pd.DataFrame(
            {"stress_value": stress_vals}, index=_pd.DatetimeIndex(dates)
        )
        target_series = _TS.from_dataframe(target_df, freq=None)

        dim = len(new_emb)
        cov_df = _pd.DataFrame(
            _np.array(all_embs, dtype=_np.float32),
            index=_pd.DatetimeIndex(all_dates),
            columns=[f"emb_{i}" for i in range(dim)],
        )
        cov_series = _TS.from_dataframe(cov_df, freq=None)

        pred = tide_model.predict(n=1, series=target_series, past_covariates=cov_series)
        score = float(pred.values()[0][0])
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
