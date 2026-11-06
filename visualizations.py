"""
BOVIGENIX - Visualization Helpers (utils/visualizations.py)
------------------------------------------------------------
This module provides interactive charts using Plotly with professional
scientific styling, clean layout margins, and guaranteed text readability.

Design Principles:
- High contrast: Dark text (#0f172a, #1e293b) on light backgrounds.
- Zero overlap: Legends placed below plots with ample margin spacing.
- Responsive dimensions: Charts have generous heights and padding to avoid clipping.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


def plot_sensor_trends(cow_df, cow_id):
    """
    Creates an interactive dual-axis time-series chart showing:
    - Left Y-Axis: Activity (Steps/h)
    - Right Y-Axis: Rumination Time (Minutes/day)
    - Clearly marked estrus events and health flags
    - Legend placed cleanly below the plot to eliminate any overlap with titles
    """
    if cow_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No sensor observations available for this cow.",
            xref="paper", yref="paper", showarrow=False,
            font=dict(size=14, color="#0f172a")
        )
        fig.update_layout(height=350, template="plotly_white")
        return fig

    # Ensure chronological sorting
    df = cow_df.sort_values("timestamp").copy()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 1. Activity Line (Left Axis)
    if "activity" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["activity"],
                name="Activity (Steps/h)",
                mode="lines+markers",
                line=dict(color="#0284c7", width=2.5),
                marker=dict(size=6, color="#0369a1"),
                hovertemplate="Time: %{x}<br>Activity: %{y} steps/h<extra></extra>"
            ),
            secondary_y=False
        )

    # 2. Rumination Line (Right Axis)
    if "rumination_time" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["rumination_time"],
                name="Rumination (min/day)",
                mode="lines+markers",
                line=dict(color="#16a34a", width=2.2, dash="dot"),
                marker=dict(size=5, color="#15803d"),
                hovertemplate="Time: %{x}<br>Rumination: %{y} min/day<extra></extra>"
            ),
            secondary_y=True
        )

    # 3. Mark Estrus Days (if labeled)
    if "estrus_label" in df.columns:
        estrus_points = df[df["estrus_label"] == 1]
        if not estrus_points.empty:
            fig.add_trace(
                go.Scatter(
                    x=estrus_points["timestamp"],
                    y=estrus_points["activity"],
                    name="Confirmed Estrus Event",
                    mode="markers",
                    marker=dict(
                        symbol="star",
                        size=15,
                        color="#ea580c",
                        line=dict(width=1.5, color="#7c2d12")
                    ),
                    hovertemplate="<b>ESTRUS EVENT</b><br>Time: %{x}<br>Activity: %{y}<extra></extra>"
                ),
                secondary_y=False
            )

    fig.update_layout(
        title=dict(
            text=f"<b>Multi-Sensor Trajectory: Cow {cow_id}</b>",
            font=dict(size=16, color="#0f172a"),
            x=0.02,
            xanchor="left"
        ),
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        hovermode="x unified",
        # Place legend below chart with ample margin to prevent ANY overlap with title
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(size=12, color="#0f172a"),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="#e2e8f0",
            borderwidth=1
        ),
        margin=dict(l=65, r=65, t=65, b=85),
        height=430
    )

    fig.update_xaxes(
        title=dict(text="<b>Observation Timestamp</b>", font=dict(color="#0f172a", size=12)),
        tickfont=dict(color="#334155", size=11),
        showgrid=True,
        gridcolor="#f1f5f9"
    )
    fig.update_yaxes(
        title=dict(text="<b>Activity (Steps/h)</b>", font=dict(color="#0284c7", size=12)),
        tickfont=dict(color="#0284c7", size=11),
        secondary_y=False,
        showgrid=True,
        gridcolor="#f1f5f9"
    )
    fig.update_yaxes(
        title=dict(text="<b>Rumination Time (min/day)</b>", font=dict(color="#16a34a", size=12)),
        tickfont=dict(color="#16a34a", size=11),
        secondary_y=True,
        showgrid=False
    )

    return fig


def plot_thi_gauge(thi_value):
    """
    Creates an intuitive speedometer/gauge chart for the Temperature-Humidity Index (THI).
    Guarantees high contrast text with clear title separation.
    """
    if thi_value is None or np.isnan(thi_value):
        thi_display = 0.0
    else:
        thi_display = float(thi_value)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=thi_display,
        title={
            'text': "<b>Current Barn THI</b><br><span style='font-size:0.82em;color:#475569;'>Thermal Comfort Gauge</span>",
            'font': {'color': '#0f172a', 'size': 16}
        },
        number={
            'font': {'color': '#0f172a', 'size': 38, 'family': 'sans-serif'},
            'suffix': ''
        },
        gauge={
            'axis': {
                'range': [50, 100],
                'tickwidth': 1.5,
                'tickcolor': "#0f172a",
                'tickfont': {'color': '#0f172a', 'size': 11}
            },
            'bar': {'color': "#0f172a", 'thickness': 0.22},
            'bgcolor': "#ffffff",
            'borderwidth': 1.5,
            'bordercolor': "#cbd5e1",
            'steps': [
                {'range': [50, 68], 'color': '#86efac'},  # Green: Comfort Zone
                {'range': [68, 72], 'color': '#fde047'},  # Yellow: Mild Stress
                {'range': [72, 79], 'color': '#fdba74'},  # Orange: Moderate Stress
                {'range': [79, 84], 'color': '#f87171'},  # Red: Severe Stress
                {'range': [84, 100], 'color': '#991b1b'}  # Dark Red: Danger
            ],
            'threshold': {
                'line': {'color': "#0f172a", 'width': 4},
                'thickness': 0.75,
                'value': thi_display
            }
        }
    ))

    fig.update_layout(
        height=320,
        margin=dict(l=35, r=35, t=75, b=30),
        template="plotly_white",
        paper_bgcolor="#ffffff"
    )
    return fig


def plot_confusion_matrix_heatmap(cm, labels=["Normal", "Estrus"]):
    """
    Plots an interactive confusion matrix heatmap with True Positives,
    False Positives, True Negatives, and False Negatives clearly labeled.
    """
    z = cm
    x = [f"Predicted {lab}" for lab in labels]
    y = [f"Actual {lab}" for lab in labels]

    annotations = []
    total = np.sum(cm)
    for i, row in enumerate(cm):
        for j, val in enumerate(row):
            pct = (val / total * 100) if total > 0 else 0
            text_color = "white" if val > (np.max(cm) * 0.5) else "#0f172a"
            annotations.append(
                dict(
                    x=x[j],
                    y=y[i],
                    text=f"<b>{val}</b><br><span style='font-size:0.85em;'>({pct:.1f}%)</span>",
                    showarrow=False,
                    font=dict(color=text_color, size=15)
                )
            )

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x,
        y=y,
        colorscale="Blues",
        showscale=False
    ))

    fig.update_layout(
        title=dict(
            text="<b>Confusion Matrix (Classification Evaluation)</b>",
            font=dict(size=16, color="#0f172a"),
            x=0.02,
            xanchor="left"
        ),
        annotations=annotations,
        xaxis=dict(
            side="bottom",
            title=dict(text="<b>Predicted Classification</b>", font=dict(color="#0f172a", size=13)),
            tickfont=dict(size=12, color="#0f172a")
        ),
        yaxis=dict(
            title=dict(text="<b>Actual Reference Label</b>", font=dict(color="#0f172a", size=13)),
            tickfont=dict(size=12, color="#0f172a")
        ),
        template="plotly_white",
        paper_bgcolor="#ffffff",
        height=380,
        margin=dict(l=80, r=40, t=65, b=65)
    )
    return fig


def plot_feature_importance(feature_names, importances, title="Feature Importance (Random Forest)"):
    """
    Plots a horizontal bar chart ranking the most predictive sensor variables.
    Generous left margin prevents truncation of feature names.
    """
    df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=True)

    fig = go.Figure(go.Bar(
        x=df["Importance"],
        y=df["Feature"],
        orientation="h",
        marker=dict(color="#0284c7", line=dict(color="#0369a1", width=1.5)),
        hovertemplate="Feature: %{y}<br>Importance Score: %{x:.3f}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=15, color="#0f172a"),
            x=0.02,
            xanchor="left"
        ),
        xaxis=dict(
            title=dict(text="<b>Relative Gini Importance</b>", font=dict(color="#0f172a", size=12)),
            tickfont=dict(size=11, color="#334155"),
            showgrid=True,
            gridcolor="#f1f5f9"
        ),
        yaxis=dict(
            title=dict(text="<b>Sensor / Context Variable</b>", font=dict(color="#0f172a", size=12)),
            tickfont=dict(size=11, color="#0f172a")
        ),
        template="plotly_white",
        paper_bgcolor="#ffffff",
        height=360,
        margin=dict(l=150, r=30, t=55, b=50)
    )
    return fig


def plot_model_comparison(comparison_df):
    """
    Visualizes performance metrics comparison between:
    - Model 1: Basic Behavioural Model
    - Model 2: Individualized & Context-Aware Model
    With legend positioned below the chart to eliminate overlap with the title.
    """
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    
    fig = go.Figure()

    colors = ["#0284c7", "#0f766e"]

    for idx, row in comparison_df.iterrows():
        model_name = row["Model"]
        values = [row.get("Accuracy", 0), row.get("Precision", 0),
                  row.get("Recall", 0), row.get("F1-Score", 0)]
        
        fig.add_trace(go.Bar(
            name=model_name,
            x=metrics,
            y=values,
            text=[f"{v*100:.1f}%" if v <= 1.0 else f"{v:.1f}%" for v in values],
            textposition="auto",
            textfont=dict(color="#ffffff", size=12),
            marker=dict(color=colors[idx % len(colors)])
        ))

    fig.update_layout(
        barmode="group",
        title=dict(
            text="<b>Model Comparison: Basic vs. Individualized & Context-Aware</b>",
            font=dict(size=16, color="#0f172a"),
            x=0.02,
            xanchor="left"
        ),
        xaxis=dict(
            tickfont=dict(size=12, color="#0f172a", family="sans-serif")
        ),
        yaxis=dict(
            title=dict(text="<b>Performance Score (0.0 to 1.0)</b>", font=dict(color="#0f172a", size=12)),
            range=[0, 1.25],
            showgrid=True,
            gridcolor="#f1f5f9",
            tickfont=dict(size=11, color="#334155")
        ),
        template="plotly_white",
        paper_bgcolor="#ffffff",
        # Place legend below to completely avoid overlapping the title
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.20,
            xanchor="center",
            x=0.5,
            font=dict(size=12, color="#0f172a"),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="#e2e8f0",
            borderwidth=1
        ),
        height=420,
        margin=dict(l=60, r=40, t=65, b=80)
    )
    return fig


def plot_herd_status_donut(status_counts):
    """
    Renders an elegant donut chart displaying the herd breakdown:
    - Center metric shows the total herd size.
    - Zero-count slices are filtered out to prevent crowded/overlapping labels.
    - Slices show clean percentages inside the ring, while labels/counts appear in a legend below.
    """
    total = sum(status_counts.values())

    # Filter out 0-value items for the pie slices to avoid overlapping 0-labels
    active_items = {k: v for k, v in status_counts.items() if v > 0}

    colors_map = {
        "Active": "#16a34a",     # Emerald Green
        "Inactive": "#94a3b8",   # Slate Gray
        "Sold": "#0284c7",       # Sky Blue
        "Deceased": "#64748b"    # Dark Charcoal Gray
    }

    if total == 0 or not active_items:
        fig = go.Figure()
        fig.add_annotation(
            text="<b>No Cattle Registered</b><br><span style='font-size:11px;color:#64748b;'>Database is empty</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#64748b")
        )
        fig.update_layout(
            height=300,
            template="plotly_white",
            margin=dict(l=20, r=20, t=30, b=20)
        )
        return fig

    labels = list(active_items.keys())
    values = list(active_items.values())
    slice_colors = [colors_map.get(k, "#cbd5e1") for k in labels]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.62,
        marker=dict(colors=slice_colors, line=dict(color="#ffffff", width=2.5)),
        # Do not put messy labels on tiny slices; show clean values or hover
        textinfo="value",
        textposition="inside",
        textfont=dict(size=13, color="#ffffff", family="sans-serif"),
        hovertemplate="<b>%{label}</b><br>Count: %{value} cows<br>Share: %{percent}<extra></extra>"
    )])

    # Center hole annotation showing total herd
    fig.add_annotation(
        text=f"<span style='font-size:24px;font-weight:800;color:#0f172a;'>{total}</span><br><span style='font-size:10px;font-weight:700;color:#64748b;letter-spacing:0.5px;'>TOTAL HERD</span>",
        x=0.5, y=0.5,
        showarrow=False,
        align="center",
        font=dict(family="sans-serif")
    )

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        height=320,
        showlegend=True,
        # Place clean legend at the bottom with counts
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.12,
            xanchor="center",
            x=0.5,
            font=dict(size=11, color="#0f172a"),
            itemclick=False,
            itemdoubleclick=False
        ),
        margin=dict(l=15, r=15, t=25, b=55)
    )
    return fig
