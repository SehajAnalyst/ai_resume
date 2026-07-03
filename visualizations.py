"""
CareerFit AI - Visualization Module
Creates charts for score breakdown, skill match, and gauge display.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


def score_breakdown_chart(result: dict) -> go.Figure:
    """
    Horizontal bar chart showing each sub-score vs its maximum.
    """
    categories = [
        "Skill Match",
        "Projects",
        "Experience",
        "Education",
        "Keywords",
        "Formatting",
    ]
    scored = [
        result["skill_score"],
        result["project_score"],
        result["experience_score"],
        result["education_score"],
        result["keyword_score"],
        result["formatting_score"],
    ]
    max_scores = [40, 20, 15, 10, 10, 5]
    remaining = [m - s for m, s in zip(max_scores, scored)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Your Score",
        y=categories,
        x=scored,
        orientation="h",
        marker_color="#4CAF50",
        text=[f"{s}/{m}" for s, m in zip(scored, max_scores)],
        textposition="inside",
    ))
    fig.add_trace(go.Bar(
        name="Remaining",
        y=categories,
        x=remaining,
        orientation="h",
        marker_color="#E0E0E0",
        showlegend=True,
    ))
    fig.update_layout(
        barmode="stack",
        title="📊 Score Breakdown",
        xaxis_title="Points",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def skill_pie_chart(result: dict) -> go.Figure:
    """
    Pie chart: matched vs missing skills.
    """
    labels = ["Matched Skills", "Missing Skills"]
    values = [result["matched_skills_count"], result["missing_skills_count"]]
    colors = ["#4CAF50", "#FF5252"]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.45,
        marker=dict(colors=colors),
        textinfo="label+percent",
    )])
    fig.update_layout(
        title="🎯 Skill Coverage",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def score_gauge(score: int, category: str) -> go.Figure:
    """
    Gauge chart showing final resume score.
    """
    color_map = {
        "Weak": "#FF5252",
        "Average": "#FFA726",
        "Good": "#66BB6A",
        "Excellent": "#1E88E5",
    }
    bar_color = color_map.get(category, "#4CAF50")

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        delta={"reference": 70, "increasing": {"color": "#4CAF50"}},
        title={"text": f"Resume Score — <b>{category}</b>", "font": {"size": 18}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": bar_color},
            "steps": [
                {"range": [0, 40],   "color": "#FFEBEE"},
                {"range": [40, 70],  "color": "#FFF3E0"},
                {"range": [70, 85],  "color": "#E8F5E9"},
                {"range": [85, 100], "color": "#E3F2FD"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 2},
                "thickness": 0.75,
                "value": score,
            },
        },
        number={"suffix": "/100"},
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig
