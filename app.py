from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed"
RISK_FILE = DATA / "risk_scoring_results.csv"
EVAL_FILE = DATA / "forecast_evaluation_results.csv"

st.set_page_config(page_title="Project FORESIGHT", page_icon="📦", layout="wide")

@st.cache_data

def load_data():
    risk = pd.read_csv(RISK_FILE, parse_dates=["forecast_week", "reference_week", "latest_inventory_snapshot_date"])
    evaluation = pd.read_csv(EVAL_FILE, parse_dates=["week_start"])
    return risk, evaluation

try:
    risk_df, evaluation_df = load_data()
except FileNotFoundError as exc:
    st.error(f"Required data file is missing: {exc}")
    st.stop()

st.title("Project FORESIGHT")
st.subheader("Demand & Inventory Intelligence")
st.caption("SKU-level forecasting, lead-time risk scoring, decisioning and inventory impact")

with st.sidebar:
    st.header("Filters")
    categories = ["All"] + sorted(risk_df["category"].dropna().unique().tolist())
    skus = ["All"] + sorted(risk_df["sku"].dropna().unique().tolist())
    quadrants = ["All"] + ["HEALTHY", "REORDER NOW", "MARKDOWN / CLEAR", "WATCH / VOLATILE"]
    selected_category = st.selectbox("Category", categories)
    selected_sku = st.selectbox("SKU", skus)
    selected_quadrant = st.selectbox("Decision", quadrants)
    risk_levels = ["All", "Low", "Medium", "High"]
    selected_stockout = st.selectbox("Stockout risk", risk_levels)

filtered = risk_df.copy()
if selected_category != "All":
    filtered = filtered[filtered["category"] == selected_category]
if selected_sku != "All":
    filtered = filtered[filtered["sku"] == selected_sku]
if selected_quadrant != "All":
    filtered = filtered[filtered["decision_quadrant"] == selected_quadrant]
if selected_stockout != "All":
    filtered = filtered[filtered["stockout_risk"] == selected_stockout]

if filtered.empty:
    st.warning("No records match the selected filters. Clear one or more filters to continue.")
    st.stop()

# KPI cards
c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.metric("Affected SKUs", filtered["sku"].nunique())
with c2: st.metric("High Stockout Records", int((filtered["stockout_risk"] == "High").sum()))
with c3: st.metric("Reorder Records", int((filtered["recommended_action"] == "REORDER").sum()))
with c4: st.metric("Markdown Records", int((filtered["recommended_action"] == "MARKDOWN").sum()))
with c5: st.metric("Watch Records", int((filtered["decision_quadrant"] == "WATCH / VOLATILE").sum()))

c6, c7 = st.columns(2)
with c6:
    st.metric("Potential Lost-Sales Exposure", f"₹{filtered['potential_lost_sales_value'].sum():,.0f}")
with c7:
    st.metric("Potential Overstock Exposure", f"₹{filtered['potential_overstock_inventory_value'].sum():,.0f}")

st.divider()

# Historical forecast evaluation
st.header("Forecast Performance")
eval_plot = evaluation_df.copy()
if selected_sku != "All":
    eval_plot = eval_plot[eval_plot["sku"] == selected_sku]
else:
    visible_skus = filtered["sku"].dropna().unique().tolist()
    eval_plot = eval_plot[eval_plot["sku"].isin(visible_skus)]

if not eval_plot.empty:
    if selected_sku == "All":
        agg = eval_plot.groupby("week_start", as_index=False).agg(
            actual_demand=("actual_demand", "sum"),
            seasonal_naive_forecast=("seasonal_naive_forecast", "sum"),
            holt_forecast=("holt_forecast", "sum"),
            ml_forecast=("ml_forecast", "sum") if "ml_forecast" in eval_plot.columns else ("seasonal_naive_forecast", "sum")
        )
        plot_cols = ["actual_demand", "seasonal_naive_forecast", "holt_forecast"]
        if "ml_forecast" in eval_plot.columns:
            plot_cols.append("ml_forecast")
        fig = px.line(agg, x="week_start", y=plot_cols, markers=True,
                      labels={"value":"Units", "week_start":"Week", "variable":"Series"},
                      title="Actual vs Forecast on the Final Historical Backtest Window")
    else:
        plot_cols = ["actual_demand", "seasonal_naive_forecast", "holt_forecast"]
        if "ml_forecast" in eval_plot.columns:
            plot_cols.append("ml_forecast")
        fig = px.line(eval_plot, x="week_start", y=plot_cols, markers=True,
                      labels={"value":"Units", "week_start":"Week", "variable":"Series"},
                      title=f"Actual vs Forecast — {selected_sku}")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Historical evaluation is unavailable for this filter.")

# Risk charts
st.header("Risk Overview")
left, right = st.columns(2)
with left:
    counts = filtered["stockout_risk"].value_counts().reindex(["Low","Medium","High"], fill_value=0).reset_index()
    counts.columns = ["Risk", "Count"]
    fig = px.bar(counts, x="Risk", y="Count", title="Stockout Risk")
    st.plotly_chart(fig, use_container_width=True)
with right:
    counts = filtered["overstock_risk"].value_counts().reindex(["Low","Medium","High"], fill_value=0).reset_index()
    counts.columns = ["Risk", "Count"]
    fig = px.bar(counts, x="Risk", y="Count", title="Overstock Risk")
    st.plotly_chart(fig, use_container_width=True)

# Four-quadrant decision grid aggregated to SKU level
st.header("SKU Decision Grid")
by_sku = (filtered.groupby("sku", as_index=False)
          .agg(stockout_score=("stockout_risk_score","max"),
               overstock_score=("overstock_risk_score","max"),
               priority_value=("priority_value","sum"),
               revenue_at_risk=("potential_lost_sales_value","sum"),
               category=("category","first")))

by_sku["decision"] = np.select(
    [
        (by_sku["stockout_score"] > 0) & (by_sku["overstock_score"] > 0),
        (by_sku["stockout_score"] > 0) & (by_sku["overstock_score"] == 0),
        (by_sku["stockout_score"] == 0) & (by_sku["overstock_score"] > 0)
    ],
    [
        "WATCH / VOLATILE",
        "REORDER NOW",
        "MARKDOWN / CLEAR"
    ],
    default="HEALTHY"
)

# Ensure display-friendly coordinates.
by_sku["stockout_axis"] = by_sku["stockout_score"].map({0.0: "Low", 0.5: "Medium", 1.0: "High"}).fillna("Low")
by_sku["overstock_axis"] = by_sku["overstock_score"].map({0.0: "Low", 0.5: "Medium", 1.0: "High"}).fillna("Low")

# Create the decision-grid chart with a fixed four-category legend

decision_colors = {
    "HEALTHY": "#8ecae6",
    "REORDER NOW": "#1f77b4",
    "MARKDOWN / CLEAR": "#ff6b6b",
    "WATCH / VOLATILE": "#9467bd"
}

fig = go.Figure()

# Keep all four decision categories visible in the legend
for quadrant in [
    "HEALTHY",
    "REORDER NOW",
    "MARKDOWN / CLEAR",
    "WATCH / VOLATILE"
]:

    subset = by_sku[by_sku["decision"] == quadrant]

    if not subset.empty:

        fig.add_trace(
            go.Scatter(
                x=subset["stockout_score"],
                y=subset["overstock_score"],
                mode="markers",
                name=quadrant,

                marker=dict(
                    size=(
                        subset["priority_value"]
                        .clip(lower=1)
                        .pow(0.5)
                        .clip(upper=35)
                    ),
                    color=decision_colors[quadrant],
                    opacity=0.9,
                    line=dict(width=1, color="white")
                ),

                text=subset["sku"],

                customdata=subset[
                    ["category", "priority_value", "revenue_at_risk"]
                ],

                hovertemplate=(
                    "<b>SKU: %{text}</b><br>"
                    "Category: %{customdata[0]}<br>"
                    "Priority Value: ₹%{customdata[1]:,.0f}<br>"
                    "Potential Lost-Sales Exposure: ₹%{customdata[2]:,.0f}<br>"
                    "<extra>" + quadrant + "</extra>"
                ),

                showlegend=True
            )
        )

    else:

        # Invisible point keeps the category visible in the legend
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=quadrant,
                marker=dict(
                    size=10,
                    color=decision_colors[quadrant]
                ),
                hoverinfo="skip",
                showlegend=True
            )
        )

# Format the X axis
fig.update_xaxes(
    tickvals=[0, 0.5, 1],
    ticktext=["Low", "Medium", "High"],
    range=[-0.05, 1.05],
    title="Stockout Risk (0=Low, 1=High)"
)

# Format the Y axis
fig.update_yaxes(
    tickvals=[0, 0.5, 1],
    ticktext=["Low", "Medium", "High"],
    range=[-0.05, 1.05],
    title="Overstock Risk (0=Low, 1=High)"
)

# Add labels to the four quadrants
fig.add_annotation(
    x=0,
    y=0,
    text="HEALTHY",
    showarrow=False,
    xshift=25,
    yshift=30
)

fig.add_annotation(
    x=1,
    y=0,
    text="REORDER NOW",
    showarrow=False,
    xshift=-70,
    yshift=30
)

fig.add_annotation(
    x=0,
    y=1,
    text="MARKDOWN / CLEAR",
    showarrow=False,
    xshift=70,
    yshift=-35
)

fig.add_annotation(
    x=1,
    y=1,
    text="WATCH / VOLATILE",
    showarrow=False,
    xshift=-70,
    yshift=-35
)

fig.update_layout(
    title="Decisioning: Stockout Risk vs Overstock Risk (one point per SKU)",
    height=500,
    legend_title="Quadrant"
)

st.plotly_chart(fig, use_container_width=True)

# Action tables
st.header("Prioritized Actions")
left, right = st.columns(2)
with left:
    st.subheader("🔴 Reorder")
    reorder = filtered[filtered["recommended_action"] == "REORDER"].sort_values("priority_value", ascending=False).drop_duplicates(subset=["sku"])
    if reorder.empty:
        st.info("No reorder records under the selected filters.")
    else:
        st.dataframe(reorder[["sku","recommended_action","stockout_risk","forecast_demand","reorder_units","reorder_value"]].rename(columns={"recommended_action":"Action", "stockout_risk":"Risk", "forecast_demand":"Forecast", "reorder_units":"Reorder Units", "reorder_value":"₹ Impact"}).head(25), use_container_width=True, hide_index=True)
with right:
    st.subheader("🟠 Markdown / Clear")
    markdown = filtered[filtered["recommended_action"] == "MARKDOWN"].sort_values("priority_value", ascending=False).drop_duplicates(subset=["sku"])
    if markdown.empty:
        st.info("No markdown records under the selected filters.")
    else:
        st.dataframe(markdown[["sku","recommended_action","overstock_risk","forecast_demand","overstock_units","potential_overstock_inventory_value"]].rename(columns={"recommended_action":"Action", "overstock_risk":"Risk", "forecast_demand":"Forecast", "overstock_units":"Overstock Units", "potential_overstock_inventory_value":"₹ Impact"}).head(25), use_container_width=True, hide_index=True)

# Selected SKU detail
if selected_sku != "All":
    st.header(f"SKU Scoring Detail — {selected_sku}")
    detail = filtered.sort_values("forecast_week")
    st.dataframe(detail[[
        "forecast_week","forecast_demand","lead_time_days","lead_time_demand",
        "inventory_before_forecast","inventory_after_lead_time","inventory_after_forecast","stockout_risk",
        "inventory_coverage_weeks","overstock_risk","decision_quadrant",
        "recommended_action","potential_lost_sales_value","potential_overstock_inventory_value",
        "reorder_units","reorder_value","priority_level"
    ]], use_container_width=True, hide_index=True)

# Download current filtered result
st.header("Export")
st.download_button(
    "Download filtered risk results (CSV)",
    filtered.to_csv(index=False).encode("utf-8"),
    file_name="foresight_filtered_risk.csv",
    mime="text/csv"
)

with st.expander("Methodology & assumptions"):
    st.markdown(
        """
        **Forecast:** feature-based HistGradientBoostingRegressor selected using four rolling-origin 13-week folds.

        **Stockout risk:** current projected inventory is compared with estimated lead-time demand and safety stock.

        **Overstock risk:** projected inventory coverage above 13 weeks is High and above 8 weeks is Medium; these are project-defined business rules.

        **Inventory timing assumption:** latest known on-hand + on-order form the starting inventory position because exact future inbound arrival dates are not available in the extract.

        **Financial values:** potential lost-sales value and potential overstock inventory value are exposure estimates, not realized losses.
        """
    )
