# streamlit_app.py
# Run: streamlit run streamlit_app.py

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Revenue Dashboard", layout="wide")


# ✅ Template-style data load
DATA_FILENAME = Path(__file__).parent / "data/gdp_data.csv"
raw_gdp_df = pd.read_csv(DATA_FILENAME)


@st.cache_data(show_spinner=False)
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Parse Month (YYYY-MM)
    df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m", errors="coerce")

    # Clean numeric columns (in case formatting shows up later)
    numeric_cols = [c for c in df.columns if c != "Month"]
    for c in numeric_cols:
        df[c] = (
            df[c]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.replace("%", "", regex=False)
        )
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Sort
    df = df.sort_values("Month").reset_index(drop=True)

    # Derived metrics
    df["Net_Customers"] = df["New_Customers"] - df["Churned_Customers"]
    df["Gross_Profit_USD"] = df["Total_Revenue_USD"] * (df["Gross_Margin_%"] / 100)

    df["Total_Revenue_MoM_%"] = df["Total_Revenue_USD"].pct_change() * 100
    df["Total_Revenue_YoY_%"] = df["Total_Revenue_USD"].pct_change(12) * 100

    return df


df = clean_data(raw_gdp_df)

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.title("Filters")

min_date = df["Month"].min()
max_date = df["Month"].max()

date_range = st.sidebar.slider(
    "Month Range",
    min_value=min_date.to_pydatetime(),
    max_value=max_date.to_pydatetime(),
    value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
    format="YYYY-MM"
)

start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
fdf = df[(df["Month"] >= start) & (df["Month"] <= end)].copy()

metric_options = [
    "Total_Revenue_USD",
    "Subscription_Revenue_USD",
    "API_Revenue_USD",
    "Gross_Profit_USD",
    "Units",
    "New_Customers",
    "Churned_Customers",
    "Net_Customers",
    "Gross_Margin_%",
    "Total_Revenue_MoM_%",
    "Total_Revenue_YoY_%"
]

selected_metrics = st.sidebar.multiselect(
    "Select Metrics to Plot",
    options=metric_options,
    default=["Total_Revenue_USD", "Subscription_Revenue_USD", "API_Revenue_USD"]
)

show_table = st.sidebar.checkbox("Show Data Table", value=True)

# -----------------------------
# Header + KPIs
# -----------------------------
st.title("Revenue & Customer Trends Dashboard")

latest = fdf.iloc[-1] if len(fdf) else df.iloc[-1]
prev = fdf.iloc[-2] if len(fdf) >= 2 else None

def money_fmt(x):
    return f"${x:,.0f}" if pd.notna(x) else "—"

def pct_fmt(x):
    return f"{x:,.2f}%" if pd.notna(x) else "—"

k1, k2, k3, k4 = st.columns(4)

k1.metric(
    "Total Revenue (Latest)",
    money_fmt(latest["Total_Revenue_USD"]),
    None if prev is None else pct_fmt((latest["Total_Revenue_USD"] / prev["Total_Revenue_USD"] - 1) * 100)
)

k2.metric(
    "Gross Margin (Latest)",
    pct_fmt(latest["Gross_Margin_%"]),
    None if prev is None else pct_fmt(latest["Gross_Margin_%"] - prev["Gross_Margin_%"])
)

k3.metric(
    "New Customers (Latest)",
    f"{int(latest['New_Customers']):,}",
    None if prev is None else f"{int(latest['New_Customers'] - prev['New_Customers']):,}"
)

k4.metric(
    "Net Customers (Latest)",
    f"{int(latest['Net_Customers']):,}",
    None if prev is None else f"{int(latest['Net_Customers'] - prev['Net_Customers']):,}"
)

st.divider()

# -----------------------------
# Chart: Revenue Breakdown
# -----------------------------
st.subheader("Revenue Breakdown Over Time")

rev_df = fdf[["Month", "Subscription_Revenue_USD", "API_Revenue_USD"]].copy()
rev_long = rev_df.melt(id_vars=["Month"], var_name="Revenue_Type", value_name="Revenue_USD")

fig_rev = px.area(
    rev_long,
    x="Month",
    y="Revenue_USD",
    color="Revenue_Type",
    title="Subscription vs API Revenue",
)
st.plotly_chart(fig_rev, use_container_width=True)

# -----------------------------
# Chart: Selected Metrics
# -----------------------------
st.subheader("Selected Metrics (Line Chart)")

if selected_metrics:
    plot_df = fdf[["Month"] + selected_metrics].copy()
    plot_long = plot_df.melt(id_vars=["Month"], var_name="Metric", value_name="Value")

    fig_line = px.line(
        plot_long,
        x="Month",
        y="Value",
        color="Metric",
        markers=True,
        title="Trends Over Time"
    )
    st.plotly_chart(fig_line, use_container_width=True)
else:
    st.info("Select at least one metric in the sidebar to display the line chart.")

# -----------------------------
# Chart: Customers (New vs Churn)
# -----------------------------
st.subheader("Customer Movement")

cust_df = fdf[["Month", "New_Customers", "Churned_Customers", "Net_Customers"]].copy()
cust_long = cust_df.melt(id_vars=["Month"], var_name="Customer_Metric", value_name="Count")

fig_cust = px.bar(
    cust_long,
    x="Month",
    y="Count",
    color="Customer_Metric",
    barmode="group",
    title="New vs Churned vs Net Customers"
)
st.plotly_chart(fig_cust, use_container_width=True)

# -----------------------------
# Table
# -----------------------------
if show_table:
    st.subheader("Filtered Data Table")
    st.dataframe(
        fdf,
        use_container_width=True,
        hide_index=True
    )
