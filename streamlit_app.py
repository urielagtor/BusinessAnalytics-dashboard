# streamlit_app.py
# Run:
#   streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
#
# Expects:
#   ./data/gdp_data.csv
#
# Note:
#   Predictive modeling uses scikit-learn. Install:
#     pip install scikit-learn
#   or add "scikit-learn" to requirements.txt

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

st.set_page_config(page_title="Revenue Dashboard", layout="wide")

# ✅ Template-style data load
DATA_FILENAME = Path(__file__).parent / "data/gdp_data.csv"
raw_gdp_df = pd.read_csv(DATA_FILENAME)

# ✅ Clean column headers (handles BOM + whitespace)
raw_gdp_df.columns = raw_gdp_df.columns.str.replace("\ufeff", "", regex=False).str.strip()


@st.cache_data(show_spinner=False)
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    required = [
        "Month",
        "Total_Revenue_USD",
        "Subscription_Revenue_USD",
        "API_Revenue_USD",
        "Units",
        "New_Customers",
        "Churned_Customers",
        "Gross_Margin_%"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            "Missing required column(s): "
            + ", ".join(missing)
            + f"\n\nColumns found: {list(df.columns)}"
        )

    # Parse Month (YYYY-MM)
    df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m", errors="coerce")
    if df["Month"].isna().any():
        bad = df.loc[df["Month"].isna(), ["Month"]].head(10)
        raise ValueError(
            "Some Month values could not be parsed as YYYY-MM.\n"
            f"Examples (first 10):\n{bad.to_string(index=False)}"
        )

    # Clean numeric columns (handles commas/$/%)
    numeric_cols = [c for c in df.columns if c != "Month"]
    for c in numeric_cols:
        df[c] = (
            df[c]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.strip()
        )
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values("Month").reset_index(drop=True)

    # Derived metrics
    df["Net_Customers"] = df["New_Customers"] - df["Churned_Customers"]
    df["Gross_Profit_USD"] = df["Total_Revenue_USD"] * (df["Gross_Margin_%"] / 100.0)
    df["Total_Revenue_MoM_%"] = df["Total_Revenue_USD"].pct_change() * 100.0
    df["Total_Revenue_YoY_%"] = df["Total_Revenue_USD"].pct_change(12) * 100.0

    # Revenue mix
    df["Subscription_Share_%"] = np.where(
        df["Total_Revenue_USD"] > 0,
        (df["Subscription_Revenue_USD"] / df["Total_Revenue_USD"]) * 100.0,
        np.nan,
    )
    df["API_Share_%"] = np.where(
        df["Total_Revenue_USD"] > 0,
        (df["API_Revenue_USD"] / df["Total_Revenue_USD"]) * 100.0,
        np.nan,
    )

    return df


def make_time_features(d: pd.DataFrame) -> pd.DataFrame:
    out = d.copy().sort_values("Month").reset_index(drop=True)
    out["t"] = np.arange(len(out))
    m = out["Month"].dt.month.astype(int)
    out["month_sin"] = np.sin(2 * np.pi * m / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * m / 12.0)
    return out


def fit_forecast(
    d: pd.DataFrame,
    target_col: str,
    horizon: int = 6,
    alpha: float = 1.0,
    test_months: int = 12,
) -> tuple[pd.DataFrame, dict]:
    """
    Ridge regression forecast using time trend + monthly seasonality.
    Backtest on last `test_months` observations (if available).
    Returns:
      forecast_df: Month, Actual, Fitted, Forecast
      metrics: MAE, MAPE (if possible)
    """
    d = d.dropna(subset=["Month", target_col]).copy()
    d = make_time_features(d)

    feature_cols = ["t", "month_sin", "month_cos"]
    X = d[feature_cols]
    y = d[target_col]

    n = len(d)
    test_size = min(test_months, max(0, n // 4))  # cap at 12-ish; avoid tiny train
    train_end = n - test_size if test_size > 0 else n

    model = Ridge(alpha=alpha)
    model.fit(X.iloc[:train_end], y.iloc[:train_end])

    fitted = model.predict(X)

    # Future months
    last_month = d["Month"].max()
    future_months = pd.date_range(
        last_month + pd.offsets.MonthBegin(1),
        periods=horizon,
        freq="MS",
    )

    future = pd.DataFrame({"Month": future_months})
    future["t"] = np.arange(n, n + horizon)
    m = future["Month"].dt.month.astype(int)
    future["month_sin"] = np.sin(2 * np.pi * m / 12.0)
    future["month_cos"] = np.cos(2 * np.pi * m / 12.0)

    y_fore = model.predict(future[feature_cols])

    # Output DF
    hist_out = d[["Month", target_col]].rename(columns={target_col: "Actual"}).copy()
    hist_out["Fitted"] = fitted
    hist_out["Forecast"] = np.nan

    fut_out = pd.DataFrame(
        {"Month": future["Month"], "Actual": np.nan, "Fitted": np.nan, "Forecast": y_fore}
    )

    forecast_df = pd.concat([hist_out, fut_out], ignore_index=True)

    # Backtest metrics
    metrics: dict = {}
    if test_size > 0:
        y_true = y.iloc[train_end:]
        y_pred = pd.Series(fitted[train_end:], index=y_true.index)

        metrics["MAE"] = float(mean_absolute_error(y_true, y_pred))

        # MAPE isn't defined well with zeros
        if (y_true == 0).any():
            metrics["MAPE"] = None
        else:
            metrics["MAPE"] = float(mean_absolute_percentage_error(y_true, y_pred) * 100.0)

        metrics["Backtest_Months"] = int(test_size)
    else:
        metrics["Backtest_Months"] = 0

    return forecast_df, metrics


# -----------------------------
# Load + Clean
# -----------------------------
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
    format="YYYY-MM",
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
    "Subscription_Share_%",
    "API_Share_%",
    "Total_Revenue_MoM_%",
    "Total_Revenue_YoY_%"
]

selected_metrics = st.sidebar.multiselect(
    "Select Metrics to Plot",
    options=metric_options,
    default=["Total_Revenue_USD", "Subscription_Revenue_USD", "API_Revenue_USD"],
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
    None
    if prev is None or prev["Total_Revenue_USD"] in (0, np.nan)
    else pct_fmt((latest["Total_Revenue_USD"] / prev["Total_Revenue_USD"] - 1) * 100.0),
)

k2.metric(
    "Gross Margin (Latest)",
    pct_fmt(latest["Gross_Margin_%"]),
    None if prev is None else pct_fmt(latest["Gross_Margin_%"] - prev["Gross_Margin_%"]),
)

k3.metric(
    "New Customers (Latest)",
    f"{int(latest['New_Customers']):,}",
    None if prev is None else f"{int(latest['New_Customers'] - prev['New_Customers']):,}",
)

k4.metric(
    "Net Customers (Latest)",
    f"{int(latest['Net_Customers']):,}",
    None if prev is None else f"{int(latest['Net_Customers'] - prev['Net_Customers']):,}",
)

st.divider()

# -----------------------------
# Charts
# -----------------------------
st.subheader("Revenue Breakdown Over Time")
rev_df = fdf.set_index("Month")[["Subscription_Revenue_USD", "API_Revenue_USD"]]
st.area_chart(rev_df, use_container_width=True)

st.subheader("Selected Metrics (Line Chart)")
if selected_metrics:
    chart_df = fdf.set_index("Month")[selected_metrics]
    st.line_chart(chart_df, use_container_width=True)
else:
    st.info("Select at least one metric in the sidebar to display the line chart.")

st.subheader("Customer Movement")
cust_df = fdf.set_index("Month")[["New_Customers", "Churned_Customers", "Net_Customers"]]
st.bar_chart(cust_df, use_container_width=True)

# -----------------------------
# Data Table
# -----------------------------
if show_table:
    st.subheader("Filtered Data Table")
    st.dataframe(fdf, use_container_width=True, hide_index=True)

# -----------------------------
# Predictive Modeling
# -----------------------------
st.divider()
st.header("Predictive Modeling (Forecast)")

target = st.selectbox(
    "Select a metric to forecast",
    options=[
        "Total_Revenue_USD",
        "Subscription_Revenue_USD",
        "API_Revenue_USD",
        "Units",
        "New_Customers",
        "Churned_Customers",
        "Net_Customers",
        "Gross_Profit_USD",
        "Gross_Margin_%",
    ],
    index=0,
)

horizon = st.slider("Forecast horizon (months)", min_value=3, max_value=24, value=6)
alpha = st.slider("Model regularization (alpha)", min_value=0.1, max_value=50.0, value=1.0)
test_months = st.slider("Backtest window (months)", min_value=0, max_value=24, value=12)

forecast_df, metrics = fit_forecast(df, target_col=target, horizon=horizon, alpha=alpha, test_months=test_months)

m1, m2, m3 = st.columns(3)
m1.metric("Backtest months", f"{metrics.get('Backtest_Months', 0)}")
m2.metric("Backtest MAE", "—" if "MAE" not in metrics else f"{metrics['MAE']:,.0f}")
mape = metrics.get("MAPE")
m3.metric("Backtest MAPE", "—" if mape is None else f"{mape:,.2f}%")

st.subheader("Actual vs Fitted vs Forecast")
chart = forecast_df.set_index("Month")[["Actual", "Fitted", "Forecast"]]
st.line_chart(chart, use_container_width=True)

st.caption(
    "Forecast model: Ridge regression with time trend + monthly seasonality features. "
    "This is a lightweight baseline model suitable for dashboards."
)
