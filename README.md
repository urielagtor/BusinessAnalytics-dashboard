# Business Analytics Dashboard

A comprehensive revenue and customer analytics dashboard built with Streamlit, featuring interactive visualizations and predictive forecasting capabilities.

## Features

- **Key Performance Indicators (KPIs)**: Real-time display of Total Revenue, Gross Margin, New Customers, and Net Customers with month-over-month comparisons
- **Revenue Breakdown**: Stacked area charts showing Subscription vs API revenue trends
- **Customer Analytics**: Grouped bar charts visualizing New, Churned, and Net customer movements
- **Custom Metric Visualization**: Select and compare multiple metrics including revenue, margins, and customer data
- **Predictive Modeling**: Ridge regression-based forecasting with configurable horizon, regularization, and backtesting parameters
- **Interactive Filters**: Date range slider and metric selection for focused analysis

## Requirements

- Python 3.8+
- streamlit
- pandas
- numpy
- plotly
- scikit-learn
- altair

## Installation

1. Clone the repository

   ```bash
   git clone <repository-url>
   cd BusinessAnalytics-dashboard
   ```

2. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

3. Ensure your data file exists at `data/gdp_data.csv` with the following required columns:
   - `Month` (format: YYYY-MM)
   - `Total_Revenue_USD`
   - `Subscription_Revenue_USD`
   - `API_Revenue_USD`
   - `Units`
   - `New_Customers`
   - `Churned_Customers`
   - `Gross_Margin_%`

## Usage

Run the dashboard locally:

```bash
streamlit run streamlit_app.py
```

Or with custom server settings:

```bash
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

## Dashboard Sections

### Sidebar Controls
- **Month Range**: Filter data by date range
- **Metric Selection**: Choose which metrics to display in the line chart
- **Data Table Toggle**: Show/hide the raw data table

### Visualizations
- **Revenue Breakdown**: Stacked area chart of subscription vs API revenue
- **Selected Metrics**: Multi-line chart for comparing chosen metrics
- **Customer Movement**: Grouped bar chart for customer acquisition and churn analysis

### Predictive Modeling
- Select any metric for forecasting
- Configure forecast horizon (3-24 months)
- Adjust model regularization (alpha parameter)
- Set backtest window for model validation
- View MAE and MAPE accuracy metrics
