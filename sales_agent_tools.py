import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Optional, Dict, List
import json

# Load the combined dataset
df_combined = pd.read_csv("./data/combined_train_forecast.csv")
df_combined['date'] = pd.to_datetime(df_combined['date'])

# Global variable to store filtered data
_filtered_data = None

def _get_working_data():
    """Get the current working dataset (filtered or full)."""
    global _filtered_data
    if _filtered_data is not None and not _filtered_data.empty:
        return _filtered_data.copy()
    return df_combined.copy()

def _aggregate_by_date(df):
    """Aggregate data by date, summing sales across all dimensions."""
    agg_df = df.groupby('date').agg({
        'sales': 'sum',
        'yhat': 'sum',
        'error': 'sum',
        'percentage_error': 'mean'
    }).reset_index()
    return agg_df


def get_sales_data(start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
    """
    Get aggregated sales data from filtered dataset at date level.
    Use filter_the_data first to specify filtering criteria.
    
    Args:
        start_date: Start date in YYYY-MM-DD format (optional)
        end_date: End date in YYYY-MM-DD format (optional)
    
    Returns:
        String with sales data summary
    """
    df = _get_working_data()
    
    if df.empty:
        return "No data available. Please use filter_the_data first to specify filtering criteria."
    
    # Filter by date range if provided
    if start_date:
        df = df[df['date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['date'] <= pd.to_datetime(end_date)]
    
    if df.empty:
        return "No data found for the specified date range."
    
    # Aggregate by date
    agg_df = _aggregate_by_date(df)
    
    # Get filter info
    filter_info = "Filtered Data" if _filtered_data is not None else "All Data"
    
    result = f"""Sales Data Summary ({filter_info}):
Date Range: {agg_df['date'].min().date()} to {agg_df['date'].max().date()}
Total Days: {len(agg_df)}
Total Sales: ${agg_df['sales'].sum():,.2f}
Average Daily Sales: ${agg_df['sales'].mean():,.2f}
Min Daily Sales: ${agg_df['sales'].min():,.2f}
Max Daily Sales: ${agg_df['sales'].max():,.2f}
Std Deviation: ${agg_df['sales'].std():,.2f}

Top 5 Sales Days:
{agg_df.nlargest(5, 'sales')[['date', 'sales']].to_string(index=False)}
"""
    return result


def get_sales_forecast(start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
    """
    Get aggregated sales forecast data from filtered dataset at date level.
    Use filter_the_data first to specify filtering criteria.
    
    Args:
        start_date: Start date in YYYY-MM-DD format (optional)
        end_date: End date in YYYY-MM-DD format (optional)
    
    Returns:
        String with forecast data summary
    """
    df = _get_working_data()
    df = df[df['yhat'].notna()].copy()
    
    if df.empty:
        return "No forecast data available. Please use filter_the_data first or check if forecast data exists."
    
    # Filter by date range if provided
    if start_date:
        df = df[df['date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['date'] <= pd.to_datetime(end_date)]
    
    if df.empty:
        return "No forecast data found for the specified date range."
    
    # Aggregate by date
    agg_df = _aggregate_by_date(df)
    
    filter_info = "Filtered Data" if _filtered_data is not None else "All Data"
    
    result = f"""Forecast Data Summary ({filter_info}):
Forecast Period: {agg_df['date'].min().date()} to {agg_df['date'].max().date()}
Total Forecast Days: {len(agg_df)}
Actual Sales: ${agg_df['sales'].sum():,.2f}
Forecasted Sales (yhat): ${agg_df['yhat'].sum():,.2f}
Average Daily Actual: ${agg_df['sales'].mean():,.2f}
Average Daily Forecast: ${agg_df['yhat'].mean():,.2f}
Mean Absolute Error: ${agg_df['error'].abs().mean():,.2f}
Mean Percentage Error: {agg_df['percentage_error'].mean():.2f}%

Top 5 Forecast Days by Error:
{agg_df.nlargest(5, 'error', key=lambda x: abs(x))[['date', 'sales', 'yhat', 'error']].to_string(index=False)}
"""
    return result


def calculate_sales_growth(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    growth_type: Optional[str] = "monthly"
) -> str:
    """
    Calculate sales growth from filtered dataset at date level.
    Use filter_the_data first to specify filtering criteria.
    
    Args:
        start_date: Start date in YYYY-MM-DD format (optional)
        end_date: End date in YYYY-MM-DD format (optional)
        growth_type: Type of growth calculation - "monthly" (MoM), "quarterly" (QoQ), 
                     "yearly" (YoY), or "period" (start to end). Default: "monthly"
    
    Returns:
        String with sales growth analysis
    """
    df = _get_working_data()
    
    if df.empty:
        return "No data available. Please use filter_the_data first to specify filtering criteria."
    
    # Aggregate by date first
    agg_df = _aggregate_by_date(df)
    
    # Apply date filters
    if start_date:
        agg_df = agg_df[agg_df['date'] >= pd.to_datetime(start_date)]
    if end_date:
        agg_df = agg_df[agg_df['date'] <= pd.to_datetime(end_date)]
    
    if len(agg_df) < 2:
        return "Not enough data to calculate growth"
    
    # Sort by date
    agg_df = agg_df.sort_values('date')
    
    # Add period columns for aggregation
    agg_df['year'] = agg_df['date'].dt.year
    agg_df['quarter'] = agg_df['date'].dt.quarter
    agg_df['month'] = agg_df['date'].dt.month
    agg_df['year_month'] = agg_df['date'].dt.to_period('M')
    agg_df['year_quarter'] = agg_df['date'].dt.to_period('Q')
    
    # Determine aggregation level description
    level_desc = "Filtered Data" if _filtered_data is not None else "All Data"
    
    # Calculate growth based on growth_type
    growth_type = growth_type.lower() if growth_type else "monthly"
    
    if growth_type in ["monthly", "mom", "month"]:
        # Month-over-Month growth
        growth_df = agg_df.groupby('year_month')['sales'].sum().reset_index()
        growth_df.columns = ['period', 'sales']
        growth_df['sales_growth'] = growth_df['sales'].pct_change() * 100
        period_label = "Month-over-Month"
        
    elif growth_type in ["quarterly", "qoq", "quarter"]:
        # Quarter-over-Quarter growth
        growth_df = agg_df.groupby('year_quarter')['sales'].sum().reset_index()
        growth_df.columns = ['period', 'sales']
        growth_df['sales_growth'] = growth_df['sales'].pct_change() * 100
        period_label = "Quarter-over-Quarter"
        
    elif growth_type in ["yearly", "yoy", "year", "annual"]:
        # Year-over-Year growth
        growth_df = agg_df.groupby('year')['sales'].sum().reset_index()
        growth_df.columns = ['period', 'sales']
        growth_df['sales_growth'] = growth_df['sales'].pct_change() * 100
        period_label = "Year-over-Year"
        
    elif growth_type in ["period", "total", "overall"]:
        # Total period growth (start to end)
        first_sales = agg_df['sales'].iloc[0]
        last_sales = agg_df['sales'].iloc[-1]
        total_growth = ((last_sales - first_sales) / first_sales) * 100
        
        result = f"""Sales Growth Analysis ({level_desc}):
Period: {agg_df['date'].min().date()} to {agg_df['date'].max().date()}
Growth Type: Total Period Growth
Starting Sales: ${first_sales:,.2f}
Ending Sales: ${last_sales:,.2f}
Total Growth: {total_growth:.2f}%
Total Days: {len(agg_df)}
"""
        return result
    else:
        return f"Invalid growth_type: {growth_type}. Use 'monthly', 'quarterly', 'yearly', or 'period'."
    
    # Calculate summary statistics
    first_period_sales = growth_df['sales'].iloc[0]
    last_period_sales = growth_df['sales'].iloc[-1]
    total_growth = ((last_period_sales - first_period_sales) / first_period_sales) * 100
    
    # Get growth statistics (excluding NaN from first period)
    growth_stats = growth_df['sales_growth'].dropna()
    
    result = f"""Sales Growth Analysis ({level_desc}):
Period: {agg_df['date'].min().date()} to {agg_df['date'].max().date()}
Growth Type: {period_label}
Total Periods: {len(growth_df)}

SALES SUMMARY:
First Period Sales: ${first_period_sales:,.2f}
Last Period Sales: ${last_period_sales:,.2f}
Overall Growth: {total_growth:.2f}%

GROWTH RATE STATISTICS:
Average {period_label} Growth: {growth_stats.mean():.2f}%
Median Growth: {growth_stats.median():.2f}%
Max Growth: {growth_stats.max():.2f}%
Min Growth: {growth_stats.min():.2f}%
Std Deviation: {growth_stats.std():.2f}%

TOP 5 GROWTH PERIODS:
{growth_df.nlargest(5, 'sales_growth')[['period', 'sales', 'sales_growth']].to_string(index=False)}

BOTTOM 5 GROWTH PERIODS:
{growth_df.nsmallest(5, 'sales_growth')[['period', 'sales', 'sales_growth']].to_string(index=False)}
"""
    return result


def calculate_forecast_accuracy(start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
    """
    Calculate forecast accuracy from filtered dataset at date level.
    Use filter_the_data first to specify filtering criteria.
    
    Args:
        start_date: Start date in YYYY-MM-DD format (optional)
        end_date: End date in YYYY-MM-DD format (optional)
    
    Returns:
        String with forecast accuracy metrics
    """
    df = _get_working_data()
    df = df[df['yhat'].notna()].copy()
    
    if df.empty:
        return "No forecast data available. Please use filter_the_data first or check if forecast data exists."
    
    # Filter by date range if provided
    if start_date:
        df = df[df['date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['date'] <= pd.to_datetime(end_date)]
    
    if df.empty:
        return "No forecast data found for the specified date range."
    
    # Aggregate by date
    agg_df = _aggregate_by_date(df)
    
    # Calculate accuracy metrics
    mae = agg_df['error'].abs().mean()
    mape = agg_df['percentage_error'].abs().mean()
    rmse = (agg_df['error'] ** 2).mean() ** 0.5
    
    # Accuracy percentage
    accuracy = 100 - mape
    
    filter_info = "Filtered Data" if _filtered_data is not None else "All Data"
    
    result = f"""Forecast Accuracy ({filter_info}):
Forecast Period: {agg_df['date'].min().date()} to {agg_df['date'].max().date()}

Accuracy Metrics:
- Mean Absolute Error (MAE): ${mae:,.2f}
- Mean Absolute Percentage Error (MAPE): {mape:.2f}%
- Root Mean Square Error (RMSE): ${rmse:,.2f}
- Forecast Accuracy: {accuracy:.2f}%

Total Actual Sales: ${agg_df['sales'].sum():,.2f}
Total Forecasted Sales: ${agg_df['yhat'].sum():,.2f}
Total Forecast Error: ${agg_df['error'].sum():,.2f}

Worst 5 Days by Absolute Error:
{agg_df.nlargest(5, 'error', key=lambda x: abs(x))[['date', 'sales', 'yhat', 'error', 'percentage_error']].to_string(index=False)}
"""
    return result


def filter_the_data(city: Optional[str] = None, shop: Optional[str] = None, 
                    brand: Optional[str] = None, container: Optional[str] = None,
                    start_date: Optional[str] = None, end_date: Optional[str] = None,
                    reset_filter: Optional[bool] = False) -> str:
    """
    PRIMARY FILTERING TOOL: Filter the data based on given criteria and store for other tools.
    All other tools will use this filtered data. Call this first before using other analysis tools.
    
    Args:
        city: City name to filter by (optional)
        shop: Shop name to filter by (optional)
        brand: Brand name to filter by (optional)
        container: Container type to filter by (optional)
        start_date: Start date in YYYY-MM-DD format (optional)
        end_date: End date in YYYY-MM-DD format (optional)
        reset_filter: Set to True to clear filters and work with all data (optional)
    
    Returns:
        String with filtered data summary
    """
    global _filtered_data
    
    # Reset filter if requested
    if reset_filter:
        _filtered_data = None
        return "Filter cleared. All tools will now use the complete dataset."
    
    df = df_combined.copy()
    
    filters_applied = []
    
    if city:
        df = df[df['city'] == city]
        filters_applied.append(f"City: {city}")
    
    if shop:
        df = df[df['shop'] == shop]
        filters_applied.append(f"Shop: {shop}")
    
    if brand:
        df = df[df['brand'] == brand]
        filters_applied.append(f"Brand: {brand}")
    
    if container:
        df = df[df['container'] == container]
        filters_applied.append(f"Container: {container}")
    
    if start_date:
        df = df[df['date'] >= pd.to_datetime(start_date)]
        filters_applied.append(f"Start Date: {start_date}")
    
    if end_date:
        df = df[df['date'] <= pd.to_datetime(end_date)]
        filters_applied.append(f"End Date: {end_date}")
    
    if df.empty:
        _filtered_data = None
        return "No data found matching the specified criteria. Filter not applied."
    
    # Store filtered data globally for other tools to use
    _filtered_data = df
    
    unique_ts = df['timeseries_id'].nunique()
    unique_cities = df['city'].nunique()
    unique_shops = df['shop'].nunique()
    unique_brands = df['brand'].nunique()
    unique_containers = df['container'].nunique()
    
    # Aggregate by date for summary
    agg_df = _aggregate_by_date(df)
    
    result = f"""✅ Data Filtered Successfully!
Filters Applied: {', '.join(filters_applied) if filters_applied else 'None (Using all data)'}

FILTERED DATASET SUMMARY:
- Unique Time Series: {unique_ts}
- Unique Cities: {unique_cities}
- Unique Shops: {unique_shops}
- Unique Brands: {unique_brands}
- Unique Containers: {unique_containers}
- Total Records: {len(df)}
- Date Range: {df['date'].min().date()} to {df['date'].max().date()}

AGGREGATED BY DATE:
- Total Days: {len(agg_df)}
- Total Sales: ${agg_df['sales'].sum():,.2f}
- Average Daily Sales: ${agg_df['sales'].mean():,.2f}

Top 5 Product Combinations by Total Sales:
{df.groupby(['city', 'shop', 'brand', 'container'])['sales'].sum().nlargest(5).to_string()}

💡 This filtered data will be used by all subsequent analysis tools.
"""
    return result


def plot_the_data(start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
    """
    Plot aggregated sales data from filtered dataset at date level.
    Use filter_the_data first to specify filtering criteria.
    
    Args:
        start_date: Start date in YYYY-MM-DD format (optional)
        end_date: End date in YYYY-MM-DD format (optional)
    
    Returns:
        String confirming plot creation
    """
    df = _get_working_data()
    
    if df.empty:
        return "No data available. Please use filter_the_data first to specify filtering criteria."
    
    # Apply date filters
    if start_date:
        df = df[df['date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['date'] <= pd.to_datetime(end_date)]
    
    if df.empty:
        return "No data found for the specified date range."
    
    # Aggregate by date
    agg_df = _aggregate_by_date(df)
    agg_df = agg_df.sort_values('date')
    
    # Create plot
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Separate training and forecast data
    train_data = agg_df[agg_df['yhat'].isna()]
    forecast_data = agg_df[agg_df['yhat'].notna()]
    
    filter_info = "Filtered Data" if _filtered_data is not None else "All Data"
    title = f"Sales Over Time ({filter_info})"
    
    # Plot training data
    if not train_data.empty:
        ax.plot(train_data['date'], train_data['sales'], label='Actual Sales (Historical)', 
                color='blue', linewidth=2, marker='o', markersize=4)
    
    # Plot forecast data
    if not forecast_data.empty:
        ax.plot(forecast_data['date'], forecast_data['sales'], label='Actual Sales (Forecast Period)', 
                color='green', linewidth=2, marker='o', markersize=4)
        ax.plot(forecast_data['date'], forecast_data['yhat'], label='Forecasted Sales', 
                color='red', linewidth=2, linestyle='--', marker='s', markersize=4)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Daily Sales ($)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return f"Plot created successfully. Showing {len(agg_df)} days of aggregated data."


def create_report(start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
    """
    Create a comprehensive report from filtered dataset at date level.
    Use filter_the_data first to specify filtering criteria.
    
    Args:
        start_date: Start date in YYYY-MM-DD format (optional)
        end_date: End date in YYYY-MM-DD format (optional)
    
    Returns:
        String with comprehensive report
    """
    df = _get_working_data()
    
    if df.empty:
        return "No data available. Please use filter_the_data first to specify filtering criteria."
    
    # Apply date filters
    if start_date:
        df = df[df['date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['date'] <= pd.to_datetime(end_date)]
    
    if df.empty:
        return "No data found for the specified date range."
    
    # Aggregate by date
    agg_df = _aggregate_by_date(df)
    
    # Calculate key metrics
    total_sales = agg_df['sales'].sum()
    avg_daily_sales = agg_df['sales'].mean()
    
    # Forecast data
    forecast_df = agg_df[agg_df['yhat'].notna()]
    
    filter_info = "Filtered Data" if _filtered_data is not None else "All Data"
    
    report = f"""
═══════════════════════════════════════════════════════════════
                    SALES ANALYSIS REPORT
═══════════════════════════════════════════════════════════════

REPORT SCOPE: {filter_info}
Date Range: {agg_df['date'].min().date()} to {agg_df['date'].max().date()}

───────────────────────────────────────────────────────────────
SALES SUMMARY (AGGREGATED BY DATE):
───────────────────────────────────────────────────────────────
Total Days: {len(agg_df)}
Unique Time Series in Filter: {df['timeseries_id'].nunique()}
Total Sales: ${total_sales:,.2f}
Average Daily Sales: ${avg_daily_sales:,.2f}
Minimum Daily Sales: ${agg_df['sales'].min():,.2f}
Maximum Daily Sales: ${agg_df['sales'].max():,.2f}
Standard Deviation: ${agg_df['sales'].std():,.2f}

───────────────────────────────────────────────────────────────
TOP PERFORMING CATEGORIES:
───────────────────────────────────────────────────────────────
By City:
{df.groupby('city')['sales'].sum().nlargest(5).to_string()}

By Brand:
{df.groupby('brand')['sales'].sum().nlargest(5).to_string()}

By Container:
{df.groupby('container')['sales'].sum().nlargest(5).to_string()}
"""
    
    if not forecast_df.empty:
        mae = forecast_df['error'].abs().mean()
        mape = forecast_df['percentage_error'].abs().mean()
        accuracy = 100 - mape
        
        report += f"""
───────────────────────────────────────────────────────────────
FORECAST PERFORMANCE:
───────────────────────────────────────────────────────────────
Forecast Days: {len(forecast_df)}
Total Actual Sales: ${forecast_df['sales'].sum():,.2f}
Total Forecasted Sales: ${forecast_df['yhat'].sum():,.2f}
Mean Absolute Error (Daily): ${mae:,.2f}
Mean Absolute Percentage Error: {mape:.2f}%
Forecast Accuracy: {accuracy:.2f}%
"""
    
    report += "\n═══════════════════════════════════════════════════════════════\n"
    
    return report
