import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Optional, Dict, List

# Load the combined dataset
df_combined = pd.read_csv("./data/combined_train_forecast.csv")
df_combined['date'] = pd.to_datetime(df_combined['date'])


def get_sales_data(timeseries_id: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
    """
    Get sales data for a given timeseries_id and optional date range.
    
    Args:
        timeseries_id: The unique identifier for the time series
        start_date: Start date in YYYY-MM-DD format (optional)
        end_date: End date in YYYY-MM-DD format (optional)
    
    Returns:
        String with sales data summary
    """
    df = df_combined[df_combined['timeseries_id'] == timeseries_id].copy()
    
    if df.empty:
        return f"No data found for timeseries_id: {timeseries_id}"
    
    # Filter by date range if provided
    if start_date:
        df = df[df['date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['date'] <= pd.to_datetime(end_date)]
    
    # Get metadata
    metadata = df[['city', 'shop', 'brand', 'container']].iloc[0]
    
    result = f"""Sales Data for {timeseries_id}:
Location: {metadata['city']} - {metadata['shop']}
Product: {metadata['brand']} - {metadata['container']}
Date Range: {df['date'].min().date()} to {df['date'].max().date()}
Total Records: {len(df)}
Total Sales: ${df['sales'].sum():,.2f}
Average Sales: ${df['sales'].mean():,.2f}
Min Sales: ${df['sales'].min():,.2f}
Max Sales: ${df['sales'].max():,.2f}
"""
    return result


def get_sales_forecast(timeseries_id: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
    """
    Get sales forecast data for a given timeseries_id and optional date range.
    
    Args:
        timeseries_id: The unique identifier for the time series
        start_date: Start date in YYYY-MM-DD format (optional)
        end_date: End date in YYYY-MM-DD format (optional)
    
    Returns:
        String with forecast data summary
    """
    df = df_combined[
        (df_combined['timeseries_id'] == timeseries_id) & 
        (df_combined['yhat'].notna())
    ].copy()
    
    if df.empty:
        return f"No forecast data found for timeseries_id: {timeseries_id}"
    
    # Filter by date range if provided
    if start_date:
        df = df[df['date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['date'] <= pd.to_datetime(end_date)]
    
    metadata = df[['city', 'shop', 'brand', 'container']].iloc[0]
    
    result = f"""Forecast Data for {timeseries_id}:
Location: {metadata['city']} - {metadata['shop']}
Product: {metadata['brand']} - {metadata['container']}
Forecast Period: {df['date'].min().date()} to {df['date'].max().date()}
Total Forecast Records: {len(df)}
Actual Sales: ${df['sales'].sum():,.2f}
Forecasted Sales (yhat): ${df['yhat'].sum():,.2f}
Average Actual: ${df['sales'].mean():,.2f}
Average Forecast: ${df['yhat'].mean():,.2f}
Mean Absolute Error: ${df['error'].abs().mean():,.2f}
Mean Percentage Error: {df['percentage_error'].mean():.2f}%
"""
    return result


def calculate_sales_growth(
    timeseries_id: Optional[str] = None,
    city: Optional[str] = None,
    shop: Optional[str] = None,
    brand: Optional[str] = None,
    container: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    growth_type: Optional[str] = "monthly"
) -> str:
    """
    Calculate sales growth at any aggregation level with different growth rate types.
    
    Args:
        timeseries_id: The unique identifier for a specific time series (optional)
        city: City name to filter by (optional)
        shop: Shop name to filter by (optional)
        brand: Brand name to filter by (optional)
        container: Container type to filter by (optional)
        start_date: Start date in YYYY-MM-DD format (optional)
        end_date: End date in YYYY-MM-DD format (optional)
        growth_type: Type of growth calculation - "monthly" (MoM), "quarterly" (QoQ), 
                     "yearly" (YoY), or "period" (start to end). Default: "monthly"
    
    Returns:
        String with sales growth analysis
    """
    # Filter data based on provided criteria
    if timeseries_id:
        df = df_combined[df_combined['timeseries_id'] == timeseries_id].copy()
    else:
        df = df_combined.copy()
        if city:
            df = df[df['city'] == city]
        if shop:
            df = df[df['shop'] == shop]
        if brand:
            df = df[df['brand'] == brand]
        if container:
            df = df[df['container'] == container]
    
    if df.empty:
        return "No data found for the specified criteria"
    
    # Apply date filters
    if start_date:
        df = df[df['date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['date'] <= pd.to_datetime(end_date)]
    
    if len(df) < 2:
        return "Not enough data to calculate growth"
    
    # Sort by date
    df = df.sort_values('date')
    
    # Add period columns for aggregation
    df['year'] = df['date'].dt.year
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year_month'] = df['date'].dt.to_period('M')
    df['year_quarter'] = df['date'].dt.to_period('Q')
    
    # Determine aggregation level description
    filters = []
    if timeseries_id:
        metadata = df[['city', 'shop', 'brand', 'container']].iloc[0]
        level_desc = f"Time Series: {metadata['city']} - {metadata['shop']} - {metadata['brand']} - {metadata['container']}"
    else:
        if city:
            filters.append(f"City: {city}")
        if shop:
            filters.append(f"Shop: {shop}")
        if brand:
            filters.append(f"Brand: {brand}")
        if container:
            filters.append(f"Container: {container}")
        level_desc = ", ".join(filters) if filters else "All Data"
    
    # Calculate growth based on growth_type
    growth_type = growth_type.lower() if growth_type else "monthly"
    
    if growth_type in ["monthly", "mom", "month"]:
        # Month-over-Month growth
        agg_df = df.groupby('year_month')['sales'].sum().reset_index()
        agg_df.columns = ['period', 'sales']
        agg_df['sales_growth'] = agg_df['sales'].pct_change() * 100
        period_label = "Month-over-Month"
        
    elif growth_type in ["quarterly", "qoq", "quarter"]:
        # Quarter-over-Quarter growth
        agg_df = df.groupby('year_quarter')['sales'].sum().reset_index()
        agg_df.columns = ['period', 'sales']
        agg_df['sales_growth'] = agg_df['sales'].pct_change() * 100
        period_label = "Quarter-over-Quarter"
        
    elif growth_type in ["yearly", "yoy", "year", "annual"]:
        # Year-over-Year growth
        agg_df = df.groupby('year')['sales'].sum().reset_index()
        agg_df.columns = ['period', 'sales']
        agg_df['sales_growth'] = agg_df['sales'].pct_change() * 100
        period_label = "Year-over-Year"
        
    elif growth_type in ["period", "total", "overall"]:
        # Total period growth (start to end)
        first_sales = df['sales'].iloc[0]
        last_sales = df['sales'].iloc[-1]
        total_growth = ((last_sales - first_sales) / first_sales) * 100
        
        result = f"""Sales Growth Analysis ({level_desc}):
Period: {df['date'].min().date()} to {df['date'].max().date()}
Growth Type: Total Period Growth
Starting Sales: ${first_sales:,.2f}
Ending Sales: ${last_sales:,.2f}
Total Growth: {total_growth:.2f}%
Total Records: {len(df)}
"""
        return result
    else:
        return f"Invalid growth_type: {growth_type}. Use 'monthly', 'quarterly', 'yearly', or 'period'."
    
    # Calculate summary statistics
    first_period_sales = agg_df['sales'].iloc[0]
    last_period_sales = agg_df['sales'].iloc[-1]
    total_growth = ((last_period_sales - first_period_sales) / first_period_sales) * 100
    
    # Get growth statistics (excluding NaN from first period)
    growth_stats = agg_df['sales_growth'].dropna()
    
    result = f"""Sales Growth Analysis ({level_desc}):
Period: {df['date'].min().date()} to {df['date'].max().date()}
Growth Type: {period_label}
Total Periods: {len(agg_df)}

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
{agg_df.nlargest(5, 'sales_growth')[['period', 'sales', 'sales_growth']].to_string(index=False)}

BOTTOM 5 GROWTH PERIODS:
{agg_df.nsmallest(5, 'sales_growth')[['period', 'sales', 'sales_growth']].to_string(index=False)}
"""
    return result


def calculate_forecast_accuracy(timeseries_id: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
    """
    Calculate forecast accuracy for a given timeseries_id and optional date range.
    
    Args:
        timeseries_id: The unique identifier for the time series
        start_date: Start date in YYYY-MM-DD format (optional)
        end_date: End date in YYYY-MM-DD format (optional)
    
    Returns:
        String with forecast accuracy metrics
    """
    df = df_combined[
        (df_combined['timeseries_id'] == timeseries_id) & 
        (df_combined['yhat'].notna())
    ].copy()
    
    if df.empty:
        return f"No forecast data found for timeseries_id: {timeseries_id}"
    
    # Filter by date range if provided
    if start_date:
        df = df[df['date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['date'] <= pd.to_datetime(end_date)]
    
    metadata = df[['city', 'shop', 'brand', 'container']].iloc[0]
    
    # Calculate accuracy metrics
    mae = df['error'].abs().mean()
    mape = df['percentage_error'].abs().mean()
    rmse = (df['error'] ** 2).mean() ** 0.5
    
    # Accuracy percentage
    accuracy = 100 - mape
    
    result = f"""Forecast Accuracy for {timeseries_id}:
Location: {metadata['city']} - {metadata['shop']}
Product: {metadata['brand']} - {metadata['container']}
Forecast Period: {df['date'].min().date()} to {df['date'].max().date()}

Accuracy Metrics:
- Mean Absolute Error (MAE): ${mae:,.2f}
- Mean Absolute Percentage Error (MAPE): {mape:.2f}%
- Root Mean Square Error (RMSE): ${rmse:,.2f}
- Forecast Accuracy: {accuracy:.2f}%

Total Actual Sales: ${df['sales'].sum():,.2f}
Total Forecasted Sales: ${df['yhat'].sum():,.2f}
Total Forecast Error: ${df['error'].sum():,.2f}
"""
    return result


def filter_the_data(city: Optional[str] = None, shop: Optional[str] = None, 
                    brand: Optional[str] = None, container: Optional[str] = None,
                    start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
    """
    Filter the data based on given criteria.
    
    Args:
        city: City name to filter by (optional)
        shop: Shop name to filter by (optional)
        brand: Brand name to filter by (optional)
        container: Container type to filter by (optional)
        start_date: Start date in YYYY-MM-DD format (optional)
        end_date: End date in YYYY-MM-DD format (optional)
    
    Returns:
        String with filtered data summary
    """
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
        return "No data found matching the specified criteria"
    
    unique_ts = df['timeseries_id'].nunique()
    
    result = f"""Filtered Data Summary:
Filters Applied: {', '.join(filters_applied) if filters_applied else 'None'}

Results:
- Unique Time Series: {unique_ts}
- Total Records: {len(df)}
- Date Range: {df['date'].min().date()} to {df['date'].max().date()}
- Total Sales: ${df['sales'].sum():,.2f}
- Average Sales: ${df['sales'].mean():,.2f}

Top 5 Time Series by Sales:
{df.groupby(['city', 'shop', 'brand', 'container'])['sales'].sum().nlargest(5).to_string()}
"""
    return result


def plot_the_data(timeseries_id: Optional[str] = None, city: Optional[str] = None, 
                  shop: Optional[str] = None, brand: Optional[str] = None, 
                  container: Optional[str] = None) -> str:
    """
    Plot the data based on given criteria.
    
    Args:
        timeseries_id: Specific time series ID to plot (optional)
        city: City name to filter by (optional)
        shop: Shop name to filter by (optional)
        brand: Brand name to filter by (optional)
        container: Container type to filter by (optional)
    
    Returns:
        String confirming plot creation
    """
    if timeseries_id:
        df = df_combined[df_combined['timeseries_id'] == timeseries_id].copy()
        if df.empty:
            return f"No data found for timeseries_id: {timeseries_id}"
    else:
        df = df_combined.copy()
        if city:
            df = df[df['city'] == city]
        if shop:
            df = df[df['shop'] == shop]
        if brand:
            df = df[df['brand'] == brand]
        if container:
            df = df[df['container'] == container]
    
    if df.empty:
        return "No data found matching the specified criteria"
    
    df = df.sort_values('date')
    
    # Create plot
    fig, ax = plt.subplots(figsize=(15, 6))
    
    if timeseries_id:
        # Single time series plot
        metadata = df[['city', 'shop', 'brand', 'container']].iloc[0]
        title = f"{metadata['city']} - {metadata['shop']} - {metadata['brand']} - {metadata['container']}"
        
        train_data = df[df['yhat'].isna()]
        forecast_data = df[df['yhat'].notna()]
        
        ax.plot(train_data['date'], train_data['sales'], label='Actual Sales (Train)', 
                color='blue', linewidth=2, marker='o', markersize=3)
        
        if not forecast_data.empty:
            ax.plot(forecast_data['date'], forecast_data['sales'], label='Actual Sales (Test)', 
                    color='green', linewidth=2, marker='o', markersize=3)
            ax.plot(forecast_data['date'], forecast_data['yhat'], label='Forecast', 
                    color='red', linewidth=2, linestyle='--', marker='s', markersize=3)
    else:
        # Aggregate plot
        agg_data = df.groupby('date')['sales'].sum().reset_index()
        ax.plot(agg_data['date'], agg_data['sales'], label='Total Sales', 
                color='blue', linewidth=2, marker='o', markersize=3)
        title = "Aggregated Sales Data"
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Sales ($)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return f"Plot created successfully for the specified criteria. Showing {len(df)} records."


def create_report(timeseries_id: Optional[str] = None, city: Optional[str] = None,
                  shop: Optional[str] = None, brand: Optional[str] = None,
                  start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
    """
    Create a comprehensive report based on given criteria.
    
    Args:
        timeseries_id: Specific time series ID (optional)
        city: City name to filter by (optional)
        shop: Shop name to filter by (optional)
        brand: Brand name to filter by (optional)
        start_date: Start date in YYYY-MM-DD format (optional)
        end_date: End date in YYYY-MM-DD format (optional)
    
    Returns:
        String with comprehensive report
    """
    if timeseries_id:
        df = df_combined[df_combined['timeseries_id'] == timeseries_id].copy()
    else:
        df = df_combined.copy()
        if city:
            df = df[df['city'] == city]
        if shop:
            df = df[df['shop'] == shop]
        if brand:
            df = df[df['brand'] == brand]
    
    if start_date:
        df = df[df['date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['date'] <= pd.to_datetime(end_date)]
    
    if df.empty:
        return "No data found matching the specified criteria"
    
    # Calculate key metrics
    total_sales = df['sales'].sum()
    avg_sales = df['sales'].mean()
    
    # Forecast data
    forecast_df = df[df['yhat'].notna()]
    
    report = f"""
═══════════════════════════════════════════════════════════════
                    SALES ANALYSIS REPORT
═══════════════════════════════════════════════════════════════

REPORT CRITERIA:
{f'Time Series ID: {timeseries_id}' if timeseries_id else ''}
{f'City: {city}' if city else ''}
{f'Shop: {shop}' if shop else ''}
{f'Brand: {brand}' if brand else ''}
Date Range: {df['date'].min().date()} to {df['date'].max().date()}

───────────────────────────────────────────────────────────────
SALES SUMMARY:
───────────────────────────────────────────────────────────────
Total Records: {len(df)}
Unique Time Series: {df['timeseries_id'].nunique()}
Total Sales: ${total_sales:,.2f}
Average Sales: ${avg_sales:,.2f}
Minimum Sales: ${df['sales'].min():,.2f}
Maximum Sales: ${df['sales'].max():,.2f}
Standard Deviation: ${df['sales'].std():,.2f}

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
Forecast Records: {len(forecast_df)}
Total Actual Sales: ${forecast_df['sales'].sum():,.2f}
Total Forecasted Sales: ${forecast_df['yhat'].sum():,.2f}
Mean Absolute Error: ${mae:,.2f}
Mean Absolute Percentage Error: {mape:.2f}%
Forecast Accuracy: {accuracy:.2f}%
"""
    
    report += "\n═══════════════════════════════════════════════════════════════\n"
    
    return report
