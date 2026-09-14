import pandas as pd
import numpy as np
import uuid
from hashlib import md5

# Read the training data
df_train = pd.read_csv('data/train.csv')

# Read the forecast data
df_forecast = pd.read_csv('data/forecast_results.csv')

# Calculate sales for training data (price * quantity)
df_train['sales'] = df_train['price'] * df_train['quantity']

# Convert date columns to datetime for proper handling
df_train['date'] = pd.to_datetime(df_train['date'], format='%d/%m/%y')
df_forecast['date'] = pd.to_datetime(df_forecast['date'])

# Prepare training data with required columns
df_train_prepared = df_train[['date', 'city', 'shop', 'brand', 'container', 'sales']].copy()
df_train_prepared['yhat'] = np.nan  # yhat is NaN for training data
df_train_prepared['error'] = np.nan  # error is NaN for training data
df_train_prepared['percentage_error'] = np.nan  # percentage_error is NaN for training data

# Prepare forecast data with required columns
df_forecast_prepared = df_forecast[['date', 'city', 'shop', 'brand', 'container', 'sales', 
                                     'yhat', 'error', 'percentage_error']].copy()

# Concatenate training and forecast data
df_combined = pd.concat([df_train_prepared, df_forecast_prepared], ignore_index=True)

# Drop rows with NaN in date column
df_combined = df_combined.dropna(subset=['date'])

# Sort by date for better organization
df_combined = df_combined.sort_values(['date', 'city', 'shop', 'brand', 'container']).reset_index(drop=True)

# Create UUID for each time series at city-shop-brand-container level
def generate_timeseries_uuid(row):
    """Generate a consistent UUID based on time series attributes"""
    # Create a unique string from the time series identifiers
    ts_string = f"{row['city']}_{row['shop']}_{row['brand']}_{row['container']}"
    # Use MD5 hash to create a consistent UUID
    hash_object = md5(ts_string.encode())
    # Create UUID from hash
    return str(uuid.UUID(hash_object.hexdigest()))

df_combined['timeseries_id'] = df_combined.apply(generate_timeseries_uuid, axis=1)

# Reorder columns to have timeseries_id at the beginning
cols = ['timeseries_id', 'date', 'city', 'shop', 'brand', 'container', 'sales', 'yhat', 'error', 'percentage_error']
df_combined = df_combined[cols]

# Display summary information
print("Combined Dataset Shape:", df_combined.shape)
print("\nColumns:", df_combined.columns.tolist())
print("\nUnique Time Series Count:", df_combined['timeseries_id'].nunique())
print("\nFirst few rows:")
print(df_combined.head(10))
print("\nLast few rows (forecast data):")
print(df_combined.tail(10))
print("\nSample time series IDs:")
print(df_combined.groupby(['city', 'shop', 'brand', 'container'])['timeseries_id'].first().head(10))
print("\nData Info:")
print(df_combined.info())
print("\nNull values per column:")
print(df_combined.isnull().sum())

# Save the combined dataset
df_combined.to_csv('./data/combined_train_forecast.csv', index=False)
print("\nCombined dataset saved to 'data/combined_train_forecast.csv'")
