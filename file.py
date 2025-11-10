# sales_forecasting.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from datetime import datetime, timedelta

# ----------------------------------------------------------
#  STEP 1: Create or Load Sales Data
# ----------------------------------------------------------
# Simulated monthly sales data for 3 years
dates = pd.date_range(start="2021-01-01", periods=36, freq="M")
sales = np.random.randint(200, 500, size=len(dates)) + np.linspace(0, 100, len(dates))
df = pd.DataFrame({"Date": dates, "Sales": sales})

print(" Sample Sales Data:\n", df.head())

# Plot data
plt.figure(figsize=(10, 5))
plt.plot(df["Date"], df["Sales"], marker="o")
plt.title(" Monthly Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.show()

# ----------------------------------------------------------
# STEP 2: Forecast using ARIMA
# ----------------------------------------------------------
print("\n🔹 ARIMA Model Forecast")

# Convert Date to index for ARIMA
df_arima = df.copy()
df_arima.set_index("Date", inplace=True)

# Build and fit ARIMA model (p, d, q)
model = ARIMA(df_arima["Sales"], order=(1, 1, 1))
model_fit = model.fit()

# Forecast next 6 months
forecast_steps = 6
forecast = model_fit.forecast(steps=forecast_steps)

future_dates = pd.date_range(df_arima.index[-1] + timedelta(days=30), periods=forecast_steps, freq="M")

plt.figure(figsize=(10, 5))
plt.plot(df_arima.index, df_arima["Sales"], label="Historical Sales")
plt.plot(future_dates, forecast, label="ARIMA Forecast", color="red")
plt.legend()
plt.title(" ARIMA Sales Forecast")
plt.show()

print("\n ARIMA Forecasted Sales:")
for d, val in zip(future_dates, forecast):
    print(f"{d.date()} → {round(val, 2)}")

# ----------------------------------------------------------
#  STEP 3: Forecast using Prophet
# ----------------------------------------------------------
print("\n Prophet Model Forecast")

# Prepare data for Prophet
df_prophet = df.rename(columns={"Date": "ds", "Sales": "y"})

# Build Prophet model
m = Prophet()
m.fit(df_prophet)

# Create future dataframe
future = m.make_future_dataframe(periods=6, freq="M")

# Predict
forecast_prophet = m.predict(future)

# Plot Prophet forecast
m.plot(forecast_prophet)
plt.title(" Prophet Sales Forecast")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.show()

# Show future predictions
print("\n Prophet Forecasted Sales:")
print(forecast_prophet[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(6))
