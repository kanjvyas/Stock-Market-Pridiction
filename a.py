import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Set page layout
st.set_page_config(layout="wide")

# Title and description
st.title("Stock Market Prediction")
st.write("Fetch stock data and predict prices using an optimized XGBoost model.")

# User input for stock ticker
ticker = st.text_input("Enter Stock Ticker (e.g., AMZN,NVDA): ", "NVDA")

# Load stock data
data = yf.download(ticker, period="2y")

# Check if data was loaded
if data.empty:
    st.error("No data available for this ticker. Please try another.")
else:
    # Select relevant columns
    data = data[['Open', 'Close', 'High', 'Low']]
    
    # Display raw data
    st.subheader("Stock Data")
    st.write(data)

    # Fill missing data
    data.fillna(method='ffill', inplace=True)

    # Add date as a feature
    data['Date'] = data.index
    data['Day'] = data['Date'].dt.dayofweek

  # Convert currency (assuming a fixed exchange rate, e.g., 1 USD = 83 INR)
    exchange_rate = 83  # Example conversion rate (USD to INR)
    data[['Open', 'Close', 'High', 'Low']] = data[['Open', 'Close', 'High', 'Low']] * exchange_rate

    # Feature engineering
    data['MA_5'] = data['Close'].rolling(window=5).mean()
    data['MA_10'] = data['Close'].rolling(window=10).mean()
    data['Target'] = data['Close'].shift(-1)
    data.dropna(inplace=True)

    # Check if there are enough rows for train-test split
    if len(data) < 2:
        st.error("Not enough data after processing for training. Please try a different ticker or time period.")
    else:
        # Define features and target
        X = data[['Open', 'High', 'Low', 'Close', 'MA_5', 'MA_10', 'Day']]
        y = data['Target']

        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Model training with tuned parameters
        model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42)
        model.fit(X_train, y_train)

        # Model prediction
        y_pred = model.predict(X_test)

        # Calculate evaluation metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # Forecast for the next 10 days
        future_days = 10
        last_data = X_scaled[-1].reshape(1, -1)
        future_predictions = []
        future_dates = [data.index[-1] + timedelta(days=i) for i in range(1, future_days + 1)]
        for _ in range(future_days):
            pred_price = model.predict(last_data)
            future_predictions.append(pred_price[0])
            new_row = np.append(last_data[:, 1:], scaler.transform([[0, 0, 0, pred_price[0], 0, 0, 0]])[:, 0]).reshape(1, -1)
            last_data = new_row

        # Display metrics
        st.subheader("Model Performance")
        st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        st.write(f"R² Score: {r2:.2f}")
        st.progress(r2)

        # Plot actual vs predicted prices
        st.subheader("Closing Price Prediction")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=data.index[-len(y_test):], y=y_test, mode='lines', name='Actual Price'))
        fig1.add_trace(go.Scatter(x=data.index[-len(y_test):], y=y_pred, mode='lines', name='Predicted Price'))
        fig1.update_layout(title="Actual vs Predicted Prices", xaxis_title="Date", yaxis_title="Price (INR)")
        st.plotly_chart(fig1)

        # Plot forecasted prices
        st.subheader("Next 10-Day Forecast")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=future_dates, y=future_predictions, mode='lines+markers', name='Forecasted Prices'))
        fig2.update_layout(title="Forecast for Next 10 Days", xaxis_title="Date", yaxis_title="Price (INR)")
        st.plotly_chart(fig2)

        # Display forecasted price table
        st.subheader("Forecasted Prices")
        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Forecasted Price (INR)": future_predictions
        })
        st.write(forecast_df)
