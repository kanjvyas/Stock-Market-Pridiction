# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import numpy as np
# from xgboost import XGBRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt

# # Set page layout
# st.set_page_config(layout="wide")

# # Title and description
# st.title("Stock Market Prediction ")
# st.write("This app fetches stock data and predicts the  price using the XGBoost model.")

# # User input for stock ticker
# ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA): ", "AAPL")

# # Load stock data
# data = yf.download(ticker, period="1y")
# data = data[['Open', 'Close', 'High', 'Low']]  # Select relevant columns

# # Display raw data
# st.subheader("Stock Data")
# st.write(data)

# # Convert currency (assuming a fixed exchange rate, e.g., 1 USD = 83 INR)
# exchange_rate = 83  # Example conversion rate (USD to INR)
# data[['Open', 'Close', 'High', 'Low']] = data[['Open', 'Close', 'High', 'Low']] * exchange_rate

# # Feature Engineering
# data['Target'] = data['Close'].shift(-1)  # Shift close price to create target variable
# data.dropna(inplace=True)  # Drop missing values

# # Split data into features and target
# X = data[['Open', 'High', 'Low', 'Close']]
# y = data['Target']

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Model training
# model = XGBRegressor()
# model.fit(X_train, y_train)

# # Model prediction
# y_pred = model.predict(X_test)

# # Calculate model accuracy
# mae = mean_absolute_error(y_test, y_pred)
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# r2 = r2_score(y_test, y_pred)

# # Forecast for the next 10 days
# future_days = 10
# last_data = X.iloc[-1].values.reshape(1, -1)
# future_predictions = []
# for _ in range(future_days):
#     pred_price = model.predict(last_data)
#     future_predictions.append(pred_price[0])
#     last_data = np.append(last_data[:, 1:], pred_price).reshape(1, -1)

# # Display accuracy metrics
# st.subheader("Model Accuracy")
# st.write(f"Mean Absolute Error: {mae:.2f}")
# st.write(f"Root Mean Squared Error: {rmse:.2f}")
# st.write(f"Model Accuracy: {r2:.2f}")
# st.progress(r2)  # Show R² score as progress

# # Plot the data
# st.subheader("Closing Price Prediction")
# plt.figure(figsize=(18, 7))
# plt.plot(data.index[-len(y_test):], y_test, label="True Closing Price (INR)")
# plt.plot(data.index[-len(y_test):], y_pred, label="Predicted Closing Price (INR)")
# plt.legend()
# st.pyplot(plt)

# # Forecast plot
# st.subheader("Forecasted Closing Prices for Next 10 Days")
# plt.figure(figsize=(18, 7))
# plt.plot(range(1, future_days + 1), future_predictions, marker='o', label="Forecasted Close Price (INR)")
# plt.xlabel("Days Ahead")
# plt.ylabel("Price (INR)")
# plt.legend()
# st.pyplot(plt)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Set page layout
st.set_page_config(layout="wide")

# Title and description
st.title("Stock Market Prediction")
st.write("This app fetches stock data and predicts the price using the XGBoost model.")

# User input for stock ticker
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA): ", "AAPL")

# Load stock data
data = yf.download(ticker, period="1y")

# Check if data was loaded
if data.empty:
    st.error("No data available for this ticker. Please try another.")
else:
    # Select relevant columns
    data = data[['Open', 'Close', 'High', 'Low']]

    # Display raw data
    st.subheader("Stock Data")
    st.write(data)

    # Convert currency (assuming a fixed exchange rate, e.g., 1 USD = 83 INR)
    exchange_rate = 83  # Example conversion rate (USD to INR)
    data[['Open', 'Close', 'High', 'Low']] = data[['Open', 'Close', 'High', 'Low']] * exchange_rate

    # Feature Engineering
    data['Target'] = data['Close'].shift(-1)  # Shift close price to create target variable
    data.dropna(inplace=True)  # Drop missing values

    # Check if there are enough rows for train-test split
    if len(data) < 2:
        st.error("Not enough data after processing for training. Please try a different ticker or time period.")
    else:
        # Split data into features and target
        X = data[['Open', 'High', 'Low', 'Close']]
        y = data['Target']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model training
        model = XGBRegressor()
        model.fit(X_train, y_train)

        # Model prediction
        y_pred = model.predict(X_test)

        # Calculate model accuracy
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # Forecast for the next 10 days
        future_days = 10
        last_data = X.iloc[-1].values.reshape(1, -1)
        future_predictions = []
        for _ in range(future_days):
            pred_price = model.predict(last_data)
            future_predictions.append(pred_price[0])
            last_data = np.append(last_data[:, 1:], pred_price).reshape(1, -1)

        # Display accuracy metrics
        st.subheader("Model Accuracy")
        st.write(f"Mean Absolute Error: {mae:.2f}")
        st.write(f"Root Mean Squared Error: {rmse:.2f}")
        st.write(f"Model Accuracy (R²): {r2:.2f}")
        st.progress(r2)  # Show R² score as progress

        # Plot the data
        st.subheader("Closing Price Prediction")
        plt.figure(figsize=(18, 7))
        plt.plot(data.index[-len(y_test):], y_test, label="True Closing Price (INR)")
        plt.plot(data.index[-len(y_test):], y_pred, label="Predicted Closing Price (INR)")
        plt.legend()
        st.pyplot(plt)

        # Forecast plot
        st.subheader("Forecasted Closing Prices for Next 10 Days")
        plt.figure(figsize=(18, 7))
        plt.plot(range(1, future_days + 1), future_predictions, marker='o', label="Forecasted Close Price (INR)")
        plt.xlabel("Days Ahead")
        plt.ylabel("Price (INR)")
        plt.legend()
        st.pyplot(plt)
