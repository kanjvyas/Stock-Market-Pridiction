import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime

# Fetch the stock data
def fetch_data(ticker):
    stock_data = yf.download(ticker, period="1y")
    stock_data = stock_data[['Open', 'High', 'Low', 'Close']]
    stock_data['Currency'] = "INR"
    return stock_data

# Prepare data for modeling
def prepare_data(df):
    df['Target'] = df['Close'].shift(-1)
    df.dropna(inplace=True)
    X = df[['Open', 'High', 'Low', 'Close']]
    y = df['Target']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
def train_model(X_train, y_train):
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5)
    model.fit(X_train, y_train)
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    accuracy = 100 - (rmse / np.mean(y_test)) * 100
    return accuracy, predictions

# Streamlit UI
st.title("Stock Market Prediction App")
st.write("Forecasted Closing Price for the next day with high accuracy.")

# User inputs
ticker = st.text_input("Enter the stock ticker (e.g., AAPL, GOOGL, MSFT):", "AAPL")
if ticker:
    df = fetch_data(ticker)
    
    if df.empty:
        st.write("No data found for the ticker symbol.")
    else:
        # Displaying basic info
        st.write("### Stock Data Overview")
        st.write(df.tail(5))

        # Model training and prediction
        X_train, X_test, y_train, y_test = prepare_data(df)
        model = train_model(X_train, y_train)
        accuracy, predictions = evaluate_model(model, X_test, y_test)

        # Display accuracy and forecasting result
        st.write(f"### Model Accuracy: {accuracy:.2f}%")
        next_day_pred = model.predict(df[['Open', 'High', 'Low', 'Close']].tail(1))
        st.write(f"**Forecasted Closing Price (INR)**: {next_day_pred[0]:.2f}")

        # Plot the predictions
        st.write("### Prediction vs Actual Closing Price")
        plt.figure(figsize=(10, 5))
        plt.plot(df.index[-len(y_test):], y_test, label="Actual Price", color="blue")
        plt.plot(df.index[-len(predictions):], predictions, label="Predicted Price", color="orange")
        plt.xlabel("Date")
        plt.ylabel("Price (INR)")
        plt.title(f"{ticker} Closing Price Prediction")
        plt.legend()
        st.pyplot(plt)
