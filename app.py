import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go

# page layout
st.set_page_config(layout="wide")

# Title and description
st.title("Stock Market Prediction")
st.write("This app fetches stock data and predicts the price using the XGBoost model.")

# User input 
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA): ", "TSLA")

# Load data
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

    # Convert currency 
    exchange_rate = 84
    data[['Open', 'Close', 'High', 'Low']] = data[['Open', 'Close', 'High', 'Low']] * exchange_rate

    # Feature Engineering
    data['Target'] = data['Close'].shift(-1)  
    data.dropna(inplace=True)  

    # Checking if there are enough rows for train-test split
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

        # model accuracy
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # Forecast for 10 days
        future_days = 10
        last_data = X.iloc[-1].values.reshape(1, -1)
        future_predictions = []
        for _ in range(future_days):
            pred_price = model.predict(last_data)
            future_predictions.append(pred_price[0])
            last_data = np.append(last_data[:, 1:], pred_price).reshape(1, -1)

        # Display accuracy 
        st.subheader("Model Accuracy")
        st.write(f"Mean Absolute Error: {mae:.2f}")
        st.write(f"Root Mean Squared Error: {rmse:.2f}")
        st.write(f"Model Accuracy (R²): {r2:.2f}")
        st.progress(r2)  

        # Plot the data 
        st.subheader("Closing Price Prediction")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=data.index[-len(y_test):], y=y_test, mode='lines', name='True Closing Price (INR)'))
        fig1.add_trace(go.Scatter(x=data.index[-len(y_test):], y=y_pred, mode='lines', name='Predicted Closing Price (INR)'))
        fig1.update_layout(
            title="Actual vs Predicted Closing Prices",
            xaxis_title="Date",
            yaxis_title="Price (INR)",
            legend_title="Legend"
        )
        st.plotly_chart(fig1)

        # Forecast plot 
        st.subheader("Forecasted Closing Prices for Next 10 Days")
        future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=future_days)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=future_dates, y=future_predictions, mode='lines+markers', name="Forecasted Close Price (INR)"))
        fig2.update_layout(
            title="Forecast for the Next 10 Days",
            xaxis_title="Date",
            yaxis_title="Price (INR)",
            legend_title="Legend"
        )
        st.plotly_chart(fig2)

        # Display forecasted price table
        st.subheader("Forecasted Prices")
        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Forecasted Price (INR)": future_predictions
        })
        st.write(forecast_df)
