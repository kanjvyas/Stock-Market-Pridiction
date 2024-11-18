# import yfinance as yf
# import pandas as pd
# import numpy as np
# import streamlit as st
# import xgboost as xgb
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# from datetime import datetime

# # Fetch the stock data
# def fetch_data(ticker):
#     stock_data = yf.download(ticker, period="1y")
#     stock_data = stock_data[['Open', 'High', 'Low', 'Close']]
#     stock_data['Currency'] = "INR"
#     return stock_data

# # Prepare data for modeling
# def prepare_data(df):
#     df['Target'] = df['Close'].shift(-1)
#     df.dropna(inplace=True)
#     X = df[['Open', 'High', 'Low', 'Close']]
#     y = df['Target']
#     return train_test_split(X, y, test_size=0.2, random_state=42)

# # Train the model
# def train_model(X_train, y_train):
#     model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5)
#     model.fit(X_train, y_train)
#     return model

# # Evaluate the model
# def evaluate_model(model, X_test, y_test):
#     predictions = model.predict(X_test)
#     mae = mean_absolute_error(y_test, predictions)
#     rmse = np.sqrt(mean_squared_error(y_test, predictions))
#     accuracy = 100 - (rmse / np.mean(y_test)) * 100
#     return accuracy, predictions

# # Streamlit UI
# st.title("Stock Market Prediction App")
# st.write("Forecasted Closing Price for the next day with high accuracy.")

# # User inputs
# ticker = st.text_input("Enter the stock ticker (e.g., AAPL, GOOGL, MSFT):", "AAPL")
# if ticker:
#     df = fetch_data(ticker)
    
#     if df.empty:
#         st.write("No data found for the ticker symbol.")
#     else:
#         # Displaying basic info
#         st.write("### Stock Data Overview")
#         st.write(df.tail(5))

#         # Model training and prediction
#         X_train, X_test, y_train, y_test = prepare_data(df)
#         model = train_model(X_train, y_train)
#         accuracy, predictions = evaluate_model(model, X_test, y_test)

#         # Display accuracy and forecasting result
#         st.write(f"### Model Accuracy: {accuracy:.2f}%")
#         next_day_pred = model.predict(df[['Open', 'High', 'Low', 'Close']].tail(1))
#         st.write(f"**Forecasted Closing Price (INR)**: {next_day_pred[0]:.2f}")

#         # Plot the predictions
#         st.write("### Prediction vs Actual Closing Price")
#         plt.figure(figsize=(10, 5))
#         plt.plot(df.index[-len(y_test):], y_test, label="Actual Price", color="blue")
#         plt.plot(df.index[-len(predictions):], predictions, label="Predicted Price", color="orange")
#         plt.xlabel("Date")
#         plt.ylabel("Price (INR)")
#         plt.title(f"{ticker} Closing Price Prediction")
#         plt.legend()
#         st.pyplot(plt)





import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from datetime import datetime, timedelta

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
        future_dates = [data.index[-1] + timedelta(days=i) for i in range(1, future_days + 1)]
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

        # Plot the actual vs predicted closing prices
        st.subheader("Closing Price Prediction")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=data.index[-len(y_test):], y=y_test, mode='lines', name='True Closing Price (INR)'))
        fig1.add_trace(go.Scatter(x=data.index[-len(y_test):], y=y_pred, mode='lines', name='Predicted Closing Price (INR)'))
        fig1.update_layout(title="Actual vs Predicted Closing Prices", xaxis_title="Date", yaxis_title="Price (INR)")
        st.plotly_chart(fig1)

        # Forecast plot for the next 10 days
        st.subheader("Forecasted Closing Prices for Next 10 Days")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=future_dates, y=future_predictions, mode='lines+markers', name='Forecasted Price (INR)'))
        fig2.update_layout(title="Forecast for the Next 10 Days", xaxis_title="Date", yaxis_title="Price (INR)")
        st.plotly_chart(fig2)

        # Display forecasted price table
        st.subheader("10-Day Forecasted Prices")
        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Forecasted Price (INR)": future_predictions
        })
        st.write(forecast_df)
