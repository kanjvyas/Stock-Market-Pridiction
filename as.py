import streamlit as st
from datetime import date, timedelta
import yfinance as yf
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import plotly.graph_objs as go

# Streamlit title and description
st.title("Stock Market Prediction App")
st.write("Predict the closing price for tomorrow and the day after tomorrow using historical data and XGBoost.")

# User input for ticker with validation
ticker = st.text_input("Enter the stock ticker symbol (e.g., AAPL, MSFT, TSLA):", "AAPL").upper()

if not ticker.isalpha() or len(ticker) < 1 or len(ticker) > 5:
    st.warning("Please enter a valid stock ticker symbol.")
else:
    # Function to load and process data
    def load_data(ticker):
        start_date = '2010-01-01'  # Start date for historical data
        end_date = date.today() - timedelta(days=1)  # End date for historical data (yesterday's date)
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            if data.empty:
                st.error(f"No data found for ticker: {ticker}")
                return None
            data = data.dropna()  # Drop rows with NaN values
            return data
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return None

    # RSI Calculation function
    def calculate_rsi(series, window=14):
        delta = series.diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    # Load and display raw data
    data = load_data(ticker)
    if data is not None:
        # Convert prices to INR (assumed conversion rate for demonstration)
        inr_conversion_rate = 82.0  # Update with actual conversion rate
        data['Close_INR'] = data['Close'] * inr_conversion_rate  # Closing prices in INR
        data['Open_INR'] = data['Open'] * inr_conversion_rate  # Opening prices in INR
        data['High_INR'] = data['High'] * inr_conversion_rate  # High prices in INR
        data['Low_INR'] = data['Low'] * inr_conversion_rate  # Low prices in INR
        
        # Label the raw data
        raw_data_labeled = data[['Close_INR', 'Open_INR', 'High_INR', 'Low_INR']].rename(
            columns={
                'Close_INR': 'Closing Price (INR)',
                'Open_INR': 'Opening Price (INR)',
                'High_INR': 'High Price (INR)',
                'Low_INR': 'Low Price (INR)'
            }
        )

        st.subheader("Raw Stock Data (Prices in INR)")
        st.dataframe(raw_data_labeled)

        # Prepare features and target
        data['Lagged_Close'] = data['Close'].shift(1)  # Previous day's closing price
        data['MA5'] = data['Close'].rolling(window=5).mean()  # 5-day moving average
        data['MA10'] = data['Close'].rolling(window=10).mean()  # 10-day moving average
        data['MA20'] = data['Close'].rolling(window=20).mean()  # 20-day moving average
        data['RSI'] = calculate_rsi(data['Close'], window=14)  # Add RSI indicator
        data = data.dropna()  # Drop rows with NaN values

        # Prepare features and target
        X = data[['Lagged_Close', 'Open', 'High', 'Low', 'Volume', 'MA5', 'MA10', 'MA20', 'RSI']]
        y = data['Close']

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Set up XGBoost with GridSearchCV for hyperparameter tuning
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse')

        # Define parameter grid for tuning
        param_grid = {
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 200],
            'subsample': [0.7, 0.8],
            'colsample_bytree': [0.7, 0.8]
        }

        # Perform GridSearchCV to find the best parameters
        grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='neg_mean_absolute_error', verbose=1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        # Predict and calculate accuracy metrics
        predictions = best_model.predict(X_test)
        mape = mean_absolute_percentage_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        st.write(f"Model Mean Absolute Percentage Error (MAPE): {mape:.2f}")
        st.write(f"Model Accuracy (R-squared): {r2 * 100:.2f}%")

        # Check if model accuracy meets the 70% threshold
        if r2 >= 0.7:
            st.success(f"Model accuracy is : {r2 * 100:.2f}%")
        else:
            st.warning(f"Model accuracy is : {r2 * 100:.2f}%")

        # Prepare data for forecasting next two days
        latest_data = data.tail(1).copy()  # Get the last row of data for prediction
        future_dates = [latest_data.index[-1] + timedelta(days=1), latest_data.index[-1] + timedelta(days=2)]
        forecasted_prices = []

        # Forecasting for each future date
        for i in range(2):
            # Update features for the next day’s prediction
            latest_data['Lagged_Close'] = forecasted_prices[-1] if forecasted_prices else latest_data['Close'].values[0]
            combined_close = pd.concat([data['Close'], pd.Series(forecasted_prices)], ignore_index=True)
            latest_data['MA5'] = combined_close.rolling(window=5).mean().iloc[-1]
            latest_data['MA10'] = combined_close.rolling(window=10).mean().iloc[-1]
            latest_data['MA20'] = combined_close.rolling(window=20).mean().iloc[-1]
            latest_data['RSI'] = calculate_rsi(combined_close, window=14).iloc[-1]

            # Predict and store the forecasted price
            forecasted_price = best_model.predict(latest_data[['Lagged_Close', 'Open', 'High', 'Low', 'Volume', 'MA5', 'MA10', 'MA20', 'RSI']])[0]
            forecasted_prices.append(forecasted_price)

        # Convert forecasted prices to INR
        forecasted_prices_inr = [price * inr_conversion_rate for price in forecasted_prices]

        # Display future predictions in INR
        st.subheader("Forecasted Closing Prices (in INR)")
        st.write(f"Predicted closing price for {future_dates[0].date()}: ₹{forecasted_prices_inr[0]:.2f}")
        st.write(f"Predicted closing price for {future_dates[1].date()}: ₹{forecasted_prices_inr[1]:.2f}")

        # Plot actual prices and forecasted prices using Plotly
        fig = go.Figure()

        # Add actual price line
        fig.add_trace(go.Scatter(x=data.index, y=data['Close_INR'], mode='lines', name='Actual Prices (INR)', line=dict(color='blue'), marker=dict(size=5)))
        
        # Add forecasted price line
        fig.add_trace(go.Scatter(x=future_dates, y=forecasted_prices_inr, mode='lines+markers', name='Forecasted Prices (INR)', line=dict(color='green', width=2), marker=dict(size=8)))

        fig.update_layout(title='Stock Price Prediction',
                          xaxis_title='Date',
                          yaxis_title='Price (INR)',
                          legend_title='Legend',
                          xaxis_tickangle=-45,
                          template='plotly_white')

        st.plotly_chart(fig)


