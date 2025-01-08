# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import numpy as np
# from xgboost import XGBRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from sklearn.model_selection import train_test_split
# import plotly.graph_objects as go

# # Page Config
# st.set_page_config(layout="wide", page_title="Stock Market Prediction", page_icon="📈")

# st.image("https://i.postimg.cc/j2tdGvv5/new-removebg-preview.png", width=300)

# # Custom CSS to style the header with transparency
# custom_css = """
# <style>
#     header[data-testid="stHeader"] {
#         background-color: rgba(0, 0, 0, 0); 
#         padding: 10px;
#     }
#     header[data-testid="stHeader"] h1 {
#         color: white;
#     }
# </style>
# """
# st.markdown(custom_css, unsafe_allow_html=True)

# # Set the path to your background image
# image_url = "https://wallpapers.com/images/hd/stock-market-green-red-candle-sticks-u8sypn2sbm22a0y3.jpg"

# # Inject CSS to set the background image
# st.markdown(
#     f"""
#     <style>
#     .stApp {{
#         background-image: url('{image_url}');
#         background-size: cover;
#         background-position: center;
#         background-repeat: no-repeat;
#         height: 100vh;
#     }}
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # Define custom CSS for translucent sidebar
# sidebar_css = """
# <style>
# [data-testid="stSidebar"] {
#     background-color: rgba(0, 0, 0, 0.1); 
# }
# </style>
# """
# st.markdown(sidebar_css, unsafe_allow_html=True)

# # Custom CSS for Yahoo Finance-like Aesthetic
# st.markdown("""
#     <style>
#         body {
#             background-color: #121212; 
#             color: #ffffff;
#             font-family: 'Arial', sans-serif;
#         }
#         .sidebar .sidebar-content {
#             background-color: #1e1e1e;
#         }
#         h1, h2, h3, h4, h5 {
#             color: #00ff7f;
#         }
#         .stProgress > div > div > div > div {
#             background-color: #00ff7f;
#         }
#         .css-1aumxhk {
#             font-size: 2.5rem;
#             font-weight: bold;
#             color: #00ff7f;
#         }
#         .css-1629p8f {
#             font-size: 1.5rem;
#             color: #d3d3d3;
#         }
#         .footer {
#             position: fixed;
#             bottom: 0;
#             left: 0;
#             width: 100%;
#             background-color: #1e1e1e;
#             color: #ffffff;
#             text-align: center;
#             padding: 10px;
#         }
#     </style>
# """, unsafe_allow_html=True)

# # Sidebar Navigation
# st.sidebar.header("📊 Stock Market Prediction")

# # Dropdown menu for stock ticker selection
# stock_tickers = ["NVDA", "TSLA", "GOOGL", "AMZN", "MSFT", "HDB", "BOAT", "ZOMATO.BO", "BAJAJFINSV.NS", "^NSEI",
# "^BSESN", "INFY.NS", "ADANIGREEN.NS", "ADANIENT.NS", "RELIANCE.NS", "ITC.NS", "HINDUNILVR.BO", "TCS.NS", "TATASTEEL.NS", "TATAPOWER.BO",
# "ADANIPOWER.NS", "AXISBANK.NS", "HDFCLIFE.BO", "ADANIPORTS.BO", "BAJAJFINSV.NS", "BAJAJ-AUTO.BO", "BHARTIARTL.BO", "CIPLA.BO", "ICICIBANK.NS", "INDUSINDBK.BO",
# "JSWSTEEL.NS", "KOTAKBANK.NS", "MARUTI.BO", "ONGC.BO", "SPARC.NS", "TATAMOTORS.BO", "TECHM.BO", "WIPRO.NS", "BANKBARODA.BO", "IRCTC.BO",
# "IDFCFIRSTB.NS", "DABUR.BO", "COALINDIA.BO", "CANBK.NS", "JINDALSTEL.BO", "BANDHANBNK.BO", "ULTRACEMCO.BO", "DELHIVERY.NS", "KALYANKJIL.BO", "ZYDUSLIFE.BO"]
# selected_ticker = st.sidebar.selectbox("Select a Stock Ticker:", stock_tickers, index=0)

# # Text input for custom ticker
# custom_ticker = st.sidebar.text_input("Or enter a custom ticker :").strip()

# # Use custom ticker if provided
# ticker = custom_ticker if custom_ticker else selected_ticker
# st.sidebar.write("Tip: Use a ticker from Yahoo Finance!")

# # Fetch Stock Data
# data = yf.download(ticker, period="1y")

# if data.empty:
#     st.error("No data available for this ticker. Please try another.")
# else:
#     # Select relevant columns
#     data = data[['Open', 'Close', 'High', 'Low']]

#     # Title and Description
#     st.title(f"{ticker.upper()} Stock Price Prediction")
#     st.subheader("Historical Stock Data")
#     st.write("The table below shows historical data for the selected stock.")

#     # Display raw data
#     st.dataframe(data.style.set_properties(**{'background-color': '#1e1e1e', 'color': '#ffffff'}), height=400)

#     # Feature Engineering
#     data['Target'] = data['Close'].shift(-1)  
#     data.dropna(inplace=True)

#     if len(data) < 2:
#         st.error("Not enough data after processing for training. Please try a different ticker.")
#     else:
#         # Split data into features and target
#         X = data[['Open', 'High', 'Low', 'Close']]
#         y = data['Target']
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#         # Model Training
#         model = XGBRegressor()
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)

#         # Model Accuracy
#         mae = mean_absolute_error(y_test, y_pred)
#         rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#         r2 = r2_score(y_test, y_pred)

#         # Forecast for 10 days
#         future_days = 10
#         last_data = X.iloc[-1].values.reshape(1, -1)
#         future_predictions = []
#         current_date = pd.Timestamp.now()

#         # Adjust future predictions to start from the current date and exclude weekends
#         while len(future_predictions) < future_days:
#             # Skip weekends (Saturday = 5, Sunday = 6)
#             if current_date.weekday() < 5:
#                 pred_price = model.predict(last_data)
#                 future_predictions.append(pred_price[0])
#                 last_data = np.append(last_data[:, 1:], pred_price).reshape(1, -1)
#             current_date += pd.Timedelta(days=1)

#         # Model Performance Metrics
#         st.subheader("📈 Model Metrics")
#         col1, col2, col3 = st.columns(3)
#         col1.metric("Mean Absolute Error", f"{mae:.2f}")
#         col2.metric("Root Mean Squared Error", f"{rmse:.2f}")
#         col3.metric("R² Score", f"{r2:.2f}")
#         st.progress(r2)

#         # Historical and Predicted Data Visualization
#         st.subheader("📉 Actual vs Predicted Prices")
#         fig1 = go.Figure()
#         fig1.add_trace(go.Scatter(x=data.index[-len(y_test):], y=y_test, mode='lines', name='True Closing Price', line=dict(color='#00ff7f')))
#         fig1.add_trace(go.Scatter(x=data.index[-len(y_test):], y=y_pred, mode='lines', name='Predicted Closing Price', line=dict(color='#ff4500')))
#         fig1.update_layout(
#             template='plotly_white',
#             title="True vs Predicted Closing Prices",
#             xaxis_title="Date",
#             yaxis_title="Price",
#             legend_title="Legend"
#         )
#         st.plotly_chart(fig1, use_container_width=True)
        
#            # Forecast Visualization
#         st.subheader("🔮 Forecasted Prices (Next 10 Business Days)")
#         # Calculate next 10 business days from today
#         future_dates = pd.date_range(start=pd.Timestamp.today(), periods=future_days, freq='B')  # 'B' for business days
#         fig2 = go.Figure()
#         fig2.add_trace(go.Scatter(x=future_dates, y=future_predictions, mode='lines+markers', name="Forecasted Prices", line=dict(color='#00ff7f')))
#         fig2.update_layout(template='plotly_dark', title="Forecast for Next 10 Business Days", xaxis_title="Date", yaxis_title="Price", legend_title="Legend")
#         st.plotly_chart(fig2, use_container_width=True)

#         # Forecast Table
# st.subheader("📋 Forecasted Prices Table")
# forecast_df = pd.DataFrame({"Date": future_dates.date, "Forecasted Price": future_predictions})  # Use .date to remove time
# st.dataframe(
#     forecast_df.style.set_properties(**{'background-color': '#1e1e1e', 'color': '#ffffff', 'border': '1px solid #444'})
# )



#         # # Forecast Visualization
#         # st.subheader("🔮 Forecasted Prices (Next 10 Days)")
#         # future_dates = pd.date_range(start=pd.Timestamp.now(), periods=future_days * 2).to_series()
#         # # Exclude weekends
#         # future_dates = future_dates[future_dates.dt.weekday < 5][:future_days]
#         # fig2 = go.Figure()
#         # fig2.add_trace(go.Scatter(x=future_dates, y=future_predictions, mode='lines+markers', name="Forecasted Prices", line=dict(color='#00ff7f')))
#         # fig2.update_layout(
#         #     template='plotly_dark',
#         #     title="Forecast for Next 10 Days",
#         #     xaxis_title="Date",
#         #     yaxis_title="Price",
#         #     legend_title="Legend"
#         # )
#         # st.plotly_chart(fig2, use_container_width=True)
        
# # # Create a DataFrame for future predictions
# # future_data = pd.DataFrame({
# #     "Date": future_dates,
# #     "Day": future_dates.dt.day_name(),
# #     "Forecasted Price": future_predictions
# # })

# # # Format the price column for better readability
# # future_data["Forecasted Price"] = future_data["Forecasted Price"].apply(lambda x: f"${x:.2f}")

# # # Display the forecast table
# # st.subheader("📅 Forecasted Prices Table")
# # st.write("The table below shows the predicted stock prices for the next 10 trading days, along with the corresponding dates and days of the week.")
# # st.table(future_data)


# # # Custom CSS to reduce table size
# # table_css = """
# # <style>
# #     .stTable {
# #         font-size: 0.8rem;
# #         width: 50%;
# #         margin
# #     }
# # </style>
# # """
# # st.markdown(table_css, unsafe_allow_html=True)


#         # # Display forecasted price table
#         # st.subheader("Forecasted Prices")
#         # forecast_df = pd.DataFrame({
#         #     "Date": future_dates,
#         #     "Forecasted Price ": future_predictions
#         # })
#         # st.write(forecast_df)

# # Footer
# st.markdown("---")  # Add a horizontal line for separation
# col1, col2, col3 = st.columns([3, 6, 1])  # Adjust column widths to center the footer
# with col2:
#     st.write("© 2024 Stock Market Prediction App. All rights reserved.", align="center")



import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
import plotly.graph_objects as go

# Page Config
st.set_page_config(layout="wide", page_title="Stock Market Prediction", page_icon="📈")

st.image("https://i.postimg.cc/j2tdGvv5/new-removebg-preview.png", width=300)

# Custom CSS to style the header with transparency
custom_css = """
<style>
    header[data-testid="stHeader"] {
        background-color: rgba(0, 0, 0, 0); 
        padding: 10px;
    }
    header[data-testid="stHeader"] h1 {
        color: white;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Set the path to your background image
image_url = "https://wallpapers.com/images/hd/stock-market-green-red-candle-sticks-u8sypn2sbm22a0y3.jpg"

# Inject CSS to set the background image
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url('{image_url}');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        height: 100vh;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Define custom CSS for translucent sidebar
sidebar_css = """
<style>
[data-testid="stSidebar"] {
    background-color: rgba(0, 0, 0, 0.1); 
}
</style>
"""
st.markdown(sidebar_css, unsafe_allow_html=True)

# Custom CSS for Yahoo Finance-like Aesthetic
st.markdown("""
    <style>
        body {
            background-color: #121212; 
            color: #ffffff;
            font-family: 'Arial', sans-serif;
        }
        .sidebar .sidebar-content {
            background-color: #1e1e1e;
        }
        h1, h2, h3, h4, h5 {
            color: #00ff7f;
        }
        .stProgress > div > div > div > div {
            background-color: #00ff7f;
        }
        .css-1aumxhk {
            font-size: 2.5rem;
            font-weight: bold;
            color: #00ff7f;
        }
        .css-1629p8f {
            font-size: 1.5rem;
            color: #d3d3d3;
        }
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #1e1e1e;
            color: #ffffff;
            text-align: center;
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.header("📊 Stock Market Prediction")

# Dropdown menu for stock ticker selection
stock_tickers = ["BOAT", "TSLA", "GOOGL", "AMZN", "MSFT", "HDB", "NVDA", "ZOMATO.BO", "BAJAJFINSV.NS", "^NSEI",
"^BSESN", "INFY.NS", "ADANIGREEN.NS", "ADANIENT.NS", "RELIANCE.NS", "ITC.NS", "HINDUNILVR.BO", "TCS.NS", "TATASTEEL.NS", "TATAPOWER.BO",
"ADANIPOWER.NS", "AXISBANK.NS", "HDFCLIFE.BO", "ADANIPORTS.BO", "BAJAJFINSV.NS", "BAJAJ-AUTO.BO", "BHARTIARTL.BO", "CIPLA.BO", "ICICIBANK.NS", "INDUSINDBK.BO",
"JSWSTEEL.NS", "KOTAKBANK.NS", "MARUTI.BO", "ONGC.BO", "SPARC.NS", "TATAMOTORS.BO", "TECHM.BO", "WIPRO.NS", "BANKBARODA.BO", "IRCTC.BO",
"IDFCFIRSTB.NS", "DABUR.BO", "COALINDIA.BO", "CANBK.NS", "JINDALSTEL.BO", "BANDHANBNK.BO", "ULTRACEMCO.BO", "DELHIVERY.NS", "KALYANKJIL.BO", "ZYDUSLIFE.BO"]
selected_ticker = st.sidebar.selectbox("Select a Stock Ticker:", stock_tickers, index=0)

# Text input for custom ticker
custom_ticker = st.sidebar.text_input("Or enter a custom ticker :").strip()

# Use custom ticker if provided
# sourcery skip: or-if-exp-identity
ticker = custom_ticker if custom_ticker else selected_ticker
st.sidebar.write("Tip: Use a ticker from Yahoo Finance!")

# Fetch Stock Data
data = yf.download(ticker, period="1y")

if data.empty:
    st.error("No data available for this ticker. Please try another.")
else:
    # Select relevant columns
    data = data[['Open', 'Close', 'High', 'Low']]

    # Title and Description
    st.title(f"{ticker.upper()} Stock Price Prediction")
    st.subheader("Historical Stock Data")
    st.write("The table below shows historical data for the selected stock.")

    # Display raw data
    st.dataframe(data.style.set_properties(**{'background-color': '#1e1e1e', 'color': '#ffffff'}), height=400)

    # Feature Engineering
    data['Target'] = data['Close'].shift(-1)  
    data['SMA_5'] = data['Close'].rolling(window=5).mean()  # 5-day moving average
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))  # Relative Strength Index (RSI)
    data['Lag_1'] = data['Close'].shift(1)  # Lag feature
    data.dropna(inplace=True)

    if len(data) < 2:
        st.error("Not enough data after processing for training. Please try a different ticker.")
    else:
        # Split data into features and target
        X = data[['Open', 'High', 'Low', 'Close', 'SMA_5', 'RSI', 'Lag_1']]
        y = data['Target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Hyperparameter Tuning with GridSearchCV
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0]
        }
        grid_search = GridSearchCV(XGBRegressor(), param_grid, cv=3, scoring='neg_mean_absolute_error')
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        # Model Training
        y_pred = best_model.predict(X_test)

        # Model Accuracy
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # Forecast for 10 days
        future_days = 10
        last_data = X.iloc[-1].values.reshape(1, -1)
        future_predictions = []
        current_date = pd.Timestamp.now()

        # Adjust future predictions to start from the current date and exclude weekends
        while len(future_predictions) < future_days:
            # Skip weekends (Saturday = 5, Sunday = 6)
            if current_date.weekday() < 5:
                pred_price = best_model.predict(last_data)
                future_predictions.append(pred_price[0])
                last_data = np.append(last_data[:, 1:], pred_price).reshape(1, -1)
            current_date += pd.Timedelta(days=1)

        # Model Performance Metrics
        st.subheader("📈 Model Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Mean Absolute Error", f"{mae:.2f}")
        col2.metric("Root Mean Squared Error", f"{rmse:.2f}")
        col3.metric("R² Score", f"{r2:.2f}")
        st.progress(r2)

        # Historical and Predicted Data Visualization
        st.subheader("📉 Actual vs Predicted Prices")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=data.index[-len(y_test):], y=y_test, mode='lines', name='True Closing Price', line=dict(color='#00ff7f')))
        fig1.add_trace(go.Scatter(x=data.index[-len(y_test):], y=y_pred, mode='lines', name='Predicted Closing Price', line=dict(color='#ff4500')))
        fig1.update_layout(
            template='plotly_white',
            title="True vs Predicted Closing Prices",
            xaxis_title="Date",
            yaxis_title="Price",
            legend_title="Legend"
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Forecast Visualization
        st.subheader("🔮 Forecasted Prices (Next 10 Business Days)")
        # Calculate next 10 business days from today
        future_dates = pd.date_range(start=pd.Timestamp.today(), periods=future_days, freq='B')  # 'B' for business days
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=future_dates, y=future_predictions, mode='lines+markers', name="Forecasted Prices", line=dict(color='#00ff7f')))
        fig2.update_layout(template='plotly_dark', title="Forecast for Next 10 Business Days", xaxis_title="Date", yaxis_title="Price", legend_title="Legend")
        st.plotly_chart(fig2, use_container_width=True)

        # Forecast Table
        st.subheader("📋 Forecasted Prices Table")
        forecast_df = pd.DataFrame({"Date": future_dates.date, "Forecasted Price": future_predictions})  # Use .date to remove time
        st.dataframe(
            forecast_df.style.set_properties(**{'background-color': '#1e1e1e', 'color': '#ffffff', 'border': '1px solid #444'})
        )

# Footer
st.markdown("---")  # Add a horizontal line for separation
col1, col2, col3 = st.columns([3, 6, 1])  # Adjust column widths to center the footer
with col2:
    st.write("© 2024 Stock Market Prediction App. All rights reserved.", align="center")
