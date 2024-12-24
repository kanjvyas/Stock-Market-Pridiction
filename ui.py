# from matplotlib.axis import Ticker
# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import numpy as np
# from xgboost import XGBRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from sklearn.model_selection import train_test_split
# import plotly.graph_objects as go

# # Set page layout
# st.set_page_config(page_title="Stock Market Prediction", layout="wide", page_icon="📈")

# st.image("https://i.postimg.cc/j2tdGvv5/new-removebg-preview.png" , width=300) 

# # Custom CSS to style the header with transparency
# custom_css = """
# <style>
#     /* Target the main header */
#     header[data-testid="stHeader"] {
#         background-color: rgba(0, 0, 0, 0); /* Black with 30% opacity */
#         padding: 10px;
#     }

#     /* Optional: Target the title and text within the header */
#     header[data-testid="stHeader"] h1 {
#         color: white; /* Set the text color */
#     }
# </style>
# """

# # Inject the custom CSS into the Streamlit app
# st.markdown(custom_css, unsafe_allow_html=True)

# # Set the path to your image
# # If the image is hosted online, use the URL
# # If it's in your project directory, use the relative path
# image_url = "https://wallpapercg.com/download/candlestick-pattern-7680x4320-19473.jpg"  # Replace with your image URL

# # Inject CSS to set the background image
# st.markdown(
#     f"""
#     <style>
#     .stApp {{
#         background-image: url('{image_url}');
#         background-size: cover;
#         background-position: center;
#         background-repeat: no-repeat;
#         height: 100vh;  /* Full height */
#     }}
#     </style>
#     """,
#     unsafe_allow_html=True
# )



# # Custom CSS for Yahoo Finance-like Aesthetic
# st.markdown("""
#     <style>
#         body {
#             background-color: #121212; 
#             color: #ffffff;
#             font-family: 'Arial', sans-serif;
           


# st.markdown(page_bg_img, unsafe_allow_html=True)
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
#         .css-1aumxhk {  /* Title font customization */
#             font-size: 2.5rem;
#             font-weight: bold;
#             color: #00ff7f;
#         }
#         .css-1629p8f {  /* Subheader font customization */
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

# page_bg_img = '''
# <style>
# body {
# background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
# background-size: cover;

# }
# </style>
# '''

# st.markdown(page_bg_img, unsafe_allow_html=True)



# # Define custom CSS for translucent sidebar
# sidebar_css = """
# <style>
# [data-testid="stSidebar"] {
#     background-color: rgba(0, 0, 0, 0.1);  /* White background with 80% opacity */
# }
# </style>
# """

# # Inject the CSS into the Streamlit app
# st.markdown(sidebar_css, unsafe_allow_html=True)

# # Sidebar inputs
# st.sidebar.title("⚙️ Settings")
# ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., HDB, BOAT):", "HDB")
# forecast_days = st.sidebar.slider("Forecast Days", min_value=1, max_value=30, value=10)

# # Title and description
# st.title(f"{ticker.upper()} Stock Price Prediction")
# st.header("Historical Stock Data")
# st.write("The table below shows historical data for the selected stock.")

# # Fetch stock data
# with st.spinner("Fetching stock data..."):
#     data = yf.download(ticker, period="1y")

# if data.empty:
#     st.error("No data available for this ticker. Please check the symbol or try another.")
# else:
#     # Display raw data
#     st.write(data)

#     # Feature Engineering
#     data = data[['Open', 'Close', 'High', 'Low']]
#     data['Target'] = data['Close'].shift(-1)
#     data.dropna(inplace=True)

#     if len(data) < 2:
#         st.error("Not enough data after processing for training. Please try a different ticker or time period.")
#     else:
#         # Split data into features and target
#         X = data[['Open', 'High', 'Low', 'Close']]
#         y = data['Target']

#         # Train-test split
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#         # Train model
#         model = XGBRegressor()
#         model.fit(X_train, y_train)

#         # Predict and evaluate
#         y_pred = model.predict(X_test)
#         mae = mean_absolute_error(y_test, y_pred)
#         rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#         r2 = r2_score(y_test, y_pred)
        
#         # Metrics Section
#         st.markdown("<h2 id='prediction'> 📈 Model Metrics</h2>", unsafe_allow_html=True)
#         col1, col2, col3 = st.columns(3)
#         col1.metric("MAE", f"{mae:.2f}")
#         col2.metric("RMSE", f"{rmse:.2f}")
#         col3.metric("R² Score", f"{r2:.2f}")
#         st.progress(r2)

#        # Historical and Predicted Data Visualization
#         st.subheader("📉 Actual vs Predicted Prices")
#         fig1 = go.Figure()
#         fig1.add_trace(go.Scatter(x=data.index[-len(y_test):], y=y_test, mode='lines', name='True Closing Price', line=dict(color='#00ff7f')))
#         fig1.add_trace(go.Scatter(x=data.index[-len(y_test):], y=y_pred, mode='lines', name='Predicted Closing Price', line=dict(color='#ff4500')))
#         fig1.update_layout(
#             template='plotly_dark',
#             title="True vs Predicted Closing Prices",
#             xaxis_title="Date",
#             yaxis_title="Price",
#             legend_title="Legend"
#         )
#         st.plotly_chart(fig1, use_container_width=True)

#         # Forecast future prices
#         last_data = X.iloc[-1].values.reshape(1, -1)
#         future_predictions = []
#         for _ in range(forecast_days):
#             pred_price = model.predict(last_data)
#             future_predictions.append(pred_price[0])
#             last_data = np.append(last_data[:, 1:], pred_price).reshape(1, -1)

#         # Plot future forecast
#         st.markdown("<h2 id='forecast'> 🔮 10-Day Future Forecast</h2>", unsafe_allow_html=True)
#         future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
#         forecast_df = pd.DataFrame({"Date": future_dates, "Forecasted Price": future_predictions})

#         fig_forecast = go.Figure()
#         fig_forecast.add_trace(go.Scatter(
#             x=forecast_df['Date'], y=forecast_df['Forecasted Price'],
#             mode='lines+markers', name='Forecasted Prices',
#             line=dict(color='#00ff7f', width=3), marker=dict(size=8)))
#         fig_forecast.update_layout(
#             title="10-Day Future Forecasted Prices",
#             xaxis_title="Date",
#             yaxis_title="Price",
#             template="plotly_white",
#             margin=dict(l=40, r=40, t=40, b=40),
#             hovermode="x unified"
#         )
#         st.plotly_chart(fig_forecast)

#         # Display forecasted data
#         st.subheader("📋 Forecasted Prices Table")
#         st.write(forecast_df)

#         # Add download button
#         csv = forecast_df.to_csv(index=False)
#         st.download_button("Download Forecast Data", data=csv, file_name="forecast.csv", mime="text/csv")

# # Footer
# st.markdown("© 2024 Stock Market Prediction App. All rights reserved.", unsafe_allow_html=True)

from matplotlib.axis import Ticker
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from scipy.stats import pearsonr

class AdvancedStockPredictor:
    def __init__(self, ticker, forecast_days=10):
        self.ticker = ticker
        self.forecast_days = forecast_days
        self.data = None
        self.model = None
        self.scaler = MinMaxScaler()

    def fetch_data(self):
        """Fetch and preprocess stock data"""
        try:
            # Fetch more historical data
            self.data = yf.download(self.ticker, period="5y")
            
            # Additional technical indicators
            self.data['MA50'] = self.data['Close'].rolling(window=50).mean()
            self.data['MA200'] = self.data['Close'].rolling(window=200).mean()
            self.data['RSI'] = self._calculate_rsi()
            self.data['MACD'], self.data['Signal'] = self._calculate_macd()
            
            # Drop NaN values
            self.data.dropna(inplace=True)
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return False
        return True

    def _calculate_rsi(self, periods=14):
        """Calculate Relative Strength Index"""
        delta = self.data['Close'].diff()
        
        # Make two series: one for lower closes and one for higher closes
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        
        # Calculate the EWMA
        roll_up = up.ewm(com=periods-1, adjust=False).mean()
        roll_down = down.ewm(com=periods-1, adjust=False).mean()
        
        # Calculate the RSI based on EWMA
        rs = roll_up / roll_down
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        return rsi

    def _calculate_macd(self, fast=12, slow=26, signal=9):
        """Calculate Moving Average Convergence Divergence"""
        exp1 = self.data['Close'].ewm(span=fast, adjust=False).mean()
        exp2 = self.data['Close'].ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

    def prepare_features(self):
        """Prepare features for model training"""
        # Select relevant features
        features_columns = [
            'Open', 'High', 'Low', 'Close', 
            'MA50', 'MA200', 'RSI', 'MACD', 'Signal'
        ]
        
        # Prepare features and target
        X = self.data[features_columns]
        y = self.data['Close'].shift(-self.forecast_days)
        
        # Remove NaN values
        combined = pd.concat([X, y], axis=1).dropna()
        X = combined[features_columns]
        y = combined['Close']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y

    def train_model(self):
        """Train advanced XGBoost model with cross-validation"""
        # Prepare data
        X, y = self.prepare_features()
        
        # Time Series Cross-Validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Advanced XGBoost model with more parameters
        model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.01,
            max_depth=10,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42
        )
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            model, X, y, 
            cv=tscv, 
            scoring='neg_mean_squared_error'
        )
        
        # Fit the model on entire data
        model.fit(X, y)
        self.model = model
        
        return -cv_scores.mean(), np.std(cv_scores)

    def forecast_prices(self):
        """Advanced forecasting with ensemble approach"""
        # Prepare last known data
        X, _ = self.prepare_features()
        last_features = X[-1].reshape(1, -1)
        
        # Multiple prediction approaches
        predictions = []
        for _ in range(self.forecast_days):
            # Predict using the model
            pred = self.model.predict(last_features)[0]
            predictions.append(pred)
            
            # Update features with prediction and some noise
            noise = np.random.normal(0, 0.02 * pred)
            last_features = np.roll(last_features, -1)
            last_features[0, -1] = pred + noise
        
        # Inverse transform to get actual prices
        return predictions

    def evaluate_model(self, X_test, y_test):
        """Comprehensive model evaluation"""
        y_pred = self.model.predict(X_test)
        
        # Multiple metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        correlation, _ = pearsonr(y_test, y_pred)
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'Correlation': correlation
        }

# Streamlit App Integration
def run_stock_prediction(ticker, forecast_days):
    # Create predictor
    predictor = AdvancedStockPredictor(ticker, forecast_days)
    
    # Fetch and prepare data
    if not predictor.fetch_data():
        st.error("Failed to fetch data")
        return
    
    # Train model
    with st.spinner("Training advanced prediction model..."):
        mse, std = predictor.train_model()
        st.success(f"Model trained successfully. MSE: {mse:.4f} ± {std:.4f}")
    
    # Forecast
    with st.spinner("Generating price forecast..."):
        forecast = predictor.forecast_prices()
        
        # Create forecast dataframe
        future_dates = pd.date_range(
            start=predictor.data.index[-1] + pd.Timedelta(days=1), 
            periods=len(forecast)
        )
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Forecasted Price': forecast
        })
    
    # Visualization
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast_df['Date'], 
        y=forecast_df['Forecasted Price'], 
        mode='lines+markers',
        name='Forecasted Prices',
        line=dict(color='#00ff7f', width=3)
    ))
    fig.update_layout(
        title=f"{ticker} Price Forecast",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark"
    )
    
    # Display results
    st.plotly_chart(fig)
    st.dataframe(forecast_df)

# Main Streamlit App
def main():
    st.title("Advanced Stock Price Predictor")
    

# Set page layout
st.set_page_config(page_title="Stock Market Prediction", layout="wide", page_icon="📈")

st.image("https://i.postimg.cc/j2tdGvv5/new-removebg-preview.png" , width=300) 

# Custom CSS to style the header with transparency
custom_css = """
<style>
    /* Target the main header */
    header[data-testid="stHeader"] {
        background-color: rgba(0, 0, 0, 0); /* Black with 30% opacity */
        padding: 10px;
    }

    /* Optional: Target the title and text within the header */
    header[data-testid="stHeader"] h1 {
        color: white; /* Set the text color */
    }
</style>
"""

# Inject the custom CSS into the Streamlit app
st.markdown(custom_css, unsafe_allow_html=True)

# Set the path to your image
# If the image is hosted online, use the URL
# If it's in your project directory, use the relative path
image_url = "https://wallpapercg.com/download/candlestick-pattern-7680x4320-19473.jpg"  # Replace with your image URL

# Inject CSS to set the background image
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url('{image_url}');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        height: 100vh;  /* Full height */
    }}
    </style>
    """,
    unsafe_allow_html=True
)

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
        .css-1aumxhk {  /* Title font customization */
            font-size: 2.5rem;
            font-weight: bold;
            color: #00ff7f;
        }
        .css-1629p8f {  /* Subheader font customization */
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

page_bg_img = '''
<style>
body {
background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

# Define custom CSS for translucent sidebar
sidebar_css = """
<style>
[data-testid="stSidebar"] {
    background-color: rgba(0, 0, 0, 0.1);  /* White background with 80% opacity */
}
</style>
"""

# Inject the CSS into the Streamlit app
st.markdown(sidebar_css, unsafe_allow_html=True)

# Sidebar inputs
st.sidebar.title("⚙️ Settings")

# Define a list of stock tickers to choose from (you can add more tickers as needed)
ticker_list = ["AAPL", "TSLA", "GOOGL", "AMZN", "MSFT", "HDB", "BOAT", "ZOMATO.BO", "BAJAJFINSV.NS", "^NSEI",
"^BSESN", "INFY.NS", "ADANIGREEN.NS", "ADANIENT.NS", "RELIANCE.NS", "ITC.NS", "HINDUNILVR.BO", "TCS.NS", "TATASTEEL.NS", "TATAPOWER.BO",
"ADANIPOWER.NS", "AXISBANK.NS", "ASIANPAINT.BO", "ADANIPORTS.BO", "BAJAJFINSV.NS", "BAJAJ-AUTO.BO", "BHARTIARTL.BO", "CIPLA.BO", "ICICIBANK.NS", "INDUSINDBK.BO",
"JSWSTEEL.NS", "KOTAKBANK.NS", "MARUTI.BO", "ONGC.BO", "SPARC.NS", "TATAMOTORS.BO", "TECHM.BO", "WIPRO.NS", "BANKBARODA.BO", "IRCTC.BO" ,
"IDFCFIRSTB.NS", "DABUR.BO", "COALINDIA.BO", "CANBK.NS", "JINDALSTEL.BO", "BANDHANBNK.BO", "ULTRACEMCO.BO", "DELHIVERY.NS", "KALYANKJIL.BO", "ZYDUSLIFE.BO"]

# Create a dropdown menu (selectbox) for the ticker
ticker = st.sidebar.selectbox("Select Stock Ticker:", ticker_list)

# Forecast days slider
forecast_days = st.sidebar.slider("Forecast Days", min_value=1, max_value=30, value=10)

# Title and description
st.title(f"{ticker.upper()} Stock Price Prediction")
st.header("Historical Stock Data")
st.write("The table below shows historical data for the selected stock.")

# Fetch stock data
with st.spinner("Fetching stock data..."):
    data = yf.download(ticker, period="1y")

if data.empty:
    st.error("No data available for this ticker. Please check the symbol or try another.")
else:
    # Display raw data
    st.write(data)

    # Feature Engineering
    data = data[['Open', 'Close', 'High', 'Low']]
    data['Target'] = data['Close'].shift(-1)
    data.dropna(inplace=True)

    if len(data) < 2:
        st.error("Not enough data after processing for training. Please try a different ticker or time period.")
    else:
        # Split data into features and target
        X = data[['Open', 'High', 'Low', 'Close']]
        y = data['Target']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = XGBRegressor()
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Metrics Section
        st.markdown("<h2 id='prediction'> 📈 Model Metrics</h2>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", f"{mae:.2f}")
        col2.metric("RMSE", f"{rmse:.2f}")
        col3.metric("R² Score", f"{r2:.2f}")
        st.progress(r2)

        # Historical and Predicted Data Visualization
        st.subheader("📉 Actual vs Predicted Prices")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=data.index[-len(y_test):], y=y_test, mode='lines', name='True Closing Price', line=dict(color='#00ff7f')))
        fig1.add_trace(go.Scatter(x=data.index[-len(y_test):], y=y_pred, mode='lines', name='Predicted Closing Price', line=dict(color='#ff4500')))
        fig1.update_layout(
            template='plotly_dark',
            title="True vs Predicted Closing Prices",
            xaxis_title="Date",
            yaxis_title="Price",
            legend_title="Legend"
        )
        st.plotly_chart(fig1, use_container_width=True)

 # Advanced Forecasting Function
        def generate_next_features(last_pred, last_features):
            """
            Generate next set of features based on last prediction and previous features
            """
            # Add some randomness to simulate market volatility
            noise = np.random.normal(0, 0.01 * last_pred)
            
            new_close = last_pred * (1 + noise)
            new_high = new_close * (1 + abs(noise))
            new_low = new_close * (1 - abs(noise))
            new_open = new_close * (1 + np.random.normal(0, 0.005))
            
            return np.array([new_open, new_high, new_low, new_close]).reshape(1, -1)

        # Forecast future prices
        last_data = X.iloc[-1].values.reshape(1, -1)  # Get the last row of data
        future_predictions = []

        for _ in range(forecast_days):
            # Predict the next price
            pred_price = model.predict(last_data)[0]
            future_predictions.append(pred_price)
            
            # Generate next input features
            last_data = generate_next_features(pred_price, last_data)

        # Plot future forecast
        st.markdown("<h2 id='forecast'> 🔮 Future Forecast</h2>", unsafe_allow_html=True)
        future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
        forecast_df = pd.DataFrame({"Date": future_dates, "Forecasted Price": future_predictions})

        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(
            x=forecast_df['Date'], y=forecast_df['Forecasted Price'],
            mode='lines+markers', name='Forecasted Prices',
            line=dict(color='#00ff7f', width=3), marker=dict(size=8)))
        fig_forecast.update_layout(
            title="Future Forecasted Prices",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_white",
            margin=dict(l=40, r=40, t=40, b=40),
            hovermode="x unified"
        )
        st.plotly_chart(fig_forecast)

        # Display forecasted data
        st.subheader("📋 Forecasted Prices Table")
        st.write(forecast_df)

        # Add download button
        csv = forecast_df.to_csv(index=False)
        st.download_button("Download Forecast Data", data=csv, file_name="forecast.csv", mime="text/csv")


# Footer
st.markdown("---")  # Add a horizontal line for separation
col1, col2, col3 = st.columns([3, 6, 1])  # Adjust column widths to center the footer
with col2:
    st.write("© 2024 Stock Market Prediction App. All rights reserved.", align="center")