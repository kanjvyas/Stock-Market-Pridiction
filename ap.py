import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go

# Page Config
st.set_page_config(layout="wide", page_title="Stock Market Prediction", page_icon="📈")


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
           


st.markdown(page_bg_img, unsafe_allow_html=True)
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

# Sidebar Navigation
st.sidebar.header("📊 Stock Market Prediction")


# User Input in Sidebar
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, MSFT):", "HDB")
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
    data.dropna(inplace=True)

    if len(data) < 2:
        st.error("Not enough data after processing for training. Please try a different ticker.")
    else:
        # Split data into features and target
        X = data[['Open', 'High', 'Low', 'Close']]
        y = data['Target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model Training
        model = XGBRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Model Accuracy
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

        # Model Performance Metrics
        st.subheader("📈 Model Accuracy")
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
            template='plotly_dark',
            title="True vs Predicted Closing Prices",
            xaxis_title="Date",
            yaxis_title="Price",
            legend_title="Legend"
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Forecast Visualization
        st.subheader("🔮 Forecasted Prices (Next 10 Days)")
        future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=future_days)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=future_dates, y=future_predictions, mode='lines+markers', name="Forecasted Prices", line=dict(color='#00ff7f')))
        fig2.update_layout(
            template='plotly_dark',
            title="Forecast for Next 10 Days",
            xaxis_title="Date",
            yaxis_title="Price",
            legend_title="Legend"
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Forecast Table
        st.subheader("📋 Forecasted Prices Table")
        forecast_df = pd.DataFrame({"Date": future_dates, "Forecasted Price": future_predictions})
        st.dataframe(forecast_df.style.set_properties(**{'background-color': '#1e1e1e', 'color': '#ffffff'}))


# Footer
st.markdown("© 2024 Stock Market Prediction App. All rights reserved.", unsafe_allow_html=True)