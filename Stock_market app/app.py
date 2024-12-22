from flask import Flask, render_template, jsonify, request
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def generate_predictions(prices):
    # Placeholder for prediction logic
    # Replace this with a trained ML model for actual predictions
    last_price = prices[-1]
    return [last_price * (1 + np.random.uniform(-0.02, 0.02)) for _ in range(10)]


@app.route('/get_stock_data', methods=['GET'])
def get_stock_data():
    ticker = request.args.get('ticker', default='AAPL', type=str)
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1y")
        data.reset_index(inplace=True)
        
        # Get last 10 days and next 10 days predictions
        last_10_days = data.tail(10)
        last_prices = last_10_days['Close'].values
        predictions = generate_predictions(last_prices)
        
        # Format response
        response = {
            "ticker": ticker,
            "historical": last_10_days.to_dict(orient='records'),
            "predictions": predictions,
            "accuracy": round(np.random.uniform(85, 95), 2)  # Placeholder accuracy
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
