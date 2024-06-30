import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os
from utility import get_config, save_to_csv

def predict_btc(df, days_ahead):
    # Set column 'Close' to predict target
    df['Prediction'] = df['Close'].shift(-days_ahead)

    # create dataset to X and Y
    X = np.array(df[['Close']])
    X = X[:-days_ahead]
    y = np.array(df['Prediction'])
    y = y[:-days_ahead]

    # split data for train, test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # build model for training
    model = LinearRegression()
    model.fit(X_train, y_train)

    # predict for data "Close"
    X_future = df[['Close']].tail(days_ahead).values
    forecast = model.predict(X_future)

    # view chart of predict
    plt.figure(figsize=(10, 6))
    plt.plot(df.index[-100:], df['Close'][-100:], label='Historical Prices')
    plt.plot(df.index[-days_ahead:], forecast, label='Predicted Prices', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('BTC Price')
    plt.title('Bitcoin Price Prediction')
    plt.legend()
    plt.show()

    return forecast

if __name__ == "__main__":
    df = pd.read_csv(get_config('file_name2'))  
    
    # number of days of predict
    days_ahead = 10
    forecasted_prices = predict_btc(df, days_ahead)
    print(f"Forecasted prices for the next {days_ahead} days: {forecasted_prices}")
