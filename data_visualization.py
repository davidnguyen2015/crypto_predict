import pandas as pd
import matplotlib.pyplot as plt
from utility import get_config
import os

def pilot_open_close_column(df):
    # convert 'Date' to datetime setting to index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # avg value of columns 'Open', 'Close'
    monthly_avg_prices = df[['Open', 'Close']].resample('M').mean()

    plt.figure(figsize=(12, 6))
    plt.plot(monthly_avg_prices.index, monthly_avg_prices['Open'], marker='o', linestyle='-', color='r', label='Average Open Price')
    plt.plot(monthly_avg_prices.index, monthly_avg_prices['Close'], marker='o', linestyle='-', color='g', label='Average Close Price')

    plt.title('Monthly Average Prices of Crypto (USD)')
    plt.xlabel('Date')
    plt.ylabel('Average Price')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
    plt.show()

def pilot_high_low_open(df):
    # convert 'Date' to datetime setting to index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # avg value of columns 'High', 'Low'
    monthly_avg_prices = df[['High', 'Low']].resample('M').mean()

    plt.figure(figsize=(12, 6))
    plt.plot(monthly_avg_prices.index, monthly_avg_prices['High'], marker='o', linestyle='-', color='r', label='Average High Price')
    plt.plot(monthly_avg_prices.index, monthly_avg_prices['Low'], marker='o', linestyle='-', color='g', label='Average Low Price')

    plt.title('Monthly Average Prices of Crypto (USD)')
    plt.xlabel('Date')
    plt.ylabel('Average Price')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
    plt.show()

def pilot_volume_column(df):
    # convert 'Date' to datetime setting to index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # avg value of columns 'Volume'
    monthly_avg_open = df['Volume'].resample('M').mean()

    plt.figure(figsize=(12, 6))
    plt.plot(monthly_avg_open.index, monthly_avg_open.values, marker='o', linestyle='-', color='b', label='Monthly Average Volume')
    plt.title('Monthly Average Volume of Crypto')
    plt.xlabel('Date')
    plt.ylabel('Average Volume')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
    plt.show()

def view_plot(df):
    plt.figure(figsize=(16,8))
    plt.title('Close Price', fontsize=24)
    plt.plot(df['Close'])
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('USD', fontsize=18)
    plt.show()

if __name__ == "__main__":
    file_name = get_config('file_name')
    df = pd.read_csv(file_name)
    
    #pilot_open_close_column(df)