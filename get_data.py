import requests
import pandas as pd
from datetime import datetime, timedelta
import time 
from utility import get_config, save_to_csv
import matplotlib.pyplot as plt
import yfinance as yf

def get_cryptocompare_data(crypto_symbol, vs_currency='USD', limit=365, start_date=None):
    # read info pf API
    api_key = get_config('api_key')
    base_url = get_config('base_url')
    
    try:       
        # init for date of request
        step_request = 0
        current_date = datetime.strptime(start_date, '%Y-%m-%d')
        all_data = [] 
        
        while current_date <= datetime.now():
            current_date += timedelta(days=limit)

            # setting param for request
            params = {
                'fsym': crypto_symbol,
                'tsym': vs_currency,
                'limit': limit,
                'toTs': int(current_date.timestamp()),
                'api_key': api_key
            }
            
            response = requests.get(base_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'Data' in data and 'Data' in data['Data']:
                    all_data.extend(data['Data']['Data'])
                    
                else:
                    print(f"Unexpected data format in the response: {data}")
                    break
            else:
                print(f"Error {response.status_code}: {response.text}")
                break
                
            step_request += 1
            print(step_request)
            
            # sleep to skip block from API 
            time.sleep(1)
        
        df = pd.DataFrame(all_data)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.rename(columns={'time': 'Date', 'high': 'High', 'low': 'Low', 'open': 'Open', 
                           'close': 'Close', 'volumefrom': 'Volume', 'volumeto': 'Market Cap'}, inplace=True)
        
        return df
    
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    print("Starting to get data...")

    crypto_symbol = 'BTC'
    currency = 'USD'

    #df = get_cryptocompare_data(crypto_symbol, currency, limit=365, start_date='2010-01-01')
    #data = df[df['Open'] != 0]

    data = yf.Ticker('BTC-USD').history(period='max')

    file_name = get_config('file_name2')
    save_to_csv(data, file_name)

    print("Finish getting data.")
