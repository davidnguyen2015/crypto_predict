import configparser
import os
import pandas as pd

def get_config(key):
    config = configparser.ConfigParser()
    config.read(os.path.dirname(__file__) + '/config.ini')

    if 'cryptocompare' not in config:
        raise Exception("Section 'cryptocompare' not found in config.ini.")
        
    return config.get('cryptocompare', key)

def save_to_csv(df, filename):
    df.to_csv(filename, index=False)
    #print(f'Data saved to {filename}')