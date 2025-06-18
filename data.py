import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

SUPPORTED_TICKERS = ['AAPL', 'AMZN', 'GOOGL', 'META', 'NFLX', 'SPY', 'VOO', 'QQQ', 'IVV', 'VTI']

def fetch_data(ticker, period='60d', interval='1m'):
    data = yf.download(ticker, period=period, interval=interval, progress=False)
    data = data[['Close']].dropna()
    return data

def prepare_sequences(data, lookback=60):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data.values)
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i-lookback:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

def get_recent_sequence(data, lookback=60, scaler=None):
    last_seq = data.values[-lookback:]
    if scaler:
        last_seq = scaler.transform(last_seq)
    last_seq = np.reshape(last_seq, (1, lookback, 1))
    return last_seq
