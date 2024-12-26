import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime

def create_output_directory(ticker):
    """
    Creates a dated directory for storing output related to the given ticker.
    """
    today = datetime.now().strftime('%Y-%m-%d')
    directory = os.path.join('data', today, ticker)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def get_stock_data(ticker, period="1y"):
    """
    Retrieves historical stock data using the specified period (e.g., '1y', '2y', etc.).
    """
    stock = yf.Ticker(ticker)
    return stock.history(period=period)

def add_moving_averages(df):
    """
    Adds various moving average columns to the DataFrame.
    """
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_100'] = df['Close'].rolling(window=100).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['SMA_300'] = df['Close'].rolling(window=300).mean()
    return df

def get_ratio_dataframe(ticker1, ticker2, date_range_input):
    """
    Calculate ratio data between two tickers, preserving OHLC structure.
    If date_range_input is something like '1y', it uses that period.
    If date_range_input has a comma, it treats it as start_date,end_date.
    """
    if ',' in date_range_input:
        start_date, end_date = [d.strip() for d in date_range_input.split(',')]
        df1 = yf.download(ticker1, start=start_date, end=end_date)
        df2 = yf.download(ticker2, start=start_date, end=end_date)
    else:
        df1 = yf.download(ticker1, period=date_range_input.strip())
        df2 = yf.download(ticker2, period=date_range_input.strip())

    ratio_df = pd.DataFrame()
    # Calculate true OHLC ratios
    ratio_df['Open'] = df1['Open'] / df2['Open']
    ratio_df['High'] = df1['High'] / df2['High']
    ratio_df['Low'] = df1['Low'] / df2['Low']
    ratio_df['Close'] = df1['Close'] / df2['Close']
    ratio_df['Volume'] = df1['Volume'] / df2['Volume']
    ratio_df.index = df1.index

    return ratio_df

