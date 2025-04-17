import yfinance as yf
import os
import pandas as pd
from config import START_DATE, END_DATE

def fetch_data(ticker, save_dir="data"):
    try:
        df = yf.download(ticker, start=START_DATE, end=END_DATE)
        if not df.empty:
            os.makedirs(save_dir, exist_ok=True)
            df.to_csv(f"{save_dir}/{ticker}.csv")
            return True
    except Exception as e:
        print(f"{ticker} 에러: {e}")
    return False
