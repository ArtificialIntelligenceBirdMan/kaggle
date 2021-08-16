import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import glob
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')
pd.set_option('max_columns',300)
pd.set_option('max_rows',500)

def calculate_wap(df,method):
    if method == 1:
        wap = (df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']) / (df['bid_size1'] + df['ask_size1'])
    if method == 2:
        wap = (df['bid_price2'] * df['ask_size2'] + df['ask_price2'] * df['bid_size2']) / (df['bid_size2'] + df['ask_size2'])
    return wap

def log_return(series):
    return np.log(series).diff()

def realized_volatility(series):
    series = log_return(series)
    return np.sqrt(np.sum(series**2))

def count_unique(series):
    return len(np.unique(series))
