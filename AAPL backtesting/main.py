import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import yfinance as yf

TICKER = "AAPL"
START_DATE = "2005-01-01"
END_DATE = "2025-01-01"

FAST_SMA = 20   #looks at a 20 day period of prices
SLOW_SMA = 100  #looks at a 100 day period of prices

INITIAL_CAPITAL = 1.0   #not dollar amounts but a normalized portfolio
TCOST = 0.0005          #5 bps or units per trade

#in all the functions df is used as a replacement for apple stock
def load_price_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    data = data[['Close']].rename(columns={'Close': 'price'})
    data.dropna(inplace=True)
    return data

def compute_features(df, fast, slow):
    df = df.copy()
    df['ret'] = df['price'].pct_change()
    df['sma_fast'] = df['price'].rolling(fast).mean()
    df['sma_slow'] = df['price'].rolling(slow).mean()
    return df

def generate_signals(df):
    df = df.copy()

    #buy when the fast SMA is above the slow SMA
    df['signal'] = 0
    df.loc[df['sma_fast'] > df['sma_slow'], 'signal'] = 1
    
    #shift the signals to avoid lookahead bias
    df['position'] = df['signal'].shift(1)
    
    return df

def apply_strategy(df, tcost):
    df = df.copy()
    
    #test to see if position is in the data frame
    if 'position' not in df.columns:
        raise KeyError("Column 'position' not found. Did you run generate_signals()?")

    df['strategy_ret'] = df['position'] * df['ret']

    #this snnipet changes out position on the stock
    df['trade'] = df['position'].diff().abs()
    df['tcost'] = df['trade'] * tcost

    df['strategy_net_ret'] = df['strategy_ret'] - df['tcost']

    return df

def compute_equity_curve(df):
    df = df.copy()

    df['buy_hold'] = (1 + df['ret']).cumprod()  #calculates the return if we didnt sell
    df['strategy'] = (1 + df['strategy_net_ret']).cumprod() #return of our strategy

    return df

def compute_metrics(returns, freq=252):
    returns = returns.dropna()

    mu = returns.mean() * freq
    sigma = returns.std() *np.sqrt(freq)
    sharpe = mu/sigma if sigma != 0 else np.nan

    max_dd = (returns.cumsum().expanding().max() - returns.cumsum()).max()

    return {
        "Annual Return": mu,
        "Annual Volume": sigma,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd
    }

def plot_results(df):
    plt.figure(figsize=(12,6))
    plt.plot(df.index, df['buy_hold'], label='Buy & Hold')
    plt.plot(df.index, df['strategy'], label='SMA Strategy')
    plt.legend()
    plt.title(f'{TICKER} SMA Backtest')
    plt.show()

df = load_price_data(TICKER, START_DATE, END_DATE)
df = compute_features(df, FAST_SMA, SLOW_SMA)
df = generate_signals(df)
df = apply_strategy(df, TCOST)
df = compute_equity_curve(df)

metrics = compute_metrics(df['strategy_net_ret'])

plot_results(df)
print(metrics)

