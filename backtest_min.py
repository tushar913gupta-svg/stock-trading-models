import random
import matplotlib.pyplot as plt
import os
import pandas as pd
import yfinance as yf
import numpy as np
import tensorflow as tf

def fetch_data(ticker):
    path = f'./extra stock data/{ticker}TEST10.csv'
    if not os.path.exists(path):
        df = yf.download(ticker, start='2015-01-01', end='2025-01-01')
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        df.to_csv(f'./extra stock data/{ticker}TEST10.csv')
    else:
        df = pd.read_csv(path)

    df = df.sort_values('Date').reset_index(drop=True)
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))

    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    df.dropna(inplace=True)

    eps = 1e-9

    df['Close_Mean'] = df['Close'].rolling(60).mean()
    df['Close_Std'] = df['Close'].rolling(60).std()
    df['Close_Norm'] = (df['Close'] - df['Close_Mean']) / (df['Close_Std'] + eps)

    df['SMA_5_Mean'] = df['SMA_5'].rolling(60).mean()
    df['SMA_5_Std'] = df['SMA_5'].rolling(60).std()
    df['SMA_5_Norm'] = (df['SMA_5'] - df['SMA_5_Mean']) / (df['SMA_5_Std'] + eps)

    df['RSI_Mean'] = df['RSI'].rolling(60).mean()
    df['RSI_Std'] = df['RSI'].rolling(60).std()
    df['RSI_Norm'] = (df['RSI'] - df['RSI_Mean']) / (df['RSI_Std'] + eps)

    df['MACD_Mean'] = df['MACD'].rolling(60).mean()
    df['MACD_Std'] = df['MACD'].rolling(60).std()
    df['MACD_Norm'] = (df['MACD'] - df['MACD_Mean']) / (df['MACD_Std'] + eps)

    df.dropna(inplace=True)

    return df.reset_index(drop=True)

class TradingEnv:
    def __init__(self, df, initial_balance=10000, window=200, max_units=1000):
        self.df = df.reset_index(drop=True)
        self.initial_balance = float(initial_balance)
        self.window = int(window)
        self.max_units = int(max_units)*(initial_balance/10000)
        self.states = ['Close_Norm', 'SMA_5_Norm', 'RSI_Norm', 'MACD_Norm']
        self.reset()
        self.state_size = len(self.get_state())
        self.action_space = 3

    def reset(self, deterministic=False):
        self.start = 0 if deterministic else random.randint(0, max(0, len(self.df) - self.window - 1))
        self.current_step = self.start
        self.end = min(len(self.df) - 1, self.start + self.window)
        self.balance = float(self.initial_balance)
        self.inventory = []
        self.total_profit = 0.0
        self.prev_equity = self.initial_balance
        self.done = False
        return self.get_state()

    def get_state(self):
        base = self.df.loc[self.current_step, self.states].values.astype(np.float32)
        pos_qty = sum(q for _, q in self.inventory)
        price = float(self.df.loc[self.current_step, 'Close'])
        equity = self.get_equity(price)
        pos_val = (pos_qty * price) / (equity + 1e-9)
        cash_frac = self.balance / (equity + 1e-9)
        return np.concatenate([base, [pos_val, cash_frac]])

    def get_equity(self, price):
        return self.balance + sum(q * price for _, q in self.inventory)

    def step(self, action):
        price = float(self.df.loc[self.current_step, 'Close'])
        reward = 0.0

        if action == 1:
            affordable = int(self.balance // price)
            if affordable > 0:
                buy_qty = max(1, int(min(affordable * 0.35, self.max_units)))
                self.inventory.append([price, buy_qty])
                self.balance -= price * buy_qty
            else:
                reward -= 0.1
                action = 0

        elif action == 2:
            if self.inventory:
                buy_price, qty = self.inventory.pop(0)
                revenue = price * qty
                profit = revenue - (buy_price * qty)
                self.balance += revenue
                self.total_profit += profit
                reward += profit
            else:
                reward -= 0.1
                action = 0

        self.current_step += 1

        if self.current_step >= self.end:
            final_price = float(self.df.loc[self.current_step, 'Close'])
            if self.inventory:
                total_qty = sum(q for _, q in self.inventory)
                total_cost = sum(bp * q for bp, q in self.inventory)
                revenue = total_qty * final_price
                self.balance += revenue
                self.total_profit += revenue - total_cost
                self.inventory = []
            self.done = True

        next_price = float(self.df.loc[self.current_step, 'Close'])
        equity = self.get_equity(next_price)
        eq_diff = equity - self.prev_equity

        reward += eq_diff * 0.035

        pos_value = sum(q for _, q in self.inventory) * next_price
        reward -= 0.0002 * pos_value

        if self.inventory:
            reward += 0.002 * eq_diff

        self.prev_equity = equity
        return self.get_state(), float(reward), self.done, self.total_profit, action

def evaluate(ticker, actor):
    df = fetch_data(ticker)
    env = TradingEnv(df, window=len(df) - 1)

    state = env.reset(deterministic=True)
    done = False
    signals = []

    while not done:
        logits = actor(state.reshape(1, -1))
        action = int(tf.argmax(logits[0]).numpy())
        state, _, done, _, action = env.step(action)
        signals.append(action)

    if len(signals) < len(df):
        signals.extend([0] * (len(df) - len(signals)))

    df_out = df.copy()
    df_out['Signal'] = signals[:len(df)]
    df_out['Signal_Label'] = df_out['Signal'].map({0: 'HOLD', 1: 'BUY', 2: 'SELL'})

    df_out['Signal'] = signals
    df_out['Signal_Label'] = df_out['Signal'].map(
        {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
    )

    final_balance = env.balance
    final_profit = final_balance - env.initial_balance

    arr = ((1+(abs(final_profit)/env.initial_balance))**0.1 - 1)*100

    return arr if final_profit > 0 else (-1)*arr


industries = {
    "finance_stocks_alt.keras": [
        "JPM", "BAC", "WFC", "GS", "MS",
        "C", "USB", "PNC", "AIG", "BLK"
    ],

    "tech_stocks_alt.keras": [
        "AAPL", "MSFT", "GOOGL", "META", "ADBE",
        "NVDA", "INTC", "AMD", "CSCO", "CRM"
    ],

    "pharma_stocks_alt.keras": [
        "PFE", "MRK", "LLY", "JNJ", "AMGN",
        "GILD", "BMY", "REGN", "VRTX", "BIIB"
    ],

    "ecom_stocks.keras": [
        "AMZN", "WMT", "COST", "TGT", "HD",
        "LOW", "DG", "DLTR", "BBY", "TJX"
    ],

    "industrial_stocks.keras": [
        "CAT", "DE", "GE", "ETN", "HON",
        "LMT", "NOC", "GD", "GM", "F"
    ],

    "logistics_stocks.keras": [
        "UPS", "FDX", "DAL", "LUV", "AAL",
        "CSX", "NSC", "UNP", "ODFL", "JBHT"
    ],

    "fintech_stocks_alt.keras": [
        "PYPL", "AFRM", "GPN", "FIS",
        "FISV", "ADP", "V", "MA", "FOUR"
    ],
}

def print_table(rows):
    print("+----------+----------------------+------------+")
    print("| ticker   | industry             | profit     |")
    print("+----------+----------------------+------------+")
    for t, ind, p in rows:
        print(f"| {t:<8} | {ind:<20} | {p:<10.4f} |")
    print("+----------+----------------------+------------+")

rows = []

for model, tickers in industries.items():
    actor = tf.keras.models.load_model(f"models/{model}")
    industry = model.replace(".keras","")
    for t in tickers:
        p = evaluate(t, actor)
        rows.append((t, industry, p))

print_table(rows)
