import random
import matplotlib.pyplot as plt
import os
import pandas as pd
import yfinance as yf
import numpy as np
import tensorflow as tf

def fetch_data(ticker):
    path = f'./Stock Data/{ticker}.csv'
    if not os.path.exists(path):
        df = yf.download(f'{ticker}', start='2022-01-01', end='2025-01-01')
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        df.to_csv(f'./Stock Data/{ticker}.csv')
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
    df['Close_Norm'] = (df['Close'] - df['Close'].mean()) / (df['Close'].std() + eps)
    df['SMA_5_Norm'] = (df['SMA_5'] - df['SMA_5'].mean()) / (df['SMA_5'].std() + eps)
    df['RSI_Norm'] = (df['RSI'] - df['RSI'].mean()) / (df['RSI'].std() + eps)
    df['MACD_Norm'] = (df['MACD'] - df['MACD'].mean()) / (df['MACD'].std() + eps)

    return df.reset_index(drop=True)

class TradingEnv:
    def __init__(self, df, initial_balance=10000, window=200, max_units=1000):
        self.df = df.reset_index(drop=True)
        self.initial_balance = float(initial_balance)
        self.window = int(window)
        self.max_units = int(max_units)
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
        return self.get_state(), float(reward), self.done, self.total_profit

def evaluate(ticker, actor):
    df = fetch_data(ticker)
    env = TradingEnv(df, window=len(df) - 1)

    state = env.reset(deterministic=True)
    done = False
    signals = []

    while not done:
        logits = actor(state.reshape(1, -1))
        action = int(tf.argmax(logits[0]).numpy())
        signals.append(action)
        state, _, done, _ = env.step(action)

    if len(signals) < len(df):
        signals.extend([0] * (len(df) - len(signals)))

    df_out = df.copy()
    df_out['Signal'] = signals[:len(df)]
    df_out['Signal_Label'] = df_out['Signal'].map({0: 'HOLD', 1: 'BUY', 2: 'SELL'})

    balance = float(env.initial_balance)
    executed = []

    for _, row in df_out.iterrows():
        price = float(row['Close'])
        signal = int(row['Signal'])

        if signal == 1:
            if balance >= price:
                executed.append(1)
                balance -= price
            else:
                executed.append(0)
        elif signal == 2:
            executed.append(2)
            balance += price
        else:
            executed.append(0)

    df_out['Signal'] = executed
    df_out['Signal_Label'] = df_out['Signal'].map(
        {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
    )

    buy_idx = df_out.index[df_out['Signal'] == 1]
    sell_idx = df_out.index[df_out['Signal'] == 2]

    plt.figure(figsize=(14, 6))
    plt.plot(df_out['Close'].values, label='Close Price')
    plt.scatter(buy_idx, df_out.loc[buy_idx, 'Close'], marker='^', s=80)
    plt.scatter(sell_idx, df_out.loc[sell_idx, 'Close'], marker='v', s=80)
    plt.title(f"{ticker} — Price with Executed BUY / SELL Signals")
    plt.xlabel("Time Step")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()

    final_balance = env.balance
    final_profit = final_balance - env.initial_balance

    return df_out, final_profit, final_balance

def test(ticker, actor):
    df, profit, balance = evaluate(f"{ticker}", actor)
    print(f"Final realized profit ({ticker}):", profit)
    print("Final balance:", balance)
    print("Total BUYs:", (df['Signal'] == 1).sum())
    print("Total SELLs:", (df['Signal'] == 2).sum())
    print("Total HOLDs:", (df['Signal'] == 0).sum())
    print(df[['Close', 'Signal', 'Signal_Label']].head(40))

while True:
    model = input("Enter the model (according to the models folder): ").lower()

    if model == 'exit':
        break

    ticker = input("Enter the stock: ").upper()
    actor = tf.keras.models.load_model(f"models/{model}.keras") #replace with whichever model you want to test
    test(f"{ticker}TEST", actor) #replace the tickers accordingly (<ticker>TEST is ytd data)



