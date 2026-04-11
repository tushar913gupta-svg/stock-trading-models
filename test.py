import random
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import tensorflow as tf
import pandas as pd


def fetch_data(ticker):
    df = yf.download(ticker, start='2025-01-01', end='2026-02-02')
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df.to_csv(f'./Stock Data/{ticker}TEST.csv')

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
        self.max_units = int(max_units) * (initial_balance / 10000)
        self.states = ['Close_Norm', 'SMA_5_Norm', 'RSI_Norm', 'MACD_Norm']
        self.sec_fee_rate = 27.80 / 1_000_000
        self.finra_taf_rate = 0.000145
        self.finra_taf_cap = 7.27
        self.spread_rate = 0.0002
        self.slippage_rate = 0.0003
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
        spread = price * self.spread_rate
        bid = price - spread / 2
        ask = price + spread / 2
        reward = 0.0

        if action == 1:
            buy_price = ask * (1 + self.slippage_rate)
            affordable = int(self.balance // buy_price)
            if affordable > 0:
                buy_qty = max(1, int(min(affordable * 0.35, self.max_units)))
                cost = buy_price * buy_qty
                self.inventory.append([buy_price, buy_qty])
                self.balance -= cost
            else:
                reward -= 0.1
                action = 0

        elif action == 2:
            if self.inventory:
                buy_price, qty = self.inventory.pop(0)
                sell_price = bid * (1 - self.slippage_rate)
                gross_revenue = sell_price * qty
                sec_fee = gross_revenue * self.sec_fee_rate
                finra_taf = min(self.finra_taf_rate * qty, self.finra_taf_cap)
                fees = sec_fee + finra_taf
                net_revenue = gross_revenue - fees
                profit = net_revenue - (buy_price * qty)
                self.balance += net_revenue
                self.total_profit += profit
                reward += profit
            else:
                reward -= 0.1
                action = 0

        self.current_step += 1

        if self.current_step >= self.end:
            final_price = float(self.df.loc[self.current_step, 'Close'])
            spread = final_price * self.spread_rate
            bid = final_price - spread / 2
            if self.inventory:
                total_qty = sum(q for _, q in self.inventory)
                total_cost = sum(bp * q for bp, q in self.inventory)
                sell_price = bid * (1 - self.slippage_rate)
                gross_revenue = sell_price * total_qty
                sec_fee = gross_revenue * self.sec_fee_rate
                finra_taf = min(self.finra_taf_rate * total_qty, self.finra_taf_cap)
                fees = sec_fee + finra_taf
                net_revenue = gross_revenue - fees
                self.balance += net_revenue
                self.total_profit += net_revenue - total_cost
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

    buy_idx = df_out.index[df_out['Signal'] == 1]
    sell_idx = df_out.index[df_out['Signal'] == 2]

    plt.figure(figsize=(14, 6))
    plt.plot(df_out['Close'].values, label='Close Price', c="black")
    plt.scatter(buy_idx, df_out.loc[buy_idx, 'Close'], marker='^', s=80, c="green")
    plt.scatter(sell_idx, df_out.loc[sell_idx, 'Close'], marker='v', s=80, c="red")
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

def stock_scoring(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        score = 0
        reasons = []

        margins = info.get('profitMargins', 0)

        if margins > 0.25:
            score += 30
        elif margins > 0.15:
            score += 20
        elif margins > 0.05:
            score += 10
        elif margins > 0:
            score += 5

        elif margins > -0.05:
            score -= 5
        elif margins > -0.15:
            score -= 10
        else:
            score -= 20

        growth = info.get('revenueGrowth', 0)

        if growth > 0.20:
            score += 20
        elif growth > 0.10:
            score += 15
        elif growth > 0:
            score += 5
        elif growth > -0.05:
            score -= 5
        elif growth > -0.15:
            score -= 10
        else:
            score -= 20

        roe = info.get('returnOnEquity', 0)

        if roe > 0.20:
            score += 20
            reasons.append(f"Elite ROE (>20%: {roe:.1%})")
        elif roe > 0.10:
            score += 10
        elif roe > 0:
            score += 5
        elif roe > -0.10:
            score -= 5
        else:
            score -= 15
            reasons.append("Management Destroying Value (Negative ROE)")

        cash = info.get('totalCash', 0)
        debt = info.get('totalDebt', 0)

        if cash > (debt * 1.5):
            score += 30
            reasons.append("Fortress Balance Sheet (Cash >> Debt)")
        elif cash > debt:
            score += 20
        elif cash > (debt * 0.5):
            score += 5
        elif debt > (cash * 3):
            score -= 25
        elif debt > (cash * 2):
            score -= 15
        else:
            score -= 5

        score = max(0, min(100, score))

        print(f"\nScore: {score}/100\n")
        print("Audit Details:")
        print(f"Debt: ${debt}")
        print(f"Cash: ${cash}")
        print(f"Return on Equity: {roe * 100}%")
        print(f"Revenue Growth: {growth * 100}%")
        print(f"Profit Margin: {margins * 100}%")

    except Exception as e:
        print(f"Error scoring {ticker}: {e}")


while True:
    mode = input("Select Mode: ").lower()

    if mode == 'exit':
        break

    if mode == 'score':
        ticker = input("Enter the stock: ").upper()
        stock_scoring(ticker)

    elif mode == 'test':
        model_name = input("Enter the model (according to the models folder): ")
        ticker = input("Enter the stock: ").upper()
        try:
            actor = tf.keras.models.load_model(f"models/{model_name}.keras")
            test(ticker, actor)
        except Exception as e:
            print(f"Error loading model: {e}")