import numpy as np
import yfinance as yf
import tensorflow as tf

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user

from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.config["SECRET_KEY"] = "dev-secret"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///trading.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

class Portfolio(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    ticker = db.Column(db.String(10), nullable=False)
    balance = db.Column(db.Float, nullable=False)

class Inventory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    portfolio_id = db.Column(db.Integer, db.ForeignKey("portfolio.id"), nullable=False)
    buy_price = db.Column(db.Float, nullable=False)
    qty = db.Column(db.Integer, nullable=False)

@login_manager.user_loader
def load_user(uid):
    return User.query.get(int(uid))

def fetch_data(ticker, lookback=100):
    df = yf.Ticker(ticker).history(period=f"{lookback}d").reset_index()
    df["SMA_5"] = df[("Close")].rolling(5).mean()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))
    exp1 = df["Close"].ewm(span=12).mean()
    exp2 = df["Close"].ewm(span=26).mean()
    df["MACD"] = exp1 - exp2
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

    return df.reset_index(drop=True)

def get_model_path(ticker):
    info = yf.Ticker(ticker).info
    sector = (info.get("sector") or "").lower()
    industry = (info.get("industry") or "").lower()

    if "semiconductor" in industry:
        return "models/semiconductor_stocks.keras"

    if "defense" in industry or "aerospace" in industry:
        return "models/industrial_stocks.keras"
    if "auto" in industry:
        return "models/industrial_stocks.keras"
    if "energy" in sector:
        return "models/industrial_stocks.keras"

    if "pharma" in industry or "biotech" in industry:
        return "models/pharma_stocks.keras"

    if "logistics" in industry or "transport" in industry:
        return "models/logistics_stocks.keras"

    if "financial" in sector or "bank" in industry or "capital markets" in industry:
        return "models/finance_stocks.keras"

    if any(word in industry for word in [
        "credit services", "payment", "fintech", "information technology services"
    ]):
        return "models/fintech_stocks_alt.keras"

    if any(word in industry for word in [
        "internet retail", "discount stores", "department stores", "specialty retail", "catalog & mail order houses"
    ]):
        return "models/ecom_stocks.keras"

    if "tech" in sector:
        return "models/tech_stocks_alt.keras"

    return "models/tech_stocks_alt.keras"

def stock_scoring(ticker):
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

    return score


class TradingEnv:
    def __init__(self, df, balance, inventory):
        self.df = df
        self.balance = balance
        self.inventory = list(inventory)
        self.states = ["Close_Norm", "SMA_5_Norm", "RSI_Norm", "MACD_Norm"]
        self.step = len(df) - 1

    def get_state(self):
        row = self.df.iloc[self.step]
        base = row[self.states].values.astype(np.float32)
        price = float(row["Close"])
        qty = sum(q for _, q in self.inventory)
        equity = self.balance + qty * price
        pos_val = (qty * price) / (equity + 1e-9)
        cash_frac = self.balance / (equity + 1e-9)
        return np.concatenate([base, [pos_val, cash_frac]])

    def apply_action(self, action):
        price = float(self.df.iloc[self.step]["Close"])
        if action == 1:
            affordable = int(self.balance // price)
            if affordable > 0:
                qty = max(1, int(affordable * 0.35))
                self.inventory.append((price, qty))
                self.balance -= qty * price
        elif action == 2:
            if self.inventory:
                bp, qty = self.inventory.pop(0)
                self.balance += qty * price

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        u = User(username=request.form["username"], password_hash=generate_password_hash(request.form["password"]))
        try:
            db.session.add(u)
            db.session.commit()
        except:
            db.session.rollback()
            flash("Username already exists", "danger")
            return redirect(url_for("signup"))
        flash("Account created", "success")
        return redirect("/login")
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = User.query.filter_by(username=request.form["username"]).first()
        if u and check_password_hash(u.password_hash, request.form["password"]):
            login_user(u)
            return redirect("/")
        flash("Invalid credentials", "danger")
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect("/login")

@app.route("/")
@login_required
def dashboard():
    portfolios = Portfolio.query.filter_by(user_id=current_user.id).all()
    sp = yf.Ticker("SPY").history(period="60d").reset_index()
    labels = sp["Date"].dt.strftime("%Y-%m-%d").tolist()
    data = sp["Close"].tolist()
    return render_template("dashboard.html", portfolios=portfolios, labels=labels, data=data)

@app.route("/portfolio", methods=["POST"])
@login_required
def create_portfolio():
    p = Portfolio(user_id=current_user.id, ticker=request.form["ticker"].upper(), balance=float(request.form["balance"]))
    db.session.add(p)
    db.session.commit()
    return redirect("/")

@app.route("/trade/<ticker>")
@login_required
def trade(ticker):
    p = Portfolio.query.filter_by(user_id=current_user.id, ticker=ticker).first()

    db_inv = Inventory.query.filter_by(portfolio_id=p.id).all()
    inventory = [(i.buy_price, i.qty) for i in db_inv]

    df = fetch_data(ticker)
    env = TradingEnv(df, p.balance, inventory)

    model_path = get_model_path(ticker)
    model = tf.keras.models.load_model(model_path)

    state = np.expand_dims(env.get_state(), 0)
    action = int(np.argmax(model(state)))

    env.apply_action(action)

    Inventory.query.filter_by(portfolio_id=p.id).delete()
    for bp, qty in env.inventory:
        db.session.add(Inventory(portfolio_id=p.id, buy_price=bp, qty=qty))

    p.balance = env.balance
    db.session.commit()

    labels = df["Date"].dt.strftime("%Y-%m-%d").tolist()
    prices = df["Close"].tolist()

    return render_template("trade.html", ticker=ticker, action=["HOLD","BUY","SELL"][action], balance=round(p.balance,2), units=sum(q for _, q in env.inventory), labels=labels, prices=prices)

@app.route("/balance/<int:pid>", methods=["POST"])
@login_required
def balance(pid):
    p = Portfolio.query.get(pid)
    if request.form["amount"] != "":
        p.balance += float(request.form["amount"])
        db.session.commit()
    return redirect(url_for("dashboard"))

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
