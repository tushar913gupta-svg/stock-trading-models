import numpy as np
import yfinance as yf
import tensorflow as tf

from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin,
    login_user, login_required, logout_user, current_user
)
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
def load_user(user_id):
    return User.query.get(int(user_id))


def fetch_data(ticker, lookback=300):
    df = yf.Ticker(ticker).history(period=f"{lookback}d").reset_index()

    df["SMA_5"] = df["Close"].rolling(5).mean()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2

    df.dropna(inplace=True)

    eps = 1e-9
    df["Close_Norm"] = (df["Close"] - df["Close"].mean()) / (df["Close"].std() + eps)
    df["SMA_5_Norm"] = (df["SMA_5"] - df["SMA_5"].mean()) / (df["SMA_5"].std() + eps)
    df["RSI_Norm"] = (df["RSI"] - df["RSI"].mean()) / (df["RSI"].std() + eps)
    df["MACD_Norm"] = (df["MACD"] - df["MACD"].mean()) / (df["MACD"].std() + eps)

    return df.reset_index(drop=True)


def get_model(ticker):
    info = yf.Ticker(ticker).info
    sector = (info.get("sector") or "").lower()
    industry = (info.get("industry") or "").lower()

    if "semiconductor" in industry:
        return "models/semiconductor_stocks.keras"

    if "defense" in industry or "aerospace" in industry:
        return "models/defence_stocks.keras"

    if "auto" in industry or "vehicle" in industry:
        return "models/auto_stocks.keras"

    if "pharmaceutical" in industry or "biotech" in industry:
        return "models/pharma_stocks.keras"

    if "oil" in industry or "gas" in industry or "energy" in sector:
        return "models/energy_stocks.keras"

    if "infrastructure" in industry or "engineering" in industry:
        return "models/infra_stocks.keras"

    if "logistics" in industry or "transport" in industry or "shipping" in industry:
        return "models/logistics_stocks.keras"

    if "industrial" in sector or "manufacturing" in industry:
        return "models/industrial_stocks.keras"

    if "financial" in sector:
        if "payment" in industry or "credit" in industry or "lending" in industry:
            return "models/fintech_stocks_alt.keras"
        return "models/finance_stocks.keras"

    if "retail" in industry or "internet retail" in industry or "consumer" in sector:
        return "models/ecom_stocks.keras"

    if "technology" in sector:
        return "models/tech_stocks_alt.keras"

    return "models/tech_stocks_alt.keras"


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
                self.balance -= price * qty
        elif action == 2 and self.inventory:
            bp, qty = self.inventory.pop(0)
            self.balance += price * qty


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        user = User(
            username=request.form["username"],
            password_hash=generate_password_hash(request.form["password"])
        )
        try:
            db.session.add(user)
            db.session.commit()
        except Exception:
            db.session.rollback()
            return redirect(url_for("login", exists=True))

        flash("Account created. Please log in.", "success")
        return redirect("/login")
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login(exists=False):
    if request.method == "POST":
        if exists:
            flash("Account already created. Please log in.", "danger")
        user = User.query.filter_by(username=request.form["username"]).first()
        if user and check_password_hash(user.password_hash, request.form["password"]):
            login_user(user)
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
    return render_template("dashboard.html", portfolios=portfolios)

@app.route("/portfolio", methods=["POST"])
@login_required
def create_portfolio():
    p = Portfolio(
        user_id=current_user.id,
        ticker=request.form["ticker"].upper(),
        balance=float(request.form["balance"])
    )
    db.session.add(p)
    db.session.commit()
    return redirect("/")

@app.route("/trade/<ticker>")
@login_required
def trade(ticker):
    p = Portfolio.query.filter_by(
        user_id=current_user.id,
        ticker=ticker
    ).first()

    db_inv = Inventory.query.filter_by(portfolio_id=p.id).all()
    inventory = [(i.buy_price, i.qty) for i in db_inv]

    df = fetch_data(ticker)
    env = TradingEnv(df, p.balance, inventory)

    model = get_model(ticker)

    model = tf.keras.models.load_model(model)
    action = int(np.argmax(model(np.expand_dims(env.get_state(), 0))))
    env.apply_action(action)

    Inventory.query.filter_by(portfolio_id=p.id).delete()

    for bp, qty in env.inventory:
        db.session.add(
            Inventory(
                portfolio_id=p.id,
                buy_price=bp,
                qty=qty
            )
        )

    p.balance = env.balance
    db.session.commit()

    return render_template(
        "trade.html",
        ticker=ticker,
        action=["HOLD", "BUY", "SELL"][action],
        balance=round(p.balance, 2),
        units=sum(q for _, q in env.inventory)
    )

@app.route("/balance/<int:pid>", methods=["POST"])
@login_required
def update_balance(pid):
    p = Portfolio.query.get(pid)
    if request.form["amount"] == '':
        return redirect(url_for("dashboard"))
    p.balance += float(request.form["amount"])
    db.session.commit()
    return redirect(url_for("dashboard"))

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
