import re
import numpy as np
import yfinance as yf
import tensorflow as tf

from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy.exc import IntegrityError
from forex_python.converter import CurrencyRates

app = Flask(__name__)
app.config["SECRET_KEY"] = "dev-secret"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///trading.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"


class User(UserMixin, db.Model):
    __tablename__ = "user"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(128), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class ExpenseCategory(db.Model):
    __tablename__ = "expense_category"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    description = db.Column(db.String(200))


class Expense(db.Model):
    __tablename__ = "expense"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    category_id = db.Column(db.Integer, db.ForeignKey("expense_category.id"))
    amount = db.Column(db.Float, nullable=False)
    date = db.Column(db.Date, nullable=False, default=datetime.utcnow)
    notes = db.Column(db.String(300))
    category = db.relationship("ExpenseCategory", backref="expenses")


class Budget(db.Model):
    __tablename__ = "budget"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    period = db.Column(db.String(20), nullable=False)
    limit_amount = db.Column(db.Float, nullable=False)


class Goal(db.Model):
    __tablename__ = "goal"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    name = db.Column(db.String(120), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Asset(db.Model):
    __tablename__ = "asset"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    name = db.Column(db.String(120), nullable=False)
    symbol = db.Column(db.String(20), unique=True, nullable=False)


class Portfolio(db.Model):
    __tablename__ = "portfolio"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    name = db.Column(db.String(120), nullable=False)
    type = db.Column(db.String(50))
    balance = db.Column(db.Float, nullable=False, default=0.0)
    holdings = db.relationship("Holding", backref="portfolio", lazy=True)


class Holding(db.Model):
    __tablename__ = "holding"
    id = db.Column(db.Integer, primary_key=True)
    portfolio_id = db.Column(db.Integer, db.ForeignKey("portfolio.id"), nullable=False)
    asset_id = db.Column(db.Integer, db.ForeignKey("asset.id"), nullable=False)
    buy_price = db.Column(db.Float, nullable=False)
    qty = db.Column(db.Integer, nullable=False)
    asset = db.relationship("Asset")


class Transaction(db.Model):
    __tablename__ = "transaction"
    id = db.Column(db.Integer, primary_key=True)
    portfolio_id = db.Column(db.Integer, db.ForeignKey("portfolio.id"), nullable=False)
    asset_id = db.Column(db.Integer, db.ForeignKey("asset.id"), nullable=False)
    type = db.Column(db.String(10), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    date = db.Column(db.DateTime, default=datetime.utcnow)
    asset = db.relationship("Asset")


class Model(db.Model):
    __tablename__ = "model"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    sector = db.Column(db.String(80))
    industry = db.Column(db.String(80))
    version = db.Column(db.String(20))


class Prediction(db.Model):
    __tablename__ = "prediction"
    id = db.Column(db.Integer, primary_key=True)
    model_id = db.Column(db.Integer, db.ForeignKey("model.id"), nullable=False)
    asset_id = db.Column(db.Integer, db.ForeignKey("asset.id"), nullable=False)
    action = db.Column(db.String(10))
    logit_hold = db.Column(db.Float)
    logit_buy = db.Column(db.Float)
    logit_sell = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    model_rel = db.relationship("Model")
    asset = db.relationship("Asset")


@login_manager.user_loader
def load_user(uid):
    return User.query.get(int(uid))


def valid_email(email):
    return re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email)


def safe_price(symbol):
    try:
        hist = yf.Ticker(symbol).history(period="1d")
        if hist.empty:
            return None
        return float(hist["Close"].iloc[-1])
    except:
        return None


def calculate_net_worth(user_id):
    portfolios = Portfolio.query.filter_by(user_id=user_id).all()
    total = 0.0
    for p in portfolios:
        total += p.balance
        for h in p.holdings:
            price = safe_price(h.asset.symbol)
            if price:
                total += h.qty * price
    return round(total, 2)


def fetch_data(symbol, lookback=100):
    try:
        df = yf.Ticker(symbol).history(period=f"{lookback}d").reset_index()
        if df.empty:
            return None
    except:
        return None

    df["SMA_5"] = df["Close"].rolling(5).mean()
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

    df["Close_Mean"] = df["Close"].rolling(60).mean()
    df["Close_Std"] = df["Close"].rolling(60).std()
    df["Close_Norm"] = (df["Close"] - df["Close_Mean"]) / (df["Close_Std"] + eps)

    df["SMA_5_Mean"] = df["SMA_5"].rolling(60).mean()
    df["SMA_5_Std"] = df["SMA_5"].rolling(60).std()
    df["SMA_5_Norm"] = (df["SMA_5"] - df["SMA_5_Mean"]) / (df["SMA_5_Std"] + eps)

    df["RSI_Mean"] = df["RSI"].rolling(60).mean()
    df["RSI_Std"] = df["RSI"].rolling(60).std()
    df["RSI_Norm"] = (df["RSI"] - df["RSI_Mean"]) / (df["RSI_Std"] + eps)

    df["MACD_Mean"] = df["MACD"].rolling(60).mean()
    df["MACD_Std"] = df["MACD"].rolling(60).std()
    df["MACD_Norm"] = (df["MACD"] - df["MACD_Mean"]) / (df["MACD_Std"] + eps)

    df.dropna(inplace=True)
    return df.reset_index(drop=True)


def fetch_data_range(symbol, start, end):
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    start_dt = (start_dt - timedelta(days=104)).strftime("%Y-%m-%d")

    df = yf.download(symbol, start=start_dt, end=end)
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df = df.reset_index()

    df["SMA_5"] = df["Close"].rolling(5).mean()
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

    df["Close_Mean"] = df["Close"].rolling(60).mean()
    df["Close_Std"] = df["Close"].rolling(60).std()
    df["Close_Norm"] = (df["Close"] - df["Close_Mean"]) / (df["Close_Std"] + eps)

    df["SMA_5_Mean"] = df["SMA_5"].rolling(60).mean()
    df["SMA_5_Std"] = df["SMA_5"].rolling(60).std()
    df["SMA_5_Norm"] = (df["SMA_5"] - df["SMA_5_Mean"]) / (df["SMA_5_Std"] + eps)

    df["RSI_Mean"] = df["RSI"].rolling(60).mean()
    df["RSI_Std"] = df["RSI"].rolling(60).std()
    df["RSI_Norm"] = (df["RSI"] - df["RSI_Mean"]) / (df["RSI_Std"] + eps)

    df["MACD_Mean"] = df["MACD"].rolling(60).mean()
    df["MACD_Std"] = df["MACD"].rolling(60).std()
    df["MACD_Norm"] = (df["MACD"] - df["MACD_Mean"]) / (df["MACD_Std"] + eps)

    df.dropna(inplace=True)
    return df.reset_index(drop=True)


def get_model_path(symbol):
    try:
        info = yf.Ticker(symbol).info
    except:
        return "models/finance_stocks_alt.keras"

    sector = (info.get("sector") or "").lower()
    industry = (info.get("industry") or "").lower()

    if "defense" in industry or "aerospace" in industry or "auto" in industry or "energy" in sector:
        return "models/industrial_stocks.keras"
    if "pharma" in industry or "biotech" in industry:
        return "models/pharma_stocks_alt.keras"
    if "logistics" in industry or "transport" in industry:
        return "models/logistics_stocks.keras"
    if "financial" in sector or "bank" in industry or "capital markets" in industry:
        return "models/finance_stocks_alt.keras"
    if any(w in industry for w in ["credit services", "payment", "fintech", "information technology services"]):
        return "models/fintech_stocks_alt.keras"
    if any(w in industry for w in ["internet retail", "discount stores", "department stores", "specialty retail"]):
        return "models/ecom_stocks.keras"
    if "tech" in industry or "semiconductor" in industry:
        return "models/tech_stocks_alt.keras"
    return "models/finance_stocks_alt.keras"


def stock_scoring(symbol):
    try:
        info = yf.Ticker(symbol).info
    except:
        return 0

    score = 0
    margins = info.get("profitMargins", 0) or 0
    if margins > 0.25: score += 30
    elif margins > 0.15: score += 20
    elif margins > 0.05: score += 10
    elif margins > 0: score += 5
    elif margins > -0.05: score -= 5
    elif margins > -0.15: score -= 10
    else: score -= 20

    growth = info.get("revenueGrowth", 0) or 0
    if growth > 0.20: score += 20
    elif growth > 0.10: score += 15
    elif growth > 0: score += 5
    elif growth > -0.05: score -= 5
    elif growth > -0.15: score -= 10
    else: score -= 20

    roe = info.get("returnOnEquity", 0) or 0
    if roe > 0.20: score += 20
    elif roe > 0.10: score += 10
    elif roe > 0: score += 5
    elif roe > -0.10: score -= 5
    else: score -= 15

    cash = info.get("totalCash", 0) or 0
    debt = info.get("totalDebt", 0) or 0
    if cash > (debt * 1.5): score += 30
    elif cash > debt: score += 20
    elif cash > (debt * 0.5): score += 5
    elif debt > (cash * 3): score -= 25
    elif debt > (cash * 2): score -= 15
    else: score -= 5

    return max(0, min(100, score))


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


def logits_to_probs(logits):
    e = np.exp(logits - np.max(logits))
    s = e / e.sum()
    return s


def _save_prediction(symbol, model_path, action_idx, logits):
    asset = Asset.query.filter_by(symbol=symbol).first()
    if not asset:
        try:
            name = yf.Ticker(symbol).info.get("longName", symbol)
        except:
            name = symbol
        asset = Asset(user_id=current_user.id, name=name, symbol=symbol)
        db.session.add(asset)
        db.session.flush()

    model_name = model_path.split("/")[-1].replace(".keras", "")
    mdl = Model.query.filter_by(name=model_name).first()
    if not mdl:
        mdl = Model(name=model_name, version="1.0")
        db.session.add(mdl)
        db.session.flush()

    action = ["HOLD", "BUY", "SELL"][action_idx]
    pred = Prediction(
        model_id=mdl.id,
        asset_id=asset.id,
        action=action,
        logit_hold=float(logits[0]),
        logit_buy=float(logits[1]),
        logit_sell=float(logits[2]),
    )
    db.session.add(pred)
    db.session.commit()


def fmt(n):
    if n is None:
        return "—"
    try:
        n = float(n)
    except:
        return "—"
    if abs(n) >= 1e12: return f"{n/1e12:.2f}T"
    if abs(n) >= 1e9: return f"{n/1e9:.2f}B"
    if abs(n) >= 1e6: return f"{n/1e6:.2f}M"
    if abs(n) >= 1e3: return f"{n/1e3:.2f}K"
    return f"{n:.2f}"


def pct(n):
    if n is None:
        return "—"
    try:
        return f"{float(n)*100:.2f}%"
    except:
        return "—"


app.jinja_env.filters["fmt"] = fmt
app.jinja_env.filters["pct"] = pct


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not username or not email or not password:
            flash("Please complete all fields.", "danger")
            return redirect("/signup")
        if not valid_email(email):
            flash("Enter a valid email address.", "danger")
            return redirect("/signup")
        if User.query.filter_by(email=email).first():
            flash("An account with this email already exists.", "warning")
            return redirect("/signup")
        if User.query.filter_by(username=username).first():
            flash("Username already taken.", "warning")
            return redirect("/signup")

        try:
            user = User(username=username, email=email, password=generate_password_hash(password))
            db.session.add(user)
            db.session.commit()

            defaults = [
                ("Food & Dining", "Restaurants, groceries, coffee"),
                ("Transport", "Fuel, transit, ride-shares"),
                ("Housing", "Rent, utilities, maintenance"),
                ("Entertainment", "Streaming, events, hobbies"),
                ("Healthcare", "Medical, dental, pharmacy"),
                ("Shopping", "Clothing, electronics, misc"),
                ("Other", "Uncategorised expenses"),
            ]
            for name, desc in defaults:
                db.session.add(ExpenseCategory(name=name, description=desc))
            db.session.commit()

            flash("Account created successfully.", "success")
            return redirect("/login")
        except IntegrityError:
            db.session.rollback()
            flash("Account already exists.", "warning")
            return redirect("/signup")

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not email or not password:
            flash("Please complete all fields.", "danger")
            return redirect("/login")

        user = User.query.filter_by(email=email).first()
        if not user:
            flash("No account found with that email.", "warning")
            return redirect("/login")
        if not check_password_hash(user.password, password):
            flash("Incorrect password.", "danger")
            return redirect("/login")

        login_user(user)
        return redirect("/")

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
    net_worth = calculate_net_worth(current_user.id)

    port_ids = [p.id for p in portfolios]
    recent = (Transaction.query
                 .filter(Transaction.portfolio_id.in_(port_ids))
                 .order_by(Transaction.date.desc())
                 .limit(5).all()) if port_ids else []

    now = datetime.utcnow()
    month = now.strftime("%Y-%m")
    budget = Budget.query.filter_by(user_id=current_user.id, period=month).first()
    month_expenses = (db.session.query(db.func.sum(Expense.amount))
                      .filter_by(user_id=current_user.id)
                      .filter(db.func.strftime("%Y-%m", Expense.date) == month)
                      .scalar() or 0.0)

    goals = Goal.query.filter_by(user_id=current_user.id).all()

    return render_template("dashboard.html",
                           portfolios=portfolios,
                           net_worth=net_worth,
                           recent_tx=recent,
                           budget=budget,
                           month_expenses=month_expenses,
                           goals=goals)


@app.route("/profile")
@login_required
def profile():
    net_worth = calculate_net_worth(current_user.id)
    portfolios = Portfolio.query.filter_by(user_id=current_user.id).all()
    goals = Goal.query.filter_by(user_id=current_user.id).all()
    return render_template("profile.html", net_worth=net_worth, portfolios=portfolios, goals=goals)


@app.route("/portfolio", methods=["POST"])
@login_required
def preview_portfolio():
    symbol = request.form.get("ticker", "").upper()
    balance = float(request.form.get("balance", 0))
    show = request.form.get("show", "true") != "false"

    stock = yf.Ticker(symbol)
    info = stock.info
    score = stock_scoring(symbol)
    financials = stock.quarterly_financials
    cur = info.get("financialCurrency") or "USD"

    try:
        c = CurrencyRates()
        rate = c.get_rate(cur, "USD")
    except:
        rate = 1.0

    def get(df, key):
        try:
            return float(df.loc[key].iloc[0])
        except:
            return None

    def to_usd(val):
        if val is None:
            return None
        try:
            return float(val) * rate
        except:
            return None

    rev_q = get(financials, "Total Revenue")
    ni_q = get(financials, "Net Income")

    fundamentals = {
        "name": info.get("longName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "marketCap": info.get("marketCap"),
        "peRatio": info.get("trailingPE"),
        "forwardPE": info.get("forwardPE"),
        "pegRatio": info.get("pegRatio"),
        "eps": info.get("trailingEps"),
        "revenue_q": to_usd(rev_q),
        "netIncome_q": to_usd(ni_q),
        "revenueGrowth": info.get("revenueGrowth"),
        "earningsGrowth": info.get("earningsGrowth"),
        "profitMargins": info.get("profitMargins"),
        "operatingMargins": info.get("operatingMargins"),
        "roe": info.get("returnOnEquity"),
        "roa": info.get("returnOnAssets"),
        "debt": to_usd(info.get("totalDebt")),
        "cash": to_usd(info.get("totalCash")),
        "currentRatio": info.get("currentRatio"),
        "quickRatio": info.get("quickRatio"),
        "dividendYield": info.get("dividendYield"),
    }

    return render_template("preview.html",
                           ticker=symbol,
                           balance=balance,
                           score=score,
                           fundamentals=fundamentals,
                           show=show)


@app.route("/confirm_portfolio", methods=["POST"])
@login_required
def confirm_portfolio():
    symbol = request.form.get("ticker", "").upper()
    balance = float(request.form.get("balance", 0))
    p_type = request.form.get("type", "equity")

    asset = Asset.query.filter_by(symbol=symbol).first()
    if not asset:
        try:
            name = yf.Ticker(symbol).info.get("longName", symbol)
        except:
            name = symbol
        asset = Asset(user_id=current_user.id, name=name, symbol=symbol)
        db.session.add(asset)
        db.session.flush()

    p = Portfolio(user_id=current_user.id, name=f"{symbol} Portfolio", type=p_type, balance=balance)
    db.session.add(p)
    db.session.commit()
    return redirect("/")


@app.route("/update_balance/<int:pid>", methods=["POST"])
@login_required
def update_balance(pid):
    p = Portfolio.query.get_or_404(pid)
    if p.user_id != current_user.id:
        return redirect("/")
    amount = float(request.form.get("amount", 0))
    action = request.form.get("action")
    if action == "add":
        p.balance += amount
    elif action == "withdraw" and p.balance >= amount:
        p.balance -= amount
    db.session.commit()
    return redirect(url_for("profile"))


@app.route("/trade/<symbol>")
@login_required
def trade(symbol):
    p = Portfolio.query.filter_by(user_id=current_user.id, name=f"{symbol} Portfolio").first_or_404()
    asset = Asset.query.filter_by(symbol=symbol).first()

    db_holdings = Holding.query.filter_by(portfolio_id=p.id).all() if asset else []
    inventory = [(h.buy_price, h.qty) for h in db_holdings]

    df = fetch_data(symbol)
    env = TradingEnv(df, p.balance, inventory)

    model_path = get_model_path(symbol)
    model = tf.keras.models.load_model(model_path)
    state = np.expand_dims(env.get_state(), 0)
    logits = model(state).numpy()[0]
    action = int(np.argmax(logits))
    pred = logits_to_probs(logits)

    _save_prediction(symbol, model_path, action, logits)

    labels = df["Date"].dt.strftime("%Y-%m-%d").tolist()
    prices = df["Close"].tolist()
    units = sum(q for _, q in inventory)

    pred_history = []
    if asset:
        pred_history = (Prediction.query
                        .filter_by(asset_id=asset.id)
                        .order_by(Prediction.timestamp.desc())
                        .limit(5).all())

    return render_template("trade.html",
                           ticker=symbol,
                           action=["HOLD", "BUY", "SELL"][action],
                           probs={"buy": round(float(pred[1]) * 100, 1),
                                  "sell": round(float(pred[2]) * 100, 1),
                                  "hold": round(float(pred[0]) * 100, 1)},
                           balance=p.balance,
                           units=units,
                           labels=labels,
                           prices=prices,
                           pred_history=pred_history)


@app.route("/execute_trade/<symbol>", methods=["POST"])
@login_required
def execute_trade(symbol):
    asset = Asset.query.filter_by(symbol=symbol).first()
    if not asset:
        flash("Asset not found.", "danger")
        return redirect("/")

    p = Portfolio.query.filter_by(user_id=current_user.id, name=f"{symbol} Portfolio").first_or_404()
    db_holdings = Holding.query.filter_by(portfolio_id=p.id).all()
    inventory = [(h.buy_price, h.qty) for h in db_holdings]

    df = fetch_data(symbol)
    env = TradingEnv(df, p.balance, inventory)

    model = tf.keras.models.load_model(get_model_path(symbol))
    state = np.expand_dims(env.get_state(), 0)
    logits = model(state).numpy()[0]
    action = int(np.argmax(logits))

    env.apply_action(action)

    Holding.query.filter_by(portfolio_id=p.id).delete()
    for bp, qty in env.inventory:
        db.session.add(Holding(portfolio_id=p.id, asset_id=asset.id, buy_price=bp, qty=qty))

    price = float(df.iloc[-1]["Close"])
    if action in (1, 2):
        tx_type = "buy" if action == 1 else "sell"
        db.session.add(Transaction(portfolio_id=p.id, asset_id=asset.id, type=tx_type, amount=price))

    p.balance = env.balance
    db.session.commit()
    return redirect(url_for("trade", symbol=symbol))


@app.route("/sell_all/<symbol>", methods=["POST"])
@login_required
def sell_all(symbol):
    p = Portfolio.query.filter_by(user_id=current_user.id, name=f"{symbol} Portfolio").first_or_404()
    asset = Asset.query.filter_by(symbol=symbol).first()
    db_holdings = Holding.query.filter_by(portfolio_id=p.id).all()
    price = safe_price(symbol)
    if price:
        total_units = sum(h.qty for h in db_holdings)
        p.balance += total_units * price
    Holding.query.filter_by(portfolio_id=p.id).delete()
    if asset:
        db.session.add(Transaction(portfolio_id=p.id, asset_id=asset.id, type="sell", amount=price or 0))
    db.session.delete(p)
    db.session.commit()
    return redirect("/")


def evaluate(symbol, actor, start, end, balance):
    df = fetch_data_range(symbol, start, end)
    env = TradingEnv(df, balance, [])
    signals = []

    for i in range(len(df)):
        env.step = i
        state = env.get_state().reshape(1, -1)
        logits = actor(state, training=False).numpy()[0]
        action = int(np.argmax(logits))
        price = float(df.iloc[i]["Close"])

        if action == 1 and env.balance < price:
            action = 0
        elif action == 2 and not env.inventory:
            action = 0

        env.apply_action(action)
        signals.append(action)

    df["Signal"] = signals
    final_balance = env.balance + sum(q * df.iloc[-1]["Close"] for _, q in env.inventory)
    profit = final_balance - balance
    return df, profit, final_balance


@app.route("/backtest", methods=["GET", "POST"])
@login_required
def backtest():
    if request.method == "POST":
        symbol = request.form.get("ticker", "").upper()
        start = request.form.get("start")
        end = request.form.get("end")
        balance = float(request.form.get("balance"))

        try:
            actor = tf.keras.models.load_model(get_model_path(symbol))
            df, profit, final_bal = evaluate(symbol, actor, start, end, balance)

            labels = df["Date"].dt.strftime("%Y-%m-%d").tolist()
            prices = df["Close"].tolist()
            signals = df["Signal"].tolist()
            buy_points = [prices[i] if signals[i] == 1 else None for i in range(len(signals))]
            sell_points = [prices[i] if signals[i] == 2 else None for i in range(len(signals))]

            results = {
                "profit": round(profit, 2),
                "balance": round(final_bal, 2),
                "buys": signals.count(1),
                "sells": signals.count(2),
                "holds": signals.count(0),
                "labels": labels,
                "prices": prices,
                "buy_points": buy_points,
                "sell_points": sell_points,
            }
        except Exception as e:
            results = None

        return render_template("backtest.html", results=results)

    return render_template("backtest.html", results=None)


@app.route("/expenses", methods=["GET", "POST"])
@login_required
def expenses():
    if request.method == "POST":
        cat_id = request.form.get("category_id")
        amount = float(request.form.get("amount", 0))
        date_s = request.form.get("date")
        notes = request.form.get("notes", "")
        date = datetime.strptime(date_s, "%Y-%m-%d").date() if date_s else datetime.utcnow().date()

        db.session.add(Expense(user_id=current_user.id, category_id=cat_id or None,
                               amount=amount, date=date, notes=notes))
        db.session.commit()
        flash("Expense recorded.", "success")
        return redirect("/expenses")

    categories = ExpenseCategory.query.all()
    expenses = (Expense.query.filter_by(user_id=current_user.id)
                .order_by(Expense.date.desc()).limit(50).all())

    month = datetime.utcnow().strftime("%Y-%m")
    budget = Budget.query.filter_by(user_id=current_user.id, period=month).first()
    spent = (db.session.query(db.func.sum(Expense.amount))
             .filter_by(user_id=current_user.id)
             .filter(db.func.strftime("%Y-%m", Expense.date) == month)
             .scalar() or 0.0)

    return render_template("expenses.html",
                           categories=categories,
                           expenses=expenses,
                           budget=budget,
                           spent=spent,
                           month=month)


@app.route("/expenses/delete/<int:eid>", methods=["POST"])
@login_required
def delete_expense(eid):
    e = Expense.query.get_or_404(eid)
    if e.user_id == current_user.id:
        db.session.delete(e)
        db.session.commit()
    return redirect("/expenses")


@app.route("/budget", methods=["GET", "POST"])
@login_required
def budget():
    if request.method == "POST":
        period = request.form.get("period")
        limit = float(request.form.get("limit_amount", 0))
        existing = Budget.query.filter_by(user_id=current_user.id, period=period).first()
        if existing:
            existing.limit_amount = limit
        else:
            db.session.add(Budget(user_id=current_user.id, period=period, limit_amount=limit))
        db.session.commit()
        flash("Budget saved.", "success")
        return redirect("/budget")

    budgets = Budget.query.filter_by(user_id=current_user.id).order_by(Budget.period.desc()).all()
    budget_data = []
    for b in budgets:
        spent = (db.session.query(db.func.sum(Expense.amount))
                 .filter_by(user_id=current_user.id)
                 .filter(db.func.strftime("%Y-%m", Expense.date) == b.period)
                 .scalar() or 0.0)
        budget_data.append({"budget": b, "spent": round(spent, 2)})

    return render_template("budget.html", budget_data=budget_data)


@app.route("/goals", methods=["GET", "POST"])
@login_required
def goals():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        amount = float(request.form.get("amount", 0))
        if name and amount > 0:
            db.session.add(Goal(user_id=current_user.id, name=name, amount=amount))
            db.session.commit()
            flash("Goal added.", "success")
        return redirect("/goals")

    all_goals = Goal.query.filter_by(user_id=current_user.id).all()
    net_worth = calculate_net_worth(current_user.id)
    return render_template("goals.html", goals=all_goals, net_worth=net_worth)


@app.route("/goals/delete/<int:gid>", methods=["POST"])
@login_required
def delete_goal(gid):
    g = Goal.query.get_or_404(gid)
    if g.user_id == current_user.id:
        db.session.delete(g)
        db.session.commit()
    return redirect("/goals")


@app.route("/assets")
@login_required
def assets():
    all_assets = Asset.query.filter_by(user_id=current_user.id).all()
    asset_data = []
    for a in all_assets:
        price = safe_price(a.symbol)
        pred = (Prediction.query.filter_by(asset_id=a.id)
                .order_by(Prediction.timestamp.desc()).first())
        if pred:
            logits = np.array([pred.logit_hold, pred.logit_buy, pred.logit_sell])
            s = logits_to_probs(logits)
            pred.pct_hold = round(float(s[0]) * 100, 1)
            pred.pct_buy = round(float(s[1]) * 100, 1)
            pred.pct_sell = round(float(s[2]) * 100, 1)
        asset_data.append({"asset": a, "price": price, "latest_pred": pred})
    return render_template("assets.html", asset_data=asset_data)


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)