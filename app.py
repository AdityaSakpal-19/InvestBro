"""
app.py — InvestBro Streamlit application

Pages (controlled via st.session_state["page"]):
  "home"  — sidebar inputs, watchlist
  "stock" — full analysis for a specific ticker
"""

import os
import json
import math
import datetime
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import google.genai as genai

from model import get_or_train_model, predict_next_price

# ─────────────────────────────────────────────
# Storage layout
# ─────────────────────────────────────────────
STORAGE_DIR = "storage"
DATA_DIR    = os.path.join(STORAGE_DIR, "data")

os.makedirs(DATA_DIR, exist_ok=True)
# storage/models/ is created by model.py


def _user_dir(username: str) -> str:
    """Return (and create) the per-user storage directory."""
    path = os.path.join(STORAGE_DIR, "users", username)
    os.makedirs(path, exist_ok=True)
    return path


def _watchlist_file(username: str) -> str:
    return os.path.join(_user_dir(username), "watchlist.json")


def _holdings_file(username: str) -> str:
    return os.path.join(_user_dir(username), "holdings.json")


def _current_user() -> str:
    """Return the active username from session state (guaranteed to exist after login)."""
    return st.session_state.get("username", "")

# ─────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(page_title="InvestBro", page_icon="📈", layout="wide")

st.markdown("""
<style>
    .main { background-color: #ffffff; }
    .block-container { padding-top: 2rem; }
    h1, h2, h3 { color: #1f2328; }
    .stMetric label { font-size: 0.82rem; color: #57606a; }

    [data-testid="stSidebar"] { background-color: #000000 !important; }
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] *,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] strong,
    [data-testid="stSidebar"] em,
    [data-testid="stSidebar"] small,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] select,
    [data-testid="stSidebar"] textarea {
        color: #ffffff !important;
        background-color: #1a1a1a !important;
        border-color: #444444 !important;
    }
    [data-testid="stSidebar"] hr     { border-color: #333333 !important; }
    [data-testid="stSidebar"] button {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        border-color: #444444 !important;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Session-state defaults
# ─────────────────────────────────────────────
if "username" not in st.session_state:
    st.session_state["username"] = ""
if "display_name" not in st.session_state:
    st.session_state["display_name"] = ""
if "page" not in st.session_state:
    st.session_state["page"] = "home"
if "stock_params" not in st.session_state:
    st.session_state["stock_params"] = {}


# ─────────────────────────────────────────────
# Watchlist  —  storage/users/<user>/watchlist.json
# ─────────────────────────────────────────────
def load_watchlist() -> list:
    path = _watchlist_file(_current_user())
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return []


def save_watchlist(wl: list):
    with open(_watchlist_file(_current_user()), "w") as f:
        json.dump(wl, f, indent=2)


def add_to_watchlist(ticker: str):
    wl = load_watchlist()
    if ticker not in wl:
        wl.append(ticker)
        save_watchlist(wl)


def remove_from_watchlist(ticker: str):
    save_watchlist([t for t in load_watchlist() if t != ticker])


# ─────────────────────────────────────────────
# Holdings  —  storage/users/<user>/holdings.json
#
# Schema per ticker:
#   { "units": float, "avg_price": float, "transactions": [...] }
#
# Avg price uses weighted average:
#   new_avg = (old_units * old_avg + added_units * price) / (old_units + added_units)
#
# On sell: avg_price stays the same (only units reduce).
# ─────────────────────────────────────────────
def load_holdings() -> dict:
    """Return the full holdings dict keyed by ticker."""
    path = _holdings_file(_current_user())
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def save_holdings(holdings: dict):
    with open(_holdings_file(_current_user()), "w") as f:
        json.dump(holdings, f, indent=2)


def get_holding(ticker: str) -> dict:
    """Return holding for one ticker or a blank skeleton."""
    return load_holdings().get(ticker, {"units": 0.0, "avg_price": 0.0, "transactions": []})


def buy_units(ticker: str, units: float, price: float):
    """
    Buy `units` at `price`.
    Updates avg_price via weighted average. Records the transaction.
    If this is the first buy, avg_price = price (as requested).
    """
    if units <= 0:
        return
    holdings = load_holdings()
    h = holdings.get(ticker, {"units": 0.0, "avg_price": 0.0, "transactions": []})

    old_units = h["units"]
    old_avg   = h["avg_price"]

    if old_units == 0:
        new_avg = price                              # first buy: avg = current price
    else:
        new_avg = (old_units * old_avg + units * price) / (old_units + units)

    h["units"]     = round(old_units + units, 4)
    h["avg_price"] = round(new_avg, 4)
    h["transactions"].append({
        "date":   datetime.date.today().isoformat(),
        "type":   "BUY",
        "units":  round(units, 4),
        "price":  round(price, 4),
    })

    holdings[ticker] = h
    save_holdings(holdings)


def sell_units(ticker: str, units: float, price: float):
    """
    Sell `units` at `price`.
    Avg_price is unchanged on sell. Records the transaction.
    """
    holdings = load_holdings()
    h = holdings.get(ticker, {"units": 0.0, "avg_price": 0.0, "transactions": []})

    units = min(units, h["units"])   # can't sell more than owned
    if units <= 0:
        return

    h["units"] = round(h["units"] - units, 4)
    h["transactions"].append({
        "date":   datetime.date.today().isoformat(),
        "type":   "SELL",
        "units":  round(units, 4),
        "price":  round(price, 4),
    })

    holdings[ticker] = h
    save_holdings(holdings)


# ─────────────────────────────────────────────
# Daily data  —  storage/data/<TICKER>.csv
# ─────────────────────────────────────────────
def _data_path(ticker: str) -> str:
    return os.path.join(DATA_DIR, f"{ticker.replace('.', '_')}.csv")


def _data_meta_path(ticker: str) -> str:
    return os.path.join(DATA_DIR, f"{ticker.replace('.', '_')}_meta.json")


def _read_data_meta(ticker: str) -> dict:
    path = _data_meta_path(ticker)
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _write_data_meta(ticker: str, meta: dict):
    with open(_data_meta_path(ticker), "w") as f:
        json.dump(meta, f, indent=2)


def load_stored_data(ticker: str) -> pd.DataFrame | None:
    path = _data_path(ticker)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df if not df.empty else None


def update_daily_data(ticker: str):
    """
    Keep storage/data/<ticker>.csv up to date. Called whenever a watched
    ticker is opened. Behaviour:

      • No CSV yet  → download full 5-year history, save, record today's date.
      • CSV exists, already updated today  → do nothing (idempotent).
      • CSV exists, not yet updated today  → fetch only the missing days
        (from last stored date + 1 day), append, record today's date.

    Never writes duplicate rows.
    """
    today     = datetime.date.today().isoformat()
    data_meta = _read_data_meta(ticker)
    path      = _data_path(ticker)

    # ── First time: seed with 5 years of history ──────────────────────
    if not os.path.exists(path):
        df = yf.download(ticker, period="5y", interval="1d",
                         progress=False, auto_adjust=True)
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.to_csv(path)
            _write_data_meta(ticker, {"last_updated": today})
        return

    # ── Already updated today: skip ───────────────────────────────────
    if data_meta.get("last_updated") == today:
        return

    # ── Append only new trading days ──────────────────────────────────
    stored = pd.read_csv(path, index_col=0, parse_dates=True)
    start  = (stored.index.max() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    new    = yf.download(ticker, start=start, interval="1d",
                         progress=False, auto_adjust=True)
    if new is not None and not new.empty:
        if isinstance(new.columns, pd.MultiIndex):
            new.columns = new.columns.get_level_values(0)
        new = new[~new.index.isin(stored.index)]
        if not new.empty:
            pd.concat([stored, new]).sort_index().to_csv(path)

    # Record that we updated today regardless of whether new rows arrived
    # (market may be closed — that is not an error)
    _write_data_meta(ticker, {"last_updated": today})


def get_training_data(ticker: str, df_live: pd.DataFrame) -> pd.DataFrame:
    """
    For watched stocks: return the stored CSV (richest dataset).
    For non-watched stocks: return the live 5-year pull.
    Stored data always wins on row count; live data wins on any overlap.
    """
    stored = load_stored_data(ticker)
    if stored is None:
        return df_live
    extra = stored[~stored.index.isin(df_live.index)]
    if extra.empty:
        return df_live
    return pd.concat([df_live, extra]).sort_index()


# ─────────────────────────────────────────────
# Stock data helpers
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def fetch_stock_data(ticker: str, period: str = "5y", interval: str = "1d"):
    df = yf.download(ticker, period=period, interval=interval,
                     progress=False, auto_adjust=True)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


@st.cache_data(show_spinner=False)
def fetch_stock_info(ticker: str) -> dict:
    try:
        return yf.Ticker(ticker).info
    except Exception:
        return {}


def get_current_price(info: dict, df: pd.DataFrame) -> float | None:
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    if price:
        return float(price)
    if df is not None and not df.empty:
        return float(df["Close"].iloc[-1])
    return None


def get_ratio(info: dict, key: str):
    val = info.get(key)
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "N/A"
    return round(val, 4) if isinstance(val, float) else val


# ─────────────────────────────────────────────
# Investment calculations
# ─────────────────────────────────────────────
def calculate_investment(amount: float, current_price: float, predicted_price: float) -> dict:
    num_shares = math.floor(amount / current_price)
    remaining  = amount - num_shares * current_price
    ret_pct    = ((predicted_price - current_price) / current_price) * 100
    ret_amt    = num_shares * (predicted_price - current_price)
    return {
        "num_shares":              num_shares,
        "remaining":               round(remaining, 2),
        "predicted_return_pct":    round(ret_pct, 2),
        "predicted_return_amount": round(ret_amt, 2),
    }


# ─────────────────────────────────────────────
# Gemini AI
# ─────────────────────────────────────────────
def get_gemini_client():
    api_key = None
    # Try secrets.toml first; catch any Streamlit/IO exception (not just KeyError)
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    # Fall back to environment variable
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


def gemini_investment_analysis(client, ticker, current_price, predicted_price,
                                growth_pct, risk, tenure, goal, ratios,
                                units_held, avg_price):
    """
    Returns a BUY / HOLD / SELL recommendation with plain-language reasoning.
    """
    unrealised_pct = ((current_price - avg_price) / avg_price * 100) if avg_price > 0 else 0
    holding_value  = units_held * current_price

    prompt = f"""
You are an expert equity investment advisor helping a retail investor decide whether to BUY, HOLD, or SELL a stock.

Stock & Model Data:
- Ticker: {ticker}
- Current Market Price: ₹{current_price:.2f}
- XGBoost Next-Period Predicted Price: ₹{predicted_price:.2f}  ({growth_pct:+.2f}%)

Investor's Current Position:
- Units Held: {int(units_held)}
- Average Buy Price: ₹{avg_price:.2f}
- Current Holding Value: ₹{holding_value:,.2f}
- Unrealised P&L: {unrealised_pct:+.2f}%

Investor Profile:
- Risk Level: {risk}
- Investment Goal: {goal}
- Tenure: {tenure}

Fundamental Ratios (Yahoo Finance):
- P/E: {ratios.get('P/E', 'N/A')}  · D/E: {ratios.get('D/E', 'N/A')}
- Dividend Yield: {ratios.get('Dividend Yield', 'N/A')}  · EPS: {ratios.get('EPS', 'N/A')}

Instructions:
1. Start with a clear ONE-WORD recommendation on its own line: **BUY**, **HOLD**, or **SELL**.
2. In 2–3 sentences explain WHY, referencing the predicted price, unrealised P&L, and fundamentals.
3. List 2–3 key reasons supporting the recommendation.
4. List 2–3 risks the investor should watch.
5. One sentence on what would change your recommendation (e.g. if price drops below X).

Rules:
- Be direct and concise. No fluff.
- Do NOT present predictions as guaranteed returns.
- Keep it suitable for someone with no finance background.
"""
    try:
        return client.models.generate_content(model="gemini-3.6-flash", contents=prompt).text
    except Exception as e:
        return f"Gemini analysis unavailable: {e}"


# ─────────────────────────────────────────────
# Model meta helpers (for display)
# ─────────────────────────────────────────────
def get_model_meta(ticker: str) -> dict:
    """Return stored meta for a ticker's model, or empty dict."""
    safe = ticker.replace(".", "_")
    meta_path = os.path.join(STORAGE_DIR, "models", f"{safe}_meta.json")
    if not os.path.exists(meta_path):
        return {}
    try:
        with open(meta_path) as f:
            return json.load(f)
    except Exception:
        return {}


# ─────────────────────────────────────────────
# ██  HOME PAGE
# ─────────────────────────────────────────────
def show_home():
    # ── Sidebar ───────────────────────────────
    with st.sidebar:
        st.title("📈 InvestBro")
        st.markdown("*Your AI-powered investment companion*")
        st.markdown("---")

        # ── User name display + switch ─────────
        display_name = st.session_state.get("display_name") or st.session_state.get("username", "")
        st.markdown(
            f"<div style='background:#1a1a1a;border:1px solid #333;border-radius:6px;"
            f"padding:0.5rem 0.75rem;margin-bottom:0.5rem;'>"
            f"<span style='font-size:0.78rem;color:#aaaaaa;'>Signed in as</span><br>"
            f"<strong style='font-size:1rem;color:#ffffff;'>👤 {display_name}</strong>"
            f"</div>",
            unsafe_allow_html=True,
        )
        if st.button("🔄 Switch User", use_container_width=True):
            st.session_state["username"]     = ""
            st.session_state["display_name"] = ""
            st.session_state["page"]         = "home"
            st.rerun()

        st.markdown("---")
        st.subheader("📋 Investment Inputs")

        st.markdown("**🔎 Stock Ticker**")
        st.caption("Yahoo Finance format · e.g. RELIANCE.NS")
        ticker = st.text_input(
            "Ticker", value="RELIANCE.NS",
            help="e.g. RELIANCE.NS · TCS.NS · INFY.NS · HDFCBANK.NS",
            label_visibility="collapsed",
        ).strip().upper()

        st.markdown("**💰 Investment Amount (₹)**")
        amount = st.number_input(
            "Amount", min_value=1000.0, value=100000.0,
            step=1000.0, format="%.2f", label_visibility="collapsed",
        )

        st.markdown("**📅 Investment Tenure**")
        tenure = st.selectbox(
            "Tenure", ["1 Year", "3 Years", "5 Years"],
            label_visibility="collapsed",
        )

        st.markdown("**⚠️ Risk Level**")
        risk = st.radio(
            "Risk", ["Low", "Medium", "High"],
            index=1, label_visibility="collapsed",
        )

        st.markdown("**🎯 Investment Goal**")
        goal = st.text_input(
            "Goal", value="Long-term growth",
            placeholder="e.g. Wealth creation, Dividend income …",
            label_visibility="collapsed",
        )

        st.markdown("---")
        analyze = st.button("🔍 Analyze Investment", use_container_width=True)
        if st.button("📊 View Portfolio", use_container_width=True):
            st.session_state["page"] = "portfolio"
            st.rerun()
        st.markdown(
            "<small style='color:#57606a;'>e.g. RELIANCE.NS · TCS.NS · INFY.NS</small>",
            unsafe_allow_html=True,
        )

        # ── Watchlist section ──────────────────
        st.markdown("---")
        st.subheader("⭐ My Watchlist")
        st.caption("Daily data collected. Model retrained every week.")

        watchlist    = load_watchlist()
        in_watchlist = ticker in watchlist

        if in_watchlist:
            if st.button("🗑 Remove from Watchlist", use_container_width=True):
                remove_from_watchlist(ticker)
                st.rerun()
        else:
            if st.button("⭐ Add to Watchlist", use_container_width=True):
                add_to_watchlist(ticker)
                update_daily_data(ticker)   # seed the CSV immediately
                st.rerun()

        if st.button("🔄 Update All Data", use_container_width=True):
            wl = load_watchlist()
            if wl:
                with st.spinner("Updating …"):
                    for t in wl:
                        update_daily_data(t)
                st.success(f"Updated {len(wl)} stock(s).")
            else:
                st.info("Watchlist is empty.")

        # Watchlist entries
        watchlist = load_watchlist()
        if watchlist:
            st.markdown("**Saved stocks:**")
            for t in watchlist:
                stored_df = load_stored_data(t)
                rows      = len(stored_df) if stored_df is not None else 0
                meta      = get_model_meta(t)
                week_tag  = meta.get("trained_week", "—")
                col_a, col_b = st.columns([3, 1])
                col_a.markdown(
                    f"**{t}**  \n"
                    f"<small style='color:#aaaaaa;'>"
                    f"{rows} day{'s' if rows != 1 else ''} stored · "
                    f"model week {week_tag}"
                    f"</small>",
                    unsafe_allow_html=True,
                )
                if col_b.button("📂", key=f"open_{t}", help=f"Analyse {t}"):
                    # Open directly with last-used or default investment params
                    st.session_state["stock_params"] = {
                        "ticker": t,
                        "amount": 100000.0,
                        "tenure": "1 Year",
                        "risk":   "Medium",
                        "goal":   "Long-term growth",
                    }
                    st.session_state["page"] = "stock"
                    st.rerun()
        else:
            st.markdown(
                "<small style='color:#888888;'>No stocks saved yet.</small>",
                unsafe_allow_html=True,
            )

    # ── Main area ──────────────────────────────
    st.title("📈 InvestBro")
    st.markdown("#### Your AI-powered stock investment companion")
    st.markdown("---")

    if not analyze:
        st.info("👈 Fill in the sidebar and click **Analyze Investment** to begin.")
        return

    # Validate
    if not ticker:
        st.error("Please enter a valid stock ticker.")
        return
    if amount <= 0:
        st.error("Investment amount must be greater than zero.")
        return

    # Navigate to stock page
    st.session_state["stock_params"] = {
        "ticker": ticker,
        "amount": amount,
        "tenure": tenure,
        "risk":   risk,
        "goal":   goal,
    }
    st.session_state["page"] = "stock"
    st.rerun()


# ─────────────────────────────────────────────
# ██  STOCK ANALYSIS PAGE
# ─────────────────────────────────────────────
def show_stock():
    params  = st.session_state.get("stock_params", {})
    ticker  = params.get("ticker", "")
    tenure  = params.get("tenure",  "1 Year")
    risk    = params.get("risk",    "Medium")
    goal    = params.get("goal",    "Long-term growth")

    if not ticker:
        st.error("No ticker selected. Please go back and try again.")
        if st.button("← Back to Home"):
            st.session_state["page"] = "home"
            st.rerun()
        return

    # ── Top bar: title + watchlist + back ──────
    watchlist    = load_watchlist()
    in_watchlist = ticker in watchlist

    top_left, top_right = st.columns([6, 2])
    with top_left:
        st.title(f"📈 {ticker}")
        if st.button("← Back to Home"):
            st.session_state["page"] = "home"
            st.rerun()
    with top_right:
        st.markdown("<div style='padding-top:1.2rem;'>", unsafe_allow_html=True)
        if in_watchlist:
            if st.button("🗑 Remove from Watchlist", use_container_width=True):
                remove_from_watchlist(ticker)
                st.rerun()
        else:
            if st.button("⭐ Add to Watchlist", use_container_width=True):
                add_to_watchlist(ticker)
                update_daily_data(ticker)
                st.rerun()
        if st.button("📊 Portfolio", use_container_width=True):
            st.session_state["page"] = "portfolio"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # ── Fetch live data ─────────────────────────
    with st.spinner(f"Fetching data for **{ticker}** …"):
        df_live = fetch_stock_data(ticker, period="5y", interval="1d")
        info    = fetch_stock_info(ticker)

    if df_live is None or df_live.empty:
        st.error(
            f"Could not retrieve data for **{ticker}**. "
            "Check the ticker symbol (e.g. RELIANCE.NS) and try again."
        )
        return

    current_price = get_current_price(info, df_live)
    if current_price is None or current_price <= 0:
        st.error("Could not determine current price. Please try a different ticker.")
        return

    # ── Daily data update (watched stocks only) ──
    is_watched = ticker in load_watchlist()
    if is_watched:
        update_daily_data(ticker)

    df_train = get_training_data(ticker, df_live)

    # ── Model (weekly gate for watched; fresh otherwise) ─
    with st.spinner("Loading / training XGBoost model …"):
        try:
            xgb_model, metrics, was_retrained = get_or_train_model(df_train, ticker)
            predicted_price = predict_next_price(xgb_model, df_train)
        except ValueError as ve:
            st.error(str(ve))
            return
        except Exception:
            st.error("Model training failed. The ticker may have insufficient historical data.")
            return

    growth_pct = ((predicted_price - current_price) / current_price) * 100

    # ── Model timestamp banner ───────────────────
    meta = get_model_meta(ticker)
    if meta:
        trained_date = meta.get("trained_date", "unknown")
        trained_week = meta.get("trained_week", "—")
        data_rows    = len(df_train)
        status_icon  = "🔄 Retrained now" if was_retrained else "✅ Loaded from cache"
        st.info(
            f"**Model status:** {status_icon} · "
            f"Week {trained_week} · Last trained {trained_date} · "
            f"Trained on {data_rows} trading days"
        )

    # ────────────────────────────────────────────────────────────
    # SECTION 1 — Holdings Tracker  (watchlist stocks only)
    # ────────────────────────────────────────────────────────────
    holding = get_holding(ticker)
    units_held = holding["units"]
    avg_price  = holding["avg_price"] if holding["avg_price"] > 0 else current_price

    if is_watched:
        st.subheader("📦 My Position")
        st.caption("Track your actual equity. Avg Price updates on every buy via weighted average.")

        # ── Current position metrics ─────────────
        holding_value    = units_held * current_price
        predicted_value  = units_held * predicted_price
        unrealised_amt   = units_held * (current_price - avg_price)
        unrealised_pct   = ((current_price - avg_price) / avg_price * 100) if avg_price > 0 else 0
        pred_return_amt  = units_held * (predicted_price - avg_price)
        pred_return_pct  = ((predicted_price - avg_price) / avg_price * 100) if avg_price > 0 else 0

        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Units Held",       f"{int(units_held):,}")
        h2.metric("Avg Buy Price (₹)", f"₹{avg_price:,.2f}")
        h3.metric("Current Value (₹)", f"₹{holding_value:,.2f}",
                  delta=f"{unrealised_pct:+.2f}%")
        h4.metric("Unrealised P&L (₹)", f"₹{unrealised_amt:,.2f}",
                  delta=f"₹{unrealised_amt:+,.2f}")

        h5, h6 = st.columns(2)
        h5.metric("Predicted Value (₹) *", f"₹{predicted_value:,.2f}",
                  delta=f"{pred_return_pct:+.2f}%")
        h6.metric("Predicted Return (₹) *", f"₹{pred_return_amt:,.2f}",
                  delta=f"₹{pred_return_amt:+,.2f}")

        st.markdown("---")

        # ── Buy / Sell panels ────────────────────
        st.markdown("**🔁 Trade Units**")
        buy_col, sell_col = st.columns(2)

        with buy_col:
            st.markdown("**🟢 Buy**")
            buy_tab1, buy_tab2 = st.tabs(["By Units", "By Amount"])

            with buy_tab1:
                buy_units_val = st.number_input(
                    "Units to buy", min_value=1, value=1, step=1,
                    key="buy_units_input",
                )
                cost = buy_units_val * current_price
                st.caption(f"Cost at current price: ₹{cost:,.2f}")
                if st.button("✅ Confirm Buy (Units)", use_container_width=True, key="confirm_buy_units"):
                    buy_units(ticker, float(buy_units_val), current_price)
                    st.success(f"Bought {buy_units_val} units @ ₹{current_price:,.2f}")
                    st.rerun()

            with buy_tab2:
                buy_amount_val = st.number_input(
                    "Amount to invest (₹)", min_value=1.0, value=10000.0, step=1000.0,
                    key="buy_amount_input",
                )
                units_from_amount = int(buy_amount_val / current_price)
                st.caption(f"≈ {units_from_amount} units at current price")
                if st.button("✅ Confirm Buy (Amount)", use_container_width=True, key="confirm_buy_amount"):
                    buy_units(ticker, float(units_from_amount), current_price)
                    st.success(f"Bought {units_from_amount} units @ ₹{current_price:,.2f} for ₹{units_from_amount * current_price:,.2f}")
                    st.rerun()

        with sell_col:
            st.markdown("**🔴 Sell**")
            if units_held > 0:
                sell_tab1, sell_tab2 = st.tabs(["By Units", "By Amount"])

                with sell_tab1:
                    sell_units_val = st.number_input(
                        "Units to sell", min_value=1, max_value=int(units_held),
                        value=min(1, int(units_held)), step=1,
                        key="sell_units_input",
                    )
                    proceeds = sell_units_val * current_price
                    realised = sell_units_val * (current_price - avg_price)
                    st.caption(f"Proceeds: ₹{proceeds:,.2f} · Realised P&L: ₹{realised:+,.2f}")
                    if st.button("✅ Confirm Sell (Units)", use_container_width=True, key="confirm_sell_units"):
                        sell_units(ticker, float(sell_units_val), current_price)
                        st.success(f"Sold {sell_units_val} units @ ₹{current_price:,.2f}")
                        st.rerun()

                with sell_tab2:
                    sell_amount_val = st.number_input(
                        "Amount to sell (₹)", min_value=1.0,
                        max_value=float(units_held * current_price),
                        value=min(10000.0, float(units_held * current_price)),
                        step=1000.0, key="sell_amount_input",
                    )
                    units_to_sell = min(int(sell_amount_val / current_price), int(units_held))
                    realised_amt  = units_to_sell * (current_price - avg_price)
                    st.caption(f"≈ {units_to_sell} units · Realised P&L: ₹{realised_amt:+,.2f}")
                    if st.button("✅ Confirm Sell (Amount)", use_container_width=True, key="confirm_sell_amount"):
                        sell_units(ticker, float(units_to_sell), current_price)
                        st.success(f"Sold {units_to_sell} units @ ₹{current_price:,.2f}")
                        st.rerun()
            else:
                st.info("No units held. Buy first to enable selling.")

        # ── Transaction history ──────────────────
        if holding["transactions"]:
            with st.expander("📋 Transaction History", expanded=False):
                tx_df = pd.DataFrame(holding["transactions"])
                tx_df.columns = [c.capitalize() for c in tx_df.columns]
                st.dataframe(tx_df, use_container_width=True, hide_index=True)

        st.markdown("---")

    # ────────────────────────────────────────────────────────────
    # SECTION 2 — Investment Summary  (price prediction)
    # ────────────────────────────────────────────────────────────
    st.subheader("💼 Investment Summary")
    st.caption("⚠️ All predicted values are model estimates, not guaranteed returns.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Current Price (₹)",     f"₹{current_price:,.2f}")
    c2.metric("Predicted Price (₹) *", f"₹{predicted_price:,.2f}", delta=f"{growth_pct:+.2f}%")
    c3.metric("Predicted Growth %",    f"{growth_pct:+.2f}%")
    st.markdown("---")

    # ── SECTION 3 — Fundamental Ratios ───────────
    st.subheader("📊 Stock Analysis")

    ratios = {
        "P/E":            get_ratio(info, "trailingPE"),
        "D/E":            get_ratio(info, "debtToEquity"),
        "Dividend Yield": get_ratio(info, "dividendYield"),
        "EPS":            get_ratio(info, "trailingEps"),
    }

    st.markdown("**Fundamental Ratios** *(Yahoo Finance)*")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("P/E Ratio",      str(ratios["P/E"]))
    r2.metric("Debt-to-Equity", str(ratios["D/E"]))
    r3.metric("Dividend Yield", str(ratios["Dividend Yield"]))
    r4.metric("EPS",            str(ratios["EPS"]))
    st.markdown("---")

    # ── SECTION 4 — Model Performance ────────────
    st.markdown("**XGBoost Model Performance** *(held-out test set)*")
    retrain_note = "just retrained 🔄" if was_retrained else "cached this week ✅"
    st.caption(f"Model {retrain_note} · Metrics reflect historical fit, not future accuracy.")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE",  f"₹{metrics['MAE']:,.2f}")
    m2.metric("RMSE", f"₹{metrics['RMSE']:,.2f}")
    m3.metric("R²",   f"{metrics['R2']:.2f}")
    st.markdown("---")

    # ── SECTION 5 — AI Recommendation ────────────
    st.subheader("🤖 AI Recommendation")
    gemini = get_gemini_client()
    if gemini is None:
        st.warning(
            "Gemini API key not found. "
            "Add **GEMINI_API_KEY** to `.streamlit/secrets.toml` or as an env variable."
        )
    else:
        with st.spinner("Generating recommendation …"):
            analysis = gemini_investment_analysis(
                gemini, ticker, current_price, predicted_price,
                growth_pct, risk, tenure, goal, ratios,
                units_held, avg_price,
            )
        st.markdown(analysis)

    st.markdown("---")

    # ── Disclaimer ───────────────────────────────
    st.markdown(
        """
        <div style="background:#f7f8fa;border:1px solid #e5e7eb;
                    border-radius:6px;padding:1rem;color:#57606a;font-size:0.82rem;">
        <strong>Disclaimer:</strong> InvestBro provides educational analysis and model-based estimates.
        It is <strong>not financial advice</strong>. Stock prices are uncertain and past performance
        does not guarantee future returns. Always consult a qualified financial advisor.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# ██  LOGIN PAGE
# ─────────────────────────────────────────────
def _get_saved_users() -> list[str]:
    """Return folder names under storage/users/ that have at least a watchlist or holdings file."""
    users_dir = os.path.join(STORAGE_DIR, "users")
    if not os.path.isdir(users_dir):
        return []
    result = []
    for entry in sorted(os.scandir(users_dir), key=lambda e: e.name.lower()):
        if entry.is_dir():
            has_data = (
                os.path.exists(os.path.join(entry.path, "watchlist.json")) or
                os.path.exists(os.path.join(entry.path, "holdings.json"))
            )
            if has_data:
                result.append(entry.name)
    return result


def show_login():
    st.title("📈 InvestBro")
    st.markdown("#### Your AI-powered stock investment companion")
    st.markdown("---")

    # ── Saved users (quick login) ──────────────
    saved_users = _get_saved_users()
    if saved_users:
        st.markdown("### 👥 Saved Users")
        st.caption("Click a name to sign in instantly.")

        # Display up to 4 per row
        cols_per_row = 4
        for i in range(0, len(saved_users), cols_per_row):
            row_users = saved_users[i : i + cols_per_row]
            cols = st.columns(cols_per_row)
            for col, uname in zip(cols, row_users):
                # Derive a readable display name (replace underscores with spaces)
                display = uname.replace("_", " ").title()
                if col.button(f"👤 {display}", key=f"saved_user_{uname}", use_container_width=True):
                    st.session_state["username"]     = uname
                    st.session_state["display_name"] = display
                    st.session_state["page"]         = "home"
                    st.rerun()

        st.markdown("---")

    # ── New user entry ─────────────────────────
    st.markdown("### ✏️ New User" if saved_users else "### 👤 Welcome! Please enter your name to continue.")
    st.markdown(
        "Your watchlist, portfolio and holdings are saved privately under your name."
    )

    col, _ = st.columns([2, 3])
    with col:
        name_input = st.text_input(
            "Your Name",
            placeholder="e.g. Aryan, Priya …",
            help="Enter any name — each name gets its own private data.",
        ).strip()
        if st.button("🚀 Get Started", use_container_width=True, disabled=not name_input):
            # Sanitise: keep only alphanumeric + hyphens/underscores for folder safety
            safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name_input)
            st.session_state["username"]     = safe_name
            st.session_state["display_name"] = name_input
            st.session_state["page"]         = "home"
            st.rerun()


# ─────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────
if not st.session_state.get("username"):
    show_login()
elif st.session_state["page"] == "stock":
    show_stock()
elif st.session_state["page"] == "portfolio":
    # Import and run portfolio page inline
    import portfolio as _portfolio
    _portfolio.show_portfolio()
else:
    show_home()
