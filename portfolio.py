"""
portfolio.py — InvestBro Portfolio Page

Shows a collective view of all watchlist stocks:
  - Per-stock: units, avg price, current value, unrealised P&L, predicted value
  - Portfolio totals
  - Navigates back to home or to any individual stock page
"""

import os
import json
import datetime

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

from model import get_or_train_model, predict_next_price

# ─────────────────────────────────────────────
# Shared storage paths (mirrors app.py constants)
# ─────────────────────────────────────────────
STORAGE_DIR    = "storage"
DATA_DIR       = os.path.join(STORAGE_DIR, "data")
WATCHLIST_FILE = os.path.join(STORAGE_DIR, "watchlist.json")
HOLDINGS_FILE  = os.path.join(STORAGE_DIR, "holdings.json")


# ─────────────────────────────────────────────
# Helpers (thin wrappers — same logic as app.py)
# ─────────────────────────────────────────────
def _load_json(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def load_watchlist() -> list:
    return _load_json(WATCHLIST_FILE, [])


def load_holdings() -> dict:
    return _load_json(HOLDINGS_FILE, {})


def get_holding(ticker: str) -> dict:
    return load_holdings().get(ticker, {"units": 0.0, "avg_price": 0.0, "transactions": []})


def load_stored_data(ticker: str):
    path = os.path.join(DATA_DIR, f"{ticker.replace('.', '_')}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df if not df.empty else None


@st.cache_data(show_spinner=False)
def _fetch_price(ticker: str) -> float | None:
    """Fetch current price for a ticker (cached per session run)."""
    try:
        info = yf.Ticker(ticker).info
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if price:
            return float(price)
    except Exception:
        pass
    # Fallback: last close from stored CSV
    df = load_stored_data(ticker)
    if df is not None and not df.empty:
        return float(df["Close"].iloc[-1])
    return None


@st.cache_data(show_spinner=False)
def _fetch_live(ticker: str):
    """Return (df_live, current_price) for prediction, cached."""
    try:
        df = yf.download(ticker, period="5y", interval="1d",
                         progress=False, auto_adjust=True)
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
    except Exception:
        pass
    return load_stored_data(ticker)


def _get_predicted_price(ticker: str) -> float | None:
    """Load or train model for ticker and return predicted next price."""
    df = _fetch_live(ticker)
    if df is None or df.empty:
        return None
    stored = load_stored_data(ticker)
    if stored is not None:
        extra = stored[~stored.index.isin(df.index)]
        if not extra.empty:
            df = pd.concat([df, extra]).sort_index()
    try:
        model, _, _ = get_or_train_model(df, ticker)
        return predict_next_price(model, df)
    except Exception:
        return None


# ─────────────────────────────────────────────
# Portfolio page
# ─────────────────────────────────────────────
def show_portfolio():
    # ── Back navigation ────────────────────────
    nav_l, nav_r = st.columns([6, 2])
    with nav_l:
        st.title("📊 Portfolio")
        if st.button("← Back to Home"):
            st.session_state["page"] = "home"
            st.rerun()
    with nav_r:
        st.markdown("<div style='padding-top:1.6rem;'>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("*Collective view of all your watched & invested stocks*")
    st.markdown("---")

    watchlist = load_watchlist()
    if not watchlist:
        st.info("Your watchlist is empty. Add stocks from the home page to see your portfolio here.")
        return

    # ── Build per-ticker rows ───────────────────
    rows = []
    with st.spinner("Loading portfolio data …"):
        for ticker in watchlist:
            current_price   = _fetch_price(ticker)
            predicted_price = _get_predicted_price(ticker)
            holding         = get_holding(ticker)

            units_held = holding["units"]
            avg_price  = holding["avg_price"] if holding["avg_price"] > 0 else (current_price or 0)

            if current_price is None:
                continue

            holding_value  = units_held * current_price
            unrealised_amt = units_held * (current_price - avg_price)
            unrealised_pct = ((current_price - avg_price) / avg_price * 100) if avg_price > 0 else 0
            cost_basis     = units_held * avg_price

            pred_value = units_held * predicted_price if predicted_price else None
            pred_pct   = ((predicted_price - current_price) / current_price * 100) if predicted_price else None

            rows.append({
                "ticker":          ticker,
                "units_held":      units_held,
                "avg_price":       avg_price,
                "current_price":   current_price,
                "cost_basis":      cost_basis,
                "holding_value":   holding_value,
                "unrealised_amt":  unrealised_amt,
                "unrealised_pct":  unrealised_pct,
                "predicted_price": predicted_price,
                "pred_value":      pred_value,
                "pred_pct":        pred_pct,
            })

    if not rows:
        st.warning("Could not fetch prices for any watchlist stock.")
        return

    # ── Portfolio totals ────────────────────────
    total_cost        = sum(r["cost_basis"]    for r in rows)
    total_value       = sum(r["holding_value"] for r in rows)
    total_unrealised  = sum(r["unrealised_amt"] for r in rows)
    total_pred_value  = sum(r["pred_value"] for r in rows if r["pred_value"] is not None)
    total_unrealised_pct = ((total_value - total_cost) / total_cost * 100) if total_cost > 0 else 0
    total_pred_pct       = ((total_pred_value - total_value) / total_value * 100) if total_value > 0 else 0

    st.subheader("💰 Portfolio Summary")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Total Invested (₹)",    f"₹{total_cost:,.2f}")
    p2.metric("Current Value (₹)",     f"₹{total_value:,.2f}",
              delta=f"{total_unrealised_pct:+.2f}%")
    p3.metric("Unrealised P&L (₹)",    f"₹{total_unrealised:,.2f}",
              delta=f"₹{total_unrealised:+,.2f}")
    p4.metric("Predicted Value (₹) *", f"₹{total_pred_value:,.2f}",
              delta=f"{total_pred_pct:+.2f}%")

    st.markdown("---")

    # ── Per-stock table ─────────────────────────
    st.subheader("📋 Holdings Detail")
    st.caption("⚠️ Predicted values are model estimates, not guaranteed returns. Returns based on Avg Price.")

    for r in rows:
        ticker = r["ticker"]
        with st.container():
            col_name, col_open = st.columns([5, 1])
            col_name.markdown(f"### {ticker}")
            if col_open.button("Open →", key=f"portfolio_open_{ticker}"):
                st.session_state["stock_params"] = {
                    "ticker": ticker,
                    "tenure": "1 Year",
                    "risk":   "Medium",
                    "goal":   "Long-term growth",
                }
                st.session_state["page"] = "stock"
                st.rerun()

            # Row 1 — position
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Units Held",        f"{int(r['units_held']):,}")
            a2.metric("Avg Buy Price (₹)", f"₹{r['avg_price']:,.2f}")
            a3.metric("Current Price (₹)", f"₹{r['current_price']:,.2f}")
            a4.metric("Cost Basis (₹)",    f"₹{r['cost_basis']:,.2f}")

            # Row 2 — P&L + prediction
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Holding Value (₹)",
                      f"₹{r['holding_value']:,.2f}",
                      delta=f"{r['unrealised_pct']:+.2f}%")
            b2.metric("Unrealised P&L (₹)",
                      f"₹{r['unrealised_amt']:,.2f}",
                      delta=f"₹{r['unrealised_amt']:+,.2f}")
            if r["predicted_price"]:
                b3.metric("Predicted Price (₹) *",
                          f"₹{r['predicted_price']:,.2f}",
                          delta=f"{r['pred_pct']:+.2f}%")
                b4.metric("Predicted Value (₹) *",
                          f"₹{r['pred_value']:,.2f}",
                          delta=f"₹{r['pred_value'] - r['holding_value']:+,.2f}")
            else:
                b3.metric("Predicted Price", "—")
                b4.metric("Predicted Value", "—")

            st.markdown("---")

    # ── Transaction log (all stocks combined) ───
    holdings = load_holdings()
    all_tx = []
    for ticker in watchlist:
        h = holdings.get(ticker, {})
        for tx in h.get("transactions", []):
            all_tx.append({**tx, "ticker": ticker})

    if all_tx:
        with st.expander("📋 Full Transaction History", expanded=False):
            tx_df = pd.DataFrame(all_tx)[["ticker", "date", "type", "units", "price"]]
            tx_df.columns = ["Ticker", "Date", "Type", "Units", "Price (₹)"]
            tx_df = tx_df.sort_values("Date", ascending=False)
            st.dataframe(tx_df, use_container_width=True, hide_index=True)

    # ── Disclaimer ─────────────────────────────
    st.markdown(
        """
        <div style="background:#f7f8fa;border:1px solid #e5e7eb;
                    border-radius:6px;padding:1rem;color:#57606a;font-size:0.82rem;">
        <strong>Disclaimer:</strong> InvestBro provides educational analysis and model-based estimates.
        It is <strong>not financial advice</strong>. Past performance does not guarantee future returns.
        Always consult a qualified financial advisor before making investment decisions.
        </div>
        """,
        unsafe_allow_html=True,
    )
