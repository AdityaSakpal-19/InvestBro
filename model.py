"""
model.py — XGBoost stock price prediction model for InvestBro
Handles feature engineering, training, weekly retraining gate, save/load.
"""

import os
import json
import datetime

import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MODELS_DIR = os.path.join("storage", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    return 100 - (100 / (1 + avg_gain / avg_loss))


def prepare_features(df: pd.DataFrame):
    """
    Features: Open, High, Low, Close, Volume, MA_20, MA_50, RSI_14
    Target:   next-period closing price
    """
    data = df.copy()
    data["MA_20"]  = data["Close"].rolling(window=20).mean()
    data["MA_50"]  = data["Close"].rolling(window=50).mean()
    data["RSI_14"] = compute_rsi(data["Close"])
    data["Target"] = data["Close"].shift(-1)
    data.dropna(inplace=True)
    cols = ["Open", "High", "Low", "Close", "Volume", "MA_20", "MA_50", "RSI_14"]
    return data[cols], data["Target"]


# ─────────────────────────────────────────────
# Save / Load helpers
# ─────────────────────────────────────────────

def _model_paths(ticker: str) -> tuple[str, str]:
    safe = ticker.replace(".", "_")
    return (
        os.path.join(MODELS_DIR, f"{safe}.joblib"),
        os.path.join(MODELS_DIR, f"{safe}_meta.json"),
    )


def _load_meta(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _save_meta(path: str, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ─────────────────────────────────────────────
# Weekly Training Gate
# ─────────────────────────────────────────────

def get_or_train_model(df: pd.DataFrame, ticker: str):
    """
    Return (model, metrics, was_retrained: bool).

    Training schedule:
      • Data collected daily via update_daily_data() in app.py.
      • Model retrained once per ISO calendar week (Mon–Sun).
        - First visit of a new week  → retrain on full accumulated dataset, save.
        - Any other visit same week  → load saved model instantly (no retraining).
      • Non-watchlist stocks have no saved model → always train fresh from live data.
    """
    model_file, meta_file = _model_paths(ticker)
    today        = datetime.date.today()
    current_week = today.strftime("%Y-W%W")   # e.g. "2025-W28"

    meta = _load_meta(meta_file)
    if os.path.exists(model_file) and meta.get("trained_week") == current_week:
        model = joblib.load(model_file)
        return model, meta["metrics"], False

    # ── Train ─────────────────────────────────
    X, y = prepare_features(df)
    if len(X) < 60:
        raise ValueError(
            "Not enough historical data to train the model. "
            "Please choose a ticker with more history."
        )

    split_idx = int(len(X) * 0.80)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    metrics = {
        "MAE":  round(float(mean_absolute_error(y_test, y_pred)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
        "R2":   round(float(r2_score(y_test, y_pred)), 4),
    }

    joblib.dump(model, model_file)
    _save_meta(meta_file, {
        "ticker":       ticker,
        "trained_week": current_week,
        "trained_date": today.isoformat(),
        "metrics":      metrics,
    })

    return model, metrics, True


def predict_next_price(model, df: pd.DataFrame) -> float:
    """Predict the next closing price using the most recent data row."""
    X, _ = prepare_features(df)
    return float(model.predict(X.iloc[[-1]])[0])
