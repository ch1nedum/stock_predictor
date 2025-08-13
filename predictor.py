# import yfinance as yf
# import pandas as pd
# import numpy as np
# from sklearn.linear_model import Ridge
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, r2_score
# import matplotlib
# import matplotlib.pyplot as plt
# matplotlib.use('Agg')
# import os
# import joblib
# from functools import lru_cache

# def fetch_data(ticker, period="6mo", interval="1d"):
#     df = yf.download(ticker, period=period, interval=interval)
#     if df is None or df.empty:
#         raise ValueError(f"No data found for ticker {ticker}. Please check the ticker symbol or try a different period.")
#     df.dropna(inplace=True)
#     return df

# def compute_RSI(series, period=14):
#     delta = series.diff()
#     gain = delta.where(delta > 0, 0).rolling(window=period).mean()
#     loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
#     rs = gain / loss
#     return 100 - (100 / (1 + rs))

# def add_features(df):
#     df["MA5"] = df["Close"].rolling(window=5).mean()
#     df["MA10"] = df["Close"].rolling(window=10).mean()
#     df["Returns"] = df["Close"].pct_change()
#     df["RSI"] = compute_RSI(df["Close"])
#     df.dropna(inplace=True)
#     return df

# def train_and_predict(df, model_type="ridge"):
#     # Target: next-day returns
#     df["Target_Return"] = df["Close"].pct_change().shift(-1)
#     df.dropna(inplace=True)

#     features = ["MA5", "MA10", "Returns", "RSI", "Volume"]
#     X = df[features]
#     y = df["Target_Return"]

#     split_idx = int(len(df) * 0.9)
#     X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
#     y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

#     if model_type == "ridge":
#         model = Ridge(alpha=1.0)
#     elif model_type == "random_forest":
#         model = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=42)
#     else:
#         raise ValueError("Invalid model type. Use 'ridge' or 'random_forest'.")

#     model.fit(X_train, y_train)

#     preds_test = model.predict(X_test)
#     mse = mean_squared_error(y_test, preds_test)
#     r2 = r2_score(y_test, preds_test)

#     # Predict next day return (last row features)
#     last_features = X.iloc[[-1]]
#     pred_return = model.predict(last_features)[0]

#     last_close = df["Close"].iloc[-1]
#     pred_price = last_close * (1 + pred_return)

#     # Basic trade signal thresholds
#     trade_signal = "Hold"
#     if pred_return > 0.002:
#         trade_signal = "Buy"
#     elif pred_return < -0.002:
#         trade_signal = "Sell"

#     return {
#         "pred_price": pred_price,
#         "pred_return": pred_return,
#         "last_close": last_close,
#         "mse": mse,
#         "r2": r2,
#         "trade_signal": trade_signal,
#         "df": df,
#         "model": model
#     }

# def plot_results(df, pred_price, ticker):
#     plt.figure(figsize=(10,5))
#     plt.plot(df.index, df["Close"], label="Historical Close")
#     plt.scatter(df.index[-1] + pd.Timedelta(days=1), pred_price, color="red", label="Predicted Next Close")
#     plt.xlabel("Date")
#     plt.ylabel("Price")
#     plt.title(f"{ticker} Price Prediction")
#     plt.legend()
#     plt.tight_layout()

#     os.makedirs("static/plots", exist_ok=True)
#     filepath = f"static/plots/{ticker}_prediction.png"
#     plt.savefig(filepath)
#     plt.close()
#     return filepath
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')  # ✅ Prevent plot hanging in Flask
import matplotlib.pyplot as plt
import os
import joblib
from functools import lru_cache

@lru_cache(maxsize=5)
def fetch_data(ticker, period="6mo", interval="1d"):
    try:
        df = yf.download(
            ticker, 
            period=period, 
            interval=interval,
            progress=False,
            threads=True
        )
    except Exception as e:
        raise ValueError(f"Data fetch failed for {ticker}: {str(e)}")

    # ✅ Ensure df is actually a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Unexpected data format returned for {ticker}")

    # ✅ Handle empty df
    if df.empty:
        raise ValueError(f"No data found for ticker {ticker}. Check symbol.")

    # ✅ Drop NaN safely
    df = df.dropna(how="any").copy()

    if df.empty:
        raise ValueError(f"No valid data after cleaning for ticker {ticker}.")

    return df

def compute_RSI(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def add_features(df):
    df = df.copy()
    df["MA5"] = df["Close"].rolling(window=5).mean()
    df["MA10"] = df["Close"].rolling(window=10).mean()
    df["Returns"] = df["Close"].pct_change()
    df["RSI"] = compute_RSI(df["Close"])
    df.dropna(inplace=True)
    return df

def train_and_predict(df, ticker, model_type="ridge"):
    df = df.copy()

    # Target: next-day returns
    df["Target_Return"] = df["Close"].pct_change().shift(-1)
    df.dropna(inplace=True)

    features = ["MA5", "MA10", "Returns", "RSI", "Volume"]
    X = df[features]
    y = df["Target_Return"]

    split_idx = int(len(df) * 0.9)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model_path = f"models/{ticker}_{model_type}.pkl"
    os.makedirs("models", exist_ok=True)

    # ✅ Load saved model if exists
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        if model_type == "ridge":
            model = Ridge(alpha=1.0)
        elif model_type == "random_forest":
            model = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=42)
        else:
            raise ValueError("Invalid model type. Use 'ridge' or 'random_forest'.")
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)

    preds_test = model.predict(X_test)
    mse = mean_squared_error(y_test, preds_test)
    r2 = r2_score(y_test, preds_test)

    # Predict next day return (last row features)
    last_features = X.iloc[[-1]]
    pred_return = float(model.predict(last_features)[0])

    last_close = float(df["Close"].iloc[-1])
    pred_price = float(last_close * (1 + pred_return))

    # Basic trade signal thresholds
    trade_signal = "Hold"
    if pred_return > 0.002:
        trade_signal = "Buy"
    elif pred_return < -0.002:
        trade_signal = "Sell"

    return {
        "pred_price": float(pred_price),
        "pred_return": float(pred_return),
        "last_close": float(last_close),
        "mse": float(mse),
        "r2": float(r2),
        "trade_signal": trade_signal,
        "df": df,
        "model": model
    }

def plot_results(df, pred_price, ticker):
    plt.figure(figsize=(7,5))
    plt.plot(df.index, df["Close"], label="Historical Close")
    plt.scatter(df.index[-1] + pd.Timedelta(days=1), pred_price, color="red", label="Predicted Next Close")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(f"{ticker} Price Prediction")
    plt.legend()
    plt.tight_layout()

    os.makedirs("static/plots", exist_ok=True)
    filepath = f"static/plots/{ticker}_prediction.png"
    plt.savefig(filepath)
    plt.close()
    return filepath

