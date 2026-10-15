"""app.py - Streamlit page for the crypto signal engine.
Run with:  streamlit run app.py
"""
import io
import os

import pandas as pd
import streamlit as st

import crypto

st.set_page_config(page_title="Crypto Signal", layout="wide")
st.title("Crypto Signal Engine")

# ---------------------------------------------------------- SIDEBAR SETTINGS
st.sidebar.header("Settings")
sec = st.sidebar.slider("Bar size (seconds)", 1, 60, 1)
ahead = st.sidebar.slider("Predict ahead (bars)", 1, 60, 10)
share = st.sidebar.slider("BUY/SELL share (%)", 5, 40, 25)
tail = share / 100

# ---------------------------------------------------------- 1. LOAD DATA
st.header("1. Load data")
uploaded = st.file_uploader("Upload a Binance trades CSV", type=["csv"])

raw = None
if uploaded is not None:
    raw = uploaded.getvalue()
elif os.path.exists(crypto.DATA_FILE):
    if st.checkbox(f"Use local file {crypto.DATA_FILE}", value=True):
        with open(crypto.DATA_FILE, "rb") as f:
            raw = f.read()

if raw is None:
    st.info("Upload a trades file to begin.")
    st.stop()


@st.cache_data
def prepare(data, sec):
    trades = crypto.load_trades(io.BytesIO(data), nrows=500000)
    return crypto.features(crypto.to_bars(trades, sec)).tail(30000)


bars = prepare(raw, sec)
st.write(f"{len(bars)} bars ready")
st.line_chart(bars.set_index("timestamp")["price"])

# ---------------------------------------------------------- 2. TRAIN MODELS
st.header("2. Train models")
if st.button("Train models"):
    rows, fitted = [], {}
    for name, model in crypto.get_models().items():
        acc, base = crypto.train(bars, model, ahead=ahead, tail=tail)
        rows.append({"Model": name, "Accuracy": round(acc, 3), "Baseline": round(base, 3)})
        fitted[name] = model
    st.session_state["fitted"] = fitted
    st.table(pd.DataFrame(rows))

# ---------------------------------------------------------- 3. LIVE SIGNAL
st.header("3. Live signal")
if st.button("Live signal"):
    fitted = st.session_state.get("fitted")
    if not fitted:
        st.warning("Train the models first.")
    else:
        try:
            signal, conf, price = crypto.live_signal(fitted["Random Forest"], sec)
            c1, c2 = st.columns(2)
            c1.metric("Signal", signal, f"{conf:.0%} confidence")
            c2.metric("Price", f"{price:,.2f}")
            st.info(crypto.paper_plan(signal, price))
        except Exception as e:
            st.error(f"Could not get live data: {e}")
