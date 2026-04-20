import streamlit as st
import pandas as pd
import numpy as np
import requests
import sqlite3
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

def init_db():
    conn = sqlite3.connect("nifty.db")
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS pe_data (time TEXT, pe REAL, price REAL)")
    conn.commit()
    conn.close()

def fetch_data():
    url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"
    headers = {"User-Agent": "Mozilla/5.0"}

    session = requests.Session()
    session.get("https://www.nseindia.com", headers=headers)
    res = session.get(url, headers=headers).json()

    pe = res['data'][0]['pe']
    price = res['data'][0]['lastPrice']

    conn = sqlite3.connect("nifty.db")
    c = conn.cursor()
    c.execute("INSERT INTO pe_data VALUES (?, ?, ?)", (datetime.now(), pe, price))
    conn.commit()
    conn.close()

def load_data():
    conn = sqlite3.connect("nifty.db")
    df = pd.read_sql("SELECT * FROM pe_data", conn)
    conn.close()

    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time')

    df['returns'] = df['price'].pct_change()
    df['pe_z'] = (df['pe'] - df['pe'].mean()) / df['pe'].std()
    df['trend'] = df['price'].rolling(20).mean() - df['price'].rolling(50).mean()
    df['vol'] = df['returns'].rolling(20).std()
    df['target'] = (df['returns'].shift(-1) > 0).astype(int)

    df = df.dropna()
    return df

def train_model(df):
    features = ['pe', 'pe_z', 'trend', 'vol']
    X = df[features]
    y = df['target']

    model = RandomForestClassifier(n_estimators=100, max_depth=5)
    model.fit(X, y)

    return model

st.title("📊 NIFTY ML APP")

init_db()

if st.button("Fetch Data"):
    fetch_data()
    st.success("Data Added")

df = load_data()

if len(df) > 50:
    model = train_model(df)

    latest = df.iloc[-1]
    prob = model.predict_proba([latest[['pe','pe_z','trend','vol']]])[0][1]

    st.metric("Bullish Probability", f"{prob:.2f}")

    if prob > 0.6:
        st.success("🚀 Bullish → Buy CE")
    elif prob < 0.4:
        st.error("💣 Bearish → Buy PE")
    else:
        st.warning("⚖️ Neutral")

    st.line_chart(df.set_index('time')['pe'])

else:
    st.warning("Click Fetch Data multiple times to build dataset")
