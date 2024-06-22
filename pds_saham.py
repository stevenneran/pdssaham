import subprocess
import sys

# Fungsi untuk menginstal modul dari requirements.txt
def install_requirements():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

# Instalasi dari requirements.txt
install_requirements()

# Import necessary libraries
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import io
import base64

# Set page config to wide layout
st.set_page_config(layout="wide")

# Title and Description
st.title("Simulasi GBM untuk Prediksi Harga Saham ASII")
st.write("Anggota Kelompok: [Argi, Azis, Cakra, Marcel, Steven]")

# Parameters for simulation
st.sidebar.header("Parameter Simulasi")
min_date = pd.to_datetime("2022-01-01").date()
start_date = st.sidebar.date_input("Tanggal Mulai", min_date)
st.sidebar.info("Tanggal mulai harus sebelum tanggal hari ini.", icon="ℹ️")
time_horizon = st.sidebar.number_input('Jangka Waktu (hari setelah hari ini)', min_value=30, value=500)
# st.sidebar.info(f"{time_horizon} hari setelah hari ini.", icon="ℹ️")
ticker = st.sidebar.text_input("Kode Saham", "ASII")

# Capitalize ticker and add .JK
ticker = ticker.upper() + ".JK"

# Download data
data = yf.download(ticker, start=start_date, end=None)
data['Log Return'] = np.log(data['Close'] / data['Close'].shift(1))
data.dropna(inplace=True)

# Preprocess data
features = data.drop(columns=['Log Return']).columns

# Split data
X = data[features]
y = data['Log Return']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model_best = GradientBoostingRegressor(learning_rate=0.1, max_depth=5, min_samples_leaf=2, min_samples_split=5, n_estimators=400, random_state=42)
model_best.fit(X_train, y_train)

# Hitung volatilitas tahunan berdasarkan pengembalian log harian
daily_returns = np.log(data["Close"].pct_change() + 1)
annualized_volatility = daily_returns.std() * np.sqrt(252)

# Fungsi untuk simulasi GBM dengan drift yang diprediksi oleh model
def gbm_sim(spot_price, volatility, time_horizon, model, features, data):
    dt = 1 / 252  # asumsi satu hari perdagangan (252 hari perdagangan dalam setahun)
    drift = model.predict(data[features])
    paths = np.zeros(time_horizon + 1)  # Change len(data) + 1 to time_horizon + 1
    paths[0] = spot_price
    
    for i in range(1, time_horizon + 1):  # Change len(data) + 1 to time_horizon + 1
        Z = np.random.normal()
        paths[i] = paths[i-1] * np.exp((drift[min(i-1, len(drift)-1)] - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * Z)
    
    return paths

# Definisikan parameter simulasi
spot_price = data['Close'].iloc[-1]  # harga penutupan terakhir sebagai harga awal
features = X.columns

# # Calculate time horizon
# time_horizon = (end_date - start_date).days
# st.write(time_horizon)

# Lakukan simulasi GBM
simulated_prices = gbm_sim(spot_price, annualized_volatility, time_horizon, model_best, features, data)
# st.write(simulated_prices)

# Buat index tanggal untuk hasil simulasi
simulated_index = pd.date_range(start=data.index[-1], periods=time_horizon+1, freq='B')
# st.write(simulated_index)

# Plot hasil simulasi dan data aktual
st.subheader("Hasil Prediksi Simulasi")
plt.figure(figsize=(12, 6))
plt.plot(simulated_index, simulated_prices, lw=0.5, label='Simulasi')
plt.plot(data.index, data['Close'], lw=0.5, label='Aktual')
plt.xlabel("Tanggal")
plt.ylabel("Harga Saham")
plt.title(f"Simulasi Harga Saham {ticker}")
plt.legend(loc='best')
plt.grid(True)
st.pyplot(plt)

# Display percentiles
percentiles = np.percentile(simulated_prices[1:], [5, 50, 95])
st.write(f"5th percentile: {percentiles[0]}")
st.write(f"Median: {percentiles[1]}")
st.write(f"95th percentile: {percentiles[2]}")

# Save results to CSV
if st.button("Simpan Hasil ke CSV"):
    results_df = pd.DataFrame(simulated_prices)
    csv = results_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="simulated_prices.csv">Download CSV File</a>'
    st.markdown(href, unsafe_allow_html=True)
