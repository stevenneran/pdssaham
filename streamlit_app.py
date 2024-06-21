import subprocess
import sys
import streamlit as st

# Fungsi untuk menginstal modul dari requirements.txt
def install_requirements():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

# Instalasi dari requirements.txt
install_requirements()

# Import necessary libraries
import streamlit as st
import yfinance as yf
import pandas as pd
from ta.utils import dropna
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import io

# Title and Description
st.title("Simulasi GBM untuk Harga Saham ASII")
st.write("Anggota Kelompok: [Nama Anggota 1, Nama Anggota 2, Nama Anggota 3, ...]")

# Parameters for simulation
st.sidebar.header("Parameter Simulasi")
start_date = st.sidebar.date_input("Tanggal Mulai", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("Tanggal Akhir", pd.to_datetime("2023-12-31"))
ticker = st.sidebar.text_input("Ticker", "ASII.JK")
simulation_days = st.sidebar.number_input("Jangka Waktu Simulasi (hari)", min_value=1, value=252)
n_simulations = st.sidebar.number_input("Jumlah Simulasi", min_value=1, value=100)

# Download data
data = yf.download(ticker, start=start_date, end=end_date)
data['Log Return'] = np.log(data['Close'] / data['Close'].shift(1))
data.dropna(inplace=True)

# Preprocess data
features = data.drop(columns=['Log Return']).columns
scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])

# Split data
X = data[features]
y = data['Log Return']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model_best = GradientBoostingRegressor(learning_rate=0.1, max_depth=5, min_samples_leaf=2, min_samples_split=5, n_estimators=400, random_state=42)
model_best.fit(X_train, y_train)

# Calculate annualized volatility
daily_returns = np.log(data["Close"].pct_change() + 1)
annualized_volatility = daily_returns.std() * np.sqrt(252)

# Define GBM simulation function
def gbm_sim(spot_price, volatility, time_horizon, model, features, data):
    dt = 1 / 252
    drift = model.predict(data[features])
    paths = np.zeros((time_horizon, n_simulations))
    paths[0] = spot_price
    
    for t in range(1, time_horizon):
        Z = np.random.normal(size=n_simulations)
        paths[t] = paths[t-1] * np.exp((drift[t-1] - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * Z)
    
    return paths

# Run simulation
spot_price = data['Close'].iloc[-1]
simulated_prices = gbm_sim(spot_price, annualized_volatility, simulation_days, model_best, features, data)

# Plot results
st.subheader("Hasil Prediksi Simulasi")
plt.figure(figsize=(12, 6))
for i in range(n_simulations):
    plt.plot(simulated_prices[:, i], lw=0.5)
plt.xlabel("Hari")
plt.ylabel("Harga Saham")
plt.title(f"Simulasi Harga Saham {ticker} Selama {simulation_days} Hari")
plt.grid(True)
st.pyplot(plt)

# Display percentiles
percentiles = np.percentile(simulated_prices[-1, :], [5, 50, 95])
st.write(f"5th percentile: {percentiles[0]}")
st.write(f"Median: {percentiles[1]}")
st.write(f"95th percentile: {percentiles[2]}")

# Save results to CSV or Excel
if st.button("Simpan Hasil ke CSV"):
    results_df = pd.DataFrame(simulated_prices)
    csv = results_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="simulated_prices.csv">Download CSV File</a>'
    st.markdown(href, unsafe_allow_html=True)

if st.button("Simpan Hasil ke Excel"):
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    results_df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.save()
    processed_data = output.getvalue()
    b64 = base64.b64encode(processed_data).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="simulated_prices.xlsx">Download Excel File</a>'
    st.markdown(href, unsafe_allow_html=True)
