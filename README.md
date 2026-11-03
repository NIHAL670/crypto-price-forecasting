# 📈 Crypto Price Forecasting (LSTM)

A Streamlit web app that forecasts short-term prices for five major cryptocurrencies with per-coin **LSTM** models trained on historical OHLCV data from Yahoo Finance.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![yfinance](https://img.shields.io/badge/yfinance-data-6001D2?style=flat-square)

> Built during the Infosys Springboard Machine Learning internship (completed with distinction).

---

## Features

- **Five coins:** Bitcoin (BTC), Ethereum (ETH), BNB, XRP and Dogecoin — one trained LSTM per coin (`*_model.h5`).
- **Live data:** pulls price history from Yahoo Finance via `yfinance` for any date range you choose.
- **Forecast view:** predicted price path, final predicted price and trend direction (up/down), plus min / max / average price cards.
- **Accounts:** sign-up and login with **bcrypt-hashed** passwords stored in a local SQLite database.
- **Export:** download the price history as CSV.

## How the model works

| Step | Detail |
|---|---|
| Data | Daily OHLCV from Yahoo Finance (sample CSVs included for each coin) |
| Scaling | MinMax scaling to [0, 1] |
| Windows | Sliding 60-day look-back windows |
| Model | 3 stacked LSTM layers (100 units each) + dense output, one model per coin |
| Output | Next-step closing price, inverse-scaled back to USD |

During the internship this LSTM was benchmarked against **GRU, Random Forest and XGBoost** under a single evaluation protocol; the best model cut RMSE by **20–30%** relative to a naive last-value baseline.

## Quick start

```bash
git clone https://github.com/NIHAL670/cryptop_prediction_model.git
cd cryptop_prediction_model
pip install -r requirements.txt
streamlit run app.py
```

Create an account on the login screen; the user database (`users.db`) is created automatically on first run and is not committed to the repository.

## Repository layout

```
├── app.py            # Streamlit app: landing page, dashboard, forecasting
├── auth.py           # SQLite + bcrypt user accounts
├── *_model.h5        # trained LSTM per coin (BTC = model.h5)
├── *_USD.csv         # sample historical data per coin
└── requirements.txt
```

## Disclaimer

Crypto markets are extremely volatile. This project is for learning and demonstration only and is **not financial advice**.

## Author

**Nihal Yadav** — M.Tech AI & ML, IIT Delhi · [GitHub](https://github.com/NIHAL670) · [LinkedIn](https://linkedin.com/in/nihal-yadav-5131582b8)
