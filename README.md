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
- **True multi-step forecasts:** choose a 1-, 7- or 30-day horizon; the model predicts one day ahead, feeds that prediction back into its input window and repeats (recursive forecasting). The chart shows the last 60 days of actual prices with the forecast continuing from them.
- **Forecast view:** final predicted price and trend direction (up/down), plus min / max / average price cards.
- **Accounts:** sign-up and login with **bcrypt-hashed** passwords stored in a local SQLite database.
- **Export:** download the price history as CSV.

## How the model works

| Step | Detail |
|---|---|
| Data | Daily OHLCV from Yahoo Finance (sample CSVs included for each coin) |
| Scaling | MinMax scaling to [0, 1] |
| Windows | Sliding look-back windows (60 days; 15 for the BTC model), read from each model's input shape |
| Model | 3 stacked LSTM layers (100 units each) + dense output, one model per coin |
| Output | Next-day closing price, rolled forward recursively for the chosen horizon and inverse-scaled to USD |

During the internship this LSTM was benchmarked against **GRU, Random Forest and XGBoost** under a single evaluation protocol; the best model cut RMSE by **20–30%** relative to a naive last-value baseline.

## Quick start

```bash
git clone https://github.com/NIHAL670/cryptop_prediction_model.git
cd cryptop_prediction_model
pip install -r requirements.txt
streamlit run app.py
```

Create an account on the login screen; the user database (`users.db`) is created automatically on first run and is not committed to the repository.

## Tests

```bash
pip install pytest
python -m pytest -q test_forecast.py
```

The tests check that each forecast step feeds the previous prediction back in, that the horizon length is respected, and that the window size is read from the model.

## Repository layout

```
├── app.py            # Streamlit app: landing page, dashboard, forecasting
├── auth.py           # SQLite + bcrypt user accounts
├── forecast.py       # recursive multi-step forecasting
├── test_forecast.py  # unit tests for the forecaster
├── *_model.h5        # trained LSTM per coin (BTC = model.h5)
├── *_USD.csv         # sample historical data per coin
└── requirements.txt
```

## Disclaimer

Crypto markets are extremely volatile. This project is for learning and demonstration only and is **not financial advice**.

## Author

**Nihal Yadav** — M.Tech AI & ML, IIT Delhi · [GitHub](https://github.com/NIHAL670) · [LinkedIn](https://linkedin.com/in/nihal-yadav-5131582b8)
