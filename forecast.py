"""Multi-step recursive forecasting for the per-coin LSTM models.

The models are trained to predict the next day's (scaled) closing price from a
fixed window of past closes. To forecast `horizon` days ahead, we predict one
step, append that prediction to the window, drop the oldest value, and repeat.
"""

import numpy as np


def model_look_back(model, default=60):
    """Window length the model was trained with (e.g. 15 for BTC, 60 for the others)."""
    try:
        steps = model.input_shape[1]
        return int(steps) if steps else default
    except (AttributeError, IndexError, TypeError):
        return default


def recursive_forecast(model, scaled_series, horizon, look_back):
    """Forecast `horizon` future steps from the last `look_back` scaled values.

    scaled_series: 1-D or (n, 1) array of MinMax-scaled closing prices, oldest first.
    Returns a 1-D array of `horizon` scaled predictions.
    """
    series = np.asarray(scaled_series, dtype="float32").reshape(-1)
    if len(series) < look_back:
        raise ValueError(f"need at least {look_back} data points, got {len(series)}")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    window = series[-look_back:].reshape(1, look_back, 1)
    preds = []
    for _ in range(horizon):
        nxt = float(np.asarray(model.predict(window, verbose=0)).reshape(-1)[0])
        preds.append(nxt)
        window = np.concatenate([window[:, 1:, :], np.array([[[nxt]]], dtype="float32")], axis=1)
    return np.array(preds)
