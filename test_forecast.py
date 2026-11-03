import numpy as np
import pytest

from forecast import model_look_back, recursive_forecast


class LastValuePlusOne:
    """Stand-in model: predicts last value in the window + 0.01, records the inputs it saw."""

    input_shape = (None, 5, 1)

    def __init__(self):
        self.seen = []

    def predict(self, x, verbose=0):
        self.seen.append(x.copy())
        return np.array([[x[0, -1, 0] + 0.01]])


def test_forecast_rolls_window_forward():
    model = LastValuePlusOne()
    series = np.linspace(0.0, 0.4, 41)
    preds = recursive_forecast(model, series, horizon=3, look_back=5)

    assert np.allclose(preds, [0.41, 0.42, 0.43])
    # each step feeds the previous prediction back in, so predictions are truly future values
    assert np.isclose(model.seen[1][0, -1, 0], 0.41)
    assert np.isclose(model.seen[2][0, -1, 0], 0.42)
    assert all(w.shape == (1, 5, 1) for w in model.seen)


def test_horizon_length_matches_request():
    for h in (1, 7, 30):
        assert len(recursive_forecast(LastValuePlusOne(), np.zeros(10), horizon=h, look_back=5)) == h


def test_look_back_read_from_model():
    assert model_look_back(LastValuePlusOne()) == 5
    assert model_look_back(object(), default=60) == 60


def test_too_little_history_raises():
    with pytest.raises(ValueError):
        recursive_forecast(LastValuePlusOne(), np.zeros(3), horizon=1, look_back=5)
