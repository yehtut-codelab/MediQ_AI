"""Loaders and predict functions for the trained models in Models/.

- waiting_time_xgb.pkl      XGBRegressor, target log1p(waiting_time_min)
- lstm_queue_forecast.pt    2-layer LSTM state dict, 15-min queue depth forecast
- clinic_hmms.pkl           per-category GaussianHMM congestion states (joblib)
"""

import json
import pickle
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn

MODELS_DIR = Path(__file__).resolve().parents[3] / "Models"


# ── metadata ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def wait_meta() -> dict:
    return json.loads((MODELS_DIR / "waiting_time_metadata.json").read_text())


@lru_cache(maxsize=1)
def lstm_meta() -> dict:
    return json.loads((MODELS_DIR / "lstm_queue_metadata.json").read_text())


@lru_cache(maxsize=1)
def hmm_meta() -> dict:
    return json.loads((MODELS_DIR / "hmm_metadata.json").read_text())


# ── XGBoost waiting time ─────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _xgb():
    with open(MODELS_DIR / "waiting_time_xgb.pkl", "rb") as f:
        return pickle.load(f)


def predict_wait_min(category: str, clinic: str, hour: int, day_of_week: int,
                     month: int, visit_position: int, visit_length: int,
                     appointment_lag_min: float, queue_depth: float) -> float:
    meta = wait_meta()
    lag_lo, lag_hi = meta["appointment_lag_clip"]
    x = np.array([[
        meta["category_encoding"].get(category, meta["category_encoding"]["other"]),
        meta["clinic_encoding"][clinic],
        hour, day_of_week, month, visit_position, visit_length,
        float(np.clip(appointment_lag_min, lag_lo, lag_hi)),
        queue_depth,
    ]])
    return float(np.expm1(_xgb().predict(x)[0]))  # target_transform: log1p / expm1


# ── LSTM queue depth forecast ────────────────────────────────────────────

class _QueueLSTM(nn.Module):
    """Architecture reconstructed from the checkpoint state dict."""

    def __init__(self, n_features: int = 8, hidden1: int = 64, hidden2: int = 32,
                 horizon: int = 4, dropout: float = 0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, hidden1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden2, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm1(x)
        out = self.dropout(out)
        out, _ = self.lstm2(out)
        return self.linear(out[:, -1, :])


@lru_cache(maxsize=1)
def _lstm() -> tuple[_QueueLSTM, object]:
    meta = lstm_meta()
    model = _QueueLSTM(
        n_features=meta["n_features"],
        hidden1=meta["architecture"]["hidden1"],
        hidden2=meta["architecture"]["hidden2"],
        horizon=meta["horizon_steps"],
        dropout=meta["architecture"]["dropout"],
    )
    model.load_state_dict(
        torch.load(MODELS_DIR / "lstm_queue_forecast.pt", map_location="cpu",
                   weights_only=False)
    )
    model.eval()
    scaler = joblib.load(MODELS_DIR / "lstm_queue_scaler.pkl")
    return model, scaler


def forecast_queue_depth(bucket_features: np.ndarray) -> list[float]:
    """bucket_features: (lookback_steps, 8) raw feature rows in metadata order.
    Returns queue depth forecast for the next horizon_steps 15-min buckets."""
    model, scaler = _lstm()
    scaled = scaler.transform(bucket_features)
    with torch.no_grad():
        out = model(torch.tensor(scaled[None], dtype=torch.float32))[0].numpy()
    # invert MinMax scaling using target column 0 (queue_depth_mean)
    lo, hi = float(scaler.data_min_[0]), float(scaler.data_max_[0])
    return [max(0.0, round(float(v) * (hi - lo) + lo, 2)) for v in out]


# ── HMM congestion states ────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _hmms() -> dict:
    return joblib.load(MODELS_DIR / "clinic_hmms.pkl")


def congestion_state(category: str, recent_waits_min: list[float]) -> dict:
    """Most likely current congestion state given recent observed waits."""
    entry = _hmms().get(category)
    if entry is None or not recent_waits_min:
        return {"category": category, "state": "unknown", "state_mean_wait": None}
    obs = np.array(recent_waits_min, dtype=float).reshape(-1, 1)
    states = entry["model"].predict(obs)
    current = int(states[-1])
    label = entry["state_map"][current]
    return {
        "category": category,
        "state": label,
        "state_mean_wait": round(float(entry["means"][current]), 1),
        "n_observations": len(recent_waits_min),
    }
