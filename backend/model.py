import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

TIME_POINTS = list(range(0, 95, 5))  # 0, 5, 10, ..., 90  (19 values)
FEATURE_COLS = [
    "heat_load_kw", "cabin_volume_m3", "pulley_ratio", "solar_w_m2",
    "ac_unit_capacity_kw", "condenser_capacity_kw", "compressor_size_cc",
    "airflow_m3_hr", "soaking_time_hr", "rpm_0_30", "rpm_31_50",
    "rpm_51_70", "rpm_71_90", "ebhs",
]
TEMP_COLS = [f"T_{t}min" for t in TIME_POINTS]
DATA_PATH = Path(__file__).parent / "data" / "vehicles_combined.csv"


# ---------------------------------------------------------------------------
# Physics model
# ---------------------------------------------------------------------------

def _make_cooling_model(T_soak: float):
    """Factory that closes over T_soak so curve_fit gets a 2-param callable."""
    def model(t, T_final, tau):
        return T_final + (T_soak - T_final) * np.exp(-t / tau)
    return model


def _fit_vehicle(time_arr: np.ndarray, temps: np.ndarray):
    """Fit Newton's Law of Cooling; returns (T_final, tau)."""
    T_soak = temps[0]
    model = _make_cooling_model(T_soak)
    p0 = [float(temps[-1]), 15.0]
    bounds = ([5.0, 0.5], [60.0, 300.0])
    try:
        popt, _ = curve_fit(model, time_arr, temps, p0=p0, bounds=bounds, maxfev=10000)
        return float(popt[0]), float(popt[1])
    except Exception:
        return float(temps[-1]), 15.0


# ---------------------------------------------------------------------------
# Data loading & model training  (runs once at import time)
# ---------------------------------------------------------------------------

def _load_and_fit() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    time_arr = np.array(TIME_POINTS, dtype=float)

    taus, T_finals, T_soaks = [], [], []
    for _, row in df.iterrows():
        temps = row[TEMP_COLS].values.astype(float)
        T_final, tau = _fit_vehicle(time_arr, temps)
        taus.append(tau)
        T_finals.append(T_final)
        T_soaks.append(temps[0])

    df["tau"] = taus
    df["T_final"] = T_finals
    df["T_soak"] = T_soaks
    return df


def _build_models(df: pd.DataFrame):
    X = df[FEATURE_COLS].values
    # Predict both tau and T_final together (multi-output)
    Y = np.column_stack([df["tau"].values, df["T_final"].values])

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_sc, Y)

    knn = KNeighborsRegressor(n_neighbors=3)
    knn.fit(X_sc, Y)

    mean_T_soak = float(df["T_soak"].mean())
    return scaler, ridge, knn, mean_T_soak


_df = _load_and_fit()
_scaler, _ridge, _knn, _mean_T_soak = _build_models(_df)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_curve(specs_dict: dict, method: str = "physics_ridge"):
    """
    Return (temperatures: list[float], tau: float, T_final: float)
    for t = 0, 5, 10, …, 90 min (19 values).
    """
    X = np.array([specs_dict[c] for c in FEATURE_COLS], dtype=float).reshape(1, -1)
    X_sc = _scaler.transform(X)

    if method == "physics_ridge":
        pred = _ridge.predict(X_sc)[0]
    else:
        pred = _knn.predict(X_sc)[0]

    tau = max(0.5, float(pred[0]))
    T_final = max(5.0, float(pred[1]))

    t_arr = np.array(TIME_POINTS, dtype=float)
    model = _make_cooling_model(_mean_T_soak)
    temps = model(t_arr, T_final, tau).tolist()
    return temps, tau, T_final


def get_all_vehicle_curves() -> list:
    """Return every training vehicle with its actual curve and fitted params."""
    result = []
    for _, row in _df.iterrows():
        result.append({
            "vehicle": row["vehicle"],
            "features": {c: row[c] for c in FEATURE_COLS},
            "time_points": TIME_POINTS,
            "temperatures": [float(row[c]) for c in TEMP_COLS],
            "tau": round(row["tau"], 3),
            "T_final": round(row["T_final"], 3),
        })
    return result
