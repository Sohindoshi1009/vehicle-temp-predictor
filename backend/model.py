import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

TIME_POINTS = list(range(0, 95, 5))   # 0, 5, ..., 90  (19 values)
T_BREAK     = 70                        # phase boundary (minutes)

FEATURE_COLS = [
    "heat_load_kw", "cabin_volume_m3", "pulley_ratio", "solar_w_m2",
    "ac_unit_capacity_kw", "condenser_capacity_kw", "compressor_size_cc",
    "airflow_m3_hr", "soaking_time_hr", "rpm_0_30", "rpm_31_50",
    "rpm_51_70", "rpm_71_90", "ebhs",
]
TEMP_COLS = [f"T_{t}min" for t in TIME_POINTS]
DATA_PATH = Path(__file__).parent / "data" / "vehicles_combined.csv"

# Feature subsets for each dedicated Ridge model
TAU1_FEATS    = ["ac_power_phase1", "airflow_m3_hr", "cabin_volume_m3", "tau_physics"]
TAU2_FEATS    = ["rpm_drop", "ac_power_phase2", "compressor_size_cc"]
T_FINAL_FEATS = ["net_cooling_power", "sealing_quality", "cooling_effectiveness", "ebhs_heat_fraction"]

# All engineered features (used by KNN and RF)
ENG_FEATS_ALL = [
    "ac_power_phase1", "ac_power_phase2",
    "heat_density", "cooling_effectiveness", "rpm_drop",
    "airflow_heat_ratio", "solar_gain",
    "net_cooling_power", "ebhs_heat_fraction", "sealing_quality",
    "infiltration_airflow_ratio", "thermal_mass", "tau_physics",
]
RF_FEATS = ENG_FEATS_ALL + ["time_min"]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _engineer_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ac_power_phase1"]            = df["compressor_size_cc"] * df["pulley_ratio"] * df["rpm_0_30"]  / 1e6
    df["ac_power_phase2"]            = df["compressor_size_cc"] * df["pulley_ratio"] * df["rpm_71_90"] / 1e6
    df["heat_density"]               = df["heat_load_kw"] / df["cabin_volume_m3"]
    df["cooling_effectiveness"]      = df["airflow_m3_hr"] / df["cabin_volume_m3"]
    df["rpm_drop"]                   = df["rpm_51_70"] - df["rpm_71_90"]
    df["airflow_heat_ratio"]         = (df["airflow_m3_hr"] * 1.2 * 1.006 * 10 / 3600) / df["heat_load_kw"]
    df["solar_gain"]                 = df["solar_w_m2"] * df["cabin_volume_m3"] / 1000
    # Physics-correct EBHS features
    df["net_cooling_power"]          = (df["ac_unit_capacity_kw"] - df["heat_load_kw"]
                                        - df["ebhs"] * 0.003 - df["solar_w_m2"] * 0.001)
    df["ebhs_heat_fraction"]         = (df["ebhs"] * 0.003) / df["heat_load_kw"]
    df["sealing_quality"]            = 1.0 / (1.0 + df["ebhs"] / 100.0)
    df["infiltration_airflow_ratio"] = df["ebhs"] / df["airflow_m3_hr"]
    df["thermal_mass"]               = df["cabin_volume_m3"] * 1.2 * 1.006
    df["tau_physics"]                = df["thermal_mass"] / df["net_cooling_power"].clip(lower=0.1)
    return df


def _engineer_single(specs: dict) -> dict:
    d = dict(specs)
    d["ac_power_phase1"]            = d["compressor_size_cc"] * d["pulley_ratio"] * d["rpm_0_30"]  / 1e6
    d["ac_power_phase2"]            = d["compressor_size_cc"] * d["pulley_ratio"] * d["rpm_71_90"] / 1e6
    d["heat_density"]               = d["heat_load_kw"] / d["cabin_volume_m3"]
    d["cooling_effectiveness"]      = d["airflow_m3_hr"] / d["cabin_volume_m3"]
    d["rpm_drop"]                   = d["rpm_51_70"] - d["rpm_71_90"]
    d["airflow_heat_ratio"]         = (d["airflow_m3_hr"] * 1.2 * 1.006 * 10 / 3600) / d["heat_load_kw"]
    d["solar_gain"]                 = d["solar_w_m2"] * d["cabin_volume_m3"] / 1000
    # Physics-correct EBHS features
    d["net_cooling_power"]          = (d["ac_unit_capacity_kw"] - d["heat_load_kw"]
                                       - d["ebhs"] * 0.003 - d["solar_w_m2"] * 0.001)
    d["ebhs_heat_fraction"]         = (d["ebhs"] * 0.003) / d["heat_load_kw"]
    d["sealing_quality"]            = 1.0 / (1.0 + d["ebhs"] / 100.0)
    d["infiltration_airflow_ratio"] = d["ebhs"] / d["airflow_m3_hr"]
    d["thermal_mass"]               = d["cabin_volume_m3"] * 1.2 * 1.006
    d["tau_physics"]                = d["thermal_mass"] / max(0.1, d["net_cooling_power"])
    return d


# ---------------------------------------------------------------------------
# Two-phase physics fitting
# ---------------------------------------------------------------------------

def _fit_two_phase(time_arr: np.ndarray, temps: np.ndarray):
    """Fit two-phase Newton's Law of Cooling; returns (T_final, tau1, tau2)."""
    T_soak = temps[0]

    # ── Phase 1: t = 0 … T_BREAK ─────────────────────────────────────────────
    m1 = time_arr <= T_BREAK
    t1, y1 = time_arr[m1], temps[m1]

    def phase1(t, T_final, tau1):
        return T_final + (T_soak - T_final) * np.exp(-t / tau1)

    try:
        popt1, _ = curve_fit(
            phase1, t1, y1,
            p0=[float(y1[-1]), 10.0],
            bounds=([5.0, 0.5], [60.0, 300.0]),
            maxfev=10000,
        )
        T_final, tau1 = float(popt1[0]), float(popt1[1])
    except Exception:
        T_final, tau1 = float(y1[-1]), 10.0

    T_at_70 = T_final + (T_soak - T_final) * np.exp(-T_BREAK / tau1)

    # ── Phase 2: t = T_BREAK … 90 ────────────────────────────────────────────
    m2 = time_arr >= T_BREAK
    t2, y2 = time_arr[m2], temps[m2]

    def phase2(t, tau2):
        return T_final + (T_at_70 - T_final) * np.exp(-(t - T_BREAK) / tau2)

    try:
        popt2, _ = curve_fit(
            phase2, t2, y2,
            p0=[10.0],
            bounds=([0.5], [300.0]),
            maxfev=10000,
        )
        tau2 = float(popt2[0])
    except Exception:
        tau2 = 10.0

    return T_final, tau1, tau2


# ---------------------------------------------------------------------------
# Data loading & model training  (runs once at import time)
# ---------------------------------------------------------------------------

def _load_and_fit() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    time_arr = np.array(TIME_POINTS, dtype=float)

    T_finals, tau1s, tau2s, T_soaks = [], [], [], []
    for _, row in df.iterrows():
        temps = row[TEMP_COLS].values.astype(float)
        T_final, tau1, tau2 = _fit_two_phase(time_arr, temps)
        T_finals.append(T_final)
        tau1s.append(tau1)
        tau2s.append(tau2)
        T_soaks.append(temps[0])

    df["T_final"] = T_finals
    df["tau1"]    = tau1s
    df["tau2"]    = tau2s
    df["T_soak"]  = T_soaks
    return _engineer_df(df)


def _make_ridge(df: pd.DataFrame, feat_cols: list, target: str):
    sc = StandardScaler()
    X_sc = sc.fit_transform(df[feat_cols].values)
    m = Ridge(alpha=1.0)
    m.fit(X_sc, df[target].values)
    return m, sc


def _build_long_format(df: pd.DataFrame) -> tuple:
    """Reshape to 266 rows (14 vehicles × 19 time steps) for RF training."""
    rows = []
    for _, row in df.iterrows():
        base = [row[f] for f in ENG_FEATS_ALL]
        for t in TIME_POINTS:
            rows.append(base + [float(t), float(row[f"T_{t}min"])])
    arr = np.array(rows, dtype=float)
    return arr[:, :-1], arr[:, -1]   # X, y


def _build_models(df: pd.DataFrame):
    ridge_tau1,    sc_tau1    = _make_ridge(df, TAU1_FEATS,    "tau1")
    ridge_tau2,    sc_tau2    = _make_ridge(df, TAU2_FEATS,    "tau2")
    ridge_T_final, sc_T_final = _make_ridge(df, T_FINAL_FEATS, "T_final")

    # KNN: all engineered features → [tau1, tau2, T_final]
    sc_knn = StandardScaler()
    X_knn  = sc_knn.fit_transform(df[ENG_FEATS_ALL].values)
    Y_knn  = np.column_stack([df["tau1"], df["tau2"], df["T_final"]])
    knn = KNeighborsRegressor(n_neighbors=3)
    knn.fit(X_knn, Y_knn)

    # Random Forest: long-format, direct temperature prediction
    X_rf, y_rf = _build_long_format(df)
    rf = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42)
    rf.fit(X_rf, y_rf)

    return (
        ridge_tau1, sc_tau1,
        ridge_tau2, sc_tau2,
        ridge_T_final, sc_T_final,
        knn, sc_knn,
        rf,
        float(df["T_soak"].mean()),
    )


_df = _load_and_fit()
(
    _ridge_tau1,    _sc_tau1,
    _ridge_tau2,    _sc_tau2,
    _ridge_T_final, _sc_T_final,
    _knn,           _sc_knn,
    _rf,
    _mean_T_soak,
) = _build_models(_df)


# ---------------------------------------------------------------------------
# Two-phase curve reconstruction
# ---------------------------------------------------------------------------

def _reconstruct(tau1: float, tau2: float, T_final: float, T_soak: float) -> list:
    T_at_70 = T_final + (T_soak - T_final) * np.exp(-T_BREAK / tau1)
    temps = []
    for t in TIME_POINTS:
        if t <= T_BREAK:
            T = T_final + (T_soak - T_final) * np.exp(-t / tau1)
        else:
            T = T_final + (T_at_70 - T_final) * np.exp(-(t - T_BREAK) / tau2)
        temps.append(float(T))
    return temps


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_curve(specs_dict: dict, method: str = "physics_ridge"):
    """Return (temperatures: list, tau1: float, tau2: float, T_final: float).

    tau1/tau2 are 0.0 for random_forest (direct temperature prediction).
    """
    eng = _engineer_single(specs_dict)

    if method == "physics_ridge":
        def _pred(model, sc, feats):
            X = np.array([eng[f] for f in feats]).reshape(1, -1)
            return float(model.predict(sc.transform(X))[0])

        tau1    = _pred(_ridge_tau1,    _sc_tau1,    TAU1_FEATS)
        tau2    = _pred(_ridge_tau2,    _sc_tau2,    TAU2_FEATS)
        T_final = _pred(_ridge_T_final, _sc_T_final, T_FINAL_FEATS)

        tau1    = max(0.5, tau1)
        tau2    = max(0.5, tau2)
        T_final = max(5.0, T_final)
        return _reconstruct(tau1, tau2, T_final, _mean_T_soak), tau1, tau2, T_final

    if method == "knn":
        X    = np.array([eng[f] for f in ENG_FEATS_ALL]).reshape(1, -1)
        pred = _knn.predict(_sc_knn.transform(X))[0]
        tau1, tau2, T_final = float(pred[0]), float(pred[1]), float(pred[2])

        tau1    = max(0.5, tau1)
        tau2    = max(0.5, tau2)
        T_final = max(5.0, T_final)
        return _reconstruct(tau1, tau2, T_final, _mean_T_soak), tau1, tau2, T_final

    # random_forest: predict temperature at each time step directly
    base_feats = [eng[f] for f in ENG_FEATS_ALL]
    X_rf = np.array([[*base_feats, float(t)] for t in TIME_POINTS], dtype=float)
    temps = _rf.predict(X_rf).tolist()
    T_final = float(min(temps))
    return temps, 0.0, 0.0, T_final


def get_all_vehicle_curves() -> list:
    result = []
    for _, row in _df.iterrows():
        result.append({
            "vehicle":      row["vehicle"],
            "features":     {c: row[c] for c in FEATURE_COLS},
            "time_points":  TIME_POINTS,
            "temperatures": [float(row[c]) for c in TEMP_COLS],
            "tau1":         round(row["tau1"],    3),
            "tau2":         round(row["tau2"],    3),
            "T_final":      round(row["T_final"], 3),
        })
    return result
