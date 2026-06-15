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
T_BREAK     = 70                        # legacy phase boundary (minutes)

FEATURE_COLS = [
    "heat_load_kw", "cabin_volume_m3", "pulley_ratio", "solar_w_m2",
    "ac_unit_capacity_kw", "condenser_capacity_kw", "compressor_size_cc",
    "airflow_m3_hr", "soaking_time_hr", "rpm_0_30", "rpm_31_50",
    "rpm_51_70", "rpm_71_90", "ebhs",
]
TEMP_COLS = [f"T_{t}min" for t in TIME_POINTS]
DATA_PATH = Path(__file__).parent / "data" / "vehicles_combined.csv"

# ---------------------------------------------------------------------------
# Per-segment Ridge feature sets — STRICT ISOLATION
# Each set uses only the AC power feature from its own RPM band.
# No cross-band features allowed.
# ---------------------------------------------------------------------------
SEG1_FEATS   = ["ac_power_0_30",  "airflow_m3_hr", "cabin_volume_m3",
                 "thermal_mass",   "net_cooling_power", "sealing_quality"]
SEG2_FEATS   = ["ac_power_31_50", "airflow_m3_hr", "cabin_volume_m3", "thermal_mass"]
SEG3_FEATS   = ["ac_power_51_70", "airflow_m3_hr", "cabin_volume_m3", "thermal_mass"]
SEG4_FEATS   = ["ac_power_71_90", "rpm_drop",      "compressor_size_cc"]
TFINAL_FEATS = ["net_cooling_power", "sealing_quality", "heat_load_fraction",
                "ac_per_volume", "net_cooling_per_volume"]

# All engineered features (KNN + RF — no isolation constraint)
ENG_FEATS_ALL = [
    "ac_power_phase1", "ac_power_phase2",
    "heat_density", "cooling_effectiveness", "rpm_drop",
    "airflow_heat_ratio", "solar_gain",
    "net_cooling_power", "ebhs_heat_fraction", "heat_load_fraction",
    "ac_per_volume", "net_cooling_per_volume", "heat_balance_ratio",
    "sealing_quality", "infiltration_airflow_ratio", "thermal_mass", "tau_physics",
    "ac_power_0_30", "ac_power_31_50", "ac_power_51_70", "ac_power_71_90",
]
RF_FEATS = ENG_FEATS_ALL + ["time_min"]

_IDX = {t: i for i, t in enumerate(TIME_POINTS)}


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _engineer_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ac_power_phase1"]            = df["compressor_size_cc"] * df["pulley_ratio"] * df["rpm_0_30"]  / 1e6
    df["ac_power_phase2"]            = df["compressor_size_cc"] * df["pulley_ratio"] * df["rpm_71_90"] / 1e6
    df["ac_power_0_30"]              = df["compressor_size_cc"] * df["pulley_ratio"] * df["rpm_0_30"]  / 1e6
    df["ac_power_31_50"]             = df["compressor_size_cc"] * df["pulley_ratio"] * df["rpm_31_50"] / 1e6
    df["ac_power_51_70"]             = df["compressor_size_cc"] * df["pulley_ratio"] * df["rpm_51_70"] / 1e6
    df["ac_power_71_90"]             = df["compressor_size_cc"] * df["pulley_ratio"] * df["rpm_71_90"] / 1e6
    df["heat_density"]               = df["heat_load_kw"] / df["cabin_volume_m3"]
    df["cooling_effectiveness"]      = df["airflow_m3_hr"] / df["cabin_volume_m3"]
    df["rpm_drop"]                   = df["rpm_51_70"] - df["rpm_71_90"]
    df["airflow_heat_ratio"]         = (df["airflow_m3_hr"] * 1.2 * 1.006 * 10 / 3600) / df["heat_load_kw"]
    df["solar_gain"]                 = df["solar_w_m2"] * df["cabin_volume_m3"] / 1000
    df["net_cooling_power"]          = (df["ac_unit_capacity_kw"] - df["heat_load_kw"]
                                        - df["ebhs"] * 0.003 - df["solar_w_m2"] * 0.001)
    df["ebhs_heat_fraction"]         = df["ebhs"] * 0.003
    df["heat_load_fraction"]         = df["heat_load_kw"] / df["ac_unit_capacity_kw"]
    df["ac_per_volume"]              = df["ac_unit_capacity_kw"] / df["cabin_volume_m3"]
    df["net_cooling_per_volume"]     = df["net_cooling_power"] / df["cabin_volume_m3"]
    df["heat_balance_ratio"]         = (df["heat_load_kw"] + df["ebhs"] * 0.003 + df["solar_w_m2"] * 0.001) / df["ac_unit_capacity_kw"]
    df["sealing_quality"]            = 1.0 / (1.0 + df["ebhs"] / 100.0)
    df["infiltration_airflow_ratio"] = df["ebhs"] / df["airflow_m3_hr"]
    df["thermal_mass"]               = df["cabin_volume_m3"] * 1.2 * 1.006
    df["tau_physics"]                = df["thermal_mass"] / df["net_cooling_power"].clip(lower=0.1)
    return df


def _engineer_single(specs: dict) -> dict:
    d = dict(specs)
    d["ac_power_phase1"]            = d["compressor_size_cc"] * d["pulley_ratio"] * d["rpm_0_30"]  / 1e6
    d["ac_power_phase2"]            = d["compressor_size_cc"] * d["pulley_ratio"] * d["rpm_71_90"] / 1e6
    d["ac_power_0_30"]              = d["compressor_size_cc"] * d["pulley_ratio"] * d["rpm_0_30"]  / 1e6
    d["ac_power_31_50"]             = d["compressor_size_cc"] * d["pulley_ratio"] * d["rpm_31_50"] / 1e6
    d["ac_power_51_70"]             = d["compressor_size_cc"] * d["pulley_ratio"] * d["rpm_51_70"] / 1e6
    d["ac_power_71_90"]             = d["compressor_size_cc"] * d["pulley_ratio"] * d["rpm_71_90"] / 1e6
    d["heat_density"]               = d["heat_load_kw"] / d["cabin_volume_m3"]
    d["cooling_effectiveness"]      = d["airflow_m3_hr"] / d["cabin_volume_m3"]
    d["rpm_drop"]                   = d["rpm_51_70"] - d["rpm_71_90"]
    d["airflow_heat_ratio"]         = (d["airflow_m3_hr"] * 1.2 * 1.006 * 10 / 3600) / d["heat_load_kw"]
    d["solar_gain"]                 = d["solar_w_m2"] * d["cabin_volume_m3"] / 1000
    d["net_cooling_power"]          = (d["ac_unit_capacity_kw"] - d["heat_load_kw"]
                                       - d["ebhs"] * 0.003 - d["solar_w_m2"] * 0.001)
    d["ebhs_heat_fraction"]         = d["ebhs"] * 0.003
    d["heat_load_fraction"]         = d["heat_load_kw"] / d["ac_unit_capacity_kw"]
    d["ac_per_volume"]              = d["ac_unit_capacity_kw"] / d["cabin_volume_m3"]
    d["net_cooling_per_volume"]     = d["net_cooling_power"] / d["cabin_volume_m3"]
    d["heat_balance_ratio"]         = (d["heat_load_kw"] + d["ebhs"] * 0.003 + d["solar_w_m2"] * 0.001) / d["ac_unit_capacity_kw"]
    d["sealing_quality"]            = 1.0 / (1.0 + d["ebhs"] / 100.0)
    d["infiltration_airflow_ratio"] = d["ebhs"] / d["airflow_m3_hr"]
    d["thermal_mass"]               = d["cabin_volume_m3"] * 1.2 * 1.006
    d["tau_physics"]                = d["thermal_mass"] / max(0.1, d["net_cooling_power"])
    return d


# ---------------------------------------------------------------------------
# Two-phase joint fit (for T_final; used as anchor for per-segment fitting)
# ---------------------------------------------------------------------------

def _fit_two_phase(time_arr: np.ndarray, temps: np.ndarray):
    """Joint fit for T_final, tau1, tau2 with tight physical bounds."""
    T_soak = temps[0]

    def model(t, tau1, T_final, tau2):
        T_at_70 = T_final + (T_soak - T_final) * np.exp(-T_BREAK / tau1)
        return np.where(
            t <= T_BREAK,
            T_final + (T_soak - T_final) * np.exp(-t / tau1),
            T_final + (T_at_70 - T_final) * np.exp(-(t - T_BREAK) / tau2),
        )

    try:
        popt, _ = curve_fit(
            model, time_arr, temps,
            p0=[5.0, 28.0, 8.0],
            bounds=([0.5, 20.0, 0.5], [20.0, 70.0, 40.0]),
            maxfev=10000,
        )
        tau1, T_final, tau2 = float(popt[0]), float(popt[1]), float(popt[2])
    except Exception:
        tau1, T_final, tau2 = 5.0, float(temps[-1]), 8.0

    return T_final, tau1, tau2


# ---------------------------------------------------------------------------
# Per-segment tau fitting
# ---------------------------------------------------------------------------

def _fit_segments(time_arr: np.ndarray, temps: np.ndarray, T_final: float):
    """Fit one tau per segment using only the temperatures in that window.

    Each tau is fitted so that T(t) = T_final + (T_start - T_final)*exp(-dt/tau)
    matches the observed temperatures in the window. Higher AC power → faster
    cooling → smaller tau (negative Ridge coefficient for ac_power features).
    """

    def _fit_one(t_arr, y_arr, T_ref, t_start, default):
        if len(y_arr) < 2 or abs(T_ref - T_final) < 0.05:
            return default

        def seg(t, tau):
            return T_final + (T_ref - T_final) * np.exp(-(t - t_start) / max(tau, 0.001))

        try:
            p, _ = curve_fit(seg, t_arr, y_arr, p0=[default],
                             bounds=([0.5], [60.0]), maxfev=5000)
            return float(np.clip(p[0], 0.5, 60.0))
        except Exception:
            return default

    T_soak = temps[0]

    # Segment 1: t = 0 … 30
    m1     = time_arr <= 30
    tau_s1 = _fit_one(time_arr[m1], temps[m1], T_soak, 0, 5.0)

    T_30 = float(temps[_IDX[30]])

    # Segment 2: t = 30 … 50
    m2     = (time_arr >= 30) & (time_arr <= 50)
    tau_s2 = _fit_one(time_arr[m2], temps[m2], T_30, 30, 8.0)

    T_50 = float(temps[_IDX[50]])

    # Segment 3: t = 50 … 70
    m3     = (time_arr >= 50) & (time_arr <= 70)
    tau_s3 = _fit_one(time_arr[m3], temps[m3], T_50, 50, 8.0)

    T_70 = float(temps[_IDX[70]])

    # Segment 4: t = 70 … 90
    m4     = time_arr >= 70
    tau_s4 = _fit_one(time_arr[m4], temps[m4], T_70, 70, 15.0)

    return tau_s1, tau_s2, tau_s3, tau_s4


# ---------------------------------------------------------------------------
# Data loading & model training  (runs once at import time)
# ---------------------------------------------------------------------------

def _load_and_fit() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    time_arr = np.array(TIME_POINTS, dtype=float)

    T_finals = []; tau1s = []; tau2s = []; T_soaks = []
    tau_s1s  = []; tau_s2s = []; tau_s3s = []; tau_s4s = []

    for _, row in df.iterrows():
        temps               = row[TEMP_COLS].values.astype(float)
        T_final, tau1, tau2 = _fit_two_phase(time_arr, temps)
        ts1, ts2, ts3, ts4  = _fit_segments(time_arr, temps, T_final)
        T_finals.append(T_final);  tau1s.append(tau1);  tau2s.append(tau2)
        T_soaks.append(temps[0])
        tau_s1s.append(ts1); tau_s2s.append(ts2)
        tau_s3s.append(ts3); tau_s4s.append(ts4)

    df["T_final"] = T_finals
    df["tau1"]    = tau1s
    df["tau2"]    = tau2s
    df["T_soak"]  = T_soaks
    df["tau_s1"]  = tau_s1s
    df["tau_s2"]  = tau_s2s
    df["tau_s3"]  = tau_s3s
    df["tau_s4"]  = tau_s4s

    print("\nFITTED PARAMETERS PER VEHICLE:")
    for name, tf, t1, t2, ts, s1, s2, s3, s4 in zip(
        df["vehicle"].values, T_finals, tau1s, tau2s, T_soaks,
        tau_s1s, tau_s2s, tau_s3s, tau_s4s,
    ):
        print(f"  {name:<20s}  T_soak={ts:.1f}°C  T_final={tf:.1f}°C  "
              f"tau1={t1:.2f}  tau2={t2:.2f}  "
              f"s1={s1:.2f}  s2={s2:.2f}  s3={s3:.2f}  s4={s4:.2f}")

    return _engineer_df(df)


def _make_ridge(df: pd.DataFrame, feat_cols: list, target: str):
    sc  = StandardScaler()
    Xsc = sc.fit_transform(df[feat_cols].values)
    m   = Ridge(alpha=1.0)
    m.fit(Xsc, df[target].values)
    return m, sc


def _build_long_format(df: pd.DataFrame) -> tuple:
    rows = []
    for _, row in df.iterrows():
        base = [row[f] for f in ENG_FEATS_ALL]
        for t in TIME_POINTS:
            rows.append(base + [float(t), float(row[f"T_{t}min"])])
    arr = np.array(rows, dtype=float)
    return arr[:, :-1], arr[:, -1]


def _build_models(df: pd.DataFrame):
    # Strictly isolated per-segment Ridge models
    model_A, sc_A   = _make_ridge(df, SEG1_FEATS,   "tau_s1")
    model_B, sc_B   = _make_ridge(df, SEG2_FEATS,   "tau_s2")
    model_C, sc_C   = _make_ridge(df, SEG3_FEATS,   "tau_s3")
    model_D, sc_D   = _make_ridge(df, SEG4_FEATS,   "tau_s4")
    model_Tf, sc_Tf = _make_ridge(df, TFINAL_FEATS, "T_final")

    # Print coefficient signs — ac_power features must be NEGATIVE (higher power → lower tau)
    print("\nSEGMENT MODEL COEFFICIENT SIGNS (ac_power must be -):")
    for label, m, feats in [
        ("SEG1->tau_s1", model_A, SEG1_FEATS),
        ("SEG2->tau_s2", model_B, SEG2_FEATS),
        ("SEG3->tau_s3", model_C, SEG3_FEATS),
        ("SEG4->tau_s4", model_D, SEG4_FEATS),
    ]:
        signs = {f: (f"{c:+.3f}") for f, c in zip(feats, m.coef_)}
        print(f"  {label}: {signs}")

    # KNN: all engineered features → [tau_s1, tau_s2, tau_s3, tau_s4, T_final]
    sc_knn = StandardScaler()
    X_knn  = sc_knn.fit_transform(df[ENG_FEATS_ALL].values)
    Y_knn  = np.column_stack([df["tau_s1"], df["tau_s2"], df["tau_s3"],
                               df["tau_s4"], df["T_final"]])
    knn = KNeighborsRegressor(n_neighbors=3)
    knn.fit(X_knn, Y_knn)

    # Random Forest: long-format direct temperature prediction
    X_rf, y_rf = _build_long_format(df)
    rf = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42)
    rf.fit(X_rf, y_rf)

    mean_T_final      = float(df["T_final"].mean())
    mean_heat_balance = float(df["heat_balance_ratio"].mean())

    return (
        model_A, sc_A,
        model_B, sc_B,
        model_C, sc_C,
        model_D, sc_D,
        model_Tf, sc_Tf,
        knn, sc_knn,
        rf,
        float(df["T_soak"].mean()),
        mean_T_final,
        mean_heat_balance,
    )


_df = _load_and_fit()
(
    _model_A,  _sc_A,
    _model_B,  _sc_B,
    _model_C,  _sc_C,
    _model_D,  _sc_D,
    _model_Tf, _sc_Tf,
    _knn,      _sc_knn,
    _rf,
    _mean_T_soak,
    _mean_T_final,
    _mean_heat_balance,
) = _build_models(_df)


# ---------------------------------------------------------------------------
# Curve reconstruction
# ---------------------------------------------------------------------------

def _reconstruct_segmented(tau_s1: float, tau_s2: float,
                            tau_s3: float, tau_s4: float,
                            T_final: float, T_soak: float) -> list:
    """4-segment Newton cooling; each segment uses its own independently fitted tau."""
    T_30 = T_final + (T_soak - T_final) * np.exp(-30 / tau_s1)
    T_50 = T_final + (T_30   - T_final) * np.exp(-20 / tau_s2)
    T_70 = T_final + (T_50   - T_final) * np.exp(-20 / tau_s3)

    temps = []
    for t in TIME_POINTS:
        if t <= 30:
            T = T_final + (T_soak - T_final) * np.exp(-t / tau_s1)
        elif t <= 50:
            T = T_final + (T_30   - T_final) * np.exp(-(t - 30) / tau_s2)
        elif t <= 70:
            T = T_final + (T_50   - T_final) * np.exp(-(t - 50) / tau_s3)
        else:
            T = T_final + (T_70   - T_final) * np.exp(-(t - 70) / tau_s4)
        temps.append(float(T))
    return temps


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_curve(specs_dict: dict, method: str = "physics_ridge"):
    """Return (temperatures, tau1, tau2, T_final).

    tau1 = segment-1 tau, tau2 = segment-4 tau (for display).
    Both 0.0 for random_forest.
    """
    eng = _engineer_single(specs_dict)

    if method == "physics_ridge":
        def _pred(model, sc, feats):
            X = np.array([eng[f] for f in feats]).reshape(1, -1)
            return float(model.predict(sc.transform(X))[0])

        tau_s1  = max(0.5, _pred(_model_A, _sc_A, SEG1_FEATS))
        tau_s2  = max(0.5, _pred(_model_B, _sc_B, SEG2_FEATS))
        tau_s3  = max(0.5, _pred(_model_C, _sc_C, SEG3_FEATS))
        tau_s4  = max(0.5, _pred(_model_D, _sc_D, SEG4_FEATS))
        T_final = max(5.0, _mean_T_final * (eng["heat_balance_ratio"] / _mean_heat_balance))

        curve = _reconstruct_segmented(tau_s1, tau_s2, tau_s3, tau_s4, T_final, _mean_T_soak)
        return curve, tau_s1, tau_s4, T_final

    if method == "knn":
        X    = np.array([eng[f] for f in ENG_FEATS_ALL]).reshape(1, -1)
        pred = _knn.predict(_sc_knn.transform(X))[0]
        tau_s1  = max(0.5, float(pred[0]))
        tau_s2  = max(0.5, float(pred[1]))
        tau_s3  = max(0.5, float(pred[2]))
        tau_s4  = max(0.5, float(pred[3]))
        T_final = max(5.0, float(pred[4]))

        curve = _reconstruct_segmented(tau_s1, tau_s2, tau_s3, tau_s4, T_final, _mean_T_soak)
        return curve, tau_s1, tau_s4, T_final

    # random_forest: direct temperature prediction
    base_feats = [eng[f] for f in ENG_FEATS_ALL]
    X_rf  = np.array([[*base_feats, float(t)] for t in TIME_POINTS], dtype=float)
    temps = _rf.predict(X_rf).tolist()
    return temps, 0.0, 0.0, float(min(temps))


def get_feature_importances() -> dict:
    def _ridge_importance(model, feat_names: list) -> dict:
        coefs     = model.coef_
        abs_coefs = np.abs(coefs)
        norm      = abs_coefs / abs_coefs.max() if abs_coefs.max() > 0 else abs_coefs
        return {
            f: {"importance": float(imp), "sign": int(np.sign(c))}
            for f, imp, c in zip(feat_names, norm, coefs)
        }

    ridge_tau1   = _ridge_importance(_model_A,  SEG1_FEATS)
    ridge_tfinal = _ridge_importance(_model_Tf, TFINAL_FEATS)

    rf_imps = _rf.feature_importances_
    rf_norm = rf_imps / rf_imps.max() if rf_imps.max() > 0 else rf_imps
    rf_signs = [
        int(np.sign(np.corrcoef(_df[f].values, _df["T_final"].values)[0, 1]))
        for f in ENG_FEATS_ALL
    ] + [-1]
    random_forest = {
        f: {"importance": float(imp), "sign": sign}
        for f, imp, sign in zip(RF_FEATS, rf_norm, rf_signs)
    }

    return {
        "ridge_tau1":    ridge_tau1,
        "ridge_tfinal":  ridge_tfinal,
        "random_forest": random_forest,
    }


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


# ---------------------------------------------------------------------------
# Validation: physical relationships
# ---------------------------------------------------------------------------

def _validate_physics() -> None:
    BASE = {
        "heat_load_kw": 4.8, "cabin_volume_m3": 3.1, "pulley_ratio": 1.5,
        "solar_w_m2": 1200.0, "ac_unit_capacity_kw": 4.4, "condenser_capacity_kw": 9.0,
        "compressor_size_cc": 130.0, "airflow_m3_hr": 550.0, "soaking_time_hr": 1.0,
        "rpm_0_30": 1600.0, "rpm_31_50": 1700.0, "rpm_51_70": 1800.0,
        "rpm_71_90": 750.0, "ebhs": 100.0,
    }

    def _vary(**kw):
        s = dict(BASE); s.update(kw); return s

    def _run(specs):
        _, tau1, _, T_final = predict_curve(specs, method="physics_ridge")
        return tau1, T_final

    checks = [
        ("Higher EBHS increases T_final",
         _run(_vary(ebhs=70))[1],   _run(_vary(ebhs=190))[1],   True,  "degC"),
        ("Higher airflow reduces tau1",
         _run(_vary(airflow_m3_hr=449))[0], _run(_vary(airflow_m3_hr=641))[0], False, " min"),
        ("Higher heat_load increases T_final",
         _run(_vary(heat_load_kw=3.6))[1], _run(_vary(heat_load_kw=5.5))[1],  True,  "degC"),
        ("Higher ac_unit_capacity reduces T_final",
         _run(_vary(ac_unit_capacity_kw=4.4))[1], _run(_vary(ac_unit_capacity_kw=5.4))[1], False, "degC"),
    ]

    print("\nVALIDATION RESULTS:")
    n_pass = 0
    for label, v_lo, v_hi, should_increase, unit in checks:
        passed = (v_hi > v_lo) if should_increase else (v_hi < v_lo)
        n_pass += passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}: {v_lo:.1f} -> {v_hi:.1f}{unit}")
    print(f"  {n_pass}/{len(checks)} checks passed\n")


# ---------------------------------------------------------------------------
# Validation: segment isolation
# ---------------------------------------------------------------------------

def _validate_segments() -> None:
    """Verify that varying RPM in band N only affects temperatures in window N."""
    BASE = {
        "heat_load_kw": 4.8, "cabin_volume_m3": 3.1, "pulley_ratio": 1.5,
        "solar_w_m2": 1200.0, "ac_unit_capacity_kw": 4.4, "condenser_capacity_kw": 9.0,
        "compressor_size_cc": 130.0, "airflow_m3_hr": 550.0, "soaking_time_hr": 1.0,
        "rpm_0_30": 1600.0, "rpm_31_50": 1700.0, "rpm_51_70": 1800.0,
        "rpm_71_90": 750.0, "ebhs": 100.0,
    }
    TOL = 0.01  # °C — threshold for "effectively constant"

    def _vary(**kw):
        s = dict(BASE); s.update(kw); return s

    def _temps(specs):
        return predict_curve(specs, method="physics_ridge")[0]

    print("\nSEGMENT ISOLATION TESTS:")
    n_pass = 0
    total  = 0

    # ── Test A: vary rpm_51_70 only ──────────────────────────────────────────
    # Expected: T@5, T@15, T@30 constant; T@55, T@65 vary
    vals = np.linspace(1400, 2230, 5)
    curves = [_temps(_vary(rpm_51_70=r)) for r in vals]
    checks_A = [
        (5,  "constant", False),
        (15, "constant", False),
        (30, "constant", False),
        (55, "vary",     True),
        (65, "vary",     True),
    ]
    for tmin, label, should_vary in checks_A:
        bucket = [c[_IDX[tmin]] for c in curves]
        spread = max(bucket) - min(bucket)
        passed = (spread > TOL) if should_vary else (spread < TOL)
        n_pass += passed; total += 1
        tag = "PASS" if passed else "FAIL"
        note = ""
        if not passed and not should_vary:
            note = "  ← chain propagation from upstream segment"
        print(f"  [{tag}] rpm_51_70 sweep — T@{tmin:2d}min must be {label:8s}: "
              f"spread={spread:.4f}°C{note}")

    # ── Test B: vary rpm_0_30 only ───────────────────────────────────────────
    # Expected: T@5, T@15 vary; T@35, T@55, T@75 constant
    vals = np.linspace(1000, 2500, 5)
    curves = [_temps(_vary(rpm_0_30=r)) for r in vals]
    checks_B = [
        (5,  "vary",     True),
        (15, "vary",     True),
        (35, "constant", False),
        (55, "constant", False),
        (75, "constant", False),
    ]
    for tmin, label, should_vary in checks_B:
        bucket = [c[_IDX[tmin]] for c in curves]
        spread = max(bucket) - min(bucket)
        passed = (spread > TOL) if should_vary else (spread < TOL)
        n_pass += passed; total += 1
        tag = "PASS" if passed else "FAIL"
        note = ""
        if not passed and not should_vary:
            note = "  ← chain propagation from upstream segment"
        print(f"  [{tag}] rpm_0_30  sweep — T@{tmin:2d}min must be {label:8s}: "
              f"spread={spread:.4f}°C{note}")

    print(f"  {n_pass}/{total} isolation checks passed\n")


_validate_physics()
_validate_segments()
