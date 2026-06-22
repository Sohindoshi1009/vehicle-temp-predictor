import os

import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

FEATURE_CONFIG = {
    "heat_load_kw":           ("Heat Load (kW)",            3.0,   7.0,   5.0,   0.1),
    "cabin_volume_m3":        ("Cabin Volume (m³)",          2.0,   5.0,   3.0,   0.1),
    "pulley_ratio":           ("Pulley Ratio",               1.0,   2.0,   1.5,   0.05),
    "solar_w_m2":             ("Solar Radiation (W/m²)",     800.,  1500., 1200., 50.),
    "ac_unit_capacity_kw":    ("AC Capacity (kW)",           3.0,   6.0,   4.4,   0.1),
    "condenser_capacity_kw":  ("Condenser Capacity (kW)",    7.0,   12.0,  9.0,   0.5),
    "compressor_size_cc":     ("Compressor Size (cc)",       100.,  200.,  140.,  5.),
    "airflow_m3_hr":          ("Airflow (m³/hr)",            400.,  700.,  550.,  10.),
    "soaking_time_hr":        ("Soaking Time (hr)",          0.5,   3.0,   1.0,   0.5),
    "rpm_0_30":               ("Engine RPM (0–30 min)",       1000., 2500., 1600., 50.),
    "rpm_31_50":              ("Engine RPM (31–50 min)",      1000., 2500., 1700., 50.),
    "rpm_51_70":              ("Engine RPM (51–70 min)",      1000., 2500., 1800., 50.),
    "rpm_71_90":              ("Engine RPM (71–90 min)",      500.,  1200., 750.,  50.),
    "ebhs":                   ("EBHS",                       60.,   200.,  100.,  5.),
}

COLORS = {
    "physics_ridge": "#E63946",
    "knn":           "#2196F3",
    "random_forest": "#4CAF50",
    "ode_solver":    "#9C27B0",
}

METHOD_LABELS = {
    "physics_ridge": "Physics + Ridge",
    "knn":           "KNN",
    "random_forest": "Random Forest",
    "ode_solver":    "ODE Solver",
}

FEATURE_EXPLANATIONS = {
    "ac_power_phase1":            "Compressor power at low speed (cc × pulley × RPM 0–30). Higher power drives faster phase-1 cooling.",
    "ac_power_phase2":            "Compressor power at highway speed (cc × pulley × RPM 71–90). Sustains cooling in phase 2.",
    "heat_density":               "Heat load per m³ of cabin. Compact, crowded cabins have higher heat density.",
    "cooling_effectiveness":      "Airflow per m³ of cabin. Higher values improve convective heat removal per unit volume.",
    "rpm_drop":                   "RPM fall from 51–70 min to 71–90 min. Steep drop reduces compressor speed in the later stage.",
    "airflow_heat_ratio":         "Convective capacity of airflow vs heat load. Higher = airflow can keep pace with heating.",
    "solar_gain":                 "Solar heat absorbed (W/m² × cabin volume). Larger, sunnier cabins absorb more radiant heat.",
    "net_cooling_power":          "AC capacity minus all heat loads (passengers + EBHS + solar). Positive = cooling wins.",
    "ebhs_heat_fraction":         "EBHS infiltration heat (EBHS × 0.003 kW). Quantifies hot-air ingress through gaps and seals.",
    "heat_load_fraction":         "Fraction of AC capacity consumed by the passenger/engine heat load. Higher = less cooling margin.",
    "ac_per_volume":              "AC power per m³ of cabin. True cooling intensity — equalises AC size against cabin size.",
    "net_cooling_per_volume":     "Net cooling surplus per m³. Higher values drive the cabin toward ambient temperature faster.",
    "heat_balance_ratio":         "Total heat input ÷ AC capacity. Values above 1.0 mean heat exceeds rated AC output.",
    "sealing_quality":            "Cabin seal factor = 1 / (1 + EBHS/100). Higher = better sealed; poor sealing lets hot air in.",
    "infiltration_airflow_ratio": "EBHS infiltration relative to airflow. High values dilute cool air with hot outside air.",
    "thermal_mass":               "Cabin air thermal mass (kg × Cp). Higher mass means slower temperature swings.",
    "tau_physics":                "Physics time constant = thermal mass ÷ net cooling power. Direct predictor of cooling speed.",
    "airflow_m3_hr":              "Cabin airflow rate. More airflow removes heat faster, cutting the phase-1 time constant.",
    "cabin_volume_m3":            "Cabin volume. Larger cabins have more thermal mass and take longer to cool.",
    "time_min":                   "Time elapsed (minutes). Temperature falls as time progresses during the cool-down cycle.",
    "compressor_size_cc":         "Compressor displacement. Larger compressors move more refrigerant per revolution.",
}


@st.cache_data(show_spinner=False)
def fetch_training_vehicles():
    try:
        resp = requests.get(f"{API_URL}/vehicles", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return []


@st.cache_data(show_spinner=False)
def fetch_feature_importance():
    try:
        resp = requests.get(f"{API_URL}/feature-importance", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def call_predict(specs: dict, method: str):
    try:
        resp = requests.post(
            f"{API_URL}/predict",
            json=specs,
            params={"method": method},
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json(), None
        return None, f"API error {resp.status_code}: {resp.text}"
    except Exception as e:
        return None, f"Cannot reach API at {API_URL}. Start the backend first.\n{e}"


def _importance_chart(model_data: dict, title: str) -> go.Figure:
    items = sorted(model_data.items(), key=lambda x: x[1]["importance"])
    features    = [it[0] for it in items]
    importances = [it[1]["importance"] for it in items]
    signs       = [it[1]["sign"] for it in items]
    colors      = ["#4CAF50" if s < 0 else "#E63946" for s in signs]

    fig = go.Figure(go.Bar(
        x=importances,
        y=features,
        orientation="h",
        marker_color=colors,
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Normalized Importance (0–1)",
        height=420,
        margin=dict(l=10, r=10, t=50, b=40),
    )
    return fig


def _top3_explanations(model_data: dict, model_type: str):
    top3 = sorted(model_data.items(), key=lambda x: -x[1]["importance"])[:3]
    for rank, (feat, vals) in enumerate(top3, 1):
        helpful = vals["sign"] < 0
        if model_type == "tau1":
            direction = "reduces τ₁ → faster initial cooling" if helpful else "increases τ₁ → slower initial cooling"
        elif model_type == "tfinal":
            direction = "lowers T_final → cooler equilibrium" if helpful else "raises T_final → warmer equilibrium"
        else:
            direction = "associated with lower cabin temp" if helpful else "associated with higher cabin temp"
        explanation = FEATURE_EXPLANATIONS.get(feat, feat)
        st.markdown(f"**{rank}. `{feat}`** — {direction}  \n{explanation}")
        if rank < 3:
            st.markdown("")


def main():
    st.set_page_config(
        page_title="Vehicle Cabin Temp Predictor",
        page_icon="🚗",
        layout="wide",
    )
    st.title("Vehicle Cabin Temperature Predictor")
    st.caption(
        "Physics-based Newton's Law of Cooling + ML regression to predict "
        "cabin cool-down over 90 minutes."
    )

    st.sidebar.header("Vehicle Specifications")
    specs = {}
    for feat, (label, mn, mx, default, step) in FEATURE_CONFIG.items():
        specs[feat] = st.sidebar.slider(label, min_value=mn, max_value=mx, value=default, step=step)

    st.sidebar.divider()
    st.sidebar.header("Comparison")
    show_training = st.sidebar.checkbox("Overlay training vehicle curves", value=False)

    tab1, tab2, tab3 = st.tabs(["Prediction", "Feature Importance", "Sensitivity Analysis"])

    # ── Tab 1: Prediction ──────────────────────────────────────────────────────
    with tab1:
        col_r, col_k, col_rf, col_ode, col_clr = st.columns([2, 2, 2, 2, 1])
        clicked_ridge = col_r.button("Predict (Physics + Ridge)", use_container_width=True)
        clicked_knn   = col_k.button("Predict (KNN)",             use_container_width=True)
        clicked_rf    = col_rf.button("Predict (Random Forest)",  use_container_width=True)
        clicked_ode   = col_ode.button("Predict (ODE Solver)",    use_container_width=True)
        clicked_clear = col_clr.button("Clear",                   use_container_width=True)

        if "predictions" not in st.session_state:
            st.session_state.predictions = {}

        if clicked_clear:
            st.session_state.predictions = {}

        for btn, method in [
            (clicked_ridge, "physics_ridge"),
            (clicked_knn,   "knn"),
            (clicked_rf,    "random_forest"),
            (clicked_ode,   "ode_solver"),
        ]:
            if btn:
                data, err = call_predict(specs, method)
                if err:
                    st.error(err)
                else:
                    st.session_state.predictions[method] = data

        fig = go.Figure()

        if show_training:
            for v in fetch_training_vehicles():
                fig.add_trace(go.Scatter(
                    x=v["time_points"],
                    y=v["temperatures"],
                    mode="lines",
                    name=v["vehicle"],
                    line=dict(dash="dot", width=1),
                    opacity=0.45,
                    hovertemplate=f"<b>{v['vehicle']}</b><br>t=%{{x}} min<br>T=%{{y:.1f}} °C<extra></extra>",
                ))

        for method, data in st.session_state.predictions.items():
            label = METHOD_LABELS[method]
            fig.add_trace(go.Scatter(
                x=data["time_points"],
                y=data["temperatures"],
                mode="lines+markers",
                name=f"Predicted ({label})",
                line=dict(color=COLORS[method], width=3),
                marker=dict(size=7),
                hovertemplate=f"<b>Predicted ({label})</b><br>t=%{{x}} min<br>T=%{{y:.2f}} °C<extra></extra>",
            ))

        fig.update_layout(
            title="Cabin Temperature vs Time",
            xaxis_title="Time (minutes)",
            yaxis_title="Temperature (°C)",
            height=480,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=80),
        )
        st.plotly_chart(fig, use_container_width=True)

        if st.session_state.predictions:
            st.subheader("Predicted Parameters")
            cols = st.columns(len(st.session_state.predictions) * 3)
            idx = 0
            for method, data in st.session_state.predictions.items():
                label = METHOD_LABELS[method]
                no_tau = method in ("random_forest", "ode_solver")
                cols[idx    ].metric(f"τ₁ — {label} (min)",    "N/A" if no_tau else f"{data['tau1']:.2f}")
                cols[idx + 1].metric(f"τ₂ — {label} (min)",    "N/A" if no_tau else f"{data['tau2']:.2f}")
                cols[idx + 2].metric(f"T_final — {label} (°C)", f"{data['T_final']:.2f}")
                idx += 3

    # ── Tab 2: Feature Importance ──────────────────────────────────────────────
    with tab2:
        fi = fetch_feature_importance()
        if fi is None:
            st.error(f"Cannot reach API at {API_URL}. Start the backend first.")
        else:
            c1, c2, c3 = st.columns(3)

            with c1:
                st.plotly_chart(
                    _importance_chart(fi["ridge_tau1"], "Ridge — Cooling Time Constant (τ₁)"),
                    use_container_width=True,
                )
                st.markdown("**Top 3 drivers of τ₁:**")
                _top3_explanations(fi["ridge_tau1"], "tau1")

            with c2:
                st.plotly_chart(
                    _importance_chart(fi["ridge_tfinal"], "Ridge — Equilibrium Temperature (T_final)"),
                    use_container_width=True,
                )
                st.markdown("**Top 3 drivers of T_final:**")
                _top3_explanations(fi["ridge_tfinal"], "tfinal")

            with c3:
                st.plotly_chart(
                    _importance_chart(fi["random_forest"], "Random Forest — Direct Temperature"),
                    use_container_width=True,
                )
                st.markdown("**Top 3 drivers (Random Forest):**")
                _top3_explanations(fi["random_forest"], "rf")

            st.divider()
            st.caption(
                "Green = helps cooling (negative Ridge coefficient or negative correlation with T_final).  "
                "Red = hurts cooling.  "
                "T_final Ridge is a diagnostic model — actual prediction uses physics scaling."
            )
            st.info(
                "**RPM bands 31–50, 51–70, 71–90 show low importance** because cooling completes "
                "in segment 1 (0–30 min) for all training vehicles (τ₁ ≈ 3–5 min). "
                "More diverse training data (higher thermal mass, lower AC power) is needed to "
                "capture late-phase RPM effects."
            )

    # ── Tab 3: Sensitivity Analysis ────────────────────────────────────────────
    with tab3:
        vehicles = fetch_training_vehicles()
        if not vehicles:
            st.error(f"Cannot reach API at {API_URL}. Start the backend first.")
        else:
            feat_keys   = list(FEATURE_CONFIG.keys())
            feat_labels = {k: FEATURE_CONFIG[k][0] for k in feat_keys}

            c_mode, c_meth = st.columns([3, 2])
            with c_mode:
                sens_mode = st.radio(
                    "Analysis mode",
                    options=["Individual band", "All bands scaled"],
                    horizontal=True,
                )
            with c_meth:
                sens_method = st.selectbox(
                    "Prediction method",
                    options=["physics_ridge", "knn", "random_forest", "ode_solver"],
                    format_func=lambda m: METHOD_LABELS[m],
                )

            selected_feat = None
            feat_min = feat_max = None

            if sens_mode == "Individual band":
                c_sel, c_run = st.columns([4, 1])
                with c_sel:
                    selected_feat = st.selectbox(
                        "Feature to vary",
                        options=feat_keys,
                        format_func=lambda k: feat_labels[k],
                    )
                with c_run:
                    st.markdown("<br>", unsafe_allow_html=True)
                    run_btn = st.button("Run Analysis", use_container_width=True)

                feat_vals_train = [
                    v["features"][selected_feat]
                    for v in vehicles
                    if selected_feat in v["features"]
                ]
                feat_min = float(min(feat_vals_train))
                feat_max = float(max(feat_vals_train))
                st.caption(
                    f"Training data range for **{feat_labels[selected_feat]}**: "
                    f"{feat_min:.2f} – {feat_max:.2f}  |  "
                    f"All other features held at current sidebar values."
                )
            else:
                c_cap, c_run = st.columns([4, 1])
                with c_cap:
                    st.caption(
                        "Scales all 4 RPM bands (0–30, 31–50, 51–70, 71–90 min) by the same "
                        "multiplier (0.7× to 1.3×). Shows how the whole curve shape changes "
                        "with uniform RPM variation."
                    )
                with c_run:
                    run_btn = st.button("Run Analysis", use_container_width=True)

            if "sensitivity" not in st.session_state:
                st.session_state.sensitivity = {}

            if run_btn:
                if sens_mode == "Individual band":
                    test_vals = np.linspace(feat_min, feat_max, 5)
                    results = []
                    with st.spinner("Running 5 predictions…"):
                        for v in test_vals:
                            test_specs = dict(specs)
                            test_specs[selected_feat] = float(v)
                            data, err = call_predict(test_specs, sens_method)
                            if data:
                                results.append((float(v), data))
                    st.session_state.sensitivity = {
                        "mode": "individual",
                        "feature": selected_feat,
                        "results": results,
                    }
                else:
                    multipliers = np.linspace(0.7, 1.3, 5)
                    results = []
                    with st.spinner("Running 5 predictions…"):
                        for mult in multipliers:
                            test_specs = dict(specs)
                            test_specs["rpm_0_30"]  = float(specs["rpm_0_30"]  * mult)
                            test_specs["rpm_31_50"] = float(specs["rpm_31_50"] * mult)
                            test_specs["rpm_51_70"] = float(specs["rpm_51_70"] * mult)
                            test_specs["rpm_71_90"] = float(specs["rpm_71_90"] * mult)
                            data, err = call_predict(test_specs, sens_method)
                            if data:
                                results.append((float(mult), data))
                    st.session_state.sensitivity = {
                        "mode": "all_bands",
                        "feature": None,
                        "results": results,
                    }

            sens = st.session_state.sensitivity
            show_results = bool(
                sens
                and sens.get("results")
                and (
                    (sens.get("mode") == "individual" and sens.get("feature") == selected_feat)
                    or (sens.get("mode") == "all_bands" and sens_mode == "All bands scaled")
                )
            )

            if show_results:
                results  = sens["results"]
                n        = len(results)
                gradient = pc.n_colors("rgb(0,0,210)", "rgb(210,0,0)", n, colortype="rgb")

                if sens["mode"] == "individual":
                    label_full  = feat_labels[selected_feat]
                    chart_title = f"Sensitivity: Varying {label_full}"
                else:
                    label_full  = "RPM Multiplier"
                    chart_title = "Sensitivity: All RPM Bands Scaled Uniformly"

                fig = go.Figure()
                for i, (val, data) in enumerate(results):
                    if sens["mode"] == "individual":
                        trace_name  = f"{label_full} = {val:.2f}"
                        hover_title = f"<b>{label_full} = {val:.2f}</b><br>"
                    else:
                        trace_name  = f"All RPMs x{val:.2f}"
                        hover_title = f"<b>All RPMs x{val:.2f}</b><br>"
                    fig.add_trace(go.Scatter(
                        x=data["time_points"],
                        y=data["temperatures"],
                        mode="lines+markers",
                        name=trace_name,
                        line=dict(color=gradient[i], width=2.5),
                        marker=dict(size=5),
                        hovertemplate=hover_title + "t=%{x} min<br>T=%{y:.2f} °C<extra></extra>",
                    ))

                fig.update_layout(
                    title=chart_title,
                    xaxis_title="Time (minutes)",
                    yaxis_title="Temperature (°C)",
                    height=460,
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(t=80),
                )
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Parameter Summary")
                time_5_idx  = 1   # TIME_POINTS[1]  ==  5 min
                time_15_idx = 3   # TIME_POINTS[3]  == 15 min
                time_30_idx = 6   # TIME_POINTS[6]  == 30 min
                time_60_idx = 12  # TIME_POINTS[12] == 60 min
                table = {
                    label_full:        [f"{v:.2f}"             for v, _    in results],
                    "τ₁ (min)":        [f"{d['tau1']:.2f}"     for _, d    in results],
                    "τ₂ (min)":        [f"{d['tau2']:.2f}"     for _, d    in results],
                    "T @ 5 min (°C)":  [f"{d['temperatures'][time_5_idx]:.2f}"  for _, d in results],
                    "T @ 15 min (°C)": [f"{d['temperatures'][time_15_idx]:.2f}" for _, d in results],
                    "T @ 30 min (°C)": [f"{d['temperatures'][time_30_idx]:.2f}" for _, d in results],
                    "T @ 60 min (°C)": [f"{d['temperatures'][time_60_idx]:.2f}" for _, d in results],
                    "T_final (°C)":    [f"{d['T_final']:.2f}"  for _, d    in results],
                }
                st.dataframe(table, use_container_width=True)

                st.info(
                    "**Note:** With τ₁ = 3–5 min, the cabin reaches equilibrium by t = 15–20 min. "
                    "RPM changes after t = 30 min have <0.1 °C effect on an already-stable temperature."
                )


if __name__ == "__main__":
    main()
