import os

import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

# Feature slider config: (min, max, default, step)
FEATURE_CONFIG = {
    "heat_load_kw":        ("Heat Load (kW)",            3.0,  7.0,   5.0,  0.1),
    "cabin_volume_m3":     ("Cabin Volume (m³)",          2.0,  5.0,   3.0,  0.1),
    "pulley_ratio":        ("Pulley Ratio",               1.0,  2.0,   1.5,  0.05),
    "solar_w_m2":          ("Solar Radiation (W/m²)",     800., 1500., 1200., 50.),
    "ac_unit_capacity_kw": ("AC Capacity (kW)",           3.0,  6.0,   4.4,  0.1),
    "condenser_capacity_kw": ("Condenser Capacity (kW)",  7.0,  12.0,  9.0,  0.5),
    "compressor_size_cc":  ("Compressor Size (cc)",       100., 200.,  140., 5.),
    "airflow_m3_hr":       ("Airflow (m³/hr)",            400., 700.,  550., 10.),
    "soaking_time_hr":     ("Soaking Time (hr)",          0.5,  3.0,   1.0,  0.5),
    "rpm_0_30":            ("RPM 0–30 km/h",              1000.,2500., 1600.,50.),
    "rpm_31_50":           ("RPM 31–50 km/h",             1000.,2500., 1700.,50.),
    "rpm_51_70":           ("RPM 51–70 km/h",             1000.,2500., 1800.,50.),
    "rpm_71_90":           ("RPM 71–90 km/h",             500., 1200.,  750.,50.),
    "ebhs":                ("EBHS",                        60.,  200.,  100., 5.),
}

COLORS = {
    "physics_ridge": "#E63946",
    "knn":           "#2196F3",
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

    # ── Sidebar ──────────────────────────────────────────────────────────────
    st.sidebar.header("Vehicle Specifications")
    specs = {}
    for feat, (label, mn, mx, default, step) in FEATURE_CONFIG.items():
        specs[feat] = st.sidebar.slider(
            label, min_value=mn, max_value=mx, value=default, step=step
        )

    st.sidebar.divider()
    st.sidebar.header("Comparison")
    show_training = st.sidebar.checkbox("Overlay training vehicle curves", value=False)

    # ── Predict buttons ───────────────────────────────────────────────────────
    col_r, col_k, col_clr = st.columns([2, 2, 1])
    clicked_ridge = col_r.button("Predict (Physics + Ridge)", use_container_width=True)
    clicked_knn   = col_k.button("Predict (KNN)",             use_container_width=True)
    clicked_clear = col_clr.button("Clear", use_container_width=True)

    if "predictions" not in st.session_state:
        st.session_state.predictions = {}

    if clicked_clear:
        st.session_state.predictions = {}

    for btn, method in [(clicked_ridge, "physics_ridge"), (clicked_knn, "knn")]:
        if btn:
            data, err = call_predict(specs, method)
            if err:
                st.error(err)
            else:
                st.session_state.predictions[method] = data

    # ── Chart ─────────────────────────────────────────────────────────────────
    fig = go.Figure()

    if show_training:
        vehicles = fetch_training_vehicles()
        for v in vehicles:
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
        label = "Physics + Ridge" if method == "physics_ridge" else "KNN"
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

    # ── Metrics ───────────────────────────────────────────────────────────────
    if st.session_state.predictions:
        st.subheader("Predicted Parameters")
        cols = st.columns(len(st.session_state.predictions) * 2)
        idx = 0
        for method, data in st.session_state.predictions.items():
            label = "Physics + Ridge" if method == "physics_ridge" else "KNN"
            cols[idx].metric(f"τ — {label} (min)", f"{data['tau']:.2f}")
            cols[idx + 1].metric(f"T_final — {label} (°C)", f"{data['T_final']:.2f}")
            idx += 2


if __name__ == "__main__":
    main()
