from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from backend.model import get_all_vehicle_curves, get_feature_importances, predict_curve
from backend.schemas import PredictionResponse, VehicleSpecs

app = FastAPI(title="Vehicle Cabin Temperature Predictor", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict", response_model=PredictionResponse)
def predict(
    specs: VehicleSpecs,
    method: str = Query("physics_ridge", enum=["physics_ridge", "knn", "random_forest", "ode_solver"]),
):
    temps, tau1, tau2, T_final, upper_band, lower_band = predict_curve(specs.model_dump(), method=method)
    return PredictionResponse(
        time_points=list(range(0, 95, 5)),
        temperatures=[round(t, 2) for t in temps],
        tau1=round(tau1, 3),
        tau2=round(tau2, 3),
        T_final=round(T_final, 3),
        method=method,
        upper_band=[round(t, 2) for t in upper_band],
        lower_band=[round(t, 2) for t in lower_band],
        confidence_level=0.90,
    )


@app.get("/vehicles")
def get_vehicles():
    return get_all_vehicle_curves()


@app.get("/feature-importance")
def feature_importance():
    return get_feature_importances()
