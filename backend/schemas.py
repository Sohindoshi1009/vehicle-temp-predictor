from typing import List

from pydantic import BaseModel


class VehicleSpecs(BaseModel):
    heat_load_kw: float
    cabin_volume_m3: float
    pulley_ratio: float
    solar_w_m2: float
    ac_unit_capacity_kw: float
    condenser_capacity_kw: float
    compressor_size_cc: float
    airflow_m3_hr: float
    soaking_time_hr: float
    rpm_0_30: float
    rpm_31_50: float
    rpm_51_70: float
    rpm_71_90: float
    ebhs: float


class PredictionResponse(BaseModel):
    time_points: List[int]
    temperatures: List[float]
    tau: float
    T_final: float
    method: str
