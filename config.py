"""Central configuration for the patent_python simulation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

# Paths
PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
CACHE_DIR: Path = DATA_DIR / "cache"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"

DEM_FILE: Path = RAW_DIR / "AP_19438_FBD_F0680_RT1.dem.tif"
DEM_CACHE_FILE: Path = CACHE_DIR / "Z_crop.npy"
DEM_CACHE_GEO_FILE: Path = CACHE_DIR / "Z_crop_geo.npz"
DEM_CACHE_META_FILE: Path = CACHE_DIR / "Z_crop_meta.json"
SIM_RESULT_FILE: Path = OUTPUTS_DIR / "simulation_results.json"
REPORT_TABLES_FILE: Path = OUTPUTS_DIR / "report_tables.json"
DOC_PAYLOAD_FILE: Path = OUTPUTS_DIR / "document_payload.json"

# Platform parameters
MASS: float = 8.5
ROTOR_RADIUS: float = 0.25
ROTOR_AREA: float = 4.0 * math.pi * ROTOR_RADIUS ** 2
GRAVITY: float = 9.8
FC_RATED_POWER: float = 3000.0
BAT_CAPACITY_AH: float = 22.0
BAT_VOLTAGE: float = 44.4
V_BUS: float = 48.0
BAT_MIN_CURRENT: float = 2.0

# Control parameters
FC_TAU: float = 5.0
FC_DP_DT_MAX: float = 600.0
FC_DP_MAX_STEP: float = 900.0
FC_RAMP_LIMIT: float = 300.0
FC_TARGET_DEMAND_MARGIN: float = 0.92
FC_TARGET_POWER_CAP_RATIO: float = 0.90
PASSIVE_BATTERY_OVERSUPPLY_RATIO: float = 1.05
BAT_SPIKE_THRESHOLD: float = 20.0
BAT_DEGRAD_THRESHOLD: float = 15.0
BATTERY_EQ_RESISTANCE: float = 0.05
PASSIVE_EQ_RESISTANCE: float = 0.08
UNDERSUPPLY_DROOP_RESISTANCE: float = 0.12
FC_STRESS_POWER_STEP_REF_W: float = 120.0
FC_STRESS_RAMP_REF_W_PER_S: float = 80.0
WIND_TRIGGER_RATIO: float = 1.05
WIND_TRIGGER_DELTA_COST_RATIO: float = WIND_TRIGGER_RATIO - 1.0
EMS_MESSAGE_LOOKAHEAD_FACTOR: float = 1.0
T3_LOOKAHEAD_SEGMENTS: int = 6
INITIAL_SOH: float = 0.90
MIN_SOH: float = 0.70
SOH_TRIGGER_THRESHOLD: float = 0.895
T4_FC_STRESS_TRIGGER: float = 20.0
T4_HEALTH_WEIGHT_TRIGGER: float = 1.02
T4_HEAVY_WEIGHT_DELTA: float = 0.30
SOH_DEGRADATION_PER_FC_STRESS: float = 2.0e-4
T4_COST_DELTA_RATIO_THRESHOLD: float = 0.01
T4_COST_DELTA_ABS_THRESHOLD: float = 0.02

# Simulation parameters
CRUISE_SPEED: float = 15.0
GRID_H_RES: float = 25.0
GRID_V_RES: float = 50.0
DEM_RES: float = 12.5
ALPHA: float = 1.0
BETA: float = 0.8
GAMMA: float = 1.5
H2_COST_SCALE: float = 1000.0
DEGRAD_CLIMB_RATE_REF_MPS: float = 3.0
DEGRAD_HIGH_POWER_RATIO: float = 0.85
DEGRAD_WIND_EXCESS_REF_MPS: float = 6.0
DEGRAD_CLIMB_WEIGHT: float = 0.35
DEGRAD_HIGH_POWER_WEIGHT: float = 0.45
DEGRAD_WIND_WEIGHT: float = 0.20
WIND_CD_BODY: float = 0.35
WIND_A_BODY: float = 0.12
WIND_HEADWIND_FRAC: float = 0.6
W1: float = 0.6
W2: float = 0.4
RHO_SEA_LEVEL: float = 1.225
SIM_DT: float = 1.0
SMOOTHING_FACTOR: float = 0.3
INITIAL_CORRIDOR_RADIUS_M: float = 600.0
CORRIDOR_ENDPOINT_MARGIN_M: float = 250.0
EVENT_CENTER_LOOKAHEAD_M: float = 150.0
T4_LOCAL_REFRESH_RADIUS_M: float = 80.0
T4_LOCAL_MAX_WAYPOINTS: int = 4
WINDSHEAR_TIME: float = 15.0
WINDSHEAR_AHEAD_M: float = 150.0
WIND_NORMAL: float = 3.0
WIND_SHEAR: float = 14.0
N_EVENTS: int = 3
EVENT_INTERVALS: list[float] = [15.0, 10.0, 8.0]

SWEEP_PRESETS: dict[str, dict[str, float]] = {
    "conservative": {
        "FC_TARGET_DEMAND_MARGIN": 0.88,
        "FC_TARGET_POWER_CAP_RATIO": 0.85,
        "PASSIVE_BATTERY_OVERSUPPLY_RATIO": 1.02,
        "FC_STRESS_POWER_STEP_REF_W": 100.0,
        "FC_STRESS_RAMP_REF_W_PER_S": 70.0,
        "WIND_TRIGGER_RATIO": 1.02,
    },
    "current": {
        "FC_TARGET_DEMAND_MARGIN": 0.92,
        "FC_TARGET_POWER_CAP_RATIO": 0.90,
        "PASSIVE_BATTERY_OVERSUPPLY_RATIO": 1.05,
        "FC_STRESS_POWER_STEP_REF_W": 120.0,
        "FC_STRESS_RAMP_REF_W_PER_S": 80.0,
        "WIND_TRIGGER_RATIO": 1.05,
    },
    "aggressive": {
        "FC_TARGET_DEMAND_MARGIN": 0.97,
        "FC_TARGET_POWER_CAP_RATIO": 0.96,
        "PASSIVE_BATTERY_OVERSUPPLY_RATIO": 1.10,
        "FC_STRESS_POWER_STEP_REF_W": 140.0,
        "FC_STRESS_RAMP_REF_W_PER_S": 95.0,
        "WIND_TRIGGER_RATIO": 1.10,
    },
}
PRIMARY_SWEEP_LABEL: str = "current"

# Mission configuration
START_LON: float = 110.0869
START_LAT: float = 34.4950
START_ALT: float = 500.0
GOAL_LON: float = 110.0781
GOAL_LAT: float = 34.4778
GOAL_ALT: float = 2154.9

PEAKS: dict[str, dict[str, float]] = {
    "south": {"lon": 110.0781, "lat": 34.4778, "elev": 2154.9},
    "north": {"lon": 110.0813, "lat": 34.4934, "elev": 1614.0},
    "east": {"lon": 110.0820, "lat": 34.4811, "elev": 2096.0},
    "west": {"lon": 110.0768, "lat": 34.4816, "elev": 2038.0},
    "center": {"lon": 110.0808, "lat": 34.4806, "elev": 2043.0},
}


def k_soh(soh: float) -> float:
    """Return the adaptive SoH weight."""
    if soh >= 0.9:
        return 1.0
    if soh >= 0.8:
        return 1.0 + (0.9 - soh) / 0.1 * 0.5
    if soh >= 0.7:
        return 1.5 + (0.8 - soh) / 0.1 * 0.7
    return 2.2


def health_reset_stage(soh: float) -> int:
    """Return a coarse health stage for deciding whether a planner reset is necessary."""
    weight = k_soh(soh)
    if weight < 1.25:
        return 0
    if weight < 1.75:
        return 1
    return 2


def air_density(altitude_m: float) -> float:
    """Standard-atmosphere density approximation."""
    return RHO_SEA_LEVEL * (1.0 - 2.2558e-5 * altitude_m) ** 4.2559


def snapshot() -> dict[str, Any]:
    """Return the runtime configuration snapshot for exported results."""
    return {
        "platform": {
            "mass_kg": MASS,
            "rotor_radius_m": ROTOR_RADIUS,
            "rotor_area_m2": ROTOR_AREA,
            "gravity_mps2": GRAVITY,
            "fc_rated_power_w": FC_RATED_POWER,
            "bat_capacity_ah": BAT_CAPACITY_AH,
            "bat_voltage_v": BAT_VOLTAGE,
            "v_bus_v": V_BUS,
        },
        "control": {
            "fc_tau_s": FC_TAU,
            "fc_dp_dt_max_wps": FC_DP_DT_MAX,
            "fc_dp_max_step_w": FC_DP_MAX_STEP,
            "fc_ramp_limit_wps": FC_RAMP_LIMIT,
            "fc_target_demand_margin": FC_TARGET_DEMAND_MARGIN,
            "fc_target_power_cap_ratio": FC_TARGET_POWER_CAP_RATIO,
            "passive_battery_oversupply_ratio": PASSIVE_BATTERY_OVERSUPPLY_RATIO,
            "bat_spike_threshold_a": BAT_SPIKE_THRESHOLD,
            "bat_degrad_threshold_a": BAT_DEGRAD_THRESHOLD,
            "battery_eq_resistance_ohm": BATTERY_EQ_RESISTANCE,
            "passive_eq_resistance_ohm": PASSIVE_EQ_RESISTANCE,
            "undersupply_droop_resistance_ohm": UNDERSUPPLY_DROOP_RESISTANCE,
            "fc_stress_power_step_ref_w": FC_STRESS_POWER_STEP_REF_W,
            "fc_stress_ramp_ref_wps": FC_STRESS_RAMP_REF_W_PER_S,
            "wind_trigger_ratio": WIND_TRIGGER_RATIO,
            "t3_lookahead_segments": T3_LOOKAHEAD_SEGMENTS,
            "initial_soh": INITIAL_SOH,
            "min_soh": MIN_SOH,
            "soh_trigger_threshold": SOH_TRIGGER_THRESHOLD,
            "t4_fc_stress_trigger": T4_FC_STRESS_TRIGGER,
            "t4_health_weight_trigger": T4_HEALTH_WEIGHT_TRIGGER,
            "t4_heavy_weight_delta": T4_HEAVY_WEIGHT_DELTA,
            "soh_degradation_per_fc_stress": SOH_DEGRADATION_PER_FC_STRESS,
            "t4_cost_delta_ratio_threshold": T4_COST_DELTA_RATIO_THRESHOLD,
            "t4_cost_delta_abs_threshold": T4_COST_DELTA_ABS_THRESHOLD,
        },
        "simulation": {
            "cruise_speed_mps": CRUISE_SPEED,
            "grid_h_res_m": GRID_H_RES,
            "grid_v_res_m": GRID_V_RES,
            "dem_res_m": DEM_RES,
            "alpha": ALPHA,
            "beta": BETA,
            "gamma": GAMMA,
            "h2_cost_scale": H2_COST_SCALE,
            "degrad_climb_rate_ref_mps": DEGRAD_CLIMB_RATE_REF_MPS,
            "degrad_high_power_ratio": DEGRAD_HIGH_POWER_RATIO,
            "degrad_wind_excess_ref_mps": DEGRAD_WIND_EXCESS_REF_MPS,
            "degrad_climb_weight": DEGRAD_CLIMB_WEIGHT,
            "degrad_high_power_weight": DEGRAD_HIGH_POWER_WEIGHT,
            "degrad_wind_weight": DEGRAD_WIND_WEIGHT,
            "sim_dt_s": SIM_DT,
            "smoothing_factor": SMOOTHING_FACTOR,
            "initial_corridor_radius_m": INITIAL_CORRIDOR_RADIUS_M,
            "corridor_endpoint_margin_m": CORRIDOR_ENDPOINT_MARGIN_M,
            "event_center_lookahead_m": EVENT_CENTER_LOOKAHEAD_M,
            "t4_local_refresh_radius_m": T4_LOCAL_REFRESH_RADIUS_M,
            "t4_local_max_waypoints": T4_LOCAL_MAX_WAYPOINTS,
            "windshear_ahead_m": WINDSHEAR_AHEAD_M,
            "wind_normal_mps": WIND_NORMAL,
            "wind_shear_mps": WIND_SHEAR,
            "n_events": N_EVENTS,
            "event_intervals_s": list(EVENT_INTERVALS),
            "sweep_presets": SWEEP_PRESETS,
            "primary_sweep_label": PRIMARY_SWEEP_LABEL,
        },
        "mission": {
            "start": {"lon": START_LON, "lat": START_LAT, "alt": START_ALT},
            "goal": {"lon": GOAL_LON, "lat": GOAL_LAT, "alt": GOAL_ALT},
        },
    }

