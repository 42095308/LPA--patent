"""Trajectory feature extraction and power-profile generation."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.interpolate import splrep, splev

import config
from energy_map import EnergyMap, _hover_power


def extract_geometry(energy_map: EnergyMap, path: List[int]) -> List[Dict[str, float]]:
    """Extract per-segment geometry and local operating conditions."""
    segments: List[Dict[str, float]] = []
    for idx in range(len(path) - 1):
        metrics = energy_map.compute_power_for_segment(path[idx], path[idx + 1])
        segments.append(
            {
                "segment_index": float(idx),
                "distance_h": metrics["distance_h"],
                "distance_3d": metrics["distance_m"],
                "delta_h": metrics["delta_h"],
                "delta_t": metrics["flight_time"],
                "climb_rate": metrics["climb_rate"],
                "alt_start": metrics["start_z"],
                "alt_end": metrics["end_z"],
                "altitude_mid": metrics["altitude_mid"],
                "wind_speed": metrics["wind_speed"],
                "p_total": metrics["p_total"],
                "start_x": metrics["start_x"],
                "start_y": metrics["start_y"],
                "start_z": metrics["start_z"],
                "end_x": metrics["end_x"],
                "end_y": metrics["end_y"],
                "end_z": metrics["end_z"],
            }
        )
    return segments


def compute_power_sequence(
    energy_map: EnergyMap,
    path: List[int],
    sample_dt: float = config.SIM_DT,
    start_time_s: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a time-based demand-power sequence from a path."""
    if len(path) < 2:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)

    time_points: list[float] = []
    power_points: list[float] = []
    cumulative_t = float(start_time_s)

    for idx in range(len(path) - 1):
        metrics = energy_map.compute_power_for_segment(path[idx], path[idx + 1])
        flight_time = float(metrics["flight_time"])
        if flight_time <= 0.0:
            continue
        p_total = float(metrics["p_total"])
        n_steps = max(1, int(np.ceil(flight_time / max(sample_dt, 1e-6))))
        step_dt = flight_time / n_steps
        for _ in range(n_steps):
            cumulative_t += step_dt
            time_points.append(cumulative_t)
            power_points.append(p_total)

    return np.array(time_points, dtype=float), np.array(power_points, dtype=float)


def smooth_power_bspline(
    time_arr: np.ndarray,
    power_arr: np.ndarray,
    smoothing_factor: float = config.SMOOTHING_FACTOR,
    resample_dt: float = config.SIM_DT,
) -> Tuple[np.ndarray, np.ndarray]:
    """Smooth and resample a power profile on the real time axis."""
    if len(time_arr) == 0 or len(power_arr) == 0:
        return time_arr, power_arr
    if len(time_arr) != len(power_arr):
        raise ValueError("time_arr and power_arr must have the same length.")
    if len(time_arr) < 4:
        return time_arr, power_arr

    start_t = float(time_arr[0])
    end_t = float(time_arr[-1])
    if end_t <= start_t:
        return time_arr, power_arr

    resampled_time = np.arange(start_t, end_t + 1e-9, resample_dt, dtype=float)
    if resampled_time[-1] < end_t:
        resampled_time = np.append(resampled_time, end_t)

    try:
        tck = splrep(time_arr, power_arr, s=smoothing_factor * len(time_arr), k=min(3, len(time_arr) - 1))
        smoothed_power = np.asarray(splev(resampled_time, tck), dtype=float)
    except Exception:
        smoothed_power = np.interp(resampled_time, time_arr, power_arr)

    return resampled_time, np.maximum(smoothed_power, 0.0)


def extract_feature_vector(
    time_arr: np.ndarray,
    power_arr: np.ndarray,
    segments: List[Dict[str, float]],
) -> Dict[str, float]:
    """Extract the feedforward feature vector from a time-based profile."""
    if len(time_arr) == 0 or len(power_arr) == 0:
        return {
            "P_peak": 0.0,
            "T_ramp": 0.0,
            "avg_climb_rate": 0.0,
            "E_hydrogen": 0.0,
        }

    peak_idx = int(np.argmax(power_arr))
    t_ramp = float(time_arr[peak_idx] - time_arr[0]) if peak_idx > 0 else 0.0
    positive_rates = [seg["climb_rate"] for seg in segments if seg["climb_rate"] > 0.0]
    avg_climb_rate = float(np.mean(positive_rates)) if positive_rates else 0.0

    shifted_time = np.concatenate(([time_arr[0] - config.SIM_DT], time_arr))
    shifted_power = np.concatenate(([power_arr[0]], power_arr))
    # Keep the trapezoidal integration logic independent from NumPy version changes.
    energy_j = float(
        np.sum(0.5 * (shifted_power[1:] + shifted_power[:-1]) * np.diff(shifted_time))
    ) if len(shifted_time) > 1 else 0.0
    h2_g = energy_j / (0.50 * 120.0e3)

    return {
        "P_peak": float(np.max(power_arr)),
        "T_ramp": t_ramp,
        "avg_climb_rate": avg_climb_rate,
        "E_hydrogen": h2_g,
    }


def build_structured_message(
    timestamp: float,
    replanning_id: int,
    feature_vec: Dict[str, float],
    time_arr: np.ndarray,
    power_arr: np.ndarray,
    t_window: float,
) -> Dict[str, object]:
    """Build the structured feedforward message for the EMS."""
    return {
        "timestamp": float(timestamp),
        "replanning_id": int(replanning_id),
        "feature_vector": {
            "P_peak": float(feature_vec["P_peak"]),
            "T_ramp": float(feature_vec["T_ramp"]),
            "avg_climb_rate": float(feature_vec["avg_climb_rate"]),
            "E_hydrogen": float(feature_vec["E_hydrogen"]),
        },
        "P_predict": {
            "time_s": [float(t) for t in time_arr.tolist()],
            "power_w": [float(p) for p in power_arr.tolist()],
        },
        "T_window": float(t_window),
    }


def classify_flight_phases(energy_map: EnergyMap, path: List[int]) -> List[Dict[str, float]]:
    """Split a path into coarse flight phases for reporting."""
    if len(path) < 3:
        return []

    segments = [energy_map.compute_power_for_segment(path[idx], path[idx + 1]) for idx in range(len(path) - 1)]
    total_segments = max(len(segments), 1)
    for idx, seg in enumerate(segments):
        seg["progress"] = idx / total_segments

    phase_ranges = [
        ("takeoff_climb", 0.0, 0.10),
        ("low_alt_cruise", 0.10, 0.35),
        ("mountain_climb", 0.35, 0.65),
        ("high_alt_cruise", 0.65, 0.80),
        ("hover_hold", 0.80, 0.85),
        ("return_descent", 0.85, 1.0),
    ]

    phases: List[Dict[str, float]] = []
    for phase_name, start_progress, end_progress in phase_ranges:
        phase_segments = [
            seg for seg in segments if start_progress <= seg["progress"] < end_progress
        ]
        if not phase_segments:
            if phase_name == "hover_hold":
                avg_alt = float(np.mean([seg["altitude_mid"] for seg in segments]))
                hover_power = _hover_power(avg_alt)
                phases.append(
                    {
                        "phase": phase_name,
                        "duration_s": config.FC_TAU,
                        "avg_power_w": hover_power,
                        "peak_power_w": hover_power,
                        "dp_req_w": 0.0,
                    }
                )
            continue

        powers = [float(seg["p_total"]) for seg in phase_segments]
        dp_values = [abs(powers[idx] - powers[idx - 1]) for idx in range(1, len(powers))]
        phases.append(
            {
                "phase": phase_name,
                "duration_s": float(sum(seg["flight_time"] for seg in phase_segments)),
                "avg_power_w": float(np.mean(powers)),
                "peak_power_w": float(np.max(powers)),
                "dp_req_w": float(np.max(dp_values)) if dp_values else 0.0,
            }
        )

    return phases
