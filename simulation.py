"""Dual-chain event-driven simulation entrypoint."""

from __future__ import annotations

import json
import math
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterator, List, Tuple

import numpy as np

import config
from dem_loader import load_dem
from ems import EMSController, PassiveEMS, fc_stress_increment
from energy_map import EnergyMap, _climb_power, _cruise_power, _hover_power
from planner import AStarPlanner, LPAStar
from trajectory import (
    build_structured_message,
    classify_flight_phases,
    compute_power_sequence,
    extract_feature_vector,
    extract_geometry,
    smooth_power_bspline,
)


@dataclass
class ScheduleSegment:
    kind: str
    duration_s: float
    power_w: float
    start_xyz: Tuple[float, float, float]
    end_xyz: Tuple[float, float, float]
    node_from: int | None = None
    node_to: int | None = None


@dataclass
class FlightState:
    elapsed_time_s: float
    current_xyz: Tuple[float, float, float]
    current_node_id: int
    path_nodes: List[int]
    segment_index: int = 0
    segment_elapsed_s: float = 0.0
    remaining_path_nodes: List[int] = field(default_factory=list)


@dataclass
class TriggerDecision:
    trigger_id: str
    triggered: bool
    reason: str
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class EventRecord:
    event_id: int
    trigger_time_s: float
    decisions: List[TriggerDecision]
    t2_trigger_s: float
    t3_message_ready_s: float
    t4_ems_ready_s: float
    t5_flight_execute_s: float
    planning_latency_ms: float
    chain_latency_ms: float
    current_xyz: Tuple[float, float, float]
    current_node_id: int
    snap_error_m: float
    new_path_length_m: float
    new_path_nodes: List[int]
    structured_message: Dict[str, object] | None
    t2_changed_nodes: int = 0
    t4_changed_nodes: int = 0
    planner_nodes_expanded: int = 0
    lpa_nodes_expanded: int = 0
    lpa_heap_rekey_count: int = 0
    compute_shortest_path_ms: float = 0.0
    path_extract_ms: float = 0.0


@dataclass
class PendingPlan:
    replanning_id: int
    activation_time_s: float
    path_nodes: List[int]
    profile_history_index: int
    event_record_index: int | None = None


@dataclass
class ChainState:
    name: str
    energy_map: EnergyMap
    goal_node_id: int
    flight_state: FlightState
    lpa_planner: LPAStar | None = None
    astar_planner: AStarPlanner | None = None
    schedule: List[ScheduleSegment] = field(default_factory=list)
    planning_latency_ms: List[float] = field(default_factory=list)
    chain_latency_ms: List[float] = field(default_factory=list)
    structured_messages: List[Dict[str, object]] = field(default_factory=list)
    power_profile_history: List[Dict[str, object]] = field(default_factory=list)
    executed_time_s: List[float] = field(default_factory=list)
    executed_power_w: List[float] = field(default_factory=list)
    event_records: List[EventRecord] = field(default_factory=list)
    initial_phases: List[Dict[str, float]] = field(default_factory=list)
    initial_path_nodes: List[int] = field(default_factory=list)
    initial_path_length_m: float = 0.0
    executed_distance_m: float = 0.0
    completed: bool = False
    initial_plan_ms: float = 0.0
    observed_max_dp_req_w: float = 0.0
    online_fc_stress_index: float = 0.0
    estimated_soh: float = config.INITIAL_SOH
    force_planner_reset: bool = False
    pending_plan: PendingPlan | None = None


def path_length_m(energy_map: EnergyMap, path_nodes: List[int]) -> float:
    total = 0.0
    for idx in range(len(path_nodes) - 1):
        total += math.dist(energy_map.nodes[path_nodes[idx]], energy_map.nodes[path_nodes[idx + 1]])
    return total


def estimate_continuous_segment(
    energy_map: EnergyMap,
    start_xyz: Tuple[float, float, float],
    end_xyz: Tuple[float, float, float],
    kind: str,
) -> ScheduleSegment | None:
    start = np.array(start_xyz, dtype=float)
    end = np.array(end_xyz, dtype=float)
    delta = end - start
    distance_m = float(np.linalg.norm(delta))
    if distance_m <= 1e-9:
        return None

    flight_time = distance_m / config.CRUISE_SPEED
    mid_xyz = 0.5 * (start + end)
    climb_rate = float(delta[2]) / max(flight_time, 0.1)
    wind_speed = energy_map._get_wind(float(mid_xyz[0]), float(mid_xyz[1]))
    p_hover = _hover_power(float(mid_xyz[2]))
    p_climb = _climb_power(max(0.0, climb_rate))
    p_cruise = _cruise_power(config.CRUISE_SPEED, wind_speed, float(mid_xyz[2]))
    return ScheduleSegment(
        kind=kind,
        duration_s=float(flight_time),
        power_w=float(p_hover + p_climb + p_cruise),
        start_xyz=(float(start[0]), float(start[1]), float(start[2])),
        end_xyz=(float(end[0]), float(end[1]), float(end[2])),
    )


def build_schedule_from_path(
    energy_map: EnergyMap,
    current_xyz: Tuple[float, float, float],
    path_nodes: List[int],
    hold_duration_s: float,
) -> Tuple[List[ScheduleSegment], float]:
    schedule: List[ScheduleSegment] = []
    if hold_duration_s > 1e-9:
        hover_power = _hover_power(current_xyz[2])
        schedule.append(
            ScheduleSegment(
                kind="hold",
                duration_s=hold_duration_s,
                power_w=hover_power,
                start_xyz=current_xyz,
                end_xyz=current_xyz,
            )
        )

    if not path_nodes:
        return schedule, 0.0

    snap_target_xyz = energy_map.position_from_node(path_nodes[0])
    snap_error_m = float(math.dist(current_xyz, snap_target_xyz))
    connector = estimate_continuous_segment(energy_map, current_xyz, snap_target_xyz, kind="connector")
    if connector is not None:
        schedule.append(connector)

    for idx in range(len(path_nodes) - 1):
        node_from = path_nodes[idx]
        node_to = path_nodes[idx + 1]
        metrics = energy_map.compute_power_for_segment(node_from, node_to)
        schedule.append(
            ScheduleSegment(
                kind="path",
                duration_s=float(metrics["flight_time"]),
                power_w=float(metrics["p_total"]),
                start_xyz=(metrics["start_x"], metrics["start_y"], metrics["start_z"]),
                end_xyz=(metrics["end_x"], metrics["end_y"], metrics["end_z"]),
                node_from=node_from,
                node_to=node_to,
            )
        )

    return schedule, snap_error_m


def build_profile_payload(
    energy_map: EnergyMap,
    path_nodes: List[int],
    execute_time_s: float,
) -> Dict[str, object]:
    geometry = extract_geometry(energy_map, path_nodes)
    raw_time, raw_power = compute_power_sequence(energy_map, path_nodes, start_time_s=0.0)
    smooth_time, smooth_power = smooth_power_bspline(raw_time, raw_power)
    feature_vector = extract_feature_vector(smooth_time, smooth_power, geometry)
    absolute_time = smooth_time + execute_time_s if len(smooth_time) else smooth_time
    return {
        "geometry": geometry,
        "feature_vector": feature_vector,
        "time_s": [float(t) for t in absolute_time.tolist()],
        "power_w": [float(p) for p in smooth_power.tolist()],
        "max_dp_req_w": max_dp_req_w(smooth_power),
        "t_window": float(smooth_time[-1]) if len(smooth_time) else 0.0,
        "path_length_m": path_length_m(energy_map, path_nodes),
    }


def apply_schedule_from_path(
    chain: ChainState,
    current_xyz: Tuple[float, float, float],
    path_nodes: List[int],
) -> float:
    chain.schedule, snap_error_m = build_schedule_from_path(
        chain.energy_map,
        current_xyz,
        path_nodes,
        hold_duration_s=0.0,
    )
    chain.flight_state.path_nodes = list(path_nodes)
    chain.flight_state.segment_index = 0
    chain.flight_state.segment_elapsed_s = 0.0
    chain.flight_state.remaining_path_nodes = list(path_nodes)
    return snap_error_m


def select_activation_path(
    energy_map: EnergyMap,
    current_xyz: Tuple[float, float, float],
    path_nodes: List[int],
) -> List[int]:
    if not path_nodes:
        return []

    current = np.array(current_xyz, dtype=float)
    distances = [
        float(np.linalg.norm(np.array(energy_map.position_from_node(node_id), dtype=float) - current))
        for node_id in path_nodes
    ]
    best_idx = int(np.argmin(np.asarray(distances, dtype=float)))
    return list(path_nodes[best_idx:])


def activate_pending_plan(chain: ChainState) -> None:
    pending = chain.pending_plan
    if pending is None:
        return

    activation_path: List[int] = []
    if chain.lpa_planner is not None:
        activation_path = chain.lpa_planner.extract_path(chain.flight_state.current_node_id)
        if not activation_path or activation_path[0] != chain.flight_state.current_node_id:
            found = chain.lpa_planner.compute_shortest_path(chain.flight_state.current_node_id)
            if found:
                activation_path = chain.lpa_planner.extract_path(chain.flight_state.current_node_id)

    if not activation_path:
        activation_path = select_activation_path(
            chain.energy_map,
            chain.flight_state.current_xyz,
            pending.path_nodes,
        )
    snap_error_m = apply_schedule_from_path(
        chain,
        tuple(chain.flight_state.current_xyz),
        activation_path,
    )

    profile_payload = build_profile_payload(
        chain.energy_map,
        activation_path,
        pending.activation_time_s,
    )
    profile_entry = chain.power_profile_history[pending.profile_history_index]
    profile_entry["predicted_path_nodes"] = list(profile_entry["path_nodes"])
    profile_entry["predicted_path_length_m"] = float(profile_entry["path_length_m"])
    profile_entry["predicted_time_s"] = list(profile_entry["time_s"])
    profile_entry["predicted_power_w"] = list(profile_entry["power_w"])
    profile_entry["activated_from_pending"] = True
    profile_entry["activation_time_s"] = float(pending.activation_time_s)
    profile_entry["path_nodes"] = list(activation_path)
    profile_entry["path_length_m"] = float(profile_payload["path_length_m"])
    profile_entry["time_s"] = list(profile_payload["time_s"])
    profile_entry["power_w"] = list(profile_payload["power_w"])
    profile_entry["max_dp_req_w"] = float(profile_payload["max_dp_req_w"])
    profile_entry["feature_vector"] = profile_payload["feature_vector"]

    if pending.event_record_index is not None:
        event_record = chain.event_records[pending.event_record_index]
        event_record.snap_error_m = snap_error_m
        event_record.new_path_nodes = list(activation_path)
        event_record.new_path_length_m = float(profile_payload["path_length_m"])

    chain.pending_plan = None


def append_power_samples(chain: ChainState, duration_s: float, power_w: float) -> None:
    if duration_s <= 1e-9:
        return
    n_steps = max(1, int(math.ceil(duration_s / config.SIM_DT)))
    step_dt = duration_s / n_steps
    current_time = chain.flight_state.elapsed_time_s
    prev_power = chain.executed_power_w[-1] if chain.executed_power_w else float(power_w)

    for _ in range(n_steps):
        current_time += step_dt
        dp_req = abs(float(power_w) - float(prev_power))
        chain.observed_max_dp_req_w = max(chain.observed_max_dp_req_w, dp_req)
        stress_inc = fc_stress_increment(dp_req, step_dt)
        chain.online_fc_stress_index += stress_inc
        chain.estimated_soh = max(
            config.MIN_SOH,
            chain.estimated_soh - stress_inc * config.SOH_DEGRADATION_PER_FC_STRESS,
        )
        chain.executed_time_s.append(float(current_time))
        chain.executed_power_w.append(float(power_w))
        prev_power = float(power_w)


def _consume_schedule_until(chain: ChainState, target_time_s: float) -> None:
    remaining = target_time_s - chain.flight_state.elapsed_time_s
    while remaining > 1e-9 and chain.schedule:
        segment = chain.schedule[0]
        original_duration = segment.duration_s
        consume = min(remaining, original_duration)
        append_power_samples(chain, consume, segment.power_w)

        if segment.kind == "hold":
            new_xyz = np.array(segment.end_xyz, dtype=float)
        else:
            start_xyz = np.array(segment.start_xyz, dtype=float)
            end_xyz = np.array(segment.end_xyz, dtype=float)
            fraction = consume / max(original_duration, 1e-9)
            new_xyz = start_xyz + fraction * (end_xyz - start_xyz)
            segment_distance = float(np.linalg.norm(end_xyz - start_xyz))
            chain.executed_distance_m += segment_distance * fraction

        chain.flight_state.elapsed_time_s += consume
        chain.flight_state.current_xyz = (float(new_xyz[0]), float(new_xyz[1]), float(new_xyz[2]))
        chain.flight_state.current_node_id = chain.energy_map.find_nearest_node_by_xyz(*chain.flight_state.current_xyz)

        if segment.kind == "path":
            if consume >= original_duration - 1e-9:
                chain.flight_state.segment_index += 1
                chain.flight_state.segment_elapsed_s = 0.0
                if (
                    chain.flight_state.remaining_path_nodes
                    and segment.node_from is not None
                    and chain.flight_state.remaining_path_nodes[0] == segment.node_from
                ):
                    chain.flight_state.remaining_path_nodes.pop(0)
            else:
                chain.flight_state.segment_elapsed_s += consume

        remaining -= consume
        segment.duration_s -= consume

        if segment.duration_s <= 1e-9:
            if segment.kind == "path" and segment.node_to is not None:
                chain.flight_state.current_node_id = segment.node_to
                chain.flight_state.current_xyz = chain.energy_map.position_from_node(segment.node_to)
            chain.schedule.pop(0)
        elif segment.kind != "hold":
            segment.start_xyz = chain.flight_state.current_xyz

    if not chain.schedule and chain.flight_state.current_node_id == chain.goal_node_id:
        chain.completed = True
        chain.flight_state.remaining_path_nodes = [chain.goal_node_id]
        chain.pending_plan = None


def advance_chain_to_time(chain: ChainState, target_time_s: float) -> None:
    if chain.completed and chain.pending_plan is None:
        return

    while chain.pending_plan is not None:
        activation_time_s = chain.pending_plan.activation_time_s
        if activation_time_s > target_time_s + 1e-9:
            break
        if chain.flight_state.elapsed_time_s < activation_time_s - 1e-9:
            _consume_schedule_until(chain, activation_time_s)
        if chain.completed:
            chain.pending_plan = None
            return
        if chain.pending_plan is not None and not chain.schedule and chain.flight_state.elapsed_time_s < activation_time_s - 1e-9:
            chain.pending_plan.activation_time_s = chain.flight_state.elapsed_time_s
            activation_time_s = chain.pending_plan.activation_time_s
        if chain.flight_state.elapsed_time_s >= activation_time_s - 1e-9:
            activate_pending_plan(chain)

    _consume_schedule_until(chain, target_time_s)


def max_dp_req_w(power_arr: np.ndarray) -> float:
    if len(power_arr) < 2:
        return 0.0
    return float(np.max(np.abs(np.diff(power_arr))))


def robust_dp_req_w(power_arr: np.ndarray, percentile: float) -> float:
    if len(power_arr) < 2:
        return 0.0
    diffs = np.abs(np.diff(power_arr))
    if len(diffs) == 0:
        return 0.0
    if len(diffs) <= 2:
        return float(np.max(diffs))
    return float(np.percentile(diffs, percentile))


def mean_or_zero(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def remaining_path_snapshot(chain: ChainState) -> List[int]:
    if chain.flight_state.remaining_path_nodes:
        return list(chain.flight_state.remaining_path_nodes)
    if chain.flight_state.path_nodes:
        return list(chain.flight_state.path_nodes)
    return [chain.flight_state.current_node_id]


def project_event_center(
    energy_map: EnergyMap,
    current_xyz: Tuple[float, float, float],
    remaining_path_nodes: List[int],
    lookahead_m: float,
) -> Tuple[float, float]:
    current = np.array(current_xyz[:2], dtype=float)
    remaining_distance = lookahead_m

    waypoints: List[np.ndarray] = [current]
    for node_id in remaining_path_nodes:
        node_xyz = energy_map.position_from_node(node_id)
        waypoints.append(np.array(node_xyz[:2], dtype=float))

    for idx in range(len(waypoints) - 1):
        start = waypoints[idx]
        end = waypoints[idx + 1]
        segment_length = float(np.linalg.norm(end - start))
        if segment_length <= 1e-9:
            continue
        if remaining_distance <= segment_length:
            ratio = remaining_distance / segment_length
            point = start + ratio * (end - start)
            return float(point[0]), float(point[1])
        remaining_distance -= segment_length

    last = waypoints[-1]
    return float(last[0]), float(last[1])


def make_stub_trigger(trigger_id: str) -> TriggerDecision:
    return TriggerDecision(trigger_id=trigger_id, triggered=False, reason="stub")


def apply_t2_trigger(chain: ChainState, event_id: int) -> TriggerDecision:
    if chain.completed:
        return TriggerDecision(trigger_id="T2", triggered=False, reason="mission_completed")

    center_xy = project_event_center(
        chain.energy_map,
        chain.flight_state.current_xyz,
        remaining_path_snapshot(chain),
        config.EVENT_CENTER_LOOKAHEAD_M,
    )
    update_stats = chain.energy_map.update_wind_field(
        center_xy,
        config.WINDSHEAR_AHEAD_M,
        config.WIND_SHEAR,
    )
    path_candidate_edges = chain.energy_map.find_edges_near_path(
        chain.flight_state.current_xyz,
        remaining_path_snapshot(chain),
        config.T2_LOCAL_REFRESH_RADIUS_M,
        max_waypoints=config.T2_LOCAL_MAX_WAYPOINTS,
    )
    path_edge_set = set(path_candidate_edges)
    candidate_edge_ids = list(update_stats["candidate_edge_ids"])
    old_costs = list(update_stats["old_costs"])
    new_costs = list(update_stats["new_costs"])
    if path_edge_set:
        filtered_indices = [idx for idx, edge_id in enumerate(candidate_edge_ids) if edge_id in path_edge_set]
        candidate_edge_ids = [candidate_edge_ids[idx] for idx in filtered_indices]
        old_costs = [old_costs[idx] for idx in filtered_indices]
        new_costs = [new_costs[idx] for idx in filtered_indices]
    else:
        candidate_edge_ids = []
        old_costs = []
        new_costs = []
    delta_filter = chain.energy_map.filter_edge_cost_delta_values(
        candidate_edge_ids,
        np.asarray(old_costs, dtype=float),
        np.asarray(new_costs, dtype=float),
        config.T2_COST_DELTA_RATIO_THRESHOLD,
        config.T2_COST_DELTA_ABS_THRESHOLD,
    )
    affected_edges = delta_filter["edge_ids"]
    triggered = len(affected_edges) >= config.T2_MIN_AFFECTED_EDGES
    updated_vertices = 0
    if triggered and chain.lpa_planner is not None and not chain.force_planner_reset:
        updated_vertices = chain.lpa_planner.update_edge_costs(affected_edges)
    return TriggerDecision(
        trigger_id="T2",
        triggered=triggered,
        reason="wind_cost_threshold_exceeded" if triggered else "wind_updated_below_threshold",
        metadata={
            "event_id": event_id,
            "center_xy": [float(center_xy[0]), float(center_xy[1])],
            "affected_edges": len(affected_edges),
            "updated_vertices": int(updated_vertices),
            "candidate_edges": int(update_stats["candidate_edges"]),
            "path_candidate_edges": int(len(path_candidate_edges)),
            "path_filtered_candidates": int(len(candidate_edge_ids)),
            "delta_filtered_edges": int(delta_filter["candidate_edges"]),
            "updated_cells": int(update_stats["updated_cells"]),
            "window_rows": int(update_stats["window_rows"]),
            "window_cols": int(update_stats["window_cols"]),
            "max_cost_delta_ratio": float(delta_filter["max_cost_delta_ratio"]),
            "max_cost_delta_abs": float(delta_filter["max_cost_delta_abs"]),
            "min_affected_edges_threshold": int(config.T2_MIN_AFFECTED_EDGES),
            "wind_speed_mps": config.WIND_SHEAR,
            "radius_m": config.WINDSHEAR_AHEAD_M,
        },
    )


def local_schedule_dp_req_w(chain: ChainState) -> float:
    powers: List[float] = []
    if chain.executed_power_w:
        powers.append(float(chain.executed_power_w[-1]))
    elif chain.schedule:
        powers.append(float(chain.schedule[0].power_w))

    for segment in chain.schedule[: config.T3_LOOKAHEAD_SEGMENTS]:
        powers.append(float(segment.power_w))

    if len(powers) < 2:
        return 0.0
    return max_dp_req_w(np.asarray(powers, dtype=float))


def future_profile_dp_req_w(chain: ChainState) -> float:
    if not chain.power_profile_history:
        return 0.0

    profile = chain.power_profile_history[-1]
    time_arr = np.asarray(profile.get("time_s", []), dtype=float)
    power_arr = np.asarray(profile.get("power_w", []), dtype=float)
    if len(time_arr) < 2 or len(power_arr) < 2:
        return 0.0

    start_t = float(chain.flight_state.elapsed_time_s)
    end_t = start_t + config.T3_PROFILE_LOOKAHEAD_S
    mask = (time_arr >= start_t - 1e-9) & (time_arr <= end_t + 1e-9)
    future_power = power_arr[mask]
    if len(future_power) < 2:
        future_idx = np.flatnonzero(time_arr >= start_t - 1e-9)
        future_power = power_arr[future_idx[: max(2, config.T3_LOOKAHEAD_SEGMENTS)]]
    if len(future_power) < 2:
        return 0.0
    return robust_dp_req_w(np.asarray(future_power, dtype=float), config.T3_PROFILE_DP_PERCENTILE)


def apply_t3_trigger(chain: ChainState, event_id: int) -> TriggerDecision:
    local_dp_req = local_schedule_dp_req_w(chain)
    profile_dp_req = future_profile_dp_req_w(chain)
    dp_req = max(local_dp_req, profile_dp_req)
    triggered = dp_req > config.FC_DP_MAX_STEP
    reason = "dp_req_threshold_exceeded" if triggered else "dp_req_within_limit"
    return TriggerDecision(
        trigger_id="T3",
        triggered=triggered,
        reason=reason,
        metadata={
            "event_id": event_id,
            "local_dp_req_w": local_dp_req,
            "future_profile_dp_req_w": profile_dp_req,
            "threshold_w": config.FC_DP_MAX_STEP,
        },
    )


def apply_t4_trigger(chain: ChainState, event_id: int) -> TriggerDecision:
    estimated_soh = float(chain.estimated_soh)
    previous_soh = float(chain.energy_map.soh)
    previous_weight = float(config.k_soh(previous_soh))
    previous_stage = int(config.health_reset_stage(previous_soh))
    health_weight = float(config.k_soh(estimated_soh))
    health_stage = int(config.health_reset_stage(estimated_soh))
    weight_delta = abs(health_weight - previous_weight)
    stress_trigger = chain.online_fc_stress_index >= config.T4_FC_STRESS_TRIGGER
    soh_trigger = estimated_soh <= config.SOH_TRIGGER_THRESHOLD
    weight_trigger = health_weight >= config.T4_HEALTH_WEIGHT_TRIGGER
    triggered = stress_trigger or soh_trigger or weight_trigger

    reasons: List[str] = []
    if stress_trigger:
        reasons.append("fc_stress_threshold_exceeded")
    if soh_trigger:
        reasons.append("soh_threshold_reached")
    if weight_trigger:
        reasons.append("health_weight_threshold_reached")
    if not reasons:
        reasons.append("health_within_limit")

    planner_action = "none"
    local_refreshed_edges = 0
    updated_vertices = 0
    chain.force_planner_reset = False
    if triggered and abs(previous_soh - estimated_soh) > 1e-9:
        chain.energy_map.set_soh(estimated_soh)
        heavy_reset = (
            health_stage != previous_stage
            and weight_delta >= config.T4_HEAVY_WEIGHT_DELTA
        ) or (weight_delta >= config.T4_HEAVY_WEIGHT_DELTA)
        if chain.lpa_planner is not None:
            if heavy_reset:
                chain.force_planner_reset = True
                planner_action = "heavy_reset"
            else:
                local_edges = chain.energy_map.find_edges_near_path(
                    chain.flight_state.current_xyz,
                    remaining_path_snapshot(chain),
                    config.T4_LOCAL_REFRESH_RADIUS_M,
                    max_waypoints=config.T4_LOCAL_MAX_WAYPOINTS,
                )
                delta_filter = chain.energy_map.filter_edges_by_cost_delta(
                    local_edges,
                    previous_soh,
                    estimated_soh,
                    config.T4_COST_DELTA_RATIO_THRESHOLD,
                    config.T4_COST_DELTA_ABS_THRESHOLD,
                )
                updated_vertices = chain.lpa_planner.update_edge_costs(delta_filter["edge_ids"])
                local_refreshed_edges = int(delta_filter["updated_edges"])
                planner_action = "local_refresh" if local_refreshed_edges > 0 else "local_refresh_filtered"
        else:
            planner_action = "global_weight_only"

    return TriggerDecision(
        trigger_id="T4",
        triggered=triggered,
        reason="+".join(reasons),
        metadata={
            "event_id": event_id,
            "estimated_soh": estimated_soh,
            "map_soh": float(chain.energy_map.soh),
            "previous_health_weight": previous_weight,
            "health_weight": health_weight,
            "previous_health_stage": previous_stage,
            "health_stage": health_stage,
            "health_weight_delta": weight_delta,
            "online_fc_stress_index": float(chain.online_fc_stress_index),
            "stress_threshold": config.T4_FC_STRESS_TRIGGER,
            "soh_threshold": config.SOH_TRIGGER_THRESHOLD,
            "planner_reset": bool(chain.force_planner_reset),
            "planner_action": planner_action,
            "candidate_local_edges": int(delta_filter["candidate_edges"]) if "delta_filter" in locals() else 0,
            "local_refreshed_edges": local_refreshed_edges,
            "updated_vertices": int(updated_vertices),
            "max_cost_delta_ratio": float(delta_filter["max_cost_delta_ratio"]) if "delta_filter" in locals() else 0.0,
            "max_cost_delta_abs": float(delta_filter["max_cost_delta_abs"]) if "delta_filter" in locals() else 0.0,
        },
    )


def assert_trigger_decisions(decisions: List[TriggerDecision]) -> None:
    trigger_ids = [decision.trigger_id for decision in decisions]
    assert trigger_ids == ["T1", "T2", "T3", "T4"], f"Unexpected trigger set: {trigger_ids}"


def evaluate_triggers(chain: ChainState, event_id: int) -> List[TriggerDecision]:
    decisions = [
        make_stub_trigger("T1"),
        apply_t2_trigger(chain, event_id),
        apply_t3_trigger(chain, event_id),
        apply_t4_trigger(chain, event_id),
    ]
    assert_trigger_decisions(decisions)
    return decisions


def compute_chain_path(chain: ChainState, current_node: int) -> Tuple[bool, List[int], float, Dict[str, float]]:
    if chain.lpa_planner is not None:
        if chain.force_planner_reset:
            chain.lpa_planner = LPAStar(chain.energy_map, chain.goal_node_id)
            chain.force_planner_reset = False
        key_shift_before = chain.lpa_planner.key_shift_count
        t_search = time.perf_counter()
        found = chain.lpa_planner.compute_shortest_path(current_node)
        compute_shortest_path_ms = (time.perf_counter() - t_search) * 1000.0
        t_extract = time.perf_counter()
        new_path = chain.lpa_planner.extract_path(current_node) if found else []
        path_extract_ms = (time.perf_counter() - t_extract) * 1000.0
        planning_latency_ms = compute_shortest_path_ms + path_extract_ms
        return found, new_path, planning_latency_ms, {
            "planner_nodes_expanded": int(chain.lpa_planner.nodes_expanded),
            "lpa_nodes_expanded": int(chain.lpa_planner.nodes_expanded),
            "lpa_heap_rekey_count": int(chain.lpa_planner.key_shift_count - key_shift_before),
            "compute_shortest_path_ms": float(compute_shortest_path_ms),
            "path_extract_ms": float(path_extract_ms),
        }

    if chain.astar_planner is not None:
        found, new_path, expanded, compute_shortest_path_ms = chain.astar_planner.plan(current_node, chain.goal_node_id)
        return found, new_path, compute_shortest_path_ms, {
            "planner_nodes_expanded": int(expanded),
            "lpa_nodes_expanded": 0,
            "lpa_heap_rekey_count": 0,
            "compute_shortest_path_ms": float(compute_shortest_path_ms),
            "path_extract_ms": 0.0,
        }

    raise RuntimeError(f"Chain {chain.name} has no planner.")


def assert_event_record(chain: ChainState, event_record: EventRecord) -> None:
    trigger_ids = [decision.trigger_id for decision in event_record.decisions]
    any_triggered = any(decision.triggered for decision in event_record.decisions)
    assert trigger_ids == ["T1", "T2", "T3", "T4"], f"Missing triggers in event {event_record.event_id}."
    assert event_record.t2_trigger_s <= event_record.t3_message_ready_s + 1e-9
    assert event_record.t3_message_ready_s <= event_record.t4_ems_ready_s + 1e-9
    assert event_record.t4_ems_ready_s <= event_record.t5_flight_execute_s + 1e-9
    if chain.name == "proposed":
        assert event_record.structured_message is not None, "Proposed event missing structured message."
        if any_triggered:
            assert event_record.t4_ems_ready_s - event_record.t3_message_ready_s >= config.FC_TAU - 1e-9
    else:
        assert event_record.structured_message is None, "Traditional chain must not depend on structured message."


def run_planning_stage(
    chain: ChainState,
    replanning_id: int,
    trigger_time_s: float,
    is_event: bool,
    trigger_decisions: List[TriggerDecision] | None = None,
) -> None:
    current_node = chain.flight_state.current_node_id
    current_xyz = tuple(chain.flight_state.current_xyz)
    decisions = trigger_decisions or []
    triggered = (not is_event) or any(decision.triggered for decision in decisions)
    planner_stats = {
        "planner_nodes_expanded": 0,
        "lpa_nodes_expanded": 0,
        "lpa_heap_rekey_count": 0,
        "compute_shortest_path_ms": 0.0,
        "path_extract_ms": 0.0,
    }

    if triggered:
        found, new_path, planning_latency_ms, planner_stats = compute_chain_path(chain, current_node)
        if not found or not new_path:
            raise RuntimeError(f"{chain.name} planner failed at stage {replanning_id}.")
    else:
        new_path = remaining_path_snapshot(chain)
        planning_latency_ms = 0.0

    t2_trigger_s = trigger_time_s if is_event else 0.0
    t3_message_ready_s = trigger_time_s + config.CONTROL_MESSAGE_DELAY_S if is_event else 0.0
    wait_for_activation = chain.name == "proposed" and is_event and triggered
    if wait_for_activation:
        t4_ems_ready_s = t3_message_ready_s + config.FC_TAU
    else:
        t4_ems_ready_s = t3_message_ready_s
    t5_flight_execute_s = t4_ems_ready_s
    chain_latency_ms = (
        max(0.0, t5_flight_execute_s - trigger_time_s) * 1000.0
        if is_event
        else 0.0
    )

    profile_payload = build_profile_payload(
        chain.energy_map,
        new_path,
        t5_flight_execute_s,
    )

    structured_message = None
    if chain.name == "proposed":
        structured_message = build_structured_message(
            timestamp=t3_message_ready_s,
            replanning_id=replanning_id,
            feature_vec=profile_payload["feature_vector"],
            time_arr=np.asarray(profile_payload["time_s"], dtype=float),
            power_arr=np.asarray(profile_payload["power_w"], dtype=float),
            t_window=float(profile_payload["t_window"]),
        )
        chain.structured_messages.append(structured_message)

    snap_error_m = 0.0
    if not wait_for_activation:
        snap_error_m = apply_schedule_from_path(chain, current_xyz, new_path)

    chain.planning_latency_ms.append(planning_latency_ms)
    if is_event:
        chain.chain_latency_ms.append(chain_latency_ms)

    path_len = float(profile_payload["path_length_m"])
    chain.power_profile_history.append(
        {
            "replanning_id": replanning_id,
            "trigger_time_s": trigger_time_s,
            "flight_execute_time_s": t5_flight_execute_s,
            "triggered": triggered,
            "decisions": [asdict(decision) for decision in decisions],
            "path_nodes": list(new_path),
            "path_length_m": path_len,
            "time_s": list(profile_payload["time_s"]),
            "power_w": list(profile_payload["power_w"]),
            "max_dp_req_w": float(profile_payload["max_dp_req_w"]),
            "feature_vector": profile_payload["feature_vector"],
            "structured_message": structured_message,
        }
    )

    if not is_event:
        chain.initial_path_nodes = list(new_path)
        chain.initial_path_length_m = path_len
        chain.initial_phases = classify_flight_phases(chain.energy_map, new_path)
        chain.initial_plan_ms = planning_latency_ms
        return

    event_record = EventRecord(
        event_id=replanning_id,
        trigger_time_s=trigger_time_s,
        decisions=decisions,
        t2_trigger_s=t2_trigger_s,
        t3_message_ready_s=t3_message_ready_s,
        t4_ems_ready_s=t4_ems_ready_s,
        t5_flight_execute_s=t5_flight_execute_s,
        planning_latency_ms=planning_latency_ms,
        chain_latency_ms=chain_latency_ms,
        current_xyz=current_xyz,
        current_node_id=current_node,
        snap_error_m=snap_error_m,
        new_path_length_m=path_len,
        new_path_nodes=list(new_path),
        structured_message=structured_message,
        t2_changed_nodes=int(
            next(
                (
                    decision.metadata.get("updated_vertices", 0)
                    for decision in decisions
                    if decision.trigger_id == "T2"
                ),
                0,
            )
        ),
        t4_changed_nodes=int(
            next(
                (
                    decision.metadata.get("updated_vertices", 0)
                    for decision in decisions
                    if decision.trigger_id == "T4"
                ),
                0,
            )
        ),
        planner_nodes_expanded=int(planner_stats["planner_nodes_expanded"]),
        lpa_nodes_expanded=int(planner_stats["lpa_nodes_expanded"]),
        lpa_heap_rekey_count=int(planner_stats["lpa_heap_rekey_count"]),
        compute_shortest_path_ms=float(planner_stats["compute_shortest_path_ms"]),
        path_extract_ms=float(planner_stats["path_extract_ms"]),
    )
    assert_event_record(chain, event_record)
    chain.event_records.append(event_record)

    if wait_for_activation:
        chain.pending_plan = PendingPlan(
            replanning_id=replanning_id,
            activation_time_s=t5_flight_execute_s,
            path_nodes=list(new_path),
            profile_history_index=len(chain.power_profile_history) - 1,
            event_record_index=len(chain.event_records) - 1,
        )


def initialize_chain(
    name: str,
    base_map: EnergyMap,
    start_node: int,
    goal_node: int,
) -> ChainState:
    energy_map = base_map.clone_dynamic_state()
    flight_state = FlightState(
        elapsed_time_s=0.0,
        current_xyz=energy_map.position_from_node(start_node),
        current_node_id=start_node,
        path_nodes=[],
        remaining_path_nodes=[],
    )

    if name == "proposed":
        chain = ChainState(
            name=name,
            energy_map=energy_map,
            goal_node_id=goal_node,
            flight_state=flight_state,
            lpa_planner=LPAStar(energy_map, goal_node),
            estimated_soh=float(energy_map.soh),
        )
    else:
        chain = ChainState(
            name=name,
            energy_map=energy_map,
            goal_node_id=goal_node,
            flight_state=flight_state,
            astar_planner=AStarPlanner(energy_map),
            estimated_soh=float(energy_map.soh),
        )

    run_planning_stage(chain, replanning_id=0, trigger_time_s=0.0, is_event=False)
    return chain


def build_chain_results(chain: ChainState) -> Dict[str, object]:
    time_arr = np.array(chain.executed_time_s, dtype=float)
    power_arr = np.array(chain.executed_power_w, dtype=float)
    event_planning_latency_ms = [event.planning_latency_ms for event in chain.event_records]
    event_chain_latency_ms = [event.chain_latency_ms for event in chain.event_records]
    total_pre_adjust_time_s = sum(
        max(0.0, event.t4_ems_ready_s - event.t3_message_ready_s) for event in chain.event_records
    )

    if chain.name == "proposed":
        ems_result = EMSController().simulate(time_arr, power_arr, chain.structured_messages)
    else:
        ems_result = PassiveEMS().simulate(time_arr, power_arr)

    final_planned_path = chain.flight_state.path_nodes
    result = {
        "initial_path_nodes": chain.initial_path_nodes,
        "initial_path_length_m": chain.initial_path_length_m,
        "final_path_nodes": final_planned_path,
        "final_planned_path_length_m": path_length_m(chain.energy_map, final_planned_path),
        "executed_distance_m": chain.executed_distance_m,
        "avg_planning_latency_ms": mean_or_zero(chain.planning_latency_ms),
        "avg_chain_latency_ms": mean_or_zero(chain.chain_latency_ms),
        "avg_event_planning_latency_ms": mean_or_zero(event_planning_latency_ms),
        "avg_event_chain_latency_ms": mean_or_zero(event_chain_latency_ms),
        "initial_plan_ms": chain.initial_plan_ms,
        "total_pre_adjust_time_s": float(total_pre_adjust_time_s),
        "max_dp_req_w": max_dp_req_w(power_arr),
        "observed_max_dp_req_w": float(chain.observed_max_dp_req_w),
        "estimated_soh_end": float(chain.estimated_soh),
        "online_fc_stress_index": float(chain.online_fc_stress_index),
        "power_profile": {
            "time_s": [float(t) for t in time_arr.tolist()],
            "power_w": [float(p) for p in power_arr.tolist()],
        },
        "structured_messages": chain.structured_messages,
        "events": [asdict(event) for event in chain.event_records],
        "power_profile_history": chain.power_profile_history,
        "initial_phases": chain.initial_phases,
        "planner": "LPA*" if chain.name == "proposed" else "A*",
        "completed": chain.completed,
    }
    result.update(ems_result)
    return result


def build_comparison(proposed_results: Dict[str, object], traditional_results: Dict[str, object]) -> Dict[str, float]:
    return {
        "speedup_ratio": (
            traditional_results["avg_chain_latency_ms"] / proposed_results["avg_chain_latency_ms"]
            if proposed_results["avg_chain_latency_ms"] > 0.0
            else 0.0
        ),
        "planning_speedup_ratio": (
            traditional_results["avg_planning_latency_ms"] / proposed_results["avg_planning_latency_ms"]
            if proposed_results["avg_planning_latency_ms"] > 0.0
            else 0.0
        ),
        "event_speedup_ratio": (
            traditional_results["avg_event_chain_latency_ms"] / proposed_results["avg_event_chain_latency_ms"]
            if proposed_results["avg_event_chain_latency_ms"] > 0.0
            else 0.0
        ),
        "event_planning_speedup_ratio": (
            traditional_results["avg_event_planning_latency_ms"] / proposed_results["avg_event_planning_latency_ms"]
            if proposed_results["avg_event_planning_latency_ms"] > 0.0
            else 0.0
        ),
    }


def summarize_scenario_for_sweep(results_data: Dict[str, object]) -> Dict[str, object]:
    proposed = results_data["chains"]["proposed"]
    traditional = results_data["chains"]["traditional"]
    metrics = [
        "h2_total_g",
        "min_bus_voltage_v",
        "battery_stress_index_as",
        "fc_stress_index",
        "max_dp_req_w",
        "avg_planning_latency_ms",
        "avg_chain_latency_ms",
    ]
    return {
        "label": results_data["scenario"]["label"],
        "config_snapshot": results_data["config_snapshot"],
        "proposed": {metric: proposed[metric] for metric in metrics},
        "traditional": {metric: traditional[metric] for metric in metrics},
        "comparison": results_data["comparison"],
    }


def build_report_tables(results_data: Dict[str, object]) -> Dict[str, object]:
    proposed = results_data["chains"]["proposed"]
    traditional = results_data["chains"]["traditional"]
    table2 = [
        {"metric": "executed_distance_m", "traditional": traditional["executed_distance_m"], "proposed": proposed["executed_distance_m"]},
        {"metric": "h2_total_g", "traditional": traditional["h2_total_g"], "proposed": proposed["h2_total_g"]},
        {"metric": "max_dp_req_w", "traditional": traditional["max_dp_req_w"], "proposed": proposed["max_dp_req_w"]},
        {"metric": "battery_stress_index_as", "traditional": traditional["battery_stress_index_as"], "proposed": proposed["battery_stress_index_as"]},
        {"metric": "fc_stress_index", "traditional": traditional["fc_stress_index"], "proposed": proposed["fc_stress_index"]},
        {"metric": "avg_planning_latency_ms", "traditional": traditional["avg_planning_latency_ms"], "proposed": proposed["avg_planning_latency_ms"]},
        {"metric": "avg_chain_latency_ms", "traditional": traditional["avg_chain_latency_ms"], "proposed": proposed["avg_chain_latency_ms"]},
        {"metric": "min_bus_voltage_v", "traditional": traditional["min_bus_voltage_v"], "proposed": proposed["min_bus_voltage_v"]},
    ]

    sweep_rows = []
    for label, entry in results_data["parameter_sweep"].items():
        sweep_rows.append(
            {
                "label": label,
                "proposed_h2_total_g": entry["proposed"]["h2_total_g"],
                "traditional_h2_total_g": entry["traditional"]["h2_total_g"],
                "proposed_min_bus_voltage_v": entry["proposed"]["min_bus_voltage_v"],
                "traditional_min_bus_voltage_v": entry["traditional"]["min_bus_voltage_v"],
                "proposed_battery_stress_index_as": entry["proposed"]["battery_stress_index_as"],
                "traditional_battery_stress_index_as": entry["traditional"]["battery_stress_index_as"],
                "proposed_fc_stress_index": entry["proposed"]["fc_stress_index"],
                "traditional_fc_stress_index": entry["traditional"]["fc_stress_index"],
                "proposed_max_dp_req_w": entry["proposed"]["max_dp_req_w"],
                "traditional_max_dp_req_w": entry["traditional"]["max_dp_req_w"],
                "planning_speedup_ratio": entry["comparison"]["planning_speedup_ratio"],
            }
        )

    return {
        "primary_parameter_set_label": results_data["primary_parameter_set_label"],
        "table1": {
            "proposed_initial_phases": proposed["initial_phases"],
            "traditional_initial_phases": traditional["initial_phases"],
        },
        "table2": table2,
        "parameter_sweep": sweep_rows,
    }


@contextmanager
def temporary_config(overrides: Dict[str, float]) -> Iterator[None]:
    previous = {key: getattr(config, key) for key in overrides}
    try:
        for key, value in overrides.items():
            setattr(config, key, value)
        config.WIND_TRIGGER_DELTA_COST_RATIO = config.WIND_TRIGGER_RATIO - 1.0
        yield
    finally:
        for key, value in previous.items():
            setattr(config, key, value)
        config.WIND_TRIGGER_DELTA_COST_RATIO = config.WIND_TRIGGER_RATIO - 1.0


def assert_chain_independence(proposed: ChainState, traditional: ChainState) -> None:
    assert proposed is not traditional
    assert proposed.schedule is not traditional.schedule
    assert proposed.flight_state.path_nodes is not traditional.flight_state.path_nodes
    assert proposed.power_profile_history is not traditional.power_profile_history
    assert proposed.event_records is not traditional.event_records
    assert proposed.structured_messages is not traditional.structured_messages


def assert_metric_helpers() -> None:
    sample = np.array([10.0, 16.0, 7.0], dtype=float)
    assert math.isclose(max_dp_req_w(sample), 9.0)


def run_scenario(
    label: str,
    base_map: EnergyMap,
    start_node: int,
    goal_node: int,
    *,
    verbose: bool,
) -> Dict[str, object]:
    proposed = initialize_chain("proposed", base_map, start_node, goal_node)
    traditional = initialize_chain("traditional", base_map, start_node, goal_node)
    assert_chain_independence(proposed, traditional)

    if verbose:
        print(f"Scenario: {label}")
        print(f"  Graph nodes: {len(base_map.nodes)}")
        print(f"  Graph edges: {len(base_map.edges)}")
        print(f"  Initial proposed path length: {proposed.initial_path_length_m:.1f} m")
        print(f"  Initial traditional path length: {traditional.initial_path_length_m:.1f} m")

    event_times = np.cumsum(np.asarray(config.EVENT_INTERVALS, dtype=float))
    for event_id, event_time in enumerate(event_times, start=1):
        if verbose:
            print(f"\nEvent {event_id} at t={event_time:.1f}s")
        for chain in (proposed, traditional):
            advance_chain_to_time(chain, float(event_time))
            if chain.completed:
                if verbose:
                    print(f"  {chain.name}: already completed before event")
                continue
            decisions = evaluate_triggers(chain, event_id)
            run_planning_stage(
                chain,
                replanning_id=event_id,
                trigger_time_s=float(event_time),
                is_event=True,
                trigger_decisions=decisions,
            )
            if verbose:
                decision_summary = ", ".join(
                    f"{decision.trigger_id}={'on' if decision.triggered else 'off'}" for decision in decisions
                )
                print(
                    f"  {chain.name}: node={chain.flight_state.current_node_id}, "
                    f"path_len={path_length_m(chain.energy_map, chain.flight_state.path_nodes):.1f} m, "
                    f"latency={chain.chain_latency_ms[-1]:.1f} ms, triggers=[{decision_summary}]"
                )

    while not proposed.completed and (proposed.schedule or proposed.pending_plan is not None):
        if proposed.pending_plan is not None:
            next_target = proposed.pending_plan.activation_time_s
            if proposed.schedule:
                next_target = min(next_target, proposed.flight_state.elapsed_time_s + proposed.schedule[0].duration_s)
        else:
            next_target = proposed.flight_state.elapsed_time_s + proposed.schedule[0].duration_s
        advance_chain_to_time(proposed, next_target)

    while not traditional.completed and (traditional.schedule or traditional.pending_plan is not None):
        if traditional.pending_plan is not None:
            next_target = traditional.pending_plan.activation_time_s
            if traditional.schedule:
                next_target = min(next_target, traditional.flight_state.elapsed_time_s + traditional.schedule[0].duration_s)
        else:
            next_target = traditional.flight_state.elapsed_time_s + traditional.schedule[0].duration_s
        advance_chain_to_time(traditional, next_target)

    proposed_results = build_chain_results(proposed)
    traditional_results = build_chain_results(traditional)
    comparison = build_comparison(proposed_results, traditional_results)

    return {
        "config_snapshot": config.snapshot(),
        "scenario": {
            "label": label,
            "graph_nodes": len(base_map.nodes),
            "graph_edges": len(base_map.edges),
            "events_count": config.N_EVENTS,
            "event_times_s": [float(t) for t in event_times.tolist()],
        },
        "chains": {
            "proposed": proposed_results,
            "traditional": traditional_results,
        },
        "comparison": comparison,
    }


def main() -> None:
    print("=" * 68)
    print("Dual-chain event-driven simulation")
    print("=" * 68)

    assert_metric_helpers()

    z_dem, lon_grid, lat_grid = load_dem()
    base_map = EnergyMap(z_dem, lon_grid, lat_grid, soh=config.INITIAL_SOH)
    base_map.build_graph()

    start_node = base_map.find_nearest_node(config.START_LON, config.START_LAT, config.START_ALT)
    goal_node = base_map.find_nearest_node(config.GOAL_LON, config.GOAL_LAT, config.GOAL_ALT)

    scenario_results: Dict[str, Dict[str, object]] = {}
    sweep_labels = tuple(config.SWEEP_PRESETS.keys())
    for label in sweep_labels:
        with temporary_config(config.SWEEP_PRESETS[label]):
            scenario_results[label] = run_scenario(
                label,
                base_map,
                start_node,
                goal_node,
                verbose=(label == config.PRIMARY_SWEEP_LABEL),
            )

    primary_results = scenario_results[config.PRIMARY_SWEEP_LABEL]
    parameter_sweep = {
        label: summarize_scenario_for_sweep(data)
        for label, data in scenario_results.items()
    }

    results_data = {
        "config_snapshot": primary_results["config_snapshot"],
        "scenario": primary_results["scenario"],
        "chains": primary_results["chains"],
        "comparison": primary_results["comparison"],
        "primary_parameter_set_label": config.PRIMARY_SWEEP_LABEL,
        "parameter_sweep": parameter_sweep,
    }
    report_tables = build_report_tables(results_data)

    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    config.SIM_RESULT_FILE.write_text(json.dumps(results_data, indent=2, ensure_ascii=False), encoding="utf-8")
    config.REPORT_TABLES_FILE.write_text(json.dumps(report_tables, indent=2, ensure_ascii=False), encoding="utf-8")

    assert config.SIM_RESULT_FILE.exists(), "simulation_results.json was not generated."
    assert config.REPORT_TABLES_FILE.exists(), "report_tables.json was not generated."

    proposed_results = primary_results["chains"]["proposed"]
    traditional_results = primary_results["chains"]["traditional"]
    comparison = primary_results["comparison"]

    print("\nSummary")
    print(f"  Proposed H2: {proposed_results['h2_total_g']:.3f} g")
    print(f"  Traditional H2: {traditional_results['h2_total_g']:.3f} g")
    print(f"  Proposed avg chain latency: {proposed_results['avg_chain_latency_ms']:.3f} ms")
    print(f"  Traditional avg chain latency: {traditional_results['avg_chain_latency_ms']:.3f} ms")
    print(f"  Speedup ratio: {comparison['speedup_ratio']:.3f}")
    print(f"  Planning speedup ratio: {comparison['planning_speedup_ratio']:.3f}")
    print(f"  Results: {config.SIM_RESULT_FILE}")
    print(f"  Report tables: {config.REPORT_TABLES_FILE}")


if __name__ == "__main__":
    main()



