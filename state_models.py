"""跨模块共享的状态对象。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List

import config


@dataclass
class PlannerState:
    current_path_nodes: List[int] = field(default_factory=list)
    replanning_id: int = 0
    planning_latency_ms: float = 0.0
    trigger_reason: str = ""


@dataclass
class EnergyState:
    fc_power_w: float = 0.0
    fc_target_power_w: float = 0.0
    battery_current_a: float = 0.0
    battery_current_limit_a: float = config.BATTERY_CURRENT_LIMIT_INIT_A
    bus_voltage_v: float = config.V_BUS
    preconditioning_target_w: float = 0.0
    preconditioning_complete: bool = True


@dataclass
class HealthState:
    soh: float = config.INITIAL_SOH
    stress_fc_online: float = 0.0


@dataclass
class TriggerState:
    replanning_id: int = 0
    last_trigger_time_s: float = -1.0e9
    min_trigger_interval_s: float = config.TRIGGER_MIN_INTERVAL_S
    trigger_reason: str = ""
    next_disturbance_index: int = 0
    next_obstacle_event_index: int = 0
    t3_condition_active: bool = False
    t4_last_stress_checkpoint: float = 0.0
    last_t4_replan_accept_s: float = -1.0e9
    observed_health_stage: int = 0
    last_t4_preview_stage: int = -1
    last_t4_preview_gain_abs: float = 0.0
    last_t4_preview_gain_ratio: float = 0.0
    last_t4_preview_path_changed: bool = False
    event_merge_lock_until_s: float = -1.0e9
    pending_wind_updates: List[Dict[str, object]] = field(default_factory=list)
    pending_obstacle_updates: List[Dict[str, object]] = field(default_factory=list)


@dataclass
class StructuredMessage:
    timestamp: float
    feature_vector: Dict[str, float]
    P_predict: Dict[str, List[float]]
    T_window: float
    meta: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        """转换为可序列化字典。"""
        return asdict(self)
