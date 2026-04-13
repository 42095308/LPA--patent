"""EMS 单步更新与结果汇总。"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

import config
from state_models import EnergyState, HealthState


def _h2_consumption(power_w: float, dt: float) -> float:
    """将燃料电池电功率换算为氢耗。"""
    if power_w <= 0.0 or dt <= 0.0:
        return 0.0
    eta_fc = 0.50
    lhv_h2 = 120.0e3
    return power_w / (eta_fc * lhv_h2) * dt


def calc_fc_step_stress_increment(dp_fc: float, dt: float) -> float:
    """按说明书公式计算单步燃料电池应力增量。"""
    ramp = dp_fc / max(dt, 1e-6)
    return (
        dp_fc / max(config.FC_STRESS_POWER_STEP_REF_W, 1e-6)
        + ramp / max(config.FC_STRESS_RAMP_REF_W_PER_S, 1e-6) * dt
    )


def _message_available_time(message: Dict[str, object]) -> float:
    meta = message.get("meta", {})
    if isinstance(meta, dict) and "t_msg_send_s" in meta:
        return float(meta["t_msg_send_s"])
    return float(message["timestamp"])


def _active_message(messages: List[Dict[str, object]], current_time_s: float) -> Dict[str, object] | None:
    active_message = None
    for message in messages:
        if _message_available_time(message) <= current_time_s + 1e-9:
            active_message = message
        else:
            break
    return active_message


def _message_preconditioning_scale(message: Dict[str, object] | None) -> float:
    if not isinstance(message, dict):
        return config.PRECONDITION_EVENT_SCALE_T3
    meta = message.get("meta", {})
    trigger_ids = meta.get("trigger_ids", []) if isinstance(meta, dict) else []
    if not isinstance(trigger_ids, list):
        trigger_ids = []
    trigger_id_set = {str(trigger_id) for trigger_id in trigger_ids}
    if "T3" in trigger_id_set:
        return config.PRECONDITION_EVENT_SCALE_T3
    if "T2" in trigger_id_set:
        return config.PRECONDITION_EVENT_SCALE_T2
    if trigger_id_set == {"T4"}:
        return config.PRECONDITION_EVENT_SCALE_T4
    return config.PRECONDITION_EVENT_SCALE_T3


def _release_profile(trigger_ids: List[str] | None) -> Dict[str, float | str]:
    trigger_id_set = {str(trigger_id) for trigger_id in (trigger_ids or [])}
    if "T3" in trigger_id_set:
        return {
            "name": "t3",
            "scale": float(config.PRECONDITION_EVENT_SCALE_T3),
            "release_power_tol_w": float(
                config.PRECONDITION_POWER_TOLERANCE_W * config.PRECONDITION_RELEASE_POWER_TOL_MULT_T3
            ),
            "release_headroom_ratio": float(
                config.PRECONDITION_BATTERY_HEADROOM_RATIO * config.PRECONDITION_RELEASE_HEADROOM_RATIO_MULT_T3
            ),
            "release_headroom_a": float(
                config.PRECONDITION_BATTERY_HEADROOM_A * config.PRECONDITION_RELEASE_HEADROOM_ABS_MULT_T3
            ),
            "min_dwell_s": float(config.PRECONDITION_RELEASE_MIN_DWELL_S_T3),
            "voltage_guard_v": float(config.PRECONDITION_RELEASE_VOLTAGE_GUARD_T3_V),
        }
    if "T2" in trigger_id_set:
        return {
            "name": "t2",
            "scale": float(config.PRECONDITION_EVENT_SCALE_T2),
            "release_power_tol_w": float(
                config.PRECONDITION_POWER_TOLERANCE_W * config.PRECONDITION_RELEASE_POWER_TOL_MULT_T2
            ),
            "release_headroom_ratio": float(
                config.PRECONDITION_BATTERY_HEADROOM_RATIO * config.PRECONDITION_RELEASE_HEADROOM_RATIO_MULT_T2
            ),
            "release_headroom_a": float(
                config.PRECONDITION_BATTERY_HEADROOM_A * config.PRECONDITION_RELEASE_HEADROOM_ABS_MULT_T2
            ),
            "min_dwell_s": float(config.PRECONDITION_RELEASE_MIN_DWELL_S_T2),
            "voltage_guard_v": float(config.PRECONDITION_RELEASE_VOLTAGE_GUARD_T2_V),
        }
    if trigger_id_set == {"T4"}:
        return {
            "name": "t4",
            "scale": float(config.PRECONDITION_EVENT_SCALE_T4),
            "release_power_tol_w": float(
                config.PRECONDITION_POWER_TOLERANCE_W * config.PRECONDITION_RELEASE_POWER_TOL_MULT_T4
            ),
            "release_headroom_ratio": float(
                config.PRECONDITION_BATTERY_HEADROOM_RATIO * config.PRECONDITION_RELEASE_HEADROOM_RATIO_MULT_T4
            ),
            "release_headroom_a": float(
                config.PRECONDITION_BATTERY_HEADROOM_A * config.PRECONDITION_RELEASE_HEADROOM_ABS_MULT_T4
            ),
            "min_dwell_s": float(config.PRECONDITION_RELEASE_MIN_DWELL_S_T4),
            "voltage_guard_v": float(config.PRECONDITION_RELEASE_VOLTAGE_GUARD_T4_V),
        }
    return {
        "name": "default",
        "scale": float(config.PRECONDITION_EVENT_SCALE_T3),
        "release_power_tol_w": float(
            config.PRECONDITION_POWER_TOLERANCE_W * config.PRECONDITION_RELEASE_POWER_TOL_MULT_T3
        ),
        "release_headroom_ratio": float(
            config.PRECONDITION_BATTERY_HEADROOM_RATIO * config.PRECONDITION_RELEASE_HEADROOM_RATIO_MULT_T3
        ),
        "release_headroom_a": float(
            config.PRECONDITION_BATTERY_HEADROOM_A * config.PRECONDITION_RELEASE_HEADROOM_ABS_MULT_T3
        ),
        "min_dwell_s": float(config.PRECONDITION_RELEASE_MIN_DWELL_S_T3),
        "voltage_guard_v": float(config.PRECONDITION_RELEASE_VOLTAGE_GUARD_T3_V),
    }


def lookup_future_demand(messages: List[Dict[str, object]], current_time_s: float) -> float:
    """查找当前时刻可见消息对应的前瞻需求功率。"""
    active_message = _active_message(messages, current_time_s)
    if active_message is None:
        return 0.0

    predict = active_message["P_predict"]
    predict_time = np.asarray(predict["time_s"], dtype=float)
    predict_power = np.asarray(predict["power_w"], dtype=float)
    if len(predict_time) == 0:
        return 0.0

    lookahead_t = current_time_s + config.FC_TAU * config.EMS_MESSAGE_LOOKAHEAD_FACTOR
    window_end_t = lookahead_t + config.FC_PREVIEW_AVG_WINDOW_S
    mask = (predict_time >= lookahead_t - 1e-9) & (predict_time <= window_end_t + 1e-9)
    if np.any(mask):
        return float(np.mean(predict_power[mask]))
    return float(
        np.interp(
            lookahead_t,
            predict_time,
            predict_power,
            left=predict_power[0],
            right=predict_power[-1],
        )
    )


def _estimate_bus_voltage(fc_power_w: float, battery_current_a: float, resistance_ohm: float) -> float:
    """根据等效供电电流估计母线电压。"""
    i_eq = fc_power_w / max(config.V_BUS, 1e-6) + battery_current_a
    return max(1.0, config.V_BUS - i_eq * resistance_ohm)


def _solve_battery_current(
    p_gap_w: float,
    fc_power_w: float,
    current_limit_a: float,
    resistance_ohm: float,
) -> Tuple[float, float, bool]:
    """在电流限值和母线电压闭环下求解电池电流。"""
    v_bus = config.V_BUS
    i_cmd = p_gap_w / max(v_bus, 1e-6)
    i_bat = min(i_cmd, current_limit_a)
    v_bus = _estimate_bus_voltage(fc_power_w, i_bat, resistance_ohm)

    i_cmd = p_gap_w / max(v_bus, 1e-6)
    i_bat = min(i_cmd, current_limit_a)
    v_bus = _estimate_bus_voltage(fc_power_w, i_bat, resistance_ohm)
    under_supply = i_cmd > i_bat + 1e-9
    return float(i_bat), float(v_bus), bool(under_supply)


def _update_battery_limit(
    energy_state: EnergyState,
    p_future_w: float,
    fc_power_w: float,
) -> float:
    """根据未来功率缺口更新电池电流限值。"""
    preview_gap = max(0.0, p_future_w - fc_power_w)
    desired_limit = max(
        config.BAT_MIN_CURRENT,
        preview_gap / max(energy_state.bus_voltage_v, 1e-6),
    )
    energy_state.battery_current_limit_a = max(desired_limit, config.BAT_MIN_CURRENT)
    return float(energy_state.battery_current_limit_a)


def preconditioning_status(
    energy_state: EnergyState,
    current_time_s: float,
    message_send_time_s: float,
    trigger_ids: List[str] | None = None,
    previous_power_error_w: float | None = None,
    previous_eval_time_s: float | None = None,
) -> Dict[str, float | bool]:
    """返回预调放行状态与各阶段门控。"""
    release_profile = _release_profile(trigger_ids)
    elapsed_s = max(0.0, current_time_s - message_send_time_s)
    release_elapsed_s = max(0.0, elapsed_s - config.FC_TAU)
    time_ready = elapsed_s >= config.FC_TAU - 1e-9
    power_error_w = abs(energy_state.fc_power_w - energy_state.preconditioning_target_w)
    strict_power_ready = power_error_w <= config.PRECONDITION_POWER_TOLERANCE_W
    release_power_ready = power_error_w <= float(release_profile["release_power_tol_w"])
    battery_headroom_a = max(0.0, energy_state.battery_current_limit_a - energy_state.battery_current_a)
    strict_headroom_threshold_a = max(
        config.PRECONDITION_BATTERY_HEADROOM_A,
        energy_state.battery_current_limit_a * config.PRECONDITION_BATTERY_HEADROOM_RATIO,
    )
    release_headroom_threshold_a = max(
        float(release_profile["release_headroom_a"]),
        energy_state.battery_current_limit_a * float(release_profile["release_headroom_ratio"]),
    )
    strict_battery_ready = battery_headroom_a >= strict_headroom_threshold_a
    release_battery_ready = battery_headroom_a >= release_headroom_threshold_a
    remaining_power_deficit_w = max(
        0.0,
        energy_state.preconditioning_target_w - energy_state.fc_power_w,
    )
    battery_buffer_w = max(
        0.0,
        battery_headroom_a * energy_state.bus_voltage_v * config.PRECONDITION_BATTERY_BUFFER_MARGIN,
    )
    battery_buffer_ready = remaining_power_deficit_w <= battery_buffer_w + 1e-9
    voltage_guard_v = float(release_profile["voltage_guard_v"])
    voltage_ready = energy_state.bus_voltage_v >= voltage_guard_v - 1e-9
    dwell_ready = release_elapsed_s >= float(release_profile["min_dwell_s"]) - 1e-9

    if (
        previous_power_error_w is None
        or previous_eval_time_s is None
        or current_time_s <= previous_eval_time_s + 1e-9
    ):
        power_error_trend_w_per_s = 0.0
        trend_ready = bool(strict_power_ready)
    else:
        dt_eval_s = max(current_time_s - previous_eval_time_s, 1e-6)
        power_error_trend_w_per_s = (power_error_w - previous_power_error_w) / dt_eval_s
        trend_ready = bool(
            power_error_trend_w_per_s <= config.PRECONDITION_TREND_RELEASE_EPS_W_PER_S
            or strict_power_ready
        )

    strict_release_ready = bool(time_ready and strict_power_ready and strict_battery_ready and voltage_ready)
    release_ready = bool(
        strict_release_ready
        or (
            time_ready
            and dwell_ready
            and voltage_ready
            and release_power_ready
            and trend_ready
            and (release_battery_ready or battery_buffer_ready)
        )
    )

    if not time_ready:
        hold_reason = "fc_ramp"
    elif not voltage_ready:
        hold_reason = "hold_voltage_guard"
    elif not (release_battery_ready or battery_buffer_ready):
        hold_reason = "hold_battery_headroom"
    elif not release_power_ready or not trend_ready:
        hold_reason = "hold_power_error"
    elif not dwell_ready:
        hold_reason = "hold_min_dwell_timer"
    else:
        hold_reason = ""

    return {
        "release_profile_name": str(release_profile["name"]),
        "release_scale": float(release_profile["scale"]),
        "elapsed_s": float(elapsed_s),
        "release_elapsed_s": float(release_elapsed_s),
        "time_ready": bool(time_ready),
        "power_ready": bool(strict_power_ready),
        "battery_ready": bool(strict_battery_ready),
        "strict_power_ready": bool(strict_power_ready),
        "strict_battery_ready": bool(strict_battery_ready),
        "release_power_ready": bool(release_power_ready),
        "release_battery_ready": bool(release_battery_ready),
        "voltage_ready": bool(voltage_ready),
        "dwell_ready": bool(dwell_ready),
        "trend_ready": bool(trend_ready),
        "battery_buffer_ready": bool(battery_buffer_ready),
        "strict_release_ready": bool(strict_release_ready),
        "release_ready": release_ready,
        "hold_reason": hold_reason,
        "power_error_w": float(power_error_w),
        "power_error_trend_w_per_s": float(power_error_trend_w_per_s),
        "strict_power_tolerance_w": float(config.PRECONDITION_POWER_TOLERANCE_W),
        "release_power_tolerance_w": float(release_profile["release_power_tol_w"]),
        "battery_headroom_a": float(battery_headroom_a),
        "strict_headroom_threshold_a": float(strict_headroom_threshold_a),
        "release_headroom_threshold_a": float(release_headroom_threshold_a),
        "remaining_power_deficit_w": float(remaining_power_deficit_w),
        "battery_buffer_w": float(battery_buffer_w),
        "voltage_guard_v": float(voltage_guard_v),
        "min_dwell_s": float(release_profile["min_dwell_s"]),
    }


def is_preconditioning_complete(
    energy_state: EnergyState,
    current_time_s: float,
    message_send_time_s: float,
) -> bool:
    """判断预调是否完成。"""
    status = preconditioning_status(energy_state, current_time_s, message_send_time_s)
    return bool(status["release_ready"])


def step_proposed_ems(
    current_time_s: float,
    dt: float,
    p_demand_w: float,
    messages: List[Dict[str, object]],
    energy_state: EnergyState,
    health_state: HealthState,
) -> Dict[str, float]:
    """执行一步说明书优先的前馈 EMS。"""
    active_message = _active_message(messages, current_time_s)
    p_future = lookup_future_demand(messages, current_time_s)
    if p_future <= 0.0:
        p_future = float(p_demand_w)

    precondition_scale = _message_preconditioning_scale(active_message)
    preview_up = max(0.0, p_future - float(p_demand_w))
    scaled_preview_up = precondition_scale * preview_up
    scaled_future_demand = min(
        float(p_demand_w) + scaled_preview_up,
        config.FC_RATED_POWER * config.FC_TARGET_POWER_CAP_RATIO,
    )
    fc_target = min(
        float(p_demand_w) * config.FC_TARGET_DEMAND_MARGIN
        + config.FC_PREVIEW_GAIN * scaled_preview_up,
        config.FC_RATED_POWER * config.FC_TARGET_POWER_CAP_RATIO,
    )
    if abs(fc_target - energy_state.fc_power_w) < config.FC_TARGET_DEADBAND_W:
        fc_target = energy_state.fc_power_w

    fc_prev = energy_state.fc_power_w
    max_change = config.FC_RAMP_LIMIT * dt
    dp = fc_target - energy_state.fc_power_w
    if abs(dp) <= max_change:
        energy_state.fc_power_w = fc_target
    elif dp > 0.0:
        energy_state.fc_power_w += max_change
    else:
        energy_state.fc_power_w -= max_change
    energy_state.fc_power_w = max(0.0, min(energy_state.fc_power_w, config.FC_RATED_POWER))
    energy_state.fc_target_power_w = float(fc_target)
    energy_state.preconditioning_target_w = float(scaled_future_demand)
    _update_battery_limit(energy_state, scaled_future_demand, energy_state.fc_power_w)

    p_gap = max(0.0, float(p_demand_w) - energy_state.fc_power_w)
    battery_current, bus_voltage, under_supply = _solve_battery_current(
        p_gap_w=p_gap,
        fc_power_w=energy_state.fc_power_w,
        current_limit_a=energy_state.battery_current_limit_a,
        resistance_ohm=config.BATTERY_EQ_RESISTANCE,
    )
    energy_state.battery_current_a = battery_current
    energy_state.bus_voltage_v = bus_voltage

    dp_fc = abs(energy_state.fc_power_w - fc_prev)
    stress_inc = calc_fc_step_stress_increment(dp_fc, dt)
    health_state.stress_fc_online += stress_inc
    health_state.soh = max(
        config.MIN_SOH,
        health_state.soh - stress_inc * config.SOH_DEGRADATION_PER_FC_STRESS,
    )

    return {
        "p_future_w": float(p_future),
        "h2_g": _h2_consumption(energy_state.fc_power_w, dt),
        "battery_stress_as": max(0.0, battery_current - config.BAT_DEGRAD_THRESHOLD) * dt,
        "fc_stress_increment": float(stress_inc),
        "under_supply": 1.0 if under_supply else 0.0,
        "power_spike": 1.0 if battery_current > config.BAT_SPIKE_THRESHOLD else 0.0,
    }


def step_passive_ems(
    current_time_s: float,
    dt: float,
    p_demand_w: float,
    energy_state: EnergyState,
    health_state: HealthState,
) -> Dict[str, float]:
    """执行一步传统被动 EMS。"""
    del current_time_s
    fc_prev = energy_state.fc_power_w
    alpha_resp = 1.0 - math.exp(-dt / config.FC_TAU)
    energy_state.fc_power_w = energy_state.fc_power_w + alpha_resp * (float(p_demand_w) - energy_state.fc_power_w)
    energy_state.fc_power_w = max(0.0, min(energy_state.fc_power_w, config.FC_RATED_POWER))
    energy_state.fc_target_power_w = energy_state.fc_power_w
    energy_state.preconditioning_target_w = energy_state.fc_power_w

    p_gap = max(0.0, float(p_demand_w) - energy_state.fc_power_w)
    energy_state.battery_current_limit_a = max(
        config.BAT_MIN_CURRENT,
        p_gap / max(config.V_BUS, 1e-6) * config.PASSIVE_BATTERY_OVERSUPPLY_RATIO,
    )
    battery_current, bus_voltage, under_supply = _solve_battery_current(
        p_gap_w=p_gap,
        fc_power_w=energy_state.fc_power_w,
        current_limit_a=energy_state.battery_current_limit_a,
        resistance_ohm=config.PASSIVE_EQ_RESISTANCE,
    )
    energy_state.battery_current_a = battery_current
    energy_state.bus_voltage_v = bus_voltage

    dp_fc = abs(energy_state.fc_power_w - fc_prev)
    stress_inc = calc_fc_step_stress_increment(dp_fc, dt)
    health_state.stress_fc_online += stress_inc
    health_state.soh = max(
        config.MIN_SOH,
        health_state.soh - stress_inc * config.SOH_DEGRADATION_PER_FC_STRESS,
    )

    return {
        "p_future_w": float(p_demand_w),
        "h2_g": _h2_consumption(energy_state.fc_power_w, dt),
        "battery_stress_as": max(0.0, battery_current - config.BAT_DEGRAD_THRESHOLD) * dt,
        "fc_stress_increment": float(stress_inc),
        "under_supply": 1.0 if under_supply else 0.0,
        "power_spike": 1.0 if battery_current > config.BAT_SPIKE_THRESHOLD else 0.0,
    }


class _BaseEMS:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.energy_state = EnergyState()
        self.health_state = HealthState()
        self.h2_total_g = 0.0
        self.power_spike_count = 0
        self.battery_stress_index_as = 0.0
        self.fc_stress_index = 0.0
        self.min_bus_voltage_v = config.V_BUS
        self.under_supply_count = 0
        self.log: List[Dict[str, float]] = []

    def _accumulate_metrics(self, step_result: Dict[str, float]) -> None:
        self.h2_total_g += float(step_result["h2_g"])
        self.battery_stress_index_as += float(step_result["battery_stress_as"])
        self.fc_stress_index += float(step_result["fc_stress_increment"])
        self.power_spike_count += int(step_result["power_spike"])
        self.under_supply_count += int(step_result["under_supply"])
        self.min_bus_voltage_v = min(self.min_bus_voltage_v, self.energy_state.bus_voltage_v)

    def _result(self) -> Dict[str, float]:
        return {
            "h2_total_g": self.h2_total_g,
            "power_spike_count": int(self.power_spike_count),
            "battery_stress_index_as": self.battery_stress_index_as,
            "fc_stress_index": self.fc_stress_index,
            "min_bus_voltage_v": self.min_bus_voltage_v,
            "under_supply_count": int(self.under_supply_count),
            "estimated_soh_end": float(self.health_state.soh),
            "online_fc_stress_index": float(self.health_state.stress_fc_online),
        }


class EMSController(_BaseEMS):
    """前馈 EMS 汇总器。"""

    def simulate(
        self,
        time_arr: np.ndarray,
        power_arr: np.ndarray,
        messages: List[Dict[str, object]] | None = None,
    ) -> Dict[str, float]:
        self.reset()
        if len(time_arr) == 0 or len(power_arr) == 0:
            return self._result()

        messages = sorted(messages or [], key=_message_available_time)
        self.energy_state.fc_power_w = float(power_arr[0])
        prev_time = 0.0

        for idx, p_demand in enumerate(power_arr):
            current_time = float(time_arr[idx])
            dt = max(current_time - prev_time, 1e-6)
            prev_time = current_time
            step_result = step_proposed_ems(
                current_time_s=current_time,
                dt=dt,
                p_demand_w=float(p_demand),
                messages=messages,
                energy_state=self.energy_state,
                health_state=self.health_state,
            )
            self._accumulate_metrics(step_result)
            self.log.append(
                {
                    "time_s": current_time,
                    "p_demand_w": float(p_demand),
                    "p_fc_w": self.energy_state.fc_power_w,
                    "i_bat_a": self.energy_state.battery_current_a,
                    "i_bat_limit_a": self.energy_state.battery_current_limit_a,
                    "v_bus_v": self.energy_state.bus_voltage_v,
                    "stress_fc_online": self.health_state.stress_fc_online,
                    "soh": self.health_state.soh,
                }
            )
        return self._result()


class PassiveEMS(_BaseEMS):
    """传统被动 EMS 汇总器。"""

    def simulate(self, time_arr: np.ndarray, power_arr: np.ndarray) -> Dict[str, float]:
        self.reset()
        if len(time_arr) == 0 or len(power_arr) == 0:
            return self._result()

        self.energy_state.fc_power_w = float(power_arr[0])
        prev_time = 0.0
        for idx, p_demand in enumerate(power_arr):
            current_time = float(time_arr[idx])
            dt = max(current_time - prev_time, 1e-6)
            prev_time = current_time
            step_result = step_passive_ems(
                current_time_s=current_time,
                dt=dt,
                p_demand_w=float(p_demand),
                energy_state=self.energy_state,
                health_state=self.health_state,
            )
            self._accumulate_metrics(step_result)
            self.log.append(
                {
                    "time_s": current_time,
                    "p_demand_w": float(p_demand),
                    "p_fc_w": self.energy_state.fc_power_w,
                    "i_bat_a": self.energy_state.battery_current_a,
                    "i_bat_limit_a": self.energy_state.battery_current_limit_a,
                    "v_bus_v": self.energy_state.bus_voltage_v,
                    "stress_fc_online": self.health_state.stress_fc_online,
                    "soh": self.health_state.soh,
                }
            )
        return self._result()
