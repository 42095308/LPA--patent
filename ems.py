"""EMS models for the proposed and traditional chains."""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

import config


def _h2_consumption(power_w: float, dt: float) -> float:
    """Convert electrical FC output to hydrogen consumption."""
    if power_w <= 0.0 or dt <= 0.0:
        return 0.0
    eta_fc = 0.50
    lhv_h2 = 120.0e3
    return power_w / (eta_fc * lhv_h2) * dt


def fc_stress_increment(dp_fc: float, dt: float) -> float:
    """Return the FC stress increment for a single time step."""
    ramp = dp_fc / max(dt, 1e-6)
    return (
        dp_fc / max(config.FC_STRESS_POWER_STEP_REF_W, 1e-6)
        + ramp / max(config.FC_STRESS_RAMP_REF_W_PER_S, 1e-6)
    ) * dt


class _BaseEMS:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.fc_power = 0.0
        self.bat_current = 0.0
        self.bus_voltage = config.V_BUS
        self.h2_total_g = 0.0
        self.power_spike_count = 0
        self.battery_stress_index_as = 0.0
        self.fc_stress_index = 0.0
        self.min_bus_voltage_v = config.V_BUS
        self.under_supply_count = 0
        self.log: List[Dict[str, float]] = []

    def _battery_stress(self, dt: float) -> None:
        if self.bat_current > config.BAT_SPIKE_THRESHOLD:
            self.power_spike_count += 1
        self.battery_stress_index_as += max(0.0, self.bat_current - config.BAT_DEGRAD_THRESHOLD) * dt

    def _fc_stress(self, dp_fc: float, dt: float) -> None:
        self.fc_stress_index += fc_stress_increment(dp_fc, dt)

    def _update_bus_voltage(self, p_demand: float, resistance_ohm: float) -> float:
        i_total = p_demand / max(config.V_BUS, 1e-6)
        i_supply = self.fc_power / max(config.V_BUS, 1e-6) + self.bat_current
        i_shortfall = max(0.0, i_total - i_supply)
        if i_shortfall > 0.0:
            self.under_supply_count += 1
        v_bus = config.V_BUS - (i_total + i_shortfall) * resistance_ohm
        self.min_bus_voltage_v = min(self.min_bus_voltage_v, v_bus)
        self.bus_voltage = v_bus
        return v_bus

    def _result(self) -> Dict[str, float]:
        return {
            "h2_total_g": self.h2_total_g,
            "power_spike_count": int(self.power_spike_count),
            "battery_stress_index_as": self.battery_stress_index_as,
            "fc_stress_index": self.fc_stress_index,
            "min_bus_voltage_v": self.min_bus_voltage_v,
            "under_supply_count": int(self.under_supply_count),
        }


class EMSController(_BaseEMS):
    """Proposed feedforward EMS with message-driven pre-adjustment."""

    def _lookup_future_demand(self, messages: List[Dict[str, object]], current_time_s: float) -> float:
        active_message = None
        for message in messages:
            if float(message["timestamp"]) <= current_time_s + 1e-9:
                active_message = message
            else:
                break

        if active_message is None:
            return 0.0

        predict = active_message["P_predict"]
        predict_time = np.asarray(predict["time_s"], dtype=float)
        predict_power = np.asarray(predict["power_w"], dtype=float)
        if len(predict_time) == 0:
            return 0.0

        lookahead_t = current_time_s + config.FC_TAU * config.EMS_MESSAGE_LOOKAHEAD_FACTOR
        return float(np.interp(lookahead_t, predict_time, predict_power, left=predict_power[0], right=predict_power[-1]))

    def simulate(
        self,
        time_arr: np.ndarray,
        power_arr: np.ndarray,
        messages: List[Dict[str, object]] | None = None,
    ) -> Dict[str, float]:
        self.reset()
        if len(time_arr) == 0 or len(power_arr) == 0:
            return self._result()

        messages = sorted(messages or [], key=lambda item: float(item["timestamp"]))
        prev_time = 0.0
        self.fc_power = float(power_arr[0])

        for idx, p_demand in enumerate(power_arr):
            current_time = float(time_arr[idx])
            dt = max(current_time - prev_time, 1e-6)
            prev_time = current_time

            p_future = self._lookup_future_demand(messages, current_time)
            if p_future <= 0.0:
                p_future = float(p_demand)

            fc_target = min(
                p_future * config.FC_TARGET_DEMAND_MARGIN,
                config.FC_RATED_POWER * config.FC_TARGET_POWER_CAP_RATIO,
            )

            fc_prev = self.fc_power
            max_change = config.FC_RAMP_LIMIT * dt
            dp = fc_target - self.fc_power
            if abs(dp) <= max_change:
                self.fc_power = fc_target
            elif dp > 0.0:
                self.fc_power += max_change
            else:
                self.fc_power -= max_change
            self.fc_power = max(0.0, min(self.fc_power, config.FC_RATED_POWER))

            p_gap = max(0.0, float(p_demand) - self.fc_power)
            self.bat_current = p_gap / max(config.V_BUS, 1e-6)

            v_bus = self._update_bus_voltage(float(p_demand), config.BATTERY_EQ_RESISTANCE)
            dp_fc = abs(self.fc_power - fc_prev)
            self._battery_stress(dt)
            self._fc_stress(dp_fc, dt)
            self.h2_total_g += _h2_consumption(self.fc_power, dt)

            self.log.append(
                {
                    "time_s": current_time,
                    "p_demand_w": float(p_demand),
                    "p_fc_w": self.fc_power,
                    "i_bat_a": self.bat_current,
                    "v_bus_v": v_bus,
                }
            )

        return self._result()


class PassiveEMS(_BaseEMS):
    """Traditional passive EMS without feedforward pre-adjustment."""

    def simulate(self, time_arr: np.ndarray, power_arr: np.ndarray) -> Dict[str, float]:
        self.reset()
        if len(time_arr) == 0 or len(power_arr) == 0:
            return self._result()

        prev_time = 0.0
        self.fc_power = float(power_arr[0])

        for idx, p_demand in enumerate(power_arr):
            current_time = float(time_arr[idx])
            dt = max(current_time - prev_time, 1e-6)
            prev_time = current_time

            fc_prev = self.fc_power
            alpha_resp = 1.0 - math.exp(-dt / config.FC_TAU)
            self.fc_power = self.fc_power + alpha_resp * (float(p_demand) - self.fc_power)
            self.fc_power = max(0.0, min(self.fc_power, config.FC_RATED_POWER))

            p_gap = max(0.0, float(p_demand) - self.fc_power)
            self.bat_current = (
                p_gap / max(config.V_BUS, 1e-6) * config.PASSIVE_BATTERY_OVERSUPPLY_RATIO
            )

            v_bus = self._update_bus_voltage(float(p_demand), config.PASSIVE_EQ_RESISTANCE)
            dp_fc = abs(self.fc_power - fc_prev)
            self._battery_stress(dt)
            self._fc_stress(dp_fc, dt)
            self.h2_total_g += _h2_consumption(self.fc_power, dt)

            self.log.append(
                {
                    "time_s": current_time,
                    "p_demand_w": float(p_demand),
                    "p_fc_w": self.fc_power,
                    "i_bat_a": self.bat_current,
                    "v_bus_v": v_bus,
                }
            )

        return self._result()
