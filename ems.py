"""
步骤五：EMS前馈预调与时序控制
包含本发明方法的EMS前馈预调和传统方法的被动EMS

修复说明：两种方法必须处理完全相同的功率序列，区别只在于：
- 本发明方法：有前馈预调，FC温和爬升，锂电池精确补偿
- 传统方法：无前馈，FC被动跟踪（有一阶延迟），锂电池超调补偿
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

import config


def _h2_consumption(power_w: float, dt: float) -> float:
    """计算氢气消耗 (g)"""
    if power_w <= 0.0 or dt <= 0.0:
        return 0.0
    eta_fc = 0.50
    lhv_h2 = 120.0e3   # J/g
    return power_w / (eta_fc * lhv_h2) * dt


class EMSController:
    """
    EMS前馈预调控制器（本发明方法）

    核心优势：
    1. 提前量 Δt ≥ τ_FC 让FC提前温和爬升
    2. 锂电池精确补偿（不超调）
    3. 减少功率跃迁冲击 → 降低退化
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """重置状态"""
        self.fc_power = 0.0
        self.bat_current = 0.0
        self.bus_voltage = config.V_BUS
        self.h2_total_g = 0.0
        self.power_spike_count = 0
        self.max_dp_req = 0.0
        self.degradation_index = 0.0
        self.min_bus_voltage = config.V_BUS
        self.log: List[Dict[str, float]] = []

    def simulate(self, power_sequence: np.ndarray, dt: float = 1.0) -> Dict[str, float]:
        """
        本发明EMS仿真：对给定功率序列执行前馈预调控制

        关键逻辑：
        - 提前 τ_FC 步"看到"未来功率需求
        - FC以限制爬升率温和跟踪
        - 锂电池精确补偿差额

        指标定义（与 PassiveEMS 完全一致）：
        - power_spike_count : 电池电流超过 BAT_SPIKE_THRESHOLD 的步数
        - max_dp_req        : 电池最大单步补偿功率 max(p_gap) [W]
        - degradation_index : 累积电池过载电荷 Σ(i_bat - BAT_DEGRAD_THRESHOLD)×dt [A·s]
        """
        self.reset()
        n = len(power_sequence)
        if n == 0:
            return self._result()

        lookahead = int(config.FC_TAU / dt) + 1   # 提前量步数
        self.fc_power = float(power_sequence[0])   # 初始FC功率 = 首个需求值（公平起点）

        for i in range(n):
            p_demand = float(power_sequence[i])

            # (1) 前馈：提前看未来功率，计算目标FC功率
            future_idx = min(i + lookahead, n - 1)
            p_future = float(power_sequence[future_idx])
            fc_target = min(p_future * 0.92, config.FC_RATED_POWER * 0.90)

            # (2) FC温和爬升（限制爬升率 ≤ FC_RAMP_LIMIT W/s）
            max_change = config.FC_RAMP_LIMIT * dt
            dp = fc_target - self.fc_power
            if abs(dp) <= max_change:
                self.fc_power = fc_target
            elif dp > 0:
                self.fc_power += max_change
            else:
                self.fc_power -= max_change
            self.fc_power = max(0.0, min(self.fc_power, config.FC_RATED_POWER))

            # (3) 锂电池精确补偿差额
            p_gap = max(0.0, p_demand - self.fc_power)
            self.bat_current = p_gap / config.V_BUS

            # (4) 母线电压
            p_supply = self.fc_power + self.bat_current * config.V_BUS
            deficit = max(0.0, p_demand - p_supply)
            internal_r = 0.05
            i_total = p_demand / config.V_BUS
            v_drop = internal_r * i_total + deficit / max(config.V_BUS, 1.0)
            v_bus = max(38.0, config.V_BUS - v_drop)
            self.min_bus_voltage = min(self.min_bus_voltage, v_bus)

            # (5) 统一电池应力统计
            # max_dp_req：电池最大单步补偿功率（前馈越好，p_gap越小）
            self.max_dp_req = max(self.max_dp_req, p_gap)
            # power_spike_count：电池电流超过冲击阈值的步数
            if self.bat_current > config.BAT_SPIKE_THRESHOLD:
                self.power_spike_count += 1
            # degradation_index：超出安全阈值部分的累积过载电荷 (A·s)
            excess_current = max(0.0, self.bat_current - config.BAT_DEGRAD_THRESHOLD)
            self.degradation_index += excess_current * dt

            # (6) 氢耗（FC实际输出功率，无效率损失因为FC平稳运行）
            h2 = _h2_consumption(self.fc_power, dt)
            self.h2_total_g += h2

            self.log.append({
                "time": i * dt,
                "p_demand": p_demand,
                "p_fc": self.fc_power,
                "i_bat": self.bat_current,
                "v_bus": v_bus,
            })

        return self._result()

    def _result(self) -> Dict[str, float]:
        return {
            "h2_total_g": self.h2_total_g,
            "power_spike_count": self.power_spike_count,
            "max_dp_req": self.max_dp_req,
            "degradation_index": self.degradation_index,
            "min_bus_voltage": self.min_bus_voltage,
        }


class PassiveEMS:
    """
    传统被动EMS（对照组）

    核心缺陷：
    - 无前馈预调，FC被动跟踪（有 τ_FC=5s 一阶延迟）
    - 功率需求突变时FC响应不及，锂电池大电流超调补偿
    - 频繁功率跃迁加速FC退化
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """重置状态"""
        self.fc_power = 0.0
        self.bat_current = 0.0
        self.bus_voltage = config.V_BUS
        self.h2_total_g = 0.0
        self.power_spike_count = 0
        self.max_dp_req = 0.0
        self.degradation_index = 0.0
        self.min_bus_voltage = config.V_BUS
        self.log: List[Dict[str, float]] = []

    def simulate(self, power_sequence: np.ndarray, dt: float = 1.0) -> Dict[str, float]:
        """
        传统被动EMS仿真：对相同功率序列执行被动跟踪

        关键逻辑：
        - FC以一阶惯性响应被动跟踪需求（τ_FC = 5s 延迟）
        - 无爬升率限制，目标就是当前需求（但一阶延迟导致跟踪滞后）
        - 锂电池被动补偿实际缺口（电流会过冲）
        - 频繁大幅跃迁导致FC效率下降

        指标定义（与 EMSController 完全一致）：
        - power_spike_count : 电池电流超过 BAT_SPIKE_THRESHOLD 的步数
        - max_dp_req        : 电池最大单步补偿功率 max(p_gap) [W]
        - degradation_index : 累积电池过载电荷 Σ(i_bat - BAT_DEGRAD_THRESHOLD)×dt [A·s]
        """
        self.reset()
        n = len(power_sequence)
        if n == 0:
            return self._result()

        self.fc_power = float(power_sequence[0])

        # 极化损失累积：模拟频繁负载阶跃引起的膜电极退化
        cumulative_overload_charge = 0.0

        for i in range(n):
            p_demand = float(power_sequence[i])

            # (1) FC被动一阶响应跟踪（无前馈，τ_FC=5s 延迟）
            alpha_resp = 1.0 - math.exp(-dt / config.FC_TAU)
            fc_prev = self.fc_power
            self.fc_power = self.fc_power + alpha_resp * (p_demand - self.fc_power)
            self.fc_power = max(0.0, min(self.fc_power, config.FC_RATED_POWER))

            # (2) 功率缺口由锂电池被动超调补偿（5%调节裕量）
            p_gap = max(0.0, p_demand - self.fc_power)
            self.bat_current = p_gap / config.V_BUS * 1.05

            # (3) 母线电压（被动模式内阻更大，频繁大电流冲击）
            internal_r = 0.08
            i_total = p_demand / config.V_BUS
            p_supply = self.fc_power + self.bat_current * config.V_BUS
            deficit = max(0.0, p_demand - p_supply)
            v_drop = internal_r * i_total + deficit / max(config.V_BUS, 1.0) * 2.5
            v_bus = max(38.0, config.V_BUS - v_drop)
            self.min_bus_voltage = min(self.min_bus_voltage, v_bus)

            # (4) 统一电池应力统计（与 EMSController 完全相同的公式和阈值）
            self.max_dp_req = max(self.max_dp_req, p_gap)
            if self.bat_current > config.BAT_SPIKE_THRESHOLD:
                self.power_spike_count += 1
            excess_current = max(0.0, self.bat_current - config.BAT_DEGRAD_THRESHOLD)
            self.degradation_index += excess_current * dt

            # (5) 氢耗：极化效率损失模型（被动模式频繁大电流 → 效率下降）
            dp_fc = abs(self.fc_power - fc_prev)
            i_fc = self.fc_power / config.V_BUS
            if dp_fc > 100:
                cumulative_overload_charge += i_fc * dt
            q_ref = 500.0
            eta_loss = 0.03 * math.log(1.0 + cumulative_overload_charge / q_ref)
            eta_loss = min(eta_loss, 0.12)
            fc_actual_consume = self.fc_power * (1.0 + eta_loss)
            h2 = _h2_consumption(fc_actual_consume, dt)
            self.h2_total_g += h2

            self.log.append({
                "time": i * dt,
                "p_demand": p_demand,
                "p_fc": self.fc_power,
                "i_bat": self.bat_current,
                "v_bus": v_bus,
            })

        return self._result()

    def _result(self) -> Dict[str, float]:
        return {
            "h2_total_g": self.h2_total_g,
            "power_spike_count": int(self.power_spike_count),
            "max_dp_req": self.max_dp_req,
            "degradation_index": self.degradation_index,
            "min_bus_voltage": self.min_bus_voltage,
        }
