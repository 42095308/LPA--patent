"""
步骤四：轨迹特征提取与结构化消息生成
从重规划路径中提取功率序列、特征向量，并生成发送至EMS的结构化消息
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
from scipy.interpolate import splprep, splev

import config
from energy_map import EnergyMap, _hover_power, _climb_power, _cruise_power


def extract_geometry(energy_map: EnergyMap, path: List[int]
                     ) -> List[Dict[str, float]]:
    """
    (1) 几何量计算：逐点计算相邻航迹点间的几何特征

    返回:
        列表，每个元素包含 delta_h, distance_h, delta_t, climb_rate
    """
    segments = []
    for i in range(len(path) - 1):
        n1 = energy_map.nodes[path[i]]
        n2 = energy_map.nodes[path[i + 1]]

        dx = n2[0] - n1[0]
        dy = n2[1] - n1[1]
        dz = n2[2] - n1[2]

        d_h = math.sqrt(dx * dx + dy * dy)
        d_3d = math.sqrt(d_h ** 2 + dz ** 2)
        dt = d_3d / config.CRUISE_SPEED if d_3d > 0 else 0.0
        climb_rate = dz / dt if dt > 0 else 0.0

        segments.append({
            "delta_h": dz,
            "distance_h": d_h,
            "distance_3d": d_3d,
            "delta_t": dt,
            "climb_rate": climb_rate,
            "alt_start": float(n1[2]),
            "alt_end": float(n2[2]),
        })

    return segments


def compute_power_sequence(energy_map: EnergyMap, path: List[int],
                           wind_speed: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    (2) 功率序列计算

    返回:
        (time_points, power_points) — 时间序列和对应功率值
    """
    time_list = [0.0]
    power_list = []
    cumulative_t = 0.0

    for i in range(len(path) - 1):
        seg = energy_map.compute_power_for_segment(path[i], path[i + 1], wind_speed)
        p_total = seg["p_total"]
        dt = seg["flight_time"]

        # 每段内按1s采样
        n_samples = max(1, int(dt))
        for k in range(n_samples):
            power_list.append(p_total)
            cumulative_t += dt / n_samples
            time_list.append(cumulative_t)

    # 调整使时间点和功率点数量一致
    time_arr = np.array(time_list[:len(power_list)], dtype=float)
    power_arr = np.array(power_list, dtype=float)

    return time_arr, power_arr


def smooth_power_bspline(time_arr: np.ndarray, power_arr: np.ndarray,
                         smoothing_factor: float = 0.3
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """
    (3) B样条平滑

    对原始功率序列进行B样条插值平滑，消除逐段计算引入的锯齿
    """
    if len(power_arr) < 4:
        return time_arr, power_arr

    # 参数化
    u = np.linspace(0.0, 1.0, len(power_arr))

    # 拟合B样条
    k = min(3, len(power_arr) - 1)
    try:
        tck, _ = splprep([power_arr], u=u, k=k, s=smoothing_factor * len(power_arr))
        u_fine = np.linspace(0.0, 1.0, len(power_arr))
        power_smooth = splev(u_fine, tck)[0]
        power_smooth = np.maximum(power_smooth, 0.0)  # 功率不能为负
    except Exception:
        power_smooth = power_arr.copy()

    return time_arr, power_smooth


def extract_feature_vector(time_arr: np.ndarray, power_arr: np.ndarray,
                           segments: List[Dict[str, float]]
                           ) -> Dict[str, float]:
    """
    (4) 特征向量提取

    Feature_vec = [P_peak, T_ramp, avg_climb_rate, E_hydrogen]
    """
    p_peak = float(np.max(power_arr)) if len(power_arr) > 0 else 0.0

    # T_ramp: 从当前功率上升到峰值所需时间
    if len(power_arr) > 1:
        p_start = float(power_arr[0])
        peak_idx = int(np.argmax(power_arr))
        if peak_idx > 0 and len(time_arr) > peak_idx:
            t_ramp = float(time_arr[peak_idx] - time_arr[0])
        else:
            t_ramp = 0.0
    else:
        t_ramp = 0.0

    # 平均爬升率
    if segments:
        rates = [s["climb_rate"] for s in segments if s["climb_rate"] > 0]
        avg_climb_rate = float(np.mean(rates)) if rates else 0.0
    else:
        avg_climb_rate = 0.0

    # 总氢气消耗估算
    total_h2 = 0.0
    eta_fc = 0.50
    lhv_h2 = 120.0e3   # J/g
    if len(power_arr) > 0 and len(time_arr) > 0:
        dt = float(time_arr[-1]) / max(len(power_arr), 1)
        for p in power_arr:
            total_h2 += float(p) / (eta_fc * lhv_h2) * dt

    return {
        "P_peak": p_peak,
        "T_ramp": t_ramp,
        "avg_climb_rate": avg_climb_rate,
        "E_hydrogen": total_h2,
    }


def build_structured_message(timestamp: float, replanning_id: int,
                              feature_vec: Dict[str, float],
                              power_predict: np.ndarray,
                              t_window: float) -> Dict:
    """
    (5) 打包结构化消息

    消息格式：
        timestamp: 重规划完成时刻
        replanning_id: 本次重规划编号
        path_feature_vector: 特征向量
        P_predict: 功率预测序列
        T_window: 重连段总飞行时间
    """
    return {
        "timestamp": timestamp,
        "replanning_id": replanning_id,
        "path_feature_vector": {
            "P_peak": feature_vec["P_peak"],
            "T_ramp": feature_vec["T_ramp"],
            "avg_climb_rate": feature_vec["avg_climb_rate"],
            "E_hydrogen": feature_vec["E_hydrogen"],
        },
        "P_predict": power_predict.tolist(),
        "T_window": t_window,
    }


def classify_flight_phases(energy_map: EnergyMap, path: List[int],
                           wind_speed: float = 3.0
                           ) -> List[Dict[str, float]]:
    """
    将路径分为飞行阶段并统计功率需求

    飞行阶段划分依据：
    - 起飞爬升：前 10% 路径且爬升率 > 2 m/s
    - 低海拔巡航：海拔 < 1000m 且水平飞行
    - 山地爬升段：海拔 1000~2000m 且爬升率 > 1 m/s
    - 高海拔巡航：海拔 > 1800m 且水平飞行
    - 悬停作业：速度接近 0（模拟）
    - 降落返航：后 15% 路径
    """
    if len(path) < 3:
        return []

    # 采集全路径功率数据
    all_segments = []
    for i in range(len(path) - 1):
        seg = energy_map.compute_power_for_segment(path[i], path[i + 1], wind_speed)
        seg["progress"] = i / max(len(path) - 1, 1)
        all_segments.append(seg)

    n = len(all_segments)
    phases = []

    # 按进度和海拔划分阶段
    phase_defs = [
        ("起飞爬升", lambda s: s["progress"] < 0.10 and s["climb_rate"] > 0.5),
        ("低海拔巡航", lambda s: s["altitude_mid"] < 1000 and abs(s["climb_rate"]) < 2.0 and s["progress"] >= 0.10),
        ("山地爬升段", lambda s: 1000 <= s["altitude_mid"] <= 2000 and s["climb_rate"] > 0.5),
        ("高海拔巡航", lambda s: s["altitude_mid"] > 1800 and abs(s["climb_rate"]) < 2.0),
        ("悬停作业", None),   # 模拟阶段
        ("降落返航", lambda s: s["progress"] > 0.85),
    ]

    # 简化处理：按进度比例分组
    phase_ranges = [
        ("起飞爬升", 0.0, 0.10),
        ("低海拔巡航", 0.10, 0.35),
        ("山地爬升段", 0.35, 0.65),
        ("高海拔巡航", 0.65, 0.80),
        ("悬停作业", 0.80, 0.85),
        ("降落返航", 0.85, 1.0),
    ]

    for phase_name, p_start, p_end in phase_ranges:
        phase_segs = [s for s in all_segments
                      if p_start <= s["progress"] < p_end]
        if not phase_segs:
            # 悬停作业阶段用模拟数据
            if phase_name == "悬停作业":
                # 取平均高度处的悬停功率
                avg_alt = float(np.mean([s["altitude_mid"] for s in all_segments]))
                p_hover = _hover_power(avg_alt)
                phases.append({
                    "phase": phase_name,
                    "duration_s": 120.0,
                    "avg_power_w": p_hover,
                    "peak_power_w": p_hover * 1.03,
                    "dp_req_w": 180.0,
                })
            continue

        total_time = sum(s["flight_time"] for s in phase_segs)
        powers = [s["p_total"] for s in phase_segs]
        avg_power = float(np.mean(powers))
        peak_power = float(np.max(powers))

        # ΔP_req：本阶段内相邻段之间的功率变化均值
        # 修复说明：原来用 max() 导致每个阶段均被同一对相邻节点的
        # 最大跃迁（约 1021W）主导，丢失阶段间差异。
        # 改为均值（或 90 分位数），能区分各阶段的典型扰动水平。
        dp_reqs = []
        for i in range(1, len(powers)):
            dp_reqs.append(abs(powers[i] - powers[i - 1]))
        if dp_reqs:
            # 样本足够时取 90 分位数（鲁棒上界）；样本少时取均值
            if len(dp_reqs) >= 5:
                dp_req = float(np.percentile(dp_reqs, 90))
            else:
                dp_req = float(np.mean(dp_reqs))
        else:
            dp_req = 0.0

        phases.append({
            "phase": phase_name,
            "duration_s": total_time,
            "avg_power_w": avg_power,
            "peak_power_w": peak_power,
            "dp_req_w": dp_req,
        })

    return phases
