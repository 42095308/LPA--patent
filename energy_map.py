"""
步骤一：构建多维能量地图
将华山地形区域离散为三维栅格图，为每条有向边定义综合代价函数 C(n,n')
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial import cKDTree

import config
from dem_loader import nearest_rc_from_lonlat


def _hover_power(altitude_m: float) -> float:
    """
    悬停功率 P_hover = (mg)^{3/2} / sqrt(2ρA)
    空气密度ρ随海拔按标准大气模型修正
    """
    rho = config.air_density(altitude_m)
    mg = config.MASS * config.GRAVITY
    return (mg ** 1.5) / math.sqrt(2.0 * rho * config.ROTOR_AREA)


def _climb_power(climb_rate: float) -> float:
    """爬升附加功率 P_climb = mg × climb_rate"""
    if climb_rate <= 0.0:
        return 0.0
    return config.MASS * config.GRAVITY * climb_rate


def _cruise_power(speed: float, wind_speed: float, altitude_m: float) -> float:
    """
    巡航功率估算（含风阻模型）
    P_cruise = P_parasite + P_wind_extra
    风速增大时额外功率消耗显著增加
    """
    rho = config.air_density(altitude_m)
    # 等效空速 = 巡航速度 + 逆风分量
    v_eff = speed + wind_speed * config.WIND_HEADWIND_FRAC
    # 气动阻力功率 = 0.5 * Cd * rho * A * v_eff^3
    p_parasite = 0.5 * config.WIND_CD_BODY * rho * config.WIND_A_BODY * v_eff ** 3
    # 额外的感应功率增量（风场干扰造成旋翼效率下降）
    # 风速越大，旋翼需要额外功率补偿姿态扰动
    p_wind_disturb = 0.5 * config.MASS * config.GRAVITY * (wind_speed / 20.0) ** 2 * (wind_speed / 5.0)
    return p_parasite + p_wind_disturb


def _degradation_penalty(dp_req: float, dp_dt: float, soh: float) -> float:
    """
    燃料电池寿命退化惩罚项
    Ψ = k(SoH) × [w₁(ΔP/P_rated)² + w₂(dP/dt / (dP/dt)_max)²]
    """
    k = config.k_soh(soh)
    term1 = config.W1 * (dp_req / config.FC_RATED_POWER) ** 2
    term2 = config.W2 * (dp_dt / config.FC_DP_DT_MAX) ** 2
    return k * (term1 + term2)


def _hydrogen_consumption(power_w: float, duration_s: float) -> float:
    """
    估算氢气消耗量 (g)
    基于典型PEMFC效率约 50%，氢的低热值 LHV = 120 MJ/kg
    H2_rate = P_FC / (η × LHV)
    """
    if duration_s <= 0.0 or power_w <= 0.0:
        return 0.0
    eta_fc = 0.50               # 燃料电池效率
    lhv_h2 = 120.0e3            # J/g (120 MJ/kg = 120000 J/g)
    h2_rate = power_w / (eta_fc * lhv_h2)  # g/s
    return h2_rate * duration_s


class EnergyMap:
    """
    三维栅格能量地图

    将华山地形区域离散为三维栅格图：
    - 水平分辨率 25m
    - 垂直分辨率 50m
    - 每条有向边定义综合代价 C(n,n') = α·D + β·E + γ·Ψ
    """

    def __init__(self, z_dem: np.ndarray, lon_grid: np.ndarray,
                 lat_grid: np.ndarray, soh: float = 0.9,
                 wind_field: np.ndarray | None = None):
        """
        参数:
            z_dem: DEM高程矩阵
            lon_grid: 经度网格
            lat_grid: 纬度网格
            soh: 燃料电池健康状态 (0~1)
            wind_field: 风场数据 (可选)
        """
        self.z_dem = z_dem
        self.lon_grid = lon_grid
        self.lat_grid = lat_grid
        self.soh = soh
        self.dem_rows, self.dem_cols = z_dem.shape

        # 风场：默认 3 m/s
        if wind_field is not None:
            self.wind_field = wind_field
        else:
            self.wind_field = np.full_like(z_dem, config.WIND_NORMAL, dtype=float)

        # 存储图结构
        self.nodes: np.ndarray = np.empty((0, 4))   # [x_m, y_m, z_m, node_id]
        self.edges: List[Tuple[int, int]] = []       # 有向边列表
        self.edge_costs: np.ndarray = np.empty(0)    # 综合代价
        self.edge_d_spatial: np.ndarray = np.empty(0)
        self.edge_e_hydrogen: np.ndarray = np.empty(0)
        self.edge_psi_degrad: np.ndarray = np.empty(0)
        self.adj: Dict[int, List[Tuple[int, int]]] = {}  # 邻接表 node -> [(neighbor, edge_idx)]

        # 经纬度→栅格索引的辅助工具
        self._node_rc: List[Tuple[int, int, int]] = []   # (dem_row, dem_col, alt_layer)

    def _dem_rc(self, x_m: float, y_m: float) -> Tuple[int, int]:
        """将地图坐标 (m) 转换为 DEM 像素坐标"""
        c = int(np.clip(x_m / config.DEM_RES, 0, self.dem_cols - 1))
        r = int(np.clip((self.dem_rows - 1) - y_m / config.DEM_RES, 0, self.dem_rows - 1))
        return r, c

    def _get_terrain(self, x_m: float, y_m: float) -> float:
        """获取某位置的地形高程"""
        r, c = self._dem_rc(x_m, y_m)
        return float(self.z_dem[r, c])

    def _get_wind(self, x_m: float, y_m: float) -> float:
        """获取某位置的风速"""
        r, c = self._dem_rc(x_m, y_m)
        return float(self.wind_field[r, c])

    def build_graph(self) -> None:
        """
        构建三维栅格图

        水平分辨率 25m, 垂直分辨率 50m
        生成节点和有向边
        """
        print("[步骤一] 构建三维栅格能量地图...")

        # 计算栅格维度
        h_res = config.GRID_H_RES
        v_res = config.GRID_V_RES
        dem_extent_x = self.dem_cols * config.DEM_RES   # 水平范围 (m)
        dem_extent_y = self.dem_rows * config.DEM_RES   # 垂直范围 (m)

        nx = int(dem_extent_x / h_res)
        ny = int(dem_extent_y / h_res)

        # 高程范围
        z_min = float(np.nanmin(self.z_dem))
        z_max = float(np.nanmax(self.z_dem))
        # 飞行高度范围：地形最低点+30m 到 地形最高点+200m
        alt_min = z_min + 30.0
        alt_max = z_max + 200.0
        nz = int((alt_max - alt_min) / v_res) + 1

        print(f"  栅格维度: {nx} x {ny} x {nz}")
        print(f"  高度范围: {alt_min:.0f}m ~ {alt_max:.0f}m")

        # 生成节点
        node_list = []
        node_rc_list = []
        node_id = 0
        # 栅格索引 (ix, iy, iz) -> node_id 的映射
        grid_to_node: Dict[Tuple[int, int, int], int] = {}

        for iy in range(ny):
            for ix in range(nx):
                x_m = (ix + 0.5) * h_res
                y_m = (iy + 0.5) * h_res
                terrain_h = self._get_terrain(x_m, y_m)
                if np.isnan(terrain_h):
                    continue

                for iz in range(nz):
                    z_m = alt_min + iz * v_res
                    # 节点必须高于地形 + 安全间距
                    if z_m < terrain_h + 20.0:
                        continue
                    # 节点不能太高于地形（不切实际）
                    if z_m > terrain_h + 300.0:
                        continue

                    grid_to_node[(ix, iy, iz)] = node_id
                    node_list.append([x_m, y_m, z_m, float(node_id)])
                    r, c = self._dem_rc(x_m, y_m)
                    node_rc_list.append((r, c, iz))
                    node_id += 1

        self.nodes = np.array(node_list, dtype=float)
        self._node_rc = node_rc_list
        n_nodes = len(self.nodes)
        print(f"  节点总数: {n_nodes}")

        # 构建有向边：相邻栅格节点之间的连接
        # 26邻域连接（水平8个方向 + 垂直上下 + 对角）
        edge_list: List[Tuple[int, int]] = []
        d_spatial_list: List[float] = []
        e_hydrogen_list: List[float] = []
        psi_degrad_list: List[float] = []
        cost_list: List[float] = []

        # 方向偏移量：水平+垂直的26邻域
        offsets = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    offsets.append((dx, dy, dz))

        prev_power = _hover_power(float(np.mean([alt_min, alt_max])))

        for (ix, iy, iz), nid in grid_to_node.items():
            for dx, dy, dz in offsets:
                nix, niy, niz = ix + dx, iy + dy, iz + dz
                neighbor_id = grid_to_node.get((nix, niy, niz))
                if neighbor_id is None:
                    continue

                n1 = self.nodes[nid]
                n2 = self.nodes[neighbor_id]

                # (1) 空间几何代价（欧氏距离，米）
                d_h = math.sqrt((n2[0] - n1[0]) ** 2 + (n2[1] - n1[1]) ** 2)
                d_v = abs(n2[2] - n1[2])
                d_spatial = math.sqrt(d_h ** 2 + d_v ** 2)

                # (2) 预测氢气消耗
                alt_mid = 0.5 * (n1[2] + n2[2])
                climb_rate = (n2[2] - n1[2]) / max(d_spatial / config.CRUISE_SPEED, 0.1)
                wind = self._get_wind(0.5 * (n1[0] + n2[0]), 0.5 * (n1[1] + n2[1]))

                p_hover = _hover_power(alt_mid)
                p_climb = _climb_power(max(0.0, climb_rate))
                p_cruise = _cruise_power(config.CRUISE_SPEED, wind, alt_mid)
                p_total = p_hover + p_climb + p_cruise

                flight_time = d_spatial / config.CRUISE_SPEED
                e_hydrogen = _hydrogen_consumption(p_total, flight_time)

                # (3) 退化惩罚
                dp_req = abs(p_total - prev_power)
                dp_dt = dp_req / max(flight_time, 0.1)
                psi_deg = _degradation_penalty(dp_req, dp_dt, self.soh)

                # 综合代价
                cost = (config.ALPHA * d_spatial
                        + config.BETA * e_hydrogen
                        + config.GAMMA * psi_deg)

                edge_list.append((nid, neighbor_id))
                d_spatial_list.append(d_spatial)
                e_hydrogen_list.append(e_hydrogen)
                psi_degrad_list.append(psi_deg)
                cost_list.append(cost)

        self.edges = edge_list
        self.edge_d_spatial = np.array(d_spatial_list, dtype=float)
        self.edge_e_hydrogen = np.array(e_hydrogen_list, dtype=float)
        self.edge_psi_degrad = np.array(psi_degrad_list, dtype=float)
        self.edge_costs = np.array(cost_list, dtype=float)

        # 构建邻接表
        self.adj = {i: [] for i in range(n_nodes)}
        for eid, (u, v) in enumerate(self.edges):
            self.adj[u].append((v, eid))

        print(f"  有向边总数: {len(self.edges)}")
        print(f"  综合代价范围: {np.min(self.edge_costs):.2f} ~ {np.max(self.edge_costs):.2f}")
        print(f"  氢耗范围: {np.min(self.edge_e_hydrogen):.4f}g ~ {np.max(self.edge_e_hydrogen):.4f}g")

    def find_nearest_node(self, lon: float, lat: float, alt: float) -> int:
        """根据经纬度和海拔找最近的节点"""
        r, c = nearest_rc_from_lonlat(self.lon_grid, self.lat_grid, lon, lat)
        x_m = c * config.DEM_RES
        y_m = (self.dem_rows - 1 - r) * config.DEM_RES

        # 在所有节点中找最近的
        dists = (self.nodes[:, 0] - x_m) ** 2 + (self.nodes[:, 1] - y_m) ** 2 + \
                ((self.nodes[:, 2] - alt) * 0.1) ** 2   # 垂直距离权重更低
        return int(np.argmin(dists))

    def get_edge_cost(self, edge_id: int) -> float:
        """获取边的综合代价"""
        return float(self.edge_costs[edge_id])

    def update_wind_field(self, region_center: Tuple[float, float],
                          radius_m: float, new_wind_speed: float) -> List[int]:
        """
        更新风场数据（模拟风切变事件）
        返回受影响的边索引列表（所有代价发生变化的边）
        """
        cx_m, cy_m = region_center
        affected_edges: List[int] = []

        # 更新风场栅格
        for r in range(self.dem_rows):
            for c in range(self.dem_cols):
                x_m = c * config.DEM_RES
                y_m = (self.dem_rows - 1 - r) * config.DEM_RES
                dist = math.sqrt((x_m - cx_m) ** 2 + (y_m - cy_m) ** 2)
                if dist <= radius_m:
                    self.wind_field[r, c] = new_wind_speed

        # 重新计算受影响边的代价
        prev_power = _hover_power(float(np.mean(self.nodes[:, 2])))
        for eid, (u, v) in enumerate(self.edges):
            n1 = self.nodes[u]
            n2 = self.nodes[v]
            mid_x = 0.5 * (n1[0] + n2[0])
            mid_y = 0.5 * (n1[1] + n2[1])

            # 检查边是否在受影响区域内（稍微扩大检测范围）
            dist = math.sqrt((mid_x - cx_m) ** 2 + (mid_y - cy_m) ** 2)
            if dist > radius_m + config.GRID_H_RES * 2:
                continue

            # 重新计算代价
            d_spatial = float(self.edge_d_spatial[eid])
            alt_mid = 0.5 * (n1[2] + n2[2])
            climb_rate = (n2[2] - n1[2]) / max(d_spatial / config.CRUISE_SPEED, 0.1)
            wind = self._get_wind(mid_x, mid_y)

            p_hover = _hover_power(alt_mid)
            p_climb = _climb_power(max(0.0, climb_rate))
            p_cruise = _cruise_power(config.CRUISE_SPEED, wind, alt_mid)
            p_total = p_hover + p_climb + p_cruise

            flight_time = d_spatial / config.CRUISE_SPEED
            e_hydrogen = _hydrogen_consumption(p_total, flight_time)

            dp_req = abs(p_total - prev_power)
            dp_dt = dp_req / max(flight_time, 0.1)
            psi_deg = _degradation_penalty(dp_req, dp_dt, self.soh)

            old_cost = float(self.edge_costs[eid])
            new_cost = (config.ALPHA * d_spatial
                        + config.BETA * e_hydrogen
                        + config.GAMMA * psi_deg)

            # 只要代价增加超过阈值（5%），就标记为受影响边
            if new_cost > old_cost * config.WIND_TRIGGER_RATIO:
                affected_edges.append(eid)

            # 无论是否超过阈值，都更新边的数据
            self.edge_e_hydrogen[eid] = e_hydrogen
            self.edge_psi_degrad[eid] = psi_deg
            self.edge_costs[eid] = new_cost

        return affected_edges

    def compute_power_for_segment(self, node_from: int, node_to: int,
                                  wind_speed: float = 3.0) -> Dict[str, float]:
        """
        计算某段飞行的功率需求

        返回:
            dict: 包含 p_hover, p_climb, p_cruise, p_total, h2_consumption 等
        """
        n1 = self.nodes[node_from]
        n2 = self.nodes[node_to]

        d_h = math.sqrt((n2[0] - n1[0]) ** 2 + (n2[1] - n1[1]) ** 2)
        d_v = n2[2] - n1[2]
        d_3d = math.sqrt(d_h ** 2 + d_v ** 2)
        flight_time = d_3d / config.CRUISE_SPEED if d_3d > 0 else 0.0

        alt_mid = 0.5 * (n1[2] + n2[2])
        climb_rate = d_v / max(flight_time, 0.1) if flight_time > 0 else 0.0

        p_hover = _hover_power(alt_mid)
        p_climb = _climb_power(max(0.0, climb_rate))
        p_cruise = _cruise_power(config.CRUISE_SPEED, wind_speed, alt_mid)
        p_total = p_hover + p_climb + p_cruise

        h2 = _hydrogen_consumption(p_total, flight_time)

        return {
            "p_hover": p_hover,
            "p_climb": p_climb,
            "p_cruise": p_cruise,
            "p_total": p_total,
            "flight_time": flight_time,
            "distance_m": d_3d,
            "delta_h": d_v,
            "climb_rate": climb_rate,
            "h2_consumption_g": h2,
            "altitude_mid": alt_mid,
        }
