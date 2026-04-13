"""能量地图与约束建图。"""

from __future__ import annotations

import math
from copy import copy
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.spatial import cKDTree

import config
from dem_loader import nearest_rc_from_lonlat


def _hover_power(altitude_m: float) -> float:
    """估计悬停功率。"""
    rho = config.air_density(altitude_m)
    mg = config.MASS * config.GRAVITY
    return (mg ** 1.5) / math.sqrt(2.0 * rho * config.ROTOR_AREA)


def _climb_power(climb_rate: float) -> float:
    """估计爬升或下降功率。"""
    base_power = config.MASS * config.GRAVITY * climb_rate
    if climb_rate >= 0.0:
        return base_power
    return config.DESCENT_POWER_FACTOR * base_power


def _cruise_power(speed: float, wind_speed: float, altitude_m: float) -> float:
    """估计巡航功率与风扰动附加项。"""
    rho = config.air_density(altitude_m)
    v_eff = speed + wind_speed * config.WIND_HEADWIND_FRAC
    p_parasite = 0.5 * config.WIND_CD_BODY * rho * config.WIND_A_BODY * v_eff ** 3
    p_wind_disturb = (
        0.5
        * config.MASS
        * config.GRAVITY
        * (wind_speed / 20.0) ** 2
        * (wind_speed / 5.0)
    )
    return p_parasite + p_wind_disturb


def _hydrogen_consumption(power_w: float, duration_s: float) -> float:
    """估计氢耗，单位为克。"""
    if duration_s <= 0.0 or power_w <= 0.0:
        return 0.0
    eta_fc = 0.50
    lhv_h2 = 120.0e3
    return power_w / (eta_fc * lhv_h2) * duration_s


class EnergyMap:
    """三维走廊图与动态代价场。"""

    def __init__(
        self,
        z_dem: np.ndarray,
        lon_grid: np.ndarray,
        lat_grid: np.ndarray,
        soh: float = 0.9,
        wind_field: np.ndarray | None = None,
    ):
        self.z_dem = z_dem
        self.lon_grid = lon_grid
        self.lat_grid = lat_grid
        self.soh = soh
        self.dem_rows, self.dem_cols = z_dem.shape
        self.wind_field = (
            np.array(wind_field, copy=True, dtype=float)
            if wind_field is not None
            else np.full_like(z_dem, config.WIND_NORMAL, dtype=float)
        )

        self.static_obstacles: List[Dict[str, Any]] = [dict(item) for item in config.STATIC_OBSTACLES]
        self.airspace_constraints: List[Dict[str, Any]] = [dict(item) for item in config.AIRSPACE_CONSTRAINTS]
        self.dynamic_obstacles: List[Dict[str, Any]] = []
        self.blocked_edge_ids: set[int] = set()

        self.nodes: np.ndarray = np.empty((0, 3), dtype=float)
        self.edges: List[Tuple[int, int]] = []
        self.adj: Dict[int, List[Tuple[int, int]]] = {}
        self.rev_adj: Dict[int, List[Tuple[int, int]]] = {}

        self.edge_distance_m: np.ndarray = np.empty(0, dtype=float)
        self.edge_flight_time_s: np.ndarray = np.empty(0, dtype=float)
        self.edge_altitude_mid_m: np.ndarray = np.empty(0, dtype=float)
        self.edge_climb_rate_mps: np.ndarray = np.empty(0, dtype=float)
        self.edge_static_power_w: np.ndarray = np.empty(0, dtype=float)
        self.edge_wind_speed_mps: np.ndarray = np.empty(0, dtype=float)
        self.edge_h2_g: np.ndarray = np.empty(0, dtype=float)
        self.edge_psi_degradation: np.ndarray = np.empty(0, dtype=float)
        self.edge_time_cost: np.ndarray = np.empty(0, dtype=float)
        self.edge_h2_cost: np.ndarray = np.empty(0, dtype=float)
        self.edge_degradation_cost: np.ndarray = np.empty(0, dtype=float)
        self.edge_constraint_cost: np.ndarray = np.empty(0, dtype=float)
        self.edge_total_cost: np.ndarray = np.empty(0, dtype=float)
        self.edge_midpoints_xy: np.ndarray = np.empty((0, 2), dtype=float)
        self.edge_wind_version: np.ndarray = np.empty(0, dtype=int)
        self.edge_health_version: np.ndarray = np.empty(0, dtype=int)
        self.edge_cost_version: np.ndarray = np.empty(0, dtype=int)
        self.edge_dirty_reason: np.ndarray = np.empty(0, dtype=object)

        self._grid_to_node: Dict[Tuple[int, int, int], int] = {}
        self._edge_lookup: Dict[Tuple[int, int], int] = {}
        self._node_search_tree: cKDTree | None = None
        self._edge_search_tree: cKDTree | None = None
        self._node_search_scale = np.array([1.0, 1.0, 0.1], dtype=float)
        self._wind_version = 0
        self._health_version = 0
        self._cost_version = 0

    def clone_dynamic_state(self) -> "EnergyMap":
        """仅复制动态态。"""
        cloned = copy(self)
        cloned.wind_field = np.array(self.wind_field, copy=True)
        cloned.edge_altitude_mid_m = np.array(self.edge_altitude_mid_m, copy=True)
        cloned.edge_climb_rate_mps = np.array(self.edge_climb_rate_mps, copy=True)
        cloned.edge_static_power_w = np.array(self.edge_static_power_w, copy=True)
        cloned.edge_wind_speed_mps = np.array(self.edge_wind_speed_mps, copy=True)
        cloned.edge_h2_g = np.array(self.edge_h2_g, copy=True)
        cloned.edge_psi_degradation = np.array(self.edge_psi_degradation, copy=True)
        cloned.edge_time_cost = np.array(self.edge_time_cost, copy=True)
        cloned.edge_h2_cost = np.array(self.edge_h2_cost, copy=True)
        cloned.edge_degradation_cost = np.array(self.edge_degradation_cost, copy=True)
        cloned.edge_constraint_cost = np.array(self.edge_constraint_cost, copy=True)
        cloned.edge_total_cost = np.array(self.edge_total_cost, copy=True)
        cloned.edge_wind_version = np.array(self.edge_wind_version, copy=True)
        cloned.edge_health_version = np.array(self.edge_health_version, copy=True)
        cloned.edge_cost_version = np.array(self.edge_cost_version, copy=True)
        cloned.edge_dirty_reason = np.array(self.edge_dirty_reason, copy=True)
        cloned.dynamic_obstacles = [dict(item) for item in self.dynamic_obstacles]
        cloned.blocked_edge_ids = set(self.blocked_edge_ids)
        return cloned

    def _dem_rc(self, x_m: float, y_m: float) -> Tuple[int, int]:
        c = int(np.clip(x_m / config.DEM_RES, 0, self.dem_cols - 1))
        r = int(np.clip((self.dem_rows - 1) - y_m / config.DEM_RES, 0, self.dem_rows - 1))
        return r, c

    def _xy_from_lonlat(self, lon: float, lat: float) -> Tuple[float, float]:
        r, c = nearest_rc_from_lonlat(self.lon_grid, self.lat_grid, lon, lat)
        x_m = c * config.DEM_RES
        y_m = (self.dem_rows - 1 - r) * config.DEM_RES
        return x_m, y_m

    def _get_terrain(self, x_m: float, y_m: float) -> float:
        r, c = self._dem_rc(x_m, y_m)
        return float(self.z_dem[r, c])

    def _get_wind(self, x_m: float, y_m: float) -> float:
        r, c = self._dem_rc(x_m, y_m)
        return float(self.wind_field[r, c])

    @staticmethod
    def _point_to_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
        segment = end - start
        seg_norm_sq = float(np.dot(segment, segment))
        if seg_norm_sq <= 1e-9:
            return float(np.linalg.norm(point - start))
        ratio = float(np.dot(point - start, segment) / seg_norm_sq)
        ratio = min(1.0, max(0.0, ratio))
        projection = start + ratio * segment
        return float(np.linalg.norm(point - projection))

    def _node_within_corridor(
        self,
        x_m: float,
        y_m: float,
        start_xy: np.ndarray,
        goal_xy: np.ndarray,
        corridor_radius_m: float,
    ) -> bool:
        point = np.array([x_m, y_m], dtype=float)
        dist_line = self._point_to_segment_distance(point, start_xy, goal_xy)
        dist_start = float(np.linalg.norm(point - start_xy))
        dist_goal = float(np.linalg.norm(point - goal_xy))
        return (
            dist_line <= corridor_radius_m
            or dist_start <= corridor_radius_m + config.CORRIDOR_ENDPOINT_MARGIN_M
            or dist_goal <= corridor_radius_m + config.CORRIDOR_ENDPOINT_MARGIN_M
        )

    def _point_inside_shape(self, spec: Dict[str, Any], x_m: float, y_m: float, z_m: float) -> bool:
        shape = str(spec.get("shape", "sphere")).lower()
        if shape == "sphere":
            cx = float(spec.get("x", 0.0))
            cy = float(spec.get("y", 0.0))
            cz = float(spec.get("z", 0.0))
            radius = float(spec.get("radius", 0.0))
            return math.dist((x_m, y_m, z_m), (cx, cy, cz)) <= radius
        if shape == "box":
            return (
                float(spec.get("x_min", -math.inf)) <= x_m <= float(spec.get("x_max", math.inf))
                and float(spec.get("y_min", -math.inf)) <= y_m <= float(spec.get("y_max", math.inf))
                and float(spec.get("z_min", -math.inf)) <= z_m <= float(spec.get("z_max", math.inf))
            )
        if shape == "cylinder":
            cx = float(spec.get("x", 0.0))
            cy = float(spec.get("y", 0.0))
            radius = float(spec.get("radius", 0.0))
            z_min = float(spec.get("z_min", -math.inf))
            z_max = float(spec.get("z_max", math.inf))
            return math.hypot(x_m - cx, y_m - cy) <= radius and z_min <= z_m <= z_max
        return False

    def _node_allowed_static_constraints(self, x_m: float, y_m: float, z_m: float) -> bool:
        for obstacle in self.static_obstacles:
            if self._point_inside_shape(obstacle, x_m, y_m, z_m):
                return False
        for obstacle in self.dynamic_obstacles:
            if self._point_inside_shape(obstacle, x_m, y_m, z_m):
                return False
        for constraint in self.airspace_constraints:
            kind = str(constraint.get("kind", "no_fly")).lower()
            if kind == "no_fly" and self._point_inside_shape(constraint, x_m, y_m, z_m):
                return False
            if kind == "max_altitude" and z_m > float(constraint.get("z_max", math.inf)):
                return False
            if kind == "min_altitude" and z_m < float(constraint.get("z_min", -math.inf)):
                return False
        return True

    def _edge_feasible(self, start_xyz: np.ndarray, end_xyz: np.ndarray) -> bool:
        sample_count = max(config.CONSTRAINT_EDGE_SAMPLE_COUNT, 2)
        for ratio in np.linspace(0.0, 1.0, sample_count):
            point = start_xyz + ratio * (end_xyz - start_xyz)
            if not self._node_allowed_static_constraints(float(point[0]), float(point[1]), float(point[2])):
                return False
        return True

    def _constraint_cost_for_edge(self, start_xyz: np.ndarray, end_xyz: np.ndarray) -> float:
        del start_xyz, end_xyz
        return 0.0

    def calc_soh_weight(self, soh: float | None = None) -> float:
        """计算 SoH 权重。"""
        return float(config.k_soh(self.soh if soh is None else soh))

    def calc_degradation_proxy(self, p_total_w: float, climb_rate_mps: float, wind_speed_mps: float) -> float:
        """计算局部退化代理项。"""
        phi_climb = max(0.0, climb_rate_mps) / max(config.DEGRAD_CLIMB_RATE_REF_MPS, 1e-6)
        high_power_threshold = config.FC_RATED_POWER * config.DEGRAD_HIGH_POWER_RATIO
        phi_power = max(0.0, p_total_w - high_power_threshold) / max(high_power_threshold, 1e-6)
        phi_wind = max(0.0, wind_speed_mps - config.WIND_NORMAL) / max(config.DEGRAD_WIND_EXCESS_REF_MPS, 1e-6)
        return config.W1 * phi_climb + config.W2 * phi_power + config.W3 * phi_wind

    def calc_edge_cost(
        self,
        flight_time_s: float,
        h2_g: float,
        psi_degradation: float,
        constraint_cost: float = 0.0,
        soh: float | None = None,
    ) -> Dict[str, float]:
        """按说明书形式计算边代价。"""
        time_cost = config.ALPHA * flight_time_s
        h2_cost = config.BETA_H2_EFF * h2_g
        degradation_cost = config.GAMMA * psi_degradation * self.calc_soh_weight(soh)
        total_cost = time_cost + h2_cost + degradation_cost + constraint_cost
        return {
            "time_cost": float(time_cost),
            "h2_cost": float(h2_cost),
            "degradation_cost": float(degradation_cost),
            "constraint_cost": float(constraint_cost),
            "total_cost": float(total_cost),
        }

    def _edge_metrics_for_nodes(self, node_from: int, node_to: int) -> Dict[str, float]:
        n1 = self.nodes[node_from]
        n2 = self.nodes[node_to]
        dx = float(n2[0] - n1[0])
        dy = float(n2[1] - n1[1])
        dz = float(n2[2] - n1[2])
        distance_m = math.sqrt(dx * dx + dy * dy + dz * dz)
        flight_time_s = distance_m / config.CRUISE_SPEED if distance_m > 0.0 else 0.0
        alt_mid = 0.5 * (n1[2] + n2[2])
        climb_rate = dz / max(flight_time_s, 0.1) if flight_time_s > 0.0 else 0.0
        mid_x = 0.5 * (n1[0] + n2[0])
        mid_y = 0.5 * (n1[1] + n2[1])
        wind_speed = self._get_wind(mid_x, mid_y)

        p_hover = _hover_power(alt_mid)
        p_climb = _climb_power(climb_rate)
        p_cruise = _cruise_power(config.CRUISE_SPEED, wind_speed, alt_mid)
        static_power = p_hover + p_climb
        p_total = static_power + p_cruise
        h2_g = _hydrogen_consumption(p_total, flight_time_s)
        psi_degradation_local = self.calc_degradation_proxy(p_total, climb_rate, wind_speed)
        constraint_cost = self._constraint_cost_for_edge(n1, n2)
        cost_breakdown = self.calc_edge_cost(
            flight_time_s=flight_time_s,
            h2_g=h2_g,
            psi_degradation=psi_degradation_local,
            constraint_cost=constraint_cost,
        )
        return {
            "distance_m": distance_m,
            "flight_time_s": flight_time_s,
            "wind_speed_mps": wind_speed,
            "static_power_w": static_power,
            "p_total_w": p_total,
            "h2_g": h2_g,
            "time_cost": cost_breakdown["time_cost"],
            "h2_cost": cost_breakdown["h2_cost"],
            "degradation_cost": cost_breakdown["degradation_cost"],
            "constraint_cost": cost_breakdown["constraint_cost"],
            "psi_degradation_local": psi_degradation_local,
            "cost": cost_breakdown["total_cost"],
            "mid_x": mid_x,
            "mid_y": mid_y,
            "climb_rate_mps": climb_rate,
            "altitude_mid_m": alt_mid,
        }

    def build_graph(self, corridor_radius_m: float = config.INITIAL_CORRIDOR_RADIUS_M) -> None:
        """构建走廊图。"""
        start_xy = np.array(self._xy_from_lonlat(config.START_LON, config.START_LAT), dtype=float)
        goal_xy = np.array(self._xy_from_lonlat(config.GOAL_LON, config.GOAL_LAT), dtype=float)

        h_res = config.GRID_H_RES
        v_res = config.GRID_V_RES
        dem_extent_x = self.dem_cols * config.DEM_RES
        dem_extent_y = self.dem_rows * config.DEM_RES
        nx = int(dem_extent_x / h_res)
        ny = int(dem_extent_y / h_res)

        z_min = float(np.nanmin(self.z_dem))
        z_max = float(np.nanmax(self.z_dem))
        alt_min = z_min + 30.0
        alt_max = z_max + 200.0
        nz = int((alt_max - alt_min) / v_res) + 1

        node_list: list[list[float]] = []
        self._grid_to_node = {}
        for iy in range(ny):
            for ix in range(nx):
                x_m = (ix + 0.5) * h_res
                y_m = (iy + 0.5) * h_res
                if not self._node_within_corridor(x_m, y_m, start_xy, goal_xy, corridor_radius_m):
                    continue
                terrain_h = self._get_terrain(x_m, y_m)
                if np.isnan(terrain_h):
                    continue
                for iz in range(nz):
                    z_m = alt_min + iz * v_res
                    if z_m < terrain_h + 20.0:
                        continue
                    if z_m > terrain_h + 300.0:
                        continue
                    if not self._node_allowed_static_constraints(x_m, y_m, z_m):
                        continue
                    node_id = len(node_list)
                    self._grid_to_node[(ix, iy, iz)] = node_id
                    node_list.append([x_m, y_m, z_m])

        self.nodes = np.array(node_list, dtype=float)
        if len(self.nodes) == 0:
            raise RuntimeError("Corridor graph build produced no nodes.")

        offsets = [
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            if not (dx == 0 and dy == 0 and dz == 0)
        ]

        edge_list: list[Tuple[int, int]] = []
        edge_distance: list[float] = []
        edge_time: list[float] = []
        edge_altitude_mid: list[float] = []
        edge_climb_rate: list[float] = []
        edge_static_power: list[float] = []
        edge_wind_speed: list[float] = []
        edge_h2: list[float] = []
        edge_psi: list[float] = []
        edge_time_cost: list[float] = []
        edge_h2_cost: list[float] = []
        edge_degradation_cost: list[float] = []
        edge_constraint_cost: list[float] = []
        edge_total_cost: list[float] = []
        edge_midpoints: list[Tuple[float, float]] = []

        for (ix, iy, iz), node_id in self._grid_to_node.items():
            start_xyz = self.nodes[node_id]
            for dx, dy, dz in offsets:
                neighbor_id = self._grid_to_node.get((ix + dx, iy + dy, iz + dz))
                if neighbor_id is None:
                    continue
                end_xyz = self.nodes[neighbor_id]
                if not self._edge_feasible(start_xyz, end_xyz):
                    continue
                metrics = self._edge_metrics_for_nodes(node_id, neighbor_id)
                edge_list.append((node_id, neighbor_id))
                edge_distance.append(metrics["distance_m"])
                edge_time.append(metrics["flight_time_s"])
                edge_altitude_mid.append(metrics["altitude_mid_m"])
                edge_climb_rate.append(metrics["climb_rate_mps"])
                edge_static_power.append(metrics["static_power_w"])
                edge_wind_speed.append(metrics["wind_speed_mps"])
                edge_h2.append(metrics["h2_g"])
                edge_psi.append(metrics["psi_degradation_local"])
                edge_time_cost.append(metrics["time_cost"])
                edge_h2_cost.append(metrics["h2_cost"])
                edge_degradation_cost.append(metrics["degradation_cost"])
                edge_constraint_cost.append(metrics["constraint_cost"])
                edge_total_cost.append(metrics["cost"])
                edge_midpoints.append((metrics["mid_x"], metrics["mid_y"]))

        self.edges = edge_list
        self._edge_lookup = {edge: idx for idx, edge in enumerate(self.edges)}
        self.edge_distance_m = np.array(edge_distance, dtype=float)
        self.edge_flight_time_s = np.array(edge_time, dtype=float)
        self.edge_altitude_mid_m = np.array(edge_altitude_mid, dtype=float)
        self.edge_climb_rate_mps = np.array(edge_climb_rate, dtype=float)
        self.edge_static_power_w = np.array(edge_static_power, dtype=float)
        self.edge_wind_speed_mps = np.array(edge_wind_speed, dtype=float)
        self.edge_h2_g = np.array(edge_h2, dtype=float)
        self.edge_psi_degradation = np.array(edge_psi, dtype=float)
        self.edge_time_cost = np.array(edge_time_cost, dtype=float)
        self.edge_h2_cost = np.array(edge_h2_cost, dtype=float)
        self.edge_degradation_cost = np.array(edge_degradation_cost, dtype=float)
        self.edge_constraint_cost = np.array(edge_constraint_cost, dtype=float)
        self.edge_total_cost = np.array(edge_total_cost, dtype=float)
        self.edge_midpoints_xy = np.array(edge_midpoints, dtype=float)
        self.edge_wind_version = np.zeros(len(self.edges), dtype=int)
        self.edge_health_version = np.zeros(len(self.edges), dtype=int)
        self.edge_cost_version = np.zeros(len(self.edges), dtype=int)
        self.edge_dirty_reason = np.full(len(self.edges), "init", dtype=object)

        self.adj = {i: [] for i in range(len(self.nodes))}
        self.rev_adj = {i: [] for i in range(len(self.nodes))}
        for edge_id, (u, v) in enumerate(self.edges):
            self.adj[u].append((v, edge_id))
            self.rev_adj[v].append((u, edge_id))

        scaled_nodes = self.nodes * self._node_search_scale
        self._node_search_tree = cKDTree(scaled_nodes)
        self._edge_search_tree = cKDTree(self.edge_midpoints_xy) if len(self.edge_midpoints_xy) else None

    def position_from_node(self, node_id: int) -> Tuple[float, float, float]:
        node = self.nodes[node_id]
        return float(node[0]), float(node[1]), float(node[2])

    def find_nearest_node(self, lon: float, lat: float, alt: float) -> int:
        x_m, y_m = self._xy_from_lonlat(lon, lat)
        return self.find_nearest_node_by_xyz(x_m, y_m, alt)

    def find_nearest_node_by_xyz(self, x_m: float, y_m: float, z_m: float) -> int:
        if self._node_search_tree is None:
            raise RuntimeError("Graph must be built before nearest-node queries.")
        query = np.array([x_m, y_m, z_m], dtype=float) * self._node_search_scale
        _, node_id = self._node_search_tree.query(query)
        return int(node_id)

    def get_edge_cost(self, edge_id: int) -> float:
        if edge_id in self.blocked_edge_ids:
            return float("inf")
        return float(self.edge_total_cost[edge_id])

    def _normalize_edge_ids(self, edge_ids: List[int] | np.ndarray | None) -> np.ndarray:
        if edge_ids is None:
            return np.arange(len(self.edges), dtype=int)
        if len(edge_ids) == 0:
            return np.empty(0, dtype=int)
        return np.unique(np.asarray(edge_ids, dtype=int))

    def _next_cost_version(self) -> int:
        self._cost_version += 1
        return self._cost_version

    def _compose_total_cost_for_edges(self, edge_ids: List[int] | np.ndarray | None) -> None:
        idx = self._normalize_edge_ids(edge_ids)
        if len(idx) == 0:
            return
        self.edge_total_cost[idx] = (
            self.edge_time_cost[idx]
            + self.edge_h2_cost[idx]
            + self.edge_degradation_cost[idx]
            + self.edge_constraint_cost[idx]
        )

    def _mark_edge_versions(
        self,
        idx: np.ndarray,
        *,
        wind_changed: bool = False,
        health_changed: bool = False,
        reason: str,
    ) -> None:
        if len(idx) == 0:
            return
        if wind_changed:
            self.edge_wind_version[idx] = self._wind_version
        if health_changed:
            self.edge_health_version[idx] = self._health_version
        self.edge_cost_version[idx] = self._next_cost_version()
        self.edge_dirty_reason[idx] = reason

    def refresh_health_terms(
        self,
        edge_ids: List[int] | np.ndarray | None = None,
        *,
        reason: str = "health_update",
    ) -> Dict[str, object]:
        idx = self._normalize_edge_ids(edge_ids)
        if len(idx) == 0:
            return {
                "edge_ids": [],
                "candidate_edges": 0,
                "updated_edges": 0,
            }

        self.edge_degradation_cost[idx] = (
            config.GAMMA * self.edge_psi_degradation[idx] * self.calc_soh_weight()
        )
        self._compose_total_cost_for_edges(idx)
        self._mark_edge_versions(idx, health_changed=True, reason=reason)
        return {
            "edge_ids": idx.astype(int).tolist(),
            "candidate_edges": int(len(idx)),
            "updated_edges": int(len(idx)),
        }

    def refresh_wind_terms(
        self,
        edge_ids: List[int] | np.ndarray,
        *,
        reason: str = "wind_update",
    ) -> Dict[str, object]:
        idx = self._normalize_edge_ids(edge_ids)
        if len(idx) == 0:
            return {
                "candidate_edge_ids": [],
                "old_costs": [],
                "new_costs": [],
                "candidate_edges": 0,
                "updated_edges": 0,
            }

        sampled_wind = np.array(
            [
                self._get_wind(
                    float(self.edge_midpoints_xy[edge_id, 0]),
                    float(self.edge_midpoints_xy[edge_id, 1]),
                )
                for edge_id in idx
            ],
            dtype=float,
        )
        changed_mask = np.abs(sampled_wind - self.edge_wind_speed_mps[idx]) > 1e-9
        changed_idx = idx[changed_mask]
        if len(changed_idx) == 0:
            return {
                "candidate_edge_ids": [],
                "old_costs": [],
                "new_costs": [],
                "candidate_edges": int(len(idx)),
                "updated_edges": 0,
            }

        sampled_changed_wind = sampled_wind[changed_mask]
        old_cost = np.array(self.edge_total_cost[changed_idx], copy=True)
        p_cruise = np.array(
            [
                _cruise_power(config.CRUISE_SPEED, float(wind_speed), float(altitude_mid))
                for wind_speed, altitude_mid in zip(
                    sampled_changed_wind,
                    self.edge_altitude_mid_m[changed_idx],
                )
            ],
            dtype=float,
        )
        p_total = self.edge_static_power_w[changed_idx] + p_cruise
        h2_g = (
            p_total
            / (0.50 * 120.0e3)
            * self.edge_flight_time_s[changed_idx]
        )
        psi = np.array(
            [
                self.calc_degradation_proxy(
                    float(power_total),
                    float(climb_rate),
                    float(wind_speed),
                )
                for power_total, climb_rate, wind_speed in zip(
                    p_total,
                    self.edge_climb_rate_mps[changed_idx],
                    sampled_changed_wind,
                )
            ],
            dtype=float,
        )

        self.edge_wind_speed_mps[changed_idx] = sampled_changed_wind
        self.edge_h2_g[changed_idx] = h2_g
        self.edge_psi_degradation[changed_idx] = psi
        self.edge_h2_cost[changed_idx] = config.BETA_H2_EFF * h2_g
        self.edge_degradation_cost[changed_idx] = config.GAMMA * psi * self.calc_soh_weight()
        self._compose_total_cost_for_edges(changed_idx)
        self._mark_edge_versions(changed_idx, wind_changed=True, reason=reason)

        return {
            "candidate_edge_ids": changed_idx.astype(int).tolist(),
            "old_costs": old_cost.astype(float).tolist(),
            "new_costs": self.edge_total_cost[changed_idx].astype(float).tolist(),
            "candidate_edges": int(len(idx)),
            "updated_edges": int(len(changed_idx)),
        }

    def set_soh(
        self,
        soh: float,
        edge_ids: List[int] | np.ndarray | None = None,
        *,
        reason: str = "health_update",
    ) -> Dict[str, object]:
        """更新健康权重，并只刷新指定脏边。"""
        clipped_soh = float(np.clip(soh, config.MIN_SOH, config.INITIAL_SOH))
        soh_changed = abs(clipped_soh - self.soh) > 1e-9
        self.soh = clipped_soh
        if len(self.edge_total_cost) == 0:
            return {
                "edge_ids": [],
                "candidate_edges": 0,
                "updated_edges": 0,
            }
        if soh_changed:
            self._health_version += 1
        return self.refresh_health_terms(edge_ids, reason=reason)

    def path_edge_ids(self, path_nodes: List[int]) -> List[int]:
        if len(path_nodes) < 2:
            return []
        edge_ids: list[int] = []
        for idx in range(len(path_nodes) - 1):
            edge_id = self._edge_lookup.get((path_nodes[idx], path_nodes[idx + 1]))
            if edge_id is not None:
                edge_ids.append(edge_id)
        return edge_ids

    def path_blocked_edges(self, path_nodes: List[int]) -> List[int]:
        return [edge_id for edge_id in self.path_edge_ids(path_nodes) if edge_id in self.blocked_edge_ids]

    def block_edges(self, edge_ids: List[int]) -> None:
        self.blocked_edge_ids.update(int(edge_id) for edge_id in edge_ids)

    def clear_blocked_edges(self, edge_ids: List[int] | None = None) -> None:
        if edge_ids is None:
            self.blocked_edge_ids.clear()
            return
        for edge_id in edge_ids:
            self.blocked_edge_ids.discard(int(edge_id))

    def add_dynamic_obstacle(self, obstacle: Dict[str, Any]) -> Dict[str, object]:
        """新增动态障碍物并标记受阻边。"""
        self.dynamic_obstacles.append(dict(obstacle))
        blocked_edges: list[int] = []
        for edge_id, (u, v) in enumerate(self.edges):
            start_xyz = self.nodes[u]
            end_xyz = self.nodes[v]
            if not self._edge_feasible(start_xyz, end_xyz):
                blocked_edges.append(edge_id)
        self.block_edges(blocked_edges)
        return {
            "blocked_edge_ids": blocked_edges,
            "blocked_edges": len(blocked_edges),
        }

    def find_edges_near_path(
        self,
        current_xyz: Tuple[float, float, float],
        remaining_path_nodes: List[int],
        radius_m: float,
        max_waypoints: int | None = None,
    ) -> List[int]:
        if self._edge_search_tree is None or len(self.edge_midpoints_xy) == 0:
            return []
        if max_waypoints is None:
            max_waypoints = config.T4_LOCAL_MAX_WAYPOINTS

        waypoints = [np.array(current_xyz[:2], dtype=float)]
        for node_id in remaining_path_nodes:
            node_xyz = self.position_from_node(node_id)
            waypoints.append(np.array(node_xyz[:2], dtype=float))

        sampled_waypoints = waypoints[:max_waypoints]
        edge_ids: set[int] = set()
        for waypoint in sampled_waypoints:
            for edge_id in self._edge_search_tree.query_ball_point(waypoint, radius_m):
                edge_ids.add(int(edge_id))
        return sorted(edge_ids)

    def filter_edges_by_cost_delta(
        self,
        edge_ids: List[int],
        previous_soh: float,
        new_soh: float,
        ratio_threshold: float,
        abs_threshold: float,
    ) -> Dict[str, object]:
        if not edge_ids:
            return {
                "edge_ids": [],
                "candidate_edges": 0,
                "updated_edges": 0,
                "max_cost_delta_ratio": 0.0,
                "max_cost_delta_abs": 0.0,
            }

        idx = np.asarray(edge_ids, dtype=int)
        old_weight = float(config.k_soh(previous_soh))
        new_weight = float(config.k_soh(new_soh))
        old_cost = self.edge_time_cost[idx] + self.edge_h2_cost[idx] + config.GAMMA * self.edge_psi_degradation[idx] * old_weight + self.edge_constraint_cost[idx]
        new_cost = self.edge_time_cost[idx] + self.edge_h2_cost[idx] + config.GAMMA * self.edge_psi_degradation[idx] * new_weight + self.edge_constraint_cost[idx]

        delta_abs = np.abs(new_cost - old_cost)
        delta_ratio = delta_abs / np.maximum(old_cost, 1e-9)
        mask = (delta_ratio >= ratio_threshold) | (delta_abs >= abs_threshold)
        filtered_ids = idx[mask].astype(int).tolist()
        return {
            "edge_ids": filtered_ids,
            "candidate_edges": int(len(idx)),
            "updated_edges": int(mask.sum()),
            "max_cost_delta_ratio": float(np.max(delta_ratio)) if len(delta_ratio) else 0.0,
            "max_cost_delta_abs": float(np.max(delta_abs)) if len(delta_abs) else 0.0,
        }

    @staticmethod
    def filter_edge_cost_delta_values(
        edge_ids: List[int],
        old_costs: np.ndarray,
        new_costs: np.ndarray,
        ratio_threshold: float,
        abs_threshold: float,
    ) -> Dict[str, object]:
        if not edge_ids:
            return {
                "edge_ids": [],
                "candidate_edges": 0,
                "updated_edges": 0,
                "max_cost_delta_ratio": 0.0,
                "max_cost_delta_abs": 0.0,
            }

        idx = np.asarray(edge_ids, dtype=int)
        old_cost = np.asarray(old_costs, dtype=float)
        new_cost = np.asarray(new_costs, dtype=float)
        delta_abs = np.abs(new_cost - old_cost)
        delta_ratio = delta_abs / np.maximum(old_cost, 1e-9)
        mask = (delta_ratio >= ratio_threshold) | (delta_abs >= abs_threshold)
        filtered_ids = idx[mask].astype(int).tolist()
        return {
            "edge_ids": filtered_ids,
            "candidate_edges": int(len(idx)),
            "updated_edges": int(mask.sum()),
            "max_cost_delta_ratio": float(np.max(delta_ratio)) if len(delta_ratio) else 0.0,
            "max_cost_delta_abs": float(np.max(delta_abs)) if len(delta_abs) else 0.0,
        }

    def update_wind_field(
        self,
        region_center: Tuple[float, float],
        radius_m: float,
        new_wind_speed: float,
        candidate_scope_edge_ids: List[int] | None = None,
    ) -> Dict[str, object]:
        """更新局部风场并回写边代价。"""
        cx_m, cy_m = region_center

        c_min = max(0, int(math.floor((cx_m - radius_m) / config.DEM_RES)))
        c_max = min(self.dem_cols - 1, int(math.ceil((cx_m + radius_m) / config.DEM_RES)))
        r_min = max(0, int(math.floor((self.dem_rows - 1) - (cy_m + radius_m) / config.DEM_RES)))
        r_max = min(self.dem_rows - 1, int(math.ceil((self.dem_rows - 1) - (cy_m - radius_m) / config.DEM_RES)))

        updated_cells = 0
        for r in range(r_min, r_max + 1):
            y_m = (self.dem_rows - 1 - r) * config.DEM_RES
            for c in range(c_min, c_max + 1):
                x_m = c * config.DEM_RES
                if math.hypot(x_m - cx_m, y_m - cy_m) <= radius_m:
                    self.wind_field[r, c] = new_wind_speed
                    updated_cells += 1

        candidate_edges: list[int] = []
        if self._edge_search_tree is not None and len(self.edge_midpoints_xy):
            candidate_edges = sorted(
                int(edge_id)
                for edge_id in self._edge_search_tree.query_ball_point(
                    np.array([cx_m, cy_m], dtype=float),
                    radius_m + config.GRID_H_RES * 2.0,
                )
            )
        scope_edge_count = 0
        if candidate_scope_edge_ids is not None:
            scope_edge_set = {int(edge_id) for edge_id in candidate_scope_edge_ids}
            scope_edge_count = len(scope_edge_set)
            candidate_edges = [edge_id for edge_id in candidate_edges if edge_id in scope_edge_set]

        if updated_cells > 0:
            self._wind_version += 1
        active_candidate_edges = [
            edge_id for edge_id in candidate_edges if edge_id not in self.blocked_edge_ids
        ]
        refresh_stats = self.refresh_wind_terms(active_candidate_edges, reason="wind_update")

        return {
            "candidate_edge_ids": list(refresh_stats["candidate_edge_ids"]),
            "old_costs": list(refresh_stats["old_costs"]),
            "new_costs": list(refresh_stats["new_costs"]),
            "candidate_edges": int(refresh_stats["candidate_edges"]),
            "updated_edges": int(refresh_stats["updated_edges"]),
            "updated_cells": int(updated_cells),
            "window_rows": int(r_max - r_min + 1) if r_max >= r_min else 0,
            "window_cols": int(c_max - c_min + 1) if c_max >= c_min else 0,
            "scope_edge_count": int(scope_edge_count),
        }

    def compute_power_for_segment(
        self,
        node_from: int,
        node_to: int,
        wind_speed: float | None = None,
    ) -> Dict[str, float]:
        """计算路径段功率与代价分量。"""
        n1 = self.nodes[node_from]
        n2 = self.nodes[node_to]
        dx = float(n2[0] - n1[0])
        dy = float(n2[1] - n1[1])
        dz = float(n2[2] - n1[2])
        d_h = math.sqrt(dx * dx + dy * dy)
        d_3d = math.sqrt(d_h ** 2 + dz ** 2)
        flight_time = d_3d / config.CRUISE_SPEED if d_3d > 0.0 else 0.0
        alt_mid = 0.5 * (n1[2] + n2[2])
        climb_rate = dz / max(flight_time, 0.1) if flight_time > 0.0 else 0.0
        mid_x = 0.5 * (n1[0] + n2[0])
        mid_y = 0.5 * (n1[1] + n2[1])
        local_wind = self._get_wind(mid_x, mid_y) if wind_speed is None else float(wind_speed)

        p_hover = _hover_power(alt_mid)
        p_climb = _climb_power(climb_rate)
        p_cruise = _cruise_power(config.CRUISE_SPEED, local_wind, alt_mid)
        p_total = p_hover + p_climb + p_cruise
        h2 = _hydrogen_consumption(p_total, flight_time)
        psi_degradation_local = self.calc_degradation_proxy(p_total, climb_rate, local_wind)
        cost_breakdown = self.calc_edge_cost(
            flight_time_s=flight_time,
            h2_g=h2,
            psi_degradation=psi_degradation_local,
            constraint_cost=self._constraint_cost_for_edge(n1, n2),
        )
        return {
            "p_hover": p_hover,
            "p_climb": p_climb,
            "p_cruise": p_cruise,
            "p_total": p_total,
            "flight_time": flight_time,
            "distance_m": d_3d,
            "distance_h": d_h,
            "delta_h": dz,
            "climb_rate": climb_rate,
            "h2_consumption_g": h2,
            "psi_degradation_local": psi_degradation_local,
            "time_cost": cost_breakdown["time_cost"],
            "h2_cost": cost_breakdown["h2_cost"],
            "degradation_cost": cost_breakdown["degradation_cost"],
            "constraint_cost": cost_breakdown["constraint_cost"],
            "edge_cost": cost_breakdown["total_cost"],
            "altitude_mid": alt_mid,
            "wind_speed": local_wind,
            "mid_x": mid_x,
            "mid_y": mid_y,
            "start_x": float(n1[0]),
            "start_y": float(n1[1]),
            "start_z": float(n1[2]),
            "end_x": float(n2[0]),
            "end_y": float(n2[1]),
            "end_z": float(n2[2]),
        }
