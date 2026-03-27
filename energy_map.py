"""Energy map and dynamic environment model."""

from __future__ import annotations

import math
from copy import copy
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial import cKDTree

import config
from dem_loader import nearest_rc_from_lonlat


def _hover_power(altitude_m: float) -> float:
    """Estimate hover power."""
    rho = config.air_density(altitude_m)
    mg = config.MASS * config.GRAVITY
    return (mg ** 1.5) / math.sqrt(2.0 * rho * config.ROTOR_AREA)


def _climb_power(climb_rate: float) -> float:
    """Estimate climb power."""
    if climb_rate <= 0.0:
        return 0.0
    return config.MASS * config.GRAVITY * climb_rate


def _cruise_power(speed: float, wind_speed: float, altitude_m: float) -> float:
    """Estimate cruise power with local wind disturbance."""
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
    """Estimate hydrogen consumption in grams."""
    if duration_s <= 0.0 or power_w <= 0.0:
        return 0.0
    eta_fc = 0.50
    lhv_h2 = 120.0e3
    return power_w / (eta_fc * lhv_h2) * duration_s


class EnergyMap:
    """Corridor graph plus dynamic wind-dependent edge costs."""

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

        self.nodes: np.ndarray = np.empty((0, 3), dtype=float)
        self.edges: List[Tuple[int, int]] = []
        self.adj: Dict[int, List[Tuple[int, int]]] = {}
        self.rev_adj: Dict[int, List[Tuple[int, int]]] = {}

        self.edge_distance_m: np.ndarray = np.empty(0, dtype=float)
        self.edge_flight_time_s: np.ndarray = np.empty(0, dtype=float)
        self.edge_h2_g: np.ndarray = np.empty(0, dtype=float)
        self.edge_psi_degradation: np.ndarray = np.empty(0, dtype=float)
        self.edge_costs: np.ndarray = np.empty(0, dtype=float)
        self.edge_midpoints_xy: np.ndarray = np.empty((0, 2), dtype=float)

        self._grid_to_node: Dict[Tuple[int, int, int], int] = {}
        self._node_search_tree: cKDTree | None = None
        self._node_search_scale = np.array([1.0, 1.0, 0.1], dtype=float)

    def clone_dynamic_state(self) -> "EnergyMap":
        """Clone only the dynamic fields so each chain can evolve independently."""
        cloned = copy(self)
        cloned.wind_field = np.array(self.wind_field, copy=True)
        cloned.edge_h2_g = np.array(self.edge_h2_g, copy=True)
        cloned.edge_psi_degradation = np.array(self.edge_psi_degradation, copy=True)
        cloned.edge_costs = np.array(self.edge_costs, copy=True)
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
        t = float(np.dot(point - start, segment) / seg_norm_sq)
        t = min(1.0, max(0.0, t))
        projection = start + t * segment
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
        p_climb = _climb_power(max(0.0, climb_rate))
        p_cruise = _cruise_power(config.CRUISE_SPEED, wind_speed, alt_mid)
        p_total = p_hover + p_climb + p_cruise
        h2_g = _hydrogen_consumption(p_total, flight_time_s)
        psi_degradation_local = self._local_degradation_proxy(p_total, climb_rate, wind_speed)
        cost = (
            config.ALPHA * flight_time_s
            + config.BETA * h2_g * config.H2_COST_SCALE
            + config.GAMMA * psi_degradation_local
        )
        return {
            "distance_m": distance_m,
            "flight_time_s": flight_time_s,
            "wind_speed_mps": wind_speed,
            "p_total_w": p_total,
            "h2_g": h2_g,
            "psi_degradation_local": psi_degradation_local,
            "cost": cost,
            "mid_x": mid_x,
            "mid_y": mid_y,
            "climb_rate_mps": climb_rate,
            "altitude_mid_m": alt_mid,
        }

    def _local_degradation_proxy(self, p_total_w: float, climb_rate_mps: float, wind_speed_mps: float) -> float:
        """Return a local static stress proxy for graph-level degradation avoidance."""
        climb_penalty = max(0.0, climb_rate_mps) / max(config.DEGRAD_CLIMB_RATE_REF_MPS, 1e-6)
        high_power_threshold = config.FC_RATED_POWER * config.DEGRAD_HIGH_POWER_RATIO
        high_power_penalty = max(0.0, p_total_w - high_power_threshold) / max(high_power_threshold, 1e-6)
        wind_penalty = max(0.0, wind_speed_mps - config.WIND_NORMAL) / max(
            config.DEGRAD_WIND_EXCESS_REF_MPS,
            1e-6,
        )
        return config.k_soh(self.soh) * (
            config.DEGRAD_CLIMB_WEIGHT * climb_penalty
            + config.DEGRAD_HIGH_POWER_WEIGHT * high_power_penalty
            + config.DEGRAD_WIND_WEIGHT * wind_penalty
        )

    def build_graph(self, corridor_radius_m: float = config.INITIAL_CORRIDOR_RADIUS_M) -> None:
        """Build a corridor graph around the mission start-goal line."""
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
                    node_id = len(node_list)
                    self._grid_to_node[(ix, iy, iz)] = node_id
                    node_list.append([x_m, y_m, z_m])

        self.nodes = np.array(node_list, dtype=float)
        if len(self.nodes) == 0:
            raise RuntimeError("No nodes generated for the corridor graph.")

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
        edge_h2: list[float] = []
        edge_psi: list[float] = []
        edge_costs: list[float] = []
        edge_midpoints: list[Tuple[float, float]] = []

        for (ix, iy, iz), node_id in self._grid_to_node.items():
            for dx, dy, dz in offsets:
                neighbor_id = self._grid_to_node.get((ix + dx, iy + dy, iz + dz))
                if neighbor_id is None:
                    continue
                metrics = self._edge_metrics_for_nodes(node_id, neighbor_id)
                edge_list.append((node_id, neighbor_id))
                edge_distance.append(metrics["distance_m"])
                edge_time.append(metrics["flight_time_s"])
                edge_h2.append(metrics["h2_g"])
                edge_psi.append(metrics["psi_degradation_local"])
                edge_costs.append(metrics["cost"])
                edge_midpoints.append((metrics["mid_x"], metrics["mid_y"]))

        self.edges = edge_list
        self.edge_distance_m = np.array(edge_distance, dtype=float)
        self.edge_flight_time_s = np.array(edge_time, dtype=float)
        self.edge_h2_g = np.array(edge_h2, dtype=float)
        self.edge_psi_degradation = np.array(edge_psi, dtype=float)
        self.edge_costs = np.array(edge_costs, dtype=float)
        self.edge_midpoints_xy = np.array(edge_midpoints, dtype=float)

        self.adj = {i: [] for i in range(len(self.nodes))}
        self.rev_adj = {i: [] for i in range(len(self.nodes))}
        for edge_id, (u, v) in enumerate(self.edges):
            self.adj[u].append((v, edge_id))
            self.rev_adj[v].append((u, edge_id))

        scaled_nodes = self.nodes * self._node_search_scale
        self._node_search_tree = cKDTree(scaled_nodes)

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
        return float(self.edge_costs[edge_id])

    def set_soh(self, soh: float) -> None:
        """Apply a new health state and refresh all graph edge costs."""
        self.soh = float(np.clip(soh, config.MIN_SOH, config.INITIAL_SOH))
        for edge_id, (u, v) in enumerate(self.edges):
            metrics = self._edge_metrics_for_nodes(u, v)
            self.edge_h2_g[edge_id] = metrics["h2_g"]
            self.edge_psi_degradation[edge_id] = metrics["psi_degradation_local"]
            self.edge_costs[edge_id] = metrics["cost"]

    def update_wind_field(
        self,
        region_center: Tuple[float, float],
        radius_m: float,
        new_wind_speed: float,
    ) -> List[int]:
        """Apply a local wind event and update only nearby edges."""
        cx_m, cy_m = region_center

        for r in range(self.dem_rows):
            for c in range(self.dem_cols):
                x_m = c * config.DEM_RES
                y_m = (self.dem_rows - 1 - r) * config.DEM_RES
                if math.hypot(x_m - cx_m, y_m - cy_m) <= radius_m:
                    self.wind_field[r, c] = new_wind_speed

        affected_edges: list[int] = []
        for edge_id, (u, v) in enumerate(self.edges):
            mid_x, mid_y = self.edge_midpoints_xy[edge_id]
            if math.hypot(mid_x - cx_m, mid_y - cy_m) > radius_m + config.GRID_H_RES * 2.0:
                continue
            old_cost = float(self.edge_costs[edge_id])
            metrics = self._edge_metrics_for_nodes(u, v)
            self.edge_h2_g[edge_id] = metrics["h2_g"]
            self.edge_psi_degradation[edge_id] = metrics["psi_degradation_local"]
            self.edge_costs[edge_id] = metrics["cost"]
            if metrics["cost"] > old_cost * config.WIND_TRIGGER_RATIO:
                affected_edges.append(edge_id)
        return affected_edges

    def compute_power_for_segment(
        self,
        node_from: int,
        node_to: int,
        wind_speed: float | None = None,
    ) -> Dict[str, float]:
        """Return the local power model for a path segment."""
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
        p_climb = _climb_power(max(0.0, climb_rate))
        p_cruise = _cruise_power(config.CRUISE_SPEED, local_wind, alt_mid)
        p_total = p_hover + p_climb + p_cruise
        h2 = _hydrogen_consumption(p_total, flight_time)
        psi_degradation_local = self._local_degradation_proxy(p_total, climb_rate, local_wind)
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
