"""Path planners for the corridor energy map."""

from __future__ import annotations

import heapq
import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

import config
from energy_map import EnergyMap


class LPAStar:
    """Reverse LPA* with fixed goal and movable start."""

    def __init__(self, energy_map: EnergyMap, goal: int):
        self.em = energy_map
        self.goal = goal
        self.current_start = goal
        self.n_nodes = len(energy_map.nodes)
        self.inf = float("inf")

        self.g = np.full(self.n_nodes, self.inf, dtype=float)
        self.rhs = np.full(self.n_nodes, self.inf, dtype=float)
        self.rhs[goal] = 0.0

        self._counter = 0
        self._heap: list[Tuple[float, float, int, int]] = []
        self._in_heap: Dict[int, Tuple[float, float]] = {}
        self._edge_cost_override: Dict[int, float] = {}

        self.nodes_expanded = 0
        self.expanded_nodes_list: List[int] = []

        self._push(goal, self._calc_key(goal))

    def _get_edge_cost(self, edge_id: int) -> float:
        if edge_id in self._edge_cost_override:
            return self._edge_cost_override[edge_id]
        return self.em.get_edge_cost(edge_id)

    def _heuristic(self, node_id: int) -> float:
        node = self.em.nodes[node_id]
        start = self.em.nodes[self.current_start]
        dx = float(start[0] - node[0])
        dy = float(start[1] - node[1])
        dz = float(start[2] - node[2])
        d3d = math.sqrt(dx * dx + dy * dy + dz * dz)
        return config.ALPHA * (d3d / config.CRUISE_SPEED)

    def _calc_key(self, node_id: int) -> Tuple[float, float]:
        base = min(self.g[node_id], self.rhs[node_id])
        return (base + self._heuristic(node_id), base)

    def _push(self, node_id: int, key: Tuple[float, float]) -> None:
        self._counter += 1
        heapq.heappush(self._heap, (key[0], key[1], self._counter, node_id))
        self._in_heap[node_id] = key

    def _pop(self) -> Tuple[Optional[int], Optional[Tuple[float, float]]]:
        while self._heap:
            k1, k2, _, node_id = heapq.heappop(self._heap)
            cur_key = self._in_heap.get(node_id)
            if cur_key is not None and cur_key == (k1, k2):
                del self._in_heap[node_id]
                return node_id, (k1, k2)
        return None, None

    def _top_key(self) -> Tuple[float, float]:
        while self._heap:
            k1, k2, _, node_id = self._heap[0]
            cur_key = self._in_heap.get(node_id)
            if cur_key is not None and cur_key == (k1, k2):
                return (k1, k2)
            heapq.heappop(self._heap)
        return (self.inf, self.inf)

    def _rekey_open_set(self) -> None:
        nodes = list(self._in_heap.keys())
        self._heap = []
        self._in_heap = {}
        for node_id in nodes:
            self._push(node_id, self._calc_key(node_id))

    def update_vertex(self, node_id: int) -> None:
        if node_id != self.goal:
            best = self.inf
            for succ, edge_id in self.em.adj.get(node_id, []):
                candidate = self._get_edge_cost(edge_id) + self.g[succ]
                if candidate < best:
                    best = candidate
            self.rhs[node_id] = best

        if node_id in self._in_heap:
            del self._in_heap[node_id]

        if not math.isclose(self.g[node_id], self.rhs[node_id], rel_tol=0.0, abs_tol=1e-12):
            self._push(node_id, self._calc_key(node_id))

    def compute_shortest_path(self, current_start: int) -> bool:
        previous_start = self.current_start
        self.current_start = current_start
        if previous_start != current_start and self._in_heap:
            self._rekey_open_set()

        self.nodes_expanded = 0
        self.expanded_nodes_list = []

        while (
            self._top_key() < self._calc_key(self.current_start)
            or not math.isclose(
                self.g[self.current_start],
                self.rhs[self.current_start],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            node_id, old_key = self._pop()
            if node_id is None or old_key is None:
                break

            new_key = self._calc_key(node_id)
            if old_key < new_key:
                self._push(node_id, new_key)
                continue

            self.nodes_expanded += 1
            self.expanded_nodes_list.append(node_id)

            if self.g[node_id] > self.rhs[node_id]:
                self.g[node_id] = self.rhs[node_id]
                for pred, _ in self.em.rev_adj.get(node_id, []):
                    self.update_vertex(pred)
            else:
                self.g[node_id] = self.inf
                self.update_vertex(node_id)
                for pred, _ in self.em.rev_adj.get(node_id, []):
                    self.update_vertex(pred)

        return np.isfinite(self.g[self.current_start])

    def update_edge_cost(self, edge_id: int, new_cost: float) -> None:
        self._edge_cost_override[edge_id] = new_cost
        node_from, _ = self.em.edges[edge_id]
        self.update_vertex(node_from)

    def extract_path(self, current_start: int) -> List[int]:
        if not np.isfinite(self.g[current_start]):
            return []

        path = [current_start]
        current = current_start
        seen = {current}
        while current != self.goal:
            best_succ = None
            best_total = self.inf
            for succ, edge_id in self.em.adj.get(current, []):
                total = self._get_edge_cost(edge_id) + self.g[succ]
                if total < best_total and succ not in seen:
                    best_total = total
                    best_succ = succ
            if best_succ is None:
                return []
            path.append(best_succ)
            current = best_succ
            seen.add(current)
        return path

    def path_length_m(self, path: List[int]) -> float:
        total = 0.0
        for idx in range(len(path) - 1):
            n1 = self.em.nodes[path[idx]]
            n2 = self.em.nodes[path[idx + 1]]
            total += math.dist(n1, n2)
        return total

    def path_total_cost(self, current_start: int) -> float:
        return float(self.g[current_start]) if np.isfinite(self.g[current_start]) else self.inf


class AStarPlanner:
    """Conventional A* full replanning baseline."""

    def __init__(self, energy_map: EnergyMap):
        self.em = energy_map
        self.n_nodes = len(energy_map.nodes)

    def _heuristic(self, node_id: int, goal: int) -> float:
        node = self.em.nodes[node_id]
        goal_node = self.em.nodes[goal]
        dx = float(goal_node[0] - node[0])
        dy = float(goal_node[1] - node[1])
        dz = float(goal_node[2] - node[2])
        d3d = math.sqrt(dx * dx + dy * dy + dz * dz)
        return config.ALPHA * (d3d / config.CRUISE_SPEED)

    def plan(
        self,
        start: int,
        goal: int,
        blocked_edges: set[int] | None = None,
    ) -> Tuple[bool, List[int], int, float]:
        blocked_edges = blocked_edges or set()
        inf = float("inf")
        dist = np.full(self.n_nodes, inf, dtype=float)
        prev = np.full(self.n_nodes, -1, dtype=int)
        closed = np.zeros(self.n_nodes, dtype=bool)

        dist[start] = 0.0
        heap: list[Tuple[float, float, int]] = [(self._heuristic(start, goal), 0.0, start)]
        expanded = 0

        t0 = time.perf_counter()
        while heap:
            _, g_cost, node_id = heapq.heappop(heap)
            if g_cost > dist[node_id] + 1e-9 or closed[node_id]:
                continue
            closed[node_id] = True
            expanded += 1

            if node_id == goal:
                break

            for succ, edge_id in self.em.adj.get(node_id, []):
                if edge_id in blocked_edges:
                    continue
                next_cost = g_cost + self.em.get_edge_cost(edge_id)
                if next_cost + 1e-9 < dist[succ]:
                    dist[succ] = next_cost
                    prev[succ] = node_id
                    heapq.heappush(heap, (next_cost + self._heuristic(succ, goal), next_cost, succ))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if not np.isfinite(dist[goal]):
            return False, [], expanded, elapsed_ms

        path = [goal]
        current = goal
        seen = {goal}
        while current != start:
            predecessor = int(prev[current])
            if predecessor < 0 or predecessor in seen:
                return False, [], expanded, elapsed_ms
            path.append(predecessor)
            current = predecessor
            seen.add(current)
        path.reverse()
        return True, path, expanded, elapsed_ms
