"""
步骤二/三：LPA* 路径规划器
包含 LPA* 增量路径规划和传统 A* 全局规划（对照组）
"""

from __future__ import annotations

import heapq
import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

import config
from energy_map import EnergyMap


class LPAStar:
    """
    LPA*（Lifelong Planning A*）增量路径规划器

    核心数据结构：
        g[s]   : 当前从起点到 s 的已知最短路径代价
        rhs[s] : 一步超前值，rhs[s] = min_{pred} [g(pred) + c(pred, s)]
        U      : 优先队列，存放不一致节点

    当 g(s) = rhs(s) 时节点局部一致，否则为局部不一致
    """

    def __init__(self, energy_map: EnergyMap, start: int, goal: int):
        self.em = energy_map
        self.start = start
        self.goal = goal
        self.n_nodes = len(energy_map.nodes)
        self.INF = float('inf')

        # g 和 rhs 初始化为 ∞
        self.g = np.full(self.n_nodes, self.INF)
        self.rhs = np.full(self.n_nodes, self.INF)

        # 起点 rhs = 0
        self.rhs[start] = 0.0

        # 优先队列
        self._counter = 0
        self._heap: List[Tuple[float, float, int, int]] = []
        self._in_heap: Dict[int, Tuple[float, float]] = {}

        self._push(start, self._calc_key(start))

        # 统计
        self.nodes_expanded = 0
        self.expanded_nodes_list: List[int] = []

        # 边代价缓存（支持动态修改）
        self._edge_cost_override: Dict[int, float] = {}

    def _get_edge_cost(self, edge_id: int) -> float:
        """获取边代价（支持动态修改）"""
        if edge_id in self._edge_cost_override:
            return self._edge_cost_override[edge_id]
        return self.em.get_edge_cost(edge_id)

    def _calc_key(self, s: int) -> Tuple[float, float]:
        """
        LPA* 优先级键：
            k1 = min(g(s), rhs(s)) + h(s)
            k2 = min(g(s), rhs(s))
        """
        base = min(self.g[s], self.rhs[s])
        return (base + self._heuristic(s), base)

    def _heuristic(self, s: int) -> float:
        """
        可容许启发式函数：基于直线距离的最低飞行代价下界
        """
        n_s = self.em.nodes[s]
        n_g = self.em.nodes[self.goal]
        dx = n_g[0] - n_s[0]
        dy = n_g[1] - n_s[1]
        dz = n_g[2] - n_s[2]
        d3d = math.sqrt(dx * dx + dy * dy + dz * dz)

        # 空间距离下界
        h_spatial = config.ALPHA * d3d
        return h_spatial

    def _push(self, node: int, key: Tuple[float, float]) -> None:
        self._counter += 1
        heapq.heappush(self._heap, (key[0], key[1], self._counter, node))
        self._in_heap[node] = key

    def _pop(self) -> Tuple[Optional[int], Optional[Tuple[float, float]]]:
        while self._heap:
            k1, k2, _, node = heapq.heappop(self._heap)
            cur_key = self._in_heap.get(node)
            if cur_key is not None and cur_key == (k1, k2):
                del self._in_heap[node]
                return node, (k1, k2)
        return None, None

    def _top_key(self) -> Tuple[float, float]:
        while self._heap:
            k1, k2, _, node = self._heap[0]
            cur_key = self._in_heap.get(node)
            if cur_key is not None and cur_key == (k1, k2):
                return (k1, k2)
            heapq.heappop(self._heap)
        return (self.INF, self.INF)

    def update_vertex(self, u: int) -> None:
        """重新计算 u 的 rhs 值"""
        if u != self.start:
            best = self.INF
            for v, eid in self.em.adj.get(u, []):
                # 反向：从前驱到u的代价
                # 注意邻接表存的是出边，需要考虑入边
                pass

            # 遍历所有可能的前驱节点
            best = self.INF
            for pred_node in range(self.n_nodes):
                for neighbor, eid in self.em.adj.get(pred_node, []):
                    if neighbor == u:
                        c = self._get_edge_cost(eid)
                        if self.g[pred_node] + c < best:
                            best = self.g[pred_node] + c
            self.rhs[u] = best

        # 维护优先队列
        if u in self._in_heap:
            del self._in_heap[u]

        if self.g[u] != self.rhs[u]:
            self._push(u, self._calc_key(u))

    def _update_vertex_fast(self, u: int) -> None:
        """
        快速更新 vertex（用反向邻接表）
        """
        if u != self.start:
            best = self.INF
            # 查找所有指向 u 的边
            for pred_node, neighbors in self.em.adj.items():
                for neighbor, eid in neighbors:
                    if neighbor == u:
                        c = self._get_edge_cost(eid)
                        val = self.g[pred_node] + c
                        if val < best:
                            best = val
            self.rhs[u] = best

        if u in self._in_heap:
            del self._in_heap[u]

        if self.g[u] != self.rhs[u]:
            self._push(u, self._calc_key(u))

    def _build_reverse_adj(self) -> None:
        """构建反向邻接表，加速 update_vertex"""
        self._rev_adj: Dict[int, List[Tuple[int, int]]] = {
            i: [] for i in range(self.n_nodes)
        }
        for node, neighbors in self.em.adj.items():
            for neighbor, eid in neighbors:
                self._rev_adj[neighbor].append((node, eid))

    def _update_vertex_with_rev(self, u: int) -> None:
        """使用反向邻接表的快速更新"""
        if u != self.start:
            best = self.INF
            for pred, eid in self._rev_adj.get(u, []):
                c = self._get_edge_cost(eid)
                val = self.g[pred] + c
                if val < best:
                    best = val
            self.rhs[u] = best

        if u in self._in_heap:
            del self._in_heap[u]

        if self.g[u] != self.rhs[u]:
            self._push(u, self._calc_key(u))

    def compute_shortest_path(self) -> bool:
        """
        核心搜索循环：
        持续处理优先队列中的不一致节点，直到终点一致且队列为空
        """
        # 确保反向邻接表存在
        if not hasattr(self, '_rev_adj'):
            self._build_reverse_adj()

        self.nodes_expanded = 0
        self.expanded_nodes_list = []

        while True:
            top_key = self._top_key()
            goal_key = self._calc_key(self.goal)

            # 终止条件
            if (top_key >= goal_key and
                    self.g[self.goal] == self.rhs[self.goal]):
                break
            if top_key[0] == self.INF:
                break

            u, k_old = self._pop()
            if u is None:
                break

            self.nodes_expanded += 1
            self.expanded_nodes_list.append(u)
            k_new = self._calc_key(u)

            if k_old < k_new:
                # key 过期，重新入队
                self._push(u, k_new)
            elif self.g[u] > self.rhs[u]:
                # 过一致：g 值可以降低
                self.g[u] = self.rhs[u]
                for neighbor, eid in self.em.adj.get(u, []):
                    self._update_vertex_with_rev(neighbor)
            else:
                # 欠一致：g 值需要提升
                self.g[u] = self.INF
                self._update_vertex_with_rev(u)
                for neighbor, eid in self.em.adj.get(u, []):
                    self._update_vertex_with_rev(neighbor)

        return self.g[self.goal] < self.INF

    def update_edge_cost(self, edge_id: int, new_cost: float) -> None:
        """
        动态修改边代价（事件触发时调用）
        只更新受影响节点的 rhs 值
        """
        self._edge_cost_override[edge_id] = new_cost
        u, v = self.em.edges[edge_id]
        if hasattr(self, '_rev_adj'):
            self._update_vertex_with_rev(v)
        else:
            self._update_vertex_fast(v)

    def extract_path(self) -> List[int]:
        """从 goal 向 start 回溯路径"""
        if self.g[self.goal] == self.INF:
            return []
        path = [self.goal]
        cur = self.goal
        seen = set()
        while cur != self.start:
            seen.add(cur)
            best_pred, best_cost = None, self.INF
            # 找最佳前驱
            for pred, eid in self._rev_adj.get(cur, []):
                c = self._get_edge_cost(eid)
                total = self.g[pred] + c
                if total < best_cost and pred not in seen:
                    best_cost = total
                    best_pred = pred
            if best_pred is None:
                break
            path.append(best_pred)
            cur = best_pred
        path.reverse()
        return path if (path and path[0] == self.start) else []

    def path_length_m(self, path: List[int]) -> float:
        """计算路径总长度（米）"""
        total = 0.0
        for k in range(len(path) - 1):
            n1 = self.em.nodes[path[k]]
            n2 = self.em.nodes[path[k + 1]]
            d = math.sqrt((n2[0] - n1[0]) ** 2 +
                          (n2[1] - n1[1]) ** 2 +
                          (n2[2] - n1[2]) ** 2)
            total += d
        return total

    def path_total_cost(self, path: List[int]) -> float:
        """计算路径总代价"""
        return float(self.g[self.goal]) if self.g[self.goal] < self.INF else float('inf')


class AStarPlanner:
    """
    传统 A* 全局路径规划器（对照组 — 静态A*规划）
    每次需要重规划时都从头开始搜索
    """

    def __init__(self, energy_map: EnergyMap):
        self.em = energy_map
        self.n_nodes = len(energy_map.nodes)

    def plan(self, start: int, goal: int,
             blocked_edges: set | None = None) -> Tuple[bool, List[int], int, float]:
        """
        A* 全局搜索

        返回:
            (成功, 路径, 遍历节点数, 耗时ms)
        """
        INF = float('inf')
        dist = np.full(self.n_nodes, INF, dtype=float)
        prev_node = np.full(self.n_nodes, -1, dtype=int)
        closed = np.zeros(self.n_nodes, dtype=bool)

        dist[start] = 0.0
        heap = [(self._heuristic(start, goal), 0.0, start)]
        expanded = 0

        if blocked_edges is None:
            blocked_edges = set()

        t0 = time.perf_counter()

        while heap:
            f, g, u = heapq.heappop(heap)
            if g > dist[u] + 1e-9:
                continue
            if closed[u]:
                continue
            closed[u] = True
            expanded += 1

            if u == goal:
                break

            for v, eid in self.em.adj.get(u, []):
                if eid in blocked_edges:
                    continue
                c = self.em.get_edge_cost(eid)
                ng = g + c
                if ng + 1e-9 < dist[v]:
                    dist[v] = ng
                    prev_node[v] = u
                    heapq.heappush(heap, (ng + self._heuristic(v, goal), ng, v))

        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000.0

        if not np.isfinite(dist[goal]):
            return False, [], expanded, elapsed_ms

        # 回溯路径
        path = [goal]
        cur = goal
        seen = set()
        while cur != start:
            seen.add(cur)
            p = int(prev_node[cur])
            if p < 0 or p in seen:
                return False, [], expanded, elapsed_ms
            path.append(p)
            cur = p
        path.reverse()
        return True, path, expanded, elapsed_ms

    def _heuristic(self, s: int, goal: int) -> float:
        """可容许启发式：直线距离 × α"""
        n_s = self.em.nodes[s]
        n_g = self.em.nodes[goal]
        dx = n_g[0] - n_s[0]
        dy = n_g[1] - n_s[1]
        dz = n_g[2] - n_s[2]
        d3d = math.sqrt(dx * dx + dy * dy + dz * dz)
        return config.ALPHA * d3d
