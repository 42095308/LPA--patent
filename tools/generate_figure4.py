"""生成图4方案B结果图。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
from matplotlib.ticker import ScalarFormatter

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from dem_loader import load_dem
from ems import EMSController, PassiveEMS
from energy_map import EnergyMap


DEFAULT_RESULT_FILE = ROOT_DIR / "outputs" / "simulation_results.json"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs"
BACKGROUND_STRIDE = 3


def configure_matplotlib() -> None:
    """配置绘图风格。"""
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "#fafafa"
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.edgecolor"] = "#555555"
    plt.rcParams["grid.color"] = "#d0d0d0"
    plt.rcParams["grid.alpha"] = 0.35


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="生成图4方案B结果图。")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULT_FILE, help="仿真结果 JSON 路径。")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="图片输出目录。")
    parser.add_argument("--dpi", type=int, default=300, help="输出图片 DPI。")
    return parser.parse_args()


def load_results(results_path: Path) -> Dict[str, object]:
    """读取仿真结果。"""
    return json.loads(results_path.read_text(encoding="utf-8"))


def build_plot_map() -> EnergyMap:
    """构建用于路径映射与事件回放的能量地图。"""
    z_dem, lon_grid, lat_grid = load_dem()
    energy_map = EnergyMap(z_dem, lon_grid, lat_grid, soh=config.INITIAL_SOH)
    energy_map.build_graph()
    return energy_map


def bilinear_sample(grid: np.ndarray, row_f: float, col_f: float) -> float:
    """对二维网格做双线性采样。"""
    rows, cols = grid.shape
    row_f = float(np.clip(row_f, 0.0, rows - 1.0))
    col_f = float(np.clip(col_f, 0.0, cols - 1.0))
    row0 = int(math.floor(row_f))
    col0 = int(math.floor(col_f))
    row1 = min(row0 + 1, rows - 1)
    col1 = min(col0 + 1, cols - 1)
    dr = row_f - row0
    dc = col_f - col0
    v00 = float(grid[row0, col0])
    v01 = float(grid[row0, col1])
    v10 = float(grid[row1, col0])
    v11 = float(grid[row1, col1])
    return (
        v00 * (1.0 - dr) * (1.0 - dc)
        + v01 * (1.0 - dr) * dc
        + v10 * dr * (1.0 - dc)
        + v11 * dr * dc
    )


def xy_to_lonlat(energy_map: EnergyMap, x_m: float, y_m: float) -> Tuple[float, float]:
    """将局部米制坐标映射回经纬度。"""
    rows, cols = energy_map.lon_grid.shape
    col_f = x_m / config.DEM_RES
    row_f = (rows - 1) - y_m / config.DEM_RES
    lon = bilinear_sample(energy_map.lon_grid, row_f, col_f)
    lat = bilinear_sample(energy_map.lat_grid, row_f, col_f)
    return lon, lat


def node_path_to_lonlat(energy_map: EnergyMap, node_ids: List[int]) -> np.ndarray:
    """将节点路径转换为经纬度折线。"""
    lonlat: List[Tuple[float, float]] = []
    for node_id in node_ids:
        x_m, y_m, _ = energy_map.position_from_node(int(node_id))
        lonlat.append(xy_to_lonlat(energy_map, x_m, y_m))
    if not lonlat:
        return np.empty((0, 2), dtype=float)
    return np.asarray(lonlat, dtype=float)


def iter_trigger_decisions(
    chain_result: Dict[str, object],
    trigger_id: str,
) -> Iterable[Tuple[Dict[str, object], Dict[str, object]]]:
    """遍历指定触发器的事件记录。"""
    for event in chain_result.get("events", []):
        for decision in event.get("decisions", []):
            if decision.get("trigger_id") == trigger_id:
                yield event, decision
                break


def collect_t2_regions(results: Dict[str, object]) -> List[Dict[str, object]]:
    """收集 T2 风场扰动区域。"""
    regions: List[Dict[str, object]] = []
    for chain_name, chain_result in results["chains"].items():
        for event, decision in iter_trigger_decisions(chain_result, "T2"):
            metadata = decision.get("metadata", {})
            center_xy = metadata.get("center_xy")
            radius_m = metadata.get("radius_m")
            if not center_xy or radius_m is None:
                continue
            current_xyz = event.get("current_xyz", [0.0, 0.0, 0.0])
            regions.append(
                {
                    "chain": chain_name,
                    "event_id": int(event.get("event_id", len(regions) + 1)),
                    "triggered": bool(decision.get("triggered", False)),
                    "center_xy": (float(center_xy[0]), float(center_xy[1])),
                    "radius_m": float(radius_m),
                    "wind_speed_mps": float(metadata.get("wind_speed_mps", config.WIND_SHEAR)),
                    "trigger_point_xy": (float(current_xyz[0]), float(current_xyz[1])),
                }
            )
    return regions


def replay_disturbance_map(base_map: EnergyMap, regions: List[Dict[str, object]]) -> EnergyMap:
    """在绘图地图上回放 T2 扰动。"""
    disturbed_map = base_map.clone_dynamic_state()
    for region in regions:
        if not region["triggered"]:
            continue
        disturbed_map.update_wind_field(
            region["center_xy"],
            region["radius_m"],
            region["wind_speed_mps"],
        )
    return disturbed_map


def region_circle_to_lonlat_polygon(
    energy_map: EnergyMap,
    center_xy: Tuple[float, float],
    radius_m: float,
    n_samples: int = 96,
) -> np.ndarray:
    """将米制圆形事件区转换为经纬度多边形。"""
    center_x, center_y = center_xy
    vertices: List[Tuple[float, float]] = []
    for angle in np.linspace(0.0, 2.0 * math.pi, n_samples, endpoint=False):
        x_m = center_x + radius_m * math.cos(float(angle))
        y_m = center_y + radius_m * math.sin(float(angle))
        vertices.append(xy_to_lonlat(energy_map, x_m, y_m))
    return np.asarray(vertices, dtype=float)


def run_ems_logs(results: Dict[str, object]) -> Dict[str, List[Dict[str, float]]]:
    """根据导出的功率轨迹重放 EMS 日志。"""
    logs: Dict[str, List[Dict[str, float]]] = {}
    for chain_name, chain_result in results["chains"].items():
        profile = chain_result.get("power_profile", {})
        time_arr = np.asarray(profile.get("time_s", []), dtype=float)
        power_arr = np.asarray(profile.get("power_w", []), dtype=float)
        if chain_name == "proposed":
            controller = EMSController()
            controller.simulate(time_arr, power_arr, chain_result.get("structured_messages", []))
            logs[chain_name] = list(controller.log)
        else:
            controller = PassiveEMS()
            controller.simulate(time_arr, power_arr)
            logs[chain_name] = list(controller.log)
    return logs


def improvement_percent(proposed_value: float, traditional_value: float, higher_is_better: bool) -> float:
    """计算相对传统方法的改进率。"""
    if higher_is_better:
        if abs(traditional_value) <= 1e-12:
            return 0.0
        return (proposed_value - traditional_value) / traditional_value * 100.0
    if abs(traditional_value) <= 1e-12:
        return 0.0
    return (traditional_value - proposed_value) / traditional_value * 100.0


def build_metric_items(results: Dict[str, object]) -> List[Dict[str, object]]:
    """构建关键指标列表。"""
    proposed = results["chains"]["proposed"]
    traditional = results["chains"]["traditional"]
    comparison = results["comparison"]
    items: List[Dict[str, object]] = [
        {
            "label": "总氢耗",
            "unit": "g",
            "proposed": float(proposed["h2_total_g"]),
            "traditional": float(traditional["h2_total_g"]),
            "higher_is_better": False,
        },
        {
            "label": "电池应力",
            "unit": "As",
            "proposed": float(proposed["battery_stress_index_as"]),
            "traditional": float(traditional["battery_stress_index_as"]),
            "higher_is_better": False,
        },
        {
            "label": "燃料电池应力",
            "unit": "-",
            "proposed": float(proposed["fc_stress_index"]),
            "traditional": float(traditional["fc_stress_index"]),
            "higher_is_better": False,
        },
        {
            "label": "最大功率跃迁",
            "unit": "W",
            "proposed": float(proposed["max_dp_req_w"]),
            "traditional": float(traditional["max_dp_req_w"]),
            "higher_is_better": False,
        },
        {
            "label": "事件级规划速度比",
            "unit": "x",
            "proposed": float(comparison["event_planning_speedup_ratio"]),
            "traditional": 1.0,
            "higher_is_better": True,
        },
    ]
    for item in items:
        item["improvement_pct"] = improvement_percent(
            float(item["proposed"]),
            float(item["traditional"]),
            bool(item["higher_is_better"]),
        )
    return items


def format_metric_value(value: float, unit: str) -> str:
    """格式化指标数值。"""
    if unit == "x":
        return f"{value:.3f}{unit}"
    if unit == "-":
        return f"{value:.2f}"
    if abs(value) >= 100.0:
        return f"{value:.2f} {unit}"
    return f"{value:.3f} {unit}"


def plot_risk_background(ax: plt.Axes, disturbed_map: EnergyMap) -> None:
    """绘制经纬度背景地形和风险区。"""
    lon_grid = disturbed_map.lon_grid[::BACKGROUND_STRIDE, ::BACKGROUND_STRIDE]
    lat_grid = disturbed_map.lat_grid[::BACKGROUND_STRIDE, ::BACKGROUND_STRIDE]
    terrain = disturbed_map.z_dem[::BACKGROUND_STRIDE, ::BACKGROUND_STRIDE]
    wind_mask = disturbed_map.wind_field[::BACKGROUND_STRIDE, ::BACKGROUND_STRIDE] > (config.WIND_NORMAL + 1e-9)

    ax.pcolormesh(
        lon_grid,
        lat_grid,
        terrain,
        shading="nearest",
        cmap="Greys",
        alpha=0.28,
    )

    risk_overlay = np.ma.masked_where(~wind_mask, wind_mask.astype(float))
    ax.pcolormesh(
        lon_grid,
        lat_grid,
        risk_overlay,
        shading="nearest",
        cmap="Reds",
        alpha=0.18,
    )


def set_lonlat_viewport(
    ax: plt.Axes,
    proposed_path: np.ndarray,
    traditional_path: np.ndarray,
    region_polygons: List[np.ndarray],
) -> None:
    """根据路径和事件区自动聚焦视野。"""
    lon_values: List[float] = []
    lat_values: List[float] = []
    if len(proposed_path):
        lon_values.extend(proposed_path[:, 0].tolist())
        lat_values.extend(proposed_path[:, 1].tolist())
    if len(traditional_path):
        lon_values.extend(traditional_path[:, 0].tolist())
        lat_values.extend(traditional_path[:, 1].tolist())
    for polygon in region_polygons:
        if len(polygon):
            lon_values.extend(polygon[:, 0].tolist())
            lat_values.extend(polygon[:, 1].tolist())
    if not lon_values or not lat_values:
        return

    lon_min = min(lon_values)
    lon_max = max(lon_values)
    lat_min = min(lat_values)
    lat_max = max(lat_values)
    lon_span = max(lon_max - lon_min, 1e-5)
    lat_span = max(lat_max - lat_min, 1e-5)
    lon_margin = max(lon_span * 0.10, 0.00018)
    lat_margin = max(lat_span * 0.10, 0.00018)

    ax.set_xlim(lon_min - lon_margin, lon_max + lon_margin)
    ax.set_ylim(lat_min - lat_margin, lat_max + lat_margin)

    ax.set_aspect("auto")


def plot_path_panel(ax: plt.Axes, energy_map: EnergyMap, results: Dict[str, object]) -> None:
    """绘制路径对比图。"""
    proposed_path = node_path_to_lonlat(energy_map, results["chains"]["proposed"]["final_path_nodes"])
    traditional_path = node_path_to_lonlat(energy_map, results["chains"]["traditional"]["final_path_nodes"])
    regions = collect_t2_regions(results)
    disturbed_map = replay_disturbance_map(energy_map, regions)
    region_polygons: List[np.ndarray] = []

    plot_risk_background(ax, disturbed_map)

    for region in regions:
        if not region["triggered"]:
            continue
        polygon = region_circle_to_lonlat_polygon(
            energy_map,
            region["center_xy"],
            region["radius_m"],
        )
        region_polygons.append(polygon)
        ax.add_patch(
            Polygon(
                polygon,
                closed=True,
                edgecolor="#d14b4b",
                facecolor="#d14b4b",
                linewidth=1.5,
                linestyle="--",
                alpha=0.12,
            )
        )

    if len(proposed_path):
        ax.plot(
            proposed_path[:, 0],
            proposed_path[:, 1],
            color="#0b7a75",
            linewidth=2.6,
            label="本发明方法路径",
        )
        ax.scatter(proposed_path[0, 0], proposed_path[0, 1], color="#0b7a75", s=52, marker="o", zorder=5)
        ax.scatter(proposed_path[-1, 0], proposed_path[-1, 1], color="#0b7a75", s=64, marker="*", zorder=5)
        ax.text(proposed_path[0, 0], proposed_path[0, 1], " 起点", color="#0b7a75", fontsize=9, va="bottom")
        ax.text(proposed_path[-1, 0], proposed_path[-1, 1], " 终点", color="#0b7a75", fontsize=9, va="bottom")

    if len(traditional_path):
        ax.plot(
            traditional_path[:, 0],
            traditional_path[:, 1],
            color="#d97706",
            linewidth=2.2,
            linestyle="--",
            label="传统解耦方法路径",
        )
        ax.scatter(traditional_path[0, 0], traditional_path[0, 1], color="#d97706", s=42, marker="s", zorder=5)
        ax.scatter(traditional_path[-1, 0], traditional_path[-1, 1], color="#d97706", s=54, marker="X", zorder=5)

    for event in results["chains"]["proposed"].get("events", []):
        lon, lat = xy_to_lonlat(energy_map, float(event["current_xyz"][0]), float(event["current_xyz"][1]))
        ax.scatter(lon, lat, color="#0f766e", s=36, marker="o", edgecolors="white", linewidths=0.8, zorder=6)
        ax.text(lon, lat, f" E{event['event_id']}", color="#0f766e", fontsize=9, weight="bold", va="bottom")
    for event in results["chains"]["traditional"].get("events", []):
        lon, lat = xy_to_lonlat(energy_map, float(event["current_xyz"][0]), float(event["current_xyz"][1]))
        ax.scatter(lon, lat, color="#b45309", s=34, marker="x", linewidths=1.2, zorder=6)

    set_lonlat_viewport(ax, proposed_path, traditional_path, region_polygons)
    ax.set_box_aspect(0.42)

    ax.set_xlabel("经度 / °\n(a) 动态事件场景路径对比", labelpad=10, fontweight="bold")
    ax.set_ylabel("纬度 / °")
    ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.grid(True, linestyle=":")

    legend_handles = [
        Line2D([0], [0], color="#0b7a75", lw=2.6, label="本发明方法路径"),
        Line2D([0], [0], color="#d97706", lw=2.2, linestyle="--", label="传统解耦方法路径"),
        Patch(facecolor="#d14b4b", edgecolor="#d14b4b", alpha=0.12, label="风场扰动/高风险区域"),
        Line2D([0], [0], color="#0f766e", marker="o", lw=0, markersize=6, label="本发明事件触发点"),
        Line2D([0], [0], color="#b45309", marker="x", lw=0, markersize=6, label="传统方法事件触发点"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True, fontsize=9)


def plot_voltage_series_panel(ax: plt.Axes, logs: Dict[str, List[Dict[str, float]]], results: Dict[str, object]) -> None:
    """绘制母线电压时序图。"""
    for chain_name, color, linestyle, label in [
        ("proposed", "#0b7a75", "-", "本发明方法"),
        ("traditional", "#d97706", "--", "传统解耦方法"),
    ]:
        chain_log = logs.get(chain_name, [])
        if not chain_log:
            continue
        time_arr = np.asarray([item["time_s"] for item in chain_log], dtype=float)
        voltage_arr = np.asarray([item["v_bus_v"] for item in chain_log], dtype=float)
        ax.plot(time_arr, voltage_arr, color=color, linewidth=2.0, linestyle=linestyle, label=label)

    for event in results["chains"]["proposed"].get("events", []):
        ax.axvline(float(event["trigger_time_s"]), color="#999999", linestyle=":", linewidth=1.0, alpha=0.8)

    ax.set_xlabel("任务时间 / s\n(b) 母线电压时序对比", labelpad=10, fontweight="bold")
    ax.set_ylabel("母线电压 / V")
    ax.grid(True, linestyle=":")
    ax.legend(loc="best", frameon=True)


def plot_metric_bar_panel(ax: plt.Axes, metric_items: List[Dict[str, object]]) -> None:
    """绘制关键指标柱状图。"""
    labels = [item["label"] for item in metric_items]
    values = [float(item["improvement_pct"]) for item in metric_items]
    y_pos = np.arange(len(labels))
    colors = ["#1f8a70" if value >= 0.0 else "#c2410c" for value in values]

    bars = ax.barh(y_pos, values, color=colors, alpha=0.88)
    ax.axvline(0.0, color="#444444", linewidth=1.0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("相对传统方法改进率 / %\n(c) 关键指标汇总", labelpad=10, fontweight="bold")
    ax.grid(True, axis="x", linestyle=":")

    if values:
        min_value = min(values)
        max_value = max(values)
    else:
        min_value = -1.0
        max_value = 1.0
    left_limit = min(-5.0, min_value - 2.0)
    right_limit = max(5.0, max_value + 9.0)
    ax.set_xlim(left_limit, right_limit)

    for bar, item, value in zip(bars, metric_items, values):
        proposed_text = format_metric_value(float(item["proposed"]), str(item["unit"]))
        traditional_text = format_metric_value(float(item["traditional"]), str(item["unit"]))
        text_x = value + (0.7 if value >= 0.0 else -0.7)
        align = "left" if value >= 0.0 else "right"
        ax.text(
            text_x,
            bar.get_y() + bar.get_height() / 2.0,
            f"{proposed_text} / {traditional_text}",
            va="center",
            ha=align,
            fontsize=8.3,
            color="#222222",
        )


def create_figure(results: Dict[str, object], energy_map: EnergyMap, output_path: Path, dpi: int) -> None:
    """生成图4方案B。"""
    fig = plt.figure(figsize=(15.8, 10.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0])
    ax_path = fig.add_subplot(grid[0, :])
    ax_voltage = fig.add_subplot(grid[1, 0])
    ax_metrics = fig.add_subplot(grid[1, 1])

    plot_path_panel(ax_path, energy_map, results)
    logs = run_ems_logs(results)
    plot_voltage_series_panel(ax_voltage, logs, results)
    plot_metric_bar_panel(ax_metrics, build_metric_items(results))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """主入口。"""
    args = parse_args()
    configure_matplotlib()
    results = load_results(args.results)
    energy_map = build_plot_map()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    create_figure(results, energy_map, output_dir / "figure4_scheme_b.png", args.dpi)
    create_figure(results, energy_map, output_dir / "figure4.png", args.dpi)

    print("图4方案B结果图已生成：")
    print(f" - {(output_dir / 'figure4_scheme_b.png').resolve()}")
    print(f" - {(output_dir / 'figure4.png').resolve()}")


if __name__ == "__main__":
    main()
