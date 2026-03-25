"""
DEM 数据加载模块
负责加载华山地形DEM数据，支持缓存机制
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import tifffile
from pyproj import Transformer

import config

# ===== 文件路径常量 =====
TIF_FILE = "AP_19438_FBD_F0680_RT1.dem.tif"
CACHE_FILE = "Z_crop.npy"
CACHE_GEO = "Z_crop_geo.npz"
CACHE_META = "Z_crop_meta.json"

EPSG_SRC = "EPSG:32649"    # WGS84 / UTM zone 49N
EPSG_WGS84 = "EPSG:4326"

# 裁剪参数
DEFAULT_CENTER_LON = 110.0798
DEFAULT_CENTER_LAT = 34.4829
DEFAULT_CROP_SIZE_M = 10_000.0


def read_tiff_with_georef(tif_path: Path) -> Tuple[np.ndarray, float, float, float, float]:
    """读取GeoTIFF文件，返回 (dem, x0, y0, sx, sy)"""
    with tifffile.TiffFile(tif_path) as tif:
        page = tif.pages[0]
        dem = page.asarray().astype(float)
        scale = page.tags["ModelPixelScaleTag"].value
        tie = page.tags["ModelTiepointTag"].value
    sx = float(scale[0])
    sy = float(scale[1])
    x0 = float(tie[3])   # 左上角 x
    y0 = float(tie[4])   # 左上角 y
    return dem, x0, y0, sx, sy


def pixel_to_xy(row: float, col: float, x0: float, y0: float,
                sx: float, sy: float) -> Tuple[float, float]:
    """像素坐标 → UTM坐标"""
    x = x0 + col * sx
    y = y0 - row * sy
    return x, y


def xy_to_pixel(x: float, y: float, x0: float, y0: float,
                sx: float, sy: float) -> Tuple[int, int]:
    """UTM坐标 → 像素坐标"""
    col = int(round((x - x0) / sx))
    row = int(round((y0 - y) / sy))
    return row, col


def bounded_crop_window(row_center: int, col_center: int, half: int,
                        n_rows: int, n_cols: int) -> Tuple[int, int, int, int]:
    """计算有界裁剪窗口"""
    row_min = row_center - half
    row_max = row_center + half
    col_min = col_center - half
    col_max = col_center + half

    if row_min < 0:
        row_max -= row_min
        row_min = 0
    if col_min < 0:
        col_max -= col_min
        col_min = 0
    if row_max > n_rows:
        shift = row_max - n_rows
        row_min -= shift
        row_max = n_rows
    if col_max > n_cols:
        shift = col_max - n_cols
        col_min -= shift
        col_max = n_cols

    row_min = max(0, row_min)
    col_min = max(0, col_min)
    row_max = min(n_rows, row_max)
    col_max = min(n_cols, col_max)
    return row_min, row_max, col_min, col_max


def build_lonlat_grids(row_min: int, row_max: int, col_min: int, col_max: int,
                       x0: float, y0: float, sx: float, sy: float
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """构建经纬度网格"""
    rows = row_max - row_min
    cols = col_max - col_min
    rr = np.arange(row_min, row_max, dtype=float)
    cc = np.arange(col_min, col_max, dtype=float)
    rr2, cc2 = np.meshgrid(rr, cc, indexing="ij")
    x2 = x0 + cc2 * sx
    y2 = y0 - rr2 * sy

    tf_to_wgs = Transformer.from_crs(EPSG_SRC, EPSG_WGS84, always_xy=True)
    lon_flat, lat_flat = tf_to_wgs.transform(x2.ravel(), y2.ravel())
    lon_grid = lon_flat.reshape(rows, cols)
    lat_grid = lat_flat.reshape(rows, cols)
    return lon_grid, lat_grid


def cache_matches(meta_path: Path, center_lon: float, center_lat: float,
                  crop_size_m: float) -> bool:
    """检查缓存是否匹配当前参数"""
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        abs(float(meta.get("center_lon", 0.0)) - center_lon) < 1e-9
        and abs(float(meta.get("center_lat", 0.0)) - center_lat) < 1e-9
        and abs(float(meta.get("crop_size_m", 0.0)) - crop_size_m) < 1e-9
        and meta.get("row0_orientation", "") == "north"
    )


def nearest_rc_from_lonlat(lon_grid: np.ndarray, lat_grid: np.ndarray,
                           lon: float, lat: float) -> Tuple[int, int]:
    """在经纬度网格中找最近的栅格坐标"""
    d2 = (lon_grid - lon) ** 2 + (lat_grid - lat) ** 2
    idx = int(np.argmin(d2))
    r, c = np.unravel_index(idx, lon_grid.shape)
    return int(r), int(c)


def load_dem(center_lon: float = DEFAULT_CENTER_LON,
             center_lat: float = DEFAULT_CENTER_LAT,
             crop_size_m: float = DEFAULT_CROP_SIZE_M,
             force_recrop: bool = False
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    加载华山DEM数据，支持缓存机制。

    返回:
        (z_crop, lon_grid, lat_grid)
        z_crop: 高程矩阵 (rows x cols)
        lon_grid: 经度网格 (rows x cols)
        lat_grid: 纬度网格 (rows x cols)
    """
    tif_path = Path(TIF_FILE)
    use_cache = (
        Path(CACHE_FILE).exists()
        and Path(CACHE_GEO).exists()
        and cache_matches(Path(CACHE_META), center_lon, center_lat, crop_size_m)
        and (not force_recrop)
    )

    if use_cache:
        print("[缓存] 使用现有DEM裁剪缓存")
        z_crop = np.asarray(np.load(CACHE_FILE), dtype=float)
        geo = np.load(CACHE_GEO)
        lon_grid = np.asarray(geo["lon_grid"], dtype=float)
        lat_grid = np.asarray(geo["lat_grid"], dtype=float)
    else:
        if not tif_path.exists():
            raise FileNotFoundError(f"缺少DEM文件: {tif_path.resolve()}")

        print("[裁剪] 读取源DEM并重新裁剪...")
        dem, x0, y0, sx, sy = read_tiff_with_georef(tif_path)
        dem[dem < -9000] = np.nan
        n_rows, n_cols = dem.shape
        half = int(round((crop_size_m / 2.0) / config.DEM_RES))

        tf_to_utm = Transformer.from_crs(EPSG_WGS84, EPSG_SRC, always_xy=True)
        x_new, y_new = tf_to_utm.transform(center_lon, center_lat)
        center_row, center_col = xy_to_pixel(x_new, y_new, x0, y0, sx, sy)

        row_min, row_max, col_min, col_max = bounded_crop_window(
            center_row, center_col, half, n_rows, n_cols
        )
        z_crop = dem[row_min:row_max, col_min:col_max]
        lon_grid, lat_grid = build_lonlat_grids(
            row_min, row_max, col_min, col_max, x0, y0, sx, sy
        )

        np.save(CACHE_FILE, z_crop.astype(np.float32))
        np.savez(CACHE_GEO, lon_grid=lon_grid.astype(np.float64),
                 lat_grid=lat_grid.astype(np.float64))
        meta = {
            "center_lon": center_lon,
            "center_lat": center_lat,
            "crop_size_m": crop_size_m,
            "row_min": int(row_min),
            "row_max": int(row_max),
            "col_min": int(col_min),
            "col_max": int(col_max),
            "row0_orientation": "north",
        }
        Path(CACHE_META).write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print("[裁剪] 缓存已保存")

    rows, cols = z_crop.shape
    print(f"[DEM] 形状: {rows} x {cols}")
    print(f"[DEM] 高程范围: {np.nanmin(z_crop):.1f}m ~ {np.nanmax(z_crop):.1f}m")
    return z_crop, lon_grid, lat_grid
