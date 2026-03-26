"""
全局参数配置
氢燃料电池与锂电池混合动力四旋翼无人机协同控制仿真
"""

import math
from pathlib import Path

# ===== Path config =====
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SAMPLES_DIR = PROJECT_ROOT / "samples"

DEM_FILE = RAW_DIR / "AP_19438_FBD_F0680_RT1.dem.tif"
DEM_CACHE_FILE = CACHE_DIR / "Z_crop.npy"
DEM_CACHE_GEO_FILE = CACHE_DIR / "Z_crop_geo.npz"
DEM_CACHE_META_FILE = CACHE_DIR / "Z_crop_meta.json"
SIM_RESULT_FILE = OUTPUTS_DIR / "simulation_results.json"

# ===== 无人机平台参数 =====
MASS = 8.5              # 整机质量 (kg)
ROTOR_RADIUS = 0.25     # 旋翼半径 (m)
ROTOR_AREA = math.pi * ROTOR_RADIUS ** 2  # 旋翼盘面积 ≈ 0.196 m²
GRAVITY = 9.8           # 重力加速度 (m/s²)

# ===== 燃料电池参数 =====
FC_RATED_POWER = 3000.0     # 额定功率 (W)
FC_TAU = 5.0                # 功率响应时间 (s)
FC_DP_DT_MAX = 600.0        # 额定最大功率爬升率 (W/s)
FC_DP_MAX_STEP = 900.0      # 额定最大功率爬升量 (W)
FC_RAMP_LIMIT = 300.0       # EMS预调功率爬升限制 (W/s)

# ===== 锂电池参数 =====
BAT_CAPACITY_AH = 22.0     # 标称容量 (Ah)
BAT_VOLTAGE = 44.4          # 标称电压 (V)
V_BUS = 48.0                # 直流母线电压 (V)
BAT_MIN_CURRENT = 2.0       # 最小保护值 (A)

# ===== EMS对比仿真统一阈值 =====
# 两种方法必须使用相同阈值，否则对比结果无意义
BAT_SPIKE_THRESHOLD = 20.0  # 电池电流冲击阈值 (A)，超过即计为一次冲击
#   物理依据：20A × 48V = 960W，约为额定需求功率的 40%，
#   属于超调性补偿，会加速电极极化和析锂风险。
BAT_DEGRAD_THRESHOLD = 15.0 # 退化累计起始阈值 (A)，超过部分才计入退化指标

# ===== 巡航参数 =====
CRUISE_SPEED = 15.0         # 巡航速度 (m/s)

# ===== 三维栅格参数 =====
GRID_H_RES = 25.0           # 水平分辨率 (m)
GRID_V_RES = 50.0           # 垂直分辨率 (m)
DEM_RES = 12.5              # DEM原始分辨率 (m/像素)

# ===== 综合代价权重 =====
ALPHA = 1.0     # 空间几何代价权重
BETA = 0.8      # 氢气消耗权重
GAMMA = 1.5     # 退化惩罚权重 (γ > α 确保退化惩罚对路径选择有实质影响)

# ===== 风阻模型参数 =====
WIND_CD_BODY = 0.35     # 机体等效阻力系数
WIND_A_BODY = 0.12      # 等效迎风面积 (m²)，四旋翼实际约 0.08~0.15
WIND_HEADWIND_FRAC = 0.6  # 逆风分量占比

# ===== 退化惩罚参数 =====
W1 = 0.6        # 功率跃迁权重
W2 = 0.4        # 功率爬升率权重

# ===== SoH自适应权重 k(SoH) =====
def k_soh(soh: float) -> float:
    """根据燃料电池健康状态返回自适应权重系数"""
    if soh >= 0.9:
        return 1.0
    elif soh >= 0.8:
        # 0.8~0.9 线性插值
        return 1.0 + (0.9 - soh) / 0.1 * 0.5
    elif soh >= 0.7:
        # 0.7~0.8 线性插值
        return 1.5 + (0.8 - soh) / 0.1 * 0.7
    else:
        return 2.2

# ===== 标准大气模型 =====
RHO_SEA_LEVEL = 1.225  # 海平面空气密度 (kg/m³)

def air_density(altitude_m: float) -> float:
    """根据海拔计算空气密度（标准大气模型）"""
    return RHO_SEA_LEVEL * (1.0 - 2.2558e-5 * altitude_m) ** 4.2559

# ===== 事件触发条件 =====
WIND_TRIGGER_RATIO = 1.05   # 条件T2：风场突变代价增量阈值（原代价的105%即触发）

# ===== 起终点坐标（WGS84） =====
START_LON = 110.0869    # 华山北峰下方（玉泉院方向） 经度
START_LAT = 34.4950     # 华山北峰下方 纬度
START_ALT = 500.0       # 海拔 (m)

GOAL_LON = 110.0781     # 华山主峰南峰 经度
GOAL_LAT = 34.4778      # 华山主峰南峰 纬度
GOAL_ALT = 2154.9       # 海拔 (m)

# ===== 风切变事件参数 =====
WINDSHEAR_TIME = 15.0       # 第一次触发时刻 (s)
WINDSHEAR_AHEAD_M = 150.0   # 前方受影响半径 (m)
# 修复说明：原值 800m 在 20M 边图中触发 40 万条边更新（占 2%），
# 导致 LPA* 重建代价接近全局搜索，失去增量优势。
# 150m ≈ 6 个水平栅格，单次影响边数 < 0.05% 总边数，
# 符合 LPA* 设计的"局部扰动"假设。
WIND_NORMAL = 3.0           # 正常风速 (m/s)
WIND_SHEAR = 14.0           # 突增风速 (m/s)

# 多次事件触发配置
N_EVENTS = 3                # 总事件次数
EVENT_INTERVALS = [15.0, 10.0, 8.0]   # 各事件触发间隔(s)，合计33s，适配60s飞行

# ===== 仿真参数 =====
SIM_DT = 1.0                # 仿真时间步长 (s)
SMOOTHING_FACTOR = 0.3      # B样条平滑因子

# ===== 华山五峰参考坐标 =====
PEAKS = {
    "南峰": {"lon": 110.0781, "lat": 34.4778, "elev": 2154.9},
    "北峰": {"lon": 110.0813, "lat": 34.4934, "elev": 1614.0},
    "东峰": {"lon": 110.0820, "lat": 34.4811, "elev": 2096.0},
    "西峰": {"lon": 110.0768, "lat": 34.4816, "elev": 2038.0},
    "中峰": {"lon": 110.0808, "lat": 34.4806, "elev": 2043.0},
}
