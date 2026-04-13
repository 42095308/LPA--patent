"""生成专利流程图与模块框图的 SVG 和 VSDX 文件。"""

from __future__ import annotations

import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape


VISIO_NS = "http://schemas.microsoft.com/office/visio/2012/main"
XML_NS = "http://www.w3.org/XML/1998/namespace"
ET.register_namespace("", VISIO_NS)
ET.register_namespace("r", "http://schemas.openxmlformats.org/officeDocument/2006/relationships")

ROOT_DIR = Path(__file__).resolve().parents[1]
TEMPLATE_VSDX = Path(r"C:\programming\anaconda3\Lib\site-packages\vsdx\media\media.vsdx")
TARGET_DIR = Path(r"C:\Users\42095\Desktop\专利模板\初稿")
METHOD_SVG = TARGET_DIR / "总体方法流程图.svg"
METHOD_VSDX = TARGET_DIR / "总体方法流程图.vsdx"
SYSTEM_SVG = TARGET_DIR / "系统总体框图.svg"
SYSTEM_VSDX = TARGET_DIR / "系统总体框图.vsdx"
PX_PER_INCH = 100.0


def qn(tag: str) -> str:
    """生成带命名空间的 XML 标签。"""
    return f"{{{VISIO_NS}}}{tag}"


def inch(px_value: float) -> float:
    """将布局像素换算为 Visio 页面英寸。"""
    return round(px_value / PX_PER_INCH, 6)


@dataclass
class SvgShape:
    """SVG 图元。"""

    kind: str
    attrs: dict
    text: str | None = None
    lines: list[tuple[str, float, int, bool]] | None = None


@dataclass
class VisioShape:
    """Visio 图元。"""

    kind: str
    x: float
    y: float
    w: float
    h: float
    points: list[tuple[float, float]] | None = None
    text: str | None = None
    font_pt: float = 12.0
    bold: bool = False
    rounded: bool = False
    no_fill: bool = False
    no_line: bool = False
    dashed: bool = False


class DiagramBuilder:
    """同时收集 SVG 和 Visio 图元。"""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.svg_shapes: list[SvgShape] = []
        self.visio_shapes: list[VisioShape] = []

    def add_rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        rounded: bool = False,
        dashed: bool = False,
        stroke: str = "#000000",
        fill: str = "#FFFFFF",
        stroke_width: float = 2.0,
    ) -> None:
        attrs = {
            "x": f"{x}",
            "y": f"{y}",
            "width": f"{w}",
            "height": f"{h}",
            "fill": fill,
            "stroke": stroke,
            "stroke-width": f"{stroke_width}",
        }
        if rounded:
            attrs["rx"] = "10"
            attrs["ry"] = "10"
        if dashed:
            attrs["stroke-dasharray"] = "8 6"
        self.svg_shapes.append(SvgShape("rect", attrs))
        self.visio_shapes.append(
            VisioShape(
                kind="rect",
                x=x,
                y=y,
                w=w,
                h=h,
                rounded=rounded,
                dashed=dashed,
            )
        )

    def add_diamond(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        stroke: str = "#000000",
        fill: str = "#FFFFFF",
        stroke_width: float = 2.0,
    ) -> None:
        points = [
            (x + w / 2.0, y),
            (x + w, y + h / 2.0),
            (x + w / 2.0, y + h),
            (x, y + h / 2.0),
        ]
        point_str = " ".join(f"{px},{py}" for px, py in points)
        attrs = {
            "points": point_str,
            "fill": fill,
            "stroke": stroke,
            "stroke-width": f"{stroke_width}",
        }
        self.svg_shapes.append(SvgShape("polygon", attrs))
        self.visio_shapes.append(
            VisioShape(kind="diamond", x=x, y=y, w=w, h=h)
        )

    def add_polyline(
        self,
        points: Sequence[tuple[float, float]],
        *,
        dashed: bool = False,
        stroke: str = "#000000",
        stroke_width: float = 2.0,
    ) -> None:
        attrs = {
            "points": " ".join(f"{px},{py}" for px, py in points),
            "fill": "none",
            "stroke": stroke,
            "stroke-width": f"{stroke_width}",
        }
        if dashed:
            attrs["stroke-dasharray"] = "8 6"
        self.svg_shapes.append(SvgShape("polyline", attrs))
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        self.visio_shapes.append(
            VisioShape(
                kind="polyline",
                x=min(xs),
                y=min(ys),
                w=max(xs) - min(xs),
                h=max(ys) - min(ys),
                points=list(points),
                dashed=dashed,
                no_fill=True,
            )
        )

    def add_arrow_head(
        self,
        tip_x: float,
        tip_y: float,
        direction: str,
        *,
        size: float = 12.0,
        fill: str = "#000000",
    ) -> None:
        if direction == "down":
            points = [(tip_x, tip_y), (tip_x - size, tip_y - size * 1.4), (tip_x + size, tip_y - size * 1.4)]
        elif direction == "up":
            points = [(tip_x, tip_y), (tip_x - size, tip_y + size * 1.4), (tip_x + size, tip_y + size * 1.4)]
        elif direction == "left":
            points = [(tip_x, tip_y), (tip_x + size * 1.4, tip_y - size), (tip_x + size * 1.4, tip_y + size)]
        elif direction == "right":
            points = [(tip_x, tip_y), (tip_x - size * 1.4, tip_y - size), (tip_x - size * 1.4, tip_y + size)]
        else:
            raise ValueError(f"未知箭头方向: {direction}")

        point_str = " ".join(f"{px},{py}" for px, py in points)
        attrs = {"points": point_str, "fill": fill, "stroke": fill, "stroke-width": "1"}
        self.svg_shapes.append(SvgShape("polygon", attrs))
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        self.visio_shapes.append(
            VisioShape(
                kind="polygon",
                x=min(xs),
                y=min(ys),
                w=max(xs) - min(xs),
                h=max(ys) - min(ys),
                points=list(points),
            )
        )

    def add_text_box(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str,
        *,
        font_px: int = 18,
        bold: bool = False,
        color: str = "#000000",
    ) -> None:
        lines = text.split("\n")
        line_gap = font_px + 6
        start_y = y + (h - line_gap * (len(lines) - 1)) / 2.0
        svg_lines = []
        for index, line in enumerate(lines):
            svg_lines.append((line, start_y + index * line_gap, font_px, bold))
        self.svg_shapes.append(
            SvgShape(
                kind="text_block",
                attrs={"x": f"{x}", "y": f"{y}", "width": f"{w}", "height": f"{h}", "color": color},
                lines=svg_lines,
            )
        )
        self.visio_shapes.append(
            VisioShape(
                kind="text",
                x=x,
                y=y,
                w=w,
                h=h,
                text=text,
                font_pt=font_px * 0.75,
                bold=bold,
                no_fill=True,
                no_line=True,
            )
        )

    def to_svg(self) -> str:
        """导出 SVG。"""
        parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" viewBox="0 0 {self.width} {self.height}">',
            f'  <rect width="{self.width}" height="{self.height}" fill="#FFFFFF"/>',
        ]
        for shape in self.svg_shapes:
            if shape.kind == "rect":
                attrs = " ".join(f'{key}="{escape(value)}"' for key, value in shape.attrs.items())
                parts.append(f"  <rect {attrs}/>")
            elif shape.kind == "polygon":
                attrs = " ".join(f'{key}="{escape(value)}"' for key, value in shape.attrs.items())
                parts.append(f"  <polygon {attrs}/>")
            elif shape.kind == "polyline":
                attrs = " ".join(f'{key}="{escape(value)}"' for key, value in shape.attrs.items())
                parts.append(f"  <polyline {attrs}/>")
            elif shape.kind == "text_block":
                color = shape.attrs["color"]
                parts.append(
                    '  <g font-family="SimSun, Microsoft YaHei, Arial, sans-serif" '
                    f'fill="{color}" text-anchor="middle">'
                )
                box_x = float(shape.attrs["x"])
                box_w = float(shape.attrs["width"])
                text_x = box_x + box_w / 2.0
                for line, line_y, font_px, bold in shape.lines or []:
                    weight = "700" if bold else "400"
                    parts.append(
                        f'    <text x="{text_x}" y="{line_y}" font-size="{font_px}" font-weight="{weight}">{escape(line)}</text>'
                    )
                parts.append("  </g>")
        parts.append("</svg>")
        return "\n".join(parts)

    def _shape_common(self, shape_id: int, x: float, y: float, w: float, h: float) -> ET.Element:
        """生成通用 Shape 节点。"""
        pin_x = inch(x + w / 2.0)
        pin_y = inch(self.height - y - h / 2.0)
        width_in = max(inch(w), 0.01)
        height_in = max(inch(h), 0.01)
        shape = ET.Element(qn("Shape"), ID=str(shape_id), Type="Shape", LineStyle="3", FillStyle="3", TextStyle="3")
        for name, value in [
            ("PinX", pin_x),
            ("PinY", pin_y),
            ("Width", width_in),
            ("Height", height_in),
            ("LocPinX", round(width_in / 2.0, 6)),
            ("LocPinY", round(height_in / 2.0, 6)),
            ("Angle", 0),
            ("FlipX", 0),
            ("FlipY", 0),
            ("ResizeMode", 0),
        ]:
            ET.SubElement(shape, qn("Cell"), N=name, V=str(value))
        ET.SubElement(shape, qn("Cell"), N="LineColor", V="#000000")
        ET.SubElement(shape, qn("Cell"), N="FillForegnd", V="#FFFFFF")
        ET.SubElement(shape, qn("Cell"), N="LineWeight", V="0.018")
        ET.SubElement(shape, qn("Cell"), N="VerticalAlign", V="1")
        return shape

    def _add_character_section(self, shape: ET.Element, font_pt: float, bold: bool) -> None:
        """设置文字字号和加粗。"""
        section = ET.SubElement(shape, qn("Section"), N="Character")
        row = ET.SubElement(section, qn("Row"), IX="0")
        ET.SubElement(row, qn("Cell"), N="Size", V=str(round(font_pt / 72.0, 6)))
        ET.SubElement(row, qn("Cell"), N="Color", V="0")
        ET.SubElement(row, qn("Cell"), N="Style", V="1" if bold else "0")
        ET.SubElement(row, qn("Cell"), N="LangID", V="2052")
        p_section = ET.SubElement(shape, qn("Section"), N="Paragraph")
        p_row = ET.SubElement(p_section, qn("Row"), IX="0")
        ET.SubElement(p_row, qn("Cell"), N="HorzAlign", V="1")

    def _add_geometry(self, shape: ET.Element, points: Sequence[tuple[float, float]], *, no_fill: bool, no_line: bool) -> None:
        """写入几何路径。"""
        section = ET.SubElement(shape, qn("Section"), N="Geometry", IX="0")
        ET.SubElement(section, qn("Cell"), N="NoFill", V="1" if no_fill else "0")
        ET.SubElement(section, qn("Cell"), N="NoLine", V="1" if no_line else "0")
        ET.SubElement(section, qn("Cell"), N="NoShow", V="0")
        ET.SubElement(section, qn("Cell"), N="NoSnap", V="0")
        ET.SubElement(section, qn("Cell"), N="NoQuickDrag", V="0")
        for index, (point_x, point_y) in enumerate(points, start=1):
            row_type = "MoveTo" if index == 1 else "LineTo"
            row = ET.SubElement(section, qn("Row"), T=row_type, IX=str(index))
            ET.SubElement(row, qn("Cell"), N="X", V=str(round(point_x, 6)))
            ET.SubElement(row, qn("Cell"), N="Y", V=str(round(point_y, 6)))

    def to_visio_page_xml(self) -> bytes:
        """导出 PageContents XML。"""
        page = ET.Element(qn("PageContents"), {f"{{{XML_NS}}}space": "preserve"})
        shapes_el = ET.SubElement(page, qn("Shapes"))
        shape_id = 1

        for spec in self.visio_shapes:
            if spec.kind == "rect":
                shape = self._shape_common(shape_id, spec.x, spec.y, spec.w, spec.h)
                if spec.rounded:
                    ET.SubElement(shape, qn("Cell"), N="Rounding", V="0.12")
                if spec.dashed:
                    ET.SubElement(shape, qn("Cell"), N="LinePattern", V="2")
                width_in = max(inch(spec.w), 0.01)
                height_in = max(inch(spec.h), 0.01)
                rect_points = [(0, 0), (width_in, 0), (width_in, height_in), (0, height_in), (0, 0)]
                self._add_geometry(shape, rect_points, no_fill=False, no_line=False)
                shapes_el.append(shape)
            elif spec.kind == "diamond":
                shape = self._shape_common(shape_id, spec.x, spec.y, spec.w, spec.h)
                width_in = max(inch(spec.w), 0.01)
                height_in = max(inch(spec.h), 0.01)
                diamond_points = [
                    (width_in / 2.0, height_in),
                    (width_in, height_in / 2.0),
                    (width_in / 2.0, 0),
                    (0, height_in / 2.0),
                    (width_in / 2.0, height_in),
                ]
                self._add_geometry(shape, diamond_points, no_fill=False, no_line=False)
                shapes_el.append(shape)
            elif spec.kind == "polyline":
                pts = spec.points or []
                min_x = min(p[0] for p in pts)
                max_x = max(p[0] for p in pts)
                min_y = min(p[1] for p in pts)
                max_y = max(p[1] for p in pts)
                shape = self._shape_common(shape_id, min_x, self.height - max_y, max(max_x - min_x, 1), max(max_y - min_y, 1))
                if spec.dashed:
                    ET.SubElement(shape, qn("Cell"), N="LinePattern", V="2")
                ET.SubElement(shape, qn("Cell"), N="FillPattern", V="0")
                local_points = []
                for point_x, point_y in pts:
                    vx = inch(point_x - min_x)
                    vy = inch(max_y - point_y)
                    local_points.append((vx, vy))
                self._add_geometry(shape, local_points, no_fill=True, no_line=False)
                shapes_el.append(shape)
            elif spec.kind == "polygon":
                pts = spec.points or []
                min_x = min(p[0] for p in pts)
                max_x = max(p[0] for p in pts)
                min_y = min(p[1] for p in pts)
                max_y = max(p[1] for p in pts)
                shape = self._shape_common(shape_id, min_x, min_y, max(max_x - min_x, 1), max(max_y - min_y, 1))
                local_points = []
                for point_x, point_y in pts:
                    vx = inch(point_x - min_x)
                    vy = inch(max_y - point_y)
                    local_points.append((vx, vy))
                local_points.append(local_points[0])
                self._add_geometry(shape, local_points, no_fill=False, no_line=False)
                shapes_el.append(shape)
            elif spec.kind == "text":
                shape = self._shape_common(shape_id, spec.x, spec.y, spec.w, spec.h)
                ET.SubElement(shape, qn("Cell"), N="FillPattern", V="0")
                self._add_character_section(shape, spec.font_pt, spec.bold)
                width_in = max(inch(spec.w), 0.01)
                height_in = max(inch(spec.h), 0.01)
                text_rect = [(0, 0), (width_in, 0), (width_in, height_in), (0, height_in), (0, 0)]
                self._add_geometry(shape, text_rect, no_fill=True, no_line=True)
                if spec.text:
                    text_node = ET.SubElement(shape, qn("Text"))
                    text_node.text = spec.text
                shapes_el.append(shape)
            shape_id += 1

        ET.SubElement(page, qn("Connects"))
        return ET.tostring(page, encoding="utf-8", xml_declaration=True)

    def pages_xml(self, page_name: str) -> bytes:
        """生成 pages.xml。"""
        pages = ET.Element(qn("Pages"), {f"{{{XML_NS}}}space": "preserve"})
        page = ET.SubElement(
            pages,
            qn("Page"),
            ID="0",
            NameU=page_name,
            Name=page_name,
            ViewScale="1",
            ViewCenterX=str(round(inch(self.width / 2.0), 6)),
            ViewCenterY=str(round(inch(self.height / 2.0), 6)),
        )
        sheet = ET.SubElement(page, qn("PageSheet"), LineStyle="0", FillStyle="0", TextStyle="0")
        ET.SubElement(sheet, qn("Cell"), N="PageWidth", V=str(round(inch(self.width), 6)))
        ET.SubElement(sheet, qn("Cell"), N="PageHeight", V=str(round(inch(self.height), 6)))
        ET.SubElement(sheet, qn("Cell"), N="ShdwOffsetX", V="0.1181102362204724")
        ET.SubElement(sheet, qn("Cell"), N="ShdwOffsetY", V="-0.1181102362204724")
        ET.SubElement(sheet, qn("Cell"), N="PageScale", V="0.03937007874015748", U="MM")
        ET.SubElement(sheet, qn("Cell"), N="DrawingScale", V="0.03937007874015748", U="MM")
        ET.SubElement(sheet, qn("Cell"), N="DrawingSizeType", V="0")
        ET.SubElement(sheet, qn("Cell"), N="DrawingScaleType", V="0")
        ET.SubElement(sheet, qn("Cell"), N="InhibitSnap", V="0")
        ET.SubElement(sheet, qn("Cell"), N="PageLockReplace", V="0", U="BOOL")
        ET.SubElement(sheet, qn("Cell"), N="PageLockDuplicate", V="0", U="BOOL")
        ET.SubElement(sheet, qn("Cell"), N="UIVisibility", V="0")
        ET.SubElement(sheet, qn("Cell"), N="ShdwType", V="0")
        ET.SubElement(sheet, qn("Cell"), N="ShdwObliqueAngle", V="0")
        ET.SubElement(sheet, qn("Cell"), N="ShdwScaleFactor", V="1")
        ET.SubElement(sheet, qn("Cell"), N="DrawingResizeType", V="1")
        ET.SubElement(sheet, qn("Cell"), N="PageShapeSplit", V="1")
        rel = ET.SubElement(page, qn("Rel"))
        rel.set("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id", "rId1")
        return ET.tostring(pages, encoding="utf-8", xml_declaration=True)


def build_method_diagram() -> DiagramBuilder:
    """构建总体方法流程图。"""
    d = DiagramBuilder(width=980, height=1670)

    # 开始
    d.add_rect(380, 26, 220, 50, rounded=True)
    d.add_text_box(380, 31, 220, 38, "开始", font_px=24, bold=True)

    # S1
    d.add_rect(170, 110, 640, 112, rounded=True)
    d.add_text_box(190, 122, 600, 28, "S1 构建三维空间图结构并定义综合代价函数", font_px=22, bold=True)
    d.add_text_box(
        192,
        154,
        596,
        58,
        "获取目标飞行区域的地形高程、局部风场、障碍物信息、空域约束信息以及无人机动力系统状态参数；\n构建三维空间图结构，并为相邻节点之间的有向边定义综合代价函数。",
        font_px=15,
    )
    d.add_text_box(250, 204, 480, 18, "空间几何代价项 + 预测氢气消耗项 + 燃料电池寿命退化惩罚项", font_px=14)

    # S2
    d.add_rect(170, 262, 640, 108, rounded=True)
    d.add_text_box(190, 274, 600, 28, "S2 执行初始路径规划并输出初始预测功率曲线", font_px=22, bold=True)
    d.add_text_box(
        192,
        306,
        596,
        48,
        "基于综合代价函数执行 LPA* 增量路径规划，得到初始最优路径；\n向能量管理系统输出对应的初始预测功率曲线 P_predict(t)。",
        font_px=15,
    )
    d.add_text_box(270, 350, 440, 14, "初始最优路径 / 初始预测功率曲线 / 前馈准备", font_px=14)

    # S3
    d.add_rect(170, 414, 640, 102, rounded=True)
    d.add_text_box(190, 426, 600, 28, "S3 飞行过程中实时监测外部环境状态与无人机健康状态", font_px=21, bold=True)
    d.add_text_box(
        192,
        458,
        596,
        48,
        "沿当前有效路径飞行；\n实时监测环境障碍物、局部风场、路径代价变化、预测功率跃迁及燃料电池健康状态。",
        font_px=15,
    )

    # D1
    d.add_diamond(170, 560, 640, 220)
    d.add_text_box(250, 582, 480, 24, "D1 是否满足事件触发条件", font_px=22, bold=True)
    d.add_text_box(
        236,
        618,
        508,
        110,
        "是否满足以下任一条件：\n障碍物或禁飞区信息更新；\n风场变化导致当前路径综合代价增加；\n预测需求功率跃迁超限；\n燃料电池健康状态变化导致当前路径综合代价突破安全约束。",
        font_px=15,
    )

    # D1 否分支说明
    d.add_rect(818, 612, 132, 72, rounded=True)
    d.add_text_box(828, 626, 112, 42, "继续沿当前有效路径飞行\n转入任务完成判断", font_px=14)
    d.add_text_box(836, 570, 48, 20, "否", font_px=18, bold=True)

    # S3-1
    d.add_rect(160, 820, 660, 116, rounded=True)
    d.add_text_box(180, 832, 620, 30, "S3-1 触发局部增量重规划并生成新的重连段路径", font_px=21, bold=True)
    d.add_text_box(
        182,
        866,
        616,
        54,
        "触发事件驱动的 LPA* 局部增量重规划；\n利用历史搜索中间量，仅对受事件影响的局部节点集合与边集合进行更新；\n生成新的重连段路径。",
        font_px=15,
    )

    # S4
    d.add_rect(160, 978, 660, 114, rounded=True)
    d.add_text_box(180, 990, 620, 28, "S4 提取重连段轨迹特征并生成结构化消息", font_px=22, bold=True)
    d.add_text_box(
        182,
        1022,
        616,
        50,
        "对重连段路径提取轨迹特征；\n生成包含路径特征向量与前瞻功率曲线 P_predict(t) 的结构化消息；\n将结构化消息发送至能量管理系统。",
        font_px=15,
    )
    d.add_text_box(214, 1072, 552, 14, "在一些实施方式中，还可包含重规划完成时间戳和预测时间窗口总长度 T_window", font_px=14)

    # S5
    d.add_rect(150, 1136, 680, 136, rounded=True)
    d.add_text_box(170, 1148, 640, 30, "S5 能量管理系统执行前馈预调并触发飞控执行", font_px=22, bold=True)
    d.add_text_box(
        172,
        1182,
        636,
        74,
        "能量管理系统解析结构化消息；\n在预设提前量 Δt 时间窗口内进行功率基线预调；\n控制燃料电池输出功率以受限爬升方式趋近预测需求功率；\n同步调整锂电池输出以承担瞬态功率补偿；\n预调完成后，触发飞控系统按照重连段执行姿态控制与推进控制。",
        font_px=15,
    )
    d.add_text_box(392, 1252, 196, 14, "Δt ≥ τ_FC", font_px=15, bold=True)

    # 返回监测闭环
    d.add_rect(220, 1318, 540, 86, rounded=True)
    d.add_text_box(240, 1332, 500, 24, "返回监测闭环", font_px=22, bold=True)
    d.add_text_box(242, 1364, 496, 24, "飞控执行新的重连段路径\n返回 S3 持续监测", font_px=15)

    # D2
    d.add_diamond(310, 1436, 360, 170)
    d.add_text_box(360, 1470, 260, 24, "D2 是否到达终点或任务完成", font_px=21, bold=True)
    d.add_text_box(360, 1512, 260, 28, "是否到达任务目标点\n或任务结束", font_px=16)
    d.add_text_box(688, 1514, 48, 20, "是", font_px=18, bold=True)
    d.add_text_box(220, 1508, 48, 20, "否", font_px=18, bold=True)

    # 结束
    d.add_rect(380, 1610, 220, 50, rounded=True)
    d.add_text_box(380, 1616, 220, 34, "结束", font_px=24, bold=True)

    # 主线箭头
    for start_y, end_y in [(76, 110), (222, 262), (370, 414), (516, 560), (780, 820), (936, 978), (1092, 1136), (1272, 1318), (1404, 1436), (1606, 1610)]:
        d.add_polyline([(490, start_y), (490, end_y)])
        d.add_arrow_head(490, end_y, "down", size=12)

    # D1 -> 右侧说明
    d.add_polyline([(810, 670), (818, 670)])
    d.add_arrow_head(818, 670, "right", size=10)
    d.add_polyline([(884, 684), (884, 1518), (670, 1518)])
    d.add_arrow_head(670, 1518, "left", size=10)

    # D1 -> S3-1
    d.add_text_box(510, 786, 50, 18, "是", font_px=18, bold=True)

    # D2 -> End
    d.add_polyline([(490, 1606), (490, 1610)])
    d.add_arrow_head(490, 1610, "down", size=10)

    # D2 否 -> 回到 S3
    d.add_polyline([(310, 1521), (90, 1521), (90, 465), (170, 465)])
    d.add_arrow_head(170, 465, "right", size=10)

    return d


def build_system_diagram() -> DiagramBuilder:
    """构建系统总体框图。"""
    d = DiagramBuilder(width=1050, height=1180)

    d.add_text_box(425, 28, 200, 24, "外部输入", font_px=18)
    d.add_text_box(850, 1128, 140, 24, "系统输出", font_px=18)

    # 输入层
    d.add_rect(70, 70, 320, 84)
    d.add_text_box(90, 92, 280, 42, "飞行区域环境、风场、障碍物\n与空域约束数据", font_px=20, bold=True)

    d.add_rect(660, 70, 320, 84)
    d.add_text_box(680, 92, 280, 42, "动力系统状态参数、历史运行数据\n与任务相关配置", font_px=20, bold=True)

    # 模块层
    d.add_rect(70, 230, 320, 126)
    d.add_text_box(96, 250, 268, 26, "多维能量地图构建模块", font_px=22, bold=True)
    d.add_text_box(
        94,
        284,
        272,
        58,
        "执行步骤 S1、S2；\n构建三维走廊子图并定义综合代价函数；\n在健康状态变化时按健康权重修正边代价。",
        font_px=15,
    )

    d.add_rect(660, 230, 320, 140)
    d.add_text_box(686, 248, 268, 26, "事件监测与触发模块", font_px=22, bold=True)
    d.add_text_box(
        682,
        282,
        276,
        74,
        "执行步骤 S6、S7 中的触发判定；\n持续采集环境扰动、路径代价变化、预测功率跃迁\n及燃料电池健康状态，并判断是否发起局部重规划请求。",
        font_px=15,
    )

    d.add_rect(365, 452, 320, 110)
    d.add_text_box(390, 470, 270, 24, "增量重规划模块", font_px=22, bold=True)
    d.add_text_box(
        386,
        504,
        278,
        42,
        "执行步骤 S3、S7；\n基于 LPA* 仅更新受影响节点与边，\n复用历史搜索结果并生成新的重连段路径。",
        font_px=15,
    )

    d.add_rect(365, 624, 320, 110)
    d.add_text_box(390, 642, 270, 24, "特征提取与消息生成模块", font_px=21, bold=True)
    d.add_text_box(
        384,
        676,
        282,
        42,
        "执行步骤 S4；\n提取重连段轨迹特征，生成包含 P_predict(t)、\nT_window、时间戳和特征向量的结构化消息。",
        font_px=15,
    )

    d.add_rect(365, 796, 320, 110)
    d.add_text_box(392, 814, 266, 24, "能量管理预调模块", font_px=22, bold=True)
    d.add_text_box(
        388,
        848,
        274,
        42,
        "执行步骤 S5；\n根据结构化消息进行前馈预调，\n并由锂电池承担瞬态功率补偿。",
        font_px=15,
    )

    d.add_rect(365, 968, 320, 110)
    d.add_text_box(392, 986, 266, 24, "飞行执行控制模块", font_px=22, bold=True)
    d.add_text_box(
        388,
        1020,
        274,
        42,
        "执行步骤 S6；\n满足预调完成条件后切换新路径执行，\n并将观测量反馈至事件监测与触发模块。",
        font_px=15,
    )

    d.add_rect(365, 1106, 320, 60)
    d.add_text_box(392, 1122, 266, 28, "协同飞行路径与控制结果", font_px=22, bold=True)

    # 输入箭头
    d.add_polyline([(230, 154), (230, 230)])
    d.add_arrow_head(230, 230, "down", size=10)
    d.add_polyline([(820, 154), (820, 230)])
    d.add_arrow_head(820, 230, "down", size=10)

    # 汇聚到增量重规划
    d.add_polyline([(390, 293), (520, 293), (520, 452)])
    d.add_arrow_head(520, 452, "down", size=10)
    d.add_text_box(396, 270, 170, 18, "三维空间图结构与综合边代价", font_px=15)

    d.add_polyline([(660, 300), (630, 300), (630, 452)])
    d.add_arrow_head(630, 452, "down", size=10)
    d.add_text_box(632, 274, 190, 18, "局部重规划请求 / 触发结果", font_px=15)

    # 主链
    for start_y, end_y in [(562, 624), (734, 796), (906, 968), (1078, 1106)]:
        d.add_polyline([(525, start_y), (525, end_y)])
        d.add_arrow_head(525, end_y, "down", size=10)

    d.add_text_box(560, 596, 150, 16, "新的重连段路径", font_px=15)
    d.add_text_box(580, 768, 170, 16, "结构化消息", font_px=15)
    d.add_text_box(570, 940, 170, 16, "预调完成后触发执行", font_px=15)

    # 反馈闭环
    d.add_polyline([(685, 1023), (930, 1023), (930, 370), (820, 370)])
    d.add_arrow_head(820, 370, "up", size=10)
    d.add_text_box(934, 742, 96, 60, "位置、电压、氢耗、\n功率跃迁与健康状态反馈", font_px=14)

    d.add_polyline([(820, 370), (820, 410), (390, 410), (390, 452)])
    d.add_arrow_head(390, 452, "down", size=10)
    d.add_text_box(438, 388, 190, 18, "触发后进入局部增量重规划", font_px=15)

    return d


def write_vsdx(builder: DiagramBuilder, page_name: str, target: Path) -> None:
    """将图元写入 VSDX 文件。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    page_xml = builder.to_visio_page_xml()
    pages_xml = builder.pages_xml(page_name)

    with zipfile.ZipFile(TEMPLATE_VSDX, "r") as source_zip:
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as target_zip:
            for info in source_zip.infolist():
                data = source_zip.read(info.filename)
                if info.filename == "visio/pages/page1.xml":
                    data = page_xml
                elif info.filename == "visio/pages/pages.xml":
                    data = pages_xml
                target_zip.writestr(info, data)


def main() -> None:
    """生成两张图的 SVG 与 VSDX 文件。"""
    method_diagram = build_method_diagram()
    METHOD_SVG.write_text(method_diagram.to_svg(), encoding="utf-8")
    write_vsdx(method_diagram, "总体方法流程图", METHOD_VSDX)

    system_diagram = build_system_diagram()
    SYSTEM_SVG.write_text(system_diagram.to_svg(), encoding="utf-8")
    write_vsdx(system_diagram, "系统总体框图", SYSTEM_VSDX)

    print("已生成文件：")
    print(f"- {METHOD_SVG}")
    print(f"- {METHOD_VSDX}")
    print(f"- {SYSTEM_SVG}")
    print(f"- {SYSTEM_VSDX}")


if __name__ == "__main__":
    main()
