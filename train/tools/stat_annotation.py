#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""统计 scanly 缺陷标注框（YOLO txt / Pascal VOC xml）的类型与尺寸分布。

参考 insect/train/train_detect/stat_annotation_from_xml.py，适配本工程目录与 convert 工具。
"""
from __future__ import annotations

import logging
import math
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import convert as cvt

logger = logging.getLogger(__name__)

AnnotationFormat = Literal["yolo", "voc"]

ANNOTATION_COLLECT_PROGRESS_EVERY = 500
SKIP_PATH_PARTS = frozenset({
    ".idea", ".git", "__pycache__", ".pytest_cache", ".vscode", "node_modules",
})
NORM_BOX_BIN_EDGES: List[float] = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0000001]
PIX_BOX_BIN_EDGES: List[float] = [0.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 1.0e9]


def _is_usable_path(path: Path) -> bool:
    name = path.name
    if name.startswith("._") or name == ".DS_Store":
        return False
    if SKIP_PATH_PARTS & set(path.parts):
        return False
    return True


def _bin_label(lo: float, hi: float, *, is_last: bool, as_percent: bool) -> str:
    if as_percent:
        if is_last:
            return f"[{lo * 100:.2f}%, 100.00%]"
        return f"[{lo * 100:.2f}%, {hi * 100:.2f}%)"
    if is_last and hi >= 1e6:
        return f">= {lo:.0f}px"
    if is_last:
        return f"[{lo:.1f}, {hi:.1f}]px"
    return f"[{lo:.1f}, {hi:.1f})px"


def histogram_bin_counts(
    values: Sequence[float],
    edges: Sequence[float],
    *,
    use_percent_labels: bool = False,
) -> Tuple[List[str], List[int]]:
    if len(edges) < 2:
        return [], []
    n = len(edges) - 1
    labels = [
        _bin_label(float(edges[i]), float(edges[i + 1]), is_last=(i == n - 1), as_percent=use_percent_labels)
        for i in range(n)
    ]
    counts = [0] * n
    for v in values:
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        placed = False
        for i in range(n):
            lo, hi = edges[i], edges[i + 1]
            last = i == n - 1
            if last:
                if lo <= x <= hi:
                    counts[i] += 1
                    placed = True
                    break
            elif lo <= x < hi:
                counts[i] += 1
                placed = True
                break
        if not placed:
            counts[0 if x < edges[0] else -1] += 1
    return labels, counts


def load_class_names_from_data_yaml(yaml_path: Path) -> list[str]:
    """轻量解析 data.yaml 的 names 列表（仅支持 `- name` 行）。"""
    names: list[str] = []
    in_names = False
    for line in yaml_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("names:"):
            in_names = True
            continue
        if not in_names:
            continue
        if stripped.startswith("- "):
            names.append(stripped[2:].strip())
            continue
        if stripped and not stripped.startswith("#"):
            break
    return names


class AnnotationStats:
    """统计 YOLO txt / VOC xml 标注框信息。"""

    def __init__(
        self,
        sample_dir: str | Path,
        annotation_format: AnnotationFormat = "yolo",
        recursive: bool = True,
        annotation_dir: str | Path | None = None,
        image_dir: str | Path | None = None,
        categories_path: str | Path | None = None,
        use_diagonal: bool = True,
    ):
        self.sample_dir = Path(sample_dir)
        self.annotation_format = annotation_format
        self.recursive = recursive
        self.annotation_dir = Path(annotation_dir) if annotation_dir else None
        self.image_dir = Path(image_dir) if image_dir else None
        self.use_diagonal = bool(use_diagonal)

        self.stats: Dict[int, Dict[str, Any]] = defaultdict(
            lambda: {
                "count": 0,
                "widths_norm": [],
                "heights_norm": [],
                "widths_px": [],
                "heights_px": [],
            }
        )
        self.image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        self.class_name_to_id: Dict[str, int] = {}
        self.id_to_name: Dict[int, str] = {}
        self.next_class_id = 0

        self.total_annotation_files = 0
        self.empty_annotation_files = 0
        self.missing_image_files = 0
        self.longest_side_extremes_px: Dict[int, Dict[str, object]] = {}

        self._init_class_maps(Path(categories_path) if categories_path else None)

    def _init_class_maps(self, categories_path: Path | None) -> None:
        if categories_path and categories_path.is_file():
            names, id_to_name = cvt.load_categories(categories_path)
            self.id_to_name = dict(id_to_name)
            self.class_name_to_id = {name: cid for cid, name in id_to_name.items()}
            self.next_class_id = max(id_to_name.keys(), default=-1) + 1
            logger.info("已加载类别表 %s，共 %d 类", categories_path, len(names))
            return

        yaml_path = self.sample_dir / "data.yaml"
        if yaml_path.is_file():
            names = load_class_names_from_data_yaml(yaml_path)
            if names:
                self.id_to_name = {i: n for i, n in enumerate(names)}
                self.class_name_to_id = {n: i for i, n in enumerate(names)}
                self.next_class_id = len(names)
                logger.info("已从 data.yaml 加载 %d 类", len(names))

    @staticmethod
    def _safe_relpath(path: Path, base: Path) -> str:
        try:
            return str(path.relative_to(base))
        except ValueError:
            return str(path)

    def _metric_cn(self) -> str:
        return "对角线长度" if self.use_diagonal else "最长边"

    def _metric_values(self, widths: Sequence[float], heights: Sequence[float]) -> List[float]:
        if self.use_diagonal:
            return [math.hypot(w, h) for w, h in zip(widths, heights)]
        return [max(w, h) for w, h in zip(widths, heights)]

    def _metric_stats(self, widths: Sequence[float], heights: Sequence[float]) -> Tuple[float, float, float]:
        vals = self._metric_values(widths, heights)
        if not vals:
            return 0.0, 0.0, 0.0
        return sum(vals) / len(vals), min(vals), max(vals)

    @staticmethod
    def _area_stats(widths: Sequence[float], heights: Sequence[float]) -> Tuple[float, float, float]:
        areas = [w * h for w, h in zip(widths, heights)]
        if not areas:
            return 0.0, 0.0, 0.0
        return sum(areas) / len(areas), min(areas), max(areas)

    def get_class_id_from_name(self, class_name: str) -> int:
        if class_name not in self.class_name_to_id:
            self.class_name_to_id[class_name] = self.next_class_id
            self.id_to_name[self.next_class_id] = class_name
            self.next_class_id += 1
        return self.class_name_to_id[class_name]

    def _class_display(self, class_id: int) -> str:
        name = self.id_to_name.get(class_id)
        if name:
            return name
        return str(class_id)

    def _find_image(self, annotation_path: Path) -> Optional[Path]:
        stem = annotation_path.stem
        for ext in self.image_extensions:
            candidate = annotation_path.with_suffix(ext)
            if candidate.is_file():
                return candidate

        search_dirs: list[Path] = []
        if self.image_dir and self.image_dir.exists():
            search_dirs.append(self.image_dir)

        # YOLO 常见布局：.../labels/train/a.txt → .../images/train/a.*
        for parent in (annotation_path.parent, *annotation_path.parents):
            if parent.name == "labels":
                try:
                    rel = annotation_path.parent.relative_to(parent)
                except ValueError:
                    rel = Path()
                search_dirs.append(parent.parent / "images" / rel)
                break

        for directory in search_dirs:
            if not directory.exists():
                continue
            for ext in self.image_extensions:
                candidate = directory / f"{stem}{ext}"
                if candidate.is_file():
                    return candidate
            found = cvt.find_image_file(directory, f"{stem}.jpg")
            if found is not None:
                return found
        return None

    def parse_txt_file(self, txt_path: Path) -> List[Tuple[int, float, float, float, float]]:
        annotations: List[Tuple[int, float, float, float, float]] = []
        image_width = image_height = 0
        image_path = self._find_image(txt_path)
        if image_path is not None:
            try:
                image_width, image_height = cvt.read_image_size(image_path)
            except ValueError as exc:
                logger.warning("%s", exc)

        try:
            lines = txt_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            logger.error("读取失败 %s: %s", txt_path, exc)
            return annotations

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                logger.warning("格式不正确，跳过 %s:%s %s", txt_path, line_num, line)
                continue
            try:
                class_id = int(parts[0])
                width_norm = float(parts[3])
                height_norm = float(parts[4])
                x_center = float(parts[1])
                y_center = float(parts[2])
            except ValueError as exc:
                logger.warning("解析失败 %s:%s %s", txt_path, line_num, exc)
                continue
            if not (
                0 <= x_center <= 1
                and 0 <= y_center <= 1
                and 0 <= width_norm <= 1
                and 0 <= height_norm <= 1
            ):
                logger.warning("归一化越界，跳过 %s:%s", txt_path, line_num)
                continue
            width_px = width_norm * image_width if image_width > 0 else 0.0
            height_px = height_norm * image_height if image_height > 0 else 0.0
            annotations.append((class_id, width_norm, height_norm, width_px, height_px))
        return annotations

    def parse_xml_file(self, xml_path: Path) -> List[Tuple[int, float, float, float, float]]:
        annotations: List[Tuple[int, float, float, float, float]] = []
        try:
            root = ET.parse(xml_path).getroot()
        except (OSError, ET.ParseError) as exc:
            logger.error("解析 XML 失败 %s: %s", xml_path, exc)
            return annotations

        size = root.find("size")
        image_width = int(size.findtext("width", "0")) if size is not None else 0
        image_height = int(size.findtext("height", "0")) if size is not None else 0
        if image_width <= 0 or image_height <= 0:
            image_path = self._find_image(xml_path)
            if image_path is not None:
                try:
                    image_width, image_height = cvt.read_image_size(image_path)
                except ValueError:
                    pass
        if image_width <= 0 or image_height <= 0:
            logger.warning("图像尺寸无效，跳过 %s", xml_path)
            return annotations

        for obj in root.findall("object"):
            class_name = (obj.findtext("name") or "").strip()
            if not class_name:
                continue
            bnd = obj.find("bndbox")
            if bnd is None:
                continue
            try:
                xmin = float(bnd.findtext("xmin", "0"))
                ymin = float(bnd.findtext("ymin", "0"))
                xmax = float(bnd.findtext("xmax", "0"))
                ymax = float(bnd.findtext("ymax", "0"))
            except ValueError:
                continue
            width_px = xmax - xmin
            height_px = ymax - ymin
            if width_px <= 0 or height_px <= 0:
                continue
            width_norm = width_px / image_width
            height_norm = height_px / image_height
            if not (0 <= width_norm <= 1 and 0 <= height_norm <= 1):
                continue
            class_id = self.get_class_id_from_name(class_name)
            annotations.append((class_id, width_norm, height_norm, width_px, height_px))
        return annotations

    def collect_stats(self) -> None:
        base_dir = self.annotation_dir if self.annotation_dir else self.sample_dir
        file_ext = "*.xml" if self.annotation_format == "voc" else "*.txt"
        pattern = f"**/{file_ext}" if self.recursive else file_ext
        annotation_files = [p for p in sorted(base_dir.glob(pattern)) if _is_usable_path(p)]
        # YOLO 根目录下可能混有 data.yaml 旁的无关 txt，仅保留像标签的文件：跳过非「行含 5 列」不是必须
        if self.annotation_format == "yolo":
            annotation_files = [
                p for p in annotation_files
                if p.name.lower() not in {"requirements.txt", "readme.txt"}
            ]

        if not annotation_files:
            logger.warning("未找到标注文件: %s (%s)", base_dir, file_ext)
            return

        self.total_annotation_files = len(annotation_files)
        logger.info("找到 %d 个标注文件（format=%s）", self.total_annotation_files, self.annotation_format)
        total_boxes = 0
        n_files = len(annotation_files)
        progress_every = ANNOTATION_COLLECT_PROGRESS_EVERY

        for file_idx, annotation_path in enumerate(annotation_files, 1):
            annotation_name = self._safe_relpath(annotation_path, base_dir)
            if self.annotation_format == "voc":
                boxes = self.parse_xml_file(annotation_path)
            else:
                boxes = self.parse_txt_file(annotation_path)
                if not boxes and not annotation_path.read_text(encoding="utf-8").strip():
                    self.empty_annotation_files += 1
                elif not boxes:
                    # 有内容但全被跳过，也计空框文件
                    self.empty_annotation_files += 1
                if self._find_image(annotation_path) is None:
                    self.missing_image_files += 1

            if self.annotation_format == "voc":
                if not boxes:
                    self.empty_annotation_files += 1
                if self._find_image(annotation_path) is None:
                    # VOC 尺寸可来自 xml，缺图不强制；仍计数便于排查
                    pass

            for class_id, width_norm, height_norm, width_px, height_px in boxes:
                if class_id not in self.id_to_name and self.annotation_format == "yolo":
                    # 未知 id 仍统计，展示为数字
                    pass
                self.stats[class_id]["count"] += 1
                self.stats[class_id]["widths_norm"].append(width_norm)
                self.stats[class_id]["heights_norm"].append(height_norm)
                if width_px > 0 and height_px > 0:
                    self.stats[class_id]["widths_px"].append(width_px)
                    self.stats[class_id]["heights_px"].append(height_px)
                    metric_px = (
                        float(math.hypot(width_px, height_px))
                        if self.use_diagonal
                        else float(max(width_px, height_px))
                    )
                    ext = self.longest_side_extremes_px.get(class_id)
                    if ext is None:
                        self.longest_side_extremes_px[class_id] = {
                            "min": metric_px,
                            "min_file": annotation_name,
                            "max": metric_px,
                            "max_file": annotation_name,
                        }
                    else:
                        if metric_px < float(ext["min"]):  # type: ignore[arg-type]
                            ext["min"] = metric_px
                            ext["min_file"] = annotation_name
                        if metric_px > float(ext["max"]):  # type: ignore[arg-type]
                            ext["max"] = metric_px
                            ext["max_file"] = annotation_name
                total_boxes += 1

            if file_idx == n_files or (progress_every > 0 and file_idx % progress_every == 0):
                logger.info("进度: %d/%d，累计框 %d", file_idx, n_files, total_boxes)

        logger.info("共统计 %d 个标注框", total_boxes)

    def _format_interval_table_lines(
        self,
        title: str,
        values: List[float],
        edges: List[float],
        *,
        use_percent_labels: bool,
    ) -> List[str]:
        lines = [title, ""]
        if not values:
            lines.extend(["  (无数据)", ""])
            return lines
        labels, counts = histogram_bin_counts(values, edges, use_percent_labels=use_percent_labels)
        total = sum(counts)
        lines.append(f"{'区间':<28} {'数量':<10} {'占比':<10}")
        lines.append("-" * 48)
        for lab, c in zip(labels, counts):
            pct = (100.0 * c / total) if total else 0.0
            lines.append(f"{lab:<28} {c:<10} {pct:>6.2f}%")
        lines.append(f"{'合计':<28} {total:<10} {'100.00%':>10}")
        lines.append("")
        return lines

    def interval_stats_lines(self) -> List[str]:
        lines = [
            "",
            f"【{self._metric_cn()}像素值区间统计（按类别）】",
            "（分箱边界可在文件顶部 PIX_BOX_BIN_EDGES 调整）",
            "",
        ]
        for class_id in sorted(self.stats.keys()):
            stat = self.stats[class_id]
            metric_px = self._metric_values(stat.get("widths_px") or [], stat.get("heights_px") or [])
            title = f"{self._class_display(class_id)} - {self._metric_cn()}(px)"
            lines.extend(
                self._format_interval_table_lines(
                    title, metric_px, PIX_BOX_BIN_EDGES, use_percent_labels=False
                )
            )
        return lines

    def _report_header_lines(self) -> List[str]:
        lines = [
            "=" * 80,
            f"标注统计结果 (格式: {self.annotation_format})",
            f"样本目录: {self.sample_dir}",
        ]
        if self.annotation_dir:
            lines.append(f"标注目录: {self.annotation_dir}")
        if self.image_dir:
            lines.append(f"图片目录: {self.image_dir}")
        if self.total_annotation_files:
            lines.append(
                f"标注文件数: {self.total_annotation_files}  "
                f"空标注文件数: {self.empty_annotation_files}  "
                f"缺图文件数: {self.missing_image_files}"
            )
        if self.id_to_name:
            lines.append("")
            lines.append("类别名称映射:")
            for class_id, name in sorted(self.id_to_name.items()):
                lines.append(f"  ID {class_id}: {name}")
        lines.append("=" * 80)
        return lines

    def _body_lines(self) -> List[str]:
        if not self.stats:
            return ["没有统计数据"]

        lines: List[str] = []
        sorted_classes = sorted(self.stats.keys())
        lines.append("")
        lines.append("【归一化值统计】")
        lines.append(
            f"{'类别':<18} {'数量':<10} "
            f"{f'{self._metric_cn()}均值':<15} {f'{self._metric_cn()}最小':<15} {f'{self._metric_cn()}最大':<15}"
        )
        lines.append("-" * 73)
        total_count = 0
        for class_id in sorted_classes:
            stat = self.stats[class_id]
            avg_m, min_m, max_m = self._metric_stats(stat["widths_norm"], stat["heights_norm"])
            lines.append(
                f"{self._class_display(class_id):<18} {stat['count']:<10} "
                f"{avg_m:<15.6f} {min_m:<15.6f} {max_m:<15.6f}"
            )
            total_count += stat["count"]
        lines.append("-" * 73)
        lines.append(f"{'总计':<18} {total_count:<10}")

        has_px = any(self.stats[cid]["widths_px"] for cid in sorted_classes)
        if has_px:
            lines.append("")
            lines.append("【原始像素值统计】")
            lines.append(
                f"{'类别':<18} {'数量':<10} "
                f"{f'{self._metric_cn()}均值(px)':<18} {f'{self._metric_cn()}最小(px)':<18} "
                f"{f'{self._metric_cn()}最大(px)':<18} "
                f"{'面积均值(px^2)':<18} {'面积最小(px^2)':<18} {'面积最大(px^2)':<18}"
            )
            lines.append("-" * 142)
            for class_id in sorted_classes:
                stat = self.stats[class_id]
                widths_px = stat["widths_px"]
                heights_px = stat["heights_px"]
                if not widths_px:
                    continue
                avg_m, min_m, max_m = self._metric_stats(widths_px, heights_px)
                avg_a, min_a, max_a = self._area_stats(widths_px, heights_px)
                lines.append(
                    f"{self._class_display(class_id):<18} {len(widths_px):<10} "
                    f"{avg_m:<18.2f} {min_m:<18.2f} {max_m:<18.2f} "
                    f"{avg_a:<18.2f} {min_a:<18.2f} {max_a:<18.2f}"
                )
            lines.append("-" * 142)

        if self.longest_side_extremes_px:
            lines.append("")
            lines.append(f"【每类{self._metric_cn()}(px)极值对应标注文件】")
            lines.append(
                f"{'类别':<18} {'min(px)':<12} {'min文件':<50} {'max(px)':<12} {'max文件'}"
            )
            lines.append("-" * 120)
            rows: List[Tuple[float, int, float, str, float, str]] = []
            for class_id in sorted_classes:
                ext = self.longest_side_extremes_px.get(class_id)
                if not ext:
                    continue
                rows.append(
                    (
                        float(ext["min"]),  # type: ignore[arg-type]
                        class_id,
                        float(ext["min"]),  # type: ignore[arg-type]
                        str(ext["min_file"]),
                        float(ext["max"]),  # type: ignore[arg-type]
                        str(ext["max_file"]),
                    )
                )
            for _, class_id, min_v, min_f, max_v, max_f in sorted(rows, key=lambda x: x[0]):
                lines.append(
                    f"{self._class_display(class_id):<18} "
                    f"{min_v:<12.2f} {min_f:<50} {max_v:<12.2f} {max_f}"
                )
            lines.append("-" * 120)

        lines.extend(self.interval_stats_lines())
        lines.append("=" * 80)
        return lines

    def print_stats(self) -> None:
        for line in self._report_header_lines() + self._body_lines():
            print(line)

    def save_stats_to_file(self, output_path: str | Path | None = None) -> Path:
        if output_path is None:
            output_path = self.sample_dir / f"annotation_stats_{self.annotation_format}.txt"
        else:
            output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        text = "\n".join(self._report_header_lines() + self._body_lines()) + "\n"
        output_path.write_text(text, encoding="utf-8")
        logger.info("统计结果已保存到: %s", output_path)
        return output_path


def run_stats(
    sample_dir: str | Path,
    *,
    annotation_format: AnnotationFormat = "yolo",
    annotation_dir: str | Path | None = None,
    image_dir: str | Path | None = None,
    categories_path: str | Path | None = None,
    recursive: bool = True,
    use_diagonal: bool = True,
    output_path: str | Path | None = None,
) -> AnnotationStats:
    stats = AnnotationStats(
        sample_dir=sample_dir,
        annotation_format=annotation_format,
        recursive=recursive,
        annotation_dir=annotation_dir,
        image_dir=image_dir,
        categories_path=categories_path,
        use_diagonal=use_diagonal,
    )
    stats.collect_stats()
    stats.print_stats()
    stats.save_stats_to_file(output_path)
    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    DEFECTS_ROOT = Path(__file__).resolve().parents[1]
    SAMPLE_ROOT = Path("/Users/shunyaoyin/Documents/code/ai-company/scanly/样本数据").resolve()

    # —— 配置（修改后直接运行）——
    SAMPLE_DIR = DEFECTS_ROOT / "output" / "yolo"
    ANNOTATION_DIR = SAMPLE_DIR / "labels" / "train"
    IMAGE_DIR = SAMPLE_DIR / "images" / "train"
    # 可选: "yolo" / "voc"
    ANNOTATION_FORMAT: AnnotationFormat = "yolo"
    CATEGORIES_PATH = SAMPLE_ROOT / "Categories.json"
    RECURSIVE = False
    USE_DIAGONAL = True
    OUTPUT_PATH = SAMPLE_DIR / f"annotation_stats_{ANNOTATION_FORMAT}.txt"

    run_stats(
        sample_dir=SAMPLE_DIR,
        annotation_format=ANNOTATION_FORMAT,
        annotation_dir=ANNOTATION_DIR,
        image_dir=IMAGE_DIR,
        categories_path=CATEGORIES_PATH,
        recursive=RECURSIVE,
        use_diagonal=USE_DIAGONAL,
        output_path=OUTPUT_PATH,
    )
