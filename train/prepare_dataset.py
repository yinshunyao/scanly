#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""将 Label Studio YOLO 导出包或扁平 Pascal VOC 过滤、对齐类别并转为可训练的 YOLO11 检测 / OBB 目录。"""
from __future__ import annotations

import json
import logging
import random
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".bmp", ".png", ".tif", ".tiff", ".webp"}
CLASSES_FILENAME = "classes.txt"
VOC_CLASSES_FILENAME = "predefined_classes.txt"
VOC_IMAGE_DIR_NAMES = ("JPEGImages", "images", "Images")
VOC_ANNOTATION_DIR_NAMES = ("Annotations", "annotations")
LabelFormat = Literal["detect", "obb"]
SourceKind = Literal["yolo", "voc"]
MIN_OBB_AREA = 1e-10


@dataclass
class SampleRecord:
    subset: str
    image_path: Path
    out_stem: str
    lines: list[str]
    box_counts: dict[str, int] = field(default_factory=dict)


def read_classes_txt(path: Path) -> list[str]:
    names: list[str] = []
    text = path.read_text(encoding="utf-8-sig")
    for raw in text.splitlines():
        name = raw.strip()
        if not name or name.startswith("#"):
            continue
        names.append(name)
    if not names:
        raise ValueError(f"类别文件为空: {path}")
    return names


def is_ls_yolo_export(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "images").is_dir()
        and (path / "labels").is_dir()
        and (path / CLASSES_FILENAME).is_file()
    )


def _find_named_subdir(root: Path, candidates: tuple[str, ...]) -> Path | None:
    for name in candidates:
        path = root / name
        if path.is_dir():
            return path
    return None


def is_voc_export(path: Path) -> bool:
    return (
        path.is_dir()
        and _find_named_subdir(path, VOC_IMAGE_DIR_NAMES) is not None
        and _find_named_subdir(path, VOC_ANNOTATION_DIR_NAMES) is not None
    )


def discover_subsets(source_root: Path) -> list[Path]:
    """发现 source_root 下全部 Label Studio YOLO 导出包（支持多层子目录自动合并）。"""
    source_root = source_root.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"原始数据集目录不存在: {source_root}")
    if is_ls_yolo_export(source_root):
        return [source_root]

    subsets: list[Path] = []
    stack = [source_root]
    while stack:
        current = stack.pop()
        if is_ls_yolo_export(current):
            subsets.append(current)
            continue
        if is_voc_export(current):
            continue
        for child in sorted(current.iterdir()):
            if child.is_dir() and not child.name.startswith("."):
                stack.append(child)

    subsets.sort(key=lambda p: str(p))
    return subsets


def discover_voc_roots(source_root: Path) -> list[Path]:
    """发现扁平 VOC 根（JPEGImages + Annotations）。"""
    source_root = source_root.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"原始数据集目录不存在: {source_root}")
    if is_voc_export(source_root):
        return [source_root]

    roots: list[Path] = []
    stack = [source_root]
    while stack:
        current = stack.pop()
        if is_voc_export(current):
            roots.append(current)
            continue
        if is_ls_yolo_export(current):
            continue
        for child in sorted(current.iterdir()):
            if child.is_dir() and not child.name.startswith("."):
                stack.append(child)
    roots.sort(key=lambda p: str(p))
    return roots


def discover_source(source_root: Path) -> tuple[SourceKind, list[Path]]:
    source_root = Path(source_root).resolve()
    yolo_subsets = discover_subsets(source_root)
    if yolo_subsets:
        return "yolo", yolo_subsets
    voc_roots = discover_voc_roots(source_root)
    if voc_roots:
        return "voc", voc_roots
    raise FileNotFoundError(
        "未找到 Label Studio YOLO 导出包（需含 images/、labels/、classes.txt）"
        f"或扁平 VOC（需含 JPEGImages/、Annotations/）: {source_root}"
    )


def _subset_key(subset: Path, source_root: Path) -> str:
    try:
        return str(subset.relative_to(source_root.resolve()))
    except ValueError:
        return subset.name


def collect_available_classes(subsets: list[Path]) -> list[str]:
    seen: list[str] = []
    for subset in subsets:
        for name in read_classes_txt(subset / CLASSES_FILENAME):
            if name not in seen:
                seen.append(name)
    return seen


def find_image_file(image_dir: Path, stem: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        candidate = image_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    matches = [
        p
        for p in image_dir.iterdir()
        if p.is_file() and p.stem == stem and p.suffix.lower() in IMAGE_SUFFIXES
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def read_image_wh(path: Path) -> tuple[int, int] | None:
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(path) as image:
            width, height = image.size
    except OSError:
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


def voc_bndbox_to_xywh(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    if width <= 0 or height <= 0:
        return None
    xmin = max(0.0, min(float(width), xmin))
    xmax = max(0.0, min(float(width), xmax))
    ymin = max(0.0, min(float(height), ymin))
    ymax = max(0.0, min(float(height), ymax))
    box_w = xmax - xmin
    box_h = ymax - ymin
    if box_w <= 0.0 or box_h <= 0.0:
        return None
    cx = (xmin + xmax) / 2.0 / width
    cy = (ymin + ymax) / 2.0 / height
    return _clip01(cx), _clip01(cy), _clip01(box_w / width), _clip01(box_h / height)


def _xml_image_size(root: ET.Element) -> tuple[int, int]:
    size = root.find("size")
    if size is None:
        return 0, 0
    try:
        width = int(float(size.findtext("width", "0") or "0"))
        height = int(float(size.findtext("height", "0") or "0"))
    except ValueError:
        return 0, 0
    return width, height


def collect_voc_class_names(voc_root: Path) -> list[str]:
    classes_path = voc_root / VOC_CLASSES_FILENAME
    if classes_path.is_file():
        return read_classes_txt(classes_path)
    annotation_dir = _find_named_subdir(voc_root, VOC_ANNOTATION_DIR_NAMES)
    if annotation_dir is None:
        return []
    names: list[str] = []
    for xml_path in sorted(annotation_dir.glob("*.xml")):
        if xml_path.name.startswith("._"):
            continue
        try:
            root = ET.parse(xml_path).getroot()
        except (OSError, ET.ParseError):
            continue
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip()
            if name and name not in names:
                names.append(name)
    return names


def collect_available_voc_classes(voc_roots: list[Path]) -> list[str]:
    seen: list[str] = []
    for voc_root in voc_roots:
        for name in collect_voc_class_names(voc_root):
            if name not in seen:
                seen.append(name)
    return seen


def convert_voc_xml(
    xml_path: Path,
    image_path: Path,
    name_to_new_id: dict[str, int],
    *,
    label_format: LabelFormat = "detect",
) -> tuple[list[str], dict[str, int], dict[str, int]]:
    counters = {
        "kept": 0,
        "dropped_class": 0,
        "dropped_invalid": 0,
        "bbox": 0,
        "polygon": 0,
        "obb4": 0,
    }
    box_counts: dict[str, int] = {}
    lines: list[str] = []
    try:
        root = ET.parse(xml_path).getroot()
    except (OSError, ET.ParseError) as exc:
        logger.warning("解析 VOC XML 失败 %s: %s", xml_path, exc)
        counters["dropped_invalid"] += 1
        return lines, box_counts, counters

    width, height = _xml_image_size(root)
    if width <= 0 or height <= 0:
        fallback = read_image_wh(image_path)
        if fallback is None:
            logger.warning("图像尺寸无效，跳过 %s", xml_path)
            counters["dropped_invalid"] += 1
            return lines, box_counts, counters
        width, height = fallback

    for obj in root.findall("object"):
        class_name = (obj.findtext("name") or "").strip()
        if not class_name:
            counters["dropped_invalid"] += 1
            continue
        if class_name not in name_to_new_id:
            counters["dropped_class"] += 1
            continue
        bnd = obj.find("bndbox")
        if bnd is None:
            counters["dropped_invalid"] += 1
            continue
        try:
            xmin = float(bnd.findtext("xmin", "0") or "0")
            ymin = float(bnd.findtext("ymin", "0") or "0")
            xmax = float(bnd.findtext("xmax", "0") or "0")
            ymax = float(bnd.findtext("ymax", "0") or "0")
        except ValueError:
            counters["dropped_invalid"] += 1
            continue
        box = voc_bndbox_to_xywh(xmin, ymin, xmax, ymax, width, height)
        if box is None:
            counters["dropped_invalid"] += 1
            logger.warning("跳过退化 VOC 框 (%s) %s", xml_path.name, class_name)
            continue
        converted = convert_line_for_format(
            name_to_new_id[class_name],
            [box[0], box[1], box[2], box[3]],
            label_format,
        )
        if converted is None:
            counters["dropped_invalid"] += 1
            continue
        out_line, kind = converted
        lines.append(out_line)
        box_counts[class_name] = box_counts.get(class_name, 0) + 1
        counters["kept"] += 1
        counters[kind] += 1
    return lines, box_counts, counters


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def polygon_to_xywh(coords: list[float]) -> tuple[float, float, float, float] | None:
    if len(coords) < 6 or len(coords) % 2 != 0:
        return None
    xs = [_clip01(v) for v in coords[0::2]]
    ys = [_clip01(v) for v in coords[1::2]]
    xmin = min(xs)
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)
    width = xmax - xmin
    height = ymax - ymin
    if width <= 0.0 or height <= 0.0:
        return None
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    return cx, cy, width, height


def xywh_to_obb_corners(box: tuple[float, float, float, float]) -> list[float]:
    """轴对齐框展开为矩形四顶点：tl, tr, br, bl。"""
    cx, cy, width, height = box
    x1 = _clip01(cx - width / 2.0)
    y1 = _clip01(cy - height / 2.0)
    x2 = _clip01(cx + width / 2.0)
    y2 = _clip01(cy + height / 2.0)
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def _polygon_area(coords: list[float]) -> float:
    xs = coords[0::2]
    ys = coords[1::2]
    area = 0.0
    n = len(xs)
    for i in range(n):
        j = (i + 1) % n
        area += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(area) / 2.0


def normalize_obb_corners(coords: list[float]) -> list[float] | None:
    if len(coords) != 8:
        return None
    clipped = [_clip01(v) for v in coords]
    if _polygon_area(clipped) <= MIN_OBB_AREA:
        return None
    return clipped


def parse_label_parts(line: str) -> tuple[int, list[float]] | None:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        class_id = int(float(parts[0]))
        nums = [float(x) for x in parts[1:]]
    except ValueError:
        return None
    if len(nums) % 2 != 0:
        return None
    return class_id, nums


def format_yolo_detect_line(class_id: int, box: tuple[float, float, float, float]) -> str:
    cx, cy, width, height = box
    return f"{class_id} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}"


def format_yolo_obb_line(class_id: int, corners: list[float]) -> str:
    body = " ".join(f"{v:.6f}" for v in corners)
    return f"{class_id} {body}"


def convert_line_for_format(
    class_id: int,
    nums: list[float],
    label_format: LabelFormat,
) -> tuple[str, str] | None:
    """
    返回 (输出行, 来源标记)。
    来源标记：bbox / polygon / obb4。
    """
    if label_format == "detect":
        if len(nums) == 4:
            cx, cy, width, height = nums
            box = (_clip01(cx), _clip01(cy), _clip01(width), _clip01(height))
            if box[2] <= 0.0 or box[3] <= 0.0:
                return None
            return format_yolo_detect_line(class_id, box), "bbox"
        box = polygon_to_xywh(nums)
        if box is None:
            return None
        return format_yolo_detect_line(class_id, box), "polygon"

    if len(nums) == 8:
        corners = normalize_obb_corners(nums)
        if corners is None:
            return None
        return format_yolo_obb_line(class_id, corners), "obb4"
    if len(nums) == 4:
        cx, cy, width, height = nums
        box = (_clip01(cx), _clip01(cy), _clip01(width), _clip01(height))
        if box[2] <= 0.0 or box[3] <= 0.0:
            return None
        return format_yolo_obb_line(class_id, xywh_to_obb_corners(box)), "bbox"
    box = polygon_to_xywh(nums)
    if box is None:
        return None
    return format_yolo_obb_line(class_id, xywh_to_obb_corners(box)), "polygon"


def convert_label_file(
    label_path: Path,
    local_names: list[str],
    name_to_new_id: dict[str, int],
    *,
    label_format: LabelFormat = "detect",
) -> tuple[list[str], dict[str, int], dict[str, int]]:
    """过滤并重映射一个标签文件。返回 (输出行, 各类框数, 计数)。"""
    counters = {
        "kept": 0,
        "dropped_class": 0,
        "dropped_invalid": 0,
        "bbox": 0,
        "polygon": 0,
        "obb4": 0,
    }
    box_counts: dict[str, int] = {}
    lines: list[str] = []

    text = label_path.read_text(encoding="utf-8")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parsed = parse_label_parts(line)
        if parsed is None:
            counters["dropped_invalid"] += 1
            logger.warning("跳过无法解析的标签行 (%s): %s", label_path.name, line[:80])
            continue
        local_id, nums = parsed
        if local_id < 0 or local_id >= len(local_names):
            counters["dropped_invalid"] += 1
            logger.warning("跳过越界 class id=%s (%s)", local_id, label_path)
            continue
        class_name = local_names[local_id]
        if class_name not in name_to_new_id:
            counters["dropped_class"] += 1
            continue
        converted = convert_line_for_format(name_to_new_id[class_name], nums, label_format)
        if converted is None:
            counters["dropped_invalid"] += 1
            logger.warning("跳过退化框 (%s): %s", label_path.name, line[:80])
            continue
        out_line, kind = converted
        lines.append(out_line)
        box_counts[class_name] = box_counts.get(class_name, 0) + 1
        counters["kept"] += 1
        counters[kind] += 1

    return lines, box_counts, counters


def _out_stem(subset: Path, source_root: Path, image_path: Path, multi_subset: bool) -> str:
    if multi_subset:
        key = _subset_key(subset, source_root).replace("/", "__").replace("\\", "__")
        return f"{key}__{image_path.stem}"
    return image_path.stem


def collect_samples(
    subsets: list[Path],
    source_root: Path,
    train_classes: list[str],
    *,
    keep_empty_labels: bool,
    label_format: LabelFormat = "detect",
) -> tuple[list[SampleRecord], dict[str, int], list[str]]:
    available = collect_available_classes(subsets)
    unknown = [name for name in train_classes if name not in available]
    if unknown:
        raise ValueError(
            "TRAIN_CLASSES 含原始数据中不存在的类别: "
            f"{unknown}；可用类别: {available}"
        )
    name_to_new_id = {name: idx for idx, name in enumerate(train_classes)}
    multi = len(subsets) > 1
    samples: list[SampleRecord] = []
    totals = {
        "label_files": 0,
        "skipped_no_image": 0,
        "skipped_empty": 0,
        "kept": 0,
        "dropped_class": 0,
        "dropped_invalid": 0,
        "bbox": 0,
        "polygon": 0,
        "obb4": 0,
    }

    for subset in subsets:
        local_names = read_classes_txt(subset / CLASSES_FILENAME)
        image_dir = subset / "images"
        label_dir = subset / "labels"
        for label_path in sorted(label_dir.glob("*.txt")):
            if label_path.name.startswith("._") or label_path.name == CLASSES_FILENAME:
                continue
            totals["label_files"] += 1
            image_path = find_image_file(image_dir, label_path.stem)
            if image_path is None:
                totals["skipped_no_image"] += 1
                logger.warning(
                    "未找到对应图像: %s / %s",
                    _subset_key(subset, source_root),
                    label_path.name,
                )
                continue
            lines, box_counts, counters = convert_label_file(
                label_path,
                local_names,
                name_to_new_id,
                label_format=label_format,
            )
            for key in (
                "kept",
                "dropped_class",
                "dropped_invalid",
                "bbox",
                "polygon",
                "obb4",
            ):
                totals[key] += counters[key]
            if not lines and not keep_empty_labels:
                totals["skipped_empty"] += 1
                continue
            samples.append(
                SampleRecord(
                    subset=_subset_key(subset, source_root),
                    image_path=image_path,
                    out_stem=_out_stem(subset, source_root, image_path, multi),
                    lines=lines,
                    box_counts=box_counts,
                )
            )

    if not samples:
        raise ValueError("按 TRAIN_CLASSES 过滤后没有可用样本")
    return samples, totals, available


def collect_voc_samples(
    voc_roots: list[Path],
    source_root: Path,
    train_classes: list[str],
    *,
    keep_empty_labels: bool,
    label_format: LabelFormat = "detect",
) -> tuple[list[SampleRecord], dict[str, int], list[str]]:
    available = collect_available_voc_classes(voc_roots)
    unknown = [name for name in train_classes if name not in available]
    if unknown:
        raise ValueError(
            "TRAIN_CLASSES 含原始数据中不存在的类别: "
            f"{unknown}；可用类别: {available}"
        )
    name_to_new_id = {name: idx for idx, name in enumerate(train_classes)}
    multi = len(voc_roots) > 1
    samples: list[SampleRecord] = []
    totals = {
        "label_files": 0,
        "skipped_no_image": 0,
        "skipped_empty": 0,
        "kept": 0,
        "dropped_class": 0,
        "dropped_invalid": 0,
        "bbox": 0,
        "polygon": 0,
        "obb4": 0,
    }

    for voc_root in voc_roots:
        image_dir = _find_named_subdir(voc_root, VOC_IMAGE_DIR_NAMES)
        annotation_dir = _find_named_subdir(voc_root, VOC_ANNOTATION_DIR_NAMES)
        if image_dir is None or annotation_dir is None:
            continue
        for xml_path in sorted(annotation_dir.glob("*.xml")):
            if xml_path.name.startswith("._"):
                continue
            totals["label_files"] += 1
            image_path = find_image_file(image_dir, xml_path.stem)
            if image_path is None:
                totals["skipped_no_image"] += 1
                logger.warning(
                    "未找到对应图像: %s / %s",
                    _subset_key(voc_root, source_root),
                    xml_path.name,
                )
                continue
            lines, box_counts, counters = convert_voc_xml(
                xml_path,
                image_path,
                name_to_new_id,
                label_format=label_format,
            )
            for key in (
                "kept",
                "dropped_class",
                "dropped_invalid",
                "bbox",
                "polygon",
                "obb4",
            ):
                totals[key] += counters[key]
            if not lines and not keep_empty_labels:
                totals["skipped_empty"] += 1
                continue
            samples.append(
                SampleRecord(
                    subset=_subset_key(voc_root, source_root),
                    image_path=image_path,
                    out_stem=_out_stem(voc_root, source_root, image_path, multi),
                    lines=lines,
                    box_counts=box_counts,
                )
            )

    if not samples:
        raise ValueError("按 TRAIN_CLASSES 过滤后没有可用样本")
    return samples, totals, available


def split_samples(
    samples: list[SampleRecord],
    *,
    val_ratio: float,
    test_ratio: float = 0.0,
    seed: int,
) -> tuple[list[SampleRecord], list[SampleRecord], list[SampleRecord]]:
    if val_ratio < 0 or test_ratio < 0:
        raise ValueError("VAL_RATIO / TEST_RATIO 不能为负")
    if val_ratio + test_ratio >= 1:
        raise ValueError("VAL_RATIO + TEST_RATIO 必须小于 1")
    if val_ratio <= 0 and test_ratio <= 0:
        return list(samples), [], []

    rng = random.Random(seed)
    grouped: dict[str, list[SampleRecord]] = {}
    for sample in samples:
        grouped.setdefault(sample.subset, []).append(sample)

    train: list[SampleRecord] = []
    val: list[SampleRecord] = []
    test: list[SampleRecord] = []
    for items in grouped.values():
        rng.shuffle(items)
        n = len(items)
        n_test = int(round(n * test_ratio)) if test_ratio > 0 else 0
        n_val = int(round(n * val_ratio)) if val_ratio > 0 else 0

        if n_test + n_val >= n:
            overflow = n_test + n_val - max(0, n - 1)
            while overflow > 0 and n_val > 0:
                n_val -= 1
                overflow -= 1
            while overflow > 0 and n_test > 0:
                n_test -= 1
                overflow -= 1

        if val_ratio > 0 and n >= 2 and n_val == 0 and n_test < n - 1:
            n_val = 1
        if test_ratio > 0 and n >= 3 and n_test == 0 and n_val + 1 < n:
            n_test = 1

        test.extend(items[:n_test])
        val.extend(items[n_test : n_test + n_val])
        train.extend(items[n_test + n_val :])
    return train, val, test


def write_data_yaml(
    output_dir: Path,
    class_names: list[str],
    *,
    has_val: bool,
    has_test: bool,
) -> None:
    val_rel = "images/val" if has_val else "images/train"
    lines = [
        f"path: {output_dir.resolve()}",
        "train: images/train",
        f"val: {val_rel}",
    ]
    if has_test:
        lines.append("test: images/test")
    lines.extend(
        [
            f"nc: {len(class_names)}",
            "names:",
            *[f"  - {name}" for name in class_names],
            "",
        ]
    )
    (output_dir / "data.yaml").write_text("\n".join(lines), encoding="utf-8")


def _reset_split_dirs(output_dir: Path) -> None:
    for name in ("images", "labels"):
        path = output_dir / name
        if path.is_dir():
            shutil.rmtree(path)


def _write_split(output_dir: Path, split: str, samples: list[SampleRecord], copy_images: bool) -> None:
    image_dir = output_dir / "images" / split
    label_dir = output_dir / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        suffix = sample.image_path.suffix
        dest_image = image_dir / f"{sample.out_stem}{suffix}"
        dest_label = label_dir / f"{sample.out_stem}.txt"
        if copy_images:
            shutil.copy2(sample.image_path, dest_image)
        else:
            if dest_image.exists() or dest_image.is_symlink():
                dest_image.unlink()
            dest_image.symlink_to(sample.image_path.resolve())
        body = "\n".join(sample.lines)
        dest_label.write_text(body + ("\n" if body else ""), encoding="utf-8")


def _sum_box_counts(samples: list[SampleRecord], train_classes: list[str]) -> dict[str, int]:
    counts = {name: 0 for name in train_classes}
    for sample in samples:
        for name, n in sample.box_counts.items():
            counts[name] = counts.get(name, 0) + n
    return counts


def prepare_dataset(
    source_root: str | Path,
    output_dir: str | Path,
    train_classes: list[str],
    *,
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
    seed: int = 42,
    keep_empty_labels: bool = False,
    copy_images: bool = True,
    label_format: LabelFormat = "detect",
) -> dict[str, Any]:
    """按配置类别过滤原始导出包或扁平 VOC，写出 YOLO11 检测或 OBB 训练目录。

    train_classes 为空时，自动使用原始数据中 classes.txt / predefined_classes.txt 的全部类别。
    val_ratio / test_ratio 为各子集内占总样本的比例，三者之和须小于 1。
    """
    if label_format not in {"detect", "obb"}:
        raise ValueError(f"不支持的 label_format: {label_format}")

    names = [str(x).strip() for x in train_classes if str(x).strip()]
    if len(names) != len(set(names)):
        raise ValueError(f"TRAIN_CLASSES 存在重复: {train_classes}")

    source_root = Path(source_root).resolve()
    output_dir = Path(output_dir)
    if output_dir.resolve() == source_root:
        raise ValueError(
            f"OUTPUT_DIR 不能与原始数据根相同（会污染源目录）: {source_root}"
        )

    source_kind, subsets = discover_source(source_root)
    if source_kind == "yolo":
        available_fn = collect_available_classes
        collect_fn = collect_samples
    else:
        available_fn = collect_available_voc_classes
        collect_fn = collect_voc_samples
        if output_dir.resolve() in {p.resolve() for p in subsets}:
            raise ValueError(
                f"OUTPUT_DIR 不能与 VOC 源目录相同（会污染 JPEGImages/Annotations）: {output_dir}"
            )

    if not names:
        names = available_fn(subsets)
        if not names:
            raise ValueError("原始数据中没有可用类别")
        logger.info("TRAIN_CLASSES 为空，使用全部可用类别: %s", names)
    samples, totals, available = collect_fn(
        subsets,
        source_root,
        names,
        keep_empty_labels=keep_empty_labels,
        label_format=label_format,
    )
    train_samples, val_samples, test_samples = split_samples(
        samples,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    if not train_samples:
        raise ValueError("划分后训练集为空，请降低 VAL_RATIO / TEST_RATIO")

    output_dir.mkdir(parents=True, exist_ok=True)
    _reset_split_dirs(output_dir)
    _write_split(output_dir, "train", train_samples, copy_images)
    if val_samples:
        _write_split(output_dir, "val", val_samples, copy_images)
    if test_samples:
        _write_split(output_dir, "test", test_samples, copy_images)

    (output_dir / CLASSES_FILENAME).write_text("\n".join(names) + "\n", encoding="utf-8")
    write_data_yaml(
        output_dir,
        names,
        has_val=bool(val_samples),
        has_test=bool(test_samples),
    )

    empty_classes = [
        name for name, n in _sum_box_counts(samples, names).items() if n == 0
    ]
    if empty_classes:
        logger.warning("配置的类别在过滤后没有框: %s", empty_classes)

    stats: dict[str, Any] = {
        "source_root": str(source_root.resolve()),
        "output_dir": str(output_dir.resolve()),
        "source_kind": source_kind,
        "label_format": label_format,
        "subsets": [_subset_key(p, source_root) for p in subsets],
        "available_classes": available,
        "train_classes": names,
        "images_train": len(train_samples),
        "images_val": len(val_samples),
        "images_test": len(test_samples),
        "boxes_by_class": _sum_box_counts(samples, names),
        "empty_classes": empty_classes,
        "kept_obb4": totals["obb4"],
        "converted_polygons": totals["polygon"],
        "already_bbox": totals["bbox"],
        "dropped_other_class": totals["dropped_class"],
        "dropped_invalid": totals["dropped_invalid"],
        "skipped_no_image": totals["skipped_no_image"],
        "skipped_empty": totals["skipped_empty"],
        "keep_empty_labels": keep_empty_labels,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "seed": seed,
    }
    (output_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "数据准备完成 kind=%s format=%s output=%s train=%s val=%s test=%s boxes=%s obb4=%s polygon=%s",
        source_kind,
        label_format,
        output_dir,
        stats["images_train"],
        stats["images_val"],
        stats["images_test"],
        stats["boxes_by_class"],
        stats["kept_obb4"],
        stats["converted_polygons"],
    )
    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    SCANLY_ROOT = Path(__file__).resolve().parents[1]
    # 支持单层（cam123-0820/BandBroken/...）或多层（cam-all-0901/cam2/BandBroken/...）自动合并
    SOURCE_DATA_ROOT = SCANLY_ROOT / "样本数据" / "cam-all-0901"
    OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "cam-all-0901"
    TRAIN_CLASSES = [
        # "BandBroken",
        # "BandBroken2",
        # "Below",
        # "Scratch",
        # "ResidualTape",
        # "Shorter",
    ]
    VAL_RATIO = 0.2
    TEST_RATIO = 0.1
    SEED = 42
    KEEP_EMPTY_LABELS = False
    COPY_IMAGES = True

    result = prepare_dataset(
        source_root=SOURCE_DATA_ROOT,
        output_dir=OUTPUT_DIR,
        train_classes=TRAIN_CLASSES,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=SEED,
        keep_empty_labels=KEEP_EMPTY_LABELS,
        copy_images=COPY_IMAGES,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
