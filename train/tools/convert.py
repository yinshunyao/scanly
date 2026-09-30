#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""将缺陷检测专用标注或多层 YOLO 导出目录转换为 YOLO11、LabelMe 或 Pascal VOC 格式。"""
from __future__ import annotations

import json
import logging
import shutil
import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".bmp", ".png", ".tif", ".tiff", ".webp"}
OutputFormat = Literal["yolo", "labelme", "voc"]


def read_image_size(image_path: Path) -> tuple[int, int]:
    """读取图像宽高，不依赖 PIL。"""
    suffix = image_path.suffix.lower()
    with image_path.open("rb") as fp:
        if suffix == ".bmp":
            fp.seek(18)
            width, height = struct.unpack("<ii", fp.read(8))
            return abs(width), abs(height)

        if suffix in {".jpg", ".jpeg"}:
            data = fp.read()
            index = 2
            while index + 9 < len(data):
                if data[index] != 0xFF:
                    break
                marker = data[index + 1]
                if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                    height = struct.unpack(">H", data[index + 5 : index + 7])[0]
                    width = struct.unpack(">H", data[index + 7 : index + 9])[0]
                    return width, height
                segment_len = struct.unpack(">H", data[index + 2 : index + 4])[0]
                index += 2 + segment_len

        if suffix == ".png" and data[:8] == b"\x89PNG\r\n\x1a\n":
            width, height = struct.unpack(">II", data[16:24])
            return width, height

    raise ValueError(f"无法解析图像尺寸: {image_path}")


def load_categories(categories_path: Path) -> tuple[list[str], dict[int, str]]:
    with categories_path.open(encoding="utf-8") as fp:
        payload = json.load(fp)

    categories = payload.get("categories", [])
    id_to_name: dict[int, str] = {}
    names: list[str] = []
    for item in sorted(categories, key=lambda x: int(x["id"])):
        cid = int(item["id"])
        name = str(item["name"])
        id_to_name[cid] = name
        names.append(name)
    return names, id_to_name


def resolve_dataset_layout(input_dir: Path) -> tuple[Path, Path, Path]:
    """返回 (categories_path, image_dir, label_dir)。"""
    categories_path = input_dir / "Categories.json"
    if not categories_path.is_file():
        raise FileNotFoundError(f"未找到类别文件: {categories_path}")

    image_dir = input_dir / "inputImages"
    label_dir = input_dir / "labelInfo"
    if image_dir.is_dir() and label_dir.is_dir():
        return categories_path, image_dir, label_dir

    return categories_path, input_dir, input_dir


def iter_label_files(label_dir: Path) -> list[Path]:
    files = []
    for path in sorted(label_dir.glob("*.json")):
        if path.name.startswith("._"):
            continue
        if path.name.lower() == "categories.json":
            continue
        files.append(path)
    return files


def label_json_to_image_name(label_path: Path) -> str:
    name = label_path.name
    if name.endswith(".json"):
        return name[:-5]
    return name


def find_image_file(image_dir: Path, file_name: str) -> Path | None:
    direct = image_dir / file_name
    if direct.is_file():
        return direct

    stem = Path(file_name).stem
    for suffix in IMAGE_SUFFIXES:
        candidate = image_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def normalize_bbox(points: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    if len(points) != 4:
        raise ValueError(f"矩形框需要 4 个坐标，实际为 {len(points)}: {points}")

    x1, y1, x2, y2 = [float(v) for v in points]
    left = max(0.0, min(x1, x2))
    right = min(float(width), max(x1, x2))
    top = max(0.0, min(y1, y2))
    bottom = min(float(height), max(y1, y2))
    return left, top, right, bottom


def bbox_to_yolo_line(
    category_id: int,
    points: list[float],
    width: int,
    height: int,
) -> str:
    x1, y1, x2, y2 = normalize_bbox(points, width, height)
    box_w = max(0.0, x2 - x1)
    box_h = max(0.0, y2 - y1)
    if box_w <= 0 or box_h <= 0:
        return ""

    cx = (x1 + x2) / 2.0 / width
    cy = (y1 + y2) / 2.0 / height
    nw = box_w / width
    nh = box_h / height
    return f"{category_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def parse_label_json(label_path: Path) -> dict[str, Any]:
    with label_path.open(encoding="utf-8") as fp:
        return json.load(fp)


def build_labelme_json(
    *,
    image_name: str,
    width: int,
    height: int,
    annotations: list[dict[str, Any]],
    id_to_name: dict[int, str],
) -> dict[str, Any]:
    shapes = []
    for ann in annotations:
        ann_type = ann.get("type")
        points = ann.get("points", [])
        if ann_type != 3 or len(points) != 4:
            logger.warning("跳过非矩形标注 type=%s points=%s", ann_type, points)
            continue

        x1, y1, x2, y2 = normalize_bbox(points, width, height)
        if x2 <= x1 or y2 <= y1:
            continue

        category_id = int(ann["category_id"])
        label = id_to_name.get(category_id, str(category_id))
        shapes.append(
            {
                "label": label,
                "points": [[x1, y1], [x2, y2]],
                "group_id": None,
                "description": "",
                "shape_type": "rectangle",
                "flags": {},
                "mask": None,
            }
        )

    return {
        "version": "5.5.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }


def bbox_to_voc_ints(
    points: list[float],
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = normalize_bbox(points, width, height)
    if x2 <= x1 or y2 <= y1:
        return None

    xmin = max(0, min(width, int(round(x1))))
    ymin = max(0, min(height, int(round(y1))))
    xmax = max(0, min(width, int(round(x2))))
    ymax = max(0, min(height, int(round(y2))))
    if xmax <= xmin or ymax <= ymin:
        return None
    return xmin, ymin, xmax, ymax


def _xml_tostring(root: ET.Element) -> str:
    elem = ET.ElementTree(root)
    try:
        ET.indent(elem, space="    ")
    except AttributeError:
        pass
    return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="unicode") + "\n"


def build_voc_xml(
    *,
    image_name: str,
    image_path: Path,
    folder_name: str,
    width: int,
    height: int,
    annotations: list[dict[str, Any]],
    id_to_name: dict[int, str],
) -> str:
    """生成 LabelImg 可打开的 Pascal VOC XML。"""
    root = ET.Element("annotation")

    ET.SubElement(root, "folder").text = folder_name
    ET.SubElement(root, "filename").text = image_name
    ET.SubElement(root, "path").text = str(image_path.resolve())

    source = ET.SubElement(root, "source")
    ET.SubElement(source, "database").text = "Unknown"

    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"

    ET.SubElement(root, "segmented").text = "0"

    for ann in annotations:
        ann_type = ann.get("type")
        points = ann.get("points", [])
        if ann_type != 3 or len(points) != 4:
            logger.warning("跳过非矩形标注 type=%s points=%s", ann_type, points)
            continue

        bbox = bbox_to_voc_ints(points, width, height)
        if bbox is None:
            continue

        xmin, ymin, xmax, ymax = bbox
        category_id = int(ann["category_id"])
        label = id_to_name.get(category_id, str(category_id))

        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = label
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(xmin)
        ET.SubElement(bndbox, "ymin").text = str(ymin)
        ET.SubElement(bndbox, "xmax").text = str(xmax)
        ET.SubElement(bndbox, "ymax").text = str(ymax)

    return _xml_tostring(root)


def write_yolo_data_yaml(output_dir: Path, class_names: list[str]) -> None:
    yaml_text = "\n".join(
        [
            f"path: {output_dir.resolve()}",
            "train: images/train",
            "val: images/train",
            f"nc: {len(class_names)}",
            "names:",
            *[f"  - {name}" for name in class_names],
            "",
        ]
    )
    (output_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")


def write_voc_predefined_classes(output_dir: Path, class_names: list[str]) -> None:
    """写入 LabelImg 预定义类别文件。"""
    (output_dir / "predefined_classes.txt").write_text(
        "\n".join(class_names) + "\n",
        encoding="utf-8",
    )


def is_yolo_package(path: Path) -> bool:
    return path.is_dir() and (path / "images").is_dir() and (path / "labels").is_dir()


def discover_yolo_packages(input_dir: Path) -> list[Path]:
    """递归发现含 images/ 与 labels/ 的 YOLO 包；进入某包后不再向下搜。"""
    input_dir = input_dir.resolve()
    if not input_dir.is_dir():
        return []
    if is_yolo_package(input_dir):
        return [input_dir]

    packages: list[Path] = []
    stack = [input_dir]
    while stack:
        current = stack.pop()
        if is_yolo_package(current):
            packages.append(current)
            continue
        for child in sorted(current.iterdir()):
            if child.is_dir() and not child.name.startswith("."):
                stack.append(child)
    packages.sort(key=lambda p: str(p))
    return packages


def read_classes_txt(path: Path) -> list[str]:
    names: list[str] = []
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        name = raw.strip()
        if not name or name.startswith("#"):
            continue
        names.append(name)
    return names


def collect_global_class_names(packages: list[Path]) -> list[str]:
    """全局类别名：优先用条目最多的 notes.json，再并入各包 classes.txt 中尚未出现的名称。"""
    best: list[str] = []
    for package in packages:
        notes = package / "notes.json"
        if not notes.is_file():
            continue
        try:
            names, _ = load_categories(notes)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        if len(names) > len(best):
            best = names

    seen = set(best)
    ordered = list(best)
    for package in packages:
        classes_path = package / "classes.txt"
        if not classes_path.is_file():
            continue
        for name in read_classes_txt(classes_path):
            if name not in seen:
                ordered.append(name)
                seen.add(name)
    if not ordered:
        raise FileNotFoundError("YOLO 包中未找到 classes.txt 或 notes.json 类别表")
    return ordered


def load_package_id_to_name(package: Path) -> dict[int, str]:
    classes_path = package / "classes.txt"
    if classes_path.is_file():
        names = read_classes_txt(classes_path)
        if names:
            return {index: name for index, name in enumerate(names)}
    notes = package / "notes.json"
    if notes.is_file():
        _, id_to_name = load_categories(notes)
        return id_to_name
    raise FileNotFoundError(f"未找到类别文件: {package / 'classes.txt'}")


def yolo_xywh_to_points(cx: float, cy: float, nw: float, nh: float, width: int, height: int) -> list[float]:
    box_w = nw * float(width)
    box_h = nh * float(height)
    x1 = cx * float(width) - box_w / 2.0
    y1 = cy * float(height) - box_h / 2.0
    return [x1, y1, x1 + box_w, y1 + box_h]


def parse_yolo_txt(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    if not label_path.is_file():
        return []

    rows: list[tuple[int, float, float, float, float]] = []
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            logger.warning("跳过非检测行 %s: %s", label_path, line)
            continue
        try:
            cid = int(float(parts[0]))
            cx, cy, nw, nh = (float(v) for v in parts[1:])
        except ValueError:
            logger.warning("跳过无法解析的检测行 %s: %s", label_path, line)
            continue
        rows.append((cid, cx, cy, nw, nh))
    return rows


def iter_yolo_images(image_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(image_dir.iterdir()):
        if not path.is_file() or path.name.startswith("._"):
            continue
        if path.suffix.lower() in IMAGE_SUFFIXES:
            files.append(path)
    return files


def resolve_flat_image_name(
    image_path: Path,
    package: Path,
    input_root: Path,
    used_names: dict[str, Path],
) -> str | None:
    """扁平输出文件名。同名同大小视为重复返回 None；同名不同大小加相对路径前缀。"""
    name = image_path.name
    previous = used_names.get(name)
    if previous is None:
        used_names[name] = image_path
        return name
    if previous.stat().st_size == image_path.stat().st_size:
        return None

    try:
        rel = package.relative_to(input_root)
    except ValueError:
        rel = Path(package.name)
    prefix = "__".join(rel.parts)
    candidate = f"{prefix}__{name}"
    if candidate in used_names:
        stem = image_path.stem
        suffix = image_path.suffix
        index = 2
        while f"{prefix}__{stem}_{index}{suffix}" in used_names:
            index += 1
        candidate = f"{prefix}__{stem}_{index}{suffix}"
    used_names[candidate] = image_path
    return candidate


def yolo_rows_to_annotations(
    rows: list[tuple[int, float, float, float, float]],
    *,
    local_id_to_name: dict[int, str],
    name_to_global_id: dict[str, int],
    width: int,
    height: int,
    label_path: Path,
) -> list[dict[str, Any]]:
    annotations: list[dict[str, Any]] = []
    for cid, cx, cy, nw, nh in rows:
        name = local_id_to_name.get(cid)
        if name is None:
            logger.warning("跳过未知类别 id=%s (%s)", cid, label_path)
            continue
        global_id = name_to_global_id.get(name)
        if global_id is None:
            logger.warning("跳过未纳入全局类别表的名称 %s (%s)", name, label_path)
            continue
        annotations.append(
            {
                "type": 3,
                "category_id": global_id,
                "points": yolo_xywh_to_points(cx, cy, nw, nh, width, height),
            }
        )
    return annotations


def convert_yolo_packages(
    input_dir: Path,
    packages: list[Path],
    output_dir: Path,
    output_format: OutputFormat,
    copy_images: bool,
) -> dict[str, int]:
    class_names = collect_global_class_names(packages)
    id_to_name = {index: name for index, name in enumerate(class_names)}
    name_to_global_id = {name: index for index, name in enumerate(class_names)}
    stats = {"images": 0, "labels": 0, "skipped": 0, "negatives": 0, "boxes": 0, "duplicates": 0}

    image_out_dir: Path | None = None
    label_out_dir: Path | None = None
    if output_format == "yolo":
        image_out_dir = output_dir / "images" / "train"
        label_out_dir = output_dir / "labels" / "train"
        image_out_dir.mkdir(parents=True, exist_ok=True)
        label_out_dir.mkdir(parents=True, exist_ok=True)
    elif output_format == "voc":
        image_out_dir = output_dir / "JPEGImages"
        label_out_dir = output_dir / "Annotations"
        image_out_dir.mkdir(parents=True, exist_ok=True)
        label_out_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    used_names: dict[str, Path] = {}
    input_root = input_dir.resolve()

    for package in packages:
        local_id_to_name = load_package_id_to_name(package)
        image_dir = package / "images"
        label_dir = package / "labels"
        for image_path in iter_yolo_images(image_dir):
            image_name = resolve_flat_image_name(image_path, package, input_root, used_names)
            if image_name is None:
                stats["duplicates"] += 1
                continue

            try:
                width, height = read_image_size(image_path)
            except ValueError as exc:
                logger.warning("%s", exc)
                stats["skipped"] += 1
                continue

            label_path = label_dir / f"{image_path.stem}.txt"
            rows = parse_yolo_txt(label_path)
            annotations = yolo_rows_to_annotations(
                rows,
                local_id_to_name=local_id_to_name,
                name_to_global_id=name_to_global_id,
                width=width,
                height=height,
                label_path=label_path,
            )
            if not annotations:
                stats["negatives"] += 1

            out_stem = Path(image_name).stem
            if output_format == "yolo":
                assert image_out_dir is not None and label_out_dir is not None
                if copy_images:
                    shutil.copy2(image_path, image_out_dir / image_name)
                yolo_lines = []
                for ann in annotations:
                    line = bbox_to_yolo_line(int(ann["category_id"]), ann.get("points", []), width, height)
                    if line:
                        yolo_lines.append(line)
                        stats["boxes"] += 1
                (label_out_dir / f"{out_stem}.txt").write_text(
                    "\n".join(yolo_lines) + ("\n" if yolo_lines else ""),
                    encoding="utf-8",
                )
            elif output_format == "voc":
                assert image_out_dir is not None and label_out_dir is not None
                image_out_path = image_out_dir / image_name
                if copy_images:
                    shutil.copy2(image_path, image_out_path)
                else:
                    image_out_path = image_path
                box_count = 0
                for ann in annotations:
                    if bbox_to_voc_ints(ann.get("points", []), width, height) is not None:
                        box_count += 1
                stats["boxes"] += box_count
                voc_xml = build_voc_xml(
                    image_name=image_name,
                    image_path=image_out_path,
                    folder_name="JPEGImages",
                    width=width,
                    height=height,
                    annotations=annotations,
                    id_to_name=id_to_name,
                )
                (label_out_dir / f"{out_stem}.xml").write_text(voc_xml, encoding="utf-8")
            else:
                if copy_images:
                    shutil.copy2(image_path, output_dir / image_name)
                labelme_payload = build_labelme_json(
                    image_name=image_name,
                    width=width,
                    height=height,
                    annotations=annotations,
                    id_to_name=id_to_name,
                )
                stats["boxes"] += len(labelme_payload["shapes"])
                (output_dir / f"{out_stem}.json").write_text(
                    json.dumps(labelme_payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )

            stats["images"] += 1
            stats["labels"] += 1

    if output_format == "yolo":
        write_yolo_data_yaml(output_dir, class_names)
    elif output_format == "voc":
        write_voc_predefined_classes(output_dir, class_names)

    logger.info(
        "转换完成 format=%s output=%s images=%s boxes=%s negatives=%s skipped=%s duplicates=%s packages=%s",
        output_format,
        output_dir,
        stats["images"],
        stats["boxes"],
        stats["negatives"],
        stats["skipped"],
        stats["duplicates"],
        len(packages),
    )
    return stats


def convert_dataset(
    input_dir: str | Path,
    output_dir: str | Path,
    output_format: OutputFormat = "yolo",
    copy_images: bool = True,
) -> dict[str, int]:
    """
    将缺陷检测标注或多层 YOLO 导出目录转换为 YOLO11、LabelMe 或 Pascal VOC 格式。

    若根目录有 Categories.json，走客户 JSON 布局；否则递归发现 images/+labels/ YOLO 包。
    VOC 输出扁平写入 JPEGImages/ 与 Annotations/。
    返回统计信息：images / labels / skipped / negatives / boxes，YOLO 包另含 duplicates。
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if output_format not in {"yolo", "labelme", "voc"}:
        raise ValueError(f"不支持的输出格式: {output_format}")

    categories_path = input_dir / "Categories.json"
    if not categories_path.is_file():
        packages = discover_yolo_packages(input_dir)
        if not packages:
            raise FileNotFoundError(
                f"未找到类别文件 {categories_path}，也未发现 YOLO 包（images/ + labels/）: {input_dir}"
            )
        return convert_yolo_packages(
            input_dir=input_dir,
            packages=packages,
            output_dir=output_dir,
            output_format=output_format,
            copy_images=copy_images,
        )

    categories_path, image_dir, label_dir = resolve_dataset_layout(input_dir)
    class_names, id_to_name = load_categories(categories_path)
    label_files = iter_label_files(label_dir)

    stats = {"images": 0, "labels": 0, "skipped": 0, "negatives": 0, "boxes": 0, "duplicates": 0}

    image_out_dir: Path | None = None
    label_out_dir: Path | None = None
    if output_format == "yolo":
        image_out_dir = output_dir / "images" / "train"
        label_out_dir = output_dir / "labels" / "train"
        image_out_dir.mkdir(parents=True, exist_ok=True)
        label_out_dir.mkdir(parents=True, exist_ok=True)
    elif output_format == "voc":
        image_out_dir = output_dir / "JPEGImages"
        label_out_dir = output_dir / "Annotations"
        image_out_dir.mkdir(parents=True, exist_ok=True)
        label_out_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    for label_path in label_files:
        payload = parse_label_json(label_path)
        images = payload.get("images") or []
        if not images:
            logger.warning("跳过无 images 字段的标注: %s", label_path)
            stats["skipped"] += 1
            continue

        image_info = images[0]
        file_name = str(image_info.get("file_name") or label_json_to_image_name(label_path))
        image_path = find_image_file(image_dir, file_name)
        if image_path is None:
            logger.warning("未找到图像文件: %s (来自 %s)", file_name, label_path)
            stats["skipped"] += 1
            continue

        try:
            width, height = read_image_size(image_path)
        except ValueError as exc:
            logger.warning("%s", exc)
            stats["skipped"] += 1
            continue

        json_w = int(image_info.get("width") or 0)
        json_h = int(image_info.get("height") or 0)
        if json_w and json_h and (json_w != width or json_h != height):
            logger.info(
                "尺寸校正 %s: json=%sx%s, actual=%sx%s",
                file_name,
                json_w,
                json_h,
                width,
                height,
            )

        annotations = payload.get("annotations") or []
        no_target = bool(image_info.get("NoTarget")) or not annotations
        if no_target:
            stats["negatives"] += 1

        image_name = image_path.name
        if output_format == "yolo":
            if copy_images:
                shutil.copy2(image_path, image_out_dir / image_name)

            yolo_lines = []
            for ann in annotations:
                if ann.get("type") != 3:
                    logger.warning("跳过非矩形标注 type=%s (%s)", ann.get("type"), label_path)
                    continue
                line = bbox_to_yolo_line(int(ann["category_id"]), ann.get("points", []), width, height)
                if line:
                    yolo_lines.append(line)
                    stats["boxes"] += 1

            label_out_path = label_out_dir / f"{image_path.stem}.txt"
            label_out_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")
        elif output_format == "voc":
            assert image_out_dir is not None and label_out_dir is not None
            image_out_path = image_out_dir / image_name
            if copy_images:
                shutil.copy2(image_path, image_out_path)
            else:
                image_out_path = image_path

            box_count = 0
            for ann in annotations:
                if ann.get("type") != 3:
                    logger.warning("跳过非矩形标注 type=%s (%s)", ann.get("type"), label_path)
                    continue
                if bbox_to_voc_ints(ann.get("points", []), width, height) is not None:
                    box_count += 1
            stats["boxes"] += box_count

            voc_xml = build_voc_xml(
                image_name=image_name,
                image_path=image_out_path,
                folder_name="JPEGImages",
                width=width,
                height=height,
                annotations=annotations,
                id_to_name=id_to_name,
            )
            voc_path = label_out_dir / f"{image_path.stem}.xml"
            voc_path.write_text(voc_xml, encoding="utf-8")
        else:
            if copy_images:
                shutil.copy2(image_path, output_dir / image_name)

            labelme_payload = build_labelme_json(
                image_name=image_name,
                width=width,
                height=height,
                annotations=annotations,
                id_to_name=id_to_name,
            )
            stats["boxes"] += len(labelme_payload["shapes"])
            labelme_path = output_dir / f"{image_path.stem}.json"
            labelme_path.write_text(
                json.dumps(labelme_payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

        stats["images"] += 1
        stats["labels"] += 1

    if output_format == "yolo":
        write_yolo_data_yaml(output_dir, class_names)
    elif output_format == "voc":
        write_voc_predefined_classes(output_dir, class_names)

    logger.info(
        "转换完成 format=%s output=%s images=%s boxes=%s negatives=%s skipped=%s",
        output_format,
        output_dir,
        stats["images"],
        stats["boxes"],
        stats["negatives"],
        stats["skipped"],
    )
    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # 修改以下变量后直接运行
    INPUT_DIR = Path("/Users/shunyaoyin/Documents/code/ai-company/scanly/样本数据/cam-all-0901")
    output_dir = Path(str(INPUT_DIR) + "_voc")
    OUTPUT_DIR_YOLO = output_dir / "yolo"
    OUTPUT_DIR_LABELME = output_dir / "labelme"
    OUTPUT_DIR_VOC = output_dir / "voc"

    # 可选: "yolo" / "labelme" / "voc"
    OUTPUT_FORMAT: OutputFormat = "voc"
    COPY_IMAGES = True

    target_dirs = {
        "yolo": OUTPUT_DIR_YOLO,
        "labelme": OUTPUT_DIR_LABELME,
        "voc": OUTPUT_DIR_VOC,
    }
    target_dir = target_dirs[OUTPUT_FORMAT]
    result = convert_dataset(
        input_dir=INPUT_DIR,
        output_dir=target_dir,
        output_format=OUTPUT_FORMAT,
        copy_images=COPY_IMAGES,
    )
    print(result)
