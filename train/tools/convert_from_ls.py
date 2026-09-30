#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""将 Label Studio 导出的矩形框标注转换为 YOLO11 或 LabelImg（Pascal VOC）格式。"""
from __future__ import annotations

import json
import logging
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable, Literal

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import convert as cvt

logger = logging.getLogger(__name__)

OutputFormat = Literal["yolo", "voc"]
UUID_PREFIX_RE = re.compile(r"^[0-9a-fA-F]{8}-")


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as fp:
        return json.load(fp)


def _as_task_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        return [payload]
    raise ValueError(f"无法识别的 Label Studio JSON 结构: {type(payload)}")


def iter_ls_tasks(
    input_path: Path,
    *,
    recursive: bool = True,
) -> Iterable[tuple[Path | None, dict[str, Any]]]:
    """
    产出 (来源文件, task)。

    - 输入为目录：遍历其中 *.json（recursive=True 时含所有子目录）
    - 输入为文件：读取单个导出 JSON
    """
    if input_path.is_dir():
        pattern = "**/*.json" if recursive else "*.json"
        for path in sorted(input_path.glob(pattern)):
            if not path.is_file():
                continue
            if path.name.startswith("._") or path.name.lower() == "categories.json":
                continue
            try:
                payload = _load_json(path)
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("跳过无法解析的 JSON: %s (%s)", path, exc)
                continue
            for task in _as_task_list(payload):
                yield path, task
        return

    if input_path.is_file():
        payload = _load_json(input_path)
        for task in _as_task_list(payload):
            yield input_path, task
        return

    raise FileNotFoundError(f"输入路径不存在: {input_path}")


def _image_name_candidates(file_name: str) -> list[str]:
    """本地图名候选：精确名 + LS 常把日期与时间之间的空格写成下划线。"""
    names = [file_name]
    if "_" in file_name:
        alt = file_name.replace("_", " ", 1)
        if alt not in names:
            names.append(alt)
    return names


def find_image_for_task(
    image_root: Path,
    file_name: str,
    *,
    source_path: Path | None = None,
    recursive: bool = True,
) -> Path | None:
    """先找标注 JSON 同目录，再在 image_root（可递归）下按文件名匹配。"""
    search_dirs: list[Path] = []
    if source_path is not None:
        search_dirs.append(source_path.parent)
    search_dirs.append(image_root)

    seen: set[Path] = set()
    for directory in search_dirs:
        directory = directory.resolve()
        if directory in seen:
            continue
        seen.add(directory)
        for name in _image_name_candidates(file_name):
            found = cvt.find_image_file(directory, name)
            if found is not None:
                return found

    if not recursive:
        return None

    for name in _image_name_candidates(file_name):
        direct = image_root / name
        if direct.is_file():
            return direct
        matches = [
            p
            for p in image_root.rglob(name)
            if p.is_file() and not p.name.startswith("._")
        ]
        if matches:
            return sorted(matches)[0]
        stem = Path(name).stem
        for suffix in cvt.IMAGE_SUFFIXES:
            matches = [
                p
                for p in image_root.rglob(f"{stem}{suffix}")
                if p.is_file() and not p.name.startswith("._")
            ]
            if matches:
                return sorted(matches)[0]
    return None


def image_name_from_ls_uri(uri: str) -> str:
    name = Path(str(uri).split("?", 1)[0]).name
    if UUID_PREFIX_RE.match(name):
        return name[9:]
    return name


def resolve_image_name(task: dict[str, Any], source_path: Path | None) -> str:
    if source_path is not None and source_path.name.lower().endswith(".json"):
        # xxx.jpg.json / xxx.bmp.json → xxx.jpg / xxx.bmp
        candidate = source_path.name[:-5]
        if Path(candidate).suffix.lower() in cvt.IMAGE_SUFFIXES:
            return candidate

    data = task.get("data") or {}
    for key in ("image", "img", "Image"):
        if data.get(key):
            return image_name_from_ls_uri(str(data[key]))
    raise ValueError("任务缺少 data.image，且无法从来源文件名推断图像名")


def extract_rectangle_results(task: dict[str, Any]) -> list[dict[str, Any]]:
    annotations = task.get("annotations") or []
    if not annotations:
        return []
    # Label Studio 导出通常取最后一次提交；这里取列表最后一项
    ann = annotations[-1]
    results = ann.get("result") or []
    return [item for item in results if isinstance(item, dict)]


def ls_value_to_points(value: dict[str, Any], width: int, height: int) -> list[float]:
    x = float(value["x"]) / 100.0 * width
    y = float(value["y"]) / 100.0 * height
    w = float(value["width"]) / 100.0 * width
    h = float(value["height"]) / 100.0 * height
    return [x, y, x + w, y + h]


def result_image_size(result_items: list[dict[str, Any]]) -> tuple[int, int] | None:
    for item in result_items:
        w = int(item.get("original_width") or 0)
        h = int(item.get("original_height") or 0)
        if w > 0 and h > 0:
            return w, h
    return None


def collect_label_names(input_path: Path, *, recursive: bool = True) -> list[str]:
    names: set[str] = set()
    for _, task in iter_ls_tasks(input_path, recursive=recursive):
        for item in extract_rectangle_results(task):
            if item.get("type") != "rectanglelabels":
                continue
            labels = (item.get("value") or {}).get("rectanglelabels") or []
            for label in labels:
                names.add(str(label))
    return sorted(names)


def build_class_maps(
    categories_path: Path | None,
    input_path: Path,
    *,
    recursive: bool = True,
) -> tuple[list[str], dict[str, int]]:
    if categories_path is not None and categories_path.is_file():
        class_names, id_to_name = cvt.load_categories(categories_path)
        name_to_id = {name: cid for cid, name in id_to_name.items()}
        return class_names, name_to_id

    class_names = collect_label_names(input_path, recursive=recursive)
    name_to_id = {name: idx for idx, name in enumerate(class_names)}
    return class_names, name_to_id


def results_to_annotations(
    result_items: list[dict[str, Any]],
    *,
    width: int,
    height: int,
    name_to_id: dict[str, int],
) -> tuple[list[dict[str, Any]], int]:
    """转为 convert.py 可消费的 annotations（type=3, points=[x1,y1,x2,y2]）。"""
    annotations: list[dict[str, Any]] = []
    skipped_unknown = 0
    for item in result_items:
        if item.get("type") != "rectanglelabels":
            logger.warning("跳过非 rectanglelabels: type=%s", item.get("type"))
            continue

        value = item.get("value") or {}
        labels = value.get("rectanglelabels") or []
        if not labels:
            continue
        label = str(labels[0])
        if label not in name_to_id:
            logger.warning("未知类别，跳过: %s", label)
            skipped_unknown += 1
            continue

        rotation = float(value.get("rotation") or 0)
        if abs(rotation) > 1e-6:
            logger.warning("存在旋转框 rotation=%s，按轴对齐框处理", rotation)

        points = ls_value_to_points(value, width, height)
        annotations.append(
            {
                "category_id": name_to_id[label],
                "type": 3,
                "points": points,
            }
        )
    return annotations, skipped_unknown


def convert_from_ls(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    output_format: OutputFormat = "yolo",
    recursive: bool = True,
    categories_path: str | Path | None = None,
    copy_images: bool = True,
) -> dict[str, int]:
    """
    将 Label Studio 导出转换为 YOLO11 或 Pascal VOC（LabelImg）。

    输入目录同时作为 JSON 与图像根目录；recursive=True（默认）时递归子目录。
    返回统计：images / labels / boxes / negatives / skipped / unknown_labels。
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if output_format not in {"yolo", "voc"}:
        raise ValueError(f"不支持的输出格式: {output_format}（仅支持 yolo / voc）")

    image_root = input_path if input_path.is_dir() else input_path.parent
    categories = Path(categories_path) if categories_path else None
    class_names, name_to_id = build_class_maps(
        categories, input_path, recursive=recursive
    )
    id_to_name = {idx: name for name, idx in name_to_id.items()}

    stats = {
        "images": 0,
        "labels": 0,
        "boxes": 0,
        "negatives": 0,
        "skipped": 0,
        "unknown_labels": 0,
    }

    if output_format == "yolo":
        image_out_dir = output_dir / "images" / "train"
        label_out_dir = output_dir / "labels" / "train"
        image_out_dir.mkdir(parents=True, exist_ok=True)
        label_out_dir.mkdir(parents=True, exist_ok=True)
    else:
        image_out_dir = output_dir / "JPEGImages"
        label_out_dir = output_dir / "Annotations"
        image_out_dir.mkdir(parents=True, exist_ok=True)
        label_out_dir.mkdir(parents=True, exist_ok=True)

    for source_path, task in iter_ls_tasks(input_path, recursive=recursive):
        try:
            file_name = resolve_image_name(task, source_path)
        except ValueError as exc:
            logger.warning("%s (%s)", exc, source_path)
            stats["skipped"] += 1
            continue

        image_path = find_image_for_task(
            image_root,
            file_name,
            source_path=source_path,
            recursive=recursive,
        )
        if image_path is None:
            logger.warning("未找到图像文件: %s", file_name)
            stats["skipped"] += 1
            continue

        result_items = extract_rectangle_results(task)
        try:
            width, height = cvt.read_image_size(image_path)
        except ValueError as exc:
            logger.warning("%s", exc)
            stats["skipped"] += 1
            continue

        meta_size = result_image_size(result_items)
        if meta_size and meta_size != (width, height):
            logger.info(
                "尺寸校正 %s: ls=%sx%s, actual=%sx%s",
                file_name,
                meta_size[0],
                meta_size[1],
                width,
                height,
            )

        annotations, skipped_unknown = results_to_annotations(
            result_items,
            width=width,
            height=height,
            name_to_id=name_to_id,
        )
        stats["unknown_labels"] += skipped_unknown

        if not annotations:
            stats["negatives"] += 1

        image_name = image_path.name
        stem = Path(image_name).stem

        if output_format == "yolo":
            if copy_images:
                shutil.copy2(image_path, image_out_dir / image_name)

            yolo_lines: list[str] = []
            for ann in annotations:
                line = cvt.bbox_to_yolo_line(
                    int(ann["category_id"]),
                    ann["points"],
                    width,
                    height,
                )
                if line:
                    yolo_lines.append(line)
                    stats["boxes"] += 1

            (label_out_dir / f"{stem}.txt").write_text(
                "\n".join(yolo_lines) + ("\n" if yolo_lines else ""),
                encoding="utf-8",
            )
        else:
            image_out_path = image_out_dir / image_name
            if copy_images:
                shutil.copy2(image_path, image_out_path)
            else:
                image_out_path = image_path

            for ann in annotations:
                if cvt.bbox_to_voc_ints(ann["points"], width, height) is not None:
                    stats["boxes"] += 1

            voc_xml = cvt.build_voc_xml(
                image_name=image_name,
                image_path=image_out_path,
                folder_name="JPEGImages",
                width=width,
                height=height,
                annotations=annotations,
                id_to_name=id_to_name,
            )
            (label_out_dir / f"{stem}.xml").write_text(voc_xml, encoding="utf-8")

        stats["images"] += 1
        stats["labels"] += 1

    if output_format == "yolo":
        cvt.write_yolo_data_yaml(output_dir, class_names)
    else:
        cvt.write_voc_predefined_classes(output_dir, class_names)

    logger.info(
        "转换完成 format=%s output=%s images=%s boxes=%s negatives=%s skipped=%s unknown=%s",
        output_format,
        output_dir,
        stats["images"],
        stats["boxes"],
        stats["negatives"],
        stats["skipped"],
        stats["unknown_labels"],
    )
    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # —— 配置（修改后直接运行）——
    INPUT_DIR = Path(
        "/Users/shunyaoyin/Documents/ai-company/scanly/样本数据/CAM1_GlueSeam600张"
    )
    OUTPUT_DIR = Path(
        "/Users/shunyaoyin/Documents/ai-company/scanly/样本数据/CAM1_GlueSeam600张-voc"
    )
    OUTPUT_FORMAT: OutputFormat = "voc"  # "yolo" / "voc"
    RECURSIVE = True  # 递归扫描输入目录下子目录中的 JSON / 图像

    result = convert_from_ls(
        input_path=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        output_format=OUTPUT_FORMAT,
        recursive=RECURSIVE,
    )
    print(result)
