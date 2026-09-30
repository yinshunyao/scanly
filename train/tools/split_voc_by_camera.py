#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""按源 YOLO 相机组（cam1&3 / cam2）将扁平 VOC 拆成两套 VOC。

分组依据：对照目录下图像 stem，不猜文件名末尾数字。
默认复制，不修改源 VOC。
"""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".bmp", ".png", ".tif", ".tiff", ".webp"}
IMAGE_DIR_NAMES = ("JPEGImages", "images", "Images")
ANNOTATION_DIR_NAMES = ("Annotations", "annotations")
CAMERA_GROUPS = ("cam1&3", "cam2")


@dataclass
class SplitStats:
    voc_dir: str
    camera_ref_dir: str
    output_dir: str
    dry_run: bool
    image_count: int = 0
    cam13_count: int = 0
    cam2_count: int = 0
    conflict_count: int = 0
    unmatched_count: int = 0
    missing_xml_count: int = 0
    copied_xml_count: int = 0
    conflicts: list[str] = field(default_factory=list)
    unmatched: list[str] = field(default_factory=list)
    missing_xml: list[str] = field(default_factory=list)


def _find_subdir(root: Path, candidates: Iterable[str]) -> Path | None:
    for name in candidates:
        path = root / name
        if path.is_dir():
            return path
    return None


def _iter_images(image_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(image_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            files.append(path)
    return files


def _collect_stems_from_camera_group(group_dir: Path) -> set[str]:
    """收集相机组目录下全部图像 stem（优先 */images/，否则递归全组）。"""
    stems: set[str] = set()
    if not group_dir.is_dir():
        return stems

    image_dirs = [p for p in group_dir.rglob("images") if p.is_dir()]
    if image_dirs:
        for image_dir in image_dirs:
            for path in _iter_images(image_dir):
                stems.add(path.stem)
        return stems

    for path in _iter_images(group_dir):
        stems.add(path.stem)
    return stems


def _resolve_xml_path(annotation_dir: Path | None, image_path: Path, image_dir: Path) -> Path | None:
    if annotation_dir is None:
        return None
    rel = image_path.relative_to(image_dir)
    candidate = annotation_dir / rel.with_suffix(".xml")
    if candidate.is_file():
        return candidate
    flat = annotation_dir / f"{image_path.stem}.xml"
    if flat.is_file():
        return flat
    return None


def _copy_file(src: Path, dest: Path, *, dry_run: bool) -> None:
    if dry_run:
        logger.info("[dry-run] copy %s -> %s", src, dest)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def split_voc_by_camera(
    *,
    voc_dir: Path,
    camera_ref_dir: Path,
    output_dir: Path,
    dry_run: bool = False,
    report_path: Path | None = None,
    max_list: int = 50,
) -> SplitStats:
    voc_dir = voc_dir.resolve()
    camera_ref_dir = camera_ref_dir.resolve()
    output_dir = output_dir.resolve()

    image_dir = _find_subdir(voc_dir, IMAGE_DIR_NAMES)
    if image_dir is None:
        raise FileNotFoundError(f"VOC 下未找到图像目录: {voc_dir} / {IMAGE_DIR_NAMES}")
    annotation_dir = _find_subdir(voc_dir, ANNOTATION_DIR_NAMES)

    stems_13 = _collect_stems_from_camera_group(camera_ref_dir / "cam1&3")
    stems_2 = _collect_stems_from_camera_group(camera_ref_dir / "cam2")
    if not stems_13 and not stems_2:
        raise FileNotFoundError(
            f"对照目录无图像: {camera_ref_dir}/cam1&3 或 cam2"
        )

    overlap = stems_13 & stems_2
    if overlap:
        logger.warning("对照目录 stem 交叉 %d 个，将记为 conflict", len(overlap))

    stats = SplitStats(
        voc_dir=str(voc_dir),
        camera_ref_dir=str(camera_ref_dir),
        output_dir=str(output_dir),
        dry_run=dry_run,
    )

    out_dirs = {
        "cam1&3": output_dir / "cam1&3",
        "cam2": output_dir / "cam2",
    }
    for group_name, group_root in out_dirs.items():
        (group_root / "JPEGImages").mkdir(parents=True, exist_ok=True)
        (group_root / "Annotations").mkdir(parents=True, exist_ok=True)

    classes_src = voc_dir / "predefined_classes.txt"
    if classes_src.is_file():
        for group_root in out_dirs.values():
            _copy_file(classes_src, group_root / "predefined_classes.txt", dry_run=dry_run)

    images = _iter_images(image_dir)
    stats.image_count = len(images)

    for image_path in images:
        stem = image_path.stem
        in_13 = stem in stems_13
        in_2 = stem in stems_2

        if in_13 and in_2:
            stats.conflict_count += 1
            if len(stats.conflicts) < max_list:
                stats.conflicts.append(stem)
            continue
        if in_13:
            group = "cam1&3"
            stats.cam13_count += 1
        elif in_2:
            group = "cam2"
            stats.cam2_count += 1
        else:
            stats.unmatched_count += 1
            if len(stats.unmatched) < max_list:
                stats.unmatched.append(stem)
            continue

        group_root = out_dirs[group]
        rel = image_path.relative_to(image_dir)
        dest_image = group_root / "JPEGImages" / rel
        _copy_file(image_path, dest_image, dry_run=dry_run)

        xml_src = _resolve_xml_path(annotation_dir, image_path, image_dir)
        if xml_src is None:
            stats.missing_xml_count += 1
            if len(stats.missing_xml) < max_list:
                stats.missing_xml.append(stem)
            continue

        dest_xml = group_root / "Annotations" / rel.with_suffix(".xml")
        _copy_file(xml_src, dest_xml, dry_run=dry_run)
        stats.copied_xml_count += 1

    if report_path is not None:
        report_path = report_path.resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(asdict(stats), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("report written: %s", report_path)

    logger.info(
        "done: images=%d cam1&3=%d cam2=%d conflict=%d unmatched=%d "
        "xml=%d missing_xml=%d dry_run=%s out=%s",
        stats.image_count,
        stats.cam13_count,
        stats.cam2_count,
        stats.conflict_count,
        stats.unmatched_count,
        stats.copied_xml_count,
        stats.missing_xml_count,
        stats.dry_run,
        stats.output_dir,
    )
    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # ===== 配置区（直接改变量后运行）=====
    SCANLY_ROOT = Path(__file__).resolve().parents[2]
    VOC_DIR = SCANLY_ROOT / "样本数据" / "cam-all-0901_voc"
    CAMERA_REF_DIR = SCANLY_ROOT / "样本数据" / "cam-all-0901"
    OUTPUT_DIR = SCANLY_ROOT / "样本数据" / "came-split-0906-voc"
    DRY_RUN = False
    REPORT_PATH = Path(__file__).resolve().parent / "split_voc_by_camera_report.json"
    # ====================================

    result = split_voc_by_camera(
        voc_dir=VOC_DIR,
        camera_ref_dir=CAMERA_REF_DIR,
        output_dir=OUTPUT_DIR,
        dry_run=DRY_RUN,
        report_path=REPORT_PATH,
    )
    print(
        json.dumps(
            {
                "voc_dir": result.voc_dir,
                "camera_ref_dir": result.camera_ref_dir,
                "output_dir": result.output_dir,
                "dry_run": result.dry_run,
                "image_count": result.image_count,
                "cam13_count": result.cam13_count,
                "cam2_count": result.cam2_count,
                "conflict_count": result.conflict_count,
                "unmatched_count": result.unmatched_count,
                "copied_xml_count": result.copied_xml_count,
                "missing_xml_count": result.missing_xml_count,
                "report_path": str(REPORT_PATH),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
