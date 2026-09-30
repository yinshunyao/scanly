#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""VOC 样本图去重：按文件内容哈希找出完全相同的图片，并将重复项（图+xml）移出。

默认策略：
- 以 JPEGImages 下图片的 MD5 判定「内容完全相同」
- 每个哈希组保留 1 张（按相对路径字典序最小的那张）
- 其余重复图及其同 stem 的 Annotations/*.xml 一并移动到目标目录
- 目标目录默认 = 源目录名 + 后缀（如 cam-all-0901_voc_bak），保持相对路径与文件名不变
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".bmp", ".png", ".tif", ".tiff", ".webp"}
IMAGE_DIR_NAMES = ("JPEGImages", "images", "Images")
ANNOTATION_DIR_NAMES = ("Annotations", "annotations", "labels")


@dataclass
class DuplicateGroup:
    content_hash: str
    keep: str
    moved: list[str] = field(default_factory=list)


@dataclass
class DedupeStats:
    source_dir: str
    dest_dir: str
    dry_run: bool
    image_count: int = 0
    unique_hash_count: int = 0
    duplicate_group_count: int = 0
    move_image_count: int = 0
    move_xml_count: int = 0
    missing_xml_count: int = 0
    groups: list[DuplicateGroup] = field(default_factory=list)


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


def _file_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _resolve_xml_path(annotation_dir: Path | None, image_path: Path, image_dir: Path) -> Path | None:
    if annotation_dir is None:
        return None
    rel = image_path.relative_to(image_dir)
    xml_path = annotation_dir / rel.with_suffix(".xml")
    if xml_path.is_file():
        return xml_path
    # 兼容扁平 Annotations：仅按 stem 匹配
    flat = annotation_dir / f"{image_path.stem}.xml"
    if flat.is_file():
        return flat
    return None


def _move_file(src: Path, dest: Path, dry_run: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        logger.info("[dry-run] move %s -> %s", src, dest)
        return
    if dest.exists():
        raise FileExistsError(f"目标已存在，拒绝覆盖: {dest}")
    shutil.move(str(src), str(dest))
    logger.info("moved %s -> %s", src, dest)


def dedupe_voc_images(
    source_dir: Path,
    dest_dir: Path | None = None,
    dest_suffix: str = "_bak",
    dry_run: bool = True,
    hash_algo: str = "md5",
    report_path: Path | None = None,
) -> DedupeStats:
    """扫描 VOC 目录，移动内容完全相同的重复图片及对应 xml。

    Args:
        source_dir: VOC 根目录（含 JPEGImages / Annotations）
        dest_dir: 重复样本目标根目录；为空则使用 source_dir 名称 + dest_suffix
        dest_suffix: dest_dir 未指定时追加的后缀，默认 `_bak`
        dry_run: True 只统计与打印，不实际移动
        hash_algo: 目前仅支持 md5（精确字节相同）
        report_path: 可选，写出 JSON 报告路径
    """
    source_dir = source_dir.resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"源目录不存在: {source_dir}")
    if hash_algo.lower() != "md5":
        raise ValueError("当前仅支持 hash_algo='md5'（精确去重）")

    if dest_dir is None:
        dest_dir = source_dir.parent / f"{source_dir.name}{dest_suffix}"
    else:
        dest_dir = dest_dir.resolve()
    if dest_dir == source_dir:
        raise ValueError("目标目录不能与源目录相同")

    image_dir = _find_subdir(source_dir, IMAGE_DIR_NAMES)
    if image_dir is None:
        raise FileNotFoundError(
            f"未找到图片目录（候选: {', '.join(IMAGE_DIR_NAMES)}）: {source_dir}"
        )
    annotation_dir = _find_subdir(source_dir, ANNOTATION_DIR_NAMES)

    images = _iter_images(image_dir)
    hash_to_paths: dict[str, list[Path]] = defaultdict(list)
    for idx, image_path in enumerate(images, start=1):
        digest = _file_md5(image_path)
        hash_to_paths[digest].append(image_path)
        if idx % 500 == 0 or idx == len(images):
            logger.info("hashed %d / %d", idx, len(images))

    stats = DedupeStats(
        source_dir=str(source_dir),
        dest_dir=str(dest_dir),
        dry_run=dry_run,
        image_count=len(images),
        unique_hash_count=len(hash_to_paths),
    )

    for digest, paths in sorted(hash_to_paths.items(), key=lambda x: x[0]):
        if len(paths) < 2:
            continue
        # 稳定保留：相对路径字典序最小
        ordered = sorted(paths, key=lambda p: str(p.relative_to(source_dir)))
        keep = ordered[0]
        group = DuplicateGroup(
            content_hash=digest,
            keep=str(keep.relative_to(source_dir)),
        )
        for dup in ordered[1:]:
            rel_image = dup.relative_to(source_dir)
            dest_image = dest_dir / rel_image
            _move_file(dup, dest_image, dry_run=dry_run)
            stats.move_image_count += 1
            group.moved.append(str(rel_image))

            xml_src = _resolve_xml_path(annotation_dir, dup, image_dir)
            if xml_src is None:
                stats.missing_xml_count += 1
                logger.warning("缺少对应 xml: %s", dup)
                continue
            rel_xml = xml_src.relative_to(source_dir)
            dest_xml = dest_dir / rel_xml
            _move_file(xml_src, dest_xml, dry_run=dry_run)
            stats.move_xml_count += 1

        stats.groups.append(group)

    stats.duplicate_group_count = len(stats.groups)

    if report_path is not None:
        report_path = report_path.resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(stats)
        report_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("report written: %s", report_path)

    logger.info(
        "done: images=%d unique=%d groups=%d move_img=%d move_xml=%d missing_xml=%d dry_run=%s dest=%s",
        stats.image_count,
        stats.unique_hash_count,
        stats.duplicate_group_count,
        stats.move_image_count,
        stats.move_xml_count,
        stats.missing_xml_count,
        stats.dry_run,
        stats.dest_dir,
    )
    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # ===== 配置区（直接改变量后运行）=====
    SCANLY_ROOT = Path(__file__).resolve().parents[2]
    SOURCE_DIR = Path("/Users/shunyaoyin/Documents/code/ai-company/scanly/样本数据/cam-all-0901_voc")

    # 目标目录二选一：
    # 1) 显式指定 DEST_DIR
    # 2) DEST_DIR=None 时，使用 SOURCE_DIR 名称 + DEST_SUFFIX
    DEST_DIR = None  # 例如: SCANLY_ROOT / "样本数据" / "cam-all-0901_voc_dup"
    DEST_SUFFIX = "_bak"  # -> .../cam-all-0901_voc_bak

    # 建议先 True 预览，确认数量后再改 False 真正移动
    DRY_RUN = False

    REPORT_PATH = Path(__file__).resolve().parent / "dedupe_voc_images_report.json"
    # ====================================

    result = dedupe_voc_images(
        source_dir=SOURCE_DIR,
        dest_dir=DEST_DIR,
        dest_suffix=DEST_SUFFIX,
        dry_run=DRY_RUN,
        report_path=REPORT_PATH,
    )
    print(
        json.dumps(
            {
                "source_dir": result.source_dir,
                "dest_dir": result.dest_dir,
                "dry_run": result.dry_run,
                "image_count": result.image_count,
                "unique_hash_count": result.unique_hash_count,
                "duplicate_group_count": result.duplicate_group_count,
                "move_image_count": result.move_image_count,
                "move_xml_count": result.move_xml_count,
                "missing_xml_count": result.missing_xml_count,
                "report_path": str(REPORT_PATH),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
